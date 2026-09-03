"""Supervised density objective for the Posterior V8 proposal network.

One simulated draw supplies one valid topology/branch/continuous sample.  The
objective does not label unobserved alternative modes as negatives and does
not use a differentiable scattering approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Mapping

import tensorflow as tf

from .branch_catalog import (
    BRANCH_PATTERN_COUNT,
    VALID_BRANCH_PATTERN_MASK,
    branch_pattern_is_valid,
    decode_branch_pattern,
)
from .contract import CYLINDER, NUM_TOPOLOGIES, TOPOLOGIES
from .model import (
    BRANCH_DIM,
    COMPONENT_PARAMETER_STRIDE,
    RESOLUTION_OFFSET,
    mask_invalid_branch_pattern_logits,
    masked_logistic_normal_nll,
)


TRAINING_OBJECTIVE_VERSION = "posterior_v8_joint_branch_mdn_objective_v1"


def _active_mask_unchecked(topology_id: int, pattern_id: int) -> tuple[bool, ...]:
    d_present, resolution_present = decode_branch_pattern(pattern_id)
    mask = [False] * BRANCH_DIM
    for slot, shape in enumerate(TOPOLOGIES[topology_id]):
        offset = slot * COMPONENT_PARAMETER_STRIDE
        mask[offset] = True
        mask[offset + 1] = True
        if shape == CYLINDER:
            mask[offset + 2] = True
            mask[offset + 3] = True
        if d_present[slot]:
            mask[offset + 4] = True
            mask[offset + 5] = True
    if resolution_present:
        mask[RESOLUTION_OFFSET] = True
        mask[RESOLUTION_OFFSET + 1] = True
    return tuple(mask)


ACTIVE_DIMENSION_MASKS = tuple(
    tuple(
        _active_mask_unchecked(topology_id, pattern_id)
        for pattern_id in range(BRANCH_PATTERN_COUNT)
    )
    for topology_id in range(NUM_TOPOLOGIES)
)


def active_dimension_mask_for(topology_id: int, pattern_id: int) -> tuple[bool, ...]:
    """Return the exact 26-D structural mask for one valid hard branch."""

    if not branch_pattern_is_valid(topology_id, pattern_id):
        raise ValueError("branch pattern is invalid for the selected topology")
    return ACTIVE_DIMENSION_MASKS[int(topology_id)][int(pattern_id)]


def _weight(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class TrainingObjectiveConfig:
    topology_weight: float = 1.0
    branch_pattern_weight: float = 1.0
    continuous_weight: float = 1.0
    topology_recall_k: int = 8
    logistic_epsilon: float = 1.0e-5
    invalid_pattern_logit: float = -1.0e9

    def __post_init__(self) -> None:
        weights = (
            _weight(self.topology_weight, "topology_weight"),
            _weight(self.branch_pattern_weight, "branch_pattern_weight"),
            _weight(self.continuous_weight, "continuous_weight"),
        )
        if not any(weights):
            raise ValueError("at least one objective weight must be positive")
        if (
            isinstance(self.topology_recall_k, bool)
            or not isinstance(self.topology_recall_k, Integral)
            or not 1 <= int(self.topology_recall_k) <= NUM_TOPOLOGIES
        ):
            raise ValueError("topology_recall_k must be an integer in [1, 34]")
        epsilon = float(self.logistic_epsilon)
        if not isfinite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError("logistic_epsilon must be finite and in (0, 0.5)")
        invalid_logit = float(self.invalid_pattern_logit)
        if not isfinite(invalid_logit) or invalid_logit >= 0.0:
            raise ValueError("invalid_pattern_logit must be finite and negative")
        object.__setattr__(self, "topology_weight", weights[0])
        object.__setattr__(self, "branch_pattern_weight", weights[1])
        object.__setattr__(self, "continuous_weight", weights[2])
        object.__setattr__(self, "topology_recall_k", int(self.topology_recall_k))
        object.__setattr__(self, "logistic_epsilon", epsilon)
        object.__setattr__(self, "invalid_pattern_logit", invalid_logit)


_OUTPUT_NAMES = (
    "topology_logits",
    "branch_pattern_logits",
    "mixture_logits",
    "mixture_loc",
    "mixture_logscale",
)
_LABEL_NAMES = (
    "topology_id",
    "branch_pattern_id",
    "target_unit",
    "active_dimension_mask",
)


def _values(mapping, names, kind):
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{kind} must be a mapping")
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"{kind} is missing required values: {missing}")
    return tuple(tf.convert_to_tensor(mapping[name]) for name in names)


def _integer_labels(value, name):
    if value.dtype == tf.bool or not value.dtype.is_integer:
        raise TypeError(f"{name} must use an integer tensor dtype")
    assertion = tf.debugging.assert_rank(value, 1, message=f"{name} must be rank one")
    with tf.control_dependencies((assertion,)):
        return tf.cast(tf.identity(value), tf.int32)


def _validated_tensors(outputs, labels):
    topology_logits, pattern_logits, mixture_logits, mixture_loc, mixture_logscale = _values(
        outputs, _OUTPUT_NAMES, "outputs"
    )
    topology_id, pattern_id, target_unit, active_mask = _values(
        labels, _LABEL_NAMES, "labels"
    )
    topology_id = _integer_labels(topology_id, "topology_id")
    pattern_id = _integer_labels(pattern_id, "branch_pattern_id")
    topology_logits = tf.cast(topology_logits, tf.float32)
    pattern_logits = tf.cast(pattern_logits, tf.float32)
    mixture_logits = tf.cast(mixture_logits, tf.float32)
    mixture_loc = tf.cast(mixture_loc, tf.float32)
    mixture_logscale = tf.cast(mixture_logscale, tf.float32)
    target_unit = tf.cast(target_unit, tf.float32)
    active_mask = tf.cast(active_mask, tf.float32)

    batch = tf.shape(topology_logits)[0]
    modes = tf.shape(mixture_logits)[1]
    assertions = [
        tf.debugging.assert_rank(topology_logits, 2),
        tf.debugging.assert_rank(pattern_logits, 3),
        tf.debugging.assert_rank(mixture_logits, 2),
        tf.debugging.assert_rank(mixture_loc, 3),
        tf.debugging.assert_rank(mixture_logscale, 3),
        tf.debugging.assert_rank(target_unit, 2),
        tf.debugging.assert_rank(active_mask, 2),
        tf.debugging.assert_equal(tf.shape(topology_logits), [batch, NUM_TOPOLOGIES]),
        tf.debugging.assert_equal(
            tf.shape(pattern_logits), [batch, NUM_TOPOLOGIES, BRANCH_PATTERN_COUNT]
        ),
        tf.debugging.assert_equal(tf.shape(mixture_logits)[0], batch),
        tf.debugging.assert_positive(modes, message="at least one mixture mode is required"),
        tf.debugging.assert_equal(tf.shape(mixture_loc), [batch, modes, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(mixture_logscale), [batch, modes, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(topology_id), [batch]),
        tf.debugging.assert_equal(tf.shape(pattern_id), [batch]),
        tf.debugging.assert_equal(tf.shape(target_unit), [batch, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(active_mask), [batch, BRANCH_DIM]),
        tf.debugging.assert_all_finite(topology_logits, "topology_logits contains NaN/Inf"),
        tf.debugging.assert_all_finite(pattern_logits, "branch_pattern_logits contains NaN/Inf"),
        tf.debugging.assert_all_finite(mixture_logits, "mixture_logits contains NaN/Inf"),
        tf.debugging.assert_all_finite(mixture_loc, "mixture_loc contains NaN/Inf"),
        tf.debugging.assert_all_finite(mixture_logscale, "mixture_logscale contains NaN/Inf"),
        tf.debugging.assert_all_finite(target_unit, "target_unit contains NaN/Inf"),
        tf.debugging.assert_all_finite(active_mask, "active_dimension_mask contains NaN/Inf"),
        tf.debugging.assert_greater_equal(topology_id, 0),
        tf.debugging.assert_less(topology_id, NUM_TOPOLOGIES),
        tf.debugging.assert_greater_equal(pattern_id, 0),
        tf.debugging.assert_less(pattern_id, BRANCH_PATTERN_COUNT),
        tf.debugging.assert_greater_equal(active_mask, 0.0),
        tf.debugging.assert_less_equal(active_mask, 1.0),
        tf.debugging.assert_equal(
            active_mask, tf.round(active_mask), message="active dimension mask must be binary"
        ),
    ]
    with tf.control_dependencies(assertions):
        tensors = tuple(
            tf.identity(value)
            for value in (
                topology_logits,
                pattern_logits,
                mixture_logits,
                mixture_loc,
                mixture_logscale,
                topology_id,
                pattern_id,
                target_unit,
                active_mask,
            )
        )

    topology_id, pattern_id = tensors[5], tensors[6]
    indices = tf.stack([topology_id, pattern_id], axis=1)
    valid = tf.gather_nd(tf.constant(VALID_BRANCH_PATTERN_MASK), indices)
    expected_mask = tf.cast(
        tf.gather_nd(tf.constant(ACTIVE_DIMENSION_MASKS), indices), tf.float32
    )
    target_for_bounds = tf.where(expected_mask > 0.5, tensors[7], 0.5)
    semantic_assertions = (
        tf.debugging.assert_equal(
            valid,
            tf.ones_like(valid),
            message="branch_pattern_id is invalid for topology_id",
        ),
        tf.debugging.assert_equal(
            tensors[8], expected_mask, message="active dimension mask does not match branch"
        ),
        tf.debugging.assert_greater_equal(
            target_for_bounds, 0.0, message="active target_unit values must be in [0, 1]"
        ),
        tf.debugging.assert_less_equal(
            target_for_bounds, 1.0, message="active target_unit values must be in [0, 1]"
        ),
    )
    with tf.control_dependencies(semantic_assertions):
        return tuple(tf.identity(value) for value in tensors)


def compute_training_objective(
    outputs: Mapping[str, object],
    labels: Mapping[str, object],
    config: TrainingObjectiveConfig = TrainingObjectiveConfig(),
) -> dict[str, tf.Tensor]:
    """Return weighted scalar loss, component losses and training metrics."""

    if not isinstance(config, TrainingObjectiveConfig):
        raise TypeError("config must be a TrainingObjectiveConfig")
    (
        topology_logits,
        pattern_logits,
        mixture_logits,
        mixture_loc,
        mixture_logscale,
        topology_id,
        pattern_id,
        target_unit,
        active_mask,
    ) = _validated_tensors(outputs, labels)

    topology_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=topology_id, logits=topology_logits
    )
    masked_pattern_logits = mask_invalid_branch_pattern_logits(
        pattern_logits, invalid_logit=config.invalid_pattern_logit
    )
    true_topology_pattern_logits = tf.gather(
        masked_pattern_logits, topology_id, axis=1, batch_dims=1
    )
    pattern_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=pattern_id, logits=true_topology_pattern_logits
    )
    continuous_per_example = masked_logistic_normal_nll(
        target_unit,
        active_mask,
        mixture_logits,
        mixture_loc,
        mixture_logscale,
        epsilon=config.logistic_epsilon,
    )

    topology_loss = tf.reduce_mean(topology_per_example)
    branch_pattern_loss = tf.reduce_mean(pattern_per_example)
    continuous_nll = tf.reduce_mean(continuous_per_example)
    loss = (
        config.topology_weight * topology_loss
        + config.branch_pattern_weight * branch_pattern_loss
        + config.continuous_weight * continuous_nll
    )
    finite_loss = tf.debugging.assert_all_finite(
        loss, "Posterior V8 training objective is non-finite"
    )
    with tf.control_dependencies((finite_loss,)):
        loss = tf.identity(loss)

    topology_prediction = tf.argmax(topology_logits, axis=-1, output_type=tf.int32)
    pattern_prediction = tf.argmax(
        true_topology_pattern_logits, axis=-1, output_type=tf.int32
    )
    topology_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(topology_prediction, topology_id), tf.float32)
    )
    branch_pattern_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pattern_prediction, pattern_id), tf.float32)
    )
    topology_recall = tf.reduce_mean(
        tf.cast(
            tf.math.in_top_k(
                targets=topology_id,
                predictions=topology_logits,
                k=config.topology_recall_k,
            ),
            tf.float32,
        )
    )
    return {
        "loss": loss,
        "topology_loss": topology_loss,
        "branch_pattern_loss": branch_pattern_loss,
        "continuous_nll": continuous_nll,
        "topology_accuracy": topology_accuracy,
        "branch_pattern_accuracy": branch_pattern_accuracy,
        "topology_recall_at_k": topology_recall,
    }


__all__ = [
    "ACTIVE_DIMENSION_MASKS",
    "TRAINING_OBJECTIVE_VERSION",
    "TrainingObjectiveConfig",
    "active_dimension_mask_for",
    "compute_training_objective",
]
