"""Dimension-normalized local-target objective for Posterior V8.

This is intentionally separate from the frozen global-target v1 objective.
Discrete topology/pattern semantics are reused unchanged; only the continuous
target contract and reduction are versioned to v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np
import tensorflow as tf

from .local_target import (
    GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    LOCAL_TARGET_COORDINATE_SEMANTICS,
    VARYING_DIMENSION_SEMANTICS,
)
from .model import (
    BRANCH_DIM,
    mask_invalid_branch_pattern_logits,
    masked_logistic_normal_nll,
)
from .training_objective import _validated_tensors


LOCAL_TRAINING_OBJECTIVE_VERSION = (
    "posterior_v8_joint_branch_local_target_dimension_normalized_nll_v2"
)
CONTINUOUS_REDUCTION_SEMANTICS = (
    "mixture_nll_sum_over_varying_dimensions_then_divide_by_varying_count_v1"
)
OBJECTIVE_TARGET_COORDINATE_SEMANTICS = LOCAL_TARGET_COORDINATE_SEMANTICS
OBJECTIVE_BRANCH_BOX_COORDINATE_SEMANTICS = GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS
OBJECTIVE_DENSITY_MASK_SEMANTICS = VARYING_DIMENSION_SEMANTICS


def _weight(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class LocalTrainingObjectiveConfig:
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
        if not any(value > 0.0 for value in weights):
            raise ValueError("at least one objective weight must be positive")
        if (
            isinstance(self.topology_recall_k, (bool, np.bool_))
            or not isinstance(self.topology_recall_k, Integral)
            or not 1 <= int(self.topology_recall_k) <= 34
        ):
            raise ValueError("topology_recall_k must be an integer in [1, 34]")
        epsilon = float(self.logistic_epsilon)
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError("logistic_epsilon must be finite and in (0, 0.5)")
        invalid = float(self.invalid_pattern_logit)
        if not np.isfinite(invalid) or invalid >= 0.0:
            raise ValueError("invalid_pattern_logit must be finite and negative")
        object.__setattr__(self, "topology_weight", weights[0])
        object.__setattr__(self, "branch_pattern_weight", weights[1])
        object.__setattr__(self, "continuous_weight", weights[2])
        object.__setattr__(self, "topology_recall_k", int(self.topology_recall_k))
        object.__setattr__(self, "logistic_epsilon", epsilon)
        object.__setattr__(self, "invalid_pattern_logit", invalid)

_REQUIRED_LABELS = (
    "topology_id",
    "branch_pattern_id",
    "target_local",
    "active_dimension_mask",
    "varying_dimension_mask",
)


def _validated_local_labels(labels: Mapping[str, object]):
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping")
    missing = [name for name in _REQUIRED_LABELS if name not in labels]
    if missing:
        raise ValueError("local-target labels are missing: " + ", ".join(missing))
    target = tf.cast(labels["target_local"], tf.float32)
    active = tf.cast(labels["active_dimension_mask"], tf.float32)
    varying = tf.cast(labels["varying_dimension_mask"], tf.float32)
    batch = tf.shape(target)[0]
    assertions = (
        tf.debugging.assert_rank(target, 2),
        tf.debugging.assert_equal(tf.shape(target), [batch, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(active), [batch, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(varying), [batch, BRANCH_DIM]),
        tf.debugging.assert_all_finite(target, "target_local contains NaN/Inf"),
        tf.debugging.assert_all_finite(active, "active_dimension_mask contains NaN/Inf"),
        tf.debugging.assert_all_finite(varying, "varying_dimension_mask contains NaN/Inf"),
        tf.debugging.assert_greater_equal(active, 0.0),
        tf.debugging.assert_less_equal(active, 1.0),
        tf.debugging.assert_greater_equal(varying, 0.0),
        tf.debugging.assert_less_equal(varying, 1.0),
        tf.debugging.assert_equal(active, tf.round(active), message="active mask must be binary"),
        tf.debugging.assert_equal(
            varying, tf.round(varying), message="varying mask must be binary"
        ),
        tf.debugging.assert_greater_equal(target, 0.0),
        tf.debugging.assert_less_equal(target, 1.0),
    )
    with tf.control_dependencies(assertions):
        target, active, varying = (
            tf.identity(value) for value in (target, active, varying)
        )
    varying_bool = varying > 0.5
    canonical = tf.logical_not(varying_bool)
    canonical_target = tf.boolean_mask(target, canonical)
    semantic_assertions = (
        tf.debugging.assert_less_equal(
            varying,
            active,
            message="varying dimensions must be semantically active",
        ),
        tf.debugging.assert_equal(
            canonical_target,
            tf.fill(tf.shape(canonical_target), tf.constant(0.5, tf.float32)),
            message="fixed and inactive local targets must equal 0.5",
        ),
    )
    with tf.control_dependencies(semantic_assertions):
        return tuple(tf.identity(value) for value in (target, active, varying))


def normalized_masked_logistic_normal_nll(
    target_local,
    varying_dimension_mask,
    mixture_logits,
    mixture_loc,
    mixture_logscale,
    *,
    epsilon: float = 1.0e-5,
):
    """Return one continuous NLL per example, normalized by varying dimensions."""

    mask = tf.cast(varying_dimension_mask, tf.float32)
    summed = masked_logistic_normal_nll(
        target_local,
        mask,
        mixture_logits,
        mixture_loc,
        mixture_logscale,
        epsilon=epsilon,
    )
    counts = tf.reduce_sum(mask, axis=-1)
    return tf.math.divide_no_nan(summed, counts)


def compute_local_training_objective(
    outputs: Mapping[str, object],
    labels: Mapping[str, object],
    config: LocalTrainingObjectiveConfig = LocalTrainingObjectiveConfig(),
) -> dict[str, tf.Tensor]:
    """Compute unchanged discrete losses plus local, dimension-normalized NLL."""

    if not isinstance(config, LocalTrainingObjectiveConfig):
        raise TypeError("config must be a LocalTrainingObjectiveConfig")
    target, active, varying = _validated_local_labels(labels)
    baseline_labels = {
        "topology_id": labels["topology_id"],
        "branch_pattern_id": labels["branch_pattern_id"],
        # The v1 helper is reused only to validate unchanged branch/discrete
        # semantics. Its continuous result is deliberately discarded below.
        "target_unit": target,
        "active_dimension_mask": active,
    }
    (
        topology_logits,
        pattern_logits,
        _,
        _,
        _,
        topology_id,
        pattern_id,
        _,
        _,
    ) = _validated_tensors(outputs, baseline_labels)
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
    topology_loss = tf.reduce_mean(topology_per_example)
    branch_pattern_loss = tf.reduce_mean(pattern_per_example)
    continuous_per_example = normalized_masked_logistic_normal_nll(
        target,
        varying,
        outputs["mixture_logits"],
        outputs["mixture_loc"],
        outputs["mixture_logscale"],
        epsilon=config.logistic_epsilon,
    )
    continuous_nll = tf.reduce_mean(continuous_per_example)
    loss = (
        config.topology_weight * topology_loss
        + config.branch_pattern_weight * branch_pattern_loss
        + config.continuous_weight * continuous_nll
    )
    with tf.control_dependencies(
        (tf.debugging.assert_all_finite(loss, "local-target v2 objective is non-finite"),)
    ):
        loss = tf.identity(loss)
    return {
        "loss": loss,
        "topology_loss": topology_loss,
        "branch_pattern_loss": branch_pattern_loss,
        "continuous_nll": continuous_nll,
        "topology_accuracy": tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(topology_logits, axis=-1, output_type=tf.int32),
                    topology_id,
                ),
                tf.float32,
            )
        ),
        "branch_pattern_accuracy": tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(
                        true_topology_pattern_logits, axis=-1, output_type=tf.int32
                    ),
                    pattern_id,
                ),
                tf.float32,
            )
        ),
        "topology_recall_at_k": tf.reduce_mean(
            tf.cast(
                tf.math.in_top_k(
                    targets=topology_id,
                    predictions=topology_logits,
                    k=config.topology_recall_k,
                ),
                tf.float32,
            )
        ),
        "mean_varying_dimension_count": tf.reduce_mean(
            tf.reduce_sum(varying, axis=-1)
        ),
    }


__all__ = [
    "CONTINUOUS_REDUCTION_SEMANTICS",
    "LOCAL_TRAINING_OBJECTIVE_VERSION",
    "LocalTrainingObjectiveConfig",
    "OBJECTIVE_BRANCH_BOX_COORDINATE_SEMANTICS",
    "OBJECTIVE_DENSITY_MASK_SEMANTICS",
    "OBJECTIVE_TARGET_COORDINATE_SEMANTICS",
    "compute_local_training_objective",
    "normalized_masked_logistic_normal_nll",
]
