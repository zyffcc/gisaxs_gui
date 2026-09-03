"""Bounds-first objective identity over the verified local-density calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf

from .bounds_first_contract import LOCAL_TARGET_SEMANTICS
from .bounds_model_contract import MODEL_VARYING_MASK_SEMANTICS
from .canonical_branch_catalog import CANONICAL_VALID_BRANCH_PATTERN_MASK
from .model import mask_invalid_branch_pattern_logits
from .training_objective import _validated_tensors
from .training_objective_v2 import (
    CONTINUOUS_REDUCTION_SEMANTICS,
    LocalTrainingObjectiveConfig,
    _validated_local_labels,
    normalized_masked_logistic_normal_nll,
)


BOUNDS_TRAINING_OBJECTIVE_VERSION = (
    "posterior_v8_bounds_first_local_dimension_normalized_mdn_objective_v3"
)
BOUNDS_OBJECTIVE_TARGET_COORDINATE_SEMANTICS = LOCAL_TARGET_SEMANTICS
BOUNDS_OBJECTIVE_DENSITY_MASK_SEMANTICS = MODEL_VARYING_MASK_SEMANTICS
BOUNDS_CONTINUOUS_REDUCTION_SEMANTICS = CONTINUOUS_REDUCTION_SEMANTICS


@dataclass(frozen=True)
class BoundsTrainingObjectiveConfig(LocalTrainingObjectiveConfig):
    """Distinct V3 config type; numeric hyperparameters remain compatible."""


def compute_bounds_training_objective(
    outputs: Mapping[str, object],
    labels: Mapping[str, object],
    config: BoundsTrainingObjectiveConfig = BoundsTrainingObjectiveConfig(),
) -> dict[str, tf.Tensor]:
    """Evaluate native local MDN loss over canonical physical branches only."""

    if not isinstance(config, BoundsTrainingObjectiveConfig):
        raise TypeError("config must be a BoundsTrainingObjectiveConfig")
    target, active, varying = _validated_local_labels(labels)
    baseline_labels = {
        "topology_id": labels["topology_id"],
        "branch_pattern_id": labels["branch_pattern_id"],
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
    canonical_mask = tf.constant(CANONICAL_VALID_BRANCH_PATTERN_MASK, tf.bool)
    canonical_label = tf.gather_nd(
        canonical_mask, tf.stack([topology_id, pattern_id], axis=1)
    )
    canonical_assertion = tf.debugging.assert_equal(
        canonical_label,
        tf.ones_like(canonical_label),
        message="branch label is not the canonical physical representative",
    )
    with tf.control_dependencies((canonical_assertion,)):
        topology_logits, pattern_logits = (
            tf.identity(topology_logits),
            tf.identity(pattern_logits),
        )
    topology_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=topology_id, logits=topology_logits
    )
    masked_pattern_logits = mask_invalid_branch_pattern_logits(
        pattern_logits, invalid_logit=config.invalid_pattern_logit
    )
    masked_pattern_logits = tf.where(
        canonical_mask,
        masked_pattern_logits,
        tf.cast(config.invalid_pattern_logit, masked_pattern_logits.dtype),
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
        (tf.debugging.assert_all_finite(loss, "bounds-first V3 objective is non-finite"),)
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
                        true_topology_pattern_logits,
                        axis=-1,
                        output_type=tf.int32,
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
    "BOUNDS_CONTINUOUS_REDUCTION_SEMANTICS",
    "BOUNDS_OBJECTIVE_DENSITY_MASK_SEMANTICS",
    "BOUNDS_OBJECTIVE_TARGET_COORDINATE_SEMANTICS",
    "BOUNDS_TRAINING_OBJECTIVE_VERSION",
    "BoundsTrainingObjectiveConfig",
    "compute_bounds_training_objective",
]
