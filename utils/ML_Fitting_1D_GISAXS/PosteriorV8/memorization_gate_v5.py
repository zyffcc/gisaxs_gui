"""Reproducible K=1 memorization diagnostic for evolving V5 contracts.

This is a pre-training wiring gate, not model acceptance evidence.  It learns
one exact-compatible local target with one logistic-normal component and
reports whether the model can drive both its configured objective and the
varying-coordinate median error down.  Model input names are discovered from
the supplied model, so V5.1 may add context tensors without changing this
runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from typing import Callable, Mapping, Protocol

import numpy as np
import tensorflow as tf

from .candidate_supervision_v5 import CANDIDATE_SUPERVISION_TENSOR_KEYS
from .grouped_artifact_v5 import canonical_json
from .training_objective_v5 import DEFAULT_LOCAL_COVERAGE_WEIGHT


V5_MEMORIZATION_GATE_SCHEMA = "gisaxs.posterior_v8.memorization_gate/v5"
V5_MEMORIZATION_GATE_VERSION = (
    "deterministic_cosine_minibatch_joint_density_center_aligned_local_target_diagnostic_v5"
)


class JoinedNumpyDataset(Protocol):
    def joined_numpy(
        self, *, include_unverified: bool = False
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]: ...


Objective = Callable[
    [Mapping[str, object], Mapping[str, object], object],
    Mapping[str, tf.Tensor],
]


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, kw_only=True)
class V5MemorizationGateConfig:
    steps: int = 18000
    batch_size: int = 32
    learning_rate: float = 3.0e-3
    final_learning_rate: float = 3.0e-5
    learning_rate_schedule: str = "cosine_decay"
    seed: int = 20260903
    local_mdn_weight: float = 1.0
    local_coverage_weight: float = DEFAULT_LOCAL_COVERAGE_WEIGHT
    operational_top_l_alignment_weight: float = 1.0
    max_final_target_median_rms: float = 0.01
    minimum_loss_reduction: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.steps, (bool, np.bool_)) or int(self.steps) != self.steps:
            raise TypeError("steps must be an integer")
        if int(self.steps) < 1:
            raise ValueError("steps must be positive")
        if isinstance(self.batch_size, (bool, np.bool_)) or int(self.batch_size) != self.batch_size:
            raise TypeError("batch_size must be an integer")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if isinstance(self.seed, (bool, np.bool_)) or int(self.seed) != self.seed:
            raise TypeError("seed must be an integer")
        if not 0 <= int(self.seed) <= (1 << 31) - 1:
            raise ValueError("seed must fit in signed int32")
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "seed", int(self.seed))
        objective_weights = {
            name: _nonnegative(getattr(self, name), name)
            for name in (
                "local_mdn_weight",
                "local_coverage_weight",
                "operational_top_l_alignment_weight",
            )
        }
        if not any(value > 0.0 for value in objective_weights.values()):
            raise ValueError("at least one local objective weight must be positive")
        for name, value in objective_weights.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "learning_rate",
            _positive(self.learning_rate, "learning_rate"),
        )
        object.__setattr__(
            self,
            "final_learning_rate",
            _positive(self.final_learning_rate, "final_learning_rate"),
        )
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final_learning_rate must not exceed learning_rate")
        if self.learning_rate_schedule not in {"constant", "cosine_decay"}:
            raise ValueError(
                "learning_rate_schedule must be 'constant' or 'cosine_decay'"
            )
        if (
            self.learning_rate_schedule == "constant"
            and self.final_learning_rate != self.learning_rate
        ):
            raise ValueError(
                "constant schedule requires final_learning_rate == learning_rate"
            )
        object.__setattr__(
            self,
            "max_final_target_median_rms",
            _positive(
                self.max_final_target_median_rms,
                "max_final_target_median_rms",
            ),
        )
        reduction = float(self.minimum_loss_reduction)
        if not isfinite(reduction) or reduction < 0.0:
            raise ValueError("minimum_loss_reduction must be finite and non-negative")
        object.__setattr__(self, "minimum_loss_reduction", reduction)


@dataclass(frozen=True)
class V5MemorizationGateResult:
    initial_loss: float
    final_loss: float
    initial_target_median_rms: float
    final_target_median_rms: float
    target_count: int
    learnable_target_count: int
    fully_fixed_target_count: int
    varying_coordinate_count: int
    model_input_keys: tuple[str, ...]
    objective_audit_sha256: str
    trajectory_sha256: str
    final_weights_sha256: str
    passed: bool
    config: V5MemorizationGateConfig
    result_sha256: str
    schema_version: str = V5_MEMORIZATION_GATE_SCHEMA
    version: str = V5_MEMORIZATION_GATE_VERSION

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "scientific_role": "wiring_and_memorization_diagnostic_not_model_acceptance",
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "initial_target_median_rms": self.initial_target_median_rms,
            "final_target_median_rms": self.final_target_median_rms,
            "target_count": self.target_count,
            "learnable_target_count": self.learnable_target_count,
            "fully_fixed_target_count": self.fully_fixed_target_count,
            "varying_coordinate_count": self.varying_coordinate_count,
            "model_input_keys": list(self.model_input_keys),
            "objective_audit_sha256": self.objective_audit_sha256,
            "trajectory_sha256": self.trajectory_sha256,
            "final_weights_sha256": self.final_weights_sha256,
            "passed": self.passed,
            "config": asdict(self.config),
            "result_sha256": self.result_sha256,
        }


def _model_input_names(model: tf.keras.Model) -> tuple[str, ...]:
    names = tuple(value.name.split(":", 1)[0] for value in model.inputs)
    if not names or len(names) != len(set(names)):
        raise ValueError("model must expose unique named inputs")
    return names


def _weights_sha256(model: tf.keras.Model) -> str:
    digest = sha256()
    for value in model.weights:
        array = np.ascontiguousarray(value.numpy())
        weight_name = getattr(value, "path", value.name)
        digest.update(weight_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _objective_hash(config: object) -> str:
    payload = config.audit_payload() if hasattr(config, "audit_payload") else asdict(config)
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _loss_and_item_rms(
    model: tf.keras.Model,
    inputs: Mapping[str, tf.Tensor],
    labels: Mapping[str, tf.Tensor],
    objective: Objective,
    objective_config: object,
    *,
    training: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    outputs = model(inputs, training=training)
    modes = tf.shape(outputs["mixture_logits"])[1]
    one_mode = tf.debugging.assert_equal(
        modes,
        1,
        message="formal K=1 memorization gate requires one MDN component",
    )
    with tf.control_dependencies((one_mode,)):
        median = tf.math.sigmoid(outputs["mixture_loc"][:, 0, :])
    target = tf.cast(labels["target_local"], tf.float32)
    varying = tf.cast(labels["varying_dimension_mask"], tf.float32)
    has_target = tf.cast(labels["has_local_target"], tf.bool)
    per_item_varying_count = tf.reduce_sum(varying, axis=-1)
    learnable = tf.logical_and(has_target, per_item_varying_count > 0.0)
    squared = tf.square(median - target) * varying
    per_item = tf.sqrt(
        tf.math.divide_no_nan(tf.reduce_sum(squared, axis=-1), tf.reduce_sum(varying, axis=-1))
    )
    selected = tf.boolean_mask(per_item, learnable)
    learnable_count = tf.size(selected)
    positive = tf.debugging.assert_positive(
        learnable_count, message="gate needs at least one learnable local target"
    )
    varying_count = tf.cast(tf.reduce_sum(tf.boolean_mask(varying, learnable)), tf.int32)
    with tf.control_dependencies((positive,)):
        selected = tf.identity(selected)
    values = objective(outputs, labels, objective_config)
    if not isinstance(values, Mapping) or "loss" not in values:
        raise ValueError("objective must return a mapping containing loss")
    loss = tf.cast(values["loss"], tf.float32)
    return loss, selected, varying_count


def _median(values: tf.Tensor) -> tf.Tensor:
    ordered = tf.sort(values)
    count = tf.size(ordered)
    middle = count // 2
    return tf.cond(
        count % 2 == 1,
        lambda: ordered[middle],
        lambda: 0.5 * (ordered[middle - 1] + ordered[middle]),
    )


def _batched_metrics(
    model: tf.keras.Model,
    inputs: Mapping[str, tf.Tensor],
    labels: Mapping[str, tf.Tensor],
    objective: Objective,
    objective_config: object,
    *,
    batch_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    row_count = int(next(iter(inputs.values())).shape[0])
    weighted_losses: list[tf.Tensor] = []
    item_rms: list[tf.Tensor] = []
    varying_counts: list[tf.Tensor] = []
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        batch_inputs = {name: value[start:stop] for name, value in inputs.items()}
        batch_labels = {name: value[start:stop] for name, value in labels.items()}
        loss, rms, varying_count = _loss_and_item_rms(
            model,
            batch_inputs,
            batch_labels,
            objective,
            objective_config,
            training=False,
        )
        weighted_losses.append(loss * tf.cast(stop - start, tf.float32))
        item_rms.append(rms)
        varying_counts.append(varying_count)
    return (
        tf.add_n(weighted_losses) / tf.cast(row_count, tf.float32),
        _median(tf.concat(item_rms, axis=0)),
        tf.add_n(varying_counts),
    )


def run_v5_memorization_gate(
    dataset: JoinedNumpyDataset,
    model_factory: Callable[[], tf.keras.Model],
    *,
    objective: Objective | None = None,
    objective_config: object | None = None,
    config: V5MemorizationGateConfig = V5MemorizationGateConfig(),
) -> tuple[tf.keras.Model, V5MemorizationGateResult]:
    """Run a deterministic bounded-memory local-MDN memorization diagnostic."""

    if not isinstance(config, V5MemorizationGateConfig):
        raise TypeError("config must be V5MemorizationGateConfig")
    if objective is None or objective_config is None:
        from .training_objective_v5 import (
            V5CandidateObjectiveConfig,
            compute_v5_candidate_training_objective,
        )

        objective = objective or compute_v5_candidate_training_objective
        objective_config = objective_config or V5CandidateObjectiveConfig(
            search_yield_weight=0.0,
            pairwise_ranking_weight=0.0,
            local_mdn_weight=config.local_mdn_weight,
            local_coverage_weight=config.local_coverage_weight,
            operational_top_l_alignment_weight=(
                config.operational_top_l_alignment_weight
            ),
        )
    if not callable(model_factory) or not callable(objective):
        raise TypeError("model_factory and objective must be callable")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    model = model_factory()
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model_factory must return a tf.keras.Model")
    input_keys = _model_input_names(model)
    raw_inputs, raw_labels = dataset.joined_numpy(include_unverified=False)
    missing_inputs = tuple(name for name in input_keys if name not in raw_inputs)
    if missing_inputs:
        raise ValueError(f"dataset is missing model inputs: {missing_inputs}")
    missing_labels = tuple(
        name for name in CANDIDATE_SUPERVISION_TENSOR_KEYS if name not in raw_labels
    )
    if missing_labels:
        raise ValueError(f"dataset is missing V5 labels: {missing_labels}")
    has_target = np.asarray(raw_labels["has_local_target"], dtype=np.bool_)
    varying = np.asarray(raw_labels["varying_dimension_mask"], dtype=np.bool_)
    learnable = has_target & (np.sum(varying, axis=-1) > 0)
    learnable_indices = np.flatnonzero(learnable).astype(np.int32, copy=False)
    if learnable_indices.size < 1:
        raise ValueError("gate needs at least one learnable local target")
    target_count = int(np.count_nonzero(has_target))
    learnable_target_count = int(learnable_indices.size)
    fully_fixed_target_count = target_count - learnable_target_count
    # Fixed queries have no density target. Avoid forwarding them through the
    # model, while retaining their audited cardinality in the result.
    inputs = {
        name: tf.gather(tf.convert_to_tensor(raw_inputs[name]), learnable_indices)
        for name in input_keys
    }
    labels = {
        name: tf.gather(tf.convert_to_tensor(raw_labels[name]), learnable_indices)
        for name in CANDIDATE_SUPERVISION_TENSOR_KEYS
    }
    if config.learning_rate_schedule == "constant":
        learning_rate: float | tf.keras.optimizers.schedules.LearningRateSchedule = (
            config.learning_rate
        )
    else:
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=config.steps,
            alpha=config.final_learning_rate / config.learning_rate,
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    initial_loss, initial_rms, varying_count = _batched_metrics(
        model,
        inputs,
        labels,
        objective,
        objective_config,
        batch_size=config.batch_size,
    )
    trajectory = [float(initial_loss.numpy())]
    effective_batch_size = min(config.batch_size, learnable_target_count)
    for step in range(config.steps):
        offset = (step * effective_batch_size) % learnable_target_count
        batch_indices = tf.math.floormod(
            tf.range(offset, offset + effective_batch_size, dtype=tf.int32),
            learnable_target_count,
        )
        batch_inputs = {name: tf.gather(value, batch_indices) for name, value in inputs.items()}
        batch_labels = {name: tf.gather(value, batch_indices) for name, value in labels.items()}
        with tf.GradientTape() as tape:
            loss, _, _ = _loss_and_item_rms(
                model,
                batch_inputs,
                batch_labels,
                objective,
                objective_config,
                training=True,
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(value is None for value in gradients):
            raise RuntimeError("memorization objective is disconnected from trainable variables")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        trajectory.append(float(loss.numpy()))
    final_loss, final_rms, final_varying_count = _batched_metrics(
        model,
        inputs,
        labels,
        objective,
        objective_config,
        batch_size=config.batch_size,
    )
    if int(final_varying_count.numpy()) != int(varying_count.numpy()):
        raise RuntimeError("varying-coordinate cardinality changed during optimization")
    initial_loss_value = float(initial_loss.numpy())
    final_loss_value = float(final_loss.numpy())
    initial_rms_value = float(initial_rms.numpy())
    final_rms_value = float(final_rms.numpy())
    trajectory_array = np.asarray([*trajectory, final_loss_value], dtype="<f8")
    trajectory_hash = sha256(trajectory_array.tobytes()).hexdigest()
    passed = bool(
        np.all(np.isfinite(trajectory_array))
        and initial_loss_value - final_loss_value >= config.minimum_loss_reduction
        and final_rms_value <= config.max_final_target_median_rms
    )
    core = {
        "schema_version": V5_MEMORIZATION_GATE_SCHEMA,
        "version": V5_MEMORIZATION_GATE_VERSION,
        "initial_loss": initial_loss_value,
        "final_loss": final_loss_value,
        "initial_target_median_rms": initial_rms_value,
        "final_target_median_rms": final_rms_value,
        "target_count": target_count,
        "learnable_target_count": learnable_target_count,
        "fully_fixed_target_count": fully_fixed_target_count,
        "varying_coordinate_count": int(varying_count.numpy()),
        "model_input_keys": list(input_keys),
        "objective_audit_sha256": _objective_hash(objective_config),
        "trajectory_sha256": trajectory_hash,
        "final_weights_sha256": _weights_sha256(model),
        "passed": passed,
        "config": asdict(config),
        "scientific_role": "wiring_and_memorization_diagnostic_not_model_acceptance",
    }
    result_hash = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return model, V5MemorizationGateResult(
        initial_loss=initial_loss_value,
        final_loss=final_loss_value,
        initial_target_median_rms=initial_rms_value,
        final_target_median_rms=final_rms_value,
        target_count=target_count,
        learnable_target_count=learnable_target_count,
        fully_fixed_target_count=fully_fixed_target_count,
        varying_coordinate_count=int(varying_count.numpy()),
        model_input_keys=input_keys,
        objective_audit_sha256=core["objective_audit_sha256"],
        trajectory_sha256=trajectory_hash,
        final_weights_sha256=core["final_weights_sha256"],
        passed=passed,
        config=config,
        result_sha256=result_hash,
    )


__all__ = [
    "V5_MEMORIZATION_GATE_SCHEMA",
    "V5_MEMORIZATION_GATE_VERSION",
    "V5MemorizationGateConfig",
    "V5MemorizationGateResult",
    "run_v5_memorization_gate",
]
