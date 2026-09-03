"""Reproducible TF 2.15 trainer for Posterior V8 Phase-2 NPZ shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from numbers import Integral
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .dataset import NumpyShard, load_shard
from .model import (
    BRANCH_DIM,
    DEFAULT_MAX_POINTS,
    DEFAULT_MIXTURE_COMPONENTS,
    GLOBAL_FEATURE_DIM,
    POINT_FEATURE_DIM,
    build_proposal_model,
)
from .proposal_training_audit import (
    CHECKPOINT_DIRECTORY,
    DATASET_AUDIT_SCHEMA,
    HISTORY_FILE,
    HISTORY_SCHEMA,
    MANIFEST_FILE,
    MODEL_FILE,
    RUN_MANIFEST_SCHEMA,
    TRAINER_VERSION,
    DatasetAudit,
    _checkpoint_prefix,
    _copy_atomic,
    _file_sha256,
    _history_template,
    _manifest_payload,
    _prepare_output,
    _prune_checkpoints,
    _require_tensorflow_215,
    _restore_checkpoint,
    _runtime_audit,
    _save_model_atomic,
    _source_hashes,
    _utc_now,
    _validate_resume,
    _write_json_atomic,
    inspect_phase2_shards,
    load_trained_proposal_model,
)
from .training_objective import TrainingObjectiveConfig, compute_training_objective


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _optional_positive_integer(value: int | None, name: str) -> int | None:
    return None if value is None else _positive_integer(value, name)


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class ProposalTrainingConfig:
    """Immutable hyperparameters bound into every training run manifest."""

    epochs: int = 50
    global_batch_size: int = 64
    learning_rate: float = 1.0e-4
    seed: int = 20260902
    max_points: int = DEFAULT_MAX_POINTS
    width: int = 128
    encoder_blocks: int = 6
    mixture_components: int = DEFAULT_MIXTURE_COMPONENTS
    shuffle_buffer: int = 8192
    gradient_clip_norm: float = 10.0
    mixed_precision: bool = False
    deterministic_ops: bool = True
    checkpoint_keep: int = 2
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    objective: TrainingObjectiveConfig = field(default_factory=TrainingObjectiveConfig)

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "global_batch_size",
            "max_points",
            "width",
            "encoder_blocks",
            "mixture_components",
            "shuffle_buffer",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        keep = _positive_integer(self.checkpoint_keep, "checkpoint_keep")
        if keep < 2:
            raise ValueError("checkpoint_keep must be at least two for crash-safe resume")
        object.__setattr__(self, "checkpoint_keep", keep)
        object.__setattr__(
            self,
            "steps_per_epoch",
            _optional_positive_integer(self.steps_per_epoch, "steps_per_epoch"),
        )
        object.__setattr__(
            self,
            "validation_steps",
            _optional_positive_integer(self.validation_steps, "validation_steps"),
        )
        object.__setattr__(
            self, "learning_rate", _positive_float(self.learning_rate, "learning_rate")
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _positive_float(self.gradient_clip_norm, "gradient_clip_norm"),
        )
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if not 0 <= int(self.seed) < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        object.__setattr__(self, "seed", int(self.seed))
        if not isinstance(self.mixed_precision, bool):
            raise TypeError("mixed_precision must be boolean")
        if not isinstance(self.deterministic_ops, bool):
            raise TypeError("deterministic_ops must be boolean")
        if not isinstance(self.objective, TrainingObjectiveConfig):
            raise TypeError("objective must be a TrainingObjectiveConfig")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalTrainingResult:
    output_dir: Path
    model_path: Path
    manifest_path: Path
    history_path: Path
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    history: tuple[Mapping[str, object], ...]


def _output_signature(max_points: int):
    inputs = {
        "x": tf.TensorSpec((None, max_points, POINT_FEATURE_DIM), tf.float32),
        "point_mask": tf.TensorSpec((None, max_points), tf.bool),
        "global_features": tf.TensorSpec((None, GLOBAL_FEATURE_DIM), tf.float32),
        "branch_topology": tf.TensorSpec((None, 34), tf.float32),
        "branch_d_present": tf.TensorSpec((None, 4), tf.float32),
        "branch_resolution_present": tf.TensorSpec((None, 1), tf.float32),
        "branch_low": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "branch_high": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "active_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
    }
    labels = {
        "topology_id": tf.TensorSpec((None,), tf.int32),
        "branch_pattern_id": tf.TensorSpec((None,), tf.int32),
        "target_unit": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
        "active_dimension_mask": tf.TensorSpec((None, BRANCH_DIM), tf.float32),
    }
    return inputs, labels


def _epoch_dataset(
    audit: DatasetAudit,
    config: ProposalTrainingConfig,
    split: str,
    *,
    epoch: int,
    shard_loader,
):
    if split not in {"train", "validation"}:
        raise ValueError("proposal trainer may only consume train or validation rows")
    records = {item["path"]: item for item in audit.payload["shards"]}

    def batches():
        for path in audit.shard_paths:
            shard = shard_loader(path)
            record = records[str(path)]
            if (
                shard.metadata.get("npz_sha256") != record["npz_sha256"]
                or _file_sha256(path.with_suffix(".json")) != record["metadata_sha256"]
            ):
                raise ValueError("shard provenance changed after run audit")
            inputs, labels = shard.training_data(split=split)
            for name, value in (*inputs.items(), *labels.items()):
                array = np.asarray(value)
                if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                    raise ValueError(f"training tensor {name!r} contains NaN/Inf")
            for start in range(0, labels["topology_id"].shape[0], 1024):
                stop = min(start + 1024, labels["topology_id"].shape[0])
                yield (
                    {name: np.asarray(value[start:stop]) for name, value in inputs.items()},
                    {name: np.asarray(value[start:stop]) for name, value in labels.items()},
                )

    dataset = tf.data.Dataset.from_generator(
        batches,
        output_signature=_output_signature(config.max_points),
    ).unbatch()
    if split == "train":
        dataset = dataset.shuffle(
            min(config.shuffle_buffer, audit.train_count),
            seed=(config.seed + epoch) % (2**31 - 1),
            reshuffle_each_iteration=False,
        )
    dataset = dataset.batch(
        config.global_batch_size,
        drop_remainder=split == "train",
    )
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return dataset.with_options(options).prefetch(1)


def _steps(config: ProposalTrainingConfig, audit: DatasetAudit):
    available_train = audit.train_count // config.global_batch_size
    available_validation = (
        audit.validation_count + config.global_batch_size - 1
    ) // config.global_batch_size
    if available_train < 1 or available_validation < 1:
        raise ValueError("train must contain a full batch and validation must be non-empty")
    train = available_train if config.steps_per_epoch is None else config.steps_per_epoch
    validation = (
        available_validation if config.validation_steps is None else config.validation_steps
    )
    if train > available_train or validation > available_validation:
        raise ValueError("requested epoch steps exceed available non-repeated split batches")
    return train, validation


def _training_steps(strategy, model, optimizer, objective_config, gradient_clip_norm):
    replicas = tf.cast(strategy.num_replicas_in_sync, tf.float32)
    mixed = isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer)

    def replica_train(inputs, labels):
        with tf.GradientTape() as tape:
            metrics = compute_training_objective(
                model(inputs, training=True), labels, objective_config
            )
            gradient_loss = metrics["loss"] / replicas
            tape_loss = optimizer.get_scaled_loss(gradient_loss) if mixed else gradient_loss
        gradients = tape.gradient(tape_loss, model.trainable_variables)
        if any(value is None for value in gradients):
            raise RuntimeError("training objective left disconnected model variables")
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        # Dynamic loss scaling must see overflowed mixed-precision gradients so
        # it can lower the scale and skip only that update.  Float32 gradients
        # have no legitimate overflow recovery path and therefore fail closed.
        finite_checks = (
            []
            if mixed
            else [
                tf.debugging.assert_all_finite(
                    value.values if isinstance(value, tf.IndexedSlices) else value,
                    "proposal gradient contains NaN/Inf",
                )
                for value in gradients
            ]
        )
        iteration_before = tf.identity(optimizer.iterations)
        with tf.control_dependencies(finite_checks):
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        variable_checks = [
            tf.debugging.assert_all_finite(value, "proposal weight contains NaN/Inf")
            for value in model.trainable_variables
        ]
        with tf.control_dependencies(variable_checks):
            result = {name: tf.identity(value) for name, value in metrics.items()}
            result["optimizer_update_applied"] = tf.cast(
                optimizer.iterations > iteration_before, tf.float32
            )
            return result

    @tf.function
    def train_step(inputs, labels):
        values = strategy.run(replica_train, args=(inputs, labels))
        return {
            name: strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            for name, value in values.items()
        }

    @tf.function
    def validation_step(inputs, labels):
        return compute_training_objective(model(inputs, training=False), labels, objective_config)

    return train_step, validation_step


def _run_epoch(dataset, step_function, step_count: int, *, sample_weighted: bool = False):
    totals = None
    total_weight = 0
    started = time.monotonic()
    completed_steps = 0
    for inputs, labels in dataset:
        if completed_steps >= step_count:
            break
        metrics = {
            name: float(value.numpy()) for name, value in step_function(inputs, labels).items()
        }
        if not all(isfinite(value) for value in metrics.values()):
            raise FloatingPointError("training step returned a NaN/Inf metric")
        if totals is None:
            totals = {name: 0.0 for name in metrics}
        weight = int(np.asarray(labels["topology_id"]).shape[0]) if sample_weighted else 1
        for name, value in metrics.items():
            totals[name] += value * weight
        total_weight += weight
        completed_steps += 1
    if totals is None or completed_steps != step_count:
        raise RuntimeError("training dataset produced fewer batches than its audit declared")
    return (
        {name: value / total_weight for name, value in totals.items()},
        time.monotonic() - started,
    )


def _result(output: Path, history) -> ProposalTrainingResult:
    best_loss = history["best_validation_loss"]
    if best_loss is None:
        raise RuntimeError("training has no validation checkpoint")
    return ProposalTrainingResult(
        output_dir=output,
        model_path=output / MODEL_FILE,
        manifest_path=output / MANIFEST_FILE,
        history_path=output / HISTORY_FILE,
        completed_epochs=int(history["completed_epochs"]),
        best_epoch=int(history["best_epoch"]),
        best_validation_loss=float(best_loss),
        history=tuple(MappingProxyType(dict(value)) for value in history["epochs"]),
    )


def train_proposal(
    shard_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    config: ProposalTrainingConfig = ProposalTrainingConfig(),
    *,
    resume: bool = False,
    overwrite: bool = False,
    shard_loader: Callable[[str | os.PathLike[str]], NumpyShard] | None = None,
    strategy: tf.distribute.Strategy | None = None,
    model_builder: Callable[..., tf.keras.Model] | None = None,
) -> ProposalTrainingResult:
    """Train one proposal model and publish the best validation snapshot."""

    if not isinstance(config, ProposalTrainingConfig):
        raise TypeError("config must be a ProposalTrainingConfig")
    if not isinstance(resume, bool) or not isinstance(overwrite, bool):
        raise TypeError("resume and overwrite must be boolean")
    _require_tensorflow_215()
    output = Path(output_dir).resolve()
    loader = load_shard if shard_loader is None else shard_loader
    audit = inspect_phase2_shards(shard_paths, shard_loader=loader)
    if audit.max_points != config.max_points:
        raise ValueError("run max_points does not match the Phase-2 shard tensors")

    if config.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(config.seed)
    distribution = tf.distribute.MirroredStrategy() if strategy is None else strategy
    replicas = int(distribution.num_replicas_in_sync)
    if config.global_batch_size % replicas:
        raise ValueError("global_batch_size must be divisible by the replica count")
    train_steps, validation_steps = _steps(config, audit)
    sources = _source_hashes()
    runtime = _runtime_audit(distribution, config.mixed_precision)
    manifest = _manifest_payload(config, audit, sources, runtime)
    _prepare_output(output, resume=resume, overwrite=overwrite)
    manifest_path = output / MANIFEST_FILE
    history_path = output / HISTORY_FILE
    if resume:
        history = _validate_resume(manifest_path, manifest, history_path)
    else:
        _write_json_atomic(manifest_path, manifest, exclusive=True)
        history = _history_template(manifest)
        _write_json_atomic(history_path, history, exclusive=True)
    (output / CHECKPOINT_DIRECTORY).mkdir(exist_ok=True)

    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if config.mixed_precision else "float32"
    )
    try:
        with distribution.scope():
            builder = build_proposal_model if model_builder is None else model_builder
            model = builder(
                max_points=config.max_points,
                width=config.width,
                encoder_blocks=config.encoder_blocks,
                mixture_components=config.mixture_components,
            )
            base_optimizer = tf.keras.optimizers.Adam(config.learning_rate)
            optimizer = (
                tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                if config.mixed_precision
                else base_optimizer
            )
            optimizer.build(model.trainable_variables)
            completed_variable = tf.Variable(0, trainable=False, dtype=tf.int64)
            best_loss_variable = tf.Variable(np.inf, trainable=False, dtype=tf.float64)
            checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer,
                completed_epochs=completed_variable,
                best_validation_loss=best_loss_variable,
            )
        if resume:
            _restore_checkpoint(checkpoint, output, history)
            if int(completed_variable.numpy()) != int(history["completed_epochs"]):
                raise ValueError("checkpoint epoch disagrees with training history")
            if history["best_validation_loss"] is not None and not np.isclose(
                float(best_loss_variable.numpy()),
                float(history["best_validation_loss"]),
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError("checkpoint best loss disagrees with training history")
            if history["best_checkpoint_file"] is not None:
                best_path = output / history["best_checkpoint_file"]
                if _file_sha256(best_path) != history["best_model_sha256"]:
                    raise ValueError("best validation checkpoint checksum mismatch")
                _copy_atomic(best_path, output / MODEL_FILE)
            _prune_checkpoints(output, history, config.checkpoint_keep)
            history["status"] = "running"
            history["failure"] = None
            history["updated_utc"] = _utc_now()
            _write_json_atomic(history_path, history)

        train_step, validation_step = _training_steps(
            distribution,
            model,
            optimizer,
            config.objective,
            config.gradient_clip_norm,
        )
        start_epoch = int(history["completed_epochs"])
        for epoch_index in range(start_epoch, config.epochs):
            train_data = distribution.experimental_distribute_dataset(
                _epoch_dataset(
                    audit,
                    config,
                    "train",
                    epoch=epoch_index,
                    shard_loader=loader,
                )
            )
            # Validation runs on the coordinator so its final short batch is
            # evaluated exactly once and epoch means can be sample-weighted.
            validation_data = _epoch_dataset(
                audit,
                config,
                "validation",
                epoch=epoch_index,
                shard_loader=loader,
            )
            train_metrics, train_seconds = _run_epoch(train_data, train_step, train_steps)
            validation_metrics, validation_seconds = _run_epoch(
                validation_data,
                validation_step,
                validation_steps,
                sample_weighted=True,
            )
            epoch_number = epoch_index + 1
            validation_loss = validation_metrics["loss"]
            best_loss = history["best_validation_loss"]
            if best_loss is None or validation_loss < float(best_loss):
                best_relative = str(
                    Path(CHECKPOINT_DIRECTORY) / f"best-epoch-{epoch_number:06d}.keras"
                )
                best_digest = _save_model_atomic(model, output / best_relative)
                history["best_epoch"] = epoch_number
                history["best_validation_loss"] = validation_loss
                history["best_checkpoint_file"] = best_relative
                history["best_model_sha256"] = best_digest
            completed_variable.assign(epoch_number)
            best_loss_variable.assign(float(history["best_validation_loss"]))
            prefix = _checkpoint_prefix(output, epoch_number)
            prefix.parent.mkdir(parents=True, exist_ok=False)
            checkpoint.write(str(prefix))
            record = {
                "epoch": epoch_number,
                "train": train_metrics,
                "validation": validation_metrics,
                "train_steps": train_steps,
                "validation_steps": validation_steps,
                "train_examples": train_steps * config.global_batch_size,
                "validation_examples": min(
                    audit.validation_count,
                    validation_steps * config.global_batch_size,
                ),
                "train_examples_not_used": (
                    audit.train_count - train_steps * config.global_batch_size
                ),
                "validation_examples_not_used": max(
                    0,
                    audit.validation_count - validation_steps * config.global_batch_size,
                ),
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "optimizer_iterations": int(optimizer.iterations.numpy()),
            }
            history["epochs"].append(record)
            history["completed_epochs"] = epoch_number
            history["last_checkpoint_prefix"] = str(prefix.relative_to(output))
            history["updated_utc"] = _utc_now()
            history["status"] = "complete" if epoch_number == config.epochs else "running"
            _write_json_atomic(history_path, history)
            best_path = output / history["best_checkpoint_file"]
            _copy_atomic(best_path, output / MODEL_FILE)
            if _file_sha256(output / MODEL_FILE) != history["best_model_sha256"]:
                raise RuntimeError("published model checksum disagrees with best checkpoint")
            _prune_checkpoints(output, history, config.checkpoint_keep)
        load_trained_proposal_model(output / MODEL_FILE)
        return _result(output, history)
    except Exception as exc:
        if history_path.is_file():
            history["status"] = "failed"
            history["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "utc": _utc_now(),
            }
            history["updated_utc"] = _utc_now()
            try:
                _write_json_atomic(history_path, history)
            except Exception:
                pass
        raise
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "DATASET_AUDIT_SCHEMA",
    "HISTORY_FILE",
    "HISTORY_SCHEMA",
    "MANIFEST_FILE",
    "MODEL_FILE",
    "RUN_MANIFEST_SCHEMA",
    "TRAINER_VERSION",
    "DatasetAudit",
    "ProposalTrainingConfig",
    "ProposalTrainingResult",
    "inspect_phase2_shards",
    "load_trained_proposal_model",
    "train_proposal",
]
