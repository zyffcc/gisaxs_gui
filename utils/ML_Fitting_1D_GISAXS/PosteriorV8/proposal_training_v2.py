"""Crash-safe local-target v2 trainer for Posterior V8.

The frozen global-target v1 trainer remains untouched. This entry point uses a
separate model identity, objective, manifest schema, and dataset adapter so an
old global-coordinate artifact cannot silently enter the local runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Sequence

import numpy as np
import tensorflow as tf

from .dataset import NumpyShard, load_shard
from .local_training_audit import (
    LOCAL_COORDINATE_CONTRACT,
    LOCAL_HISTORY_SCHEMA,
    LOCAL_RUN_MANIFEST_SCHEMA,
    LOCAL_TRAINER_VERSION,
    history_template,
    load_local_trained_proposal_model,
    manifest_payload,
    prepare_output,
    source_hashes,
    validate_local_training_run,
    validate_resume,
)
from .local_training_data import (
    LOCAL_DATASET_ADAPTER_VERSION,
    epoch_dataset,
    require_scientific_data_contract,
    run_hybrid_validation,
    training_steps,
)
from .model_v2 import LOCAL_PROPOSAL_MODEL_NAME, build_local_proposal_model
from .proposal_training import (
    ProposalTrainingResult,
    _optional_positive_integer,
    _positive_float,
    _positive_integer,
    _run_epoch,
    inspect_phase2_shards,
)
from .proposal_training_audit import (
    CHECKPOINT_DIRECTORY,
    HISTORY_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    DatasetAudit,
    _checkpoint_prefix,
    _copy_atomic,
    _file_sha256,
    _prune_checkpoints,
    _require_tensorflow_215,
    _restore_checkpoint,
    _runtime_audit,
    _save_model_atomic,
    _utc_now,
    _write_json_atomic,
)
from .training_objective_v2 import LocalTrainingObjectiveConfig


@dataclass(frozen=True)
class LocalProposalTrainingConfig:
    epochs: int = 50
    global_batch_size: int = 64
    learning_rate: float = 1.0e-4
    seed: int = 20260902
    max_points: int = 1000
    width: int = 128
    encoder_blocks: int = 6
    mixture_components: int = 12
    shuffle_buffer: int = 8192
    gradient_clip_norm: float = 10.0
    mixed_precision: bool = False
    deterministic_ops: bool = True
    checkpoint_keep: int = 2
    steps_per_epoch: int | None = None
    objective: LocalTrainingObjectiveConfig = field(
        default_factory=LocalTrainingObjectiveConfig
    )
    allow_truth_centered_range_pilot: bool = False

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
        if not isinstance(self.mixed_precision, bool) or not isinstance(
            self.deterministic_ops, bool
        ):
            raise TypeError("mixed_precision and deterministic_ops must be boolean")
        if not isinstance(self.allow_truth_centered_range_pilot, bool):
            raise TypeError("allow_truth_centered_range_pilot must be boolean")
        if not isinstance(self.objective, LocalTrainingObjectiveConfig):
            raise TypeError("objective must be a LocalTrainingObjectiveConfig")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def train_local_proposal(
    shard_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    config: LocalProposalTrainingConfig = LocalProposalTrainingConfig(),
    *,
    resume: bool = False,
    overwrite: bool = False,
    shard_loader: Callable[[str | os.PathLike[str]], NumpyShard] | None = None,
    strategy: tf.distribute.Strategy | None = None,
    model_builder: Callable[..., tf.keras.Model] | None = None,
) -> ProposalTrainingResult:
    """Train local-target v2 without changing the frozen v1 trainer contract."""

    if not isinstance(config, LocalProposalTrainingConfig):
        raise TypeError("config must be a LocalProposalTrainingConfig")
    if not isinstance(resume, bool) or not isinstance(overwrite, bool):
        raise TypeError("resume and overwrite must be boolean")
    _require_tensorflow_215()
    output = Path(output_dir).resolve()
    loader = load_shard if shard_loader is None else shard_loader
    audit = inspect_phase2_shards(shard_paths, shard_loader=loader)
    range_semantics = require_scientific_data_contract(
        audit,
        allow_truth_centered_range_pilot=config.allow_truth_centered_range_pilot,
    )
    if audit.max_points != config.max_points:
        raise ValueError("run max_points does not match the Phase-2 shard tensors")
    if config.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(config.seed)
    distribution = tf.distribute.MirroredStrategy() if strategy is None else strategy
    replicas = int(distribution.num_replicas_in_sync)
    if config.global_batch_size % replicas:
        raise ValueError("global_batch_size must be divisible by the replica count")
    available_train_steps = audit.train_count // config.global_batch_size
    train_steps = (
        available_train_steps if config.steps_per_epoch is None else config.steps_per_epoch
    )
    if train_steps < 1 or train_steps > available_train_steps:
        raise ValueError("requested train steps exceed available full split batches")
    runtime = _runtime_audit(distribution, config.mixed_precision)
    manifest = manifest_payload(config, audit, source_hashes(), runtime, range_semantics)
    prepare_output(output, resume=resume, overwrite=overwrite)
    manifest_path, history_path = output / MANIFEST_FILE, output / HISTORY_FILE
    if resume:
        history = validate_resume(manifest_path, manifest, history_path)
    else:
        _write_json_atomic(manifest_path, manifest, exclusive=True)
        history = history_template(manifest)
        _write_json_atomic(history_path, history, exclusive=True)
    (output / CHECKPOINT_DIRECTORY).mkdir(exist_ok=True)

    previous_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if config.mixed_precision else "float32"
    )
    try:
        with distribution.scope():
            builder = build_local_proposal_model if model_builder is None else model_builder
            model = builder(
                max_points=config.max_points,
                width=config.width,
                encoder_blocks=config.encoder_blocks,
                mixture_components=config.mixture_components,
            )
            if model.name != LOCAL_PROPOSAL_MODEL_NAME:
                raise ValueError("v2 model_builder must return the local-target model identity")
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
                raise ValueError("checkpoint epoch disagrees with local-target history")
            if history["best_validation_loss"] is not None and not np.isclose(
                float(best_loss_variable.numpy()),
                float(history["best_validation_loss"]),
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError("checkpoint best loss disagrees with local-target history")
            if history["best_checkpoint_file"] is not None:
                best_path = output / history["best_checkpoint_file"]
                if _file_sha256(best_path) != history["best_model_sha256"]:
                    raise ValueError("best local-target checkpoint checksum mismatch")
                _copy_atomic(best_path, output / MODEL_FILE)
            _prune_checkpoints(output, history, config.checkpoint_keep)
            history["status"], history["failure"] = "running", None
            history["updated_utc"] = _utc_now()
            _write_json_atomic(history_path, history)

        train_step, distributed_validation_step, coordinator_validation_step = (
            training_steps(
                distribution,
                model,
                optimizer,
                config.objective,
                config.gradient_clip_norm,
            )
        )
        for epoch_index in range(int(history["completed_epochs"]), config.epochs):
            train_data = distribution.experimental_distribute_dataset(
                epoch_dataset(
                    audit,
                    config,
                    "train",
                    epoch=epoch_index,
                    shard_loader=loader,
                    range_semantics=range_semantics,
                )
            )
            validation_data = epoch_dataset(
                audit,
                config,
                "validation",
                epoch=epoch_index,
                shard_loader=loader,
                range_semantics=range_semantics,
            )
            train_metrics, train_seconds = _run_epoch(train_data, train_step, train_steps)
            validation_metrics, validation_seconds, validation_audit = run_hybrid_validation(
                validation_data,
                strategy=distribution,
                global_batch_size=config.global_batch_size,
                expected_examples=audit.validation_count,
                distributed_step=distributed_validation_step,
                coordinator_step=coordinator_validation_step,
            )
            epoch_number = epoch_index + 1
            validation_loss = validation_metrics["loss"]
            if history["best_validation_loss"] is None or validation_loss < float(
                history["best_validation_loss"]
            ):
                best_relative = str(
                    Path(CHECKPOINT_DIRECTORY) / f"best-epoch-{epoch_number:06d}.keras"
                )
                history["best_model_sha256"] = _save_model_atomic(
                    model, output / best_relative
                )
                history["best_epoch"] = epoch_number
                history["best_validation_loss"] = validation_loss
                history["best_checkpoint_file"] = best_relative
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
                "train_examples": train_steps * config.global_batch_size,
                "train_examples_not_used": audit.train_count
                - train_steps * config.global_batch_size,
                **validation_audit,
                "validation_examples_not_used": 0,
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
            _copy_atomic(output / history["best_checkpoint_file"], output / MODEL_FILE)
            if _file_sha256(output / MODEL_FILE) != history["best_model_sha256"]:
                raise RuntimeError("published local model checksum disagrees with checkpoint")
            _prune_checkpoints(output, history, config.checkpoint_keep)
        load_local_trained_proposal_model(output)
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
    "LOCAL_COORDINATE_CONTRACT",
    "LOCAL_DATASET_ADAPTER_VERSION",
    "LOCAL_HISTORY_SCHEMA",
    "LOCAL_RUN_MANIFEST_SCHEMA",
    "LOCAL_TRAINER_VERSION",
    "DatasetAudit",
    "LocalProposalTrainingConfig",
    "load_local_trained_proposal_model",
    "train_local_proposal",
    "validate_local_training_run",
]
