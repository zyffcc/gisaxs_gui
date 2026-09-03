"""Dataset provenance and crash-safe run artifacts for proposal training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .dataset import (
    DATASET_GENERATOR_VERSION,
    DATASET_SCHEMA_VERSION,
    PILOT_PHASE,
    SPLIT_CODE,
    NumpyShard,
    load_shard,
)
from .model import MODEL_VERSION
from .training_objective import TRAINING_OBJECTIVE_VERSION


TRAINER_VERSION = "posterior_v8_phase2_npz_trainer_v1"
RUN_MANIFEST_SCHEMA = "gisaxs.posterior_v8.proposal_training_run/v1"
HISTORY_SCHEMA = "gisaxs.posterior_v8.proposal_training_history/v1"
DATASET_AUDIT_SCHEMA = "gisaxs.posterior_v8.training_dataset_audit/v1"
MANIFEST_FILE = "training_manifest.json"
HISTORY_FILE = "history.json"
MODEL_FILE = "model.keras"
CHECKPOINT_DIRECTORY = "checkpoints"
_CHECKPOINT_RE = re.compile(r"epoch-(\d{6})\Z")
_BEST_MODEL_RE = re.compile(r"best-epoch-(\d{6})\.keras\Z")
_EXPECTED_INPUTS = {
    "x",
    "point_mask",
    "global_features",
    "branch_topology",
    "branch_d_present",
    "branch_resolution_present",
    "branch_low",
    "branch_high",
    "active_dimension_mask",
}
_EXPECTED_OUTPUTS = {
    "topology_logits",
    "branch_pattern_logits",
    "mixture_logits",
    "mixture_loc",
    "mixture_logscale",
}


@dataclass(frozen=True)
class DatasetAudit:
    payload: Mapping[str, object]
    shard_paths: tuple[Path, ...]
    train_count: int
    validation_count: int
    calibration_count: int
    test_count: int
    max_points: int

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("training audit is not finite canonical JSON") from exc
    return text.encode("utf-8")


def _file_sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object, *, exclusive: bool = False) -> None:
    data = _canonical_json(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite {path}") from exc
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"could not read training JSON {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"training JSON must contain one object: {path}")
    return result


def inspect_phase2_shards(
    shard_paths: Sequence[str | os.PathLike[str]],
    *,
    shard_loader: Callable[[str | os.PathLike[str]], NumpyShard] | None = None,
) -> DatasetAudit:
    """Validate all shards and return content-bound split/provenance audit."""

    if not shard_paths:
        raise ValueError("at least one Phase-2 NPZ shard is required")
    loader = load_shard if shard_loader is None else shard_loader
    paths = tuple(sorted(Path(value).resolve() for value in shard_paths))
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate shard paths are not allowed")
    identity = None
    rows_seen: set[tuple[int, int]] = set()
    records = []
    split_counts = {name: 0 for name in SPLIT_CODE}
    max_points = None
    for path in paths:
        if path.suffix != ".npz" or path.is_symlink() or not path.is_file():
            raise ValueError(f"shard must be a regular .npz file: {path}")
        metadata_path = path.with_suffix(".json")
        shard = loader(path)
        metadata = shard.metadata
        if _file_sha256(path) != metadata.get("npz_sha256"):
            raise ValueError("shard checksum changed during training audit")
        current_identity = {
            "dataset_schema_version": metadata.get("dataset_schema_version"),
            "dataset_generator_version": metadata.get("dataset_generator_version"),
            "phase": metadata.get("phase"),
            "versions": metadata.get("versions"),
            "config": metadata.get("config"),
            "preprocessing_contract": metadata.get("preprocessing_contract"),
            "pilot_limitations": metadata.get("pilot_limitations"),
            "source_sha256_aggregate": metadata.get("source_sha256_aggregate"),
        }
        if current_identity["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
            raise ValueError("trainer requires the current Phase-2 dataset schema")
        if current_identity["dataset_generator_version"] != DATASET_GENERATOR_VERSION:
            raise ValueError("trainer requires the current Phase-2 dataset generator")
        if current_identity["phase"] != PILOT_PHASE:
            raise ValueError("trainer only accepts the declared Phase-2 pilot")
        if identity is None:
            identity = current_identity
        elif _canonical_json(identity) != _canonical_json(current_identity):
            raise ValueError("shards do not share one dataset/source identity")
        points = int(shard.arrays["x"].shape[1])
        max_points = points if max_points is None else max_points
        if points != max_points:
            raise ValueError("shards disagree on padded curve length")
        for recipe, view in zip(shard.arrays["recipe_index"], shard.arrays["view_index"]):
            key = (int(recipe), int(view))
            if key in rows_seen:
                raise ValueError("duplicate recipe_index/view_index across shards")
            rows_seen.add(key)
        shard_splits = {}
        for name, code in SPLIT_CODE.items():
            count = int(np.count_nonzero(shard.arrays["assigned_split"] == code))
            split_counts[name] += count
            shard_splits[name] = count
        if sum(shard_splits.values()) != shard.sample_count:
            raise ValueError("shard contains an unknown split code")
        records.append(
            {
                "path": str(path),
                "npz_sha256": metadata["npz_sha256"],
                "metadata_sha256": _file_sha256(metadata_path),
                "sample_count": shard.sample_count,
                "split_counts": shard_splits,
                "recipe_start": int(np.min(shard.arrays["recipe_index"])),
                "recipe_stop_exclusive": int(np.max(shard.arrays["recipe_index"])) + 1,
            }
        )
    assert identity is not None and max_points is not None
    if split_counts["train"] == 0 or split_counts["validation"] == 0:
        raise ValueError("both train and validation splits must contain samples")
    fingerprint_payload = {"identity": identity, "shards": records}
    payload = {
        "schema_version": DATASET_AUDIT_SCHEMA,
        "identity": identity,
        "shards": records,
        "shard_count": len(records),
        "sample_count": sum(split_counts.values()),
        "split_counts": split_counts,
        "max_points": max_points,
        "fingerprint_sha256": sha256(_canonical_json(fingerprint_payload)).hexdigest(),
    }
    return DatasetAudit(
        MappingProxyType(payload),
        paths,
        split_counts["train"],
        split_counts["validation"],
        split_counts["calibration"],
        split_counts["test"],
        max_points,
    )


def _source_hashes() -> dict[str, str]:
    root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "src").is_dir() and (parent / "utils").is_dir()
        ),
        None,
    )
    if root is None:
        raise RuntimeError("could not locate Posterior V8 source root")
    names = (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/proposal_training.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/proposal_training_audit.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/dataset.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_catalog.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_codec.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/contract.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/preprocessing.py",
    )
    return {name: _file_sha256(root / name) for name in names}


def _runtime_audit(strategy: tf.distribute.Strategy, mixed_precision: bool):
    devices = tf.config.list_logical_devices()
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "tensorflow_version": tf.__version__,
        "host": platform.node(),
        "strategy": type(strategy).__name__,
        "num_replicas": int(strategy.num_replicas_in_sync),
        "logical_devices": [f"{item.device_type}:{item.name}" for item in devices],
        "mixed_precision_policy": "mixed_float16" if mixed_precision else "float32",
    }


def _require_tensorflow_215() -> None:
    if not re.match(r"^2\.15(?:\.|$)", tf.__version__):
        raise RuntimeError(f"Posterior V8 trainer requires TensorFlow 2.15.x, got {tf.__version__}")


def _owned_run_directory(path: Path) -> bool:
    manifest = path / MANIFEST_FILE
    if not manifest.is_file() or manifest.is_symlink():
        return False
    try:
        return _read_json(manifest).get("schema_version") == RUN_MANIFEST_SCHEMA
    except ValueError:
        return False


def _prepare_output(path: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if path.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if not path.exists():
        if resume:
            raise FileNotFoundError("cannot resume a missing training run")
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise ValueError("output path must be a directory")
    entries = tuple(path.iterdir())
    if resume:
        if not _owned_run_directory(path):
            raise ValueError("resume requires a valid Posterior V8 training manifest")
        return
    if not entries:
        return
    if not overwrite:
        raise FileExistsError("output directory is not empty; use resume or overwrite")
    if not _owned_run_directory(path):
        raise ValueError("refusing to overwrite a directory not owned by this trainer")
    allowed = {MANIFEST_FILE, HISTORY_FILE, MODEL_FILE, CHECKPOINT_DIRECTORY}
    unexpected = sorted(item.name for item in entries if item.name not in allowed)
    if unexpected:
        raise ValueError(f"refusing to overwrite run with unknown entries: {unexpected}")
    for item in entries:
        if item.is_symlink():
            raise ValueError("refusing to overwrite a run containing symlinks")
    for item in entries:
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def _manifest_payload(config, audit, sources, runtime):
    config_payload = config.to_dict()
    source_aggregate = sha256(_canonical_json(sources)).hexdigest()
    resume_contract = {
        "trainer_version": TRAINER_VERSION,
        "model_version": MODEL_VERSION,
        "objective_version": TRAINING_OBJECTIVE_VERSION,
        "run_config": config_payload,
        "dataset_fingerprint": audit.payload["fingerprint_sha256"],
        "source_sha256_aggregate": source_aggregate,
        "tensorflow_version": runtime["tensorflow_version"],
        "num_replicas": runtime["num_replicas"],
        "mixed_precision_policy": runtime["mixed_precision_policy"],
    }
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "trainer_version": TRAINER_VERSION,
        "created_utc": _utc_now(),
        "model_version": MODEL_VERSION,
        "objective_version": TRAINING_OBJECTIVE_VERSION,
        "run_config": config_payload,
        "run_config_sha256": sha256(_canonical_json(config_payload)).hexdigest(),
        "dataset_audit": audit.to_dict(),
        "source_sha256": sources,
        "source_sha256_aggregate": source_aggregate,
        "runtime": runtime,
        "resume_contract_sha256": sha256(_canonical_json(resume_contract)).hexdigest(),
        "best_model_file": MODEL_FILE,
        "history_file": HISTORY_FILE,
        "checkpoint_directory": CHECKPOINT_DIRECTORY,
    }


def _history_template(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": HISTORY_SCHEMA,
        "resume_contract_sha256": manifest["resume_contract_sha256"],
        "status": "running",
        "updated_utc": _utc_now(),
        "completed_epochs": 0,
        "target_epochs": manifest["run_config"]["epochs"],
        "best_epoch": 0,
        "best_validation_loss": None,
        "best_checkpoint_file": None,
        "best_model_sha256": None,
        "last_checkpoint_prefix": None,
        "failure": None,
        "epochs": [],
    }


def _validate_resume(manifest_path: Path, expected, history_path: Path):
    actual = _read_json(manifest_path)
    required = set(expected)
    if set(actual) != required or actual["schema_version"] != RUN_MANIFEST_SCHEMA:
        raise ValueError("training manifest schema is incomplete or unsupported")
    for name in required - {"created_utc", "runtime"}:
        if actual[name] != expected[name]:
            raise ValueError(f"resume contract mismatch in {name}")
    history = _read_json(history_path)
    expected_history = set(_history_template(expected))
    if set(history) != expected_history or history["schema_version"] != HISTORY_SCHEMA:
        raise ValueError("training history schema is incomplete or unsupported")
    if history["resume_contract_sha256"] != expected["resume_contract_sha256"]:
        raise ValueError("training history belongs to a different run contract")
    completed = history["completed_epochs"]
    epochs = history["epochs"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or not isinstance(epochs, list)
        or completed != len(epochs)
        or completed > expected["run_config"]["epochs"]
        or history["target_epochs"] != expected["run_config"]["epochs"]
        or history["status"] not in {"running", "failed", "complete"}
    ):
        raise ValueError("training history epoch state is inconsistent")
    for index, record in enumerate(epochs, start=1):
        if not isinstance(record, dict) or record.get("epoch") != index:
            raise ValueError("training history contains a malformed epoch record")
        for split in ("train", "validation"):
            metrics = record.get(split)
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("training history is missing epoch metrics")
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
                for value in metrics.values()
            ):
                raise ValueError("training history contains a non-finite metric")
    if completed:
        best_epoch = history["best_epoch"]
        best_loss = history["best_validation_loss"]
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or not 1 <= best_epoch <= completed
            or isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
            or not isfinite(float(best_loss))
            or not isinstance(history["best_checkpoint_file"], str)
            or not isinstance(history["best_model_sha256"], str)
            or len(history["best_model_sha256"]) != 64
        ):
            raise ValueError("training history has invalid best-checkpoint state")
    return history


def _save_model_atomic(model: tf.keras.Model, path: Path) -> str:
    temporary = path.parent / f".{path.stem}-{os.getpid()}-{time.time_ns()}.keras"
    try:
        model.save(temporary)
        digest = _file_sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def load_trained_proposal_model(path: str | os.PathLike[str]) -> tf.keras.Model:
    """Cold-process-safe loader that registers and verifies the V8 graph contract."""

    model_path = Path(path)
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("proposal model must be a regular non-symlink .keras file")
    model = tf.keras.models.load_model(model_path, safe_mode=True, compile=False)
    input_names = {value.name.split(":", 1)[0] for value in model.inputs}
    output_names = set(model.output_names)
    if (
        model.name != "posterior_v8_proposal"
        or input_names != _EXPECTED_INPUTS
        or output_names != _EXPECTED_OUTPUTS
    ):
        raise ValueError("saved proposal model has an incompatible input/output contract")
    return model


def _copy_atomic(source: Path, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_prefix(output: Path, epoch: int) -> Path:
    return output / CHECKPOINT_DIRECTORY / f"epoch-{epoch:06d}" / "ckpt"


def _restore_checkpoint(checkpoint, output: Path, history):
    relative = history["last_checkpoint_prefix"]
    if relative is None:
        if history["completed_epochs"] != 0:
            raise ValueError("completed history is missing its training checkpoint")
        return
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("history contains an unsafe checkpoint path")
    prefix = output / relative
    if not prefix.with_suffix(".index").is_file():
        raise ValueError("resume checkpoint is missing")
    # ``write``/``read`` deliberately omit Checkpoint.save_counter.
    status = checkpoint.read(str(prefix))
    status.assert_consumed()


def _prune_checkpoints(output: Path, history, keep: int) -> None:
    directory = output / CHECKPOINT_DIRECTORY
    if not directory.exists():
        return
    completed = int(history["completed_epochs"])
    keep_epochs = set(range(max(1, completed - keep + 1), completed + 1))
    best_file = history["best_checkpoint_file"]
    for item in directory.iterdir():
        epoch_match = _CHECKPOINT_RE.fullmatch(item.name)
        best_match = _BEST_MODEL_RE.fullmatch(item.name)
        if item.is_symlink():
            raise ValueError("checkpoint directory must not contain symlinks")
        if epoch_match and item.is_dir():
            if int(epoch_match.group(1)) not in keep_epochs:
                shutil.rmtree(item)
        elif best_match and item.is_file():
            if str(Path(CHECKPOINT_DIRECTORY) / item.name) != best_file:
                item.unlink()
        else:
            raise ValueError(f"unknown checkpoint artifact {item.name!r}")


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
    "inspect_phase2_shards",
    "load_trained_proposal_model",
]
