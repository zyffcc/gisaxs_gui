"""Manifest and loading contract for local-target v2 proposal models."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf

from .local_target import (
    LOCAL_TARGET_CONTRACT,
    TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS,
)
from .local_training_data import LOCAL_DATASET_ADAPTER_VERSION
from .model_v2 import (
    LOCAL_PROPOSAL_MODEL_NAME,
    LOCAL_PROPOSAL_MODEL_VERSION,
    MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS,
    MODEL_OUTPUT_COORDINATE_SEMANTICS,
)
from .proposal_training_audit import (
    CHECKPOINT_DIRECTORY,
    HISTORY_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    _canonical_json,
    _file_sha256,
    _read_json,
    _utc_now,
)
from .training_objective_v2 import (
    CONTINUOUS_REDUCTION_SEMANTICS,
    LOCAL_TRAINING_OBJECTIVE_VERSION,
)


LOCAL_TRAINER_VERSION = "posterior_v8_local_target_npz_trainer_v2"
LOCAL_RUN_MANIFEST_SCHEMA = "gisaxs.posterior_v8.proposal_training_run/v2"
LOCAL_HISTORY_SCHEMA = "gisaxs.posterior_v8.proposal_training_history/v2"
LOCAL_COORDINATE_CONTRACT = {
    **LOCAL_TARGET_CONTRACT,
    "model_input_bounds": MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS,
    "model_output": MODEL_OUTPUT_COORDINATE_SEMANTICS,
    "continuous_reduction": CONTINUOUS_REDUCTION_SEMANTICS,
}


def _scientific_scope(range_semantics: str) -> str:
    if range_semantics != TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS:
        raise ValueError("this trainer only adapts truth-centered Phase-2 global boxes")
    return "engineering_smoke_only_truth_centered_ranges"


def source_hashes() -> dict[str, str]:
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
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/proposal_training_v2.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/local_training_audit.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/local_training_data.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective_v2.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model_v2.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/local_target.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/dataset.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/proposal_training.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/proposal_training_audit.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_catalog.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_codec.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_contract.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/contract.py",
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/preprocessing.py",
    )
    return {name: _file_sha256(root / name) for name in names}


def manifest_payload(config, audit, sources, runtime, range_semantics):
    config_payload = config.to_dict()
    source_aggregate = sha256(_canonical_json(sources)).hexdigest()
    resume_contract = {
        "trainer_version": LOCAL_TRAINER_VERSION,
        "model_version": LOCAL_PROPOSAL_MODEL_VERSION,
        "objective_version": LOCAL_TRAINING_OBJECTIVE_VERSION,
        "dataset_adapter_version": LOCAL_DATASET_ADAPTER_VERSION,
        "coordinate_contract": LOCAL_COORDINATE_CONTRACT,
        "range_construction_semantics": range_semantics,
        "run_config": config_payload,
        "dataset_fingerprint": audit.payload["fingerprint_sha256"],
        "source_sha256_aggregate": source_aggregate,
        "tensorflow_version": runtime["tensorflow_version"],
        "num_replicas": runtime["num_replicas"],
        "mixed_precision_policy": runtime["mixed_precision_policy"],
    }
    return {
        "schema_version": LOCAL_RUN_MANIFEST_SCHEMA,
        "trainer_version": LOCAL_TRAINER_VERSION,
        "created_utc": _utc_now(),
        "model_version": LOCAL_PROPOSAL_MODEL_VERSION,
        "model_name": LOCAL_PROPOSAL_MODEL_NAME,
        "objective_version": LOCAL_TRAINING_OBJECTIVE_VERSION,
        "dataset_adapter_version": LOCAL_DATASET_ADAPTER_VERSION,
        "coordinate_contract": LOCAL_COORDINATE_CONTRACT,
        "range_construction_semantics": range_semantics,
        "scientific_scope": _scientific_scope(range_semantics),
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


def history_template(manifest):
    return {
        "schema_version": LOCAL_HISTORY_SCHEMA,
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


def _owned_run_directory(path: Path) -> bool:
    manifest = path / MANIFEST_FILE
    if not manifest.is_file() or manifest.is_symlink():
        return False
    try:
        return _read_json(manifest).get("schema_version") == LOCAL_RUN_MANIFEST_SCHEMA
    except ValueError:
        return False


def prepare_output(path: Path, *, resume: bool, overwrite: bool) -> None:
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
            raise ValueError("resume requires a local-target v2 training manifest")
        return
    if not entries:
        return
    if not overwrite:
        raise FileExistsError("output directory is not empty; use resume or overwrite")
    if not _owned_run_directory(path):
        raise ValueError("refusing to overwrite a directory not owned by the v2 trainer")
    allowed = {MANIFEST_FILE, HISTORY_FILE, MODEL_FILE, CHECKPOINT_DIRECTORY}
    if any(item.name not in allowed or item.is_symlink() for item in entries):
        raise ValueError("refusing to overwrite a v2 run with unknown or symlink entries")
    for item in entries:
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def validate_resume(manifest_path: Path, expected, history_path: Path):
    actual = _read_json(manifest_path)
    if set(actual) != set(expected) or actual.get("schema_version") != LOCAL_RUN_MANIFEST_SCHEMA:
        raise ValueError("local-target training manifest is incomplete or unsupported")
    for name in set(expected) - {"created_utc", "runtime"}:
        if actual[name] != expected[name]:
            raise ValueError(f"local-target resume contract mismatch in {name}")
    history = _read_json(history_path)
    if set(history) != set(history_template(expected)) or history.get(
        "schema_version"
    ) != LOCAL_HISTORY_SCHEMA:
        raise ValueError("local-target training history is incomplete or unsupported")
    if history["resume_contract_sha256"] != expected["resume_contract_sha256"]:
        raise ValueError("local-target history belongs to a different run contract")
    completed, epochs = history["completed_epochs"], history["epochs"]
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
        raise ValueError("local-target training history epoch state is inconsistent")
    for index, record in enumerate(epochs, start=1):
        if not isinstance(record, dict) or record.get("epoch") != index:
            raise ValueError("local-target history contains a malformed epoch record")
        for split in ("train", "validation"):
            metrics = record.get(split)
            if not isinstance(metrics, dict) or not metrics or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
                for value in metrics.values()
            ):
                raise ValueError("local-target history contains invalid epoch metrics")
    if completed:
        best_epoch, best_loss = history["best_epoch"], history["best_validation_loss"]
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
            raise ValueError("local-target history has invalid best-checkpoint state")
    return history


def validate_local_training_run(path: str | os.PathLike[str]) -> Mapping[str, object]:
    """Validate a complete v2 run and reject global/local semantic mismatch."""

    root = Path(path)
    if root.is_file() and root.name == MODEL_FILE:
        root = root.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("local-target run must be a regular directory")
    manifest_path, history_path, model_path = (
        root / MANIFEST_FILE,
        root / HISTORY_FILE,
        root / MODEL_FILE,
    )
    if any(
        value.is_symlink() or not value.is_file()
        for value in (manifest_path, history_path, model_path)
    ):
        raise ValueError("local-target run is missing a regular manifest, history or model")
    manifest, history = _read_json(manifest_path), _read_json(history_path)
    expected = {
        "schema_version": LOCAL_RUN_MANIFEST_SCHEMA,
        "trainer_version": LOCAL_TRAINER_VERSION,
        "model_version": LOCAL_PROPOSAL_MODEL_VERSION,
        "model_name": LOCAL_PROPOSAL_MODEL_NAME,
        "objective_version": LOCAL_TRAINING_OBJECTIVE_VERSION,
        "dataset_adapter_version": LOCAL_DATASET_ADAPTER_VERSION,
        "coordinate_contract": LOCAL_COORDINATE_CONTRACT,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"local-target run has incompatible {name}")
    range_semantics = manifest.get("range_construction_semantics")
    if (
        range_semantics != TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS
        or manifest.get("scientific_scope")
        != "engineering_smoke_only_truth_centered_ranges"
    ):
        raise ValueError("local-target run has incompatible scientific scope")
    if manifest.get("run_config", {}).get("allow_truth_centered_range_pilot") is not True:
        raise ValueError("truth-centered local-target run lacks its explicit pilot gate")
    if history.get("schema_version") != LOCAL_HISTORY_SCHEMA or history.get(
        "status"
    ) != "complete":
        raise ValueError("local-target training history is not complete")
    if history.get("resume_contract_sha256") != manifest.get("resume_contract_sha256"):
        raise ValueError("local-target manifest and history contracts disagree")
    digest = history.get("best_model_sha256")
    if not isinstance(digest, str) or _file_sha256(model_path) != digest:
        raise ValueError("local-target model checksum does not match training history")
    return MappingProxyType(manifest)


def load_local_trained_proposal_model(path: str | os.PathLike[str]) -> tf.keras.Model:
    """Load only a manifest-bound local-target v2 model."""

    root = Path(path)
    if root.is_file():
        root = root.parent
    validate_local_training_run(root)
    model_path = root / MODEL_FILE
    model = tf.keras.models.load_model(model_path, safe_mode=True, compile=False)
    input_names = {value.name.split(":", 1)[0] for value in model.inputs}
    output_names = set(model.output_names)
    expected_inputs = {
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
    expected_outputs = {
        "topology_logits",
        "branch_pattern_logits",
        "mixture_logits",
        "mixture_loc",
        "mixture_logscale",
    }
    if (
        model.name != LOCAL_PROPOSAL_MODEL_NAME
        or input_names != expected_inputs
        or output_names != expected_outputs
    ):
        raise ValueError("saved model identity or graph is not local-target v2")
    model.posterior_v8_model_version = LOCAL_PROPOSAL_MODEL_VERSION
    model.posterior_v8_input_bounds_coordinate_semantics = (
        MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS
    )
    model.posterior_v8_output_coordinate_semantics = MODEL_OUTPUT_COORDINATE_SEMANTICS
    return model


__all__ = [
    "LOCAL_COORDINATE_CONTRACT",
    "LOCAL_HISTORY_SCHEMA",
    "LOCAL_RUN_MANIFEST_SCHEMA",
    "LOCAL_TRAINER_VERSION",
    "history_template",
    "load_local_trained_proposal_model",
    "manifest_payload",
    "prepare_output",
    "source_hashes",
    "validate_local_training_run",
    "validate_resume",
]
