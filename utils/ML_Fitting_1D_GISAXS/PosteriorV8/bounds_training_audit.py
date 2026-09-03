"""Manifest, resume, and strict loading contract for V3 bounds models."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf

from .bounds_first_shards import PILOT_LIMITATIONS, SOLUTION_ONLY_PHASE
from .bounds_model_contract import (
    BOUNDS_MODEL_COORDINATE_CONTRACT,
    BOUNDS_PROPOSAL_MODEL_NAME,
    BOUNDS_PROPOSAL_MODEL_VERSION,
    MODEL_FIXED_DIMENSIONS,
    MODEL_INPUT_KEYS,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
    MODEL_OUTPUT_KEYS,
    MODEL_PHYSICAL_BRANCH_COUNT,
)
from .branch_catalog import BRANCH_PATTERN_COUNT
from .bounds_training_data import (
    BOUNDS_DATASET_ADAPTER_VERSION,
    EXCLUDED_TRAINING_SPLITS,
    TRAINING_SPLITS,
)
from .bounds_training_sources import bounds_training_source_hashes
from .local_target import BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS
from .model_v3 import build_bounds_proposal_model  # noqa: F401 - registers Keras layer
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
from .training_objective_v3 import (
    BOUNDS_CONTINUOUS_REDUCTION_SEMANTICS,
    BOUNDS_TRAINING_OBJECTIVE_VERSION,
)


BOUNDS_TRAINER_VERSION = "posterior_v8_bounds_first_npz_trainer_v3"
BOUNDS_RUN_MANIFEST_SCHEMA = "gisaxs.posterior_v8.bounds_proposal_training_run/v3"
BOUNDS_HISTORY_SCHEMA = "gisaxs.posterior_v8.bounds_proposal_training_history/v3"
BOUNDS_SCIENTIFIC_SCOPE = "solution_only_recipe_grouped_interpolation_pilot_no_parameter_guard_band"


source_hashes = bounds_training_source_hashes


def _resume_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    runtime = manifest["runtime"]
    return {
        "trainer_version": manifest["trainer_version"],
        "model_version": manifest["model_version"],
        "branch_catalog_contract": manifest["branch_catalog_contract"],
        "component_slots_version": manifest["component_slots_version"],
        "objective_version": manifest["objective_version"],
        "dataset_adapter_version": manifest["dataset_adapter_version"],
        "coordinate_contract": manifest["coordinate_contract"],
        "range_construction_semantics": manifest["range_construction_semantics"],
        "scientific_scope": manifest["scientific_scope"],
        "run_config": manifest["run_config"],
        "dataset_fingerprint": manifest["dataset_audit"]["fingerprint_sha256"],
        "source_sha256_aggregate": manifest["source_sha256_aggregate"],
        "tensorflow_version": runtime["tensorflow_version"],
        "num_replicas": runtime["num_replicas"],
        "mixed_precision_policy": runtime["mixed_precision_policy"],
    }


def manifest_payload(config, audit, sources, runtime) -> dict[str, object]:
    config_payload = config.to_dict()
    source_aggregate = sha256(_canonical_json(sources)).hexdigest()
    result = {
        "schema_version": BOUNDS_RUN_MANIFEST_SCHEMA,
        "trainer_version": BOUNDS_TRAINER_VERSION,
        "created_utc": _utc_now(),
        "model_version": BOUNDS_PROPOSAL_MODEL_VERSION,
        "model_name": BOUNDS_PROPOSAL_MODEL_NAME,
        "branch_catalog_contract": {
            "version": MODEL_BRANCH_CATALOG_VERSION,
            "physical_branch_count": MODEL_PHYSICAL_BRANCH_COUNT,
            "wire_pattern_dimension": 32,
            "noncanonical_pattern_logits": (
                "excluded_by_training_objective_and_canonical_inference_ranking"
            ),
        },
        "component_slots_version": MODEL_COMPONENT_SLOTS_VERSION,
        "objective_version": BOUNDS_TRAINING_OBJECTIVE_VERSION,
        "continuous_reduction_semantics": BOUNDS_CONTINUOUS_REDUCTION_SEMANTICS,
        "dataset_adapter_version": BOUNDS_DATASET_ADAPTER_VERSION,
        "coordinate_contract": dict(BOUNDS_MODEL_COORDINATE_CONTRACT),
        "range_construction_semantics": BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS,
        "scientific_scope": BOUNDS_SCIENTIFIC_SCOPE,
        "dataset_phase": SOLUTION_ONLY_PHASE,
        "pilot_limitations": list(PILOT_LIMITATIONS),
        "claim_limits": {
            "production_model": False,
            "paper_holdout": False,
            "ood_generalization": False,
            "no_solution_classification": False,
        },
        "training_split_contract": {
            "consumed": list(TRAINING_SPLITS),
            "excluded": list(EXCLUDED_TRAINING_SPLITS),
            "calibration_test_rows_consumed": 0,
        },
        "run_config": config_payload,
        "run_config_sha256": sha256(_canonical_json(config_payload)).hexdigest(),
        "dataset_audit": audit.to_dict(),
        "source_sha256": sources,
        "source_sha256_aggregate": source_aggregate,
        "runtime": runtime,
        "best_model_file": MODEL_FILE,
        "history_file": HISTORY_FILE,
        "checkpoint_directory": CHECKPOINT_DIRECTORY,
    }
    result["resume_contract_sha256"] = sha256(_canonical_json(_resume_contract(result))).hexdigest()
    return result


def history_template(manifest) -> dict[str, object]:
    return {
        "schema_version": BOUNDS_HISTORY_SCHEMA,
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
        return _read_json(manifest).get("schema_version") == BOUNDS_RUN_MANIFEST_SCHEMA
    except ValueError:
        return False


def prepare_output(path: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if path.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if not path.exists():
        if resume:
            raise FileNotFoundError("cannot resume a missing V3 training run")
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise ValueError("output path must be a directory")
    entries = tuple(path.iterdir())
    if resume:
        if not _owned_run_directory(path):
            raise ValueError("resume requires a bounds-conditioned V3 manifest")
        return
    if not entries:
        return
    if not overwrite:
        raise FileExistsError("output directory is not empty; use resume or overwrite")
    if not _owned_run_directory(path):
        raise ValueError("refusing to overwrite a directory not owned by the V3 trainer")
    allowed = {MANIFEST_FILE, HISTORY_FILE, MODEL_FILE, CHECKPOINT_DIRECTORY}
    if any(item.name not in allowed or item.is_symlink() for item in entries):
        raise ValueError("refusing to overwrite a V3 run with unknown/symlink entries")
    for item in entries:
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def _validate_manifest_static(manifest: Mapping[str, object]) -> None:
    expected = {
        "schema_version": BOUNDS_RUN_MANIFEST_SCHEMA,
        "trainer_version": BOUNDS_TRAINER_VERSION,
        "model_version": BOUNDS_PROPOSAL_MODEL_VERSION,
        "model_name": BOUNDS_PROPOSAL_MODEL_NAME,
        "branch_catalog_contract": {
            "version": MODEL_BRANCH_CATALOG_VERSION,
            "physical_branch_count": MODEL_PHYSICAL_BRANCH_COUNT,
            "wire_pattern_dimension": 32,
            "noncanonical_pattern_logits": (
                "excluded_by_training_objective_and_canonical_inference_ranking"
            ),
        },
        "component_slots_version": MODEL_COMPONENT_SLOTS_VERSION,
        "objective_version": BOUNDS_TRAINING_OBJECTIVE_VERSION,
        "continuous_reduction_semantics": BOUNDS_CONTINUOUS_REDUCTION_SEMANTICS,
        "dataset_adapter_version": BOUNDS_DATASET_ADAPTER_VERSION,
        "coordinate_contract": dict(BOUNDS_MODEL_COORDINATE_CONTRACT),
        "range_construction_semantics": BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS,
        "scientific_scope": BOUNDS_SCIENTIFIC_SCOPE,
        "dataset_phase": SOLUTION_ONLY_PHASE,
        "pilot_limitations": list(PILOT_LIMITATIONS),
        "claim_limits": {
            "production_model": False,
            "paper_holdout": False,
            "ood_generalization": False,
            "no_solution_classification": False,
        },
        "training_split_contract": {
            "consumed": list(TRAINING_SPLITS),
            "excluded": list(EXCLUDED_TRAINING_SPLITS),
            "calibration_test_rows_consumed": 0,
        },
        "best_model_file": MODEL_FILE,
        "history_file": HISTORY_FILE,
        "checkpoint_directory": CHECKPOINT_DIRECTORY,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"bounds-conditioned V3 run has incompatible {name}")
    config = manifest.get("run_config")
    sources = manifest.get("source_sha256")
    dataset = manifest.get("dataset_audit")
    if (
        not isinstance(config, Mapping)
        or manifest.get("run_config_sha256") != sha256(_canonical_json(config)).hexdigest()
    ):
        raise ValueError("V3 run config digest is invalid")
    if (
        not isinstance(sources, Mapping)
        or manifest.get("source_sha256_aggregate") != sha256(_canonical_json(sources)).hexdigest()
    ):
        raise ValueError("V3 source digest aggregate is invalid")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("consumed_splits") != list(TRAINING_SPLITS)
        or dataset.get("excluded_splits") != list(EXCLUDED_TRAINING_SPLITS)
        or dataset.get("calibration_test_rows_consumed") != 0
    ):
        raise ValueError("V3 dataset audit permits holdout leakage")
    try:
        expected_resume = sha256(_canonical_json(_resume_contract(manifest))).hexdigest()
    except (KeyError, TypeError) as exc:
        raise ValueError("V3 resume contract is incomplete") from exc
    if manifest.get("resume_contract_sha256") != expected_resume:
        raise ValueError("V3 resume contract digest is invalid")


def _validate_history(history, manifest, *, require_complete: bool) -> None:
    if (
        set(history) != set(history_template(manifest))
        or history.get("schema_version") != BOUNDS_HISTORY_SCHEMA
    ):
        raise ValueError("V3 training history is incomplete or unsupported")
    if history["resume_contract_sha256"] != manifest["resume_contract_sha256"]:
        raise ValueError("V3 history belongs to a different run contract")
    completed, epochs = history["completed_epochs"], history["epochs"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or not isinstance(epochs, list)
        or completed != len(epochs)
        or completed > manifest["run_config"]["epochs"]
        or history["target_epochs"] != manifest["run_config"]["epochs"]
        or history["status"] not in {"running", "failed", "complete"}
    ):
        raise ValueError("V3 training history epoch state is inconsistent")
    if require_complete and history["status"] != "complete":
        raise ValueError("V3 training history is not complete")
    for index, record in enumerate(epochs, start=1):
        if not isinstance(record, dict) or record.get("epoch") != index:
            raise ValueError("V3 history contains a malformed epoch record")
        for split in ("train", "validation"):
            metrics = record.get(split)
            if (
                not isinstance(metrics, dict)
                or not metrics
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isfinite(float(value))
                    for value in metrics.values()
                )
            ):
                raise ValueError("V3 history contains invalid epoch metrics")
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
            raise ValueError("V3 history has invalid best-checkpoint state")


def validate_resume(manifest_path: Path, expected, history_path: Path):
    actual = _read_json(manifest_path)
    _validate_manifest_static(actual)
    if set(actual) != set(expected):
        raise ValueError("V3 training manifest fields changed")
    for name in set(expected) - {"created_utc", "runtime"}:
        if actual[name] != expected[name]:
            raise ValueError(f"V3 resume contract mismatch in {name}")
    history = _read_json(history_path)
    _validate_history(history, expected, require_complete=False)
    return history


def validate_bounds_training_run(path: str | os.PathLike[str]) -> Mapping[str, object]:
    """Validate one complete V3 run and reject V1/V2 artifacts fail-closed."""

    root = Path(path)
    if root.is_file() and root.name == MODEL_FILE:
        root = root.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("bounds-conditioned V3 run must be a regular directory")
    manifest_path, history_path, model_path = (
        root / MANIFEST_FILE,
        root / HISTORY_FILE,
        root / MODEL_FILE,
    )
    if any(
        value.is_symlink() or not value.is_file()
        for value in (manifest_path, history_path, model_path)
    ):
        raise ValueError("V3 run is missing a regular manifest, history, or model")
    manifest, history = _read_json(manifest_path), _read_json(history_path)
    _validate_manifest_static(manifest)
    _validate_history(history, manifest, require_complete=True)
    digest = history["best_model_sha256"]
    if _file_sha256(model_path) != digest:
        raise ValueError("V3 model checksum does not match training history")
    return MappingProxyType(manifest)


def load_bounds_trained_proposal_model(
    path: str | os.PathLike[str],
) -> tf.keras.Model:
    """Load only a manifest-bound bounds-conditioned local V3 artifact."""

    root = Path(path)
    if root.is_file():
        root = root.parent
    manifest = validate_bounds_training_run(root)
    model = tf.keras.models.load_model(root / MODEL_FILE, safe_mode=True, compile=False)
    _validate_loaded_model_contract(model, manifest)
    model.posterior_v8_model_version = BOUNDS_PROPOSAL_MODEL_VERSION
    model.posterior_v8_branch_catalog_version = MODEL_BRANCH_CATALOG_VERSION
    model.posterior_v8_component_slots_version = MODEL_COMPONENT_SLOTS_VERSION
    model.posterior_v8_input_bounds_coordinate_semantics = BOUNDS_MODEL_COORDINATE_CONTRACT[
        "model_input_bounds"
    ]
    model.posterior_v8_output_coordinate_semantics = BOUNDS_MODEL_COORDINATE_CONTRACT[
        "model_output"
    ]
    return model


def _tensor_shape(value: tf.Tensor) -> tuple[int | None, ...]:
    return tuple(None if item is None else int(item) for item in value.shape)


def _validate_loaded_model_contract(model: tf.keras.Model, manifest: Mapping[str, object]) -> None:
    """Bind serialized tensor dimensions/dtypes to the signed run config."""

    if not isinstance(model, tf.keras.Model):
        raise TypeError("loaded V3 graph must be a Keras Model")
    config = manifest.get("run_config")
    if not isinstance(config, Mapping):
        raise ValueError("V3 manifest has no run_config")
    try:
        max_points = int(config["max_points"])
        mixture_components = int(config["mixture_components"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V3 run_config has invalid graph dimensions") from exc
    if max_points < 1 or mixture_components < 1:
        raise ValueError("V3 run_config graph dimensions must be positive")
    input_by_name = {value.name.split(":", 1)[0]: value for value in model.inputs}
    output_by_name = dict(zip(model.output_names, model.outputs))
    if (
        model.name != BOUNDS_PROPOSAL_MODEL_NAME
        or tuple(sorted(input_by_name)) != tuple(sorted(MODEL_INPUT_KEYS))
        or tuple(sorted(output_by_name)) != tuple(sorted(MODEL_OUTPUT_KEYS))
    ):
        raise ValueError("saved graph identity is not bounds-conditioned local V3")
    dims = MODEL_FIXED_DIMENSIONS
    expected_inputs = {
        "x": ((None, max_points, dims["point_feature_dim"]), "float32"),
        "point_mask": ((None, max_points), "bool"),
        "global_features": ((None, dims["global_feature_dim"]), "float32"),
        "branch_topology": ((None, dims["topology_dim"]), "float32"),
        "branch_d_present": ((None, dims["d_presence_dim"]), "float32"),
        "branch_resolution_present": (
            (None, dims["resolution_presence_dim"]),
            "float32",
        ),
        "bounds_embedding": ((None, dims["bounds_embedding_dim"]), "float32"),
        "active_dimension_mask": (
            (None, dims["local_coordinate_dim"]),
            "float32",
        ),
        "varying_dimension_mask": (
            (None, dims["local_coordinate_dim"]),
            "float32",
        ),
    }
    expected_outputs = {
        "topology_logits": ((None, dims["topology_dim"]), "float32"),
        "branch_pattern_logits": (
            (None, dims["topology_dim"], BRANCH_PATTERN_COUNT),
            "float32",
        ),
        "mixture_logits": ((None, mixture_components), "float32"),
        "mixture_loc": (
            (None, mixture_components, dims["local_coordinate_dim"]),
            "float32",
        ),
        "mixture_logscale": (
            (None, mixture_components, dims["local_coordinate_dim"]),
            "float32",
        ),
    }
    for kind, actual, expected in (
        ("input", input_by_name, expected_inputs),
        ("output", output_by_name, expected_outputs),
    ):
        for name, (shape, dtype) in expected.items():
            tensor = actual[name]
            if _tensor_shape(tensor) != tuple(shape) or tf.as_dtype(tensor.dtype).name != dtype:
                raise ValueError(
                    f"saved V3 graph {kind} {name!r} disagrees with manifest shape/dtype"
                )


__all__ = [
    "BOUNDS_HISTORY_SCHEMA",
    "BOUNDS_RUN_MANIFEST_SCHEMA",
    "BOUNDS_SCIENTIFIC_SCOPE",
    "BOUNDS_TRAINER_VERSION",
    "history_template",
    "load_bounds_trained_proposal_model",
    "manifest_payload",
    "prepare_output",
    "source_hashes",
    "validate_bounds_training_run",
    "validate_resume",
]
