"""Write-free, content-bound launch plan for K1 IID calibration production."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from .k1_dataset_disjointness_v5 import validate_v5_k1_dataset_disjointness_receipt
from .k1_iid_calibration_plan_v5 import (
    V5_K1_IID_CALIBRATION_TOTAL_SAMPLES,
    build_v5_k1_iid_calibration_plan,
    validate_v5_k1_iid_calibration_plan,
)
from .k1_staging_files_v5 import file_sha256, read_regular_bytes
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import fingerprint_v5_k1_training_source
from .package_source_snapshot_v5 import verify_extracted_source_snapshot


V5_K1_IID_CALIBRATION_LAUNCH_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_compatibility_calibration_launch_plan/v2"
)
V5_K1_IID_CALIBRATION_LAUNCH_VERSION = (
    "posterior_v8_v5_2_k1_iid_60_stratum_worker_array_overflow_safe_sigma_"
    "completion_last_v2"
)
K1_IID_CALIBRATION_LAUNCH_PLAN_FILENAME = "k1-iid-calibration-launch-plan-v2.json"
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
_TOP_FIELDS = {
    "schema",
    "version",
    "scientific_role",
    "source",
    "three_way_disjointness_receipt",
    "calibration_population_plan",
    "array",
    "tasks",
    "layout",
    "claim_limits",
}
_TASK_FIELDS = {
    "array_task_id",
    "stratum_ordinal",
    "sample_ordinal_start",
    "sample_count",
    "design_coordinate",
    "compatibility_stratum",
    "output",
    "completion",
}
_LAYOUT_FIELDS = {
    "run_root",
    "plan",
    "logs",
    "data",
    "task_completion",
    "audit",
    "calibration_artifact",
    "four_way_disjointness_receipt",
    "collection_completion",
    "held_submission_receipt",
    "launch_completion",
    "launch_failure",
}


@dataclass(frozen=True, kw_only=True)
class V5K1IIDCalibrationLaunchConfig:
    source_root: Path
    source_archive: Path
    expected_source_archive_sha256: str
    three_way_disjointness_receipt: Path
    expected_three_way_receipt_file_sha256: str
    run_root: Path


def _under_root(path: Path, root: Path, name: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    allowed = root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {allowed}") from exc
    if not relative.parts:
        raise ValueError(f"{name} must not be the allowed root")
    current = allowed
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink")
        if not current.exists():
            break
    resolved = lexical.resolve(strict=must_exist)
    if not resolved.is_relative_to(allowed):
        raise ValueError(f"{name} resolves outside {allowed}")
    return resolved


def _immutable(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a real regular file")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _source_identity(
    config: V5K1IIDCalibrationLaunchConfig,
    allowed_root: Path,
) -> dict[str, object]:
    source_root = _under_root(config.source_root, allowed_root, "source_root", must_exist=True)
    source = fingerprint_v5_k1_training_source(source_root)
    if source["source_snapshot_write_bits_set"] or source["source_files_with_write_bits"]:
        raise ValueError("source snapshot and required files must be read-only")
    archive = _under_root(
        config.source_archive, allowed_root, "source_archive", must_exist=True
    )
    _immutable(archive, "source_archive")
    archive_sha = file_sha256(archive, "source archive")
    if archive_sha != digest(
        config.expected_source_archive_sha256, "expected_source_archive_sha256"
    ):
        raise ValueError("source archive SHA-256 differs from the explicit expectation")
    tree = verify_extracted_source_snapshot(
        archive,
        source_root,
        expected_archive_sha256=archive_sha,
    )
    if tree.get("verified") is not True or tree.get("read_only_tree_verified") is not True:
        raise RuntimeError("source archive/tree binding is incomplete")
    return {
        **source,
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_tree_binding": tree,
    }


def _receipt_identity(
    config: V5K1IIDCalibrationLaunchConfig,
    allowed_root: Path,
) -> dict[str, object]:
    path = _under_root(
        config.three_way_disjointness_receipt,
        allowed_root,
        "three_way_disjointness_receipt",
        must_exist=True,
    )
    _immutable(path, "three_way_disjointness_receipt")
    observed = file_sha256(path, "three-way disjointness receipt")
    if observed != digest(
        config.expected_three_way_receipt_file_sha256,
        "expected_three_way_receipt_file_sha256",
    ):
        raise ValueError("three-way receipt file SHA-256 differs from the expectation")
    receipt = validate_v5_k1_dataset_disjointness_receipt(
        _json(path, "three-way disjointness receipt", 64 * 1024 * 1024)
    )
    if (
        receipt["all_recipe_sets_pairwise_disjoint"] is not True
        or receipt["all_clean_group_sets_pairwise_disjoint"] is not True
    ):
        raise ValueError("three-way parent disjointness is not proven")
    return {
        "path": str(path),
        "file_sha256": observed,
        "receipt_sha256": receipt["receipt_sha256"],
        "phase_c_exclusion_claim_sha256": receipt["phase_c_exclusion_claim_sha256"],
        "population_counts": {
            role: receipt["populations"][role]["clean_parent_count"]
            for role in ("train", "tuning_validation", "phase_c_holdout")
        },
    }


def _tasks(run_root: Path, calibration_plan: Mapping[str, object]) -> list[dict[str, object]]:
    result = []
    for entry in calibration_plan["strata"]:
        ordinal = entry["stratum_ordinal"]
        stem = f"stratum-{ordinal:02d}"
        result.append(
            {
                "array_task_id": ordinal,
                "stratum_ordinal": ordinal,
                "sample_ordinal_start": 0,
                "sample_count": entry["sample_count"],
                "design_coordinate": entry["design_coordinate"],
                "compatibility_stratum": entry["compatibility_stratum"],
                "output": str(run_root / "data" / f"{stem}.json"),
                "completion": str(run_root / "task-completion" / f"{stem}.json"),
            }
        )
    return result


def build_v5_k1_iid_calibration_launch_plan(
    config: V5K1IIDCalibrationLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    if not isinstance(config, V5K1IIDCalibrationLaunchConfig):
        raise TypeError("config must be V5K1IIDCalibrationLaunchConfig")
    run_root = _under_root(config.run_root, allowed_root, "run_root", must_exist=False)
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("refusing to reuse an IID calibration run root")
    source = _source_identity(config, allowed_root)
    receipt = _receipt_identity(config, allowed_root)
    calibration = build_v5_k1_iid_calibration_plan()
    tasks = _tasks(run_root, calibration)
    paths = tuple(value[name] for value in tasks for name in ("output", "completion"))
    if len(paths) != len(set(paths)) or any(Path(value).exists() for value in paths):
        raise FileExistsError("calibration task paths collide or already exist")
    core = {
        "schema": V5_K1_IID_CALIBRATION_LAUNCH_SCHEMA,
        "version": V5_K1_IID_CALIBRATION_LAUNCH_VERSION,
        "scientific_role": "formal_iid_measurement_compatibility_calibration_production",
        "source": source,
        "three_way_disjointness_receipt": receipt,
        "calibration_population_plan": calibration,
        "array": {
            "array_spec": "0-59%60",
            "task_count": 60,
            "expected_sample_count": V5_K1_IID_CALIBRATION_TOTAL_SAMPLES,
        },
        "tasks": tasks,
        "layout": {
            "run_root": str(run_root),
            "plan": str(run_root / K1_IID_CALIBRATION_LAUNCH_PLAN_FILENAME),
            "logs": str(run_root / "logs"),
            "data": str(run_root / "data"),
            "task_completion": str(run_root / "task-completion"),
            "audit": str(run_root / "audit"),
            "calibration_artifact": str(run_root / "artifacts" / "k1-iid-compatibility-v2.json"),
            "four_way_disjointness_receipt": str(
                run_root / "audit" / "train-tune-phase-c-calibration-disjointness-v2.json"
            ),
            "collection_completion": str(run_root / "audit" / "calibration-completion-v2.json"),
            "held_submission_receipt": str(run_root / "audit" / "held-submission-receipt-v2.json"),
            "launch_completion": str(run_root / "audit" / "launch-completion-v2.json"),
            "launch_failure": str(run_root / "audit" / "launch-failure-v2.json"),
        },
        "claim_limits": {
            "training_authorization_granted": False,
            "model_selection_authorization_granted": False,
            "phase_c_acceptance_granted": False,
            "compatibility_threshold_requires_successful_collection": True,
        },
    }
    return {**core, "plan_sha256": sha256(canonical_json(core).encode()).hexdigest()}


def validate_v5_k1_iid_calibration_launch_plan(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("calibration launch plan must be an object")
    value = dict(payload)
    supplied = digest(value.pop("plan_sha256", None), "plan_sha256")
    if supplied != sha256(canonical_json(value).encode()).hexdigest():
        raise ValueError("calibration launch plan self-hash does not reproduce")
    if set(value) != _TOP_FIELDS:
        raise ValueError("calibration launch plan fields are incomplete or unsupported")
    if (
        value["schema"] != V5_K1_IID_CALIBRATION_LAUNCH_SCHEMA
        or value["version"] != V5_K1_IID_CALIBRATION_LAUNCH_VERSION
        or value["array"]
        != {
            "array_spec": "0-59%60",
            "task_count": 60,
            "expected_sample_count": V5_K1_IID_CALIBRATION_TOTAL_SAMPLES,
        }
        or len(value["tasks"]) != 60
    ):
        raise ValueError("calibration launch plan formal population drifted")
    validate_v5_k1_iid_calibration_plan(value["calibration_population_plan"])
    layout = value["layout"]
    if not isinstance(layout, Mapping) or set(layout) != _LAYOUT_FIELDS:
        raise ValueError("calibration launch layout is incomplete or unsupported")
    run_root = Path(layout["run_root"])
    expected_layout = {
        "run_root": str(run_root),
        "plan": str(run_root / K1_IID_CALIBRATION_LAUNCH_PLAN_FILENAME),
        "logs": str(run_root / "logs"),
        "data": str(run_root / "data"),
        "task_completion": str(run_root / "task-completion"),
        "audit": str(run_root / "audit"),
        "calibration_artifact": str(
            run_root / "artifacts" / "k1-iid-compatibility-v2.json"
        ),
        "four_way_disjointness_receipt": str(
            run_root / "audit" / "train-tune-phase-c-calibration-disjointness-v2.json"
        ),
        "collection_completion": str(
            run_root / "audit" / "calibration-completion-v2.json"
        ),
        "held_submission_receipt": str(
            run_root / "audit" / "held-submission-receipt-v2.json"
        ),
        "launch_completion": str(run_root / "audit" / "launch-completion-v2.json"),
        "launch_failure": str(run_root / "audit" / "launch-failure-v2.json"),
    }
    if dict(layout) != expected_layout:
        raise ValueError("calibration launch layout drifted")
    for index, task in enumerate(value["tasks"]):
        expected = value["calibration_population_plan"]["strata"][index]
        if (
            not isinstance(task, Mapping)
            or set(task) != _TASK_FIELDS
            or task["array_task_id"] != index
            or task["stratum_ordinal"] != index
            or task["sample_ordinal_start"] != 0
            or task["sample_count"] != expected["sample_count"]
            or task["design_coordinate"] != expected["design_coordinate"]
            or task["compatibility_stratum"] != expected["compatibility_stratum"]
            or task["output"] != str(run_root / "data" / f"stratum-{index:02d}.json")
            or task["completion"]
            != str(run_root / "task-completion" / f"stratum-{index:02d}.json")
        ):
            raise ValueError("calibration array task mapping drifted")
    if any(value["claim_limits"][name] is not False for name in (
        "training_authorization_granted",
        "model_selection_authorization_granted",
        "phase_c_acceptance_granted",
    )) or value["claim_limits"]["compatibility_threshold_requires_successful_collection"] is not True:
        raise ValueError("calibration launch claim limits drifted")
    return dict(payload)


def replay_v5_k1_iid_calibration_launch_inputs(
    payload: Mapping[str, object],
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    plan = validate_v5_k1_iid_calibration_launch_plan(payload)
    source_root = _under_root(
        Path(plan["source"]["root"]), allowed_root, "source_root", must_exist=True
    )
    source = fingerprint_v5_k1_training_source(source_root)
    archive = _under_root(
        Path(plan["source"]["archive_path"]),
        allowed_root,
        "source_archive",
        must_exist=True,
    )
    receipt_path = _under_root(
        Path(plan["three_way_disjointness_receipt"]["path"]),
        allowed_root,
        "three_way_disjointness_receipt",
        must_exist=True,
    )
    _immutable(archive, "source_archive")
    _immutable(receipt_path, "three_way_disjointness_receipt")
    receipt = validate_v5_k1_dataset_disjointness_receipt(
        _json(receipt_path, "three-way disjointness receipt", 64 * 1024 * 1024)
    )
    observed = {
        "source_bundle_sha256": source["bundle_sha256"],
        "source_archive_sha256": file_sha256(archive, "source archive"),
        "three_way_receipt_file_sha256": file_sha256(
            receipt_path, "three-way disjointness receipt"
        ),
        "three_way_receipt_sha256": receipt["receipt_sha256"],
        "phase_c_exclusion_claim_sha256": receipt["phase_c_exclusion_claim_sha256"],
        "calibration_population_plan_sha256": plan["calibration_population_plan"][
            "plan_sha256"
        ],
    }
    expected = {
        "source_bundle_sha256": plan["source"]["bundle_sha256"],
        "source_archive_sha256": plan["source"]["archive_sha256"],
        "three_way_receipt_file_sha256": plan["three_way_disjointness_receipt"][
            "file_sha256"
        ],
        "three_way_receipt_sha256": plan["three_way_disjointness_receipt"][
            "receipt_sha256"
        ],
        "phase_c_exclusion_claim_sha256": plan["three_way_disjointness_receipt"][
            "phase_c_exclusion_claim_sha256"
        ],
        "calibration_population_plan_sha256": plan["calibration_population_plan"][
            "plan_sha256"
        ],
    }
    if observed != expected:
        raise RuntimeError("calibration immutable input identity changed")
    return observed


def write_v5_k1_iid_calibration_launch_plan(
    path: str | os.PathLike[str],
    payload: Mapping[str, object],
) -> Path:
    plan = validate_v5_k1_iid_calibration_launch_plan(payload)
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a calibration launch plan")
    if not target.parent.is_dir():
        raise FileNotFoundError("calibration launch-plan parent does not exist")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    _immutable(target, "calibration launch plan")
    return target


__all__ = [
    "K1_IID_CALIBRATION_LAUNCH_PLAN_FILENAME",
    "V5_K1_IID_CALIBRATION_LAUNCH_SCHEMA",
    "V5_K1_IID_CALIBRATION_LAUNCH_VERSION",
    "V5K1IIDCalibrationLaunchConfig",
    "build_v5_k1_iid_calibration_launch_plan",
    "replay_v5_k1_iid_calibration_launch_inputs",
    "validate_v5_k1_iid_calibration_launch_plan",
    "write_v5_k1_iid_calibration_launch_plan",
]
