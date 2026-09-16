"""Write-free production plan for the formal K1 Phase-C holdout identities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping

from .k1_balanced_dataset_launch_plan_v5 import MAXWELL_DUST_ROOT
from .k1_dataset_disjointness_v5 import (
    validate_v5_k1_train_tuning_disjointness_receipt,
)
from .k1_phase_c_plan_v5 import v5_k1_phase_c_plan_from_payload
from .k1_staging_files_v5 import file_sha256, read_regular_bytes
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import fingerprint_v5_k1_training_source
from .package_source_snapshot_v5 import verify_extracted_source_snapshot


V5_K1_PHASE_C_HOLDOUT_LAUNCH_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_holdout_launch_plan/v1"
)
V5_K1_PHASE_C_HOLDOUT_LAUNCH_VERSION = (
    "posterior_v8_v5_2_formal_all12_holdout_identity_array_v1"
)
K1_PHASE_C_HOLDOUT_LAUNCH_PLAN_FILENAME = "k1-phase-c-holdout-launch-plan-v1.json"
K1_PHASE_C_HOLDOUT_FORMAL_RECIPES_PER_SHARD = 288


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCHoldoutLaunchConfig:
    source_root: Path
    source_archive: Path
    expected_source_archive_sha256: str
    phase_c_plan: Path
    expected_phase_c_plan_file_sha256: str
    train_tuning_receipt: Path
    expected_train_tuning_receipt_file_sha256: str
    run_root: Path
    recipes_per_shard: int = K1_PHASE_C_HOLDOUT_FORMAL_RECIPES_PER_SHARD


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


def _immutable_file(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a real regular file")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _read_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _input_bindings(config: V5K1PhaseCHoldoutLaunchConfig, allowed_root: Path):
    source_root = _under_root(config.source_root, allowed_root, "source_root", must_exist=True)
    source = fingerprint_v5_k1_training_source(source_root)
    if source["source_snapshot_write_bits_set"] or source["source_files_with_write_bits"]:
        raise ValueError("source snapshot and required files must be read-only")
    archive = _under_root(
        config.source_archive, allowed_root, "source_archive", must_exist=True
    )
    _immutable_file(archive, "source_archive")
    archive_sha = file_sha256(archive, "Phase-C source archive")
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

    phase_c_path = _under_root(
        config.phase_c_plan, allowed_root, "phase_c_plan", must_exist=True
    )
    _immutable_file(phase_c_path, "phase_c_plan")
    phase_c_file_sha = file_sha256(phase_c_path, "Phase-C plan")
    if phase_c_file_sha != digest(
        config.expected_phase_c_plan_file_sha256,
        "expected_phase_c_plan_file_sha256",
    ):
        raise ValueError("Phase-C plan file SHA-256 differs from the expectation")
    phase_c = v5_k1_phase_c_plan_from_payload(
        _read_json(phase_c_path, "Phase-C plan", 1024 * 1024)
    )
    if not phase_c.formal or phase_c.total_parent_count != 13824:
        raise ValueError("holdout production requires the exact formal Phase-C plan")

    train_tuning_path = _under_root(
        config.train_tuning_receipt,
        allowed_root,
        "train_tuning_receipt",
        must_exist=True,
    )
    _immutable_file(train_tuning_path, "train_tuning_receipt")
    train_tuning_file_sha = file_sha256(train_tuning_path, "train/tuning receipt")
    if train_tuning_file_sha != digest(
        config.expected_train_tuning_receipt_file_sha256,
        "expected_train_tuning_receipt_file_sha256",
    ):
        raise ValueError("train/tuning receipt file SHA-256 differs from the expectation")
    train_tuning = validate_v5_k1_train_tuning_disjointness_receipt(
        _read_json(train_tuning_path, "train/tuning receipt", 16 * 1024 * 1024)
    )
    return {
        "source_root": source_root,
        "archive": archive,
        "archive_sha": archive_sha,
        "source": source,
        "tree": tree,
        "phase_c_path": phase_c_path,
        "phase_c_file_sha": phase_c_file_sha,
        "phase_c": phase_c,
        "train_tuning_path": train_tuning_path,
        "train_tuning_file_sha": train_tuning_file_sha,
        "train_tuning": train_tuning,
    }


def _task_rows(phase_c, *, run_root: Path, recipes_per_shard: int):
    tasks = []
    for block in phase_c.sobol_blocks:
        for shard_index in range(math.ceil(block.parent_count / recipes_per_shard)):
            offset = shard_index * recipes_per_shard
            count = min(recipes_per_shard, block.parent_count - offset)
            stem = f"branch-{block.branch_ordinal:02d}-shard-{shard_index:04d}"
            tasks.append(
                {
                    "array_task_id": len(tasks),
                    "branch_id": block.branch_id,
                    "branch_ordinal": block.branch_ordinal,
                    "phase_c_sobol_block_sha256": block.block_sha256,
                    "shard_index": shard_index,
                    "split_offset": offset,
                    "recipe_count": count,
                    "output": str(run_root / "data" / f"{stem}.json"),
                    "completion": str(run_root / "completion" / f"{stem}.json"),
                }
            )
    return tasks


def build_v5_k1_phase_c_holdout_launch_plan(
    config: V5K1PhaseCHoldoutLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    if not isinstance(config, V5K1PhaseCHoldoutLaunchConfig):
        raise TypeError("config must be V5K1PhaseCHoldoutLaunchConfig")
    if isinstance(config.recipes_per_shard, bool) or not isinstance(
        config.recipes_per_shard, int
    ) or config.recipes_per_shard < 1:
        raise ValueError("recipes_per_shard must be a positive integer")
    bound = _input_bindings(config, allowed_root)
    run_root = _under_root(config.run_root, allowed_root, "run_root", must_exist=False)
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("refusing to reuse a Phase-C holdout run root")
    tasks = _task_rows(
        bound["phase_c"],
        run_root=run_root,
        recipes_per_shard=config.recipes_per_shard,
    )
    expected_count = sum(value["recipe_count"] for value in tasks)
    if expected_count != bound["phase_c"].total_parent_count:
        raise RuntimeError("Phase-C task inventory does not cover its formal population")
    paths = tuple(str(value[name]) for value in tasks for name in ("output", "completion"))
    if len(paths) != len(set(paths)) or any(Path(value).exists() for value in paths):
        raise FileExistsError("Phase-C task outputs collide or already exist")
    core = {
        "schema": V5_K1_PHASE_C_HOLDOUT_LAUNCH_SCHEMA,
        "version": V5_K1_PHASE_C_HOLDOUT_LAUNCH_VERSION,
        "scientific_role": "formal_phase_c_holdout_identity_production_not_model_acceptance",
        "source": {
            **bound["source"],
            "archive_path": str(bound["archive"]),
            "archive_sha256": bound["archive_sha"],
            "archive_tree_binding": bound["tree"],
        },
        "phase_c_plan": {
            "path": str(bound["phase_c_path"]),
            "file_sha256": bound["phase_c_file_sha"],
            "plan_sha256": bound["phase_c"].sha256,
            "contract_sha256": bound["phase_c"].contract_sha256,
        },
        "train_tuning_receipt": {
            "path": str(bound["train_tuning_path"]),
            "file_sha256": bound["train_tuning_file_sha"],
            "receipt_sha256": bound["train_tuning"]["receipt_sha256"],
            "train_tuning_claim_sha256": bound["train_tuning"][
                "train_tuning_claim_sha256"
            ],
        },
        "layout": {
            "run_root": str(run_root),
            "logs": str(run_root / "logs"),
            "data": str(run_root / "data"),
            "completion": str(run_root / "completion"),
            "audit": str(run_root / "audit"),
            "plan": str(run_root / K1_PHASE_C_HOLDOUT_LAUNCH_PLAN_FILENAME),
            "three_way_disjointness_receipt": str(
                run_root / "audit" / "train-tuning-phase-c-disjointness-v1.json"
            ),
            "holdout_completion": str(
                run_root / "audit" / "phase-c-holdout-completion-v1.json"
            ),
            "held_submission_receipt": str(
                run_root / "audit" / "held-submission-receipt-v1.json"
            ),
            "launch_completion": str(run_root / "audit" / "launch-completion-v1.json"),
            "launch_failure": str(run_root / "audit" / "launch-failure-v1.json"),
        },
        "configuration": {
            "formal_phase_c_holdout_production": (
                config.recipes_per_shard == K1_PHASE_C_HOLDOUT_FORMAL_RECIPES_PER_SHARD
            ),
            "recipes_per_shard": config.recipes_per_shard,
            "parent_count": bound["phase_c"].total_parent_count,
            "parents_per_branch": bound["phase_c"].parents_per_branch,
            "branch_count": len(bound["phase_c"].sobol_blocks),
        },
        "array": {
            "task_count": len(tasks),
            "array_spec": f"0-{len(tasks) - 1}",
            "expected_clean_parent_count": expected_count,
        },
        "tasks": tasks,
        "execution_contract": {
            "materialization": "Slurm_CPU_worker_only",
            "submit_all_jobs_held_before_release": True,
            "completion_written_after_artifact_seal": True,
            "artifact_and_completion_mode_octal": "0400",
            "artifact_and_completion_nlink": 1,
            "existing_output_reuse_allowed": False,
            "three_way_actual_identity_receipt_required": True,
            "training_authorization_granted": False,
            "phase_c_acceptance_granted": False,
        },
    }
    return {**core, "plan_sha256": sha256(canonical_json(core).encode()).hexdigest()}


def validate_v5_k1_phase_c_holdout_launch_plan(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Phase-C holdout launch plan must be an object")
    value = dict(payload)
    supplied = digest(value.pop("plan_sha256", None), "plan_sha256")
    if supplied != sha256(canonical_json(value).encode()).hexdigest():
        raise ValueError("Phase-C holdout launch plan self-hash does not reproduce")
    if value.get("schema") != V5_K1_PHASE_C_HOLDOUT_LAUNCH_SCHEMA or value.get(
        "version"
    ) != V5_K1_PHASE_C_HOLDOUT_LAUNCH_VERSION:
        raise ValueError("Phase-C holdout launch plan schema/version drifted")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != value["array"]["task_count"]:
        raise ValueError("Phase-C holdout task inventory is incomplete")
    if [row.get("array_task_id") for row in tasks] != list(range(len(tasks))):
        raise ValueError("Phase-C holdout array mapping is not contiguous")
    if sum(row.get("recipe_count", 0) for row in tasks) != value["array"][
        "expected_clean_parent_count"
    ]:
        raise ValueError("Phase-C holdout task counts do not reproduce")
    return {**value, "plan_sha256": supplied}


def replay_v5_k1_phase_c_holdout_launch_inputs(
    plan: Mapping[str, object], *, allowed_root: Path = MAXWELL_DUST_ROOT
) -> dict[str, object]:
    checked = validate_v5_k1_phase_c_holdout_launch_plan(plan)
    source_expected = checked["source"]
    source_root = _under_root(
        Path(source_expected["root"]), allowed_root, "source_root", must_exist=True
    )
    source = fingerprint_v5_k1_training_source(source_root)
    expected_source = {
        name: source_expected[name]
        for name in (
            "root",
            "bundle_sha256",
            "required_file_sha256",
            "source_snapshot_write_bits_set",
            "source_files_with_write_bits",
        )
    }
    if source != expected_source:
        raise RuntimeError("Phase-C source snapshot changed after launch planning")
    archive = _under_root(
        Path(source_expected["archive_path"]),
        allowed_root,
        "source_archive",
        must_exist=True,
    )
    _immutable_file(archive, "source_archive")
    if file_sha256(archive, "Phase-C source archive") != source_expected[
        "archive_sha256"
    ]:
        raise RuntimeError("Phase-C source archive changed after launch planning")
    tree_expected = source_expected["archive_tree_binding"]
    tree = verify_extracted_source_snapshot(
        archive,
        source_root,
        expected_archive_sha256=source_expected["archive_sha256"],
        expected_manifest_sha256=tree_expected["manifest_sha256"],
        expected_source_tree_sha256=tree_expected["source_tree_sha256"],
    )
    if tree != tree_expected:
        raise RuntimeError("Phase-C source archive/tree binding changed")

    phase_c_expected = checked["phase_c_plan"]
    phase_c_path = _under_root(
        Path(phase_c_expected["path"]),
        allowed_root,
        "phase_c_plan",
        must_exist=True,
    )
    _immutable_file(phase_c_path, "phase_c_plan")
    phase_c_file_sha = file_sha256(phase_c_path, "Phase-C plan")
    phase_c = v5_k1_phase_c_plan_from_payload(
        _read_json(phase_c_path, "Phase-C plan", 1024 * 1024)
    )
    if {
        "path": str(phase_c_path),
        "file_sha256": phase_c_file_sha,
        "plan_sha256": phase_c.sha256,
        "contract_sha256": phase_c.contract_sha256,
    } != phase_c_expected:
        raise RuntimeError("Phase-C authoring plan changed after launch planning")

    train_tuning_expected = checked["train_tuning_receipt"]
    train_tuning_path = _under_root(
        Path(train_tuning_expected["path"]),
        allowed_root,
        "train_tuning_receipt",
        must_exist=True,
    )
    _immutable_file(train_tuning_path, "train_tuning_receipt")
    train_tuning_file_sha = file_sha256(
        train_tuning_path, "train/tuning receipt"
    )
    train_tuning = validate_v5_k1_train_tuning_disjointness_receipt(
        _read_json(train_tuning_path, "train/tuning receipt", 16 * 1024 * 1024)
    )
    if {
        "path": str(train_tuning_path),
        "file_sha256": train_tuning_file_sha,
        "receipt_sha256": train_tuning["receipt_sha256"],
        "train_tuning_claim_sha256": train_tuning["train_tuning_claim_sha256"],
    } != train_tuning_expected:
        raise RuntimeError("train/tuning receipt changed after launch planning")
    expected_tasks = _task_rows(
        phase_c,
        run_root=Path(checked["layout"]["run_root"]),
        recipes_per_shard=checked["configuration"]["recipes_per_shard"],
    )
    if checked["tasks"] != expected_tasks:
        raise RuntimeError("Phase-C holdout task inventory changed after planning")
    return {
        "source_bundle_sha256": source["bundle_sha256"],
        "source_archive_sha256": source_expected["archive_sha256"],
        "source_manifest_sha256": tree["manifest_sha256"],
        "source_tree_sha256": tree["source_tree_sha256"],
        "phase_c_plan_file_sha256": phase_c_file_sha,
        "phase_c_plan_sha256": phase_c.sha256,
        "train_tuning_receipt_file_sha256": train_tuning_file_sha,
        "train_tuning_receipt_sha256": train_tuning["receipt_sha256"],
        "train_tuning_claim_sha256": train_tuning["train_tuning_claim_sha256"],
    }


def write_v5_k1_phase_c_holdout_launch_plan(
    path: str | os.PathLike[str], payload: Mapping[str, object]
) -> Path:
    value = validate_v5_k1_phase_c_holdout_launch_plan(payload)
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a Phase-C holdout launch plan")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    _immutable_file(target, "Phase-C holdout launch plan")
    return target


__all__ = [
    "K1_PHASE_C_HOLDOUT_FORMAL_RECIPES_PER_SHARD",
    "K1_PHASE_C_HOLDOUT_LAUNCH_PLAN_FILENAME",
    "V5_K1_PHASE_C_HOLDOUT_LAUNCH_SCHEMA",
    "V5_K1_PHASE_C_HOLDOUT_LAUNCH_VERSION",
    "V5K1PhaseCHoldoutLaunchConfig",
    "build_v5_k1_phase_c_holdout_launch_plan",
    "replay_v5_k1_phase_c_holdout_launch_inputs",
    "validate_v5_k1_phase_c_holdout_launch_plan",
    "write_v5_k1_phase_c_holdout_launch_plan",
]
