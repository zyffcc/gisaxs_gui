"""Write-free production plan for balanced all-K1 grouped datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Integral
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np

from .k1_balanced_dataset_plan_v5 import (
    K1_BALANCED_FORMAL_RECIPES_PER_SHARD,
    K1_BALANCED_FORMAL_VIEW_INDICES,
    V5K1BalancedDatasetPlan,
    is_frozen_v5_k1_balanced_dataset_plan,
    v5_k1_balanced_formal_configuration_payload,
    v5_k1_balanced_dataset_plan_from_payload,
)
from .k1_staging_files_v5 import file_sha256, read_regular_bytes
from .k1_training_chain_contract_v5 import canonical_json, digest
from .k1_training_chain_plan_v5 import fingerprint_v5_k1_training_source
from .package_source_snapshot_v5 import verify_extracted_source_snapshot


V5_K1_BALANCED_DATASET_LAUNCH_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_dataset_launch_plan/v2"
)
V5_K1_BALANCED_DATASET_LAUNCH_VERSION = (
    "posterior_v8_v5_2_all12_branch_pure_grouped_dataset_array_v2"
)
K1_BALANCED_DATASET_LAUNCH_PLAN_FILENAME = "k1-balanced-dataset-launch-plan-v2.json"
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")


@dataclass(frozen=True, kw_only=True)
class V5K1BalancedDatasetLaunchConfig:
    source_root: Path
    source_archive: Path
    expected_source_archive_sha256: str
    balanced_plan: Path
    expected_balanced_plan_file_sha256: str
    run_root: Path
    recipes_per_shard: int
    view_indices: tuple[int, ...]


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _views(values: object) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("view_indices must be a tuple")
    result = tuple(
        _nonnegative_integer(value, f"view_indices[{index}]")
        for index, value in enumerate(values)
    )
    if not result or len(result) != len(set(result)):
        raise ValueError("view_indices must be non-empty and duplicate-free")
    return result


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


def _read_balanced_plan(path: Path) -> tuple[V5K1BalancedDatasetPlan, str]:
    encoded = read_regular_bytes(
        path,
        "K1 balanced dataset authoring plan",
        maximum_bytes=1024 * 1024,
    )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("balanced dataset authoring plan is not valid JSON") from exc
    plan = v5_k1_balanced_dataset_plan_from_payload(payload)
    return plan, sha256(encoded).hexdigest()


def _readonly_single_link(path: Path, name: str) -> None:
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _source_binding(config: V5K1BalancedDatasetLaunchConfig, allowed_root: Path):
    source_root = _under_root(
        config.source_root, allowed_root, "source_root", must_exist=True
    )
    source = fingerprint_v5_k1_training_source(source_root)
    if source["source_snapshot_write_bits_set"] or source["source_files_with_write_bits"]:
        raise ValueError("source_root and required files must be read-only")
    archive = _under_root(
        config.source_archive, allowed_root, "source_archive", must_exist=True
    )
    if not archive.is_file():
        raise ValueError("source_archive must be a regular file")
    _readonly_single_link(archive, "source_archive")
    expected = digest(
        config.expected_source_archive_sha256, "expected_source_archive_sha256"
    )
    actual = file_sha256(archive, "K1 balanced dataset source archive")
    if actual != expected:
        raise ValueError("source archive SHA-256 differs from the explicit expectation")
    tree = verify_extracted_source_snapshot(
        archive,
        source_root,
        expected_archive_sha256=expected,
    )
    if tree.get("verified") is not True or tree.get("read_only_tree_verified") is not True:
        raise RuntimeError("source archive/tree binding is incomplete")
    return source_root, archive, source, actual, tree


def _task_rows(
    plan: V5K1BalancedDatasetPlan,
    *,
    run_root: Path,
    recipes_per_shard: int,
) -> list[dict[str, object]]:
    tasks = []
    for block in plan.blocks:
        for shard_index in range(math.ceil(block.parent_count / recipes_per_shard)):
            start = shard_index * recipes_per_shard
            count = min(recipes_per_shard, block.parent_count - start)
            stem = f"branch-{block.branch_ordinal:02d}-shard-{shard_index:06d}"
            tasks.append(
                {
                    "array_task_id": len(tasks),
                    "role": block.role,
                    "split_id": block.split_id,
                    "branch_id": block.branch_id,
                    "branch_ordinal": block.branch_ordinal,
                    "balanced_sobol_block_sha256": block.block_sha256,
                    "shard_index": shard_index,
                    "split_offset": start,
                    "recipe_count": count,
                    "output": str(run_root / "data" / block.role / f"{stem}.gvd5"),
                    "completion": str(
                        run_root / "completion" / block.role / f"{stem}.json"
                    ),
                }
            )
    return tasks


def build_v5_k1_balanced_dataset_launch_plan(
    config: V5K1BalancedDatasetLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Bind a frozen source and explicit all12 authoring inputs without writing."""

    if not isinstance(config, V5K1BalancedDatasetLaunchConfig):
        raise TypeError("config must be V5K1BalancedDatasetLaunchConfig")
    shard_size = _positive_integer(config.recipes_per_shard, "recipes_per_shard")
    views = _views(config.view_indices)
    source_root, archive, source, archive_sha, tree = _source_binding(
        config, allowed_root
    )
    balanced_path = _under_root(
        config.balanced_plan, allowed_root, "balanced_plan", must_exist=True
    )
    if not balanced_path.is_file():
        raise ValueError("balanced_plan must be a regular file")
    _readonly_single_link(balanced_path, "balanced_plan")
    balanced, balanced_file_sha = _read_balanced_plan(balanced_path)
    if balanced_file_sha != digest(
        config.expected_balanced_plan_file_sha256,
        "expected_balanced_plan_file_sha256",
    ):
        raise ValueError("balanced plan file SHA-256 differs from the explicit expectation")
    run_root = _under_root(config.run_root, allowed_root, "run_root", must_exist=False)
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("refusing to reuse a balanced K1 dataset run root")
    if run_root == source_root or run_root.is_relative_to(source_root):
        raise ValueError("run_root cannot be inside the immutable source snapshot")
    tasks = _task_rows(balanced, run_root=run_root, recipes_per_shard=shard_size)
    paths = tuple(
        str(value[name]) for value in tasks for name in ("output", "completion")
    )
    if len(paths) != len(set(paths)) or any(Path(value).exists() for value in paths):
        raise FileExistsError("planned artifact/completion paths collide or already exist")
    expected_counts = {
        role: sum(int(value["recipe_count"]) for value in tasks if value["role"] == role)
        for role in ("train", "tuning_validation")
    }
    if expected_counts != {
        "train": balanced.train_parents_per_branch * 12,
        "tuning_validation": balanced.tuning_parents_per_branch * 12,
    }:
        raise RuntimeError("balanced dataset task coverage does not reproduce")
    frozen_configuration = v5_k1_balanced_formal_configuration_payload()
    formal_e1_production = bool(
        is_frozen_v5_k1_balanced_dataset_plan(balanced)
        and shard_size == K1_BALANCED_FORMAL_RECIPES_PER_SHARD
        and views == K1_BALANCED_FORMAL_VIEW_INDICES
    )
    core = {
        "schema": V5_K1_BALANCED_DATASET_LAUNCH_SCHEMA,
        "version": V5_K1_BALANCED_DATASET_LAUNCH_VERSION,
        "scientific_role": "balanced_all12_train_tuning_dataset_production_not_model_acceptance",
        "source": {
            **source,
            "archive_path": str(archive),
            "archive_sha256": archive_sha,
            "archive_tree_binding": tree,
        },
        "balanced_dataset_plan": {
            "path": str(balanced_path),
            "file_sha256": balanced_file_sha,
            "plan_sha256": balanced.sha256,
            "phase_c_formal_plan_sha256": balanced.phase_c_plan_sha256,
        },
        "layout": {
            "run_root": str(run_root),
            "logs": str(run_root / "logs"),
            "data": str(run_root / "data"),
            "completion": str(run_root / "completion"),
            "audit": str(run_root / "audit"),
            "plan": str(run_root / K1_BALANCED_DATASET_LAUNCH_PLAN_FILENAME),
            "train_tuning_disjointness_receipt": str(
                run_root / "audit" / "train-tuning-disjointness-v2.json"
            ),
            "dataset_completion": str(
                run_root / "audit" / "balanced-dataset-completion-v2.json"
            ),
            "held_submission_receipt": str(
                run_root / "audit" / "held-submission-receipt-v2.json"
            ),
            "launch_completion": str(
                run_root / "audit" / "launch-completion-v2.json"
            ),
            "launch_failure": str(run_root / "audit" / "launch-failure-v2.json"),
        },
        "configuration": {
            "formal_e1_dataset_production": formal_e1_production,
            "formal_configuration_sha256": frozen_configuration[
                "configuration_sha256"
            ],
            "recipes_per_shard": shard_size,
            "view_indices": list(views),
            "train_master_scramble_seed": balanced.train_master_scramble_seed,
            "tuning_master_scramble_seed": balanced.tuning_master_scramble_seed,
            "parents_per_branch": {
                "train": balanced.train_parents_per_branch,
                "tuning_validation": balanced.tuning_parents_per_branch,
            },
        },
        "array": {
            "task_count": len(tasks),
            "array_spec": f"0-{len(tasks) - 1}",
            "expected_clean_parent_counts": expected_counts,
        },
        "tasks": tasks,
        "execution_contract": {
            "dataset_generation": "Slurm_CPU_worker_only",
            "submit_all_jobs_held_before_release": True,
            "completion_written_after_artifact_seal": True,
            "artifact_and_completion_mode_octal": "0400",
            "artifact_and_completion_nlink": 1,
            "existing_output_reuse_allowed": False,
            "actual_recipe_hash_disjointness_receipt_required": True,
            "training_authorization_granted": False,
            "phase_c_acceptance_granted": False,
            "submit_requires_formal_e1_dataset_production": True,
        },
    }
    return {**core, "plan_sha256": sha256(canonical_json(core).encode()).hexdigest()}


def validate_v5_k1_balanced_dataset_launch_plan(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate self-hash and the deterministic task inventory without I/O."""

    if not isinstance(payload, Mapping):
        raise TypeError("balanced dataset launch plan must be an object")
    value = dict(payload)
    supplied = digest(value.pop("plan_sha256", None), "plan_sha256")
    if supplied != sha256(canonical_json(value).encode()).hexdigest():
        raise ValueError("balanced dataset launch plan SHA-256 does not reproduce")
    required = {
        "schema",
        "version",
        "scientific_role",
        "source",
        "balanced_dataset_plan",
        "layout",
        "configuration",
        "array",
        "tasks",
        "execution_contract",
    }
    if set(value) != required:
        raise ValueError("balanced dataset launch plan fields are unsupported")
    if (
        value["schema"] != V5_K1_BALANCED_DATASET_LAUNCH_SCHEMA
        or value["version"] != V5_K1_BALANCED_DATASET_LAUNCH_VERSION
    ):
        raise ValueError("balanced dataset launch plan schema/version drifted")
    tasks = value["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("balanced dataset launch plan has no tasks")
    if [item.get("array_task_id") for item in tasks] != list(range(len(tasks))):
        raise ValueError("balanced dataset array task IDs are not contiguous")
    array = value["array"]
    if not isinstance(array, Mapping) or (
        array.get("task_count") != len(tasks)
        or array.get("array_spec") != f"0-{len(tasks) - 1}"
    ):
        raise ValueError("balanced dataset array metadata disagrees with tasks")
    outputs = tuple(
        item.get(name) for item in tasks for name in ("output", "completion")
    )
    if not all(isinstance(path, str) and Path(path).is_absolute() for path in outputs):
        raise ValueError("balanced dataset task paths must be absolute")
    if len(outputs) != len(set(outputs)):
        raise ValueError("balanced dataset task paths are not unique")
    return {**value, "plan_sha256": supplied}


def replay_v5_k1_balanced_dataset_launch_inputs(
    payload: Mapping[str, object],
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Rehash every immutable launch input and reject any post-plan drift."""

    plan = validate_v5_k1_balanced_dataset_launch_plan(payload)
    source_expected = plan["source"]
    source_root = _under_root(
        Path(source_expected["root"]), allowed_root, "source_root", must_exist=True
    )
    # A published dataset owns its historical inventory. The current training
    # inventory may legitimately grow; it is only authoritative for new plans.
    # The pinned archive/tree check below still verifies the entire old snapshot.
    inventory = source_expected["required_file_sha256"]
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("balanced dataset historical source inventory is empty")
    write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    writable_files = []
    for relative, expected_sha in inventory.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("historical source inventory requires relative paths")
        if Path(relative).as_posix() != relative or ".." in Path(relative).parts:
            raise ValueError("historical source inventory path is not canonical")
        path = _under_root(
            source_root / relative, source_root, "historical source", must_exist=True
        )
        if file_sha256(path, "historical source") != expected_sha:
            raise RuntimeError("balanced dataset source snapshot changed after planning")
        if stat.S_IMODE(path.stat().st_mode) & write_mask:
            writable_files.append(relative)
    if (
        bool(stat.S_IMODE(source_root.stat().st_mode) & write_mask)
        != source_expected["source_snapshot_write_bits_set"]
        or sorted(writable_files) != sorted(source_expected["source_files_with_write_bits"])
    ):
        raise RuntimeError("balanced dataset source permissions changed after planning")
    archive = _under_root(
        Path(source_expected["archive_path"]),
        allowed_root,
        "source_archive",
        must_exist=True,
    )
    _readonly_single_link(archive, "source_archive")
    if file_sha256(archive, "K1 balanced dataset source archive") != source_expected[
        "archive_sha256"
    ]:
        raise RuntimeError("balanced dataset source archive changed after planning")
    tree_expected = source_expected["archive_tree_binding"]
    tree = verify_extracted_source_snapshot(
        archive,
        source_root,
        expected_archive_sha256=source_expected["archive_sha256"],
        expected_manifest_sha256=tree_expected["manifest_sha256"],
        expected_source_tree_sha256=tree_expected["source_tree_sha256"],
    )
    if tree != tree_expected:
        raise RuntimeError("balanced dataset source archive/tree binding changed")
    balanced_expected = plan["balanced_dataset_plan"]
    balanced_path = _under_root(
        Path(balanced_expected["path"]),
        allowed_root,
        "balanced_plan",
        must_exist=True,
    )
    _readonly_single_link(balanced_path, "balanced_plan")
    balanced, balanced_file_sha = _read_balanced_plan(balanced_path)
    balanced_identity = {
        "path": str(balanced_path),
        "file_sha256": balanced_file_sha,
        "plan_sha256": balanced.sha256,
        "phase_c_formal_plan_sha256": balanced.phase_c_plan_sha256,
    }
    if balanced_identity != balanced_expected:
        raise RuntimeError("balanced dataset authoring plan changed after planning")
    expected_tasks = _task_rows(
        balanced,
        run_root=Path(plan["layout"]["run_root"]),
        recipes_per_shard=plan["configuration"]["recipes_per_shard"],
    )
    if plan["tasks"] != expected_tasks:
        raise RuntimeError("balanced dataset task inventory changed after planning")
    return {
        "source_bundle_sha256": source_expected["bundle_sha256"],
        "source_archive_sha256": source_expected["archive_sha256"],
        "source_manifest_sha256": tree["manifest_sha256"],
        "source_tree_sha256": tree["source_tree_sha256"],
        "balanced_plan_file_sha256": balanced_file_sha,
        "balanced_plan_sha256": balanced.sha256,
        "phase_c_formal_plan_sha256": balanced.phase_c_plan_sha256,
    }


def write_v5_k1_balanced_dataset_launch_plan(
    path: str | Path,
    payload: Mapping[str, object],
) -> Path:
    """Exclusively publish a validated launch plan as 0400/nlink1."""

    plan = validate_v5_k1_balanced_dataset_launch_plan(payload)
    target = Path(path)
    if not target.is_absolute() or str(target) != plan["layout"]["plan"]:
        raise ValueError("launch plan path must equal its frozen layout path")
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a balanced dataset launch plan")
    if not target.parent.is_dir():
        raise FileNotFoundError("balanced dataset launch plan parent does not exist")
    encoded = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    _readonly_single_link(target, "balanced dataset launch plan")
    return target


__all__ = [
    "K1_BALANCED_DATASET_LAUNCH_PLAN_FILENAME",
    "MAXWELL_DUST_ROOT",
    "V5_K1_BALANCED_DATASET_LAUNCH_SCHEMA",
    "V5_K1_BALANCED_DATASET_LAUNCH_VERSION",
    "V5K1BalancedDatasetLaunchConfig",
    "build_v5_k1_balanced_dataset_launch_plan",
    "replay_v5_k1_balanced_dataset_launch_inputs",
    "validate_v5_k1_balanced_dataset_launch_plan",
    "write_v5_k1_balanced_dataset_launch_plan",
]
