"""Slurm worker runtime for one balanced all-K1 grouped-dataset shard."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .build_k1_balanced_grouped_shard_v5 import (
    build_v5_k1_balanced_grouped_shard,
    plan_v5_k1_balanced_grouped_shard,
)
from .grouped_artifact_v5 import canonical_json
from .k1_balanced_dataset_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_balanced_dataset_launch_inputs,
    validate_v5_k1_balanced_dataset_launch_plan,
)
from .k1_balanced_dataset_plan_v5 import v5_k1_balanced_dataset_plan_from_payload
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest


V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_dataset_task_completion/v2"
)
V5_K1_BALANCED_DATASET_COMPLETION_VERSION = (
    "posterior_v8_v5_2_completion_last_branch_pure_grouped_shard_v2"
)


def _load_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _load_plan(path: Path, expected_sha256: str | None) -> dict[str, object]:
    value = validate_v5_k1_balanced_dataset_launch_plan(
        _load_json(path, "K1 balanced dataset launch plan", 16 * 1024 * 1024)
    )
    if expected_sha256 is not None and value["plan_sha256"] != digest(
        expected_sha256, "expected_plan_sha256"
    ):
        raise ValueError("launch plan SHA-256 differs from the Slurm export binding")
    if str(path.resolve(strict=True)) != value["layout"]["plan"]:
        raise ValueError("launch plan path is not its frozen publication path")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError("launch plan must be 0400/nlink1")
    return value


def _task(plan: Mapping[str, object], array_task_id: int) -> Mapping[str, object]:
    if isinstance(array_task_id, bool) or not isinstance(array_task_id, int):
        raise TypeError("array_task_id must be an integer")
    tasks = plan["tasks"]
    if not 0 <= array_task_id < len(tasks):
        raise ValueError("array_task_id is outside the frozen task inventory")
    selected = tasks[array_task_id]
    if selected["array_task_id"] != array_task_id:
        raise RuntimeError("array task mapping does not reproduce")
    return selected


def _worker_guard(
    *, dry_run: bool, hostname: str, environment: Mapping[str, str]
) -> None:
    if dry_run:
        return
    if hostname.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("balanced K1 dataset generation is forbidden on Maxwell login nodes")
    job_id = environment.get("SLURM_JOB_ID", "")
    task_id = environment.get("SLURM_ARRAY_TASK_ID", "")
    if not job_id.isdigit() or not task_id.isdigit():
        raise RuntimeError("balanced K1 dataset generation requires a Slurm array worker")


def _write_completion(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite a K1 dataset task completion")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("K1 dataset task completion is not 0400/nlink1")


def run_v5_k1_balanced_dataset_task(
    plan_path: str | os.PathLike[str],
    array_task_id: int,
    *,
    expected_plan_sha256: str | None = None,
    dry_run: bool = False,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate one frozen task and publish completion only after the shard."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    _worker_guard(dry_run=dry_run, hostname=host, environment=env)
    path = lexical_no_symlinks(
        Path(plan_path), "K1 balanced dataset launch plan"
    ).resolve(strict=True)
    plan = _load_plan(path, expected_plan_sha256)
    selected = _task(plan, array_task_id)
    inputs_before = replay_v5_k1_balanced_dataset_launch_inputs(
        plan, allowed_root=allowed_root
    )
    balanced_path = Path(plan["balanced_dataset_plan"]["path"])
    balanced = v5_k1_balanced_dataset_plan_from_payload(
        _load_json(balanced_path, "K1 balanced dataset authoring plan", 1024 * 1024)
    )
    shard = plan_v5_k1_balanced_grouped_shard(
        dataset_plan=balanced,
        block_sha256=selected["balanced_sobol_block_sha256"],
        start=selected["split_offset"],
        count=selected["recipe_count"],
        view_indices=tuple(plan["configuration"]["view_indices"]),
    )
    if (
        shard.block.role != selected["role"]
        or shard.block.split_id != selected["split_id"]
        or shard.block.branch_id != selected["branch_id"]
        or shard.block.branch_ordinal != selected["branch_ordinal"]
        or len(shard.selected_indices) != selected["recipe_count"]
    ):
        raise RuntimeError("balanced K1 shard task does not reproduce its branch window")
    output = Path(selected["output"])
    completion = Path(selected["completion"])
    if output.exists() or output.is_symlink() or completion.exists() or completion.is_symlink():
        raise FileExistsError("refusing to reuse a K1 dataset artifact or completion")
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": array_task_id,
            "selection_sha256": shard.selection_sha256,
            "input_identity": inputs_before,
            "training_authorization_granted": False,
        }
    if str(array_task_id) != env["SLURM_ARRAY_TASK_ID"]:
        raise RuntimeError("array_task_id differs from SLURM_ARRAY_TASK_ID")
    if not output.parent.is_dir() or not completion.parent.is_dir():
        raise FileNotFoundError("task output and completion parents must already exist")
    _, receipt = build_v5_k1_balanced_grouped_shard(
        shard,
        output,
        allowed_root=allowed_root,
    )
    metadata = output.stat()
    if (
        receipt.path != output
        or file_sha256(output, "balanced K1 grouped shard") != receipt.artifact_sha256
        or metadata.st_size != receipt.byte_count
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
    ):
        raise RuntimeError("published balanced K1 grouped shard identity is incomplete")
    inputs_after = replay_v5_k1_balanced_dataset_launch_inputs(
        plan, allowed_root=allowed_root
    )
    if inputs_after != inputs_before:
        raise RuntimeError("balanced K1 immutable inputs changed during generation")
    core = {
        "schema": V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
        "version": V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "plan_sha256": plan["plan_sha256"],
        "array_task_id": array_task_id,
        "slurm_job_id": env["SLURM_JOB_ID"],
        "hostname": host,
        "task": dict(selected),
        "selection_sha256": shard.selection_sha256,
        "artifact": {
            "path": str(output),
            "artifact_sha256": receipt.artifact_sha256,
            "manifest_sha256": receipt.manifest_sha256,
            "byte_count": receipt.byte_count,
            "mode_octal": "0400",
            "nlink": 1,
        },
        "immutable_input_identity_pre": inputs_before,
        "immutable_input_identity_post": inputs_after,
        "completion_written_after_artifact_seal": True,
    }
    result = {
        **core,
        "completion_sha256": sha256(canonical_json(core).encode()).hexdigest(),
    }
    _write_completion(completion, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--array-task-id", required=True, type=int)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_v5_k1_balanced_dataset_task(
        args.plan,
        args.array_task_id,
        expected_plan_sha256=args.expected_plan_sha256,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA",
    "V5_K1_BALANCED_DATASET_COMPLETION_VERSION",
    "main",
    "run_v5_k1_balanced_dataset_task",
]
