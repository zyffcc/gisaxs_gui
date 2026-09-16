"""Slurm worker for one formal K1 Phase-C holdout identity shard."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_holdout_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_phase_c_holdout_launch_inputs,
    validate_v5_k1_phase_c_holdout_launch_plan,
)
from .k1_phase_c_holdout_shard_v5 import (
    plan_v5_k1_phase_c_holdout_shard,
    validate_v5_k1_phase_c_holdout_shard_payload,
    write_v5_k1_phase_c_holdout_shard,
)
from .k1_phase_c_plan_v5 import v5_k1_phase_c_plan_from_payload
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest


V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_holdout_task_completion/v1"
)
V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION = (
    "posterior_v8_v5_2_completion_last_recipe_identity_shard_v1"
)


def _load_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _write_completion(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite Phase-C task completion")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("Phase-C task completion is not 0400/nlink1")


def run_v5_k1_phase_c_holdout_task(
    plan_path: str | os.PathLike[str],
    array_task_id: int,
    *,
    expected_plan_sha256: str,
    dry_run: bool = False,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    if not dry_run:
        if host.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
            raise RuntimeError("Phase-C holdout materialization is forbidden on Maxwell login nodes")
        if not env.get("SLURM_JOB_ID", "").isdigit() or not env.get(
            "SLURM_ARRAY_TASK_ID", ""
        ).isdigit():
            raise RuntimeError("Phase-C holdout materialization requires a Slurm array worker")
    path = lexical_no_symlinks(Path(plan_path), "Phase-C holdout launch plan").resolve(
        strict=True
    )
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError("Phase-C holdout launch plan must be 0400/nlink1")
    plan = validate_v5_k1_phase_c_holdout_launch_plan(
        _load_json(path, "Phase-C holdout launch plan", 16 * 1024 * 1024)
    )
    if plan["plan_sha256"] != digest(expected_plan_sha256, "expected_plan_sha256"):
        raise ValueError("Phase-C launch plan differs from the Slurm export")
    if str(path) != plan["layout"]["plan"]:
        raise ValueError("Phase-C launch plan path drifted")
    if isinstance(array_task_id, bool) or not isinstance(array_task_id, int) or not (
        0 <= array_task_id < len(plan["tasks"])
    ):
        raise ValueError("array_task_id is outside the Phase-C task inventory")
    task = plan["tasks"][array_task_id]
    if task["array_task_id"] != array_task_id:
        raise RuntimeError("Phase-C task mapping does not reproduce")
    inputs_before = replay_v5_k1_phase_c_holdout_launch_inputs(
        plan, allowed_root=allowed_root
    )
    phase_c = v5_k1_phase_c_plan_from_payload(
        _load_json(Path(plan["phase_c_plan"]["path"]), "Phase-C plan", 1024 * 1024)
    )
    shard_plan = plan_v5_k1_phase_c_holdout_shard(
        phase_c_plan=phase_c,
        block_sha256=task["phase_c_sobol_block_sha256"],
        start=task["split_offset"],
        count=task["recipe_count"],
    )
    if (
        shard_plan.block.branch_id != task["branch_id"]
        or shard_plan.block.branch_ordinal != task["branch_ordinal"]
        or len(shard_plan.selected_indices) != task["recipe_count"]
    ):
        raise RuntimeError("Phase-C task window does not reproduce")
    output = Path(task["output"])
    completion = Path(task["completion"])
    if output.exists() or output.is_symlink() or completion.exists() or completion.is_symlink():
        raise FileExistsError("refusing to reuse Phase-C task output/completion")
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": array_task_id,
            "selection_sha256": shard_plan.selection_sha256,
            "immutable_input_identity": inputs_before,
            "training_authorization_granted": False,
        }
    if str(array_task_id) != env["SLURM_ARRAY_TASK_ID"]:
        raise RuntimeError("array_task_id differs from SLURM_ARRAY_TASK_ID")
    if not output.parent.is_dir() or not completion.parent.is_dir():
        raise FileNotFoundError("Phase-C task output parents must exist")
    artifact = write_v5_k1_phase_c_holdout_shard(
        shard_plan, output, allowed_root=allowed_root
    )
    artifact = validate_v5_k1_phase_c_holdout_shard_payload(artifact)
    output_meta = output.stat()
    inputs_after = replay_v5_k1_phase_c_holdout_launch_inputs(
        plan, allowed_root=allowed_root
    )
    if inputs_after != inputs_before:
        raise RuntimeError("Phase-C immutable inputs changed during task execution")
    core = {
        "schema": V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA,
        "version": V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "plan_sha256": plan["plan_sha256"],
        "array_task_id": array_task_id,
        "slurm_job_id": env["SLURM_JOB_ID"],
        "hostname": host,
        "task": task,
        "selection_sha256": shard_plan.selection_sha256,
        "artifact": {
            "path": str(output),
            "file_sha256": file_sha256(output, "Phase-C holdout shard"),
            "artifact_self_sha256": artifact["artifact_self_sha256"],
            "manifest_sha256": artifact["manifest"]["manifest_sha256"],
            "byte_count": output_meta.st_size,
            "mode_octal": "0400",
            "nlink": 1,
        },
        "immutable_input_identity_pre": inputs_before,
        "immutable_input_identity_post": inputs_after,
        "completion_written_after_artifact_seal": True,
    }
    result = {**core, "completion_sha256": sha256(canonical_json(core).encode()).hexdigest()}
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
    result = run_v5_k1_phase_c_holdout_task(
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
    "V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA",
    "V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION",
    "main",
    "run_v5_k1_phase_c_holdout_task",
]
