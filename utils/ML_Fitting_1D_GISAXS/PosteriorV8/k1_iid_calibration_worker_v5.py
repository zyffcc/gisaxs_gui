"""Slurm worker for one formal K1 IID calibration acquisition stratum."""

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
from .k1_iid_calibration_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_iid_calibration_launch_inputs,
    validate_v5_k1_iid_calibration_launch_plan,
)
from .k1_iid_calibration_shard_v5 import (
    build_v5_k1_iid_calibration_shard,
    validate_v5_k1_iid_calibration_shard,
    write_v5_k1_iid_calibration_shard,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest


V5_K1_IID_CALIBRATION_TASK_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_calibration_task_completion/v2"
)
V5_K1_IID_CALIBRATION_TASK_COMPLETION_VERSION = (
    "posterior_v8_v5_2_iid_stratum_score_shard_overflow_safe_sigma_"
    "completion_last_v2"
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


def _immutable(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a real regular file")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _write_completion(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite a calibration task completion")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    _immutable(path, "calibration task completion")


def run_v5_k1_iid_calibration_task(
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
            raise RuntimeError("IID calibration score generation is forbidden on Maxwell login nodes")
        if not env.get("SLURM_JOB_ID", "").isdigit() or not env.get(
            "SLURM_ARRAY_TASK_ID", ""
        ).isdigit():
            raise RuntimeError("IID calibration generation requires a Slurm array worker")
    path = lexical_no_symlinks(Path(plan_path), "IID calibration launch plan").resolve(
        strict=True
    )
    _immutable(path, "IID calibration launch plan")
    plan = validate_v5_k1_iid_calibration_launch_plan(
        _load_json(path, "IID calibration launch plan", 16 * 1024 * 1024)
    )
    if plan["plan_sha256"] != digest(expected_plan_sha256, "expected_plan_sha256"):
        raise ValueError("calibration launch plan differs from the Slurm export")
    if str(path) != plan["layout"]["plan"]:
        raise ValueError("calibration launch plan path drifted")
    if isinstance(array_task_id, bool) or not isinstance(array_task_id, int) or not (
        0 <= array_task_id < len(plan["tasks"])
    ):
        raise ValueError("array_task_id is outside the calibration task inventory")
    task = plan["tasks"][array_task_id]
    if task["array_task_id"] != array_task_id:
        raise RuntimeError("calibration task mapping does not reproduce")
    inputs_before = replay_v5_k1_iid_calibration_launch_inputs(
        plan, allowed_root=allowed_root
    )
    output = Path(task["output"])
    completion = Path(task["completion"])
    if output.exists() or output.is_symlink() or completion.exists() or completion.is_symlink():
        raise FileExistsError("refusing to reuse calibration task output/completion")
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": array_task_id,
            "immutable_input_identity": inputs_before,
            "compatibility_threshold_authorization_granted": False,
        }
    if str(array_task_id) != env["SLURM_ARRAY_TASK_ID"]:
        raise RuntimeError("array_task_id differs from SLURM_ARRAY_TASK_ID")
    if not output.parent.is_dir() or not completion.parent.is_dir():
        raise FileNotFoundError("calibration task output parents do not exist")
    shard = build_v5_k1_iid_calibration_shard(
        calibration_plan=plan["calibration_population_plan"],
        stratum_ordinal=task["stratum_ordinal"],
        sample_ordinal_start=task["sample_ordinal_start"],
        sample_count=task["sample_count"],
    )
    write_v5_k1_iid_calibration_shard(output, shard, allowed_root=allowed_root)
    checked = validate_v5_k1_iid_calibration_shard(
        _load_json(output, "IID calibration shard", 128 * 1024 * 1024),
        calibration_plan=plan["calibration_population_plan"],
    )
    inputs_after = replay_v5_k1_iid_calibration_launch_inputs(
        plan, allowed_root=allowed_root
    )
    if inputs_after != inputs_before:
        raise RuntimeError("calibration immutable inputs changed during generation")
    metadata = output.stat()
    core = {
        "schema": V5_K1_IID_CALIBRATION_TASK_COMPLETION_SCHEMA,
        "version": V5_K1_IID_CALIBRATION_TASK_COMPLETION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "model_selection_authorization_granted": False,
        "compatibility_threshold_authorization_granted": False,
        "plan_sha256": plan["plan_sha256"],
        "array_task_id": array_task_id,
        "slurm_job_id": env["SLURM_JOB_ID"],
        "hostname": host,
        "task": task,
        "artifact": {
            "path": str(output),
            "file_sha256": file_sha256(output, "IID calibration shard"),
            "artifact_self_sha256": checked["artifact_self_sha256"],
            "manifest_sha256": checked["manifest"]["manifest_sha256"],
            "byte_count": metadata.st_size,
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
    result = run_v5_k1_iid_calibration_task(
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
    "V5_K1_IID_CALIBRATION_TASK_COMPLETION_SCHEMA",
    "V5_K1_IID_CALIBRATION_TASK_COMPLETION_VERSION",
    "main",
    "run_v5_k1_iid_calibration_task",
]
