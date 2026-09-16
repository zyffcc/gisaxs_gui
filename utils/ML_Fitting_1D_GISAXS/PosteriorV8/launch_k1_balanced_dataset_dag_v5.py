"""Hardened held-readback launcher for balanced all-K1 dataset production."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
from typing import Callable, Mapping, Sequence

from .grouped_artifact_v5 import canonical_json
from .k1_balanced_dataset_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    V5_K1_BALANCED_DATASET_LAUNCH_VERSION,
    V5K1BalancedDatasetLaunchConfig,
    build_v5_k1_balanced_dataset_launch_plan,
    replay_v5_k1_balanced_dataset_launch_inputs,
    write_v5_k1_balanced_dataset_launch_plan,
)
from .k1_balanced_dataset_plan_v5 import K1_BALANCED_CANONICAL_RUNTIME_PACKAGE
from .k1_staging_files_v5 import file_sha256


POSTERIOR_ROOT_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
DATASET_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_balanced_dataset_cpu.sbatch"
)
COLLECTOR_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_balanced_dataset_collect_cpu.sbatch"
)
V5_K1_BALANCED_DATASET_HELD_RECEIPT_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_dataset_held_submission_receipt/v2"
)
V5_K1_BALANCED_DATASET_LAUNCH_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_dataset_launch_completion/v2"
)
K1_BALANCED_DATASET_LOG_PREFIX = "k1-balanced-all12-v4"
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")
_FIELD = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")


@dataclass(frozen=True)
class V5CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], V5CommandResult]


def _default_runner(argv: Sequence[str]) -> V5CommandResult:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603
    return V5CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _run(runner: CommandRunner, argv: Sequence[str], stage: str) -> V5CommandResult:
    result = runner(tuple(argv))
    if not isinstance(result, V5CommandResult):
        raise TypeError("command runner must return V5CommandResult")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "command failed"
        raise RuntimeError(f"{stage} failed ({result.returncode}): {detail[:1000]}")
    return result


def _job_id(result: V5CommandResult, stage: str) -> str:
    token = result.stdout.strip().split(";", 1)[0]
    if not token.isdigit() or int(token) < 1:
        raise RuntimeError(f"{stage} returned an invalid parsable Slurm job ID")
    return token


def _safe_export(values: Mapping[str, object]) -> str:
    fields = []
    for name, value in values.items():
        text = str(value)
        if any(mark in text for mark in (",", "\n", "\r", "\0")):
            raise ValueError(f"{name} contains an unsafe Slurm export character")
        fields.append(f"{name}={text}")
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *fields))


def _environment(plan: Mapping[str, object]) -> dict[str, str]:
    return {
        "POSTERIOR_V8_SOURCE_ROOT": plan["source"]["root"],
        "POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN": plan["layout"]["plan"],
        "POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN_SHA256": plan["plan_sha256"],
    }


def _dataset_command(plan: Mapping[str, object]) -> tuple[str, ...]:
    wrapper = Path(plan["source"]["root"]) / DATASET_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--array={plan['array']['array_spec']}",
        f"--job-name=gisaxs-v5-{K1_BALANCED_DATASET_LOG_PREFIX}-data",
        f"--output={plan['layout']['logs']}/{K1_BALANCED_DATASET_LOG_PREFIX}-data-%A_%a.out",
        f"--error={plan['layout']['logs']}/{K1_BALANCED_DATASET_LOG_PREFIX}-data-%A_%a.err",
        _safe_export(_environment(plan)),
        str(wrapper),
    )


def _collector_command(plan: Mapping[str, object], dataset_job_id: str) -> tuple[str, ...]:
    wrapper = Path(plan["source"]["root"]) / COLLECTOR_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--dependency=afterok:{dataset_job_id}",
        f"--job-name=gisaxs-v5-{K1_BALANCED_DATASET_LOG_PREFIX}-collect",
        f"--output={plan['layout']['logs']}/{K1_BALANCED_DATASET_LOG_PREFIX}-collect-%j.out",
        f"--error={plan['layout']['logs']}/{K1_BALANCED_DATASET_LOG_PREFIX}-collect-%j.err",
        _safe_export(_environment(plan)),
        str(wrapper),
    )


def _snapshot(runner: CommandRunner, job_id: str, *, held: bool) -> dict[str, str]:
    result = _run(
        runner,
        ("scontrol", "show", "job", "--oneliner", job_id),
        f"read back Slurm job {job_id}",
    )
    fields = {name: value for name, value in _FIELD.findall(result.stdout)}
    if fields.get("JobId") != job_id:
        raise RuntimeError(f"Slurm readback returned another job for {job_id}")
    state = fields.get("JobState", "")
    reason = fields.get("Reason", "")
    if held:
        if state != "PENDING" or reason != "JobHeldUser":
            raise RuntimeError(f"Slurm job {job_id} is not user-held after submission")
    elif reason == "JobHeldUser":
        raise RuntimeError(f"Slurm job {job_id} remains user-held after release")
    return {"job_id": job_id, "job_state": state, "reason": reason, "raw": result.stdout.strip()}


def _require_submission_host(host: str) -> None:
    if re.fullmatch(r"(?:max-wgs[0-9]*|max-fs-display[0-9]*)", host.split(".", 1)[0]) is None:
        raise RuntimeError("submission requires a Maxwell login host (max-wgs or max-fs-display)")


def _verify_whole_array_afterok(snapshot: Mapping[str, str], array_job_id: str) -> None:
    """Require one exact dependency, including Slurm's whole-array spelling."""
    if not isinstance(array_job_id, str) or re.fullmatch(r"[1-9][0-9]*", array_job_id) is None:
        raise ValueError("array job ID must be a positive decimal identifier")
    dependencies = [
        value for name, value in _FIELD.findall(snapshot["raw"])
        if name == "Dependency"
    ]
    # A substring elsewhere (notably SubmitLine) cannot prove the live dependency.
    expected = rf"afterok:{re.escape(array_job_id)}(?:_\*)?(?:\(unfulfilled\))?"
    if len(dependencies) != 1 or re.fullmatch(expected, dependencies[0]) is None:
        raise RuntimeError("collector readback lost its exact whole-array afterok dependency")


def _sealed_json(path: Path, core: Mapping[str, object], hash_field: str) -> dict[str, object]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite launch evidence: {path}")
    payload = {
        **core,
        hash_field: sha256(canonical_json(core).encode()).hexdigest(),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("launch evidence is not 0400/nlink1")
    return payload


def _plan_file_identity(path: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "path": str(path),
        "file_sha256": file_sha256(path, "balanced dataset launch plan"),
        "byte_count": metadata.st_size,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "nlink": metadata.st_nlink,
    }


def _prepare_layout(plan: Mapping[str, object]) -> Path:
    run_root = Path(plan["layout"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "data", "completion", "audit"):
        Path(plan["layout"][name]).mkdir(exist_ok=False)
    for task in plan["tasks"]:
        Path(task["output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(task["completion"]).parent.mkdir(parents=True, exist_ok=True)
    return run_root


def launch_v5_k1_balanced_dataset_dag(
    config: V5K1BalancedDatasetLaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
) -> dict[str, object]:
    """Preview or submit the held dataset-array/collector dependency pair."""

    if type(submit) is not bool:
        raise TypeError("submit must be a bool")
    plan = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=allowed_root)
    preview = {
        "dataset_array": list(_dataset_command(plan)),
        "collector": list(_collector_command(plan, "DATASET_ARRAY_JOB_ID")),
    }
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "plan": plan,
            "submission_preview": preview,
        }
    if plan["configuration"].get("formal_e1_dataset_production") is not True:
        raise RuntimeError(
            "submit mode requires the exact frozen E1 counts, seeds, shard size, and views"
        )
    host = socket.gethostname() if hostname is None else hostname
    _require_submission_host(host)
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("balanced K1 dataset launcher cannot run in a Slurm allocation")
    _prepare_layout(plan)
    plan_path = write_v5_k1_balanced_dataset_launch_plan(plan["layout"]["plan"], plan)
    command_runner = _default_runner if runner is None else runner
    job_ids: dict[str, str] = {}
    held_snapshots: dict[str, dict[str, str]] = {}
    release_snapshots: dict[str, dict[str, str]] = {}
    stage = "submit_dataset_array"
    try:
        replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=allowed_root)
        result = _run(command_runner, _dataset_command(plan), stage)
        job_ids["dataset_array"] = _job_id(result, stage)
        stage = "submit_collector"
        replay_v5_k1_balanced_dataset_launch_inputs(plan, allowed_root=allowed_root)
        result = _run(
            command_runner,
            _collector_command(plan, job_ids["dataset_array"]),
            stage,
        )
        job_ids["collector"] = _job_id(result, stage)
        for name in ("dataset_array", "collector"):
            stage = f"readback_held_{name}"
            held_snapshots[name] = _snapshot(
                command_runner, job_ids[name], held=True
            )
        _verify_whole_array_afterok(held_snapshots["collector"], job_ids["dataset_array"])
        plan_identity = _plan_file_identity(plan_path)
        receipt_core = {
            "schema": V5_K1_BALANCED_DATASET_HELD_RECEIPT_SCHEMA,
            "version": V5_K1_BALANCED_DATASET_LAUNCH_VERSION,
            "status": "ALL_JOBS_HELD",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_file_identity": plan_identity,
            "job_ids": job_ids,
            "held_scheduler_snapshots": held_snapshots,
            "dependency": f"afterok:{job_ids['dataset_array']}",
            "all_jobs_submitted_held": True,
            "cancellation_attempted": False,
        }
        receipt_path = Path(plan["layout"]["held_submission_receipt"])
        receipt = _sealed_json(receipt_path, receipt_core, "receipt_sha256")
        for name in ("collector", "dataset_array"):
            stage = f"release_{name}"
            _run(command_runner, ("scontrol", "release", job_ids[name]), stage)
            release_snapshots[name] = _snapshot(
                command_runner, job_ids[name], held=False
            )
        completion_core = {
            "schema": V5_K1_BALANCED_DATASET_LAUNCH_COMPLETION_SCHEMA,
            "version": V5_K1_BALANCED_DATASET_LAUNCH_VERSION,
            "status": "ALL_JOBS_RELEASED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_file_identity": plan_identity,
            "held_receipt": {
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path, "held submission receipt"),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "job_ids": job_ids,
            "release_order": ["collector", "dataset_array"],
            "released_scheduler_snapshots": release_snapshots,
            "all_jobs_released": True,
            "training_authorization_granted": False,
            "phase_c_acceptance_granted": False,
            "cancellation_attempted": False,
        }
        return _sealed_json(
            Path(plan["layout"]["launch_completion"]),
            completion_core,
            "completion_sha256",
        )
    except (Exception, KeyboardInterrupt) as exc:
        failure_core = {
            "schema": V5_K1_BALANCED_DATASET_LAUNCH_COMPLETION_SCHEMA,
            "version": V5_K1_BALANCED_DATASET_LAUNCH_VERSION,
            "status": "FAILED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:2000],
            "plan_sha256": plan["plan_sha256"],
            "plan_file_identity": _plan_file_identity(plan_path),
            "job_ids": job_ids,
            "held_scheduler_snapshots": held_snapshots,
            "released_scheduler_snapshots": release_snapshots,
            "cancellation_attempted": False,
            "submitted_jobs_may_remain_held": True,
        }
        _sealed_json(
            Path(plan["layout"]["launch_failure"]),
            failure_core,
            "failure_sha256",
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--balanced-plan", required=True, type=Path)
    parser.add_argument("--balanced-plan-file-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--recipes-per-shard", required=True, type=int)
    parser.add_argument("--view-indices", required=True)
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.submit and __package__ != K1_BALANCED_CANONICAL_RUNTIME_PACKAGE:
        raise RuntimeError(
            "submit mode requires the canonical repository package module path"
        )
    views = tuple(int(value.strip()) for value in args.view_indices.split(","))
    result = launch_v5_k1_balanced_dataset_dag(
        V5K1BalancedDatasetLaunchConfig(
            source_root=args.source_root,
            source_archive=args.source_archive,
            expected_source_archive_sha256=args.source_archive_sha256,
            balanced_plan=args.balanced_plan,
            expected_balanced_plan_file_sha256=args.balanced_plan_file_sha256,
            run_root=args.run_root,
            recipes_per_shard=args.recipes_per_shard,
            view_indices=views,
        ),
        submit=args.submit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5CommandResult",
    "launch_v5_k1_balanced_dataset_dag",
    "main",
]
