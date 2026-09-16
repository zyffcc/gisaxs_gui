"""Hardened held-readback launcher for formal K1 IID calibration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Mapping, Sequence

from .k1_iid_calibration_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    V5_K1_IID_CALIBRATION_LAUNCH_VERSION,
    V5K1IIDCalibrationLaunchConfig,
    build_v5_k1_iid_calibration_launch_plan,
    replay_v5_k1_iid_calibration_launch_inputs,
    write_v5_k1_iid_calibration_launch_plan,
)
from .k1_staging_files_v5 import file_sha256
from .launch_k1_balanced_dataset_dag_v5 import (
    CommandRunner,
    V5CommandResult,
    _default_runner,
    _job_id,
    _run,
    _safe_export,
    _sealed_json,
    _snapshot,
    _require_submission_host,
    _verify_whole_array_afterok,
)


POSTERIOR_ROOT_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
CALIBRATION_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_iid_calibration_cpu.sbatch"
)
COLLECTOR_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_iid_calibration_collect_cpu.sbatch"
)
V5_K1_IID_CALIBRATION_HELD_RECEIPT_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_calibration_held_submission_receipt/v2"
)
V5_K1_IID_CALIBRATION_LAUNCH_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_iid_calibration_launch_completion/v2"
)
K1_IID_CALIBRATION_LOG_PREFIX = "k1-iid-calibration-v3"
_CANONICAL_PACKAGE = "utils.ML_Fitting_1D_GISAXS.PosteriorV8"


def _environment(plan: Mapping[str, object]) -> dict[str, str]:
    return {
        "POSTERIOR_V8_SOURCE_ROOT": plan["source"]["root"],
        "POSTERIOR_V8_K1_IID_CALIBRATION_PLAN": plan["layout"]["plan"],
        "POSTERIOR_V8_K1_IID_CALIBRATION_PLAN_SHA256": plan["plan_sha256"],
    }


def _calibration_command(plan: Mapping[str, object]) -> tuple[str, ...]:
    wrapper = Path(plan["source"]["root"]) / CALIBRATION_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--array={plan['array']['array_spec']}",
        "--job-name=gisaxs-v5-k1-iid-calibration-v3-data",
        f"--output={plan['layout']['logs']}/{K1_IID_CALIBRATION_LOG_PREFIX}-data-%A_%a.out",
        f"--error={plan['layout']['logs']}/{K1_IID_CALIBRATION_LOG_PREFIX}-data-%A_%a.err",
        _safe_export(_environment(plan)),
        str(wrapper),
    )


def _collector_command(
    plan: Mapping[str, object], calibration_job_id: str
) -> tuple[str, ...]:
    wrapper = Path(plan["source"]["root"]) / COLLECTOR_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--dependency=afterok:{calibration_job_id}",
        "--job-name=gisaxs-v5-k1-iid-calibration-v3-collect",
        f"--output={plan['layout']['logs']}/{K1_IID_CALIBRATION_LOG_PREFIX}-collect-%j.out",
        f"--error={plan['layout']['logs']}/{K1_IID_CALIBRATION_LOG_PREFIX}-collect-%j.err",
        _safe_export(_environment(plan)),
        str(wrapper),
    )


def _prepare_layout(plan: Mapping[str, object]) -> Path:
    run_root = Path(plan["layout"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "data", "task_completion", "audit"):
        Path(plan["layout"][name]).mkdir(exist_ok=False)
    Path(plan["layout"]["calibration_artifact"]).parent.mkdir(exist_ok=False)
    return run_root


def _plan_file_identity(path: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "path": str(path),
        "file_sha256": file_sha256(path, "IID calibration launch plan"),
        "byte_count": metadata.st_size,
        "mode_octal": "0400",
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "nlink": metadata.st_nlink,
    }


def launch_v5_k1_iid_calibration_dag(
    config: V5K1IIDCalibrationLaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
) -> dict[str, object]:
    """Preview or submit the held 60-stratum array and collector."""

    if type(submit) is not bool:
        raise TypeError("submit must be a bool")
    plan = build_v5_k1_iid_calibration_launch_plan(config, allowed_root=allowed_root)
    preview = {
        "calibration_array": list(_calibration_command(plan)),
        "collector": list(_collector_command(plan, "CALIBRATION_ARRAY_JOB_ID")),
    }
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "plan": plan,
            "submission_preview": preview,
        }
    if (
        plan["array"]
        != {
            "array_spec": "0-59%60",
            "task_count": 60,
            "expected_sample_count": 160020,
        }
        or plan["calibration_population_plan"]["randomized_sobol_or_qmc_points_used"]
        is not False
    ):
        raise RuntimeError("submit mode requires the exact formal IID calibration plan")
    host = socket.gethostname() if hostname is None else hostname
    _require_submission_host(host)
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("IID calibration launcher cannot run in a Slurm allocation")
    _prepare_layout(plan)
    plan_path = write_v5_k1_iid_calibration_launch_plan(plan["layout"]["plan"], plan)
    command_runner = _default_runner if runner is None else runner
    job_ids: dict[str, str] = {}
    held_snapshots: dict[str, dict[str, str]] = {}
    release_snapshots: dict[str, dict[str, str]] = {}
    stage = "submit_calibration_array"
    try:
        replay_v5_k1_iid_calibration_launch_inputs(plan, allowed_root=allowed_root)
        result = _run(command_runner, _calibration_command(plan), stage)
        job_ids["calibration_array"] = _job_id(result, stage)
        stage = "submit_collector"
        replay_v5_k1_iid_calibration_launch_inputs(plan, allowed_root=allowed_root)
        result = _run(
            command_runner,
            _collector_command(plan, job_ids["calibration_array"]),
            stage,
        )
        job_ids["collector"] = _job_id(result, stage)
        for name in ("calibration_array", "collector"):
            stage = f"readback_held_{name}"
            held_snapshots[name] = _snapshot(
                command_runner, job_ids[name], held=True
            )
        _verify_whole_array_afterok(
            held_snapshots["collector"], job_ids["calibration_array"]
        )
        plan_identity = _plan_file_identity(plan_path)
        receipt_core = {
            "schema": V5_K1_IID_CALIBRATION_HELD_RECEIPT_SCHEMA,
            "version": V5_K1_IID_CALIBRATION_LAUNCH_VERSION,
            "status": "ALL_JOBS_HELD",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_file_identity": plan_identity,
            "job_ids": job_ids,
            "held_scheduler_snapshots": held_snapshots,
            "dependency": f"afterok:{job_ids['calibration_array']}",
            "all_jobs_submitted_held": True,
            "release_order": ["collector", "calibration_array"],
            "cancellation_attempted": False,
        }
        receipt_path = Path(plan["layout"]["held_submission_receipt"])
        receipt = _sealed_json(receipt_path, receipt_core, "receipt_sha256")
        for name in ("collector", "calibration_array"):
            stage = f"release_{name}"
            _run(command_runner, ("scontrol", "release", job_ids[name]), stage)
            release_snapshots[name] = _snapshot(
                command_runner, job_ids[name], held=False
            )
        completion_core = {
            "schema": V5_K1_IID_CALIBRATION_LAUNCH_COMPLETION_SCHEMA,
            "version": V5_K1_IID_CALIBRATION_LAUNCH_VERSION,
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
            "release_order": ["collector", "calibration_array"],
            "released_scheduler_snapshots": release_snapshots,
            "all_jobs_released": True,
            "compatibility_threshold_authorization_granted": False,
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
            "schema": V5_K1_IID_CALIBRATION_LAUNCH_COMPLETION_SCHEMA,
            "version": V5_K1_IID_CALIBRATION_LAUNCH_VERSION,
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
    parser.add_argument("--three-way-receipt", required=True, type=Path)
    parser.add_argument("--three-way-receipt-file-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.submit and __package__ != _CANONICAL_PACKAGE:
        raise RuntimeError("submit mode requires the canonical repository package path")
    result = launch_v5_k1_iid_calibration_dag(
        V5K1IIDCalibrationLaunchConfig(
            source_root=args.source_root,
            source_archive=args.source_archive,
            expected_source_archive_sha256=args.source_archive_sha256,
            three_way_disjointness_receipt=args.three_way_receipt,
            expected_three_way_receipt_file_sha256=(
                args.three_way_receipt_file_sha256
            ),
            run_root=args.run_root,
        ),
        submit=args.submit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "K1_IID_CALIBRATION_LOG_PREFIX",
    "V5CommandResult",
    "V5_K1_IID_CALIBRATION_HELD_RECEIPT_SCHEMA",
    "V5_K1_IID_CALIBRATION_LAUNCH_COMPLETION_SCHEMA",
    "launch_v5_k1_iid_calibration_dag",
    "main",
]
