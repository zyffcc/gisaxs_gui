"""Held-readback launcher for balanced all-K1 full-search supervision."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from typing import Callable, Mapping, Sequence

from .calibrated_search_threshold_v5 import (
    read_v5_checked_compatibility_calibration,
)
from .exact_search_schedule_v5 import read_v5_frozen_local_sobol_schedule
from .grouped_artifact_v5 import canonical_json
from .k1_balanced_full_search_plan_v5 import (
    MAXWELL_DUST_ROOT,
    validate_v5_k1_balanced_full_search_plan,
    write_v5_k1_balanced_full_search_plan,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks
from .k1_training_chain_plan_v5 import fingerprint_v5_k1_training_source
from .launch_k1_balanced_dataset_dag_v5 import _verify_whole_array_afterok


POSTERIOR_ROOT_RELATIVE = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
WORKER_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_balanced_full_search_cpu.sbatch"
)
COLLECTOR_WRAPPER_RELATIVE = (
    POSTERIOR_ROOT_RELATIVE / "slurm/v5_k1_balanced_full_search_collect_cpu.sbatch"
)
V5_K1_BALANCED_FULL_SEARCH_HELD_RECEIPT_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_held_submission_receipt/v3"
)
V5_K1_BALANCED_FULL_SEARCH_LAUNCH_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_launch_completion/v3"
)
V5_K1_BALANCED_FULL_SEARCH_LAUNCH_VERSION = (
    "posterior_v8_v5_2_all60_maxwell_submission_host_held_readback_"
    "exact_whole_array_dependency_reverse_release_v3"
)
K1_BALANCED_FULL_SEARCH_LOG_PREFIX = "k1-balanced-full-search-v3"
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")
_FIELD = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")
_MAXWELL_SUBMISSION_HOST = re.compile(r"(?:max-wgs\d*|max-fs-display\d*)\Z")


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


def _sealed_file(path: Path, name: str, expected_sha256: str | None = None) -> str:
    selected = lexical_no_symlinks(path, name).resolve(strict=True)
    metadata = selected.stat()
    if (
        not selected.is_file()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a 0400/nlink1 regular file")
    observed = file_sha256(selected, name)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"{name} file SHA-256 drifted")
    return observed


def _under_root(path: Path, allowed_root: Path, name: str) -> Path:
    selected = lexical_no_symlinks(path, name).resolve(strict=True)
    root = Path(allowed_root).resolve(strict=True)
    if selected == root or root not in selected.parents:
        raise ValueError(f"{name} must remain below {allowed_root}")
    return selected


def replay_v5_k1_balanced_full_search_launch_inputs(
    plan_payload: Mapping[str, object],
    *,
    source_root: str | os.PathLike[str],
    source_archive: str | os.PathLike[str],
    local_sobol_schedule_path: str | os.PathLike[str],
    calibration_path: str | os.PathLike[str],
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Rehash every external launch input and replay schedule/calibration logic."""

    plan = validate_v5_k1_balanced_full_search_plan(plan_payload)
    source = _under_root(Path(source_root), allowed_root, "source root")
    if not source.is_dir():
        raise ValueError("source root must be a directory")
    fingerprint = fingerprint_v5_k1_training_source(source)
    if (
        fingerprint["bundle_sha256"] != plan["source"]["bundle_sha256"]
        or fingerprint["source_snapshot_write_bits_set"]
        or fingerprint["source_files_with_write_bits"]
    ):
        raise RuntimeError("balanced full-search source snapshot drifted")
    archive = _under_root(Path(source_archive), allowed_root, "source archive")
    archive_sha = _sealed_file(
        archive, "source archive", plan["source"]["archive_sha256"]
    )
    schedule_path = _under_root(
        Path(local_sobol_schedule_path), allowed_root, "local Sobol schedule"
    )
    schedule_file_sha = _sealed_file(schedule_path, "local Sobol schedule")
    schedule, schedule_receipt = read_v5_frozen_local_sobol_schedule(schedule_path)
    stage = plan["k1_stage"]["payload"]
    if (
        schedule.sha256 != stage["local_sobol_schedule_sha256"]
        or schedule.audit_payload() != stage["local_sobol_schedule"]
    ):
        raise ValueError("local Sobol schedule escaped the frozen K1 stage")
    calibration = _under_root(Path(calibration_path), allowed_root, "calibration")
    calibration_file_sha = _sealed_file(calibration, "compatibility calibration")
    calibration_identity = stage["protocol"]["calibration_identity"]
    checked_calibration = read_v5_checked_compatibility_calibration(
        calibration,
        expected_artifact_sha256=calibration_identity["artifact_sha256"],
        expected_file_sha256=calibration_identity["file_sha256"],
    )
    return {
        "source_root": str(source),
        "source_bundle_sha256": fingerprint["bundle_sha256"],
        "source_archive_path": str(archive),
        "source_archive_sha256": archive_sha,
        "local_sobol_schedule_path": str(schedule_path),
        "local_sobol_schedule_file_sha256": schedule_file_sha,
        "local_sobol_schedule_sha256": schedule.sha256,
        "local_sobol_schedule_artifact_sha256": schedule_receipt.artifact_sha256,
        "calibration_path": str(calibration),
        "calibration_file_sha256": calibration_file_sha,
        "calibration_artifact_sha256": checked_calibration.identity.artifact_sha256,
    }


def _environment(
    plan: Mapping[str, object],
    inputs: Mapping[str, object],
    plan_file_sha256: str,
) -> dict[str, str]:
    return {
        "POSTERIOR_V8_SOURCE_ROOT": str(inputs["source_root"]),
        "POSTERIOR_V8_SOURCE_ARCHIVE": str(inputs["source_archive_path"]),
        "POSTERIOR_V8_K1_FULL_SEARCH_PLAN": plan["layout"]["plan"],
        "POSTERIOR_V8_K1_FULL_SEARCH_PLAN_SHA256": plan["plan_sha256"],
        "POSTERIOR_V8_K1_FULL_SEARCH_PLAN_FILE_SHA256": plan_file_sha256,
        "POSTERIOR_V8_K1_FULL_SEARCH_LOCAL_SOBOL_SCHEDULE": str(
            inputs["local_sobol_schedule_path"]
        ),
        "POSTERIOR_V8_K1_FULL_SEARCH_CALIBRATION": str(inputs["calibration_path"]),
    }


def _array_command(
    plan: Mapping[str, object], environment: Mapping[str, str]
) -> tuple[str, ...]:
    wrapper = Path(environment["POSTERIOR_V8_SOURCE_ROOT"]) / WORKER_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--array={plan['array']['array_spec']}",
        "--job-name=gisaxs-v5-k1-balanced-full-search-v3",
        f"--output={plan['layout']['logs']}/{K1_BALANCED_FULL_SEARCH_LOG_PREFIX}-%A_%a.out",
        f"--error={plan['layout']['logs']}/{K1_BALANCED_FULL_SEARCH_LOG_PREFIX}-%A_%a.err",
        _safe_export(environment),
        str(wrapper),
    )


def _collector_command(
    plan: Mapping[str, object], environment: Mapping[str, str], array_job_id: str
) -> tuple[str, ...]:
    wrapper = Path(environment["POSTERIOR_V8_SOURCE_ROOT"]) / COLLECTOR_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--dependency=afterok:{array_job_id}",
        "--job-name=gisaxs-v5-k1-balanced-full-search-v3-collect",
        f"--output={plan['layout']['logs']}/{K1_BALANCED_FULL_SEARCH_LOG_PREFIX}-collect-%j.out",
        f"--error={plan['layout']['logs']}/{K1_BALANCED_FULL_SEARCH_LOG_PREFIX}-collect-%j.err",
        _safe_export(environment),
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
    if held and (state != "PENDING" or reason != "JobHeldUser"):
        raise RuntimeError(f"Slurm job {job_id} is not user-held after submission")
    if not held and reason == "JobHeldUser":
        raise RuntimeError(f"Slurm job {job_id} remains user-held after release")
    return {"job_id": job_id, "job_state": state, "reason": reason, "raw": result.stdout.strip()}


def _write_evidence(
    path: Path, core: Mapping[str, object], hash_field: str
) -> dict[str, object]:
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
        raise RuntimeError(f"launch evidence {path.name} is not 0400/nlink1")
    return payload


def _file_identity(path: Path, name: str) -> dict[str, object]:
    metadata = path.stat()
    if (
        not path.is_file()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a 0400/nlink1 regular file")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path, name),
        "byte_count": metadata.st_size,
        "mode_octal": "0400",
        "nlink": 1,
    }


def _prepare_layout(plan: Mapping[str, object]) -> None:
    root = Path(plan["layout"]["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    Path(plan["layout"]["logs"]).mkdir()
    Path(plan["layout"]["shards"]).mkdir()
    Path(plan["layout"]["completion"]).parent.mkdir()


def launch_v5_k1_balanced_full_search_dag(
    plan_payload: Mapping[str, object],
    *,
    source_root: str | os.PathLike[str],
    source_archive: str | os.PathLike[str],
    local_sobol_schedule_path: str | os.PathLike[str],
    calibration_path: str | os.PathLike[str],
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
) -> dict[str, object]:
    """Preview or submit the held array/collector dependency pair."""

    if type(submit) is not bool:
        raise TypeError("submit must be a bool")
    plan = validate_v5_k1_balanced_full_search_plan(plan_payload)
    inputs = replay_v5_k1_balanced_full_search_launch_inputs(
        plan,
        source_root=source_root,
        source_archive=source_archive,
        local_sobol_schedule_path=local_sobol_schedule_path,
        calibration_path=calibration_path,
        allowed_root=allowed_root,
    )
    preview_environment = _environment(plan, inputs, "PLAN_FILE_SHA256_AFTER_WRITE")
    preview = {
        "array": list(_array_command(plan, preview_environment)),
        "collector": list(
            _collector_command(plan, preview_environment, "ARRAY_JOB_ID")
        ),
    }
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "plan": plan,
            "input_identity": inputs,
            "submission_preview": preview,
        }
    host = socket.gethostname() if hostname is None else hostname
    short_host = host.split(".", 1)[0]
    if _MAXWELL_SUBMISSION_HOST.fullmatch(short_host) is None:
        raise RuntimeError(
            "balanced full-search submit mode requires a Maxwell submission host"
        )
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("balanced full-search launcher cannot run in Slurm")
    _prepare_layout(plan)
    plan_path = write_v5_k1_balanced_full_search_plan(plan["layout"]["plan"], plan)
    plan_identity = _file_identity(plan_path, "balanced full-search plan")
    plan_file_sha = plan_identity["file_sha256"]
    environment = _environment(plan, inputs, plan_file_sha)
    command_runner = _default_runner if runner is None else runner
    root = Path(plan["layout"]["run_root"])
    receipt_path = root / "audit" / "held-submission-receipt-v3.json"
    launch_completion_path = root / "audit" / "launch-completion-v3.json"
    launch_failure_path = root / "audit" / "launch-failure-v3.json"
    job_ids: dict[str, str] = {}
    held_snapshots: dict[str, dict[str, str]] = {}
    released_snapshots: dict[str, dict[str, str]] = {}
    stage = "submit_array"
    try:
        replay = replay_v5_k1_balanced_full_search_launch_inputs(
            plan,
            source_root=source_root,
            source_archive=source_archive,
            local_sobol_schedule_path=local_sobol_schedule_path,
            calibration_path=calibration_path,
            allowed_root=allowed_root,
        )
        if replay != inputs:
            raise RuntimeError("full-search launch inputs changed before submission")
        result = _run(command_runner, _array_command(plan, environment), stage)
        job_ids["array"] = _job_id(result, stage)
        stage = "submit_collector"
        result = _run(
            command_runner,
            _collector_command(plan, environment, job_ids["array"]),
            stage,
        )
        job_ids["collector"] = _job_id(result, stage)
        for name in ("array", "collector"):
            stage = f"readback_held_{name}"
            held_snapshots[name] = _snapshot(
                command_runner, job_ids[name], held=True
            )
        _verify_whole_array_afterok(held_snapshots["collector"], job_ids["array"])
        receipt_core = {
            "schema": V5_K1_BALANCED_FULL_SEARCH_HELD_RECEIPT_SCHEMA,
            "version": V5_K1_BALANCED_FULL_SEARCH_LAUNCH_VERSION,
            "status": "ALL_JOBS_HELD",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_identity": plan_identity,
            "input_identity": inputs,
            "job_ids": job_ids,
            "held_readback": held_snapshots,
            "submission_commands": {
                "array": list(_array_command(plan, environment)),
                "collector": list(
                    _collector_command(plan, environment, job_ids["array"])
                ),
            },
            "all_jobs_submitted_held": True,
            "release_order": ["collector", "array"],
            "gradient_training_authorized": False,
        }
        receipt = _write_evidence(receipt_path, receipt_core, "receipt_sha256")
        for name in ("collector", "array"):
            stage = f"release_{name}"
            _run(command_runner, ("scontrol", "release", job_ids[name]), stage)
            released_snapshots[name] = _snapshot(
                command_runner, job_ids[name], held=False
            )
        replay_after = replay_v5_k1_balanced_full_search_launch_inputs(
            plan,
            source_root=source_root,
            source_archive=source_archive,
            local_sobol_schedule_path=local_sobol_schedule_path,
            calibration_path=calibration_path,
            allowed_root=allowed_root,
        )
        if replay_after != inputs:
            raise RuntimeError("full-search launch inputs changed during release")
        completion_core = {
            "schema": V5_K1_BALANCED_FULL_SEARCH_LAUNCH_COMPLETION_SCHEMA,
            "version": V5_K1_BALANCED_FULL_SEARCH_LAUNCH_VERSION,
            "status": "ALL_JOBS_RELEASED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_identity": plan_identity,
            "receipt": {
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path, "held receipt"),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "job_ids": job_ids,
            "held_readback": held_snapshots,
            "released_readback": released_snapshots,
            "input_identity_pre": inputs,
            "input_identity_post": replay_after,
            "completion_written_after_reverse_release": True,
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
        }
        return _write_evidence(
            launch_completion_path, completion_core, "completion_sha256"
        )
    except (Exception, KeyboardInterrupt) as exc:
        if not launch_failure_path.exists() and not launch_failure_path.is_symlink():
            failure_core = {
                "schema": V5_K1_BALANCED_FULL_SEARCH_LAUNCH_COMPLETION_SCHEMA,
                "version": V5_K1_BALANCED_FULL_SEARCH_LAUNCH_VERSION,
                "status": "FAIL",
                "failed_stage": stage,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:2000],
                "plan_sha256": plan["plan_sha256"],
                "job_ids": job_ids,
                "held_readback": held_snapshots,
                "released_readback": released_snapshots,
                "submitted_jobs_are_not_cancelled_automatically": True,
                "partial_outputs_retained": True,
            }
            _write_evidence(launch_failure_path, failure_core, "failure_sha256")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = launch_v5_k1_balanced_full_search_dag(
        plan,
        source_root=args.source_root,
        source_archive=args.source_archive,
        local_sobol_schedule_path=args.local_sobol_schedule,
        calibration_path=args.calibration,
        submit=args.submit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "K1_BALANCED_FULL_SEARCH_LOG_PREFIX",
    "V5CommandResult",
    "V5_K1_BALANCED_FULL_SEARCH_HELD_RECEIPT_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_LAUNCH_COMPLETION_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_LAUNCH_VERSION",
    "launch_v5_k1_balanced_full_search_dag",
    "main",
    "replay_v5_k1_balanced_full_search_launch_inputs",
]
