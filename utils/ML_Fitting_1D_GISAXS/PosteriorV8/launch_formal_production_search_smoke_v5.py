"""Maxwell launcher for V5 formal-production contract smoke.

The launcher is dry-run by default.  Submit mode writes the canonical global
plan durably before the first ``sbatch``, submits all six stage/split arrays on
hold, then releases them in one control operation.  It never submits training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Callable, Mapping, Sequence

from .formal_production_search_contract_v5 import (
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_ALLOWED_SPLITS,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
)
from .formal_production_search_plan_v5 import V5FormalProductionSearchPlan
from .formal_production_search_worker_contract_v5 import (
    prepare_v5_formal_production_worker_shard,
    validate_v5_formal_production_global_plan_payload,
)
from .frozen_search_launch_contracts_v5 import (
    MAXWELL_DUST_ROOT,
    fingerprint_v5_frozen_search_source,
    under_v5_launch_root,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_warmup_launch_plan_v5 import safe_export_value
from .launch_frozen_search_v5 import V5SearchLaunchCommandResult


V5_FORMAL_PRODUCTION_PLAN_FILENAME = "formal-production-search-plan-v5.json"
V5_FORMAL_PRODUCTION_LAUNCH_RECEIPT_FILENAME = (
    "formal-production-contract-smoke-launch.json"
)
V5_FORMAL_PRODUCTION_WRAPPER_RELATIVE = Path(
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/"
    "v5_formal_production_search_smoke_cpu.sbatch"
)
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")


@dataclass(frozen=True)
class V5FormalProductionSmokeLaunchConfig:
    global_plan_payload: Mapping[str, object]
    expected_global_plan_sha256: str
    source_root: Path
    run_root: Path
    split_plan: Path
    sobol_design: Path
    calibration_artifact: Path
    local_sobol_schedules: Mapping[str, Path]

    @classmethod
    def from_plan(
        cls,
        plan: V5FormalProductionSearchPlan,
        **paths: object,
    ) -> "V5FormalProductionSmokeLaunchConfig":
        if not isinstance(plan, V5FormalProductionSearchPlan):
            raise TypeError("plan must be a V5FormalProductionSearchPlan")
        return cls(
            global_plan_payload=plan.to_payload(),
            expected_global_plan_sha256=plan.sha256,
            **paths,
        )

    def __post_init__(self) -> None:
        payload = json.loads(
            json.dumps(self.global_plan_payload, allow_nan=False)
        )
        validate_v5_formal_production_global_plan_payload(
            payload, expected_plan_sha256=self.expected_global_plan_sha256
        )
        schedules = {str(key): Path(value) for key, value in self.local_sobol_schedules.items()}
        if set(schedules) != set(V5_FORMAL_PRODUCTION_STAGE_IDS):
            raise ValueError("one local-Sobol artifact is required for every formal stage")
        object.__setattr__(self, "global_plan_payload", payload)
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(self, "split_plan", Path(self.split_plan))
        object.__setattr__(self, "sobol_design", Path(self.sobol_design))
        object.__setattr__(
            self, "calibration_artifact", Path(self.calibration_artifact)
        )
        object.__setattr__(self, "local_sobol_schedules", schedules)


CommandRunner = Callable[[Sequence[str]], V5SearchLaunchCommandResult]


class V5FormalProductionSmokeLaunchError(RuntimeError):
    def __init__(self, message: str, *, receipt_path: Path) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_global_plan_before_submit(
    path: Path, payload: Mapping[str, object]
) -> None:
    _write_json_exclusive(path, payload)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _verify_planned_source(config: V5FormalProductionSmokeLaunchConfig) -> str:
    identity = V5FormalProductionSourceIdentity.from_fingerprint(
        fingerprint_v5_frozen_search_source(config.source_root)
    )
    plan = config.global_plan_payload
    if (
        identity.audit_payload() != plan["source"]
        or identity.sha256 != plan["source_identity_sha256"]
    ):
        raise RuntimeError("source bundle changed after formal planning")
    return identity.bundle_sha256


def _groups(payload: Mapping[str, object]) -> tuple[tuple[str, str, list[dict]], ...]:
    shards = payload["shards"]
    if not isinstance(shards, list):
        raise ValueError("formal plan has no shard list")
    result = []
    for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS:
        for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
            selected = [
                dict(value)
                for value in shards
                if isinstance(value, Mapping)
                and value.get("stage_id") == stage_id
                and value.get("target_split") == split
            ]
            if not selected:
                raise ValueError("every formal stage/split array must be non-empty")
            result.append((stage_id, split, selected))
    return tuple(result)


def _export_argument(values: Mapping[str, object]) -> str:
    assignments = [
        f"{name}={safe_export_value(str(value), name)}"
        for name, value in values.items()
    ]
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *assignments))


def _array_command(
    config: V5FormalProductionSmokeLaunchConfig,
    *,
    plan_path: Path,
    logs: Path,
    stage_id: str,
    split: str,
    shards: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    expected_shas = ":".join(str(value["shard_plan_sha256"]) for value in shards)
    environment = {
        "POSTERIOR_V8_SOURCE_ROOT": config.source_root.resolve(),
        "POSTERIOR_V8_V5_FORMAL_GLOBAL_PLAN": plan_path.resolve(),
        "POSTERIOR_V8_V5_EXPECTED_GLOBAL_PLAN_SHA256": (
            config.expected_global_plan_sha256
        ),
        "POSTERIOR_V8_V5_EXPECTED_SHARD_SHA256S": expected_shas,
        "POSTERIOR_V8_V5_FORMAL_STAGE_ID": stage_id,
        "POSTERIOR_V8_V5_FORMAL_TARGET_SPLIT": split,
        "POSTERIOR_V8_V5_FORMAL_RUN_ROOT": config.run_root.resolve(),
        "POSTERIOR_V8_V5_SPLIT_PLAN": config.split_plan.resolve(),
        "POSTERIOR_V8_V5_SOBOL_DESIGN": config.sobol_design.resolve(),
        "POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE": config.local_sobol_schedules[
            stage_id
        ].resolve(),
        "POSTERIOR_V8_V5_CALIBRATION_ARTIFACT": (
            config.calibration_artifact.resolve()
        ),
    }
    wrapper = config.source_root.resolve() / V5_FORMAL_PRODUCTION_WRAPPER_RELATIVE
    label = f"{stage_id.lower()}-{split.replace('_', '-')}"
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--array=0-{len(shards) - 1}",
        f"--job-name=gisaxs-v5-formal-{label}",
        f"--output={logs}/{label}-%A_%a.out",
        f"--error={logs}/{label}-%A_%a.err",
        _export_argument(environment),
        str(wrapper),
    )


def _default_runner(argv: Sequence[str]) -> V5SearchLaunchCommandResult:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603
    return V5SearchLaunchCommandResult(
        completed.returncode, completed.stdout, completed.stderr
    )


def _job_id(result: V5SearchLaunchCommandResult, label: str) -> str:
    if not isinstance(result, V5SearchLaunchCommandResult):
        raise TypeError("command runner returned an invalid result")
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "sbatch failed"
        raise RuntimeError(f"{label} submission failed: {detail[:1000]}")
    token = result.stdout.strip().split(";", 1)[0]
    if not token.isdigit() or int(token) < 1:
        raise RuntimeError(f"{label} returned an invalid Slurm job ID")
    return token


def _control(
    runner: CommandRunner, command: Sequence[str], label: str
) -> V5SearchLaunchCommandResult:
    result = runner(command)
    if not isinstance(result, V5SearchLaunchCommandResult):
        raise TypeError("command runner returned an invalid result")
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "control failed"
        raise RuntimeError(f"{label} failed: {detail[:1000]}")
    return result


def _preflight_stage_contracts(
    config: V5FormalProductionSmokeLaunchConfig, plan_path: Path
) -> None:
    """Replay one member per stage/split without generating any curve.

    The global payload already binds every shard.  Limiting login-node replay
    to the six distinct array contracts avoids production-size work there;
    every worker still replays its own exact member before curve generation.
    """

    for stage_id, split, shards in _groups(config.global_plan_payload):
        row = shards[0]
        prepare_v5_formal_production_worker_shard(
            global_plan_path=plan_path,
            expected_global_plan_sha256=config.expected_global_plan_sha256,
            expected_shard_plan_sha256=row["shard_plan_sha256"],
            stage_id=stage_id,
            target_split=split,
            array_task_id=0,
            source_root=config.source_root,
            run_root=config.run_root,
            split_plan_path=config.split_plan,
            sobol_design_path=config.sobol_design,
            local_sobol_schedule_path=config.local_sobol_schedules[stage_id],
            calibration_path=config.calibration_artifact,
        )


def launch_v5_formal_production_contract_smoke(
    config: V5FormalProductionSmokeLaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
) -> dict[str, object]:
    """Preview or atomically release all held formal contract-smoke arrays."""

    if not isinstance(config, V5FormalProductionSmokeLaunchConfig):
        raise TypeError("config has an invalid type")
    for name in (
        "source_root",
        "split_plan",
        "sobol_design",
        "calibration_artifact",
    ):
        under_v5_launch_root(
            getattr(config, name), allowed_root, name, must_exist=True
        )
    for stage_id, path in config.local_sobol_schedules.items():
        under_v5_launch_root(
            path, allowed_root, f"local_sobol_schedules[{stage_id}]", must_exist=True
        )
    under_v5_launch_root(config.run_root, allowed_root, "run_root", must_exist=False)
    groups = _groups(config.global_plan_payload)
    plan_path = config.run_root / V5_FORMAL_PRODUCTION_PLAN_FILENAME
    logs = config.run_root / "logs"
    commands = {
        f"{stage_id}/{split}": _array_command(
            config,
            plan_path=plan_path,
            logs=logs,
            stage_id=stage_id,
            split=split,
            shards=shards,
        )
        for stage_id, split, shards in groups
    }
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "training_submitted": False,
            "global_plan_sha256": config.expected_global_plan_sha256,
            "submission_preview": {
                key: list(value) for key, value in commands.items()
            },
        }
    host = socket.gethostname() if hostname is None else hostname
    if not host.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("submit mode is restricted to a max-wgs login node")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal launcher cannot run inside a Slurm worker")
    config.run_root.mkdir(parents=True, exist_ok=False)
    _write_global_plan_before_submit(plan_path, config.global_plan_payload)
    logs.mkdir(exist_ok=False)
    receipt_path = config.run_root / V5_FORMAL_PRODUCTION_LAUNCH_RECEIPT_FILENAME
    command_runner = _default_runner if runner is None else runner
    job_ids: dict[str, str] = {}
    attempts: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    status = "failed"
    released = False
    failure: dict[str, object] | None = None
    stage = "preflight"
    try:
        _preflight_stage_contracts(config, plan_path)
        for label, command in commands.items():
            stage = f"submit:{label}"
            _verify_planned_source(config)
            result = command_runner(command)
            attempts.append(
                {
                    "label": label,
                    "argv": list(command),
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip()[:1000],
                    "stderr": result.stderr.strip()[:1000],
                }
            )
            job_ids[label] = _job_id(result, label)
        stage = "release"
        _verify_planned_source(config)
        release_command = ("scontrol", "release", *job_ids.values())
        release_result = _control(command_runner, release_command, stage)
        controls.append(
            {"label": stage, "argv": list(release_command), "returncode": release_result.returncode}
        )
        released = True
        status = "submitted"
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        if job_ids:
            cancel = ("scancel", *job_ids.values())
            try:
                result = _control(command_runner, cancel, "cancel held arrays")
                controls.append(
                    {"label": "cancel", "argv": list(cancel), "returncode": result.returncode}
                )
            except Exception as cancel_exc:
                failure["cancellation_failure"] = str(cancel_exc)[:1000]
    receipt_core = {
        "schema": "gisaxs.posterior_v8.formal_production_contract_smoke_launch/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "global_plan_path": str(plan_path),
        "global_plan_sha256": config.expected_global_plan_sha256,
        "job_ids": job_ids,
        "all_arrays_submitted_held": len(job_ids) == len(groups),
        "arrays_released_together": released,
        "submission_attempts": attempts,
        "control_attempts": controls,
        "failure": failure,
        "contract_smoke_only": True,
        "training_submitted": False,
        "heavy_compute_performed_on_login_node": False,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": sha256(
            canonical_json(receipt_core).encode("utf-8")
        ).hexdigest(),
    }
    _write_json_exclusive(receipt_path, receipt)
    if status != "submitted":
        raise V5FormalProductionSmokeLaunchError(
            f"formal contract-smoke launch failed during {stage}",
            receipt_path=receipt_path,
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-plan-source", type=Path, required=True)
    parser.add_argument("--expected-global-plan-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--sobol-design", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS:
        parser.add_argument(
            f"--{stage_id.lower()}-local-sobol-schedule", type=Path, required=True
        )
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.global_plan_source.read_text(encoding="utf-8"))
    config = V5FormalProductionSmokeLaunchConfig(
        global_plan_payload=payload,
        expected_global_plan_sha256=args.expected_global_plan_sha256,
        source_root=args.source_root,
        run_root=args.run_root,
        split_plan=args.split_plan,
        sobol_design=args.sobol_design,
        calibration_artifact=args.calibration_artifact,
        local_sobol_schedules={
            stage_id: getattr(args, f"{stage_id.lower()}_local_sobol_schedule")
            for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
        },
    )
    result = launch_v5_formal_production_contract_smoke(
        config, submit=args.submit
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5FormalProductionSmokeLaunchConfig",
    "V5FormalProductionSmokeLaunchError",
    "V5_FORMAL_PRODUCTION_LAUNCH_RECEIPT_FILENAME",
    "V5_FORMAL_PRODUCTION_PLAN_FILENAME",
    "launch_v5_formal_production_contract_smoke",
    "main",
]
