"""Dry-run or submit Maxwell CPU arrays for V5.1 frozen search labels.

Submit mode is restricted to ``max-wgs`` and performs only lightweight
fingerprint checks, directory reservation, and held Slurm submissions.  Both
arrays are released together only after both held submissions succeed; any
failure cancels every submitted job.  It deliberately does not submit
training, and its throughput-pilot outputs are not eligible for the full
trainer without a separately versioned promotion gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Callable, Mapping, Sequence

from .frozen_search_launch_plan_v5 import (
    FROZEN_SEARCH_WRAPPER_RELATIVE,
    LAUNCH_MANIFEST_FILENAME,
    MAXWELL_DUST_ROOT,
    V5FrozenSearchLaunchConfig,
    build_v5_frozen_search_launch_plan,
    replay_v5_frozen_search_launch_fingerprints,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_warmup_launch_plan_v5 import safe_export_value
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)


@dataclass(frozen=True)
class V5SearchLaunchCommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], V5SearchLaunchCommandResult]
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")


class V5FrozenSearchLaunchError(RuntimeError):
    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


def _export_argument(values: Mapping[str, object]) -> str:
    assignments = [
        f"{name}={safe_export_value(str(value), name)}"
        for name, value in values.items()
    ]
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *assignments))


def _array_command(plan: Mapping[str, object], split: str) -> tuple[str, ...]:
    contracts = plan["contracts"]
    config = plan["configuration"]
    arrays = plan["arrays"]
    layout = plan["layout"]
    source = plan["source"]
    protocol = contracts["protocol"]
    optimizer = contracts["optimizer_schedule"]
    split_start_key = "train_start" if split == "train" else "validation_start"
    total_key = "train_recipes" if split == "train" else "validation_recipes"
    output_key = "train_shards" if split == "train" else "validation_shards"
    environment = {
        "POSTERIOR_V8_SOURCE_ROOT": source["source_root"],
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_BUNDLE_SHA256": source["bundle_sha256"],
        "POSTERIOR_V8_V5_LAUNCH_PLAN_SHA256": plan["plan_sha256"],
        "POSTERIOR_V8_V5_SPLIT_PLAN": contracts["split_plan"]["path"],
        "POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_SHA256": contracts["split_plan"][
            "contract_sha256"
        ],
        "POSTERIOR_V8_V5_EXPECTED_SPLIT_PLAN_FILE_SHA256": contracts[
            "split_plan"
        ]["file_sha256"],
        "POSTERIOR_V8_V5_SOBOL_DESIGN": contracts["sobol_design"]["path"],
        "POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_SHA256": contracts[
            "sobol_design"
        ]["contract_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_SOBOL_DESIGN_FILE_SHA256": contracts[
            "sobol_design"
        ]["file_sha256"],
        "POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE": contracts[
            "local_sobol_schedule"
        ]["path"],
        "POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_SHA256": contracts[
            "local_sobol_schedule"
        ]["schedule_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_ARTIFACT_SHA256": contracts[
            "local_sobol_schedule"
        ]["artifact_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_LOCAL_SCHEDULE_MANIFEST_SHA256": contracts[
            "local_sobol_schedule"
        ]["manifest_sha256"],
        "POSTERIOR_V8_V5_SEARCH_OUTPUT_DIR": layout[output_key],
        "POSTERIOR_V8_V5_TARGET_SPLIT": split,
        "POSTERIOR_V8_V5_SPLIT_START": config[split_start_key],
        "POSTERIOR_V8_V5_RECIPES_PER_SHARD": config["recipes_per_shard"],
        "POSTERIOR_V8_V5_TOTAL_RECIPES": config[total_key],
        "POSTERIOR_V8_V5_VIEW_INDICES": ":".join(
            str(value) for value in config["view_indices"]
        ),
        "POSTERIOR_V8_V5_TOPOLOGY_SCHEDULE_ID": contracts[
            "topology_schedule"
        ]["schedule_id"],
        "POSTERIOR_V8_V5_SELECTED_TOPOLOGY_IDS": ":".join(
            str(value) for value in config["selected_topology_ids"]
        ),
        "POSTERIOR_V8_V5_OPTIMIZER_SCHEDULE_ID": optimizer["schedule_id"],
        "POSTERIOR_V8_V5_DIRECT_SCOUT_SEEDS": optimizer[
            "direct_scout_seed_count"
        ],
        "POSTERIOR_V8_V5_PER_SEED_FORWARD_LIMIT": optimizer[
            "per_seed_forward_evaluation_limit"
        ],
        "POSTERIOR_V8_V5_FTOL": optimizer["ftol"],
        "POSTERIOR_V8_V5_XTOL": optimizer["xtol"],
        "POSTERIOR_V8_V5_GTOL": optimizer["gtol"],
        "POSTERIOR_V8_V5_PROTOCOL_ID": protocol["protocol_id"],
        "POSTERIOR_V8_V5_PROTOCOL_TIER": protocol["protocol_tier"],
    }
    if protocol["protocol_tier"] == V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT:
        environment.update(
            {
                "POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_NAME": protocol[
                    "threshold_name"
                ],
                "POSTERIOR_V8_V5_STANDARDIZED_THRESHOLD_VALUE": protocol[
                    "threshold_value"
                ],
                "POSTERIOR_V8_V5_RAW_THRESHOLD_NAME": protocol[
                    "missing_acceptance_sigma_threshold_name"
                ],
                "POSTERIOR_V8_V5_RAW_THRESHOLD_VALUE": protocol[
                    "missing_acceptance_sigma_threshold_value"
                ],
                "POSTERIOR_V8_V5_THRESHOLD_SOURCE_ID": protocol[
                    "threshold_source_id"
                ],
            }
        )
    elif protocol["protocol_tier"] == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED:
        calibration = contracts["compatibility_calibration"]
        if calibration is None:  # pragma: no cover - launch-plan invariant
            raise RuntimeError("formal protocol lost its calibration artifact")
        environment.update(
            {
                "POSTERIOR_V8_V5_CALIBRATION_ARTIFACT": calibration["path"],
                "POSTERIOR_V8_V5_EXPECTED_CALIBRATION_SHA256": calibration[
                    "identity"
                ]["artifact_sha256"],
                "POSTERIOR_V8_V5_EXPECTED_CALIBRATION_FILE_SHA256": calibration[
                    "identity"
                ]["file_sha256"],
            }
        )
    else:  # pragma: no cover - checked protocol contract
        raise RuntimeError("unsupported frozen-search protocol tier")
    wrapper = Path(str(source["source_root"])) / FROZEN_SEARCH_WRAPPER_RELATIVE
    label = "train" if split == "train" else "validation"
    return (
        "sbatch",
        "--parsable",
        "--hold",
        f"--array={arrays[split]['array_spec']}",
        f"--job-name=gisaxs-v5-{label}-search",
        f"--output={layout['logs']}/{label}-search-%A_%a.out",
        f"--error={layout['logs']}/{label}-search-%A_%a.err",
        _export_argument(environment),
        str(wrapper),
    )


def _default_runner(argv: Sequence[str]) -> V5SearchLaunchCommandResult:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603
    return V5SearchLaunchCommandResult(
        completed.returncode, completed.stdout, completed.stderr
    )


def _job_id(result: V5SearchLaunchCommandResult, stage: str) -> str:
    if not isinstance(result, V5SearchLaunchCommandResult):
        raise TypeError("command runner must return V5SearchLaunchCommandResult")
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "sbatch failed"
        raise RuntimeError(f"{stage} submission failed: {detail[:1000]}")
    token = result.stdout.strip().split(";", 1)[0]
    if not token.isdigit() or int(token) < 1:
        raise RuntimeError(f"{stage} returned an invalid parsable Slurm job ID")
    return token


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _submit_one(
    stage: str,
    command: Sequence[str],
    runner: CommandRunner,
    attempts: list[dict[str, object]],
) -> str:
    try:
        result = runner(command)
    except (Exception, KeyboardInterrupt) as exc:
        attempts.append(
            {
                "stage": stage,
                "argv": list(command),
                "runner_exception_type": type(exc).__name__,
                "runner_exception_message": str(exc)[:1000],
            }
        )
        raise
    if not isinstance(result, V5SearchLaunchCommandResult):
        raise TypeError("command runner must return V5SearchLaunchCommandResult")
    attempts.append(
        {
            "stage": stage,
            "argv": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:1000],
            "stderr": result.stderr.strip()[:1000],
        }
    )
    return _job_id(result, stage)


def _run_control(
    stage: str,
    command: Sequence[str],
    runner: CommandRunner,
    controls: list[dict[str, object]],
) -> None:
    try:
        result = runner(command)
    except (Exception, KeyboardInterrupt) as exc:
        controls.append(
            {
                "stage": stage,
                "argv": list(command),
                "runner_exception_type": type(exc).__name__,
                "runner_exception_message": str(exc)[:1000],
            }
        )
        raise
    if not isinstance(result, V5SearchLaunchCommandResult):
        raise TypeError("command runner must return V5SearchLaunchCommandResult")
    controls.append(
        {
            "stage": stage,
            "argv": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:1000],
            "stderr": result.stderr.strip()[:1000],
        }
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "control command failed"
        raise RuntimeError(f"{stage} failed: {detail[:1000]}")


def launch_v5_frozen_search_labels(
    config: V5FrozenSearchLaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
) -> dict[str, object]:
    """Preview by default; atomically hold/release train and validation arrays."""

    plan = build_v5_frozen_search_launch_plan(config, allowed_root=allowed_root)
    commands = {
        split: _array_command(plan, split)
        for split in ("train", "tuning_validation")
    }
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "plan": plan,
            "submission_preview": {
                split: list(command) for split, command in commands.items()
            },
        }

    host = socket.gethostname() if hostname is None else hostname
    if not host.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("submit mode is restricted to a max-wgs login node")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("launcher must run on max-wgs, not inside Slurm")
    run_root = Path(str(plan["run_root"]))
    run_root.mkdir(parents=True, exist_ok=False)
    Path(str(plan["layout"]["logs"])).mkdir(exist_ok=False)
    Path(str(plan["layout"]["train_shards"])).mkdir(parents=True, exist_ok=False)
    Path(str(plan["layout"]["validation_shards"])).mkdir(
        parents=True, exist_ok=False
    )
    manifest_path = run_root / LAUNCH_MANIFEST_FILENAME
    command_runner = _default_runner if runner is None else runner
    attempts: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    job_ids: dict[str, str] = {}
    failure = None
    stage = "train"
    status = "failed"
    arrays_released = False
    cancellation_attempted = False
    expected_fingerprints = {
        "source": plan["source"],
        "contracts": plan["contracts"],
    }
    try:
        for stage, split in (
            ("train", "train"),
            ("tuning_validation", "tuning_validation"),
        ):
            observed = replay_v5_frozen_search_launch_fingerprints(
                config, allowed_root=allowed_root
            )
            if observed != expected_fingerprints:
                raise RuntimeError("source or frozen contracts changed during launch")
            job_ids[stage] = _submit_one(
                stage, commands[split], command_runner, attempts
            )
        stage = "pre_release_fingerprint"
        observed = replay_v5_frozen_search_launch_fingerprints(
            config, allowed_root=allowed_root
        )
        if observed != expected_fingerprints:
            raise RuntimeError("source or frozen contracts changed during launch")
        stage = "release_held_arrays"
        _run_control(
            stage,
            ("scontrol", "release", job_ids["train"], job_ids["tuning_validation"]),
            command_runner,
            controls,
        )
        arrays_released = True
        status = "submitted"
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        if job_ids:
            cancellation_attempted = True
            cancel_command = ("scancel", *job_ids.values())
            try:
                _run_control(
                    "cancel_submitted_held_arrays",
                    cancel_command,
                    command_runner,
                    controls,
                )
            except Exception as cancel_exc:  # preserve both failure boundaries
                failure["cancellation_failure_type"] = type(cancel_exc).__name__
                failure["cancellation_failure_message"] = str(cancel_exc)[:2000]

    handoff = dict(plan["afterok_handoff"])
    if set(job_ids) == {"train", "tuning_validation"}:
        handoff["dependency"] = (
            f"afterok:{job_ids['train']}:{job_ids['tuning_validation']}"
        )
    manifest_core = {
        "schema": plan["schema"],
        "version": plan["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "plan": plan,
        "execution_environment": {
            "hostname": host,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "submission_attempts": attempts,
        "control_attempts": controls,
        "job_ids": job_ids,
        "arrays_submitted_held": set(job_ids) == {"train", "tuning_validation"},
        "submitted_held_job_ids": list(job_ids.values()),
        "arrays_released": arrays_released,
        "afterok_handoff": handoff,
        "failure": failure,
        "heavy_compute_performed_on_login_node": False,
        "downstream_training_submitted": False,
        "cancellation_attempted": cancellation_attempted,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": sha256(
            canonical_json(manifest_core).encode("utf-8")
        ).hexdigest(),
    }
    _write_manifest(manifest_path, manifest)
    if status != "submitted":
        raise V5FrozenSearchLaunchError(
            f"frozen-search launch failed during {failure['stage']}; audit preserved",
            manifest_path=manifest_path,
        )
    return manifest


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.replace(":", ",").split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--sobol-design", required=True, type=Path)
    parser.add_argument("--local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-recipes", required=True, type=int)
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--validation-recipes", required=True, type=int)
    parser.add_argument("--recipes-per-shard", required=True, type=int)
    parser.add_argument("--view-indices", type=_integer_tuple, default=(0,))
    parser.add_argument("--topology-schedule-id", required=True)
    parser.add_argument("--selected-topology-ids", required=True, type=_integer_tuple)
    parser.add_argument("--optimizer-schedule-id", required=True)
    parser.add_argument("--direct-scout-seed-count", required=True, type=int)
    parser.add_argument("--per-seed-forward-evaluation-limit", required=True, type=int)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--xtol", type=float, default=1.0e-8)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--standardized-threshold-name")
    parser.add_argument("--standardized-threshold-value", type=float)
    parser.add_argument("--raw-threshold-name")
    parser.add_argument("--raw-threshold-value", type=float)
    parser.add_argument("--threshold-source-id")
    parser.add_argument("--compatibility-calibration", type=Path)
    parser.add_argument("--pilot-throughput-source-id", required=True)
    parser.add_argument(
        "--pilot-effective-seconds-per-exact-forward-call",
        required=True,
        type=float,
    )
    parser.add_argument("--runtime-safety-factor", type=float, default=10.0)
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = V5FrozenSearchLaunchConfig(
        **{
            name: value
            for name, value in vars(args).items()
            if name != "submit"
        }
    )
    result = launch_v5_frozen_search_labels(config, submit=args.submit)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5FrozenSearchLaunchError",
    "V5SearchLaunchCommandResult",
    "launch_v5_frozen_search_labels",
    "main",
]
