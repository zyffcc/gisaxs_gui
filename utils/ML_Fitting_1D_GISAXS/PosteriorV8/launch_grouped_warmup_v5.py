"""Audit and submit an exclusive Maxwell V5.1 grouped warmup pilot.

Dry-run is the default.  Explicit submit mode performs only three
``sbatch --parsable`` calls on max-wgs: independent train and validation data
arrays followed by an ``afterok`` GPU warmup.  It never computes curves or
trains on the login node.
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

from .grouped_warmup_launch_plan_v5 import (
    DATASET_WRAPPER_RELATIVE,
    LAUNCH_MANIFEST_FILENAME,
    MAXWELL_DUST_ROOT,
    TRAIN_WRAPPER_RELATIVE,
    V5WarmupLaunchConfig,
    build_v5_warmup_launch_plan,
    canonical_json,
    replay_v5_warmup_fingerprints,
    safe_export_value,
)


@dataclass(frozen=True)
class V5CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], V5CommandResult]
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")


class V5WarmupLaunchError(RuntimeError):
    """A submit-mode launch failed after its exclusive run root was reserved."""

    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


def _export_argument(values: Mapping[str, object]) -> str:
    assignments = [
        f"{key}={safe_export_value(str(value), key)}" for key, value in values.items()
    ]
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *assignments))


def _dataset_command(plan: Mapping[str, object], split: str) -> tuple[str, ...]:
    layout = plan["layout"]
    arrays = plan["arrays"]
    config = plan["configuration"]
    source = plan["source"]
    contracts = plan["contracts"]
    label = "train" if split == "train" else "validation"
    output_key = "train_data" if split == "train" else "validation_data"
    total_key = "train_recipes" if split == "train" else "validation_recipes"
    environment = {
        "POSTERIOR_V8_SOURCE_ROOT": source["source_root"],
        "POSTERIOR_V8_V5_SPLIT_PLAN": contracts["split_plan"]["path"],
        "POSTERIOR_V8_V5_SOBOL_DESIGN": contracts["sobol_design"]["path"],
        "POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR": layout[output_key],
        "POSTERIOR_V8_V5_TARGET_SPLIT": split,
        "POSTERIOR_V8_V5_RECIPES_PER_SHARD": config["recipes_per_shard"],
        "POSTERIOR_V8_V5_TOTAL_RECIPES": config[total_key],
    }
    wrapper = Path(str(source["source_root"])) / DATASET_WRAPPER_RELATIVE
    array_spec = arrays[split]["array_spec"]
    return (
        "sbatch",
        "--parsable",
        f"--array={array_spec}",
        f"--job-name=gisaxs-v5-{label}-data",
        f"--output={layout['logs']}/{label}-dataset-%A_%a.out",
        f"--error={layout['logs']}/{label}-dataset-%A_%a.err",
        _export_argument(environment),
        str(wrapper),
    )


def _training_command(
    plan: Mapping[str, object],
    train_job_id: str,
    validation_job_id: str,
) -> tuple[str, ...]:
    layout = plan["layout"]
    config = plan["configuration"]
    source = plan["source"]
    arrays = plan["arrays"]
    train_paths = ":".join(item["output"] for item in arrays["train"]["windows"])
    validation_paths = ":".join(
        item["output"] for item in arrays["tuning_validation"]["windows"]
    )
    environment = {
        "POSTERIOR_V8_SOURCE_ROOT": source["source_root"],
        "POSTERIOR_V8_V5_TRAIN_DATASETS": train_paths,
        "POSTERIOR_V8_V5_VALIDATION_DATASETS": validation_paths,
        "POSTERIOR_V8_V5_GROUPED_OUTPUT": layout["model"],
        "POSTERIOR_V8_WARMUP_EPOCHS": config["warmup_epochs"],
        "POSTERIOR_V8_FULL_EPOCHS": 0,
        "POSTERIOR_V8_SEED": config["seed"],
    }
    wrapper = Path(str(source["source_root"])) / TRAIN_WRAPPER_RELATIVE
    return (
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{train_job_id}:{validation_job_id}",
        "--job-name=gisaxs-v5-warmup",
        f"--output={layout['logs']}/warmup-%j.out",
        f"--error={layout['logs']}/warmup-%j.err",
        _export_argument(environment),
        str(wrapper),
    )


def _default_runner(argv: Sequence[str]) -> V5CommandResult:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603
    return V5CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _job_id(result: V5CommandResult, stage: str) -> str:
    if not isinstance(result, V5CommandResult):
        raise TypeError("command runner must return V5CommandResult")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "sbatch failed without output"
        raise RuntimeError(f"{stage} submission failed ({result.returncode}): {detail[:1000]}")
    token = result.stdout.strip().split(";", 1)[0]
    if not token.isdigit() or int(token) < 1:
        raise RuntimeError(f"{stage} returned an invalid parsable Slurm job id")
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
    result = runner(command)
    if not isinstance(result, V5CommandResult):
        raise TypeError("command runner must return V5CommandResult")
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


def launch_v5_grouped_warmup(
    config: V5WarmupLaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
) -> dict[str, object]:
    """Dry-run by default; explicitly submit and persist one immutable audit."""

    plan = build_v5_warmup_launch_plan(config, allowed_root=allowed_root)
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "plan": plan,
            "submission_preview": [
                list(_dataset_command(plan, "train")),
                list(_dataset_command(plan, "tuning_validation")),
                list(_training_command(plan, "TRAIN_JOB_ID", "VALIDATION_JOB_ID")),
            ],
        }
    host = socket.gethostname() if hostname is None else hostname
    if not host.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("submit mode is restricted to a max-wgs login node")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the launcher must run on max-wgs, not inside a Slurm allocation")

    run_root = Path(str(plan["run_root"]))
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "logs").mkdir(exist_ok=False)
    manifest_path = run_root / LAUNCH_MANIFEST_FILENAME
    command_runner = _default_runner if runner is None else runner
    attempts: list[dict[str, object]] = []
    job_ids: dict[str, str] = {}
    stage = "train_dataset"
    status = "failed"
    failure: dict[str, str] | None = None
    try:
        for stage, split in (
            ("train_dataset", "train"),
            ("validation_dataset", "tuning_validation"),
        ):
            replay = replay_v5_warmup_fingerprints(config, allowed_root)
            if replay != {"source": plan["source"], "contracts": plan["contracts"]}:
                raise RuntimeError("source or frozen contracts changed during launch")
            job_ids[stage] = _submit_one(
                stage, _dataset_command(plan, split), command_runner, attempts
            )

        stage = "gpu_warmup"
        replay = replay_v5_warmup_fingerprints(config, allowed_root)
        if replay != {"source": plan["source"], "contracts": plan["contracts"]}:
            raise RuntimeError("source or frozen contracts changed during launch")
        command = _training_command(
            plan,
            job_ids["train_dataset"],
            job_ids["validation_dataset"],
        )
        job_ids[stage] = _submit_one(stage, command, command_runner, attempts)
        status = "submitted"
    except (Exception, KeyboardInterrupt) as exc:  # preserve partial audit; never cancel jobs
        failure = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:2000]}

    manifest: dict[str, object] = {
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
        "job_ids": job_ids,
        "failure": failure,
        "heavy_compute_performed_on_login_node": False,
        "cancellation_attempted": False,
        "manifest_write_policy": "single exclusive create; never overwrite",
    }
    manifest["manifest_sha256"] = sha256(canonical_json(manifest).encode()).hexdigest()
    _write_manifest(manifest_path, manifest)
    if status != "submitted":
        raise V5WarmupLaunchError(
            f"V5.1 warmup launch failed during {failure['stage']}; audit preserved",
            manifest_path=manifest_path,
        )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--sobol-design", required=True, type=Path)
    parser.add_argument("--train-recipes", required=True, type=int)
    parser.add_argument("--validation-recipes", required=True, type=int)
    parser.add_argument("--recipes-per-shard", required=True, type=int)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit two dataset arrays and their afterok GPU warmup dependency.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = launch_v5_grouped_warmup(
        V5WarmupLaunchConfig(
            source_root=args.source_root,
            run_root=args.run_root,
            split_plan=args.split_plan,
            sobol_design=args.sobol_design,
            train_recipes=args.train_recipes,
            validation_recipes=args.validation_recipes,
            recipes_per_shard=args.recipes_per_shard,
            warmup_epochs=args.warmup_epochs,
            seed=args.seed,
        ),
        submit=args.submit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LAUNCH_MANIFEST_FILENAME",
    "MAXWELL_DUST_ROOT",
    "V5CommandResult",
    "V5WarmupLaunchConfig",
    "V5WarmupLaunchError",
    "build_v5_warmup_launch_plan",
    "launch_v5_grouped_warmup",
    "main",
]
