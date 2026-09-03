"""Dry-run or submit the Maxwell balanced-K1 training seed array.

Submit mode is restricted to max-wgs and performs only immutable plan
publication plus ``sbatch`` calls.  Dataset inspection and training execute in
Slurm workers.  Dry-run is the default.
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

from .k1_training_chain_contract_v5 import canonical_json
from .k1_training_chain_plan_v5 import (
    K1_TRAINING_LAUNCH_RECEIPT_FILENAME,
    MAXWELL_DUST_ROOT,
    V5K1TrainingChainConfig,
    build_v5_k1_training_chain_plan,
    replay_v5_k1_training_chain_fingerprints,
)


@dataclass(frozen=True)
class V5CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], V5CommandResult]
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")


class V5K1TrainingLaunchError(RuntimeError):
    def __init__(self, message: str, *, receipt_path: Path) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path


def _safe_export(value: object, name: str) -> str:
    result = str(value)
    if any(character in result for character in (",", "\n", "\r", "\0")):
        raise ValueError(f"{name} cannot enter the Slurm export contract")
    return result


def _export_argument(values: Mapping[str, object]) -> str:
    assignments = [f"{name}={_safe_export(value, name)}" for name, value in values.items()]
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *assignments))


def _training_command(plan: Mapping[str, object]) -> tuple[str, ...]:
    return (
        "sbatch",
        "--parsable",
        f"--array={plan['slurm']['training_array_spec']}",
        "--job-name=gisaxs-v5-k1-train",
        f"--output={plan['layout']['logs']}/k1-train-%A_%a.out",
        f"--error={plan['layout']['logs']}/k1-train-%A_%a.err",
        _export_argument(
            {
                "POSTERIOR_V8_SOURCE_ROOT": plan["source"]["root"],
                "POSTERIOR_V8_K1_TRAINING_PLAN": plan["layout"]["plan"],
                "POSTERIOR_V8_K1_TRAINING_PLAN_SHA256": plan["plan_sha256"],
            }
        ),
        plan["slurm"]["training_wrapper"],
    )


def _collection_command(plan: Mapping[str, object], training_job_id: str) -> tuple[str, ...]:
    return (
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{training_job_id}",
        "--job-name=gisaxs-v5-k1-tuning-handoff",
        f"--output={plan['layout']['logs']}/k1-tuning-handoff-%j.out",
        f"--error={plan['layout']['logs']}/k1-tuning-handoff-%j.err",
        _export_argument(
            {
                "POSTERIOR_V8_SOURCE_ROOT": plan["source"]["root"],
                "POSTERIOR_V8_K1_TRAINING_PLAN": plan["layout"]["plan"],
                "POSTERIOR_V8_K1_TRAINING_PLAN_SHA256": plan["plan_sha256"],
            }
        ),
        plan["slurm"]["collection_wrapper"],
    )


def _default_runner(argv: Sequence[str]) -> V5CommandResult:
    result = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603
    return V5CommandResult(result.returncode, result.stdout, result.stderr)


def _job_id(result: V5CommandResult, stage: str) -> str:
    if not isinstance(result, V5CommandResult):
        raise TypeError("command runner must return V5CommandResult")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "sbatch failed"
        raise RuntimeError(f"{stage} submission failed: {detail[:1000]}")
    token = result.stdout.strip().split(";", 1)[0]
    if not token.isdigit() or int(token) < 1:
        raise RuntimeError(f"{stage} returned an invalid parsable Slurm job ID")
    return token


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def launch_v5_k1_training_chain(
    config: V5K1TrainingChainConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a dry-run preview or submit one GPU array and its CPU handoff."""

    if type(submit) is not bool:
        raise TypeError("submit must be a bool")
    plan = build_v5_k1_training_chain_plan(config, allowed_root=allowed_root)
    preview = {
        "training": list(_training_command(plan)),
        "tuning_handoff": list(_collection_command(plan, "TRAINING_ARRAY_JOB_ID")),
    }
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "submissions_performed": False,
            "plan": plan,
            "submission_preview": preview,
        }
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    if not host.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("submit mode is restricted to a max-wgs login node")
    if env.get("SLURM_JOB_ID"):
        raise RuntimeError("launcher must run on max-wgs, not inside a Slurm allocation")
    if plan["execution_gate"]["submission_allowed"] is not True:
        blockers = "; ".join(plan["execution_gate"]["blockers"])
        raise RuntimeError(f"K1 training chain is fail-closed: {blockers}")

    run_root = Path(plan["layout"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "audit", "models"):
        Path(plan["layout"][name]).mkdir(exist_ok=False)
    plan_path = Path(plan["layout"]["plan"])
    _write_json_exclusive(plan_path, plan)
    command_runner = _default_runner if runner is None else runner
    attempts = []
    job_ids = {}
    status = "failed"
    stage = "training_seed_array"
    failure = None
    try:
        replay_v5_k1_training_chain_fingerprints(plan, allowed_root=allowed_root)
        command = _training_command(plan)
        result = command_runner(command)
        attempts.append(
            {
                "stage": stage,
                "argv": list(command),
                "returncode": result.returncode,
                "stdout": result.stdout.strip()[:1000],
                "stderr": result.stderr.strip()[:1000],
            }
        )
        job_ids[stage] = _job_id(result, stage)
        stage = "tuning_handoff"
        replay_v5_k1_training_chain_fingerprints(plan, allowed_root=allowed_root)
        command = _collection_command(plan, job_ids["training_seed_array"])
        result = command_runner(command)
        attempts.append(
            {
                "stage": stage,
                "argv": list(command),
                "returncode": result.returncode,
                "stdout": result.stdout.strip()[:1000],
                "stderr": result.stderr.strip()[:1000],
            }
        )
        job_ids[stage] = _job_id(result, stage)
        status = "submitted"
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:2000]}

    core = {
        "schema": plan["schema"],
        "version": plan["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "plan_sha256": plan["plan_sha256"],
        "plan_path": str(plan_path),
        "job_ids": job_ids,
        "submission_attempts": attempts,
        "failure": failure,
        "execution_environment": {
            "hostname": host,
            "python": platform.python_version(),
        },
        "heavy_compute_performed_on_login_node": False,
        "cancellation_attempted": False,
        "paper_model_eligible": False,
        "k1_phase_c_passed": False,
    }
    receipt = {
        **core,
        "receipt_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    receipt_path = Path(plan["layout"]["launch_receipt"])
    if receipt_path.name != K1_TRAINING_LAUNCH_RECEIPT_FILENAME:
        raise RuntimeError("launch receipt filename escaped the contract")
    _write_json_exclusive(receipt_path, receipt)
    if status != "submitted":
        raise V5K1TrainingLaunchError(
            f"K1 training launch failed during {stage}; audit preserved",
            receipt_path=receipt_path,
        )
    return receipt


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one model seed is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--input-inventory", required=True, type=Path)
    parser.add_argument("--input-inventory-file-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("engineering_e1", "formal_multiseed"), required=True)
    parser.add_argument("--model-seeds", required=True, type=_seeds)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--full-epochs", type=int, default=0)
    parser.add_argument("--recipes-per-replica", type=int, default=4)
    parser.add_argument("--validation-recipes-per-batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--encoder-blocks", type=int, default=6)
    parser.add_argument("--mixture-components", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = launch_v5_k1_training_chain(
        V5K1TrainingChainConfig(
            source_root=args.source_root,
            source_archive=args.source_archive,
            expected_source_archive_sha256=args.source_archive_sha256,
            input_inventory=args.input_inventory,
            expected_inventory_file_sha256=args.input_inventory_file_sha256,
            run_root=args.run_root,
            mode=args.mode,
            model_seeds=args.model_seeds,
            warmup_epochs=args.warmup_epochs,
            full_epochs=args.full_epochs,
            recipes_per_replica=args.recipes_per_replica,
            validation_recipes_per_batch=args.validation_recipes_per_batch,
            width=args.width,
            encoder_blocks=args.encoder_blocks,
            mixture_components=args.mixture_components,
            learning_rate=args.learning_rate,
            mixed_precision=not args.no_mixed_precision,
        ),
        submit=args.submit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5CommandResult",
    "V5K1TrainingLaunchError",
    "launch_v5_k1_training_chain",
    "main",
]
