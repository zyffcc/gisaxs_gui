"""Author one immutable all-K1 full-search plan on a Slurm worker.

Plan authoring replays every balanced dataset publication and therefore is
not permitted on a Maxwell submission host.  The resulting candidate is
written outside the future search root; the launch transaction later creates
that root and publishes the same validated plan there with ``O_EXCL``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .calibrated_search_threshold_v5 import (
    V5CheckedCompatibilityCalibration,
    read_v5_checked_compatibility_calibration,
)
from .exact_search_executor_v5 import build_v5_frozen_exact_search_protocol
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
    read_v5_frozen_local_sobol_schedule,
)
from .formal_production_search_contract_v5 import (
    V5FormalProductionSearchStage,
    topology_ids_for_v5_formal_production_stage,
)
from .frozen_search_pipeline_contract_v5 import V5SelectedTopologySearchSchedule
from .grouped_artifact_v5 import canonical_json
from .k1_balanced_full_search_plan_runtime_v5 import (
    V5K1BalancedFullSearchPlanConfig,
    build_v5_k1_balanced_full_search_plan_from_publications,
)
from .k1_balanced_full_search_plan_v5 import (
    MAXWELL_DUST_ROOT,
    validate_v5_k1_balanced_full_search_plan,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks
from .k1_training_chain_plan_v5 import fingerprint_v5_k1_training_source
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)


V5_K1_BALANCED_FULL_SEARCH_PLAN_AUTHORING_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_plan_authoring/v1"
)
V5_K1_BALANCED_FULL_SEARCH_PLAN_AUTHORING_VERSION = (
    "posterior_v8_worker_only_historical_inventory_noise_preflight_completion_last_v6"
)
V5_K1_BALANCED_FULL_SEARCH_DIRECT_SCOUT_COUNT = 16
V5_K1_BALANCED_FULL_SEARCH_PER_SEED_FORWARD_LIMIT = 256
V5_K1_BALANCED_FULL_SEARCH_EXACT_BUDGET = 4096


@dataclass(frozen=True, kw_only=True)
class V5K1BalancedFullSearchAuthoringConfig:
    source_root: Path
    source_archive: Path
    expected_source_archive_sha256: str
    balanced_dataset_launch_plan_path: Path
    balanced_dataset_launch_plan_file_sha256: str
    balanced_dataset_launch_plan_sha256: str
    balanced_dataset_completion_path: Path
    balanced_dataset_completion_file_sha256: str
    balanced_dataset_completion_sha256: str
    train_tuning_receipt_path: Path
    train_tuning_receipt_file_sha256: str
    train_tuning_receipt_sha256: str
    train_tuning_claim_sha256: str
    phase_c_completion_path: Path
    phase_c_completion_file_sha256: str
    phase_c_completion_sha256: str
    three_way_receipt_path: Path
    three_way_receipt_file_sha256: str
    three_way_receipt_sha256: str
    phase_c_exclusion_claim_sha256: str
    local_sobol_schedule_path: Path
    calibration_path: Path
    expected_calibration_artifact_sha256: str
    expected_calibration_file_sha256: str
    search_run_root: Path
    candidate_plan_path: Path
    completion_path: Path


def _under_root(path: Path, allowed_root: Path, name: str, *, exists: bool) -> Path:
    selected = lexical_no_symlinks(path, name)
    root = Path(allowed_root).resolve(strict=True)
    resolved = selected.resolve(strict=exists)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{name} must remain below {root}")
    return resolved


def _sealed_file_identity(path: Path, allowed_root: Path, name: str) -> dict[str, object]:
    selected = _under_root(path, allowed_root, name, exists=True)
    metadata = selected.stat()
    if (
        not selected.is_file()
        or selected.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a 0400/nlink1 regular file")
    return {
        "path": str(selected),
        "file_sha256": file_sha256(selected, name),
        "byte_count": metadata.st_size,
        "mode_octal": "0400",
        "nlink": 1,
    }


def _assert_worker_context(
    environ: Mapping[str, str] | None = None,
    *,
    hostname: str | None = None,
) -> None:
    values = os.environ if environ is None else environ
    if not values.get("SLURM_JOB_ID"):
        raise RuntimeError("full-search plan authoring must run through Slurm")
    if values.get("SLURM_ARRAY_TASK_ID"):
        raise RuntimeError("full-search plan authoring cannot be an array task")
    short = (socket.gethostname() if hostname is None else hostname).split(".", 1)[0]
    if short.startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("full-search plan authoring cannot run on a login host")


def _build_k1_stage(
    schedule: V5FrozenLocalSobolSchedule,
    calibration: V5CheckedCompatibilityCalibration,
) -> V5FormalProductionSearchStage:
    if schedule.point_count != V5_K1_BALANCED_FULL_SEARCH_EXACT_BUDGET:
        raise ValueError("all-K1 full search requires the frozen B4096 schedule")
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id="gisaxs-v5-2-k1-balanced-full-search-n16-b4096-optimizer-v1",
        direct_scout_seed_count=V5_K1_BALANCED_FULL_SEARCH_DIRECT_SCOUT_COUNT,
        per_seed_forward_evaluation_limit=(
            V5_K1_BALANCED_FULL_SEARCH_PER_SEED_FORWARD_LIMIT
        ),
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
    )
    protocol = build_v5_frozen_exact_search_protocol(
        protocol_id="gisaxs-v5-2-k1-balanced-full-search-paper-calibrated-v1",
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
        calibration_identity=calibration.identity,
    )
    return V5FormalProductionSearchStage(
        stage_id="K1",
        topology_schedule=V5SelectedTopologySearchSchedule(
            schedule_id="gisaxs-v5-2-k1-balanced-full-search-all-k1-topologies-v1",
            selected_topology_ids=topology_ids_for_v5_formal_production_stage("K1"),
        ),
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        protocol=protocol,
    )


def _write_read_only_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    allowed_root: Path,
) -> dict[str, object]:
    target = _under_root(path, allowed_root, "authoring output", exists=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite authoring output: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FileNotFoundError("authoring output parent must already exist")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    return _sealed_file_identity(target, allowed_root, "authoring output")


def _plan_config(
    config: V5K1BalancedFullSearchAuthoringConfig,
    *,
    source_bundle_sha256: str,
    stage: V5FormalProductionSearchStage,
) -> V5K1BalancedFullSearchPlanConfig:
    return V5K1BalancedFullSearchPlanConfig(
        source_archive_sha256=config.expected_source_archive_sha256,
        source_bundle_sha256=source_bundle_sha256,
        balanced_dataset_launch_plan_path=config.balanced_dataset_launch_plan_path,
        balanced_dataset_launch_plan_file_sha256=(
            config.balanced_dataset_launch_plan_file_sha256
        ),
        balanced_dataset_launch_plan_sha256=config.balanced_dataset_launch_plan_sha256,
        balanced_dataset_completion_path=config.balanced_dataset_completion_path,
        balanced_dataset_completion_file_sha256=(
            config.balanced_dataset_completion_file_sha256
        ),
        balanced_dataset_completion_sha256=(
            config.balanced_dataset_completion_sha256
        ),
        train_tuning_receipt_path=config.train_tuning_receipt_path,
        train_tuning_receipt_file_sha256=config.train_tuning_receipt_file_sha256,
        train_tuning_receipt_sha256=config.train_tuning_receipt_sha256,
        train_tuning_claim_sha256=config.train_tuning_claim_sha256,
        phase_c_completion_path=config.phase_c_completion_path,
        phase_c_completion_file_sha256=config.phase_c_completion_file_sha256,
        phase_c_completion_sha256=config.phase_c_completion_sha256,
        three_way_receipt_path=config.three_way_receipt_path,
        three_way_receipt_file_sha256=config.three_way_receipt_file_sha256,
        three_way_receipt_sha256=config.three_way_receipt_sha256,
        phase_c_exclusion_claim_sha256=config.phase_c_exclusion_claim_sha256,
        k1_stage=stage,
        search_run_root=str(config.search_run_root),
    )


def author_v5_k1_balanced_full_search_plan(
    config: V5K1BalancedFullSearchAuthoringConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Replay publications twice and publish a candidate plus completion last."""

    if not isinstance(config, V5K1BalancedFullSearchAuthoringConfig):
        raise TypeError("config has an invalid type")
    # The shared contract root is lexical; downstream publication I/O needs Path.
    allowed_root = Path(allowed_root)
    source = _under_root(config.source_root, allowed_root, "source root", exists=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source root must be a real directory")
    future_root = _under_root(
        config.search_run_root, allowed_root, "future search root", exists=False
    )
    if future_root.exists() or future_root.is_symlink():
        raise FileExistsError("future search root already exists")
    archive_pre = _sealed_file_identity(
        config.source_archive, allowed_root, "source archive"
    )
    if archive_pre["file_sha256"] != config.expected_source_archive_sha256:
        raise ValueError("source archive SHA-256 differs from the authorization")
    schedule_identity_pre = _sealed_file_identity(
        config.local_sobol_schedule_path, allowed_root, "local Sobol schedule"
    )
    calibration_identity_pre = _sealed_file_identity(
        config.calibration_path, allowed_root, "compatibility calibration"
    )
    schedule, schedule_receipt = read_v5_frozen_local_sobol_schedule(
        config.local_sobol_schedule_path
    )
    calibration = read_v5_checked_compatibility_calibration(
        config.calibration_path,
        expected_artifact_sha256=config.expected_calibration_artifact_sha256,
        expected_file_sha256=config.expected_calibration_file_sha256,
    )
    stage = _build_k1_stage(schedule, calibration)
    source_pre = fingerprint_v5_k1_training_source(source)
    if source_pre["source_snapshot_write_bits_set"] or source_pre[
        "source_files_with_write_bits"
    ]:
        raise ValueError("source snapshot is not read-only")
    runtime_config = _plan_config(
        config,
        source_bundle_sha256=str(source_pre["bundle_sha256"]),
        stage=stage,
    )
    first = build_v5_k1_balanced_full_search_plan_from_publications(
        runtime_config, allowed_root=allowed_root
    )
    validate_v5_k1_balanced_full_search_plan(first)
    second = build_v5_k1_balanced_full_search_plan_from_publications(
        runtime_config, allowed_root=allowed_root
    )
    if first != second:
        raise RuntimeError("publication replay changed during plan authoring")
    source_post = fingerprint_v5_k1_training_source(source)
    archive_post = _sealed_file_identity(
        config.source_archive, allowed_root, "source archive"
    )
    schedule_identity_post = _sealed_file_identity(
        config.local_sobol_schedule_path, allowed_root, "local Sobol schedule"
    )
    calibration_identity_post = _sealed_file_identity(
        config.calibration_path, allowed_root, "compatibility calibration"
    )
    if (
        source_post != source_pre
        or archive_post != archive_pre
        or schedule_identity_post != schedule_identity_pre
        or calibration_identity_post != calibration_identity_pre
    ):
        raise RuntimeError("an immutable authoring input changed during replay")
    candidate_identity = _write_read_only_json(
        config.candidate_plan_path, first, allowed_root=allowed_root
    )
    completion_core = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_PLAN_AUTHORING_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_PLAN_AUTHORING_VERSION,
        "status": "PASS",
        "plan_sha256": first["plan_sha256"],
        "plan_candidate": candidate_identity,
        "future_search_run_root": str(future_root),
        "source_identity_pre": source_pre,
        "source_identity_post": source_post,
        "source_archive_identity_pre": archive_pre,
        "source_archive_identity_post": archive_post,
        "local_sobol_schedule_identity_pre": schedule_identity_pre,
        "local_sobol_schedule_identity_post": schedule_identity_post,
        "local_sobol_schedule_artifact_sha256": schedule_receipt.artifact_sha256,
        "calibration_identity_pre": calibration_identity_pre,
        "calibration_identity_post": calibration_identity_post,
        "calibration_artifact_sha256": calibration.identity.artifact_sha256,
        "k1_stage_sha256": stage.sha256,
        "direct_scout_seed_count": V5_K1_BALANCED_FULL_SEARCH_DIRECT_SCOUT_COUNT,
        "per_seed_forward_evaluation_limit": (
            V5_K1_BALANCED_FULL_SEARCH_PER_SEED_FORWARD_LIMIT
        ),
        "exact_forward_call_budget_per_branch": (
            V5_K1_BALANCED_FULL_SEARCH_EXACT_BUDGET
        ),
        "publication_replay_pre_post_equal": True,
        "completion_written_after_candidate_seal": True,
        "scientific_acceptance_evidence": False,
        "gradient_training_authorized": False,
    }
    completion = {
        **completion_core,
        "completion_sha256": sha256(
            canonical_json(completion_core).encode("utf-8")
        ).hexdigest(),
    }
    completion_identity = _write_read_only_json(
        config.completion_path, completion, allowed_root=allowed_root
    )
    return {**completion, "completion_file_identity": completion_identity}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, destination in (
        ("source-root", "source_root"),
        ("source-archive", "source_archive"),
        ("balanced-dataset-launch-plan", "balanced_dataset_launch_plan_path"),
        ("balanced-dataset-completion", "balanced_dataset_completion_path"),
        ("train-tuning-receipt", "train_tuning_receipt_path"),
        ("phase-c-completion", "phase_c_completion_path"),
        ("three-way-receipt", "three_way_receipt_path"),
        ("local-sobol-schedule", "local_sobol_schedule_path"),
        ("calibration", "calibration_path"),
        ("search-run-root", "search_run_root"),
        ("candidate-plan", "candidate_plan_path"),
        ("completion", "completion_path"),
    ):
        parser.add_argument(f"--{name}", dest=destination, required=True, type=Path)
    for name in (
        "expected-source-archive-sha256",
        "balanced-dataset-launch-plan-file-sha256",
        "balanced-dataset-launch-plan-sha256",
        "balanced-dataset-completion-file-sha256",
        "balanced-dataset-completion-sha256",
        "train-tuning-receipt-file-sha256",
        "train-tuning-receipt-sha256",
        "train-tuning-claim-sha256",
        "phase-c-completion-file-sha256",
        "phase-c-completion-sha256",
        "three-way-receipt-file-sha256",
        "three-way-receipt-sha256",
        "phase-c-exclusion-claim-sha256",
        "expected-calibration-artifact-sha256",
        "expected-calibration-file-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _assert_worker_context()
    args = _parser().parse_args(argv)
    result = author_v5_k1_balanced_full_search_plan(
        V5K1BalancedFullSearchAuthoringConfig(
            **vars(args),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5K1BalancedFullSearchAuthoringConfig",
    "V5_K1_BALANCED_FULL_SEARCH_DIRECT_SCOUT_COUNT",
    "V5_K1_BALANCED_FULL_SEARCH_EXACT_BUDGET",
    "V5_K1_BALANCED_FULL_SEARCH_PER_SEED_FORWARD_LIMIT",
    "V5_K1_BALANCED_FULL_SEARCH_PLAN_AUTHORING_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_PLAN_AUTHORING_VERSION",
    "author_v5_k1_balanced_full_search_plan",
    "main",
]
