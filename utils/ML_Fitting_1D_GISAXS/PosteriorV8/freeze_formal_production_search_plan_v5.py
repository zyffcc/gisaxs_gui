"""Freeze a replayable three-stage V5 formal contract-smoke plan.

This login-node-safe command only reads checked artifacts, fingerprints source,
selects explicit Sobol parent membership, and writes one plan with ``O_EXCL``.
It performs no curve generation, exact search, Slurm submission, or training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
from typing import Mapping, Sequence

from .calibrated_search_threshold_v5 import (
    V5CheckedCompatibilityCalibration,
    inspect_v5_compatibility_calibration,
)
from .contract import NUM_TOPOLOGIES, topology_from_id
from .exact_search_executor_v5 import build_v5_frozen_exact_search_protocol
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
    read_v5_frozen_local_sobol_schedule,
)
from .formal_production_search_contract_v5 import (
    V5FormalProductionSearchStage,
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_ALLOWED_SPLITS,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
    plan_v5_formal_production_search_shard,
    topology_ids_for_v5_formal_production_stage,
)
from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchPlan,
    build_v5_formal_production_search_plan,
)
from .formal_production_search_worker_contract_v5 import (
    read_v5_formal_production_global_plan,
    validate_v5_formal_production_global_plan_payload,
)
from .frozen_search_launch_contracts_v5 import fingerprint_v5_frozen_search_source
from .frozen_search_pipeline_contract_v5 import V5SelectedTopologySearchSchedule
from .sobol_design_v5 import (
    V5DesignPoint,
    V5SobolDesign,
    materialize_v5_design_points_for_indices,
)
from .sobol_recipe_coordinates_v5 import V5_SOBOL_RECIPE_COORDINATE_INDEX
from .split_design_v5 import V5SplitPlan
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)


V5_FORMAL_CONTRACT_SMOKE_FREEZE_SCHEMA = "gisaxs.posterior_v8.formal_contract_smoke_plan_freeze/v1"
V5_FORMAL_CONTRACT_SMOKE_FREEZE_VERSION = (
    "earliest_disjoint_stage_band_parents_exclusive_global_plan_v1"
)
V5_FORMAL_CONTRACT_SMOKE_PARENT_POLICY = (
    "earliest_sobol_parent_per_split_in_new_topology_band_k1_exact1_k2_exact2_all34_k3_or_k4"
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_file(path: Path, name: str) -> Path:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"{name} must be an existing non-symlink file")
    return selected.resolve(strict=True)


def _read_json_contract(path: Path, name: str, reader):
    selected = _checked_file(path, name)
    before = selected.read_bytes()
    value = reader(before.decode("utf-8"))
    if selected.read_bytes() != before:
        raise RuntimeError(f"{name} changed while it was being read")
    return value, sha256(before).hexdigest()


def _generating_topology_id(point: V5DesignPoint) -> int:
    coordinate = point.unit_coordinates[V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]]
    return min(int(coordinate * NUM_TOPOLOGIES), NUM_TOPOLOGIES - 1)


def _stage_band_for_point(point: V5DesignPoint) -> str:
    component_count = len(topology_from_id(_generating_topology_id(point)))
    if component_count == 1:
        return "K1"
    if component_count == 2:
        return "K2"
    return "ALL34"


@dataclass(frozen=True, kw_only=True)
class V5FormalContractSmokeParentSelection:
    stage_id: str
    target_split: str
    sobol_indices: tuple[int, ...]


def select_v5_formal_contract_smoke_parent_indices(
    split_plan: V5SplitPlan,
    sobol_design: V5SobolDesign,
    *,
    parents_per_stage_split: int = 1,
    scan_chunk_size: int = 1024,
) -> tuple[V5FormalContractSmokeParentSelection, ...]:
    """Select the earliest parent in each newly introduced topology band."""

    if not isinstance(split_plan, V5SplitPlan) or not isinstance(sobol_design, V5SobolDesign):
        raise TypeError("split_plan and sobol_design have invalid types")
    count = _positive_integer(parents_per_stage_split, "parents_per_stage_split")
    chunk_size = _positive_integer(scan_chunk_size, "scan_chunk_size")
    selected: dict[tuple[str, str], list[int]] = {
        (stage_id, split): []
        for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
        for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS
    }
    for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
        block = next(value for value in split_plan.blocks if value.name == split)
        for start in range(block.start, block.stop, chunk_size):
            indices = tuple(range(start, min(start + chunk_size, block.stop)))
            for point in materialize_v5_design_points_for_indices(
                split_plan, sobol_design, indices
            ):
                key = (_stage_band_for_point(point), split)
                if len(selected[key]) < count:
                    selected[key].append(point.sobol_index)
            if all(
                len(selected[(stage_id, split)]) == count
                for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
            ):
                break
        missing = {
            stage_id: count - len(selected[(stage_id, split)])
            for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
            if len(selected[(stage_id, split)]) != count
        }
        if missing:
            raise ValueError(f"{split} has too few parents for the formal stage bands: {missing}")
    result = tuple(
        V5FormalContractSmokeParentSelection(
            stage_id=stage_id,
            target_split=split,
            sobol_indices=tuple(selected[(stage_id, split)]),
        )
        for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
        for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS
    )
    flattened = [index for value in result for index in value.sobol_indices]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("formal contract-smoke parent selections overlap")
    return result


def _build_stages(
    *,
    study_id: str,
    calibration: V5CheckedCompatibilityCalibration,
    schedules: Mapping[str, V5FrozenLocalSobolSchedule],
    direct_scout_seed_count: int,
    per_seed_forward_evaluation_limit: int,
    ftol: float,
    xtol: float,
    gtol: float,
) -> tuple[V5FormalProductionSearchStage, ...]:
    if set(schedules) != set(V5_FORMAL_PRODUCTION_STAGE_IDS):
        raise ValueError("exactly one local-Sobol schedule is required for K1, K2, and ALL34")
    stages = []
    for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS:
        seed_schedule = schedules[stage_id]
        optimizer = V5FrozenExactOptimizerSchedule(
            schedule_id=f"{study_id}-{stage_id.lower()}-contract-smoke-optimizer-v1",
            direct_scout_seed_count=direct_scout_seed_count,
            per_seed_forward_evaluation_limit=per_seed_forward_evaluation_limit,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )
        protocol = build_v5_frozen_exact_search_protocol(
            protocol_id=f"{study_id}-{stage_id.lower()}-contract-smoke-protocol-v1",
            seed_schedule=seed_schedule,
            optimizer_schedule=optimizer,
            protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
            calibration_identity=calibration.identity,
        )
        stages.append(
            V5FormalProductionSearchStage(
                stage_id=stage_id,
                topology_schedule=V5SelectedTopologySearchSchedule(
                    schedule_id=f"{study_id}-{stage_id.lower()}-topologies-v1",
                    selected_topology_ids=topology_ids_for_v5_formal_production_stage(stage_id),
                ),
                seed_schedule=seed_schedule,
                optimizer_schedule=optimizer,
                protocol=protocol,
            )
        )
    return tuple(stages)


def build_v5_formal_contract_smoke_plan(
    *,
    study_id: str,
    source: V5FormalProductionSourceIdentity,
    split_plan: V5SplitPlan,
    sobol_design: V5SobolDesign,
    calibration: V5CheckedCompatibilityCalibration,
    local_sobol_schedules: Mapping[str, V5FrozenLocalSobolSchedule],
    parents_per_stage_split: int = 1,
    parents_per_shard: int = 1,
    candidate_view_indices: Sequence[int] = (0, 1),
    direct_scout_seed_count: int = 1,
    per_seed_forward_evaluation_limit: int = 1,
    ftol: float = 1.0e-8,
    xtol: float = 1.0e-8,
    gtol: float = 1.0e-8,
) -> V5FormalProductionSearchPlan:
    """Compose the immutable non-training global plan from checked objects."""

    study = _text(study_id, "study_id")
    count = _positive_integer(parents_per_stage_split, "parents_per_stage_split")
    shard_size = _positive_integer(parents_per_shard, "parents_per_shard")
    scout_count = _positive_integer(direct_scout_seed_count, "direct_scout_seed_count")
    per_seed_limit = _positive_integer(
        per_seed_forward_evaluation_limit, "per_seed_forward_evaluation_limit"
    )
    stages = _build_stages(
        study_id=study,
        calibration=calibration,
        schedules=dict(local_sobol_schedules),
        direct_scout_seed_count=scout_count,
        per_seed_forward_evaluation_limit=per_seed_limit,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )
    if any(scout_count > stage.seed_schedule.point_count for stage in stages):
        raise ValueError("direct scout count exceeds a staged local-Sobol budget")
    selections = select_v5_formal_contract_smoke_parent_indices(
        split_plan,
        sobol_design,
        parents_per_stage_split=count,
    )
    selection_by_key = {
        (value.stage_id, value.target_split): value.sobol_indices for value in selections
    }
    shards = []
    for stage in stages:
        for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
            indices = selection_by_key[(stage.stage_id, split)]
            for ordinal, start in enumerate(range(0, len(indices), shard_size)):
                shards.append(
                    plan_v5_formal_production_search_shard(
                        split_plan=split_plan,
                        sobol_design=sobol_design,
                        stage=stage,
                        target_split=split,
                        sobol_indices=indices[start : start + shard_size],
                        output_relative_path=(
                            f"labels/{stage.stage_id.lower()}/{split}/"
                            f"contract-smoke-shard-{ordinal:06d}"
                        ),
                        candidate_view_indices=candidate_view_indices,
                    )
                )
    return build_v5_formal_production_search_plan(
        study_id=study,
        source=source,
        split_plan=split_plan,
        sobol_design=sobol_design,
        calibration=calibration,
        candidate_view_indices=candidate_view_indices,
        stages=stages,
        shards=shards,
    )


def summarize_v5_formal_contract_smoke_plan(
    plan: V5FormalProductionSearchPlan,
) -> dict[str, object]:
    rows = []
    for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS:
        for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
            selected = tuple(
                shard
                for shard in plan.shards
                if shard.stage.stage_id == stage_id and shard.target_split == split
            )
            rows.append(
                {
                    "stage_id": stage_id,
                    "split": split,
                    "shards": len(selected),
                    "queries": sum(value.expected_query_count for value in selected),
                    "branches": sum(value.expected_branch_count for value in selected),
                    "exact_forward_calls": sum(
                        value.expected_exact_forward_calls for value in selected
                    ),
                }
            )
    return {
        "schema": V5_FORMAL_CONTRACT_SMOKE_FREEZE_SCHEMA,
        "version": V5_FORMAL_CONTRACT_SMOKE_FREEZE_VERSION,
        "plan_sha256": plan.sha256,
        "parent_selection_policy": V5_FORMAL_CONTRACT_SMOKE_PARENT_POLICY,
        "training_promotion_enabled": False,
        "stage_split_workload": rows,
        "totals": {
            key: sum(int(row[key]) for row in rows)
            for key in ("shards", "queries", "branches", "exact_forward_calls")
        },
    }


@dataclass(frozen=True, kw_only=True)
class V5FormalContractSmokeFreezeConfig:
    study_id: str
    source_root: Path
    split_plan_path: Path
    sobol_design_path: Path
    calibration_path: Path
    local_sobol_schedule_paths: Mapping[str, Path]
    output_path: Path
    parents_per_stage_split: int = 1
    parents_per_shard: int = 1
    candidate_view_indices: tuple[int, ...] = (0, 1)
    direct_scout_seed_count: int = 1
    per_seed_forward_evaluation_limit: int = 1
    ftol: float = 1.0e-8
    xtol: float = 1.0e-8
    gtol: float = 1.0e-8


def _input_file_hashes(config: V5FormalContractSmokeFreezeConfig) -> dict[str, str]:
    paths = {
        "split_plan": config.split_plan_path,
        "sobol_design": config.sobol_design_path,
        "calibration": config.calibration_path,
        **{
            f"local_sobol_schedule[{stage_id}]": path
            for stage_id, path in config.local_sobol_schedule_paths.items()
        },
    }
    return {
        name: _file_sha256(_checked_file(Path(path), name)) for name, path in sorted(paths.items())
    }


def _write_plan_exclusive(path: Path, encoded: str) -> None:
    if not path.parent.is_dir():
        raise ValueError("output plan parent directory must already exist")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def freeze_v5_formal_contract_smoke_plan(
    config: V5FormalContractSmokeFreezeConfig,
) -> dict[str, object]:
    """Read, replay, and exclusively publish one immutable global plan."""

    if not isinstance(config, V5FormalContractSmokeFreezeConfig):
        raise TypeError("config must be V5FormalContractSmokeFreezeConfig")
    output_path = Path(config.output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite global plan: {output_path}")
    if set(config.local_sobol_schedule_paths) != set(V5_FORMAL_PRODUCTION_STAGE_IDS):
        raise ValueError("schedule paths must contain exactly K1, K2, and ALL34")
    source_before = V5FormalProductionSourceIdentity.from_fingerprint(
        fingerprint_v5_frozen_search_source(Path(config.source_root))
    )
    file_hashes_before = _input_file_hashes(config)
    split_plan, _ = _read_json_contract(
        Path(config.split_plan_path), "split_plan", V5SplitPlan.from_json
    )
    sobol_design, _ = _read_json_contract(
        Path(config.sobol_design_path), "sobol_design", V5SobolDesign.from_json
    )
    calibration = inspect_v5_compatibility_calibration(config.calibration_path)
    schedules = {}
    for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS:
        schedule, _receipt = read_v5_frozen_local_sobol_schedule(
            config.local_sobol_schedule_paths[stage_id]
        )
        schedule.verify_runtime_replay()
        schedules[stage_id] = schedule
    plan = build_v5_formal_contract_smoke_plan(
        study_id=config.study_id,
        source=source_before,
        split_plan=split_plan,
        sobol_design=sobol_design,
        calibration=calibration,
        local_sobol_schedules=schedules,
        parents_per_stage_split=config.parents_per_stage_split,
        parents_per_shard=config.parents_per_shard,
        candidate_view_indices=config.candidate_view_indices,
        direct_scout_seed_count=config.direct_scout_seed_count,
        per_seed_forward_evaluation_limit=config.per_seed_forward_evaluation_limit,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
    )
    validate_v5_formal_production_global_plan_payload(
        plan.to_payload(), expected_plan_sha256=plan.sha256
    )
    source_after = V5FormalProductionSourceIdentity.from_fingerprint(
        fingerprint_v5_frozen_search_source(Path(config.source_root))
    )
    if source_after.audit_payload() != source_before.audit_payload():
        raise RuntimeError("source bundle changed while the formal plan was being frozen")
    if _input_file_hashes(config) != file_hashes_before:
        raise RuntimeError("a checked scientific input changed while the plan was being frozen")
    encoded = plan.to_json()
    _write_plan_exclusive(output_path, encoded)
    read_v5_formal_production_global_plan(output_path, expected_plan_sha256=plan.sha256)
    return {
        **summarize_v5_formal_contract_smoke_plan(plan),
        "output_path": str(output_path.resolve()),
        "plan_file_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
        "source_bundle_sha256": plan.source.bundle_sha256,
        "calibration_artifact_sha256": plan.calibration.identity.artifact_sha256,
        "local_sobol_schedule_sha256": {
            value.stage_id: value.seed_schedule.sha256 for value in plan.stages
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--sobol-design", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--k1-local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--k2-local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--all34-local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--parents-per-stage-split", type=int, default=1)
    parser.add_argument("--parents-per-shard", type=int, default=1)
    parser.add_argument("--candidate-view-index", action="append", type=int)
    parser.add_argument("--direct-scout-seed-count", type=int, default=1)
    parser.add_argument("--per-seed-forward-evaluation-limit", type=int, default=1)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--xtol", type=float, default=1.0e-8)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = freeze_v5_formal_contract_smoke_plan(
        V5FormalContractSmokeFreezeConfig(
            study_id=args.study_id,
            source_root=args.source_root,
            split_plan_path=args.split_plan,
            sobol_design_path=args.sobol_design,
            calibration_path=args.calibration,
            local_sobol_schedule_paths={
                "K1": args.k1_local_sobol_schedule,
                "K2": args.k2_local_sobol_schedule,
                "ALL34": args.all34_local_sobol_schedule,
            },
            output_path=args.output_plan,
            parents_per_stage_split=args.parents_per_stage_split,
            parents_per_shard=args.parents_per_shard,
            candidate_view_indices=tuple(args.candidate_view_index or (0, 1)),
            direct_scout_seed_count=args.direct_scout_seed_count,
            per_seed_forward_evaluation_limit=args.per_seed_forward_evaluation_limit,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
        )
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5FormalContractSmokeFreezeConfig",
    "V5FormalContractSmokeParentSelection",
    "V5_FORMAL_CONTRACT_SMOKE_FREEZE_SCHEMA",
    "V5_FORMAL_CONTRACT_SMOKE_FREEZE_VERSION",
    "V5_FORMAL_CONTRACT_SMOKE_PARENT_POLICY",
    "build_v5_formal_contract_smoke_plan",
    "freeze_v5_formal_contract_smoke_plan",
    "main",
    "select_v5_formal_contract_smoke_parent_indices",
    "summarize_v5_formal_contract_smoke_plan",
]
