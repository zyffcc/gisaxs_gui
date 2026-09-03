"""Command-line entry point for one V5.1 frozen-search worker shard."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from .calibrated_search_threshold_v5 import (
    read_v5_checked_compatibility_calibration,
)
from .exact_search_executor_v5 import build_v5_frozen_exact_search_protocol
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    read_v5_frozen_local_sobol_schedule,
)
from .frozen_search_pipeline_v5 import (
    V5FrozenSearchExecution,
    V5SelectedTopologySearchSchedule,
    V5_SEARCH_PIPELINE_ALLOWED_SPLITS,
    execute_v5_frozen_search_shard,
    plan_v5_frozen_search_shard,
)
from .frozen_search_launch_plan_v5 import fingerprint_v5_frozen_search_source
from .sobol_design_v5 import V5SobolDesign
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
)
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
    V5_SEARCH_PROTOCOL_TIERS,
)
from .split_design_v5 import V5SplitPlan


def _integer_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            int(item.strip())
            for item in value.replace(":", ",").split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer list must be non-empty and unique")
    return result


def _load_plan_and_design(
    split_plan: Path, sobol_design: Path
) -> tuple[V5SplitPlan, V5SobolDesign]:
    plan = V5SplitPlan.from_json(split_plan.read_text(encoding="utf-8"))
    design = V5SobolDesign.from_json(sobol_design.read_text(encoding="utf-8"))
    if (
        design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
        or design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Sobol design is not bound to the direct V5 recipe contract")
    return plan, design


def _execution_from_args(args: argparse.Namespace) -> V5FrozenSearchExecution:
    schedule, receipt = read_v5_frozen_local_sobol_schedule(args.local_sobol_schedule)
    if (
        schedule.sha256 != args.expected_local_schedule_sha256
        or receipt.artifact_sha256
        != args.expected_local_schedule_artifact_sha256
        or receipt.manifest_sha256
        != args.expected_local_schedule_manifest_sha256
    ):
        raise RuntimeError(
            "local-Sobol schedule identity changed after launch planning"
        )
    schedule.verify_runtime_replay()
    optimizer = V5FrozenExactOptimizerSchedule(
        schedule_id=args.optimizer_schedule_id,
        direct_scout_seed_count=args.direct_scout_seed_count,
        per_seed_forward_evaluation_limit=args.per_seed_forward_evaluation_limit,
        ftol=args.ftol,
        xtol=args.xtol,
        gtol=args.gtol,
    )
    protocol_kwargs = {
        "protocol_id": args.protocol_id,
        "seed_schedule": schedule,
        "optimizer_schedule": optimizer,
    }
    calibration = None
    if args.protocol_tier == V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT:
        if args.calibration_artifact is not None or any(
            value is not None
            for value in (
                args.expected_calibration_sha256,
                args.expected_calibration_file_sha256,
            )
        ):
            raise ValueError("engineering pilot cannot accept a calibration artifact")
        protocol = build_v5_frozen_exact_search_protocol(
            **protocol_kwargs,
            standardized_threshold_name=args.standardized_threshold_name,
            standardized_threshold_value=args.standardized_threshold_value,
            raw_threshold_name=args.raw_threshold_name,
            raw_threshold_value=args.raw_threshold_value,
            threshold_source_id=args.threshold_source_id,
        )
    else:
        if any(
            value is not None
            for value in (
                args.standardized_threshold_name,
                args.standardized_threshold_value,
                args.raw_threshold_name,
                args.raw_threshold_value,
                args.threshold_source_id,
            )
        ):
            raise ValueError("paper/full protocol forbids scalar threshold arguments")
        if (
            args.calibration_artifact is None
            or args.expected_calibration_sha256 is None
            or args.expected_calibration_file_sha256 is None
        ):
            raise ValueError("paper/full protocol requires checked calibration arguments")
        calibration = read_v5_checked_compatibility_calibration(
            args.calibration_artifact,
            expected_artifact_sha256=args.expected_calibration_sha256,
            expected_file_sha256=args.expected_calibration_file_sha256,
        )
        protocol = build_v5_frozen_exact_search_protocol(
            **protocol_kwargs,
            protocol_tier=V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
            calibration_identity=calibration.identity,
        )
    return V5FrozenSearchExecution(
        seed_schedule=schedule,
        optimizer_schedule=optimizer,
        protocol=protocol,
        launch_source_bundle_sha256=args.expected_source_bundle_sha256,
        launch_plan_sha256=args.launch_plan_sha256,
        calibration=calibration,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--launch-plan-sha256", required=True)
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--expected-split-plan-sha256", required=True)
    parser.add_argument("--expected-split-plan-file-sha256", required=True)
    parser.add_argument("--sobol-design", required=True, type=Path)
    parser.add_argument("--expected-sobol-design-sha256", required=True)
    parser.add_argument("--expected-sobol-design-file-sha256", required=True)
    parser.add_argument(
        "--target-split", required=True, choices=V5_SEARCH_PIPELINE_ALLOWED_SPLITS
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--start", type=int)
    selector.add_argument("--shard-index", type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--view-indices", type=_integer_list, default=(0,))
    parser.add_argument("--topology-schedule-id", required=True)
    parser.add_argument("--selected-topology-ids", required=True, type=_integer_list)
    parser.add_argument("--local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--expected-local-schedule-sha256", required=True)
    parser.add_argument("--expected-local-schedule-artifact-sha256", required=True)
    parser.add_argument("--expected-local-schedule-manifest-sha256", required=True)
    parser.add_argument("--optimizer-schedule-id", required=True)
    parser.add_argument("--direct-scout-seed-count", required=True, type=int)
    parser.add_argument("--per-seed-forward-evaluation-limit", required=True, type=int)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--xtol", type=float, default=1.0e-8)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument(
        "--protocol-tier",
        choices=V5_SEARCH_PROTOCOL_TIERS,
        default=V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT,
    )
    parser.add_argument("--standardized-threshold-name")
    parser.add_argument("--standardized-threshold-value", type=float)
    parser.add_argument("--raw-threshold-name")
    parser.add_argument("--raw-threshold-value", type=float)
    parser.add_argument("--threshold-source-id")
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--expected-calibration-sha256")
    parser.add_argument("--expected-calibration-file-sha256")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = fingerprint_v5_frozen_search_source(args.source_root)
    if source["bundle_sha256"] != args.expected_source_bundle_sha256:
        raise RuntimeError(
            "worker source bundle changed after launch planning; refusing physics/search work"
        )
    if (
        sha256(args.split_plan.read_bytes()).hexdigest()
        != args.expected_split_plan_file_sha256
        or sha256(args.sobol_design.read_bytes()).hexdigest()
        != args.expected_sobol_design_file_sha256
    ):
        raise RuntimeError(
            "split-plan or Sobol-design file changed after launch planning"
        )
    plan, design = _load_plan_and_design(args.split_plan, args.sobol_design)
    if (
        plan.sha256 != args.expected_split_plan_sha256
        or design.sha256 != args.expected_sobol_design_sha256
    ):
        raise RuntimeError("split plan or Sobol design changed after launch planning")
    execution = _execution_from_args(args)
    topology_schedule = V5SelectedTopologySearchSchedule(
        schedule_id=args.topology_schedule_id,
        selected_topology_ids=args.selected_topology_ids,
    )
    shard_plan = plan_v5_frozen_search_shard(
        plan=plan,
        design=design,
        target_split=args.target_split,
        start=args.start,
        shard_index=args.shard_index,
        count=args.count,
        view_indices=args.view_indices,
        topology_schedule=topology_schedule,
    )
    if args.output_root.exists() and not args.execute:
        raise FileExistsError(
            f"refusing a dry-run against an existing output: {args.output_root}"
        )
    result = (
        execute_v5_frozen_search_shard(
            shard_plan,
            execution,
            args.output_root,
            allow_local_smoke=args.allow_local_smoke,
            source_bundle_fingerprint=lambda: str(
                fingerprint_v5_frozen_search_source(args.source_root)[
                    "bundle_sha256"
                ]
            ),
        )
        if args.execute
        else {
            "status": "dry_run",
            "writes_performed": False,
            "output_root": str(args.output_root),
            "pipeline_plan": shard_plan.audit_payload(),
            "pipeline_plan_sha256": shard_plan.sha256,
            "launch_plan_sha256": execution.launch_plan_sha256,
            "seed_schedule_sha256": execution.seed_schedule.sha256,
            "optimizer_schedule_sha256": execution.optimizer_schedule.sha256,
            "protocol_sha256": execution.protocol.sha256,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
