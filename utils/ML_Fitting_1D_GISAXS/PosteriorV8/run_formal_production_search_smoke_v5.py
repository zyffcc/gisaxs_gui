"""CLI for one non-training V5 formal-production contract-smoke shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .formal_production_search_worker_contract_v5 import (
    prepare_v5_formal_production_worker_shard,
)
from .frozen_search_pipeline_v5 import execute_v5_frozen_search_shard
from .search_evidence_receipt_v5 import (
    V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one formal-production shard; dry-run by default."
    )
    parser.add_argument("--global-plan", type=Path, required=True)
    parser.add_argument("--expected-global-plan-sha256", required=True)
    parser.add_argument("--expected-shard-plan-sha256", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--array-task-id", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--sobol-design", type=Path, required=True)
    parser.add_argument("--local-sobol-schedule", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare_v5_formal_production_worker_shard(
        global_plan_path=args.global_plan,
        expected_global_plan_sha256=args.expected_global_plan_sha256,
        expected_shard_plan_sha256=args.expected_shard_plan_sha256,
        stage_id=args.stage_id,
        target_split=args.target_split,
        array_task_id=args.array_task_id,
        source_root=args.source_root,
        run_root=args.run_root,
        split_plan_path=args.split_plan,
        sobol_design_path=args.sobol_design,
        local_sobol_schedule_path=args.local_sobol_schedule,
        calibration_path=args.calibration_artifact,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "writes_performed": False,
                    "training_promotion_enabled": False,
                    "label_purpose": V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE,
                    "global_plan_sha256": prepared.global_plan_sha256,
                    "shard_plan_sha256": prepared.runtime_shard.sha256,
                    "output_root": str(prepared.output_root),
                    "selected_view_indices": [
                        prepared.runtime_shard.view_indices_for_recipe(index)[0]
                        for index in range(len(prepared.runtime_shard.points))
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    completion = execute_v5_frozen_search_shard(
        prepared.runtime_shard,
        prepared.execution,
        prepared.output_root,
        allow_local_smoke=args.allow_local_smoke,
        source_bundle_fingerprint=prepared.verify_source_bundle,
    )
    if (
        completion.get("label_purpose")
        != V5_SEARCH_LABEL_PURPOSE_FORMAL_CONTRACT_SMOKE
        or completion.get("full_training_label_claimed") is not False
    ):
        raise RuntimeError("formal contract smoke crossed the TRAINING boundary")
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
