"""Build one immutable, single-split direct-Sobol V5.1 grouped shard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import socket
from typing import Sequence

import numpy as np

from .build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_v5_grouped_solution_dataset,
)
from .grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from .grouped_dataset_v5 import V5GroupedDataset, write_v5_grouped_dataset
from .grouped_shard_metadata_v5 import build_formal_sobol_shard_selection
from .sobol_design_v5 import (
    V5SobolDesign,
    materialize_v5_design_points_for_indices,
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
)
from .sobol_recipe_v5 import materialize_v5_sobol_clean_recipe
from .split_design_v5 import MAIN_SPLITS, V5SplitPlan


V5_FORMAL_SOBOL_GROUPED_BUILDER_SCHEMA = "gisaxs.posterior_v8.formal_sobol_grouped_shard_builder/v1"
V5_FORMAL_SOBOL_GROUPED_BUILDER_VERSION = (
    "posterior_v8_v5_1_single_split_direct_coordinate_cpu_shard_v1"
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _view_indices(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("view indices must be comma-separated integers") from exc
    else:
        selected = tuple(value)
    result = tuple(_integer(item, "view_index") for item in selected)
    if not result or len(set(result)) != len(result):
        raise ValueError("view indices must be non-empty and unique")
    return result


def _load_inputs(
    split_plan_path: str | os.PathLike[str],
    sobol_design_path: str | os.PathLike[str],
) -> tuple[V5SplitPlan, V5SobolDesign]:
    plan = V5SplitPlan.from_json(Path(split_plan_path).read_text(encoding="utf-8"))
    design = V5SobolDesign.from_json(Path(sobol_design_path).read_text(encoding="utf-8"))
    if (
        design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
        or design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Sobol design is not bound to the frozen direct-recipe coordinates")
    return plan, design


def _target_block(plan: V5SplitPlan, target_split: str):
    if target_split not in MAIN_SPLITS:
        raise ValueError(f"target_split must be one of {MAIN_SPLITS}")
    if target_split == "ood":
        raise ValueError("current OOD design is fail-closed and cannot generate formal shards")
    return next(block for block in plan.blocks if block.name == target_split)


@dataclass(frozen=True)
class V5FormalSobolShardPlan:
    plan: V5SplitPlan
    design: V5SobolDesign
    target_split: str
    split_offset: int
    requested_count: int
    selected_indices: tuple[int, ...]
    view_indices: tuple[int, ...]
    generating_only: bool
    selection_mode: str
    shard_index: int | None
    selection: dict[str, object]

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_FORMAL_SOBOL_GROUPED_BUILDER_SCHEMA,
            "version": V5_FORMAL_SOBOL_GROUPED_BUILDER_VERSION,
            "target_split": self.target_split,
            "split_offset": self.split_offset,
            "requested_count": self.requested_count,
            "actual_count": len(self.selected_indices),
            "selection_mode": self.selection_mode,
            "shard_index": self.shard_index,
            "view_indices": list(self.view_indices),
            "generating_candidate_only": self.generating_only,
            "split_plan_sha256": self.plan.sha256,
            "sobol_design_sha256": self.design.sha256,
            "sobol_design_scipy_version": self.design.payload()["scipy_version"],
            "coordinate_contract_sha256": self.design.coordinate_contract_sha256,
            "runtime_binding": "Sobol design scipy_version must match the shard builder runtime",
            "selection_sha256": self.selection["selection_sha256"],
            "sobol_index_runs": self.selection["sobol_index_runs"],
            "sobol_indices_contiguous": self.selection["sobol_indices_contiguous"],
        }


def plan_formal_sobol_grouped_shard(
    *,
    plan: V5SplitPlan,
    design: V5SobolDesign,
    target_split: str,
    count: int,
    view_indices: Sequence[int] = (0,),
    start: int | None = None,
    shard_index: int | None = None,
    generating_only: bool = True,
) -> V5FormalSobolShardPlan:
    """Resolve a split-relative start/count or array shard index without physics work."""

    if not isinstance(plan, V5SplitPlan) or not isinstance(design, V5SobolDesign):
        raise TypeError("plan/design have invalid types")
    if (
        design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
        or design.coordinate_contract_sha256 != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Sobol design is not bound to the frozen direct-recipe coordinates")
    if (start is None) == (shard_index is None):
        raise ValueError("supply exactly one of start or shard_index")
    selected_count = _integer(count, "count", minimum=1)
    if start is not None:
        offset = _integer(start, "start")
        selected_shard = None
        mode = "start_count"
    else:
        selected_shard = _integer(shard_index, "shard_index")
        offset = selected_shard * selected_count
        mode = "shard_index"
    views = _view_indices(view_indices)
    if type(generating_only) is not bool:
        raise TypeError("generating_only must be a bool")
    block = _target_block(plan, target_split)
    if offset >= block.count:
        raise ValueError("requested shard starts beyond the target split")
    stop = min(block.start + offset + selected_count, block.stop)
    indices = tuple(range(block.start + offset, stop))
    selection = build_formal_sobol_shard_selection(
        target_split=target_split,
        selection_mode=mode,
        split_offset=offset,
        requested_count=selected_count,
        selected_indices=indices,
        shard_index=selected_shard,
        view_indices=views,
        generating_only=generating_only,
        split_plan_sha256=plan.sha256,
        sobol_design_sha256=design.sha256,
        coordinate_contract_sha256=V5_SOBOL_RECIPE_COORDINATE_SHA256,
    )
    return V5FormalSobolShardPlan(
        plan=plan,
        design=design,
        target_split=target_split,
        split_offset=offset,
        requested_count=selected_count,
        selected_indices=indices,
        view_indices=views,
        generating_only=generating_only,
        selection_mode=mode,
        shard_index=selected_shard,
        selection=selection,
    )


def _assert_build_host() -> None:
    if socket.gethostname().split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("formal dataset generation is forbidden on the Maxwell login node")


def build_formal_sobol_grouped_shard(
    shard_plan: V5FormalSobolShardPlan,
    output: str | os.PathLike[str],
) -> tuple[V5GroupedDataset, V5ArtifactReceipt]:
    """Materialize and exclusively publish one direct-Sobol grouped shard."""

    if not isinstance(shard_plan, V5FormalSobolShardPlan):
        raise TypeError("shard_plan must be V5FormalSobolShardPlan")
    _assert_build_host()
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    points = materialize_v5_design_points_for_indices(
        shard_plan.plan,
        shard_plan.design,
        shard_plan.selected_indices,
    )
    recipes = tuple(materialize_v5_sobol_clean_recipe(point, shard_plan.design) for point in points)
    specs = tuple(
        V5GroupedRecipeSpec.from_design_point(
            recipe,
            point,
            split_plan_sha256=shard_plan.plan.sha256,
            sobol_design_sha256=shard_plan.design.sha256,
            view_indices=shard_plan.view_indices,
        )
        for recipe, point in zip(recipes, points)
    )
    identity = {
        "builder_schema": V5_FORMAL_SOBOL_GROUPED_BUILDER_SCHEMA,
        "builder_version": V5_FORMAL_SOBOL_GROUPED_BUILDER_VERSION,
        "selection_sha256": shard_plan.selection["selection_sha256"],
    }
    dataset_id = f"formal-sobol-v5-{sha256(canonical_json(identity).encode()).hexdigest()}"
    dataset = build_v5_grouped_solution_dataset(
        specs,
        dataset_id=dataset_id,
        generating_only=shard_plan.generating_only,
        shard_selection=shard_plan.selection,
    )
    return dataset, write_v5_grouped_dataset(dataset, target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument(
        "--sobol-design",
        required=True,
        type=Path,
        help="frozen design JSON created with the same SciPy runtime as this builder",
    )
    parser.add_argument("--target-split", required=True, choices=MAIN_SPLITS)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--start", type=int, help="zero-based offset within target split")
    selector.add_argument("--shard-index", type=int, help="count-sized split-relative shard")
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--view-indices", default="0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-unverified-branches", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, design = _load_inputs(args.split_plan, args.sobol_design)
    shard_plan = plan_formal_sobol_grouped_shard(
        plan=plan,
        design=design,
        target_split=args.target_split,
        start=args.start,
        shard_index=args.shard_index,
        count=args.count,
        view_indices=_view_indices(args.view_indices),
        generating_only=not args.include_unverified_branches,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")
    result = {**shard_plan.audit_payload(), "output": str(args.output), "dry_run": args.dry_run}
    if not args.dry_run:
        _, receipt = build_formal_sobol_grouped_shard(shard_plan, args.output)
        result.update(
            {
                "artifact_sha256": receipt.artifact_sha256,
                "manifest_sha256": receipt.manifest_sha256,
                "byte_count": receipt.byte_count,
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "V5_FORMAL_SOBOL_GROUPED_BUILDER_SCHEMA",
    "V5_FORMAL_SOBOL_GROUPED_BUILDER_VERSION",
    "V5FormalSobolShardPlan",
    "build_formal_sobol_grouped_shard",
    "main",
    "plan_formal_sobol_grouped_shard",
]
