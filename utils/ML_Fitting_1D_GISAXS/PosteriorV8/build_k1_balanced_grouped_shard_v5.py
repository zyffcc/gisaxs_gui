"""Build one immutable branch-pure shard for balanced all-K1 training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import socket
import stat
from typing import Sequence

import numpy as np

from .build_grouped_dataset_v5 import (
    V5GroupedRecipeSpec,
    build_v5_grouped_solution_dataset,
)
from .grouped_artifact_v5 import V5ArtifactReceipt, canonical_json
from .grouped_dataset_v5 import V5GroupedDataset, write_v5_grouped_dataset
from .k1_balanced_dataset_plan_v5 import (
    V5K1BalancedDatasetPlan,
    V5K1BalancedSobolBlock,
    v5_k1_balanced_dataset_plan_from_payload,
    validate_v5_k1_balanced_dataset_plan,
)
from .k1_forced_sobol_recipe_v5 import V5K1ForcedSobolCleanRecipe
from .sobol_design_v5 import materialize_v5_unit_coordinates_for_indices
from .sobol_recipe_coordinates_v5 import v5_sobol_recipe_design


V5_K1_BALANCED_GROUPED_SHARD_BUILDER_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_grouped_shard_builder/v2"
)
V5_K1_BALANCED_GROUPED_SHARD_BUILDER_VERSION = (
    "posterior_v8_v5_2_branch_pure_forced_sobol_grouped_shard_v2"
)
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _view_indices(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("view indices must be comma-separated integers") from exc
    else:
        selected = tuple(value)
    result = tuple(_nonnegative_integer(item, "view_index") for item in selected)
    if not result or len(result) != len(set(result)):
        raise ValueError("view indices must be non-empty and unique")
    return result


def _load_authoring_plan(path: Path) -> V5K1BalancedDatasetPlan:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("balanced K1 authoring plan is not valid JSON") from exc
    return v5_k1_balanced_dataset_plan_from_payload(payload)


def _under_root(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("output must be absolute")
    allowed = root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"output must be under {allowed}") from exc
    if not relative.parts:
        raise ValueError("output must not be the allowed root")
    current = allowed
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("output must not traverse a symlink")
        if not current.exists():
            break
    resolved_parent = lexical.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(allowed):
        raise ValueError("output parent resolves outside the allowed root")
    return lexical


def _assert_build_host() -> None:
    if socket.gethostname().split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 grouped-dataset generation is forbidden on the login node")


@dataclass(frozen=True)
class V5K1BalancedGroupedShardPlan:
    dataset_plan: V5K1BalancedDatasetPlan
    block: V5K1BalancedSobolBlock
    split_offset: int
    requested_count: int
    selected_indices: tuple[int, ...]
    view_indices: tuple[int, ...]
    shard_index: int | None
    selection_mode: str
    selection_sha256: str

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_K1_BALANCED_GROUPED_SHARD_BUILDER_SCHEMA,
            "version": V5_K1_BALANCED_GROUPED_SHARD_BUILDER_VERSION,
            "balanced_dataset_plan_sha256": self.dataset_plan.sha256,
            "balanced_sobol_block": self.block.audit_payload(),
            "split_offset": self.split_offset,
            "requested_count": self.requested_count,
            "actual_count": len(self.selected_indices),
            "selected_sobol_indices": list(self.selected_indices),
            "view_indices": list(self.view_indices),
            "selection_mode": self.selection_mode,
            "shard_index": self.shard_index,
            "generating_candidate_only": True,
            "selection_sha256": self.selection_sha256,
        }


def plan_v5_k1_balanced_grouped_shard(
    *,
    dataset_plan: V5K1BalancedDatasetPlan,
    block_sha256: str,
    count: int,
    view_indices: Sequence[int] = (0,),
    start: int | None = None,
    shard_index: int | None = None,
) -> V5K1BalancedGroupedShardPlan:
    """Resolve one branch-local shard without simulating any curves."""

    plan = validate_v5_k1_balanced_dataset_plan(dataset_plan)
    matches = tuple(value for value in plan.blocks if value.block_sha256 == block_sha256)
    if len(matches) != 1:
        raise ValueError("block_sha256 must select exactly one balanced K1 block")
    if (start is None) == (shard_index is None):
        raise ValueError("supply exactly one of start or shard_index")
    selected_count = _positive_integer(count, "count")
    if start is not None:
        offset = _nonnegative_integer(start, "start")
        selected_shard = None
        mode = "start_count"
    else:
        selected_shard = _nonnegative_integer(shard_index, "shard_index")
        offset = selected_shard * selected_count
        mode = "shard_index"
    block = matches[0]
    if offset >= block.parent_count:
        raise ValueError("requested shard starts beyond the selected K1 block")
    stop = min(
        block.sobol_index_start + offset + selected_count,
        block.sobol_index_start + block.parent_count,
    )
    indices = tuple(range(block.sobol_index_start + offset, stop))
    views = _view_indices(view_indices)
    core = {
        "balanced_dataset_plan_sha256": plan.sha256,
        "balanced_sobol_block_sha256": block.block_sha256,
        "split_offset": offset,
        "requested_count": selected_count,
        "actual_count": len(indices),
        "selected_sobol_indices": list(indices),
        "view_indices": list(views),
        "selection_mode": mode,
        "shard_index": selected_shard,
        "generating_candidate_only": True,
    }
    selection_sha = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return V5K1BalancedGroupedShardPlan(
        dataset_plan=plan,
        block=block,
        split_offset=offset,
        requested_count=selected_count,
        selected_indices=indices,
        view_indices=views,
        shard_index=selected_shard,
        selection_mode=mode,
        selection_sha256=selection_sha,
    )


def validate_v5_k1_balanced_grouped_shard_plan(
    shard_plan: V5K1BalancedGroupedShardPlan,
) -> V5K1BalancedGroupedShardPlan:
    if not isinstance(shard_plan, V5K1BalancedGroupedShardPlan):
        raise TypeError("shard_plan must be V5K1BalancedGroupedShardPlan")
    if shard_plan.selection_mode == "start_count":
        selector = {"start": shard_plan.split_offset, "shard_index": None}
    elif shard_plan.selection_mode == "shard_index":
        selector = {"start": None, "shard_index": shard_plan.shard_index}
    else:
        raise ValueError("unsupported K1 balanced shard selection mode")
    replay = plan_v5_k1_balanced_grouped_shard(
        dataset_plan=shard_plan.dataset_plan,
        block_sha256=shard_plan.block.block_sha256,
        count=shard_plan.requested_count,
        view_indices=shard_plan.view_indices,
        **selector,
    )
    if replay != shard_plan:
        raise ValueError("K1 balanced grouped-shard plan identity does not reproduce")
    return shard_plan


def materialize_v5_k1_grouped_recipe_specs(
    shard_plan: V5K1BalancedGroupedShardPlan,
) -> tuple[V5GroupedRecipeSpec, ...]:
    """Materialize recipe/spec rows but not observations, curves, or oracle searches."""

    shard_plan = validate_v5_k1_balanced_grouped_shard_plan(shard_plan)
    design = v5_sobol_recipe_design(scramble_seed=shard_plan.block.scramble_seed)
    if design.sha256 != shard_plan.block.sobol_design_sha256:
        raise ValueError("selected K1 block Sobol design does not reproduce")
    coordinates = materialize_v5_unit_coordinates_for_indices(
        design,
        shard_plan.selected_indices,
    )
    recipes = tuple(
        V5K1ForcedSobolCleanRecipe.create(
            plan=shard_plan.dataset_plan,
            block=shard_plan.block,
            sobol_index=index,
            original_unit_coordinates=point,
        )
        for index, point in zip(shard_plan.selected_indices, coordinates, strict=True)
    )
    return tuple(
        V5GroupedRecipeSpec(
            recipe=recipe,
            split_id=recipe.assigned_split,
            view_indices=shard_plan.view_indices,
            clean_group_id=recipe.clean_group_id,
            sobol_index=recipe.sobol_index,
            split_plan_sha256=recipe.plan.sha256,
            sobol_design_sha256=recipe.sobol_design_sha256,
        )
        for recipe in recipes
    )


def build_v5_k1_balanced_grouped_shard(
    shard_plan: V5K1BalancedGroupedShardPlan,
    output: str | os.PathLike[str],
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> tuple[V5GroupedDataset, V5ArtifactReceipt]:
    """Materialize and exclusively publish one immutable branch-pure shard."""

    _assert_build_host()
    target = _under_root(Path(output), allowed_root)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    specs = materialize_v5_k1_grouped_recipe_specs(shard_plan)
    dataset_id = f"balanced-k1-v5-{shard_plan.selection_sha256}"
    dataset = build_v5_grouped_solution_dataset(
        specs,
        dataset_id=dataset_id,
        generating_only=True,
        shard_selection=None,
    )
    receipt = write_v5_grouped_dataset(dataset, target)
    target.chmod(0o400)
    metadata = target.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("published K1 grouped shard is not 0400/nlink1")
    return dataset, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-plan", required=True, type=Path)
    parser.add_argument("--block-sha256", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--start", type=int)
    selector.add_argument("--shard-index", type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--view-indices", default="0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_plan = _load_authoring_plan(args.balanced_plan)
    shard_plan = plan_v5_k1_balanced_grouped_shard(
        dataset_plan=dataset_plan,
        block_sha256=args.block_sha256,
        start=args.start,
        shard_index=args.shard_index,
        count=args.count,
        view_indices=_view_indices(args.view_indices),
    )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")
    result = {**shard_plan.audit_payload(), "output": str(args.output), "dry_run": args.dry_run}
    if not args.dry_run:
        _, receipt = build_v5_k1_balanced_grouped_shard(shard_plan, args.output)
        result.update(
            {
                "artifact_sha256": receipt.artifact_sha256,
                "manifest_sha256": receipt.manifest_sha256,
                "byte_count": receipt.byte_count,
                "mode_octal": "0400",
                "nlink": 1,
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_BALANCED_GROUPED_SHARD_BUILDER_SCHEMA",
    "V5_K1_BALANCED_GROUPED_SHARD_BUILDER_VERSION",
    "V5K1BalancedGroupedShardPlan",
    "build_v5_k1_balanced_grouped_shard",
    "main",
    "materialize_v5_k1_grouped_recipe_specs",
    "plan_v5_k1_balanced_grouped_shard",
    "validate_v5_k1_balanced_grouped_shard_plan",
]
