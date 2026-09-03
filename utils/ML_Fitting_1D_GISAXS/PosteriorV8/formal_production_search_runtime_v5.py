"""Executable replay of one preregistered formal-production search shard.

The object in this module is deliberately constructed only after the global
plan and all of its external artifacts have been checked by the worker.  It
adapts the explicit formal shard contract to the established frozen-search
pipeline without weakening the engineering-pilot path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .formal_production_search_contract_v5 import V5FormalProductionSearchShard
from .frozen_search_pipeline_contract_v5 import V5SelectedTopologySearchSchedule
from .sobol_design_v5 import (
    V5DesignPoint,
    V5SobolDesign,
    materialize_v5_design_points_for_indices,
)
from .sobol_universal_query_design_v5 import (
    V5SobolUniversalTopologyQueryDesign,
    materialize_v5_sobol_universal_topology_query_design,
)
from .split_design_v5 import V5SplitPlan


@dataclass(frozen=True)
class V5FormalProductionExecutableShard:
    """Fully replayed, curve-free input accepted by the frozen pipeline."""

    formal_shard: V5FormalProductionSearchShard
    split_plan: V5SplitPlan
    sobol_design: V5SobolDesign
    points: tuple[V5DesignPoint, ...]
    query_designs: tuple[V5SobolUniversalTopologyQueryDesign, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.formal_shard, V5FormalProductionSearchShard):
            raise TypeError("formal_shard has an invalid type")
        if not isinstance(self.split_plan, V5SplitPlan):
            raise TypeError("split_plan has an invalid type")
        if not isinstance(self.sobol_design, V5SobolDesign):
            raise TypeError("sobol_design has an invalid type")
        shard = self.formal_shard
        if (
            shard.split_plan_sha256 != self.split_plan.sha256
            or shard.sobol_design_sha256 != self.sobol_design.sha256
        ):
            raise ValueError("formal executable shard escaped its frozen designs")
        points = tuple(self.points)
        queries = tuple(self.query_designs)
        if len(points) != len(shard.recipes) or len(queries) != len(points):
            raise ValueError("formal executable shard has incomplete recipe replay")
        for member, point, query in zip(shard.recipes, points, queries):
            if (
                point.sobol_index != member.sobol_index
                or point.clean_group_id != member.clean_group_id
                or query.sobol_index != member.sobol_index
                or query.clean_group_id != member.clean_group_id
                or query.design_point_sha256 != member.design_point_sha256
                or query.sha256 != member.query_design_sha256
                or query.generating_topology_id != member.generating_topology_id
                or query.selected_topology_ids != shard.stage.selected_topology_ids
            ):
                raise ValueError("formal recipe/query membership does not replay")
            branch_count = sum(
                len(value.feasible_wire_pattern_ids)
                for value in query.topology_queries
            )
            if branch_count != member.branch_count:
                raise ValueError("formal recipe branch count does not replay")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "query_designs", queries)

    @classmethod
    def replay(
        cls,
        shard: V5FormalProductionSearchShard,
        *,
        split_plan: V5SplitPlan,
        sobol_design: V5SobolDesign,
    ) -> "V5FormalProductionExecutableShard":
        indices = tuple(value.sobol_index for value in shard.recipes)
        points = materialize_v5_design_points_for_indices(
            split_plan, sobol_design, indices
        )
        queries = tuple(
            materialize_v5_sobol_universal_topology_query_design(
                point,
                sobol_design,
                selected_topology_ids=shard.stage.selected_topology_ids,
            )
            for point in points
        )
        return cls(
            formal_shard=shard,
            split_plan=split_plan,
            sobol_design=sobol_design,
            points=points,
            query_designs=queries,
        )

    @property
    def topology_schedule(self) -> V5SelectedTopologySearchSchedule:
        return self.formal_shard.stage.topology_schedule

    @property
    def sha256(self) -> str:
        return self.formal_shard.sha256

    @property
    def target_split(self) -> str:
        return self.formal_shard.target_split

    def view_indices_for_recipe(self, recipe_index: int) -> tuple[int, ...]:
        """Return the one curve-blind, sigma-present view frozen for a recipe."""

        try:
            member = self.formal_shard.recipes[recipe_index]
        except IndexError as exc:
            raise IndexError("formal recipe index is outside the shard") from exc
        return (member.observation_selection.selected_view_index,)


__all__ = ["V5FormalProductionExecutableShard"]
