"""Executable adapter for one authorized balanced all-K1 search task."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
)
from .frozen_search_pipeline_contract_v5 import (
    V5SelectedTopologySearchSchedule,
)
from .grouped_dataset_v5 import V5GroupedDataset
from .k1_balanced_full_search_authorization_v5 import (
    V5K1BalancedFullSearchTaskAuthorization,
    V5K1BalancedFullSearchTaskMembership,
    build_v5_k1_balanced_full_search_task_membership,
)
from .k1_balanced_full_search_parent_v5 import (
    V5K1BalancedFullSearchParentProjection,
)
from .k1_forced_sobol_recipe_v5 import V5K1PersistedForcedCleanRecipe
from .k1_forced_universal_query_v5 import V5K1ForcedUniversalQuerySet
from .universal_query_contract_v5 import V5TopologyQuery


@dataclass(frozen=True)
class V5K1BalancedFullSearchQueryDesign:
    """Compatibility view of one strictly replayed forced all-K1 query set."""

    sobol_index: int
    assigned_split: str
    clean_group_id: str
    sobol_design_sha256: str
    design_point_sha256: str
    generating_topology_id: int
    selected_topology_ids: tuple[int, ...]
    topology_queries: tuple[V5TopologyQuery, ...]
    canonical_json: str
    sha256: str

    @classmethod
    def from_replay(
        cls,
        recipe: V5K1PersistedForcedCleanRecipe,
        query_set: V5K1ForcedUniversalQuerySet,
    ) -> "V5K1BalancedFullSearchQueryDesign":
        if not isinstance(recipe, V5K1PersistedForcedCleanRecipe) or not isinstance(
            query_set, V5K1ForcedUniversalQuerySet
        ):
            raise TypeError("forced recipe/query-set replay types are invalid")
        if query_set.identity != recipe.identity:
            raise ValueError("forced query set escaped its persisted recipe")
        return cls(
            sobol_index=recipe.sobol_index,
            assigned_split=recipe.assigned_split,
            clean_group_id=recipe.clean_group_id,
            sobol_design_sha256=recipe.sobol_design_sha256,
            design_point_sha256=recipe.sha256,
            generating_topology_id=query_set.generating_topology_id,
            selected_topology_ids=query_set.selected_topology_ids,
            topology_queries=query_set.topology_queries,
            canonical_json=query_set.canonical_json,
            sha256=query_set.sha256,
        )

    def to_json(self) -> str:
        payload = json.loads(self.canonical_json)
        payload["artifact_sha256"] = self.sha256
        return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, eq=False)
class V5K1BalancedFullSearchExecutableTask:
    """Live, fully replayed task accepted by the frozen search pipeline."""

    projection: V5K1BalancedFullSearchParentProjection
    task_authorization: V5K1BalancedFullSearchTaskAuthorization
    topology_schedule: V5SelectedTopologySearchSchedule

    def __post_init__(self) -> None:
        if not isinstance(self.projection, V5K1BalancedFullSearchParentProjection):
            raise TypeError("projection has an invalid type")
        if not isinstance(
            self.task_authorization, V5K1BalancedFullSearchTaskAuthorization
        ):
            raise TypeError("task_authorization has an invalid type")
        if not isinstance(self.topology_schedule, V5SelectedTopologySearchSchedule):
            raise TypeError("topology_schedule has an invalid type")
        membership = self.membership
        replay = build_v5_k1_balanced_full_search_task_membership(
            self.projection,
            array_task_id=membership.array_task_id,
            shard_index=membership.shard_index,
            split_offset=membership.split_offset,
            source_selection_sha256=membership.source_selection_sha256,
        )
        if replay.to_payload() != membership.to_payload():
            raise ValueError("executable projection escaped its task authorization")
        if self.topology_schedule.selected_topology_ids != tuple(
            self.projection.query_sets[0].selected_topology_ids
        ) or any(
            value.selected_topology_ids != self.topology_schedule.selected_topology_ids
            for value in self.projection.query_sets
        ):
            raise ValueError("executable query sets escaped the K1 topology schedule")
        formal = self.formal_production_authorization
        if (
            formal.shard_plan_sha256 != self.sha256
            or formal.target_split != membership.split_id
            or formal.recipe_sobol_indices
            != tuple(value.sobol_index for value in membership.members)
            or formal.clean_group_ids
            != tuple(value.clean_group_id for value in membership.members)
        ):
            raise ValueError("executable task escaped its formal authorization")

    @property
    def membership(self) -> V5K1BalancedFullSearchTaskMembership:
        return V5K1BalancedFullSearchTaskMembership.from_payload(
            self.task_authorization.task_plan["membership"]
        )

    @property
    def formal_production_authorization(
        self,
    ) -> V5FormalProductionSearchAuthorization:
        return self.task_authorization.formal_authorization

    @property
    def parent_dataset(self) -> V5GroupedDataset:
        return self.projection.parent

    @property
    def recipes(self) -> tuple[V5K1PersistedForcedCleanRecipe, ...]:
        return self.projection.recipes

    @property
    def points(self):
        return self.membership.members

    @property
    def query_designs(self) -> tuple[V5K1BalancedFullSearchQueryDesign, ...]:
        return tuple(
            V5K1BalancedFullSearchQueryDesign.from_replay(recipe, query_set)
            for recipe, query_set in zip(
                self.projection.recipes,
                self.projection.query_sets,
                strict=True,
            )
        )

    @property
    def sha256(self) -> str:
        return self.task_authorization.task_plan_sha256

    @property
    def target_split(self) -> str:
        return self.membership.split_id

    def view_indices_for_recipe(self, recipe_index: int) -> tuple[int, ...]:
        try:
            selected = self.projection.observation_selections[recipe_index]
        except IndexError as exc:
            raise IndexError("projected recipe index is outside the task") from exc
        return (selected.selected_view_index,)


__all__ = [
    "V5K1BalancedFullSearchExecutableTask",
    "V5K1BalancedFullSearchQueryDesign",
]
