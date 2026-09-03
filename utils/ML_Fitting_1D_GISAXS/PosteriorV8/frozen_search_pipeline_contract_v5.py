"""Immutable shard and topology-schedule planning for V5.1 search mining."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Sequence

import numpy as np

from .build_formal_sobol_grouped_shard_v5 import (
    V5FormalSobolShardPlan,
    plan_formal_sobol_grouped_shard,
)
from .contract import topology_from_id
from .grouped_artifact_v5 import canonical_json
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


V5_FROZEN_SEARCH_PIPELINE_SCHEMA = (
    "gisaxs.posterior_v8.frozen_cross_topology_search_pipeline/v2"
)
V5_FROZEN_SEARCH_PIPELINE_VERSION = (
    "posterior_v8_task_bound_pilot_and_formal_contract_smoke_nonpromotion_v2"
)
V5_TOPOLOGY_SEARCH_SCHEDULE_SCHEMA = (
    "gisaxs.posterior_v8.selected_topology_search_schedule/v1"
)
V5_TOPOLOGY_SEARCH_SCHEDULE_VERSION = (
    "posterior_v8_one_explicit_topology_subset_for_every_clean_parent_v1"
)
V5_SEARCH_PIPELINE_ALLOWED_SPLITS = ("train", "tuning_validation")
V5_SEARCH_PIPELINE_SCOPE = (
    "K1_E1_pipeline_pilot_only_no_paper_scale_or_performance_claim"
)
V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX = (
    "pipeline-pilot-not-full-training-"
)
V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX = (
    "formal-calibrated-contract-smoke-not-training-"
)
V5_SEARCH_PIPELINE_FORMAL_SCOPE = (
    "paper_full_calibrated_contract_smoke_only_not_training_eligible"
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer_list(value: str | Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = value.replace(":", ",").split(",")
        try:
            supplied = tuple(int(item.strip()) for item in raw if item.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated integers") from exc
    else:
        supplied = tuple(value)
    values = tuple(_integer(item, name) for item in supplied)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be non-empty and unique")
    return values


@dataclass(frozen=True, kw_only=True)
class V5SelectedTopologySearchSchedule:
    """One explicit topology subset applied to every point in a pilot shard."""

    schedule_id: str
    selected_topology_ids: tuple[int, ...]
    schema_version: str = V5_TOPOLOGY_SEARCH_SCHEDULE_SCHEMA
    version: str = V5_TOPOLOGY_SEARCH_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule_id", _text(self.schedule_id, "schedule_id"))
        selected = _integer_list(self.selected_topology_ids, "selected_topology_ids")
        for topology_id in selected:
            topology_from_id(topology_id)
        object.__setattr__(self, "selected_topology_ids", tuple(sorted(selected)))
        if self.schema_version != V5_TOPOLOGY_SEARCH_SCHEDULE_SCHEMA:
            raise ValueError("unsupported topology-search schedule schema")
        if self.version != V5_TOPOLOGY_SEARCH_SCHEDULE_VERSION:
            raise ValueError("unsupported topology-search schedule version")

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "schedule_id": self.schedule_id,
            "selected_topology_ids": list(self.selected_topology_ids),
            "selected_topologies": [
                list(topology_from_id(value)) for value in self.selected_topology_ids
            ],
            "same_subset_for_every_clean_parent": True,
            "generating_topology_must_be_in_subset": True,
            "every_feasible_wire_branch_within_subset_is_searched": True,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V5FrozenSearchShardPlan:
    grouped_plan: V5FormalSobolShardPlan
    topology_schedule: V5SelectedTopologySearchSchedule
    points: tuple[V5DesignPoint, ...]
    query_designs: tuple[V5SobolUniversalTopologyQueryDesign, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.grouped_plan, V5FormalSobolShardPlan):
            raise TypeError("grouped_plan must be V5FormalSobolShardPlan")
        if self.grouped_plan.generating_only:
            raise ValueError("frozen search parent must retain every source branch")
        if not isinstance(self.topology_schedule, V5SelectedTopologySearchSchedule):
            raise TypeError("topology_schedule has an invalid type")
        if tuple(value.sobol_index for value in self.points) != (
            self.grouped_plan.selected_indices
        ):
            raise ValueError("design points escaped the selected Sobol shard")
        if len(self.query_designs) != len(self.points):
            raise ValueError("every clean parent requires one topology-query design")
        for point, design in zip(self.points, self.query_designs):
            if (
                design.sobol_index != point.sobol_index
                or design.clean_group_id != point.clean_group_id
                or design.selected_topology_ids
                != self.topology_schedule.selected_topology_ids
            ):
                raise ValueError("topology-query design escaped its frozen parent/schedule")

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_FROZEN_SEARCH_PIPELINE_SCHEMA,
            "version": V5_FROZEN_SEARCH_PIPELINE_VERSION,
            "scientific_scope": V5_SEARCH_PIPELINE_SCOPE,
            "grouped_parent_plan": self.grouped_plan.audit_payload(),
            "topology_schedule": self.topology_schedule.audit_payload(),
            "topology_schedule_sha256": self.topology_schedule.sha256,
            "query_designs": [
                {
                    "sobol_index": value.sobol_index,
                    "clean_group_id": value.clean_group_id,
                    "query_design_sha256": value.sha256,
                    "design_point_sha256": value.design_point_sha256,
                }
                for value in self.query_designs
            ],
            "claim_limits": {
                "paper_scale_training_labels": False,
                "no_solution_certificate": False,
                "model_scores_used_by_search": False,
                "completed_negative_is_budget_limited": True,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def plan_v5_frozen_search_shard(
    *,
    plan: V5SplitPlan,
    design: V5SobolDesign,
    target_split: str,
    start: int | None,
    shard_index: int | None,
    count: int,
    view_indices: Sequence[int],
    topology_schedule: V5SelectedTopologySearchSchedule,
) -> V5FrozenSearchShardPlan:
    """Resolve a contiguous pilot shard and freeze every topology-query artifact."""

    if target_split not in V5_SEARCH_PIPELINE_ALLOWED_SPLITS:
        raise ValueError(
            f"target_split must be one of {V5_SEARCH_PIPELINE_ALLOWED_SPLITS}"
        )
    grouped = plan_formal_sobol_grouped_shard(
        plan=plan,
        design=design,
        target_split=target_split,
        start=start,
        shard_index=shard_index,
        count=count,
        view_indices=view_indices,
        generating_only=False,
    )
    points = materialize_v5_design_points_for_indices(
        plan, design, grouped.selected_indices
    )
    query_designs = tuple(
        materialize_v5_sobol_universal_topology_query_design(
            point,
            design,
            selected_topology_ids=topology_schedule.selected_topology_ids,
        )
        for point in points
    )
    return V5FrozenSearchShardPlan(grouped, topology_schedule, points, query_designs)


__all__ = [
    "V5FrozenSearchShardPlan",
    "V5SelectedTopologySearchSchedule",
    "V5_FROZEN_SEARCH_PIPELINE_SCHEMA",
    "V5_FROZEN_SEARCH_PIPELINE_VERSION",
    "V5_SEARCH_PIPELINE_ALLOWED_SPLITS",
    "V5_SEARCH_PIPELINE_FORMAL_SCOPE",
    "V5_SEARCH_PIPELINE_FORMAL_SIDECAR_PREFIX",
    "V5_SEARCH_PIPELINE_PILOT_SIDECAR_PREFIX",
    "V5_SEARCH_PIPELINE_SCOPE",
    "V5_TOPOLOGY_SEARCH_SCHEDULE_SCHEMA",
    "V5_TOPOLOGY_SEARCH_SCHEDULE_VERSION",
    "plan_v5_frozen_search_shard",
]
