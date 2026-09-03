"""Small-pilot workload accounting for frozen V5.1 exact-search arrays."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contract import NUM_TOPOLOGIES, topology_from_id
from .grouped_artifact_v5 import canonical_json
from .sobol_design_v5 import V5SobolDesign
from .sobol_recipe_coordinates_v5 import V5_SOBOL_RECIPE_COORDINATE_INDEX
from .sobol_universal_query_design_v5 import (
    materialize_v5_sobol_universal_topology_query_design,
)


V5_FROZEN_SEARCH_WORKER_WALLTIME_SECONDS = 24 * 60 * 60
V5_FROZEN_SEARCH_MAX_PREDICTED_SECONDS = 20 * 60 * 60
V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT = 8
V5_FROZEN_SEARCH_PILOT_TOPOLOGY_IDS = tuple(
    topology_id
    for topology_id in range(NUM_TOPOLOGIES)
    if len(topology_from_id(topology_id)) == 1
)


def _positive_float(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= minimum:
        raise ValueError(f"{name} must be finite and greater than {minimum}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, kw_only=True)
class V5FrozenSearchPilotThroughput:
    source_id: str
    effective_seconds_per_exact_forward_call: float
    runtime_safety_factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "effective_seconds_per_exact_forward_call",
            _positive_float(
                self.effective_seconds_per_exact_forward_call,
                "effective_seconds_per_exact_forward_call",
            ),
        )
        object.__setattr__(
            self,
            "runtime_safety_factor",
            _positive_float(
                self.runtime_safety_factor,
                "runtime_safety_factor",
                minimum=1.0,
            ),
        )

    def predicted_seconds(self, exact_forward_calls: int) -> int:
        return math.ceil(
            exact_forward_calls
            * self.effective_seconds_per_exact_forward_call
            * self.runtime_safety_factor
        )

    def audit_payload(self, *, exact_forward_calls_per_branch: int) -> dict[str, object]:
        return {
            "scope": "K1_throughput_pilot_only",
            "pilot_throughput_source_id": self.source_id,
            "pilot_effective_seconds_per_exact_forward_call": (
                self.effective_seconds_per_exact_forward_call
            ),
            "runtime_safety_factor": self.runtime_safety_factor,
            "prediction_formula": (
                "ceil(shard_exact_forward_calls * "
                "pilot_effective_seconds_per_exact_forward_call * "
                "runtime_safety_factor)"
            ),
            "exact_forward_calls_per_branch": exact_forward_calls_per_branch,
            "worker_walltime_seconds": V5_FROZEN_SEARCH_WORKER_WALLTIME_SECONDS,
            "maximum_permitted_predicted_seconds": (
                V5_FROZEN_SEARCH_MAX_PREDICTED_SECONDS
            ),
            "input_is_not_a_paper_scale_runtime_claim": True,
        }


def validate_v5_frozen_search_pilot_scope(
    *,
    train_recipes: int,
    validation_recipes: int,
    recipes_per_shard: int,
    selected_topology_ids: tuple[int, ...],
) -> None:
    if train_recipes > V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT or (
        validation_recipes > V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT
    ):
        raise ValueError(
            "this non-resumable K1 throughput pipeline is limited to "
            f"{V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT} recipes per split"
        )
    if recipes_per_shard != 1:
        raise ValueError(
            "the non-resumable throughput pilot requires exactly one recipe per shard"
        )
    if selected_topology_ids != V5_FROZEN_SEARCH_PILOT_TOPOLOGY_IDS:
        raise ValueError(
            "this launcher is restricted to the complete K1 topology set for its "
            "first throughput pilot"
        )


def point_workloads_v5(
    points: Sequence[object],
    *,
    design: V5SobolDesign,
    selected_topology_ids: tuple[int, ...],
    view_indices: Sequence[int],
    exact_forward_calls_per_branch: int,
) -> list[dict[str, object]]:
    rows = []
    for point in points:
        query_design = materialize_v5_sobol_universal_topology_query_design(
            point,
            design,
            selected_topology_ids=selected_topology_ids,
        )
        branch_count = sum(
            len(value.feasible_wire_pattern_ids)
            for value in query_design.topology_queries
        )
        observation_queries = [
            {
                "sobol_index": point.sobol_index,
                "clean_group_id": point.clean_group_id,
                "view_index": view_index,
                "branch_count": branch_count,
                "exact_forward_calls_per_branch": exact_forward_calls_per_branch,
                "exact_forward_calls_total": (
                    branch_count * exact_forward_calls_per_branch
                ),
            }
            for view_index in view_indices
        ]
        rows.append(
            {
                "observation_query_count": len(observation_queries),
                "branch_count_total": branch_count * len(observation_queries),
                "exact_forward_calls_total": sum(
                    int(value["exact_forward_calls_total"])
                    for value in observation_queries
                ),
                "observation_queries": observation_queries,
            }
        )
    return rows


def _generating_topology_id(point) -> int:
    coordinate = point.unit_coordinates[
        V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]
    ]
    return min(int(coordinate * NUM_TOPOLOGIES), NUM_TOPOLOGIES - 1)


def shard_windows_v5(
    *,
    split: str,
    points: Sequence[object],
    point_workloads: Sequence[Mapping[str, object]],
    split_start: int,
    shard_size: int,
    output_directory: Path,
    throughput: V5FrozenSearchPilotThroughput,
) -> list[dict[str, object]]:
    if len(points) != len(point_workloads):
        raise ValueError("point workloads must align with the frozen Sobol points")
    rows = []
    for task_index in range(math.ceil(len(points) / shard_size)):
        first = task_index * shard_size
        selected = points[first : first + shard_size]
        workloads = point_workloads[first : first + shard_size]
        identity = [
            {
                "sobol_index": value.sobol_index,
                "clean_group_id": value.clean_group_id,
                "generating_topology_id": _generating_topology_id(value),
            }
            for value in selected
        ]
        exact_calls = sum(
            int(value["exact_forward_calls_total"]) for value in workloads
        )
        rows.append(
            {
                "array_task_id": task_index,
                "split_offset": split_start + first,
                "recipe_count": len(selected),
                "output_root": str(output_directory / f"{split}-shard-{task_index:06d}"),
                "point_identity_sha256": sha256(
                    canonical_json(identity).encode("utf-8")
                ).hexdigest(),
                "workload": {
                    "observation_query_count": sum(
                        int(value["observation_query_count"]) for value in workloads
                    ),
                    "branch_count_total": sum(
                        int(value["branch_count_total"]) for value in workloads
                    ),
                    "exact_forward_calls_total": exact_calls,
                    "observation_queries": [
                        query
                        for value in workloads
                        for query in value["observation_queries"]
                    ],
                    "predicted_worker_seconds": throughput.predicted_seconds(
                        exact_calls
                    ),
                },
            }
        )
    return rows


def validate_v5_frozen_search_runtime_gate(
    windows: Sequence[Mapping[str, object]],
) -> int:
    predicted = [
        int(value["workload"]["predicted_worker_seconds"]) for value in windows
    ]
    maximum = max(predicted)
    if maximum >= V5_FROZEN_SEARCH_MAX_PREDICTED_SECONDS:
        raise ValueError(
            "predicted search shard runtime reaches the 20-hour safety ceiling "
            "inside the fixed 24-hour worker walltime"
        )
    return maximum


__all__ = [
    "V5FrozenSearchPilotThroughput",
    "V5_FROZEN_SEARCH_MAX_PILOT_RECIPES_PER_SPLIT",
    "V5_FROZEN_SEARCH_MAX_PREDICTED_SECONDS",
    "V5_FROZEN_SEARCH_PILOT_TOPOLOGY_IDS",
    "V5_FROZEN_SEARCH_WORKER_WALLTIME_SECONDS",
    "point_workloads_v5",
    "shard_windows_v5",
    "validate_v5_frozen_search_pilot_scope",
    "validate_v5_frozen_search_runtime_gate",
]
