"""Truth-independent factorial branch/bounds schedule for V4 recipes."""

from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral

import numpy as np

from .bounds_first_contract import BOUND_PLACEMENTS, RANGE_REGIMES
from .canonical_branch_catalog import CANONICAL_VALID_BRANCH_PATTERN_MASK
from .contract import NUM_TOPOLOGIES


BOUNDS_BRANCH_SCHEDULE_VERSION = "posterior_v8_topology_occurrence_factorial_v3"
TOPOLOGY_SCHEDULES = {
    "k1": tuple(range(3)),
    "k1_k2": tuple(range(9)),
    "all34": tuple(range(NUM_TOPOLOGIES)),
}


def _index(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("global_recipe_index must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("global_recipe_index must be non-negative")
    return result


def topology_id_for_index(schedule: str, global_recipe_index: int) -> int:
    try:
        topology_ids = TOPOLOGY_SCHEDULES[str(schedule).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown topology schedule {schedule!r}") from exc
    return topology_ids[_index(global_recipe_index) % len(topology_ids)]


def full_factorial_prefix_recipe_count(schedule: str) -> int:
    """Smallest start-at-zero prefix completing every scheduled topology.

    Topologies with fewer valid D/Resolution patterns repeat while the largest
    branch catalog finishes its 18 bounds combinations.  Counts refer to
    independent clean recipes, not correlated observation views.
    """

    try:
        topology_ids = TOPOLOGY_SCHEDULES[str(schedule).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown topology schedule {schedule!r}") from exc
    largest_pattern_count = max(
        int(np.count_nonzero(CANONICAL_VALID_BRANCH_PATTERN_MASK[topology_id]))
        for topology_id in topology_ids
    )
    return len(topology_ids) * len(RANGE_REGIMES) * len(BOUND_PLACEMENTS) * largest_pattern_count


def schedule_policy() -> dict[str, object]:
    return {
        "version": BOUNDS_BRANCH_SCHEDULE_VERSION,
        "inputs": ["topology_schedule", "global_recipe_index"],
        "truth_access": False,
        "topology_schedules": {
            name: list(topology_ids)
            for name, topology_ids in TOPOLOGY_SCHEDULES.items()
        },
        "minimum_start_zero_full_factorial_recipe_counts": {
            name: full_factorial_prefix_recipe_count(name)
            for name in TOPOLOGY_SCHEDULES
        },
        "topology_occurrence": "global_recipe_index // topology_schedule_length",
        "bounds_combinations": "3_range_regimes_x_6_bound_placements",
        "bounds_combo_permutation": "(5 * occurrence_mod_18 + 7 * topology_id) mod 18",
        "valid_branch_patterns": (
            "all permutation-quotiented topology-valid D/Resolution bit patterns"
        ),
        "factorial_pairing": "pattern_block_plus_5x_combo_base_plus_topology_id",
        "full_cycle_per_topology": "18 * valid_branch_pattern_count",
    }


SCHEDULE_SEMANTICS_SHA256 = sha256(
    json.dumps(schedule_policy(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def scheduled_branch_and_bounds(schedule: str, global_recipe_index: int):
    """Return topology occurrence, full-factorial pattern, and bounds condition."""

    name = str(schedule).strip().lower()
    try:
        topology_ids = TOPOLOGY_SCHEDULES[name]
    except KeyError as exc:
        raise ValueError(f"unknown topology schedule {schedule!r}") from exc
    index = _index(global_recipe_index)
    topology_id = topology_ids[index % len(topology_ids)]
    occurrence = index // len(topology_ids)
    valid_patterns = tuple(
        pattern
        for pattern, valid in enumerate(
            CANONICAL_VALID_BRANCH_PATTERN_MASK[topology_id]
        )
        if valid
    )
    combo_base = occurrence % 18
    pattern_block = (occurrence // 18) % len(valid_patterns)
    combo_id = (5 * combo_base + 7 * topology_id) % 18
    pattern_position = (
        pattern_block + 5 * combo_base + topology_id
    ) % len(valid_patterns)
    pattern_id = valid_patterns[pattern_position]
    regime = RANGE_REGIMES[combo_id // len(BOUND_PLACEMENTS)]
    placement = BOUND_PLACEMENTS[combo_id % len(BOUND_PLACEMENTS)]
    return topology_id, occurrence, pattern_id, combo_id, regime, placement


__all__ = [
    "BOUNDS_BRANCH_SCHEDULE_VERSION",
    "SCHEDULE_SEMANTICS_SHA256",
    "TOPOLOGY_SCHEDULES",
    "schedule_policy",
    "scheduled_branch_and_bounds",
    "full_factorial_prefix_recipe_count",
    "topology_id_for_index",
]
