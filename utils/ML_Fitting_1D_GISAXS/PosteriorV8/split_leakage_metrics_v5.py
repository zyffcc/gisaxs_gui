"""Exact-duplicate and nearest-neighbour metrics for V5 split audits."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .split_leakage_contract_v5 import V5LeakageAuditRow, ViolationCollector


def _array(parent: Mapping[str, object], name: str) -> np.ndarray:
    return np.asarray(parent[name], dtype=np.float64)


def nearest(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
) -> dict[str, object] | None:
    if not left or not right:
        return None
    left_matrix = np.stack(
        [np.concatenate([_array(item, name) for name in fields]) for item in left]
    )
    right_matrix = np.stack(
        [np.concatenate([_array(item, name) for name in fields]) for item in right]
    )
    distances, indices = cKDTree(right_matrix).query(left_matrix, k=1, workers=1)
    selected = int(np.argmin(distances))
    target = int(indices[selected])
    lhs, rhs = left[selected], right[target]
    rms = float(distances[selected] / np.sqrt(left_matrix.shape[1]))
    return {
        "rms_distance": rms,
        "left_group_id": lhs["clean_group_id"],
        "right_group_id": rhs["clean_group_id"],
        "same_topology": lhs["topology_id"] == rhs["topology_id"],
        "same_branch": lhs["branch_pattern_id"] == rhs["branch_pattern_id"],
        "shared_acquisition_policy": bool(set(lhs["policies"]) & set(rhs["policies"])),
    }


def nearest_by_category(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
    include_policy: bool,
) -> dict[str, object] | None:
    left_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    right_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for collection, target in ((left, left_groups), (right, right_groups)):
        for item in collection:
            base = (item["topology_id"], item["branch_pattern_id"])
            if include_policy:
                for policy in item["policies"]:
                    target[(*base, policy)].append(item)
            else:
                target[base].append(item)
    best = None
    for category in sorted(set(left_groups) & set(right_groups), key=str):
        candidate = nearest(left_groups[category], right_groups[category], fields=fields)
        if candidate is None:
            continue
        candidate["category"] = list(category)
        key = (
            candidate["rms_distance"],
            candidate["left_group_id"],
            candidate["right_group_id"],
        )
        if best is None or key < best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


def duplicate_summary(
    parents: Sequence[Mapping[str, object]],
    rows: Sequence[V5LeakageAuditRow],
    violations: ViolationCollector,
) -> dict[str, object]:
    physics: dict[tuple[object, ...], Mapping[str, object]] = {}
    query: dict[tuple[object, ...], Mapping[str, object]] = {}
    observation: dict[tuple[object, ...], V5LeakageAuditRow] = {}
    counts = Counter()
    for parent in parents:
        parameter_key = (
            parent["topology_id"],
            parent["branch_pattern_id"],
            tuple(parent["parameters"]),
        )
        context_key = (
            parent["topology_id"],
            parent["branch_pattern_id"],
            tuple(parent["query_bounds"]),
        )
        previous = physics.get(parameter_key)
        if previous is not None and previous["assigned_split"] != parent["assigned_split"]:
            counts["physics_exact_cross_split"] += 1
            violations.add(
                "physics_exact_cross_split",
                left_group_id=previous["clean_group_id"],
                right_group_id=parent["clean_group_id"],
            )
        else:
            physics[parameter_key] = parent
        previous = query.get(context_key)
        if previous is not None and previous["assigned_split"] != parent["assigned_split"]:
            counts["query_context_exact_cross_split"] += 1
        else:
            query[context_key] = parent
    for row in rows:
        key = (
            row.topology_id,
            row.branch_pattern_id,
            row.normalized_continuous_parameters,
            row.normalized_query_bounds,
            row.acquisition_policy_id,
        )
        previous = observation.get(key)
        if previous is not None and previous.assigned_split != row.assigned_split:
            counts["observation_exact_cross_split"] += 1
            violations.add(
                "observation_exact_cross_split",
                left_group_id=previous.clean_group_id,
                right_group_id=row.clean_group_id,
                acquisition_policy_id=row.acquisition_policy_id,
            )
        else:
            observation[key] = row
    return {
        "physics_exact_cross_split": counts["physics_exact_cross_split"],
        "query_context_exact_cross_split_report_only": counts["query_context_exact_cross_split"],
        "observation_exact_cross_split": counts["observation_exact_cross_split"],
        "query_context_duplicates_are_a_failure_by_themselves": False,
    }


__all__ = ["duplicate_summary", "nearest", "nearest_by_category"]
