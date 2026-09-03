"""Fail-closed group, support, duplicate, and nearest-neighbour V5 audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Sequence

from .sobol_design_v5 import V5SobolDesign, v5_clean_group_id
from .split_design_v5 import MAIN_SPLITS, OOD_LABELS, V5SplitPlan
from .split_leakage_contract_v5 import (
    V5_LEAKAGE_AUDIT_SCHEMA,
    V5_LEAKAGE_AUDIT_VERSION,
    V5LeakageAudit,
    V5LeakageAuditConfig,
    V5LeakageAuditRow,
    V5LeakageViolation,
    V5OODDefinition,
    ViolationCollector,
    canonical_json,
    parent_signature,
    row_fingerprint,
    support_flags,
)
from .split_leakage_features_v5 import (
    V5_CLEAN_PHYSICS_LEAKAGE_DIM,
    V5_QUERY_BOUNDS_LEAKAGE_DIM,
)
from .split_leakage_metrics_v5 import duplicate_summary, nearest, nearest_by_category


def _validate_rows_against_design(
    records: Sequence[V5LeakageAuditRow],
    *,
    plan: V5SplitPlan,
    design: V5SobolDesign,
    ood_definition: V5OODDefinition,
    violations: ViolationCollector,
) -> dict[str, list[V5LeakageAuditRow]]:
    grouped: dict[str, list[V5LeakageAuditRow]] = defaultdict(list)
    for row in records:
        grouped[row.clean_group_id].append(row)
        try:
            expected_split = plan.split_for_index(row.design_index)
            expected_ood = plan.ood_label_for_index(row.design_index)
            expected_group = v5_clean_group_id(plan, design, row.design_index)
        except ValueError:
            violations.add("unassigned_or_guard_design_index", design_index=row.design_index)
            continue
        if (row.assigned_split, row.ood_label) != (expected_split, expected_ood):
            violations.add(
                "split_or_ood_label_disagrees_with_plan",
                clean_group_id=row.clean_group_id,
                design_index=row.design_index,
            )
        if row.clean_group_id != expected_group:
            violations.add(
                "clean_group_id_disagrees_with_design",
                clean_group_id=row.clean_group_id,
                design_index=row.design_index,
            )
        observed_support = support_flags(row, ood_definition)
        expected_support = {name: name == row.ood_label for name in OOD_LABELS}
        if observed_support != expected_support:
            violations.add(
                "ood_support_not_isolated_or_label_incorrect",
                clean_group_id=row.clean_group_id,
                view_index=row.view_index,
                expected=expected_support,
                observed=observed_support,
            )
    return grouped


def _collapse_clean_parents(
    grouped: dict[str, list[V5LeakageAuditRow]],
    violations: ViolationCollector,
) -> tuple[list[dict[str, object]], dict[int, str]]:
    parents = []
    index_owner = {}
    for group_id in sorted(grouped):
        group_rows = sorted(grouped[group_id], key=lambda value: value.view_index)
        first = group_rows[0]
        if any(parent_signature(value) != parent_signature(first) for value in group_rows[1:]):
            violations.add("views_do_not_share_clean_parent", clean_group_id=group_id)
        views = [value.view_index for value in group_rows]
        if len(views) != len(set(views)):
            violations.add("duplicate_view_index_within_parent", clean_group_id=group_id)
        previous_group = index_owner.get(first.design_index)
        if previous_group is not None and previous_group != group_id:
            violations.add(
                "multiple_clean_groups_share_design_index",
                design_index=first.design_index,
                left_group_id=previous_group,
                right_group_id=group_id,
            )
        index_owner[first.design_index] = group_id
        parents.append(
            {
                "clean_group_id": group_id,
                "design_index": first.design_index,
                "assigned_split": first.assigned_split,
                "ood_label": first.ood_label,
                "parameters": first.normalized_continuous_parameters,
                "query_bounds": first.normalized_query_bounds,
                "topology_id": first.topology_id,
                "branch_pattern_id": first.branch_pattern_id,
                "policies": tuple(sorted({value.acquisition_policy_id for value in group_rows})),
            }
        )
    return parents, index_owner


def _nearest_split_pairs(
    parents: Sequence[dict[str, object]],
    config: V5LeakageAuditConfig,
    violations: ViolationCollector,
) -> list[dict[str, object]]:
    split_parents = {
        name: [value for value in parents if value["assigned_split"] == name]
        for name in MAIN_SPLITS
    }
    records = []
    for left_index, left_name in enumerate(MAIN_SPLITS):
        for right_name in MAIN_SPLITS[left_index + 1 :]:
            left = split_parents[left_name]
            right = split_parents[right_name]
            record = {
                "left_split": left_name,
                "right_split": right_name,
                "parameter_unconditional": nearest(left, right, fields=("parameters",)),
                "query_bounds_unconditional": nearest(left, right, fields=("query_bounds",)),
                "joint_unconditional": nearest(left, right, fields=("parameters", "query_bounds")),
                "parameter_same_topology_branch": nearest_by_category(
                    left,
                    right,
                    fields=("parameters",),
                    include_policy=False,
                ),
                "joint_same_topology_branch_policy": nearest_by_category(
                    left,
                    right,
                    fields=("parameters", "query_bounds"),
                    include_policy=True,
                ),
            }
            physics_nearest = record["parameter_same_topology_branch"]
            if (
                physics_nearest is not None
                and physics_nearest["rms_distance"] <= config.physics_near_duplicate_rms_threshold
            ):
                violations.add(
                    "physics_near_duplicate_cross_split",
                    left_split=left_name,
                    right_split=right_name,
                    **physics_nearest,
                )
            observation_nearest = record["joint_same_topology_branch_policy"]
            if (
                observation_nearest is not None
                and observation_nearest["rms_distance"]
                <= config.observation_near_duplicate_rms_threshold
            ):
                violations.add(
                    "observation_near_duplicate_cross_split",
                    left_split=left_name,
                    right_split=right_name,
                    **observation_nearest,
                )
            records.append(record)
    return records


def audit_v5_split_leakage(
    rows: Sequence[V5LeakageAuditRow],
    *,
    plan: V5SplitPlan,
    design: V5SobolDesign,
    ood_definition: V5OODDefinition,
    config: V5LeakageAuditConfig = V5LeakageAuditConfig(),
    fail_on_violation: bool = True,
) -> V5LeakageAudit:
    """Audit one complete V5 design; failures carry their JSON-ready report."""

    if not isinstance(plan, V5SplitPlan) or not isinstance(design, V5SobolDesign):
        raise TypeError("plan/design have invalid types")
    if not isinstance(ood_definition, V5OODDefinition):
        raise TypeError("ood_definition must be V5OODDefinition")
    if not isinstance(config, V5LeakageAuditConfig):
        raise TypeError("config must be V5LeakageAuditConfig")
    records = tuple(rows)
    if not records or not all(isinstance(value, V5LeakageAuditRow) for value in records):
        raise ValueError("rows must contain V5LeakageAuditRow records")

    violations = ViolationCollector(config.maximum_reported_violations)
    grouped = _validate_rows_against_design(
        records,
        plan=plan,
        design=design,
        ood_definition=ood_definition,
        violations=violations,
    )
    parents, index_owner = _collapse_clean_parents(grouped, violations)
    expected_indices = set(plan.assigned_indices())
    observed_indices = set(index_owner)
    if config.require_complete_plan and observed_indices != expected_indices:
        violations.add(
            "incomplete_or_extra_design_indices",
            missing_count=len(expected_indices - observed_indices),
            extra_count=len(observed_indices - expected_indices),
        )

    duplicates = duplicate_summary(parents, records, violations)
    nearest_pairs = _nearest_split_pairs(parents, config, violations)
    fingerprints = sorted(
        (row_fingerprint(value) for value in records),
        key=lambda value: (
            value["design_index"],
            value["clean_group_id"],
            value["view_index"],
        ),
    )
    split_counts = Counter(value["assigned_split"] for value in parents)
    ood_counts = Counter(value["ood_label"] for value in parents if value["ood_label"])
    policy_counts = Counter(value.acquisition_policy_id for value in records)
    payload = {
        "schema": V5_LEAKAGE_AUDIT_SCHEMA,
        "version": V5_LEAKAGE_AUDIT_VERSION,
        "status": "passed" if violations.total == 0 else "failed",
        "plan": {
            "sha256": plan.sha256,
            "schema": json.loads(plan.canonical_json)["schema"],
            "guard_band": plan.guard_band,
            "prefix_stop": plan.prefix_stop,
        },
        "sobol_design": {**design.payload(), "sha256": design.sha256},
        "config": asdict(config),
        "ood_definition": asdict(ood_definition),
        "feature_contract": {
            "continuous_parameters": "canonical_global_physical_geometry_and_coefficients_33d",
            "continuous_parameter_dimension": V5_CLEAN_PHYSICS_LEAKAGE_DIM,
            "query_bounds": (
                "normalized_physical_GUI_geometry_bounds_78d_plus_"
                "absolute_reference_amplitude_bounds_21d"
            ),
            "query_bounds_dimension": V5_QUERY_BOUNDS_LEAKAGE_DIM,
            "categorical": ["topology_id", "branch_pattern_id", "acquisition_policy_id"],
            "nearest_distance": "RMS_Euclidean_in_reported_normalized_feature_block",
        },
        "population": {
            "row_count": len(records),
            "clean_group_count": len(parents),
            "split_clean_group_counts": {name: split_counts[name] for name in MAIN_SPLITS},
            "ood_clean_group_counts": {name: ood_counts[name] for name in OOD_LABELS},
            "acquisition_policy_row_counts": dict(sorted(policy_counts.items())),
            "row_fingerprint_sha256": sha256(
                canonical_json(fingerprints).encode("utf-8")
            ).hexdigest(),
        },
        "exact_duplicates": duplicates,
        "nearest_neighbours": {
            "split_pairs": nearest_pairs,
            "query_bound_nearness_is_reported_not_a_failure_by_itself": True,
            "categorical_coordinates_are_matched_not_encoded_as_ordinal_distance": True,
        },
        "violations": {
            "total_count": violations.total,
            "reported_count": len(violations.records),
            "truncated": violations.total > len(violations.records),
            "records": violations.records,
        },
        "claim_limits": {
            "nearest_neighbour_separation_proves_no_distribution_overlap": False,
            "guarded_sobol_indices_prove_physical_separation_without_audit": False,
            "ordinary_random_holdout_is_ood": False,
        },
    }
    audit = V5LeakageAudit.create(payload)
    if fail_on_violation and not audit.passed:
        raise V5LeakageViolation(audit)
    return audit


__all__ = [
    "V5_LEAKAGE_AUDIT_SCHEMA",
    "V5_LEAKAGE_AUDIT_VERSION",
    "V5LeakageAudit",
    "V5LeakageAuditConfig",
    "V5LeakageAuditRow",
    "V5LeakageViolation",
    "V5OODDefinition",
    "audit_v5_split_leakage",
]
