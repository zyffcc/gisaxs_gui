"""Fail-closed assessment for the full K=1 Phase-C generalization gate."""

from __future__ import annotations

from collections import Counter
from math import fsum
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np

from .candidate_refinement_contract_v5 import V5_EXACT_FORWARD_BUDGET_UNIT
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_AMPLITUDE_AXIS_REGIMES,
    K1_PHASE_C_BASELINE_IDS,
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_BRANCH_BY_ID,
    K1_PHASE_C_BRANCH_CATALOG_SIZE,
    K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS,
    K1_PHASE_C_GEOMETRY_AXIS_REGIMES,
    K1_PHASE_C_METHOD_IDS,
    K1_PHASE_C_OBSERVATION_EFFECTS,
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_PRODUCT_METHOD_ID,
    K1_PHASE_C_RANGE_STRESS_STRATA,
    K1_PHASE_C_REFERENCE_EXACT_BUDGET,
    K1_PHASE_C_REFERENCE_OUTPUT_CAP,
    K1_PHASE_C_SPLIT_ID,
    K1_PHASE_C_TOP_K,
    digest,
    validate_v5_k1_phase_c_contract,
    v5_k1_phase_c_contract_payload,
)
from .k1_phase_c_plan_v5 import V5K1PhaseCPlan, validate_v5_k1_phase_c_plan
from .k1_phase_c_records_v5 import V5K1PhaseCMethodAudit, V5K1PhaseCParentRecord
from .paper_budget_evaluator_v5 import (
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
)
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_unit(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _method_reasons(
    value: object,
    *,
    parent: V5K1PhaseCParentRecord,
) -> list[str]:
    if not isinstance(value, V5K1PhaseCMethodAudit):
        return ["method_audit_has_wrong_type"]
    reasons = []
    if value.method_id not in K1_PHASE_C_METHOD_IDS:
        reasons.append("method_id_is_not_frozen")
    for name in (
        "evaluator_config_sha256",
        "comparison_protocol_sha256",
        "exact_judge_sha256",
        "reference_set_sha256",
        "trace_artifact_sha256",
        "trace_ledger_sha256",
    ):
        try:
            digest(getattr(value, name), name)
        except ValueError:
            reasons.append(f"{name}_is_invalid")
    if (
        value.evaluator_schema != V5_PAPER_BUDGET_EVALUATOR_SCHEMA
        or value.evaluator_version != V5_PAPER_BUDGET_EVALUATOR_VERSION
    ):
        reasons.append("paper_budget_evaluator_identity_drifted")
    if value.reference_set_sha256 != parent.reference_set_sha256:
        reasons.append("method_reference_set_does_not_match_parent")
    if value.exact_forward_budget_unit != V5_EXACT_FORWARD_BUDGET_UNIT:
        reasons.append("exact_forward_budget_unit_drifted")
    if value.exact_forward_budgets != EXACT_FORWARD_BUDGETS:
        reasons.append("exact_forward_budget_schedule_drifted")
    if value.output_cap != K1_PHASE_C_REFERENCE_OUTPUT_CAP:
        reasons.append("primary_output_cap_is_not_16")
    if value.configured_exact_call_budget != K1_PHASE_C_REFERENCE_EXACT_BUDGET:
        reasons.append("configured_exact_call_budget_is_not_4096")
    if value.complete_contiguous_trace is not True:
        reasons.append("exact_call_trace_is_partial")
    try:
        _finite_unit(value.primary_log2_budget_auc, "primary_log2_budget_auc")
    except (TypeError, ValueError):
        reasons.append("primary_log2_budget_auc_is_invalid")
    return reasons


def _range_reasons(record: V5K1PhaseCParentRecord) -> list[str]:
    reasons = []
    try:
        geometry_count = _integer(record.geometry_axis_count, "geometry_axis_count", minimum=1)
        amplitude_count = _integer(record.amplitude_axis_count, "amplitude_axis_count", minimum=1)
    except (TypeError, ValueError):
        return ["geometry_or_amplitude_axis_count_is_invalid"]
    geometry_regimes = tuple(record.geometry_axis_regimes)
    geometry_placements = tuple(record.geometry_axis_placements)
    geometry_ids = tuple(record.geometry_axis_coordinate_sha256s)
    amplitude_regimes = tuple(record.amplitude_axis_regimes)
    amplitude_ids = tuple(record.amplitude_axis_coordinate_sha256s)
    if not (
        len(geometry_regimes) == len(geometry_placements) == len(geometry_ids) == geometry_count
        and len(amplitude_regimes) == len(amplitude_ids) == amplitude_count
    ):
        reasons.append("geometry_or_amplitude_axis_audit_lengths_do_not_match")
        return reasons
    if any(value not in K1_PHASE_C_GEOMETRY_AXIS_REGIMES for value in geometry_regimes):
        reasons.append("geometry_axis_regime_is_not_frozen")
    if any(value not in K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS for value in geometry_placements):
        reasons.append("geometry_axis_placement_is_not_frozen")
    if any(value not in K1_PHASE_C_AMPLITUDE_AXIS_REGIMES for value in amplitude_regimes):
        reasons.append("amplitude_axis_regime_is_not_frozen")
    coordinate_ids = geometry_ids + amplitude_ids
    for value in coordinate_ids:
        try:
            digest(value, "range_axis_coordinate_sha256")
        except ValueError:
            reasons.append("range_axis_coordinate_sha256_is_invalid")
    if len(set(coordinate_ids)) != len(coordinate_ids):
        reasons.append("range_axes_reuse_one_coordinate")
    if record.axis_independent_range_coordinates is not True:
        reasons.append("range_axes_are_not_independently_derived")
    if record.range_generated_before_truth is not True:
        reasons.append("range_generation_is_not_truth_independent")

    edge_placements = ("edge_low", "edge_high")
    amplitude_edges = ("edge_low", "edge_high")
    geometry_families = tuple(
        "edge" if placement in edge_placements else regime
        for regime, placement in zip(geometry_regimes, geometry_placements)
    )
    amplitude_families = tuple(
        "edge" if regime in amplitude_edges else regime for regime in amplitude_regimes
    )
    families = geometry_families + amplitude_families
    stratum = record.range_stress_stratum
    if stratum not in K1_PHASE_C_RANGE_STRESS_STRATA:
        reasons.append("range_stress_stratum_is_not_frozen")
    elif stratum == "full" and (
        any(value != "full" for value in geometry_regimes + amplitude_regimes)
        or any(value != "interior" for value in geometry_placements)
    ):
        reasons.append("full_range_stratum_semantics_failed")
    elif stratum == "narrow" and (
        any(value != "narrow" for value in geometry_regimes + amplitude_regimes)
        or any(value in edge_placements for value in geometry_placements)
    ):
        reasons.append("narrow_range_stratum_semantics_failed")
    elif stratum == "fixed" and (
        any(value != "fixed" for value in geometry_regimes + amplitude_regimes)
        or any(value in edge_placements for value in geometry_placements)
    ):
        reasons.append("fixed_range_stratum_semantics_failed")
    elif stratum == "edge" and "edge" not in families:
        reasons.append("edge_range_stratum_semantics_failed")
    elif stratum == "mixed" and len(set(families)) < 2:
        reasons.append("mixed_range_stratum_semantics_failed")
    return reasons


def _expected_stress_cell(sobol_index: int) -> tuple[str, str]:
    return (
        K1_PHASE_C_RANGE_STRESS_STRATA[sobol_index % len(K1_PHASE_C_RANGE_STRESS_STRATA)],
        K1_PHASE_C_OBSERVATION_STRESS_STRATA[
            (sobol_index // len(K1_PHASE_C_RANGE_STRESS_STRATA))
            % len(K1_PHASE_C_OBSERVATION_STRESS_STRATA)
        ],
    )


def _record_reasons(
    record: V5K1PhaseCParentRecord,
    *,
    plan: V5K1PhaseCPlan,
    protocol_sha256: str,
    coordinate_contract: Mapping[str, object],
    range_contract: Mapping[str, object],
) -> list[str]:
    reasons = []
    for name in (
        "clean_parent_sha256",
        "universal_query_sha256",
        "protocol_sha256",
        "model_artifact_sha256",
        "source_bundle_sha256",
        "reference_set_sha256",
        "split_disjointness_receipt_sha256",
        "range_coordinate_contract_sha256",
        "sobol_block_sha256",
        "sobol_design_sha256",
    ):
        try:
            digest(getattr(record, name), name)
        except ValueError:
            reasons.append(f"{name}_is_invalid")
    if record.protocol_sha256 != protocol_sha256:
        reasons.append("study_protocol_identity_drifted")
    if record.split_id != K1_PHASE_C_SPLIT_ID:
        reasons.append("clean_parent_split_is_not_the_independent_phase_c_split")
    if record.split_disjointness_verified is not True:
        reasons.append("clean_parent_split_disjointness_is_not_verified")
    coordinate_identity = (
        record.range_coordinate_contract_schema,
        record.range_coordinate_contract_version,
        record.range_coordinate_contract_sha256,
        record.range_coordinate_contract_dimension,
    )
    expected_coordinate_identity = (
        coordinate_contract["coordinate_schema"],
        coordinate_contract["coordinate_version"],
        coordinate_contract["coordinate_contract_sha256"],
        coordinate_contract["coordinate_dimension"],
    )
    if coordinate_identity != expected_coordinate_identity:
        reasons.append("range_coordinate_contract_identity_drifted")
    if (
        record.amplitude_range_assignment_schema,
        record.amplitude_range_assignment_version,
    ) != (
        range_contract["amplitude_range_assignment_schema"],
        range_contract["amplitude_range_assignment_version"],
    ):
        reasons.append("amplitude_range_assignment_identity_drifted")
    branch = K1_PHASE_C_BRANCH_BY_ID.get(record.generating_branch_id)
    if branch is None:
        reasons.append("generating_branch_is_not_in_full_k1_catalog")
        block = None
    else:
        block = next(value for value in plan.sobol_blocks if value.branch_id == branch.branch_id)
        if (record.topology_id, record.pattern_id) != (branch.topology_id, branch.pattern_id):
            reasons.append("generating_branch_wire_identity_drifted")
        if (
            record.sobol_block_sha256 != block.block_sha256
            or record.sobol_design_sha256 != block.sobol_design_sha256
        ):
            reasons.append("clean_parent_sobol_block_identity_drifted")
    try:
        sobol_index = _integer(record.sobol_index, "sobol_index")
    except (TypeError, ValueError):
        reasons.append("sobol_index_is_invalid")
        sobol_index = -1
    if block is not None and not (
        block.sobol_index_start <= sobol_index < block.sobol_index_start + block.parent_count
    ):
        reasons.append("sobol_index_is_outside_generating_branch_block")
    if sobol_index >= 0 and (
        record.range_stress_stratum,
        record.observation_stress_stratum,
    ) != _expected_stress_cell(sobol_index):
        reasons.append("stress_cell_does_not_match_frozen_index_assignment")
    reasons.extend(_range_reasons(record))
    expected_effects = K1_PHASE_C_OBSERVATION_EFFECTS.get(record.observation_stress_stratum)
    if expected_effects is None:
        reasons.append("observation_stress_stratum_is_not_frozen")
    elif tuple(record.observation_effects) != expected_effects:
        reasons.append("observation_effects_do_not_match_one_factor_stratum")
    if record.observation_generated_before_curve is not True:
        reasons.append("observation_design_is_not_curve_independent")

    count_fields = (
        "branch_catalog_size",
        "completed_branch_outcomes",
        "unverified_branch_outcomes",
        "positive_branch_count",
        "matched_positive_branches_at_4",
        "reference_representative_count",
        "matched_reference_representative_count_at_n16_b4096",
        "verified_candidate_count",
        "bounds_compliant_candidate_count",
        "physics_compliant_candidate_count",
        "amplitude_compliant_candidate_count",
        "compatible_candidate_count_before_dedup_at_n16_b4096",
        "duplicate_candidate_count_at_n16_b4096",
    )
    counts = {}
    for name in count_fields:
        try:
            counts[name] = _integer(getattr(record, name), name)
        except (TypeError, ValueError):
            reasons.append(f"{name}_is_invalid")
    if len(counts) == len(count_fields):
        if (
            counts["branch_catalog_size"] != K1_PHASE_C_BRANCH_CATALOG_SIZE
            or counts["completed_branch_outcomes"] != K1_PHASE_C_BRANCH_CATALOG_SIZE
            or counts["unverified_branch_outcomes"] != 0
        ):
            reasons.append("all_twelve_branch_outcomes_are_not_complete")
        positive = counts["positive_branch_count"]
        matched = counts["matched_positive_branches_at_4"]
        if not 1 <= positive <= K1_PHASE_C_BRANCH_CATALOG_SIZE or not 0 <= matched <= min(
            positive, K1_PHASE_C_TOP_K
        ):
            reasons.append("branch_recall_counts_are_invalid")
        references = counts["reference_representative_count"]
        reference_hits = counts["matched_reference_representative_count_at_n16_b4096"]
        if references < 1 or not 0 <= reference_hits <= min(
            references, K1_PHASE_C_REFERENCE_OUTPUT_CAP
        ):
            reasons.append("reference_recall_counts_are_invalid")
        verified = counts["verified_candidate_count"]
        for name in (
            "bounds_compliant_candidate_count",
            "physics_compliant_candidate_count",
            "amplitude_compliant_candidate_count",
        ):
            if counts[name] > verified:
                reasons.append(f"{name}_exceeds_verified_denominator")
        compatible = counts["compatible_candidate_count_before_dedup_at_n16_b4096"]
        duplicates = counts["duplicate_candidate_count_at_n16_b4096"]
        if duplicates > compatible:
            reasons.append("duplicate_count_exceeds_compatible_denominator")
        if record.any_exact_compatible is not (compatible > 0):
            reasons.append("any_exact_compatible_disagrees_with_candidate_count")
    if record.reference_bank_qualified is not True:
        reasons.append("reference_bank_is_not_qualified")
    if record.reference_bank_saturated is not True:
        reasons.append("reference_bank_is_not_saturated")

    methods = tuple(record.method_audits)
    if not all(isinstance(value, V5K1PhaseCMethodAudit) for value in methods):
        reasons.append("method_audits_contain_wrong_types")
    method_ids = tuple(
        value.method_id for value in methods if isinstance(value, V5K1PhaseCMethodAudit)
    )
    if len(methods) != len(K1_PHASE_C_METHOD_IDS) or set(method_ids) != set(K1_PHASE_C_METHOD_IDS):
        reasons.append("paired_method_set_is_incomplete_or_duplicated")
    for value in methods:
        reasons.extend(_method_reasons(value, parent=record))
    valid_methods = tuple(value for value in methods if isinstance(value, V5K1PhaseCMethodAudit))
    for name in ("evaluator_config_sha256", "comparison_protocol_sha256", "exact_judge_sha256"):
        if len({getattr(value, name) for value in valid_methods}) > 1:
            reasons.append(f"paired_methods_do_not_share_{name}")
    return reasons


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(fsum(values) / len(values))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _branch_metrics(
    records: Sequence[V5K1PhaseCParentRecord],
) -> dict[str, dict[str, object]]:
    result = {}
    for branch in K1_PHASE_C_BRANCHES:
        rows = [value for value in records if value.generating_branch_id == branch.branch_id]
        method_auc = {
            method_id: _mean(
                [
                    next(
                        audit.primary_log2_budget_auc
                        for audit in row.method_audits
                        if audit.method_id == method_id
                    )
                    for row in rows
                ]
            )
            for method_id in K1_PHASE_C_METHOD_IDS
        }
        result[branch.branch_id] = {
            "clean_parent_count": len(rows),
            "branch_recall_at_4": _mean(
                [row.matched_positive_branches_at_4 / row.positive_branch_count for row in rows]
            ),
            "any_exact_compatible_rate": _mean([float(row.any_exact_compatible) for row in rows]),
            "reference_recall_at_n16_b4096": _mean(
                [
                    row.matched_reference_representative_count_at_n16_b4096
                    / row.reference_representative_count
                    for row in rows
                ]
            ),
            "duplicate_rate": _ratio(
                sum(row.duplicate_candidate_count_at_n16_b4096 for row in rows),
                sum(row.compatible_candidate_count_before_dedup_at_n16_b4096 for row in rows),
            ),
            "primary_log2_budget_auc_by_method": method_auc,
            "product_minus_baseline_auc": {
                baseline: (
                    None
                    if method_auc[K1_PHASE_C_PRODUCT_METHOD_ID] is None
                    or method_auc[baseline] is None
                    else method_auc[K1_PHASE_C_PRODUCT_METHOD_ID] - method_auc[baseline]
                )
                for baseline in K1_PHASE_C_BASELINE_IDS
            },
        }
    return result


def _minimum(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return min(finite) if len(finite) == len(values) and finite else None


def _maximum(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return max(finite) if len(finite) == len(values) and finite else None


def _assess_v5_k1_phase_c_records(
    records: Sequence[V5K1PhaseCParentRecord],
    *,
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Internal common assessment after the caller's evidence role is established."""

    frozen = validate_v5_k1_phase_c_contract(
        v5_k1_phase_c_contract_payload() if contract is None else contract
    )
    validate_v5_k1_phase_c_plan(plan, contract=frozen)
    values = tuple(records)
    if not all(isinstance(value, V5K1PhaseCParentRecord) for value in values):
        raise TypeError("records must contain V5K1PhaseCParentRecord values")
    integrity_reasons = []
    if len(values) != plan.total_parent_count:
        integrity_reasons.append("clean_parent_count_does_not_match_plan")
    parent_ids = [value.clean_parent_sha256 for value in values]
    query_ids = [value.universal_query_sha256 for value in values]
    if len(parent_ids) != len(set(parent_ids)):
        integrity_reasons.append("clean_parent_sha256_values_are_not_unique")
    if len(query_ids) != len(set(query_ids)):
        integrity_reasons.append("universal_query_sha256_values_are_not_unique")
    if len({value.model_artifact_sha256 for value in values}) != (1 if values else 0):
        integrity_reasons.append("records_do_not_share_one_frozen_model_artifact")
    if len({value.source_bundle_sha256 for value in values}) != (1 if values else 0):
        integrity_reasons.append("records_do_not_share_one_frozen_source_bundle")
    if len({value.split_disjointness_receipt_sha256 for value in values}) != (1 if values else 0):
        integrity_reasons.append("records_do_not_share_one_split_disjointness_receipt")

    protocol_sha = str(frozen["protocol_binding"]["protocol_sha256"])
    coordinate_contract = frozen["clean_parent_design"]
    range_contract = frozen["range_stress"]
    per_parent_reasons = {
        f"{index}:{record.clean_parent_sha256}": _record_reasons(
            record,
            plan=plan,
            protocol_sha256=protocol_sha,
            coordinate_contract=coordinate_contract,
            range_contract=range_contract,
        )
        for index, record in enumerate(values)
    }
    if any(per_parent_reasons.values()):
        integrity_reasons.append("one_or_more_parent_records_are_invalid")

    coverage = {}
    for block in plan.sobol_blocks:
        rows = [value for value in values if value.generating_branch_id == block.branch_id]
        indices = [value.sobol_index for value in rows]
        expected_indices = list(
            range(block.sobol_index_start, block.sobol_index_start + block.parent_count)
        )
        cells = Counter(
            (value.range_stress_stratum, value.observation_stress_stratum) for value in rows
        )
        coverage[block.branch_id] = {
            "parent_count": len(rows),
            "sobol_indices_complete": sorted(indices) == expected_indices,
            "stress_cell_counts": [
                [range_name, observation_name, cells[(range_name, observation_name)]]
                for observation_name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
                for range_name in K1_PHASE_C_RANGE_STRESS_STRATA
            ],
            "all_25_stress_cells_present": len(cells) == 25 and all(cells.values()),
            "stress_cell_max_minus_min": (
                max(cells.values()) - min(cells.values()) if len(cells) == 25 else None
            ),
        }
        if sorted(indices) != expected_indices:
            integrity_reasons.append(f"{block.branch_id}:sobol_block_is_partial_or_duplicated")
        if (
            len(cells) != 25
            or not all(cells.values())
            or max(cells.values()) - min(cells.values()) > 1
        ):
            integrity_reasons.append(f"{block.branch_id}:stress_cells_are_not_balanced")

    valid = [
        record
        for index, record in enumerate(values)
        if not per_parent_reasons[f"{index}:{record.clean_parent_sha256}"]
    ]
    branches = _branch_metrics(valid)
    verified = sum(value.verified_candidate_count for value in valid)
    compliance = {
        "verified_candidate_denominator": verified,
        "bounds_compliance_fraction": _ratio(
            sum(value.bounds_compliant_candidate_count for value in valid), verified
        ),
        "physics_compliance_fraction": _ratio(
            sum(value.physics_compliant_candidate_count for value in valid), verified
        ),
        "amplitude_compliance_fraction": _ratio(
            sum(value.amplitude_compliant_candidate_count for value in valid), verified
        ),
    }
    metrics = {
        "clean_parent_count": len(valid),
        "minimum_generating_branch_macro_branch_recall_at_4": _minimum(
            [value["branch_recall_at_4"] for value in branches.values()]
        ),
        "minimum_generating_branch_any_exact_compatible_rate": _minimum(
            [value["any_exact_compatible_rate"] for value in branches.values()]
        ),
        "minimum_generating_branch_reference_recall_at_n16_b4096": _minimum(
            [value["reference_recall_at_n16_b4096"] for value in branches.values()]
        ),
        "bounds_compliance_fraction": compliance["bounds_compliance_fraction"],
        "physics_compliance_fraction": compliance["physics_compliance_fraction"],
        "amplitude_compliance_fraction": compliance["amplitude_compliance_fraction"],
        "maximum_generating_branch_duplicate_rate": _maximum(
            [value["duplicate_rate"] for value in branches.values()]
        ),
        "minimum_generating_branch_product_minus_each_baseline_auc": _minimum(
            [
                value["product_minus_baseline_auc"][baseline]
                for value in branches.values()
                for baseline in K1_PHASE_C_BASELINE_IDS
            ]
        ),
    }
    gates = frozen["gates"]
    metric_for_gate = {
        name: metrics[
            name.removesuffix("_gte").removesuffix("_lt").removesuffix("_eq").removesuffix("_gt")
        ]
        for name in gates
    }
    decisions = {}
    integrity_ok = not integrity_reasons
    for name, threshold_value in gates.items():
        observed = metric_for_gate[name]
        threshold = float(threshold_value)
        if name.endswith("_gte"):
            operator, verdict = "gte", observed is not None and observed >= threshold
        elif name.endswith("_lt"):
            operator, verdict = "lt", observed is not None and observed < threshold
        elif name.endswith("_eq"):
            operator, verdict = "eq", observed is not None and observed == threshold
        elif name.endswith("_gt"):
            operator, verdict = "gt", observed is not None and observed > threshold
        else:  # pragma: no cover - live contract validation freezes suffixes
            raise RuntimeError(f"unsupported K1 Phase-C gate operator: {name}")
        decisions[name] = {
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(integrity_ok and verdict),
        }
    numeric_pass = integrity_ok and all(value["passed"] for value in decisions.values())
    return {
        "schema": frozen["schema"],
        "version": frozen["version"],
        "scientific_role": frozen["scientific_role"],
        "plan_sha256": plan.sha256,
        "contract_sha256": frozen["contract_sha256"],
        "formal_plan": plan.formal,
        "statistical_unit": "independent_clean_parent_macro_with_worst_branch_safety_gate",
        "integrity_passed": integrity_ok,
        "integrity_failure_reasons": integrity_reasons,
        "per_parent_failure_reasons": per_parent_reasons,
        "branch_and_stress_coverage": coverage,
        "per_generating_branch_metrics": branches,
        "bounds_physics_amplitude_compliance": compliance,
        "metrics": metrics,
        "gate_decisions": decisions,
        "fixture_gate_passed": bool(not plan.formal and numeric_pass),
        "full_k1_phase_c_generalization_gate_passed": bool(plan.formal and numeric_pass),
        "final_paper_model_acceptance": False,
        "performance_claim_allowed_from_phase_c_alone": False,
    }


def assess_v5_k1_phase_c_records(
    records: Sequence[V5K1PhaseCParentRecord],
    *,
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assess non-claiming fixtures; formal claims require a sealed replay receipt."""

    if plan.formal:
        raise ValueError("formal K1 Phase-C assessment rejects manually assembled records")
    return _assess_v5_k1_phase_c_records(records, plan=plan, contract=contract)


def assess_v5_k1_phase_c_replay_receipt(
    receipt: object,
    *,
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assess only records minted by a byte-checked, freshly replayed runner."""

    from .k1_phase_c_replay_receipt_v5 import validate_v5_k1_phase_c_checked_receipt

    checked = validate_v5_k1_phase_c_checked_receipt(receipt)
    frozen = validate_v5_k1_phase_c_contract(
        v5_k1_phase_c_contract_payload() if contract is None else contract
    )
    validate_v5_k1_phase_c_plan(plan, contract=frozen)
    if (
        checked.plan_sha256 != plan.sha256
        or checked.contract_sha256 != frozen["contract_sha256"]
        or checked.formal is not plan.formal
    ):
        raise ValueError("checked replay receipt escaped the requested plan or contract")
    return _assess_v5_k1_phase_c_records(checked.records, plan=plan, contract=frozen)


__all__ = ["assess_v5_k1_phase_c_records", "assess_v5_k1_phase_c_replay_receipt"]
