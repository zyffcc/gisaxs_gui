from __future__ import annotations

from dataclasses import replace

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_PRODUCT_METHOD_ID,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_evaluation_v5 import (
    assess_v5_k1_phase_c_records,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_fixture_v5 import (
    build_v5_k1_phase_c_fixture_records,
    run_v5_k1_phase_c_fixture_gate,
)


def test_small_fixture_exercises_every_gate_but_can_never_claim_formal_acceptance():
    result = run_v5_k1_phase_c_fixture_gate()

    assert result["integrity_passed"] is True
    assert result["fixture_gate_passed"] is True
    assert result["formal_plan"] is False
    assert result["full_k1_phase_c_generalization_gate_passed"] is False
    assert result["final_paper_model_acceptance"] is False
    assert result["performance_claim_allowed_from_phase_c_alone"] is False
    assert all(value["passed"] for value in result["gate_decisions"].values())
    assert len(result["per_generating_branch_metrics"]) == 12
    assert all(
        value["all_25_stress_cells_present"] is True and value["stress_cell_max_minus_min"] == 0
        for value in result["branch_and_stress_coverage"].values()
    )
    assert result["metrics"] == {
        "clean_parent_count": 300,
        "minimum_generating_branch_macro_branch_recall_at_4": 1.0,
        "minimum_generating_branch_any_exact_compatible_rate": 1.0,
        "minimum_generating_branch_reference_recall_at_n16_b4096": 1.0,
        "bounds_compliance_fraction": 1.0,
        "physics_compliance_fraction": 1.0,
        "amplitude_compliance_fraction": 1.0,
        "maximum_generating_branch_duplicate_rate": 0.0,
        "minimum_generating_branch_product_minus_each_baseline_auc": (
            result["metrics"]["minimum_generating_branch_product_minus_each_baseline_auc"]
        ),
    }
    assert result["metrics"]["minimum_generating_branch_product_minus_each_baseline_auc"] > 0.0


def test_fixture_preserves_geometry_placement_and_amplitude_edge_regime_semantics():
    _, records = build_v5_k1_phase_c_fixture_records()
    edge = next(value for value in records if value.range_stress_stratum == "edge")

    assert "edge_low" in edge.geometry_axis_placements
    assert "edge_high" in edge.geometry_axis_placements
    assert "edge_low" in edge.amplitude_axis_regimes
    assert "edge_high" in edge.amplitude_axis_regimes
    assert set(edge.geometry_axis_coordinate_sha256s).isdisjoint(
        edge.amplitude_axis_coordinate_sha256s
    )


def test_reusing_one_range_coordinate_fails_every_gate_closed():
    plan, records = build_v5_k1_phase_c_fixture_records()
    first = records[0]
    coordinate = first.geometry_axis_coordinate_sha256s[0]
    broken = replace(
        first,
        geometry_axis_coordinate_sha256s=(coordinate,) * first.geometry_axis_count,
    )
    result = assess_v5_k1_phase_c_records((broken, *records[1:]), plan=plan)

    assert result["integrity_passed"] is False
    assert result["fixture_gate_passed"] is False
    assert not any(value["passed"] for value in result["gate_decisions"].values())
    assert (
        "range_axes_reuse_one_coordinate"
        in result["per_parent_failure_reasons"][f"0:{first.clean_parent_sha256}"]
    )


def test_partial_or_unequal_exact_budget_method_trace_fails_closed():
    plan, records = build_v5_k1_phase_c_fixture_records()
    first = records[0]
    methods = list(first.method_audits)
    methods[0] = replace(
        methods[0],
        configured_exact_call_budget=4095,
        complete_contiguous_trace=False,
    )
    broken = replace(first, method_audits=tuple(methods))
    result = assess_v5_k1_phase_c_records((broken, *records[1:]), plan=plan)

    assert result["fixture_gate_passed"] is False
    reasons = result["per_parent_failure_reasons"][f"0:{first.clean_parent_sha256}"]
    assert "configured_exact_call_budget_is_not_4096" in reasons
    assert "exact_call_trace_is_partial" in reasons


def test_split_receipt_or_split_role_drift_fails_closed():
    plan, records = build_v5_k1_phase_c_fixture_records()
    first = records[0]
    broken = replace(first, split_id="training")
    result = assess_v5_k1_phase_c_records((broken, *records[1:]), plan=plan)

    assert result["fixture_gate_passed"] is False
    assert (
        "clean_parent_split_is_not_the_independent_phase_c_split"
        in result["per_parent_failure_reasons"][f"0:{first.clean_parent_sha256}"]
    )


def test_amplitude_assignment_or_named_coordinate_contract_drift_fails_closed():
    plan, records = build_v5_k1_phase_c_fixture_records()
    first = records[0]
    broken = replace(
        first,
        amplitude_range_assignment_version="drifted",
        range_coordinate_contract_sha256="0" * 64,
    )
    result = assess_v5_k1_phase_c_records((broken, *records[1:]), plan=plan)

    assert result["fixture_gate_passed"] is False
    reasons = result["per_parent_failure_reasons"][f"0:{first.clean_parent_sha256}"]
    assert "amplitude_range_assignment_identity_drifted" in reasons
    assert "range_coordinate_contract_identity_drifted" in reasons


def test_worst_generating_branch_recall_prevents_pooled_success():
    plan, records = build_v5_k1_phase_c_fixture_records()
    first_branch = records[0].generating_branch_id
    changed = []
    failures = 0
    for value in records:
        if value.generating_branch_id == first_branch and failures < 2:
            changed.append(replace(value, matched_positive_branches_at_4=0))
            failures += 1
        else:
            changed.append(value)
    result = assess_v5_k1_phase_c_records(tuple(changed), plan=plan)

    assert result["integrity_passed"] is True
    assert result["metrics"]["minimum_generating_branch_macro_branch_recall_at_4"] == 0.92
    decision = result["gate_decisions"]["minimum_generating_branch_macro_branch_recall_at_4_gte"]
    assert decision["passed"] is False
    assert result["fixture_gate_passed"] is False


def test_product_must_strictly_beat_both_baselines_in_every_generating_branch():
    plan, records = build_v5_k1_phase_c_fixture_records()
    first_branch = records[0].generating_branch_id
    changed = []
    for value in records:
        if value.generating_branch_id != first_branch:
            changed.append(value)
            continue
        methods = tuple(
            replace(audit, primary_log2_budget_auc=0.60)
            if audit.method_id == K1_PHASE_C_PRODUCT_METHOD_ID
            else audit
            for audit in value.method_audits
        )
        changed.append(replace(value, method_audits=methods))
    result = assess_v5_k1_phase_c_records(tuple(changed), plan=plan)

    assert result["integrity_passed"] is True
    assert result["metrics"]["minimum_generating_branch_product_minus_each_baseline_auc"] < 0.0
    assert (
        result["gate_decisions"]["minimum_generating_branch_product_minus_each_baseline_auc_gt"][
            "passed"
        ]
        is False
    )
    assert result["fixture_gate_passed"] is False
