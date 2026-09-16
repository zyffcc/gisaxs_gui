from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import stat

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
    V5_AMPLITUDE_RANGE_REGIMES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    branch_pattern_is_valid,
    decode_branch_pattern,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import SHAPES, topology_id_for
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_BASELINE_IDS,
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
    K1_PHASE_C_SPLIT_ID,
    validate_v5_k1_phase_c_contract,
    v5_k1_phase_c_contract_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
    build_v5_k1_phase_c_plan,
    planned_v5_k1_phase_c_stress_cell,
    v5_k1_phase_c_plan_from_payload,
    validate_v5_k1_phase_c_plan,
    write_v5_k1_phase_c_authoring_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_DIM,
)


def test_k1_phase_c_freezes_exactly_all_twelve_legal_labelled_branches():
    assert len(K1_PHASE_C_BRANCHES) == 12
    assert len({value.branch_id for value in K1_PHASE_C_BRANCHES}) == 12
    assert len({(value.topology_id, value.pattern_id) for value in K1_PHASE_C_BRANCHES}) == 12
    assert {value.shape for value in K1_PHASE_C_BRANCHES} == set(SHAPES)

    for shape in SHAPES:
        rows = [value for value in K1_PHASE_C_BRANCHES if value.shape == shape]
        assert len(rows) == 4
        assert {value.pattern_id for value in rows} == {0, 1, 16, 17}
        assert {value.topology_id for value in rows} == {topology_id_for((shape,))}
        assert {(value.d_present, value.resolution_present) for value in rows} == {
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        }
        for value in rows:
            assert branch_pattern_is_valid(value.topology_id, value.pattern_id)
            d_flags, resolution = decode_branch_pattern(value.pattern_id)
            assert d_flags == (value.d_present, False, False, False)
            assert resolution is value.resolution_present


def test_contract_is_post_phase_b_generalization_not_a_formal_paper_test():
    value = validate_v5_k1_phase_c_contract(v5_k1_phase_c_contract_payload())

    population = value["population"]
    assert population["stage_order"] == "after_phase_a_and_phase_b"
    assert population["role"] == "independent_generalization_gate_not_formal_paper_test"
    assert population["split_id"] == K1_PHASE_C_SPLIT_ID
    assert population["recipe_count"] == 13_824
    assert population["parents_per_generating_branch"] == 1_152
    assert population["all_twelve_branches_compete_for_every_parent"] is True

    clean = value["clean_parent_design"]
    assert clean["independent_digital_scramble_per_generating_branch"] is True
    assert clean["cross_branch_coordinate_reuse"] is False
    assert clean["split_disjointness_receipt_required"] is True
    assert "k1_phase_b_single_branch_parents" in clean["independent_from"]
    assert "formal_paper_test_parents" in clean["independent_from"]

    assert tuple(value["range_stress"]["strata"]) == K1_PHASE_C_RANGE_STRESS_STRATA
    assert value["range_stress"]["one_global_range_regime_for_all_axes_allowed"] is False
    assert (
        value["range_stress"][
            "geometry_and_amplitude_axis_decisions_use_distinct_named_coordinates"
        ]
        is True
    )
    assert tuple(value["range_stress"]["amplitude_axis_regimes"]) == (V5_AMPLITUDE_RANGE_REGIMES)
    assert value["range_stress"]["amplitude_edge_encoding"].startswith("regime_is_")
    assert value["range_stress"]["geometry_edge_encoding"].startswith("placement_is_")
    assert (
        value["range_stress"]["amplitude_range_assignment_schema"]
        == V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA
    )
    assert (
        value["range_stress"]["amplitude_range_assignment_version"]
        == V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION
    )
    assert (
        value["clean_parent_design"]["coordinate_contract_sha256"]
        == V5_SOBOL_RECIPE_COORDINATE_SHA256
    )
    assert value["clean_parent_design"]["coordinate_dimension"] == V5_SOBOL_RECIPE_DIM
    assert tuple(value["observation_stress"]["strata"]) == (K1_PHASE_C_OBSERVATION_STRESS_STRATA)
    assert value["claim_limits"]["phase_c_is_final_model_acceptance"] is False


def test_contract_freezes_equal_budget_baselines_and_all_numeric_gates():
    value = v5_k1_phase_c_contract_payload()
    comparison = value["paired_comparison"]

    assert tuple(comparison["baseline_method_ids"]) == K1_PHASE_C_BASELINE_IDS
    assert tuple(comparison["exact_forward_budgets"]) == EXACT_FORWARD_BUDGETS
    assert comparison["reference_output_cap"] == 16
    assert comparison["reference_exact_budget"] == 4096
    assert comparison["same_observation_query_reference_set_exact_judge_and_budget"] is True
    assert comparison["complete_contiguous_exact_call_trace_required"] is True
    assert value["gates"] == {
        "minimum_generating_branch_macro_branch_recall_at_4_gte": 0.95,
        "minimum_generating_branch_any_exact_compatible_rate_gte": 0.95,
        "minimum_generating_branch_reference_recall_at_n16_b4096_gte": 0.85,
        "bounds_compliance_fraction_eq": 1.0,
        "physics_compliance_fraction_eq": 1.0,
        "amplitude_compliance_fraction_eq": 1.0,
        "maximum_generating_branch_duplicate_rate_lt": 0.10,
        "minimum_generating_branch_product_minus_each_baseline_auc_gt": 0.0,
    }


def test_contract_tamper_is_rejected_even_when_top_level_shape_is_retained():
    value = v5_k1_phase_c_contract_payload()
    value["gates"] = {**value["gates"], "bounds_compliance_fraction_eq": 0.99}
    with pytest.raises(ValueError, match="SHA-256 does not reproduce"):
        validate_v5_k1_phase_c_contract(value)


def test_plan_has_twelve_independent_balanced_sobol_blocks_and_replays():
    plan = build_v5_k1_phase_c_plan()

    assert plan.formal is True
    assert plan.parents_per_branch == 1_152
    assert plan.total_parent_count == 13_824
    assert len(plan.sobol_blocks) == 12
    assert len({value.scramble_seed for value in plan.sobol_blocks}) == 12
    assert len({value.sobol_design_sha256 for value in plan.sobol_blocks}) == 12
    assert len({value.block_sha256 for value in plan.sobol_blocks}) == 12
    assert {value.split_id for value in plan.sobol_blocks} == {K1_PHASE_C_SPLIT_ID}
    assert validate_v5_k1_phase_c_plan(plan) is plan


def test_fixture_plan_covers_every_5_by_5_stress_cell_without_claiming_formal():
    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    block = plan.sobol_blocks[0]
    cells = {
        planned_v5_k1_phase_c_stress_cell(
            plan,
            branch_id=block.branch_id,
            sobol_index=index,
        )
        for index in range(25)
    }
    assert cells == {
        (range_name, observation_name)
        for range_name in K1_PHASE_C_RANGE_STRESS_STRATA
        for observation_name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
    }
    assert plan.formal is False

    with pytest.raises(ValueError, match="at least 25"):
        build_v5_k1_phase_c_plan(formal=False, parents_per_branch=24)
    with pytest.raises(ValueError, match="frozen parent count"):
        build_v5_k1_phase_c_plan(formal=True, parents_per_branch=25)


def test_plan_identity_tamper_fails_closed():
    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    tampered = replace(plan, total_parent_count=plan.total_parent_count - 1)
    with pytest.raises(ValueError, match="identity does not reproduce"):
        validate_v5_k1_phase_c_plan(tampered)


def test_phase_c_authoring_plan_is_strict_canonical_and_write_once(tmp_path: Path):
    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    target = tmp_path / "phase-c-plan.json"

    write_v5_k1_phase_c_authoring_plan(target, plan)

    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert target.stat().st_nlink == 1
    persisted = json.loads(target.read_text(encoding="utf-8"))
    assert v5_k1_phase_c_plan_from_payload(persisted) == plan
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_k1_phase_c_authoring_plan(target, plan)
    persisted["total_parent_count"] -= 1
    with pytest.raises(ValueError, match="does not reproduce"):
        v5_k1_phase_c_plan_from_payload(persisted)
