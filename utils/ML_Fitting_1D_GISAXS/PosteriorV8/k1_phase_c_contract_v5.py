"""Frozen, fail-closed contract for the full K=1 generalization gate.

Phase-A and Phase-B intentionally exercise only one Sphere/no-D/no-Resolution
branch.  This module defines the next falsifiable step without claiming model
acceptance: all twelve K=1 branches compete for every clean parent, user ranges
are stressed per axis, and the learned method is compared with two baselines
under the same exact-forward budget.

No curve is generated here.  The module only owns immutable identities, the
balanced Sobol-block plan, and deterministic stress-cell assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
import re
from typing import Mapping

import numpy as np

from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
    V5_AMPLITUDE_RANGE_REGIMES,
)
from .bounds_query_v5 import AXIS_RANGE_PLACEMENTS, AXIS_RANGE_REGIMES
from .branch_catalog import branch_pattern_id, branch_pattern_is_valid, decode_branch_pattern
from .candidate_refinement_contract_v5 import V5_EXACT_FORWARD_BUDGET_UNIT
from .contract import MAX_COMPONENTS, SHAPES, topology_id_for
from .contextual_reference_bank_v5 import (
    V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
    V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
)
from .paper_budget_evaluator_v5 import (
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
)
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS, PRIMARY_OUTPUT_CAP
from .sobol_design_v5 import V5_SOBOL_DESIGN_SCHEMA, V5_SOBOL_DESIGN_VERSION
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    V5_SOBOL_RECIPE_DIM,
)
from .study_protocol import protocol_payload, validate_protocol


V5_K1_PHASE_C_SCHEMA = "gisaxs.posterior_v8.k1_full_branch_generalization_gate/v1"
V5_K1_PHASE_C_VERSION = "posterior_v8_v5_2_k1_all12_axis_independent_paired_budget_gate_v1"
V5_K1_PHASE_C_ROLE = (
    "k1_all_legal_branch_generalization_and_baseline_gate_not_final_model_acceptance"
)
V5_K1_PHASE_C_SOBOL_BLOCK_VERSION = (
    "posterior_v8_independent_scrambled_sobol_block_per_generating_branch_v1"
)
V5_K1_PHASE_C_STRESS_ASSIGNMENT_VERSION = (
    "posterior_v8_k1_range_fast_observation_slow_balanced_5x5_cycle_v1"
)

K1_PHASE_C_MASTER_SCRAMBLE_SEED = 20260903
K1_PHASE_C_BRANCH_CATALOG_SIZE = 12
K1_PHASE_C_SPLIT_ID = "engineering_e1_k1_phase_c_generalization_holdout"
K1_PHASE_C_TOP_K = 4
K1_PHASE_C_REFERENCE_OUTPUT_CAP = 16
K1_PHASE_C_REFERENCE_EXACT_BUDGET = 4096

K1_PHASE_C_RANGE_STRESS_STRATA = ("full", "narrow", "fixed", "edge", "mixed")
K1_PHASE_C_OBSERVATION_STRESS_STRATA = (
    "clean_control",
    "noisy",
    "masked",
    "cropped",
    "q_grid",
)
K1_PHASE_C_OBSERVATION_EFFECTS = {
    "clean_control": (),
    "noisy": ("noise",),
    "masked": ("mask",),
    "cropped": ("crop",),
    "q_grid": ("q_grid",),
}
K1_PHASE_C_GEOMETRY_AXIS_REGIMES = tuple(AXIS_RANGE_REGIMES)
K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS = tuple(AXIS_RANGE_PLACEMENTS)
K1_PHASE_C_AMPLITUDE_AXIS_REGIMES = tuple(V5_AMPLITUDE_RANGE_REGIMES)

K1_PHASE_C_PRODUCT_METHOD_ID = "v5_2_search_yield_ranked_mdn_plus_exact_refinement"
K1_PHASE_C_SOBOL_BASELINE_ID = "solver_only_sobol_or_independent_global_search"
K1_PHASE_C_RETRIEVAL_BASELINE_ID = "retrieval_seeded_exact_solver"
K1_PHASE_C_METHOD_IDS = (
    K1_PHASE_C_PRODUCT_METHOD_ID,
    K1_PHASE_C_SOBOL_BASELINE_ID,
    K1_PHASE_C_RETRIEVAL_BASELINE_ID,
)
K1_PHASE_C_BASELINE_IDS = (
    K1_PHASE_C_SOBOL_BASELINE_ID,
    K1_PHASE_C_RETRIEVAL_BASELINE_ID,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def unit_interval(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True, order=True)
class V5K1PhaseCBranch:
    """One of the twelve labelled K=1 shape/D/Resolution branches."""

    branch_id: str
    shape: str
    topology_id: int
    pattern_id: int
    d_present: bool
    resolution_present: bool

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


def _make_k1_branches() -> tuple[V5K1PhaseCBranch, ...]:
    values = []
    for shape in SHAPES:
        topology_id = topology_id_for((shape,))
        for d_present in (False, True):
            for resolution_present in (False, True):
                d_flags = (d_present,) + (False,) * (MAX_COMPONENTS - 1)
                pattern_id = branch_pattern_id(d_flags, resolution_present)
                if not branch_pattern_is_valid(topology_id, pattern_id):  # pragma: no cover
                    raise RuntimeError("constructed K1 branch is not legal")
                replay_d, replay_resolution = decode_branch_pattern(pattern_id)
                if replay_d != d_flags or replay_resolution is not resolution_present:
                    raise RuntimeError("K1 branch wire codec does not replay")
                values.append(
                    V5K1PhaseCBranch(
                        branch_id=(
                            f"k1:{shape}:D={'present' if d_present else 'absent'}:"
                            f"Resolution={'present' if resolution_present else 'absent'}"
                        ),
                        shape=shape,
                        topology_id=topology_id,
                        pattern_id=pattern_id,
                        d_present=d_present,
                        resolution_present=resolution_present,
                    )
                )
    result = tuple(values)
    if len(result) != K1_PHASE_C_BRANCH_CATALOG_SIZE:  # pragma: no cover
        raise RuntimeError("K1 Phase-C must contain exactly twelve legal branches")
    if len({(value.topology_id, value.pattern_id) for value in result}) != len(result):
        raise RuntimeError("K1 Phase-C branches must have unique wire identities")
    return result


K1_PHASE_C_BRANCHES = _make_k1_branches()
K1_PHASE_C_BRANCH_BY_ID = {value.branch_id: value for value in K1_PHASE_C_BRANCHES}


def _live_contract_core() -> dict[str, object]:
    protocol = validate_protocol(protocol_payload())
    stage = protocol["stages"]["engineering_e1"]
    schedule = protocol["topology_schedule_contract"]["k1"]
    stage_gates = stage.get("gates")
    expected_gate_keys = {
        "frozen_search_yield_ranking_recall_at_4_gte",
        "at_least_one_exact_compatible_candidate_rate_gte",
        "reference_representative_recall_at_n16_b4096_gte",
        "primary_auc_must_outperform",
    }
    if not isinstance(stage_gates, Mapping) or set(stage_gates) != expected_gate_keys:
        raise ValueError("engineering_e1 gates are incomplete, extended, or unsupported")
    if (
        stage.get("topology_schedule") != "k1"
        or stage.get("purpose") != "k1_method_selection"
        or schedule.get("topology_count") != 3
        or schedule.get("legal_branch_count") != K1_PHASE_C_BRANCH_CATALOG_SIZE
    ):
        raise ValueError("engineering_e1 is not the frozen full-K1 method-selection stage")
    baselines = tuple(stage_gates["primary_auc_must_outperform"])
    if baselines != K1_PHASE_C_BASELINE_IDS:
        raise ValueError("engineering_e1 baseline list changed incompatibly")
    recipes = positive_integer(stage["independent_clean_recipes"], "engineering_e1 recipes")
    if recipes % K1_PHASE_C_BRANCH_CATALOG_SIZE:
        raise ValueError("engineering_e1 recipes must balance exactly across twelve branches")

    branch_recall = unit_interval(
        stage_gates["frozen_search_yield_ranking_recall_at_4_gte"],
        "branch recall threshold",
    )
    compatible_rate = unit_interval(
        stage_gates["at_least_one_exact_compatible_candidate_rate_gte"],
        "exact-compatible threshold",
    )
    reference_recall = unit_interval(
        stage_gates["reference_representative_recall_at_n16_b4096_gte"],
        "reference recall threshold",
    )
    duplicate_rate = unit_interval(
        protocol["final_acceptance"]["duplicate_parameter_representative_rate_lt"],
        "duplicate-rate threshold",
    )
    if tuple(protocol["exact_forward_budgets"]) != EXACT_FORWARD_BUDGETS:
        raise ValueError("study protocol exact-forward budgets drifted")
    if K1_PHASE_C_REFERENCE_OUTPUT_CAP != PRIMARY_OUTPUT_CAP:
        raise RuntimeError("K1 Phase-C output cap must use the paper primary output cap")
    if K1_PHASE_C_REFERENCE_EXACT_BUDGET != EXACT_FORWARD_BUDGETS[-1]:
        raise RuntimeError("K1 Phase-C reference budget must use the largest paper budget")

    return {
        "schema": V5_K1_PHASE_C_SCHEMA,
        "version": V5_K1_PHASE_C_VERSION,
        "scientific_role": V5_K1_PHASE_C_ROLE,
        "protocol_binding": {
            "schema": protocol["schema_version"],
            "version": protocol["protocol_version"],
            "protocol_sha256": protocol["protocol_sha256"],
            "json_pointer": "/stages/engineering_e1",
        },
        "population": {
            "statistical_unit": "independent_clean_parent_physical_recipe",
            "split_id": K1_PHASE_C_SPLIT_ID,
            "stage_order": "after_phase_a_and_phase_b",
            "role": "independent_generalization_gate_not_formal_paper_test",
            "recipe_count": recipes,
            "parents_per_generating_branch": recipes // K1_PHASE_C_BRANCH_CATALOG_SIZE,
            "generating_branches": [value.audit_payload() for value in K1_PHASE_C_BRANCHES],
            "all_twelve_branches_compete_for_every_parent": True,
            "completed_search_outcomes_required_per_parent": K1_PHASE_C_BRANCH_CATALOG_SIZE,
            "branch_balance_is_exact": True,
        },
        "clean_parent_design": {
            "block_version": V5_K1_PHASE_C_SOBOL_BLOCK_VERSION,
            "sobol_schema": V5_SOBOL_DESIGN_SCHEMA,
            "sobol_version": V5_SOBOL_DESIGN_VERSION,
            "coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
            "coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
            "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
            "coordinate_dimension": V5_SOBOL_RECIPE_DIM,
            "independent_digital_scramble_per_generating_branch": True,
            "cross_branch_coordinate_reuse": False,
            "sobol_indices_are_local_to_each_block": True,
            "discrete_shape_d_resolution_are_forced_by_the_block_before_curve_generation": True,
            "range_or_acquisition_may_not_read_truth_or_curve": True,
            "training_tuning_calibration_reference_block_overlap_allowed": False,
            "independent_from": [
                "k1_phase_a_memorization_parents",
                "k1_phase_b_single_branch_parents",
                "model_training_parents",
                "tuning_parents",
                "calibration_parents",
                "formal_paper_test_parents",
                "network_free_reference_search_parents",
            ],
            "split_disjointness_receipt_required": True,
        },
        "range_stress": {
            "strata": list(K1_PHASE_C_RANGE_STRESS_STRATA),
            "geometry_axis_regimes": list(K1_PHASE_C_GEOMETRY_AXIS_REGIMES),
            "geometry_axis_placements": list(K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS),
            "geometry_edge_encoding": "placement_is_edge_low_or_edge_high",
            "amplitude_axis_regimes": list(K1_PHASE_C_AMPLITUDE_AXIS_REGIMES),
            "amplitude_edge_encoding": "regime_is_edge_low_or_edge_high_no_placement_field",
            "amplitude_range_assignment_schema": V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
            "amplitude_range_assignment_version": V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
            "geometry_and_amplitude_axis_decisions_use_distinct_named_coordinates": True,
            "one_global_range_regime_for_all_axes_allowed": False,
            "full": "every_available_axis_uses_its_full_domain",
            "narrow": "every_available_axis_is_narrow_with_axis_local_position",
            "fixed": "every_available_axis_is_fixed_at_an_axis_local_value",
            "edge": (
                "at_least_one_geometry_placement_or_amplitude_regime_explicitly_"
                "encodes_edge_low_or_edge_high"
            ),
            "mixed": (
                "at_least_two_raw_regime_families_across_geometry_and_amplitude;"
                "amplitude_edge_low_edge_high_remain_explicit_not_relabelled_as_placement"
            ),
        },
        "observation_stress": {
            "strata": list(K1_PHASE_C_OBSERVATION_STRESS_STRATA),
            "effects": {
                name: list(K1_PHASE_C_OBSERVATION_EFFECTS[name])
                for name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
            },
            "one_factor_control_semantics": True,
            "noise_mask_crop_and_q_grid_choices_precede_curve_evaluation": True,
        },
        "stress_assignment": {
            "version": V5_K1_PHASE_C_STRESS_ASSIGNMENT_VERSION,
            "cycle": "range_fast_observation_slow_5x5",
            "every_branch_must_cover_all_25_cells": True,
            "within_branch_cell_count_max_minus_min_lte": 1,
        },
        "paired_comparison": {
            "method_ids": list(K1_PHASE_C_METHOD_IDS),
            "product_method_id": K1_PHASE_C_PRODUCT_METHOD_ID,
            "baseline_method_ids": list(K1_PHASE_C_BASELINE_IDS),
            "budget_unit": V5_EXACT_FORWARD_BUDGET_UNIT,
            "exact_forward_budgets": list(EXACT_FORWARD_BUDGETS),
            "reference_output_cap": K1_PHASE_C_REFERENCE_OUTPUT_CAP,
            "reference_exact_budget": K1_PHASE_C_REFERENCE_EXACT_BUDGET,
            "paper_evaluator_schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
            "paper_evaluator_version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
            "same_observation_query_reference_set_exact_judge_and_budget": True,
            "complete_contiguous_exact_call_trace_required": True,
            "method_failure_is_zero_not_missing": True,
        },
        "reference": {
            "schema": V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
            "version": V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
            "network_free": True,
            "saturation_required": True,
            "finite_search_operational_representatives_not_all_mathematical_solutions": True,
        },
        "gates": {
            "minimum_generating_branch_macro_branch_recall_at_4_gte": branch_recall,
            "minimum_generating_branch_any_exact_compatible_rate_gte": compatible_rate,
            "minimum_generating_branch_reference_recall_at_n16_b4096_gte": (reference_recall),
            "bounds_compliance_fraction_eq": 1.0,
            "physics_compliance_fraction_eq": 1.0,
            "amplitude_compliance_fraction_eq": 1.0,
            "maximum_generating_branch_duplicate_rate_lt": duplicate_rate,
            "minimum_generating_branch_product_minus_each_baseline_auc_gt": 0.0,
        },
        "aggregation": {
            "parent_metrics": "unweighted_independent_clean_parent",
            "branch_safety": "every_generating_branch_must_pass_not_only_pooled_macro",
            "reference_recall": "matched_reference_count_divided_by_frozen_reference_count",
            "duplicate_rate": "duplicates_removed_divided_by_compatible_candidates_before_dedup",
            "baseline_comparison": "paired_parent_auc_then_macro_within_generating_branch",
        },
        "claim_limits": {
            "phase_c_is_final_model_acceptance": False,
            "finite_search_failure_proves_no_solution": False,
            "reference_bank_contains_all_mathematical_solutions": False,
            "synthetic_same_forward_results_prove_real_world_parameter_accuracy": False,
        },
    }


def v5_k1_phase_c_contract_payload() -> dict[str, object]:
    core = _live_contract_core()
    return {
        **core,
        "contract_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def validate_v5_k1_phase_c_contract(payload: Mapping[str, object]) -> dict[str, object]:
    """Require byte-equivalent scientific content to the live study protocol."""

    if not isinstance(payload, Mapping):
        raise TypeError("K1 Phase-C contract must be a mapping")
    value = dict(payload)
    if set(value) != set(_live_contract_core()) | {"contract_sha256"}:
        raise ValueError("K1 Phase-C contract fields are incomplete or unsupported")
    supplied = digest(value.pop("contract_sha256"), "contract_sha256")
    if sha256(canonical_json(value).encode("utf-8")).hexdigest() != supplied:
        raise ValueError("K1 Phase-C contract SHA-256 does not reproduce")
    live = v5_k1_phase_c_contract_payload()
    if dict(payload) != live:
        raise ValueError("K1 Phase-C contract drifted from the live study protocol")
    return live


__all__ = [
    "K1_PHASE_C_AMPLITUDE_AXIS_REGIMES",
    "K1_PHASE_C_BASELINE_IDS",
    "K1_PHASE_C_BRANCHES",
    "K1_PHASE_C_BRANCH_BY_ID",
    "K1_PHASE_C_BRANCH_CATALOG_SIZE",
    "K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS",
    "K1_PHASE_C_GEOMETRY_AXIS_REGIMES",
    "K1_PHASE_C_MASTER_SCRAMBLE_SEED",
    "K1_PHASE_C_METHOD_IDS",
    "K1_PHASE_C_OBSERVATION_EFFECTS",
    "K1_PHASE_C_OBSERVATION_STRESS_STRATA",
    "K1_PHASE_C_PRODUCT_METHOD_ID",
    "K1_PHASE_C_RANGE_STRESS_STRATA",
    "K1_PHASE_C_SPLIT_ID",
    "K1_PHASE_C_REFERENCE_EXACT_BUDGET",
    "K1_PHASE_C_REFERENCE_OUTPUT_CAP",
    "K1_PHASE_C_TOP_K",
    "V5_K1_PHASE_C_SOBOL_BLOCK_VERSION",
    "V5_K1_PHASE_C_STRESS_ASSIGNMENT_VERSION",
    "V5_K1_PHASE_C_ROLE",
    "V5_K1_PHASE_C_SCHEMA",
    "V5_K1_PHASE_C_VERSION",
    "V5K1PhaseCBranch",
    "canonical_json",
    "digest",
    "positive_integer",
    "unit_interval",
    "v5_k1_phase_c_contract_payload",
    "validate_v5_k1_phase_c_contract",
]
