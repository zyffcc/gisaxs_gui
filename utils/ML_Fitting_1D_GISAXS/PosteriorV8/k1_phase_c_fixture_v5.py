"""Lightweight structural fixture for K1 Phase-C; it is never formal evidence."""

from __future__ import annotations

from hashlib import sha256
import json

from .candidate_refinement_contract_v5 import V5_EXACT_FORWARD_BUDGET_UNIT
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_AMPLITUDE_AXIS_REGIMES,
    K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS,
    K1_PHASE_C_METHOD_IDS,
    K1_PHASE_C_OBSERVATION_EFFECTS,
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_PRODUCT_METHOD_ID,
    K1_PHASE_C_RANGE_STRESS_STRATA,
    K1_PHASE_C_REFERENCE_EXACT_BUDGET,
    K1_PHASE_C_REFERENCE_OUTPUT_CAP,
    K1_PHASE_C_RETRIEVAL_BASELINE_ID,
    K1_PHASE_C_SOBOL_BASELINE_ID,
    K1_PHASE_C_SPLIT_ID,
    v5_k1_phase_c_contract_payload,
)
from .k1_phase_c_evaluation_v5 import (
    V5K1PhaseCMethodAudit,
    V5K1PhaseCParentRecord,
    assess_v5_k1_phase_c_records,
)
from .k1_phase_c_plan_v5 import build_v5_k1_phase_c_plan
from .paper_budget_evaluator_v5 import (
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
)
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _axis_fixture(stratum: str, parent_id: str):
    geometry_count = amplitude_count = 4
    if stratum == "full":
        geometry_regimes = ("full",) * geometry_count
        geometry_placements = ("interior",) * geometry_count
        amplitude_regimes = ("full",) * amplitude_count
    elif stratum == "narrow":
        geometry_regimes = ("narrow",) * geometry_count
        geometry_placements = (
            "interior",
            "asymmetric_low",
            "asymmetric_high",
            "interior",
        )
        amplitude_regimes = ("narrow",) * amplitude_count
    elif stratum == "fixed":
        geometry_regimes = ("fixed",) * geometry_count
        geometry_placements = ("interior",) * geometry_count
        amplitude_regimes = ("fixed",) * amplitude_count
    elif stratum == "edge":
        geometry_regimes = ("narrow",) * geometry_count
        geometry_placements = ("edge_low", "interior", "edge_high", "interior")
        amplitude_regimes = ("edge_low", "narrow", "edge_high", "narrow")
    elif stratum == "mixed":
        geometry_regimes = ("full", "wide", "narrow", "fixed")
        geometry_placements = ("interior", "asymmetric_low", "interior", "edge_high")
        amplitude_regimes = ("full", "edge_low", "fixed", "wide")
    else:  # pragma: no cover - caller uses the frozen strata
        raise ValueError("unsupported fixture range stratum")
    if any(value not in K1_PHASE_C_GEOMETRY_AXIS_PLACEMENTS for value in geometry_placements):
        raise RuntimeError("fixture uses an unsupported geometry-axis placement")
    if any(value not in K1_PHASE_C_AMPLITUDE_AXIS_REGIMES for value in amplitude_regimes):
        raise RuntimeError("fixture uses an unsupported amplitude-axis regime")
    geometry_ids = tuple(
        _digest(f"{parent_id}:geometry-range-axis:{index}") for index in range(geometry_count)
    )
    amplitude_ids = tuple(
        _digest(f"{parent_id}:amplitude-range-axis:{index}") for index in range(amplitude_count)
    )
    return (
        geometry_count,
        geometry_regimes,
        geometry_placements,
        geometry_ids,
        amplitude_count,
        amplitude_regimes,
        amplitude_ids,
    )


def _method_audits(parent_id: str, reference_sha256: str):
    common = {
        "evaluator_schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
        "evaluator_version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
        "evaluator_config_sha256": _digest("k1-phase-c-fixture-evaluator"),
        "comparison_protocol_sha256": _digest("k1-phase-c-fixture-comparison"),
        "exact_judge_sha256": _digest("k1-phase-c-fixture-exact-judge"),
        "reference_set_sha256": reference_sha256,
        "exact_forward_budget_unit": V5_EXACT_FORWARD_BUDGET_UNIT,
        "exact_forward_budgets": EXACT_FORWARD_BUDGETS,
        "output_cap": K1_PHASE_C_REFERENCE_OUTPUT_CAP,
        "configured_exact_call_budget": K1_PHASE_C_REFERENCE_EXACT_BUDGET,
        "complete_contiguous_trace": True,
    }
    auc = {
        K1_PHASE_C_PRODUCT_METHOD_ID: 0.92,
        K1_PHASE_C_SOBOL_BASELINE_ID: 0.60,
        K1_PHASE_C_RETRIEVAL_BASELINE_ID: 0.70,
    }
    return tuple(
        V5K1PhaseCMethodAudit(
            method_id=method_id,
            trace_artifact_sha256=_digest(f"{parent_id}:{method_id}:trace"),
            trace_ledger_sha256=_digest(f"{parent_id}:{method_id}:ledger"),
            primary_log2_budget_auc=auc[method_id],
            **common,
        )
        for method_id in K1_PHASE_C_METHOD_IDS
    )


def build_v5_k1_phase_c_fixture_records():
    """Return a 12 x 25 structural fixture covering every required stress cell."""

    contract = v5_k1_phase_c_contract_payload()
    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    protocol_sha = str(contract["protocol_binding"]["protocol_sha256"])
    coordinate_contract = contract["clean_parent_design"]
    range_contract = contract["range_stress"]
    model_sha = _digest("k1-phase-c-fixture-model")
    source_sha = _digest("k1-phase-c-fixture-source")
    split_receipt_sha = _digest("k1-phase-c-fixture-disjoint-split-receipt")
    records = []
    for block in plan.sobol_blocks:
        branch = next(
            value
            for value in contract["population"]["generating_branches"]
            if value["branch_id"] == block.branch_id
        )
        for sobol_index in range(block.parent_count):
            parent_id = f"{block.branch_id}:{sobol_index}"
            range_stratum = K1_PHASE_C_RANGE_STRESS_STRATA[
                sobol_index % len(K1_PHASE_C_RANGE_STRESS_STRATA)
            ]
            observation_stratum = K1_PHASE_C_OBSERVATION_STRESS_STRATA[
                (sobol_index // len(K1_PHASE_C_RANGE_STRESS_STRATA))
                % len(K1_PHASE_C_OBSERVATION_STRESS_STRATA)
            ]
            axis_fixture = _axis_fixture(range_stratum, parent_id)
            reference_sha = _digest(f"{parent_id}:reference")
            records.append(
                V5K1PhaseCParentRecord(
                    clean_parent_sha256=_digest(f"{parent_id}:clean"),
                    universal_query_sha256=_digest(f"{parent_id}:query"),
                    protocol_sha256=protocol_sha,
                    model_artifact_sha256=model_sha,
                    source_bundle_sha256=source_sha,
                    reference_set_sha256=reference_sha,
                    split_id=K1_PHASE_C_SPLIT_ID,
                    split_disjointness_receipt_sha256=split_receipt_sha,
                    split_disjointness_verified=True,
                    generating_branch_id=block.branch_id,
                    topology_id=int(branch["topology_id"]),
                    pattern_id=int(branch["pattern_id"]),
                    sobol_block_sha256=block.block_sha256,
                    sobol_design_sha256=block.sobol_design_sha256,
                    sobol_index=sobol_index,
                    range_stress_stratum=range_stratum,
                    observation_stress_stratum=observation_stratum,
                    observation_effects=K1_PHASE_C_OBSERVATION_EFFECTS[observation_stratum],
                    geometry_axis_count=axis_fixture[0],
                    geometry_axis_regimes=axis_fixture[1],
                    geometry_axis_placements=axis_fixture[2],
                    geometry_axis_coordinate_sha256s=axis_fixture[3],
                    amplitude_axis_count=axis_fixture[4],
                    amplitude_axis_regimes=axis_fixture[5],
                    amplitude_axis_coordinate_sha256s=axis_fixture[6],
                    amplitude_range_assignment_schema=str(
                        range_contract["amplitude_range_assignment_schema"]
                    ),
                    amplitude_range_assignment_version=str(
                        range_contract["amplitude_range_assignment_version"]
                    ),
                    range_coordinate_contract_sha256=str(
                        coordinate_contract["coordinate_contract_sha256"]
                    ),
                    range_coordinate_contract_schema=str(coordinate_contract["coordinate_schema"]),
                    range_coordinate_contract_version=str(
                        coordinate_contract["coordinate_version"]
                    ),
                    range_coordinate_contract_dimension=int(
                        coordinate_contract["coordinate_dimension"]
                    ),
                    axis_independent_range_coordinates=True,
                    range_generated_before_truth=True,
                    observation_generated_before_curve=True,
                    branch_catalog_size=12,
                    completed_branch_outcomes=12,
                    unverified_branch_outcomes=0,
                    positive_branch_count=4,
                    matched_positive_branches_at_4=4,
                    any_exact_compatible=True,
                    reference_bank_qualified=True,
                    reference_bank_saturated=True,
                    reference_representative_count=4,
                    matched_reference_representative_count_at_n16_b4096=4,
                    verified_candidate_count=16,
                    bounds_compliant_candidate_count=16,
                    physics_compliant_candidate_count=16,
                    amplitude_compliant_candidate_count=16,
                    compatible_candidate_count_before_dedup_at_n16_b4096=8,
                    duplicate_candidate_count_at_n16_b4096=0,
                    method_audits=_method_audits(parent_id, reference_sha),
                )
            )
    return plan, tuple(records)


def run_v5_k1_phase_c_fixture_gate() -> dict[str, object]:
    plan, records = build_v5_k1_phase_c_fixture_records()
    return assess_v5_k1_phase_c_records(records, plan=plan)


def main() -> int:
    result = run_v5_k1_phase_c_fixture_gate()
    summary = {
        "fixture_gate_passed": result["fixture_gate_passed"],
        "formal_plan": result["formal_plan"],
        "formal_gate_passed": result["full_k1_phase_c_generalization_gate_passed"],
        "final_paper_model_acceptance": result["final_paper_model_acceptance"],
        "plan_sha256": result["plan_sha256"],
        "metrics": result["metrics"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["fixture_gate_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_v5_k1_phase_c_fixture_records",
    "run_v5_k1_phase_c_fixture_gate",
]
