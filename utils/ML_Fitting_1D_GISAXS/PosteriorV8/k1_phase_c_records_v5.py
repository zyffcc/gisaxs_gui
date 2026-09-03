"""Immutable audit records consumed by the K1 Phase-C evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCMethodAudit:
    method_id: str
    evaluator_schema: str
    evaluator_version: str
    evaluator_config_sha256: str
    comparison_protocol_sha256: str
    exact_judge_sha256: str
    reference_set_sha256: str
    exact_forward_budget_unit: str
    exact_forward_budgets: tuple[int, ...]
    output_cap: int
    configured_exact_call_budget: int
    complete_contiguous_trace: bool
    trace_artifact_sha256: str
    trace_ledger_sha256: str
    primary_log2_budget_auc: float

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCParentRecord:
    clean_parent_sha256: str
    universal_query_sha256: str
    protocol_sha256: str
    model_artifact_sha256: str
    source_bundle_sha256: str
    reference_set_sha256: str
    split_id: str
    split_disjointness_receipt_sha256: str
    split_disjointness_verified: bool
    generating_branch_id: str
    topology_id: int
    pattern_id: int
    sobol_block_sha256: str
    sobol_design_sha256: str
    sobol_index: int
    range_stress_stratum: str
    observation_stress_stratum: str
    observation_effects: tuple[str, ...]
    geometry_axis_count: int
    geometry_axis_regimes: tuple[str, ...]
    geometry_axis_placements: tuple[str, ...]
    geometry_axis_coordinate_sha256s: tuple[str, ...]
    amplitude_axis_count: int
    amplitude_axis_regimes: tuple[str, ...]
    amplitude_axis_coordinate_sha256s: tuple[str, ...]
    amplitude_range_assignment_schema: str
    amplitude_range_assignment_version: str
    range_coordinate_contract_sha256: str
    range_coordinate_contract_schema: str
    range_coordinate_contract_version: str
    range_coordinate_contract_dimension: int
    axis_independent_range_coordinates: bool
    range_generated_before_truth: bool
    observation_generated_before_curve: bool
    branch_catalog_size: int
    completed_branch_outcomes: int
    unverified_branch_outcomes: int
    positive_branch_count: int
    matched_positive_branches_at_4: int
    any_exact_compatible: bool
    reference_bank_qualified: bool
    reference_bank_saturated: bool
    reference_representative_count: int
    matched_reference_representative_count_at_n16_b4096: int
    verified_candidate_count: int
    bounds_compliant_candidate_count: int
    physics_compliant_candidate_count: int
    amplitude_compliant_candidate_count: int
    compatible_candidate_count_before_dedup_at_n16_b4096: int
    duplicate_candidate_count_at_n16_b4096: int
    method_audits: tuple[V5K1PhaseCMethodAudit, ...]

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["V5K1PhaseCMethodAudit", "V5K1PhaseCParentRecord"]
