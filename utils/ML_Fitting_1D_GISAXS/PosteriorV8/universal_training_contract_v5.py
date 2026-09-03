"""Claim boundary between V5.1 warmup and cross-topology supervision.

The current grouped solution-stage shards evaluate the generating parameter
vector exactly once.  They are useful for local-density wiring and
memorization, but an observation paired only with its generating topology
contains no evidence for choosing between topology hypotheses.  This module
freezes that distinction without changing the in-progress grouped builder.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


V5_UNIVERSAL_TRAINING_CONTRACT_SCHEMA = (
    "gisaxs.posterior_v8.cross_topology_training_scope/v1"
)
V5_UNIVERSAL_TRAINING_CONTRACT_VERSION = (
    "posterior_v8_generating_warmup_vs_verified_cross_topology_search_v1"
)
V5_GENERATING_TOPOLOGY_WARMUP_STAGE = "generating_topology_local_density_warmup"
V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE = "cross_topology_verified_search"


def universal_training_contract_v5_payload() -> dict[str, object]:
    """Return the exact methodological claim contract for universal V5.1."""

    return {
        "schema_version": V5_UNIVERSAL_TRAINING_CONTRACT_SCHEMA,
        "version": V5_UNIVERSAL_TRAINING_CONTRACT_VERSION,
        "stages": {
            V5_GENERATING_TOPOLOGY_WARMUP_STAGE: {
                "observation_pairing": (
                    "generating_topology_only_with_generating_branch_known_truth_positive_"
                    "and_any_other_materialized_same_topology_branches_unverified"
                ),
                "permitted_objectives": [
                    "branch_local_mdn_wiring",
                    "codec_memorization",
                    "within_generating_branch_local_density_pretraining",
                ],
                "search_yield_bce_allowed": False,
                "cross_topology_model_selection_claim_allowed": False,
                "branch_ranking_claim_allowed": False,
                "generating_mismatch_is_negative": False,
                "competing_topology_outcomes_available": False,
                "reason": (
                    "the_same_observation_was_not_searched_under_competing_topologies"
                ),
            },
            V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE: {
                "observation_pairing": (
                    "same_observation_with_every_user_selected_topology_query_and_every_"
                    "codec_feasible_contextual_wire_branch"
                ),
                "global_branch_identity": "stable_topology_id_plus_wire_pattern_id",
                "required_outcomes": [
                    "compatible_found",
                    "no_compatible_found_within_frozen_search_budget",
                    "unverified",
                ],
                "outcome_source": (
                    "independent_frozen_bounded_exact_search_for_each_topology_specific_"
                    "geometry_amplitude_query_and_each_feasible_wire_branch"
                ),
                "competing_topology_label_from_generating_topology_forbidden": True,
                "competing_topology_label_from_generating_parameters_forbidden": True,
                "search_yield_bce_allowed_only_for_completed_outcomes": True,
                "unverified_bce_weight": 0.0,
                "generating_mismatch_is_negative": False,
                "negative_is_no_solution_certificate": False,
                "local_mdn_targets": (
                    "exact_compatible_branch_local_representatives_only"
                ),
                "conditional_ranking_requires": (
                    "completed_outcomes_for_the_full_preregistered_selected_topology_and_"
                    "feasible_branch_catalog_with_at_least_one_positive"
                ),
                "cross_topology_model_selection_claim_allowed": (
                    "only_after_held_out_full_catalog_ranking_and_exact_verified_candidate_"
                    "metrics_pass_the_frozen_protocol"
                ),
            },
        },
        "shared_invariants": {
            "one_clean_parent_split_for_all_topologies_and_views": True,
            "network_score_is_not_posterior_probability": True,
            "network_score_is_not_mathematical_solvability": True,
            "final_candidates_require_gui_consistent_exact_forward": True,
            "bounds_and_physics_failures_never_become_accepted_candidates": True,
            "zero_found_within_budget_is_not_no_solution": True,
            "parameter_representative_deduplication_runs_after_exact_verification": True,
            "parameter_distance_deduplication_is_within_declared_topology": True,
            "different_topology_component_slots_are_never_distance_aligned": True,
            "curve_equivalence_groups_may_span_topologies": True,
        },
    }


def validate_universal_training_contract_v5(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on missing, extended, or semantically changed scope claims."""

    if not isinstance(payload, Mapping):
        raise TypeError("universal training contract must be a mapping")
    value = deepcopy(dict(payload))
    if value != universal_training_contract_v5_payload():
        raise ValueError("universal training contract is missing or incompatible")
    return value


__all__ = [
    "V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE",
    "V5_GENERATING_TOPOLOGY_WARMUP_STAGE",
    "V5_UNIVERSAL_TRAINING_CONTRACT_SCHEMA",
    "V5_UNIVERSAL_TRAINING_CONTRACT_VERSION",
    "universal_training_contract_v5_payload",
    "validate_universal_training_contract_v5",
]
