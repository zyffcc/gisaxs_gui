"""Machine-readable methods template for the V5.2 paper model.

The architectural document explains why these decisions exist.  This module
contains the numeric gates and implementation identities that must be frozen
before any holdout is inspected.  It is not an external preregistration and it
does not claim that a concrete paper run or model artifact has already frozen.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


# These two identifiers are intentionally available before the heavier pure-
# Python contract imports below.  Grouped artifact validation binds them while
# formal-production modules are imported, so defining them later would create a
# circular-import failure.
STUDY_PROTOCOL_SCHEMA = "gisaxs.posterior_v8.multisolution_study_protocol/v17"
STUDY_PROTOCOL_VERSION = (
    "posterior_v8_v5_2_methods_template_iid_conformal_typed_outputs_decimal80_"
    "multiseed_rqmc_numeric_query_bound_operational_mass_contract_20260903_v17"
)
STUDY_PROTOCOL_STATUS = (
    "methods_template_frozen_design_before_holdout_no_run_or_artifact_yet_frozen"
)

from .artifact import canonical_json_bytes
from .amplitude_query_v5 import (
    AMPLITUDE_QUERY_EMBEDDING_DIM,
    V5_AMPLITUDE_EMBEDDING_VERSION,
    V5_AMPLITUDE_QUERY_SCHEMA,
    V5_AMPLITUDE_QUERY_VERSION,
)
from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
    V5_AMPLITUDE_RANGE_REGIMES,
)
from .bounds_first_schedule import full_factorial_prefix_recipe_count
from .branch_codec import BRANCH_CODEC_VERSION
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .calibrated_search_threshold_v5 import (
    V5_CALIBRATION_IDENTITY_SCHEMA,
    V5_CALIBRATION_IDENTITY_VERSION,
    V5_OBSERVATION_THRESHOLD_SCHEMA,
    V5_OBSERVATION_THRESHOLD_VERSION,
)
from .candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_V5_SCHEMA,
    CANDIDATE_SUPERVISION_V5_VERSION,
    SEARCH_OUTCOME_STATES,
)
from .candidate_refinement_contract_v5 import (
    V5_EXACT_REFINEMENT_SCHEMA,
    V5_EXACT_REFINEMENT_VERSION,
)
from .compatibility_calibration import (
    ACQUISITION_POLICY_ID_VERSION,
    ACQUISITION_POLICY_REQUIRED_COMPONENTS,
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    DESIGN_STRATUM_UNIVERSE_FIELDS,
    DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    DESIGN_STRATUM_UNIVERSE_SHA256,
    DESIGN_STRATUM_UNIVERSE_VERSION,
    MEASUREMENT_SIGMA_POLICY,
    PREREGISTERED_DESIGN_STRATUM_COUNT,
    RESERVED_CALIBRATION_SPLIT_ID,
)
from .contextual_branch_catalog import CONTEXTUAL_BRANCH_CATALOG_VERSION
from .contextual_reference_bank_v5 import (
    V5_CONTEXTUAL_DEDUPLICATION_VERSION,
    V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
    V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
    V5_CONTEXTUAL_REFERENCE_CLAIM,
    V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION,
    V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION,
)
from .contract import FORWARD_MODEL_VERSION
from .exact_search_executor_v5 import (
    V5_EXACT_SEARCH_COMPLETION_RULE,
    V5_EXACT_SEARCH_EVALUATOR_VERSION,
    V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
    V5_EXACT_SEARCH_EXECUTOR_VERSION,
    V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
    V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256,
)
from .exact_search_schedule_v5 import (
    V5_EXACT_SEARCH_OPTIMIZER_SCHEMA,
    V5_EXACT_SEARCH_OPTIMIZER_VERSION,
)
from .formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION,
)
from .gui_amplitude_constraints import (
    CANONICAL_AMPLITUDE_GAUGE,
    COEFFICIENT_POLYTOPE_SCHEMA,
    GUI_AMPLITUDE_CONSTRAINT_SCHEMA,
    GUI_AMPLITUDE_CONSTRAINT_VERSION,
)
from .model_v5_contract import (
    MODEL_V5_INPUT_KEYS,
    MODEL_V5_NAME,
    MODEL_V5_OUTPUT_KEYS,
    MODEL_V5_SCHEMA,
    MODEL_V5_SCORE_SEMANTICS,
    MODEL_V5_VERSION,
)
from .paper_budget_evaluator_v5 import (
    V5_EQUIVALENCE_MATCHING_ENGINE,
    V5_EQUIVALENCE_MATCHING_VERSION,
    V5_PAIRED_BOOTSTRAP_RNG,
    V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE,
    V5_PAIRED_BOOTSTRAP_VERSION,
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
)
from .paper_checkpoint_selector_v5 import (
    V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID,
    V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA,
    V5_PAPER_CHECKPOINT_SELECTION_RULE_ID,
    V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA,
    V5_PAPER_CHECKPOINT_SELECTOR_VERSION,
    V5_SELECTION_ONLY_STATUS,
    V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
    V5_TUNING_SELECTION_SPLIT,
)
from .paper_endpoint_metrics import (
    CURVE_EQUIVALENCE_LOG_RMSE_MAX,
    EXACT_FORWARD_BUDGETS,
    OUTPUT_CAPS,
    PAPER_ENDPOINT_METRICS_VERSION,
    PARAMETER_CLUSTER_DISTANCE_MAX,
    PRIMARY_ENDPOINT_NAME,
    PRIMARY_OUTPUT_CAP,
    REFERENCE_QUALIFICATION_SCHEMA,
    REFERENCE_QUALIFICATION_STATUSES,
    REFERENCE_QUALIFICATION_VERSION,
    REFERENCE_MATCH_DISTANCE_MAX,
    THRESHOLD_SENSITIVITY_MULTIPLIERS,
)
from .paper_representative_payload_v5 import (
    V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
    V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
)
from .paper_rqmc_design_v5 import (
    V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT,
    V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED,
    V5_PAPER_RQMC_DESIGN_SCHEMA,
    V5_PAPER_RQMC_DESIGN_VERSION,
    V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_IDENTITY_SCHEMA,
    V5_PAPER_RQMC_INFERENCE_RULE,
    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_RANDOMIZATION_UNIT,
    V5_PAPER_RQMC_SPLITS,
    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES,
)
from .paper_rqmc_evaluator_v5 import (
    V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE,
    V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE,
    V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA,
    V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION,
    V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS,
    V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA,
    V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION,
    V5_PAPER_RQMC_REPLICATE_MEAN_SCHEMA,
    V5_PAPER_RQMC_SHARED_BASELINE_POLICY,
    V5_PAPER_RQMC_TRAINING_SCOPE,
)
from .preprocessing import PREPROCESSING_VERSION
from .proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
)
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
)
from .query_bound_evaluation_v5 import (
    V5_QUERY_BOUND_EVALUATION_SCHEMA,
    V5_QUERY_BOUND_EVALUATION_SHA256,
    V5_QUERY_BOUND_EVALUATION_VERSION,
    V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
)
from .split_design_v5 import (
    MAIN_SPLITS,
    OOD_LABELS,
    V5_SPLIT_PLAN_SCHEMA,
    V5_SPLIT_PLAN_VERSION,
)
from .universal_inference_contract_v5 import (
    V5_UNIVERSAL_BRANCH_RANKING,
    V5_UNIVERSAL_INFERENCE_SCHEMA,
    V5_UNIVERSAL_INFERENCE_VERSION,
    V5_UNIVERSAL_MODE_STOP,
    V5_UNIVERSAL_SEED_SCHEDULE,
)
from .uncertainty_provenance_v5 import (
    UNCERTAINTY_KINDS,
    V5_UNCERTAINTY_SCHEMA,
    V5_UNCERTAINTY_VERSION,
)


# Kept as identifiers rather than importing TensorFlow into this manifest CLI.
V5_OBJECTIVE_MODULE = "PosteriorV8.training_objective_v5"
V5_OBJECTIVE_SCHEMA = "gisaxs.posterior_v8.candidate_training_objective/v7"
V5_OBJECTIVE_VERSION = (
    "posterior_v8_recipe_pairwise_yield_local_mdn_mass_coverage_soft_rank_top4_v5"
)


def _formal_production_identity() -> dict[str, object]:
    """Read formal identities lazily to avoid the grouped-artifact import cycle."""

    from .formal_production_search_contract_v5 import (
        V5_FORMAL_PRODUCTION_ALLOWED_SPLITS,
        V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256,
        V5_FORMAL_PRODUCTION_SHARD_SCHEMA,
        V5_FORMAL_PRODUCTION_SHARD_VERSION,
        V5_FORMAL_PRODUCTION_SOURCE_SCHEMA,
        V5_FORMAL_PRODUCTION_SOURCE_VERSION,
        V5_FORMAL_PRODUCTION_STAGE_IDS,
        V5_FORMAL_PRODUCTION_STAGE_SCHEMA,
        V5_FORMAL_PRODUCTION_STAGE_VERSION,
    )
    from .formal_production_search_membership_v5 import (
        V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA,
    )
    from .formal_production_search_plan_v5 import (
        V5_FORMAL_PRODUCTION_SCIENTIFIC_SCOPE,
        V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA,
        V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION,
        V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED,
    )
    from .formal_production_search_runtime_v5 import V5FormalProductionExecutableShard

    return {
        "source_schema": V5_FORMAL_PRODUCTION_SOURCE_SCHEMA,
        "source_version": V5_FORMAL_PRODUCTION_SOURCE_VERSION,
        "stage_schema": V5_FORMAL_PRODUCTION_STAGE_SCHEMA,
        "stage_version": V5_FORMAL_PRODUCTION_STAGE_VERSION,
        "stage_ids": list(V5_FORMAL_PRODUCTION_STAGE_IDS),
        "shard_schema": V5_FORMAL_PRODUCTION_SHARD_SCHEMA,
        "shard_version": V5_FORMAL_PRODUCTION_SHARD_VERSION,
        "allowed_splits": list(V5_FORMAL_PRODUCTION_ALLOWED_SPLITS),
        "query_contract_sha256": V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256,
        "plan_schema": V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA,
        "plan_version": V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION,
        "scientific_scope": V5_FORMAL_PRODUCTION_SCIENTIFIC_SCOPE,
        "membership_proof_schema": V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA,
        "runtime_module": V5FormalProductionExecutableShard.__module__,
        "runtime_type": V5FormalProductionExecutableShard.__name__,
        "training_promotion_enabled_in_bound_implementation": (
            V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED
        ),
    }


def _scientific_generation_identity() -> dict[str, object]:
    """Bind label semantics lazily without creating the dataset import cycle."""

    from .bounds_query_v5 import V5_LOCAL_TARGET_VERSION
    from .grouped_dataset_v5 import V5_GROUPED_DATASET_SCHEMA, V5_GROUPED_DATASET_VERSION
    from .sobol_amplitude_recipe_v5 import (
        V5_DIRECT_AMPLITUDE_SCHEMA,
        V5_DIRECT_AMPLITUDE_VERSION,
    )
    from .sobol_recipe_physics_v5 import V5_DIRECT_PHYSICS_VERSION
    from .sobol_numeric_canonicalization_v5 import (
        V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
        v5_numeric_policy_payload,
        v5_numeric_policy_sha256,
    )
    from .sobol_recipe_coordinates_v5 import (
        V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        V5_SOBOL_RECIPE_COORDINATE_SHA256,
        V5_SOBOL_RECIPE_COORDINATE_VERSION,
        V5_SOBOL_RECIPE_DIM,
    )
    from .sobol_recipe_v5 import (
        V5_SOBOL_CLEAN_RECIPE_SCHEMA,
        V5_SOBOL_CLEAN_RECIPE_VERSION,
    )
    from .synthetic_recipe_v5 import V5_CLEAN_RECIPE_SCHEMA, V5_CLEAN_RECIPE_VERSION

    return {
        "local_target_version": V5_LOCAL_TARGET_VERSION,
        "clean_recipe_schema": V5_CLEAN_RECIPE_SCHEMA,
        "clean_recipe_version": V5_CLEAN_RECIPE_VERSION,
        "direct_physics_version": V5_DIRECT_PHYSICS_VERSION,
        "direct_amplitude_schema": V5_DIRECT_AMPLITUDE_SCHEMA,
        "direct_amplitude_version": V5_DIRECT_AMPLITUDE_VERSION,
        "sobol_numeric_policy_version": V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
        "sobol_numeric_policy_contract": v5_numeric_policy_payload(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
        "sobol_numeric_policy_sha256": v5_numeric_policy_sha256(
            V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
        ),
        "amplitude_query_sampler_version": V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
        "amplitude_range_assignment_schema": V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
        "amplitude_range_assignment_version": V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
        "amplitude_range_regimes": list(V5_AMPLITUDE_RANGE_REGIMES),
        "active_amplitude_axes_receive_independent_range_assignments": True,
        "sobol_coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        "sobol_coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
        "sobol_coordinate_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "sobol_coordinate_dimension": V5_SOBOL_RECIPE_DIM,
        "sobol_clean_recipe_schema": V5_SOBOL_CLEAN_RECIPE_SCHEMA,
        "sobol_clean_recipe_version": V5_SOBOL_CLEAN_RECIPE_VERSION,
        "grouped_dataset_schema": V5_GROUPED_DATASET_SCHEMA,
        "grouped_dataset_version": V5_GROUPED_DATASET_VERSION,
    }


def _stage(
    *,
    topology_schedule: str,
    recipes: int,
    training_augmentation_views: int,
    seeds: int,
    purpose: str,
    gates: Mapping[str, object],
) -> dict[str, object]:
    return {
        "topology_schedule": topology_schedule,
        "distinct_clean_parent_design_points": recipes,
        "independent_clean_recipes": recipes,
        "independent_clean_recipes_compatibility_field_semantics": (
            "legacy_key_for_distinct_design_points_not_an_iid_claim"
        ),
        "recipe_count_semantics": (
            "distinct_clean_parent_design_points_not_iid_statistical_replicates"
        ),
        "training_augmentation_views_per_recipe": training_augmentation_views,
        "formal_search_label_views_per_recipe": 1,
        "formal_label_candidate_view_indices": list(V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES),
        "independent_training_seeds": seeds,
        "purpose": purpose,
        "gates": dict(gates),
    }


def _protocol_core() -> dict[str, object]:
    formal_production = _formal_production_identity()
    scientific_generation = _scientific_generation_identity()
    return {
        "schema_version": STUDY_PROTOCOL_SCHEMA,
        "protocol_version": STUDY_PROTOCOL_VERSION,
        "status": STUDY_PROTOCOL_STATUS,
        "frozen_date": "2026-09-03",
        "status_contract": {
            "document_kind": "methods_template",
            "design_must_be_frozen_before_holdout_inspection": True,
            "external_timestamped_preregistration_claimed": False,
            "concrete_run_plan_frozen": False,
            "trained_model_artifact_frozen": False,
            "holdout_or_real_case_results_inspected_under_this_version": False,
            "performance_claim_allowed_from_template_alone": False,
        },
        "paper_identity": {
            "preferred_title": (
                "Forward-Model-Verified Amortized Multisolution Inversion for "
                "User-Constrained Empirical 1D GISAXS Cuts"
            ),
            "required_forward_terminology": (
                "authoritative non-neural versioned empirical forward model"
            ),
            "disallowed_title_claim": "Physics-Verified",
            "absolute_first_ever_claim_allowed": False,
            "conservative_novelty_scope": (
                "the_combination_of_user_range_conditioning_finite_mixed_topology_"
                "set_valued_proposals_authoritative_bounded_refinement_and_equal_"
                "exact_forward_budget_equivalence_class_recall"
            ),
            "novelty_search_is_mathematically_exhaustive": False,
        },
        "scientific_scope": (
            "forward_model_verified_set_valued_inversion_of_the_versioned_"
            "empirical_1d_gisaxs_cut_forward"
        ),
        "authoritative_forward_contract": {
            "authority": "authoritative_non_neural_versioned_empirical_forward_model",
            "forward_version": FORWARD_MODEL_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "branch_codec_version": BRANCH_CODEC_VERSION,
            "component_slot_canonicalization_version": CANONICAL_COMPONENT_SLOTS_VERSION,
            "component_slot_exchangeability_requires": (
                "identical_geometry_bounds_D_policy_branch_D_presence_and_Int_i_bounds"
            ),
            "missing_amplitude_context_allows_slot_exchange": False,
            "exact_refinement_coordinates": "codec_varying_indices_only",
            "fixed_active_axes_consume_optimizer_dimensions": False,
            "neural_role": "amortized_candidate_proposal_and_search_yield_ranking_only",
            "compatibility_judge": (
                "versioned_non_neural_empirical_forward_plus_frozen_range_and_physics_gates"
            ),
            "forward_version_and_digest_required_in_every_paper_artifact": True,
            "neural_surrogate_may_replace_final_exact_verification": False,
            "agreement_scope": "agreement_with_this_versioned_empirical_forward_not_full_physics",
            "gui_formula": ("I(q)=BG+k*(sum_i(Int_i*P_i(q)*S_i(q))+int_Res*R(q))"),
            "gui_BG_k_Int_i_and_int_Res_are_independent_inputs": True,
            "effective_linear_coefficients": ("a_i=k*Int_i_and_a_res=k*int_Res_with_BG_unchanged"),
            "shared_auxiliary_projection": (
                "exists_kappa_in_user_k_interval_with_Int_i_low*kappa_lte_a_i_"
                "lte_Int_i_high*kappa_and_equivalent_optional_int_Res_constraints"
            ),
            "shared_k_gauge_quotient": (
                "different_explicit_k_Int_witnesses_with_identical_effective_"
                "coefficients_are_one_forward_parameterization"
            ),
            "k_equals_sum_particle_coefficients_assumption_allowed": False,
        },
        "implementation_identity": {
            "proposal_model": {
                "schema": MODEL_V5_SCHEMA,
                "version": MODEL_V5_VERSION,
                "name": MODEL_V5_NAME,
            },
            "gui_amplitude_constraint": {
                "schema": GUI_AMPLITUDE_CONSTRAINT_SCHEMA,
                "version": GUI_AMPLITUDE_CONSTRAINT_VERSION,
                "coefficient_polytope_schema": COEFFICIENT_POLYTOPE_SCHEMA,
                "shared_gauge_contract": CANONICAL_AMPLITUDE_GAUGE,
            },
            "training_objective": {
                "module": V5_OBJECTIVE_MODULE,
                "schema": V5_OBJECTIVE_SCHEMA,
                "version": V5_OBJECTIVE_VERSION,
                "imported_at_manifest_runtime": False,
                "reason": "keep_manifest_tensorflow_free",
            },
            "proposal_execution_policy": {
                **V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
                "sha256": V5_PROPOSAL_EXECUTION_POLICY_SHA256,
                "bound_by_grouped_training_and_universal_inference": True,
            },
            "component_slot_canonicalization": {
                "version": CANONICAL_COMPONENT_SLOTS_VERSION,
                "equivalence_source": (
                    "contextual_branch_catalog.component_slot_contract_equivalence_classes"
                ),
                "requires_geometry_D_and_component_intensity_range_context": True,
                "missing_component_intensity_context_policy": "singleton_slots_no_permutation",
            },
            "exact_refinement": {
                "schema": V5_EXACT_REFINEMENT_SCHEMA,
                "version": V5_EXACT_REFINEMENT_VERSION,
                "optimization_coordinates": "codec_varying_indices_only",
                "fixed_active_axes_are_template_coordinates": True,
                "zero_varying_axes_policy": "one_exact_profile_no_scipy",
            },
            "exact_search_optimizer": {
                "schema": V5_EXACT_SEARCH_OPTIMIZER_SCHEMA,
                "version": V5_EXACT_SEARCH_OPTIMIZER_VERSION,
                "finite_difference_budget_denominator": "varying_dimension_count_plus_one",
            },
            "scientific_data_generation": scientific_generation,
            "compatibility_calibration": {
                "schema": CALIBRATION_SCHEMA,
                "version": CALIBRATION_VERSION,
                "checked_identity_schema": V5_CALIBRATION_IDENTITY_SCHEMA,
                "checked_identity_version": V5_CALIBRATION_IDENTITY_VERSION,
                "observation_threshold_schema": V5_OBSERVATION_THRESHOLD_SCHEMA,
                "observation_threshold_version": V5_OBSERVATION_THRESHOLD_VERSION,
            },
            "formal_label_observation": {
                "schema": V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA,
                "version": V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION,
                "policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
                "policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
                "candidate_view_indices": list(V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES),
                "selected_sigma_present_views_per_recipe": 1,
            },
            "formal_production_search": formal_production,
            "query_parameter_distance": {
                "schema": V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
                "version": V5_QUERY_PARAMETER_DISTANCE_VERSION,
                "scope": V5_QUERY_PARAMETER_DISTANCE_SCOPE,
                "sha256": V5_QUERY_PARAMETER_DISTANCE_SHA256,
            },
            "exact_search_executor": {
                "schema": V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
                "version": V5_EXACT_SEARCH_EXECUTOR_VERSION,
                "evaluator_version": V5_EXACT_SEARCH_EVALUATOR_VERSION,
                "representative_distance_id": (V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID),
                "representative_distance_sha256": (V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256),
                "completion_rule": V5_EXACT_SEARCH_COMPLETION_RULE,
            },
            "contextual_reference_bank": {
                "schema": V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
                "version": V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
                "distance_version": V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION,
                "slot_equivalence_version": V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION,
                "deduplication_version": V5_CONTEXTUAL_DEDUPLICATION_VERSION,
                "claim": V5_CONTEXTUAL_REFERENCE_CLAIM,
            },
            "paper_budget_evaluator": {
                "schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
                "version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
                "matching_version": V5_EQUIVALENCE_MATCHING_VERSION,
                "matching_engine": V5_EQUIVALENCE_MATCHING_ENGINE,
                "paired_bootstrap_version": V5_PAIRED_BOOTSTRAP_VERSION,
                "paired_bootstrap_rng": V5_PAIRED_BOOTSTRAP_RNG,
                "paired_bootstrap_claim_scope": V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE,
                "actual_emitted_representative_payload_is_primary_match_input": True,
                "hidden_cluster_members_may_raise_primary_recall": False,
                "representative_payload_schema": V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
                "representative_payload_version": V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
                "one_payload_contains_exactly_one_typed_parameter_representative": True,
                "payload_hash_binds_query_branch_source_parameter_and_exact_curve": True,
                "cross_query_reference_or_emission_payload_allowed": False,
            },
            "paper_rqmc_design": {
                "schema": V5_PAPER_RQMC_DESIGN_SCHEMA,
                "version": V5_PAPER_RQMC_DESIGN_VERSION,
                "identity_schema": V5_PAPER_RQMC_IDENTITY_SCHEMA,
                "formal_splits": list(V5_PAPER_RQMC_SPLITS),
                "engineering_minimum_independent_replicates": (
                    V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES
                ),
                "engineering_minimum_allows_formal_inference": False,
                "formal_minimum_independent_replicates": (
                    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
                ),
                "paper_target_independent_replicates": (
                    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
                ),
                "randomization_unit": V5_PAPER_RQMC_RANDOMIZATION_UNIT,
                "inference_rule": V5_PAPER_RQMC_INFERENCE_RULE,
                "finite_sample_conformal_calibration_included": (
                    V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED
                ),
                "compatibility_calibration_sampling_requirement": (
                    V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT
                ),
                "authoritative_coordinate_schema_version_sha_dimension_and_names_bound": True,
                "names_only_coordinate_identity_allowed": False,
            },
            "paper_rqmc_evaluator": {
                "schema": V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA,
                "version": V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION,
                "replicate_mean_schema": V5_PAPER_RQMC_REPLICATE_MEAN_SCHEMA,
                "training_artifact_set_schema": V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA,
                "training_artifact_set_version": V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION,
                "formal_training_artifact_set_scope": (
                    V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE
                ),
                "engineering_training_artifact_set_scope": (
                    V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE
                ),
                "formal_minimum_predeclared_training_seeds": (
                    V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS
                ),
                "engineering_tiny_artifact_set_may_enter_formal_summary": False,
                "training_scope": V5_PAPER_RQMC_TRAINING_SCOPE,
                "primary_solver_baseline_policy": V5_PAPER_RQMC_SHARED_BASELINE_POLICY,
                "inference_seed_set_sha256_required": True,
                "study_protocol_and_endpoint_metric_sha256_required_per_cell": True,
                "frozen_cohort_and_qualification_artifact_sha256_required_per_cell": True,
                "frozen_seed_set_contrast": (
                    "mean_paired_difference_across_predeclared_training_seeds_within_"
                    "each_scramble_then_student_t_across_independent_scrambles"
                ),
                "point_bootstrap_available": False,
            },
            "one_click_inference": {
                "schema": V5_UNIVERSAL_INFERENCE_SCHEMA,
                "version": V5_UNIVERSAL_INFERENCE_VERSION,
                "branch_ranking": V5_UNIVERSAL_BRANCH_RANKING,
                "seed_schedule": V5_UNIVERSAL_SEED_SCHEDULE,
                "stop_rule": V5_UNIVERSAL_MODE_STOP,
                "query_bound_evaluator": {
                    "schema": V5_QUERY_BOUND_EVALUATION_SCHEMA,
                    "version": V5_QUERY_BOUND_EVALUATION_VERSION,
                    "sha256": V5_QUERY_BOUND_EVALUATION_SHA256,
                    "reference_matching": V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
                    "actual_emitted_representatives_only": True,
                    "legacy_global_evaluator_is_product_endpoint": False,
                },
                "proposal_execution_policy_sha256": (
                    V5_PROPOSAL_EXECUTION_POLICY_SHA256
                ),
            },
        },
        "model_contract": {
            "schema": MODEL_V5_SCHEMA,
            "version": MODEL_V5_VERSION,
            "name": MODEL_V5_NAME,
            "input_keys": list(MODEL_V5_INPUT_KEYS),
            "output_keys": list(MODEL_V5_OUTPUT_KEYS),
            "contextual_branch_catalog_version": CONTEXTUAL_BRANCH_CATALOG_VERSION,
            "candidate_supervision_schema": CANDIDATE_SUPERVISION_V5_SCHEMA,
            "candidate_supervision_version": CANDIDATE_SUPERVISION_V5_VERSION,
            "candidate_supervision_outcomes": list(SEARCH_OUTCOME_STATES),
            "objective_module": V5_OBJECTIVE_MODULE,
            "objective_schema": V5_OBJECTIVE_SCHEMA,
            "objective_version": V5_OBJECTIVE_VERSION,
            "objective_terms": {
                "search_yield_bce": "operational_yield_calibration",
                "pairwise_ranking": ("all_completed_positive_negative_pairs_within_clean_recipe"),
                "local_density": "expanded_representative_masked_local_mdn_nll",
                "local_coverage": (
                    "mixture_mass_weighted_negative_temperature_logsumexp_of_"
                    "per_expanded_representative_median_masked_local_rms"
                ),
                "operational_top_l_alignment": (
                    "distance_affinity_weighted_pairwise_sigmoid_soft_rank_"
                    "penalty_above_frozen_top4"
                ),
                "hard_best_of_m_local_rms": "diagnostic_only_not_a_loss_term",
                "operational_top_l": 4,
                "operational_metrics": [
                    "best_local_rms_top_l",
                    "target_recall_top_l",
                    "miss_censored_hit_mixture_rank_top_l",
                    "top_l_mixture_mass",
                    "mixture_entropy",
                    "effective_mixture_components",
                    "effective_mixture_utilization",
                    "duplicate_fraction_top_l",
                ],
                "warmup_pairwise_ranking_weight": 0.0,
            },
            "multisolution_mass_objective_status": {
                "matched_set_or_mass_aware_objective_implemented": True,
                "matched_set_or_hungarian_implemented": False,
                "current_objective": (
                    "expanded_local_mdn_nll_plus_mixture_mass_aware_soft_coverage"
                ),
                "current_objective_is_posterior_mass_calibrated": False,
                "explicit_low_weight_ghost_mode_mass_penalty_implemented": True,
                "mass_aware_performance_hypothesis_enabled": True,
                "versioned_objective_and_focused_regression_tests_required": True,
            },
            "uncertainty_schema": V5_UNCERTAINTY_SCHEMA,
            "uncertainty_version": V5_UNCERTAINTY_VERSION,
            "uncertainty_states": list(UNCERTAINTY_KINDS),
            "score_semantics": MODEL_V5_SCORE_SEMANTICS,
            "candidate_enumeration": "external_contextual_catalog_per_user_query",
            "continuous_output": "branch_conditioned_local_coordinate_diagonal_mdn",
            "artifact_identity_must_be_frozen_before_holdout": True,
            "paper_training_v5_2_requirement": {
                "status": "satisfied_by_bound_v5_2_model_contract",
                "geometry_embedding_input": "geometry_bounds_embedding",
                "amplitude_embedding_input": "amplitude_bounds_embedding",
                "amplitude_query_schema": V5_AMPLITUDE_QUERY_SCHEMA,
                "amplitude_query_version": V5_AMPLITUDE_QUERY_VERSION,
                "amplitude_embedding_version": V5_AMPLITUDE_EMBEDDING_VERSION,
                "amplitude_embedding_dimension": AMPLITUDE_QUERY_EMBEDDING_DIM,
                "explicit_query_conditioning": [
                    "geometry_ranges",
                    "BG_range",
                    "k_range",
                    "Int_i_ranges",
                    "int_Res_range_when_resolution_present",
                ],
                "same_ranges_define_exact_gui_amplitude_polytope": True,
                "geometry_and_amplitude_graph_dependency_audit_required": True,
                "new_model_schema_version_and_artifact_digest_required": True,
                "paper_training_or_holdout_allowed_without_this_contract": False,
                "paper_holdout_allowed_before_artifact_identity_freeze": False,
            },
        },
        "search_yield_estimand": {
            "positive": (
                "frozen_generation_and_refinement_protocol_finds_at_least_one_"
                "exact_compatible_representative_for_query_branch"
            ),
            "negative": "no_compatible_found_within_frozen_search_budget",
            "unverified_enters_bce": False,
            "negative_requires_completed_frozen_search_and_budget_provenance": True,
            "generating_parameter_mismatch_is_negative": False,
            "posterior_probability": False,
            "mathematical_branch_solvability": False,
            "no_solution_certificate": False,
        },
        "search_yield_ranking_endpoint": {
            "unit": "one_query_over_its_codec_feasible_contextual_branch_catalog",
            "positive_branch": "compatible_found_under_the_same_frozen_search_protocol",
            "ranking": "descending_proposal_search_yield_logit_with_frozen_tie_break",
            "recall_at_L": "positive_branches_in_top_L_divided_by_all_positive_branches",
            "query_eligibility": "every_codec_feasible_contextual_branch_has_completed_outcome",
            "top_L_when_catalog_smaller": "rank_all_min_L_and_catalog_size_branches",
            "zero_positive_catalog": (
                "undefined_for_conditional_ranking_recall_and_count_zero_in_"
                "preselected_full_query_sensitivity"
            ),
            "partially_unverified_catalog": (
                "exclude_from_conditional_ranking_endpoint_and_count_zero_in_"
                "preselected_full_query_sensitivity"
            ),
            "outcome_label_search_uses_evaluated_model_score": False,
            "outcome_label_search_budget_schedule_frozen_before_scoring": True,
            "operational_not_branch_solvability": True,
        },
        "local_mdn_diagnostic_contract": {
            "distance": "rms_over_varying_dimension_mask_only",
            "fixed_and_inactive_axes_enter_distance": False,
            "exact_refinement_coordinates": "codec_varying_indices_only",
            "zero_varying_axes_refinement": "one_exact_profile_no_optimizer",
            "memorization_reports": [
                "single_draw_median_local_rms",
                "best_of_32_median_and_p90_local_rms",
                "exact_post_refine_raw_logrmse_and_compatible_rate",
            ],
            "generating_target_error_is_primary_on_ambiguous_test": False,
        },
        "comparison_fairness": {
            "shared_inputs": [
                "observed_curve_and_uncertainty_provenance",
                "geometry_and_amplitude_user_ranges",
                "codec_feasible_contextual_branch_catalog",
            ],
            "shared_scientific_judge": (
                "same_authoritative_non_neural_versioned_empirical_forward_"
                "compatibility_and_range_physics_gates"
            ),
            "paired_budgets": ["exact_forward_calls", "wall_clock"],
            "candidate_output_cap_and_budget_prefix_rules_identical": True,
            "attempts_consuming_exact_budget": [
                "exact_compatible",
                "exact_incompatible",
                "unverified_after_an_exact_call_started",
            ],
            "output_cap_counts": (
                "only_exact_compatible_verified_returned_cluster_representatives"
            ),
            "incompatible_or_unverified_attempts_consume_output_cap": False,
            "legacy_method_missing_capability_is_reported_not_silently_repaired": True,
        },
        "claim_limits": {
            "mathematical_solution_completeness": False,
            "full_dwba_gisaxs": False,
            "legacy_resolution_is_instrument_convolution": False,
            "refined_candidates_are_posterior_samples": False,
            "finite_search_failure_proves_no_solution": False,
            "complete_linkage_diameter_cluster_representatives_are_mathematical_modes": (False),
            "representatives_from_distinct_complete_linkage_clusters_are_guaranteed_"
            "pairwise_farther_than_delta": False,
            "versioned_forward_agreement_proves_full_scattering_physics": False,
            "real_case_without_ground_truth_supports_parameter_accuracy": False,
            "real_case_study_supports_population_generalization": False,
        },
        "statistical_unit": (
            "clean_parent_physical_recipe_is_the_within_scramble_observation_unit_"
            "and_independent_scramble_replicate_mean_is_the_formal_randomization_unit"
        ),
        "exact_forward_budgets": list(EXACT_FORWARD_BUDGETS),
        "primary_endpoint": {
            "name": PRIMARY_ENDPOINT_NAME,
            "metrics_version": PAPER_ENDPOINT_METRICS_VERSION,
            "conditioning": "frozen_reference_search_qualification",
            "unconditional_population_performance_claimed": False,
            "scientific_object": (
                "deterministic_complete_linkage_diameter_delta_cluster_"
                "representatives_of_exact_compatible_parameters_not_mathematical_"
                "or_posterior_modes"
            ),
            "reported_output_caps": list(OUTPUT_CAPS),
            "primary_output_cap": PRIMARY_OUTPUT_CAP,
            "summary": (
                "recipe_macro_normalized_trapezoid_auc_over_log2_exact_forward_budget_"
                "then_report_each_seed_and_seed_mean"
            ),
            "candidate_eligibility": (
                "exact_compatible_and_bounds_pass_and_physics_pass_independent_of_"
                "observability_or_minimality"
            ),
            "reference_population": (
                "all_reference_discovered_exact_compatible_representatives_including_"
                "observability_unknown_and_confirmed_redundant"
            ),
            "strict_minimal_reference_role": "secondary_sensitivity_only",
            "cumulative_budget_prefix_rule": (
                "candidate_enters_B_only_if_refinement_and_final_exact_verification_"
                "complete_at_cumulative_exact_forward_calls_lte_B"
            ),
            "output_cap_filter_order": (
                "within_budget_prefix_filter_exact_compatible_and_verified_then_"
                "take_first_N_emitted_cluster_representatives_by_frozen_reference_"
                "independent_ranking"
            ),
            "matching_candidate_payload": (
                "exact_payload_of_actual_user_visible_emitted_cluster_representative"
            ),
            "matching_candidate_payload_schema": V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
            "matching_candidate_payload_version": V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
            "hidden_accepted_cluster_member_may_satisfy_primary_match": False,
            "output_cap_counts_incompatible_or_unverified": False,
            "incompatible_and_unverified_exact_calls_still_consume_B": True,
            "observability_budget_rule": (
                "post_search_separate_budget_excluded_from_primary_exact_forward_B"
            ),
            "ranking": "frozen_product_ranking_independent_of_reference",
            "cluster_semantics": (
                "complete_linkage_enforces_within_cluster_diameter_lte_delta_but_"
                "does_not_guarantee_pairwise_distance_gt_delta_between_returned_"
                "representatives"
            ),
            "matching": "maximum_cardinality_then_minimum_distance_bipartite",
            "per_recipe_recall": "matched_reference_count_divided_by_M_ref",
            "M_ref_gt_N": (
                "retain_denominator_M_ref_and_report_ceiling_min_N_M_ref_divided_by_M_ref"
            ),
            "auc_formula_for_frozen_doubling_budgets": ("(0.5*R256+R512+R1024+R2048+0.5*R4096)/4"),
            "aggregation": {
                "within_scramble_and_seed": ("unweighted_macro_mean_over_qualified_clean_recipes"),
                "formal_randomization_unit": V5_PAPER_RQMC_RANDOMIZATION_UNIT,
                "across_training_seeds": (
                    "report_every_seed_and_within_each_scramble_average_the_frozen_"
                    "predeclared_seed_set_before_paired_scramble_inference"
                ),
                "pooled_reference_mode_micro_average_is_primary": False,
            },
            "method_crash_or_algorithm_failure": ("zero_recall_at_every_budget_not_missing"),
        },
        "operational_representative_thresholds": {
            "parameter_complete_linkage_distance_max": PARAMETER_CLUSTER_DISTANCE_MAX,
            "reference_match_distance_max": REFERENCE_MATCH_DISTANCE_MAX,
            "raw_curve_equivalence_log_rmse_max": CURVE_EQUIVALENCE_LOG_RMSE_MAX,
            "sensitivity_multipliers": list(THRESHOLD_SENSITIVITY_MULTIPLIERS),
            "interpretation": "frozen_design_operational_not_physical_constants",
            "distance_schema": V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
            "distance_version": V5_QUERY_PARAMETER_DISTANCE_VERSION,
            "distance_sha256": V5_QUERY_PARAMETER_DISTANCE_SHA256,
            "distance_retains_absolute_scale": [
                "background",
                "effective_particle_total",
            ],
            "distance_quotients_only": "exact_shared_k_GUI_gauge",
            "complete_linkage_guarantee": "within_cluster_diameter_lte_delta",
            "between_representative_pairwise_gt_delta_guaranteed": False,
            "parameter_metric_missing_linear_solution": (
                "fail_closed_for_full_metric_never_drop_composition_dimensions"
            ),
            "geometry_only_analysis": "separate_versioned_secondary_metric_only",
        },
        "reference_qualification_and_missingness": {
            "schema": REFERENCE_QUALIFICATION_SCHEMA,
            "version": REFERENCE_QUALIFICATION_VERSION,
            "typed_statuses": list(REFERENCE_QUALIFICATION_STATUSES),
            "preselected_cohort_recipe_ids_and_source_sha256_frozen_before_search": True,
            "qualification_artifact_contains_exactly_one_typed_status_per_cohort_recipe": True,
            "qualified": (
                "independent_network_free_search_satisfies_frozen_saturation_and_has_"
                "at_least_one_exact_compatible_reference"
            ),
            "unqualified_or_unsaturated": (
                "exclude_from_conditional_primary_without_replacement_and_count_as_zero_"
                "in_conservative_full_preselected_set_sensitivity"
            ),
            "empty_without_infeasibility_certificate": "unresolved_not_no_solution",
            "certified_no_solution": (
                "evaluate_in_separate_abstention_endpoint_not_representative_recall"
            ),
            "missing_reference_linear_solution": (
                "reference_metric_ineligible_for_full_metric_not_geometry_fallback"
            ),
            "missing_sigma_or_unseen_calibration_stratum": (
                "descriptive_raw_metrics_only_no_formal_compatibility_endpoint"
            ),
            "reference_qualification_fraction": "always_report",
            "complete_preselected_cohort_lower_bound": (
                "qualified_endpoint_sum_divided_by_preselected_count_with_all_"
                "unresolved_recipes_imputed_zero"
            ),
            "complete_preselected_cohort_unresolved_universal_upper_bound": (
                "qualified_endpoint_sum_plus_one_per_unresolved_recipe_divided_by_preselected_count"
            ),
            "qualification_strata_required": [
                "component_count",
                "range_width_regime",
                "weak_component_status",
                "ambiguity_stratum",
            ],
            "headline_blocked_when_frozen_stratified_qualification_gate_fails": True,
        },
        "secondary_endpoints": [
            "exact_compatible_curve_success",
            "compatible_candidate_yield",
            "unique_compatible_representative_yield",
            "conservative_reference_match_fraction_lower_bound_with_unmatched_compatible_separate",
            "strict_minimal_confirmed_effective_reference_sensitivity",
            "observability_status_distribution",
            "bounds_and_physics_compliance",
            "duplicate_parameter_representative_rate",
            "wall_clock_and_core_or_gpu_time",
            "frozen_search_yield_branch_ranking_recall",
            "local_mdn_best_of_m_local_coordinate_rms",
        ],
        "required_method_comparisons": [
            "legacy_single_or_fixed_multihead_regression",
            "range_conditioned_single_point_mse_regression_plus_same_exact_polish",
            "topology_classifier_then_top_k_topologies_then_exact_solver",
            "solver_only_sobol_or_independent_global_search",
            "pure_sobol_multistart_exact_solver_ensemble",
            "differential_evolution_or_genetic_algorithm_plus_same_exact_solver",
            "retrieval_seeded_exact_solver",
            "matched_branch_conditioned_conditional_flow_or_cinn",
            "exact_forward_mcmc_or_nuts_stratified_bayesian_reference",
            "v5_2_branch_conditioned_mdn_only_without_exact_refinement",
            "v5_2_search_yield_ranked_mdn_plus_exact_refinement",
            "v5_2_search_yield_ranked_mdn_plus_exact_refinement_and_rescue",
        ],
        "method_comparison_estimands": {
            "primary_operational_estimand": (
                "finite_budget_recall_of_deterministic_complete_linkage_diameter_"
                "delta_cluster_representatives_of_exact_compatible_parameters_"
                "under_user_ranges"
            ),
            "matched_branch_conditioned_conditional_flow_or_cinn": {
                "same_estimand_as_primary": True,
                "matching_controls": [
                    "identical_training_recipes_and_splits",
                    "identical_contextual_branch_conditioning",
                    "identical_exact_forward_budgets_and_output_caps",
                    "identical_final_authoritative_forward_verification",
                ],
            },
            "exact_forward_mcmc_or_nuts_stratified_bayesian_reference": {
                "scope": "predeclared_stratified_subset",
                "same_estimand_as_primary": False,
                "estimand": (
                    "posterior_mass_and_credible_regions_under_a_declared_prior_and_likelihood"
                ),
                "forward": "authoritative_non_neural_versioned_empirical_forward_model",
                "role": "bayesian_reference_and_uncertainty_diagnostic_not_primary_mode_recall",
                "must_report_prior_likelihood_chains_and_convergence_diagnostics": True,
            },
        },
        "oracle_diagnostics": {
            "role": "diagnostic_error_decomposition_only_not_a_deployable_method",
            "reference_information_may_train_or_select_product_model": False,
            "topology_ranking": {
                "intervention": (
                    "rank_reference_discovered_compatible_topologies_and_contextual_"
                    "branches_before_incompatible_branches"
                ),
                "held_fixed": "learned_continuous_proposal_refinement_and_final_verification",
                "generating_topology_is_oracle_truth": False,
            },
            "continuous_proposal": {
                "intervention": (
                    "seed_each_reference_discovered_compatible_branch_from_its_"
                    "complete_linkage_diameter_cluster_representative_basin"
                ),
                "held_fixed": "operational_refinement_and_final_verification",
                "generating_parameter_is_unique_truth": False,
            },
            "refinement": {
                "intervention": (
                    "replace_operational_refinement_with_joint_gold_optimization_from_"
                    "identical_ranked_continuous_proposals"
                ),
                "held_fixed": "topology_ranking_continuous_proposals_and_exact_call_budget",
                "final_authoritative_forward_verification_required": True,
            },
            "reporting": (
                "paired_primary_endpoint_deltas_attribute_headroom_to_topology_ranking_"
                "continuous_proposal_and_refinement_without_claiming_causal_independence"
            ),
        },
        "required_ablations": [
            "remove_query_bounds_conditioning",
            "replace_local_with_global_coordinates",
            "remove_contextual_candidate_branch_enumeration",
            "remove_frozen_search_yield_ranking",
            "replace_profiled_amplitude_initialization_with_direct_or_joint_only_initialization_keep_exact_polytope",
            "replace_coupled_gui_amplitude_polytope_with_axis_box_during_search_keep_final_exact_gui_range_gate",
            "remove_BG_k_Int_i_int_Res_query_conditioning_but_keep_exact_polytope",
            "collapse_uncertainty_provenance_states",
            "uncertainty_provenance_strata_measured_vs_simulated_vs_encoder_proxy",
            (
                "pipeline_stage_proposal_only_vs_exact_scoring_only_vs_refinement_"
                "and_final_verification"
            ),
            "remove_exact_refinement",
            "replace_search_yield_ranking_with_uniform_and_seeded_random_branch_order",
            (
                "explicit_GUI_k_Int_witness_vs_effective_coefficients_with_exact_"
                "shared_k_gauge_quotient"
            ),
            (
                "representative_delta_output_cap_N_compatibility_threshold_and_"
                "exact_forward_budget_sensitivity"
            ),
            "remove_rescue_keep_ranked_mdn_and_exact_refinement",
            "remove_ambiguity_bank_mining",
            "identifiable_ambiguity_and_hard_negative_training_contribution",
            "guarded_sobol_blocks_vs_random_record_split_leakage_diagnostic",
            "training_scale_learning_curve",
        ],
        "ablation_reporting_contract": {
            "pipeline_stages": [
                "proposal_only_without_exact_scoring_or_refinement",
                "proposal_plus_authoritative_exact_scoring_without_refinement",
                "proposal_plus_refinement_plus_final_authoritative_exact_verification",
            ],
            "branch_ranking_controls": [
                "uniform_contextual_branch_priority_with_frozen_tie_break",
                "seeded_random_contextual_branch_order",
            ],
            "uncertainty_strata": list(UNCERTAINTY_KINDS),
            "amplitude_parameterizations": [
                "direct_gui_BG_k_Int_i_and_optional_int_Res_coordinates",
                (
                    "effective_BG_a_i_optional_a_res_with_exact_shared_auxiliary_"
                    "kappa_projection_and_explicit_GUI_witness"
                ),
            ],
            "sensitivity_axes": {
                "representative_delta": PARAMETER_CLUSTER_DISTANCE_MAX,
                "output_cap_N": list(OUTPUT_CAPS),
                "raw_curve_compatibility_threshold": CURVE_EQUIVALENCE_LOG_RMSE_MAX,
                "reference_match_threshold": REFERENCE_MATCH_DISTANCE_MAX,
                "threshold_multipliers": list(THRESHOLD_SENSITIVITY_MULTIPLIERS),
                "exact_forward_budget": list(EXACT_FORWARD_BUDGETS),
            },
            "training_case_contributions": [
                "identifiable_cases",
                "ambiguous_multi_representative_cases",
                "completed_search_hard_negative_branches",
            ],
            "split_leakage_diagnostic": {
                "valid_design": "guarded_disjoint_clean_parent_sobol_blocks",
                "leaky_control": "seeded_random_record_split_after_view_and_branch_expansion",
                "leaky_control_may_support_headline_results": False,
            },
        },
        "uncertainty_contract": {
            "proposal": (
                "search_yield_reliability_brier_nll_for_frozen_protocol_and_"
                "branch_conditioned_held_out_nll_coverage_and_rank_for_frozen_"
                "target_construction_distribution"
            ),
            "continuous_density_is_posterior": False,
            "sbc_name_allowed_only_for_unbiased_declared_prior_likelihood_joint_draws": True,
            "post_refinement_outputs_enter_sbc": False,
            "input_provenance": list(UNCERTAINTY_KINDS),
            "measurement": (
                "independent_acquisition_only_marginal_split_conformal_compatibility_calibration"
            ),
            "search": "reference_saturation_and_optimizer_disagreement",
        },
        "observation_view_contract": {
            "formal_search_label_candidate_pool": list(
                V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES
            ),
            "formal_search_label_selection": (
                "curve_blind_exactly_one_measurement_sigma_present_view_per_recipe"
            ),
            "formal_search_label_views_per_recipe": 1,
            "zero_or_multiple_sigma_present_candidates": "fail_closed",
            "policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
            "policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
            "training_augmentation_views": (
                "separate_stage_specific_related_views_grouped_with_clean_parent"
            ),
            "training_augmentation_view_count_changes_formal_label_count": False,
        },
        "compatibility_calibration": {
            "primary_semantics": "acquisition_only_marginal_coverage",
            "schema": CALIBRATION_SCHEMA,
            "version": CALIBRATION_VERSION,
            "reserved_split_id": RESERVED_CALIBRATION_SPLIT_ID,
            "sampling_design": (
                "separate_iid_exchangeable_clean_parent_recipes_within_frozen_strata"
            ),
            "finite_sample_exchangeability_required": True,
            "rqmc_design_points_used_to_fit_split_conformal_threshold": False,
            "paper_rqmc_design_includes_calibration": (
                V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED
            ),
            "sampling_requirement": V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT,
            "reachable_design_strata": PREREGISTERED_DESIGN_STRATUM_COUNT,
            "compatibility_stratum_version": COMPATIBILITY_STRATUM_VERSION,
            "compatibility_stratum_fields": list(COMPATIBILITY_STRATUM_FIELDS),
            "design_stratum_universe_version": DESIGN_STRATUM_UNIVERSE_VERSION,
            "design_stratum_universe_fields": list(DESIGN_STRATUM_UNIVERSE_FIELDS),
            "design_stratum_universe_semantics": DESIGN_STRATUM_UNIVERSE_SEMANTICS,
            "design_stratum_universe_sha256": DESIGN_STRATUM_UNIVERSE_SHA256,
            "acquisition_policy_id_version": ACQUISITION_POLICY_ID_VERSION,
            "acquisition_policy_required_components": list(ACQUISITION_POLICY_REQUIRED_COMPONENTS),
            "measurement_sigma_policy": MEASUREMENT_SIGMA_POLICY,
            "formal_lookup_requires_full_policy_membership_in_selected_stratum": True,
            "marginalized_within_stratum": [
                "grid_kind",
                "crop_policy",
                "mask_policy",
                "realized_valid_point_count",
            ],
            "candidate_or_generating_component_count_in_primary_stratum": False,
            "target_coverage": 0.95,
            "minimum_independent_recipes_per_stratum": 200,
            "k_conditional_calibration": (
                "secondary_label_conditional_sensitivity_not_topology_precision"
            ),
            "adaptive_search_branch_familywise_false_positive_controlled": False,
        },
        "split_contract": {
            "schema": V5_SPLIT_PLAN_SCHEMA,
            "version": V5_SPLIT_PLAN_VERSION,
            "legacy_engineering_mutually_exclusive_clean_parent_sobol_blocks": list(MAIN_SPLITS),
            "ood_subblocks": list(OOD_LABELS),
            "group_by": "clean_parent_recipe_id_all_views_and_candidate_branches_follow_parent",
            "legacy_engineering_partition": (
                "contiguous_disjoint_sobol_blocks_with_unused_guard_bands"
            ),
            "formal_paper_rqmc_splits": list(V5_PAPER_RQMC_SPLITS),
            "formal_paper_calibration_partition": (
                "separate_iid_exchangeable_sampling_not_a_randomized_sobol_net"
            ),
            "roles": {
                "train": "gradient_updates_only",
                "tuning_validation": (
                    "epoch_checkpoint_selection_architecture_early_stopping_and_"
                    "family_promotion_only"
                ),
                "calibration": (
                    "iid_exchangeable_measurement_compatibility_threshold_only"
                ),
                "test": "locked_id_synthetic_final_evaluation_only",
                "reference": "network_free_gold_reference_construction_only",
                "ood": "predeclared_challenges_excluded_from_id_headline",
            },
            "test_used_for_model_or_threshold_selection": False,
            "reference_used_for_model_or_threshold_selection": False,
            "external_real_cases_are_not_sobol_split_members": True,
            "real_cases_used_for_training_calibration_or_selection": False,
        },
        "statistical_inference": {
            "minimum_independent_training_seeds": V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS,
            "seed_reporting": "report_every_seed_then_arithmetic_seed_mean_never_best_seed_only",
            "formal_randomization_unit": V5_PAPER_RQMC_RANDOMIZATION_UNIT,
            "minimum_independent_scramble_replicates": (V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES),
            "target_independent_scramble_replicates": (
                V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
            ),
            "engineering_four_scramble_design_may_enter_formal_inference": False,
            "confidence_interval": V5_PAPER_RQMC_INFERENCE_RULE,
            "within_scramble_points_are_iid": False,
            "recipe_or_point_bootstrap_is_formal_population_interval": False,
            "formal_design_schema": V5_PAPER_RQMC_DESIGN_SCHEMA,
            "formal_design_version": V5_PAPER_RQMC_DESIGN_VERSION,
            "formal_evaluator_schema": V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA,
            "formal_evaluator_version": V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION,
            "formal_training_scope": V5_PAPER_RQMC_TRAINING_SCOPE,
            "formal_training_artifact_set_schema": (V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA),
            "formal_training_artifact_set_version": (V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION),
            "formal_artifact_set_constructor_enforces_minimum_training_seeds": True,
            "frozen_seed_set_summary_order": (
                "average_seed_level_paired_differences_within_scramble_then_student_t_"
                "over_independent_scramble_means"
            ),
            "fixed_cohort_secondary_interval": {
                "version": V5_PAIRED_BOOTSTRAP_VERSION,
                "rng": V5_PAIRED_BOOTSTRAP_RNG,
                "claim_scope": V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE,
                "bootstrap_replicates": 10_000,
            },
            "paired_methods_use_identical_recipes_budgets_and_training_seeds": True,
            "primary_solver_baseline_model_seed_policy": (V5_PAPER_RQMC_SHARED_BASELINE_POLICY),
            "solver_baseline_inference_seed_set_sha256_required": True,
            "comparison_method_inference_seed_set_sha256_required": True,
            "training_seed_population_interval_claimed": False,
            "primary_confirmatory_contrast": (
                "v5_2_search_yield_ranked_mdn_plus_exact_refinement_and_rescue_"
                "versus_solver_only_sobol_or_independent_global_search"
            ),
            "primary_two_sided_alpha": 0.05,
            "secondary_familywise_plan": "holm_adjustment_within_each_endpoint_family",
            "exploratory_strata_are_labelled_not_confirmatory": True,
            "views_or_reference_representatives_are_independent_replicates": False,
        },
        "real_data_claim_contract": {
            "role": "blinded_case_study_after_model_threshold_and_protocol_freeze",
            "population_performance_claim_allowed": False,
            "parameter_accuracy_without_traceable_ground_truth_allowed": False,
            "reported_without_ground_truth": [
                "exact_forward_residual",
                "bounds_and_physics_compliance",
                "candidate_diversity_and_stability",
                "expert_blinded_assessment",
                "latency_and_failure_audit",
            ],
            "external_characterization": "concordance_only_unless_declared_as_traceable_ground_truth",
        },
        "compute_reporting": {
            "latency": [
                "end_to_end_median_and_p90",
                "proposal_time",
                "exact_refinement_time",
                "cold_and_warm_start",
            ],
            "compute": [
                "exact_forward_calls",
                "gpu_hours",
                "cpu_core_hours",
                "peak_memory",
                "hardware_and_software_environment",
            ],
            "energy_if_metered": "report_kWh_with_measurement_method_and_coverage",
            "carbon_if_reported": (
                "descriptive_CO2e_with_energy_source_region_time_and_emissions_factor_provenance"
            ),
            "missing_energy_meter_is_reported_not_imputed": True,
        },
        "topology_schedule_contract": {
            "single_branch_sphere_pattern0": {
                "declared_topology": ["sphere"],
                "branch_pattern_id": 0,
                "d_present": [False],
                "resolution_present": False,
                "purpose": "isolated_representation_and_objective_capacity_diagnostic",
                "full_k1_coverage_claim_allowed": False,
            },
            "k1": {
                "component_counts": [1],
                "shape_multisets": "all",
                "codec_feasible_branch_patterns_per_topology": "all",
                "topology_count": 3,
                "legal_branch_count": 12,
            },
            "k1_k2": {
                "component_counts": [1, 2],
                "shape_multisets": "all",
                "codec_feasible_branch_patterns_per_topology": "all",
                "topology_count": 9,
                "legal_branch_count": 60,
            },
            "all34": {
                "component_counts": [1, 2, 3, 4],
                "shape_multisets": "all",
                "codec_feasible_branch_patterns_per_topology": "all",
                "topology_count": 34,
                "legal_branch_count": 700,
            },
        },
        "stages": {
            "contract_smoke": _stage(
                topology_schedule="k1",
                recipes=full_factorial_prefix_recipe_count("k1"),
                training_augmentation_views=2,
                seeds=1,
                purpose="contract_and_plumbing_only",
                gates={
                    "bounds_and_physics_compliance": 1.0,
                    "codec_roundtrip_max_abs_error_lt": 1.0e-6,
                    "exact_forward_log_rmse_lt": 1.0e-4,
                    "performance_claim_allowed": False,
                },
            ),
            "k1_memorization": _stage(
                topology_schedule="single_branch_sphere_pattern0",
                recipes=512,
                training_augmentation_views=1,
                seeds=1,
                purpose="single_branch_capacity_and_objective_diagnostic",
                gates={
                    "branch_conditioned_local_mdn_single_draw_local_rms_median_lt": 0.05,
                    "branch_conditioned_local_mdn_best_of_32_local_rms_median_lt": 0.01,
                    "branch_conditioned_local_mdn_best_of_32_local_rms_p90_lt": 0.03,
                    "exact_post_refine_raw_log_rmse_p90_lt": 1.0e-3,
                    "exact_post_refine_compatible_rate_gte": 0.99,
                },
            ),
            "engineering_e1": _stage(
                topology_schedule="k1",
                recipes=13_824,
                training_augmentation_views=3,
                seeds=1,
                purpose="k1_method_selection",
                gates={
                    "frozen_search_yield_ranking_recall_at_4_gte": 0.95,
                    "at_least_one_exact_compatible_candidate_rate_gte": 0.95,
                    "reference_representative_recall_at_n16_b4096_gte": 0.85,
                    "primary_auc_must_outperform": [
                        "solver_only_sobol_or_independent_global_search",
                        "retrieval_seeded_exact_solver",
                    ],
                },
            ),
            "engineering_e2": _stage(
                topology_schedule="k1_k2",
                recipes=82_944,
                training_augmentation_views=3,
                seeds=1,
                purpose="k1_k2_multisolution_gate",
                gates={
                    "frozen_search_yield_ranking_recall_at_8_gte": 0.90,
                    "at_least_one_exact_compatible_candidate_rate_gte": 0.90,
                    "reference_representative_recall_at_n16_b4096_gte": 0.75,
                    "primary_log2_budget_auc_report_required": True,
                    "reference_bank_curve_count_range": [200, 500],
                },
            ),
            "all34_plumbing": _stage(
                topology_schedule="all34",
                recipes=full_factorial_prefix_recipe_count("all34"),
                training_augmentation_views=3,
                seeds=1,
                purpose="throughput_loss_and_nan_checks_only",
                gates={
                    "performance_claim_allowed": False,
                    "required_smoke_metrics": [
                        "frozen_search_yield_ranking_recall",
                        "at_least_one_exact_compatible_candidate_rate",
                        "reference_representative_recall",
                        "primary_log2_budget_auc",
                    ],
                },
            ),
            "engineering_e3": _stage(
                topology_schedule="all34",
                recipes=235_008,
                training_augmentation_views=3,
                seeds=1,
                purpose="learning_curve_and_failure_strata",
                gates={
                    "all_34_topologies_and_contextual_catalog_audited": True,
                    "frozen_search_yield_ranking_recall_at_16_gte": 0.80,
                    "at_least_one_exact_compatible_candidate_rate_gte": 0.80,
                    "reference_representative_recall_at_n16_b4096_gte": 0.65,
                    "primary_log2_budget_auc_report_required": True,
                },
            ),
            "paper_id_train": _stage(
                topology_schedule="all34",
                recipes=940_032,
                training_augmentation_views=2,
                seeds=5,
                purpose="frozen_paper_training_minimum",
                gates={
                    "full_epochs_minimum": 1,
                    "all_completed_full_epoch_candidates_retained": True,
                    "checkpoint_selected_by_tuning_exact_budget_auc_and_ttfc": True,
                    "training_scale_recipe_checkpoints": [
                        14_688,
                        58_752,
                        235_008,
                        940_032,
                    ],
                    "open_larger_scale_only_if_235008_to_940032_auc_gain_gt": 0.02,
                },
            ),
        },
        "final_acceptance": {
            "bounds_and_physics_compliance": 1.0,
            "paper_model_explicitly_conditions_all_geometry_and_amplitude_ranges": True,
            "geometry_and_amplitude_graph_dependency_test_required": True,
            "amplitude_range_conditioning_audit_coverage": 1.0,
            "proposal_and_exact_solver_share_identical_gui_amplitude_ranges": True,
            "frozen_search_yield_ranking_recall_report_required": True,
            "at_least_one_exact_compatible_candidate_rate_report_required": True,
            "reference_representative_recall_at_n16_b4096_gte": 0.80,
            "primary_log2_budget_auc_report_required": True,
            "compatible_candidate_yield_report_required": True,
            "unique_compatible_representative_yield_report_required": True,
            "reference_match_fraction_lower_bound_report_required": True,
            "unmatched_compatible_representatives_report_required": True,
            "duplicate_parameter_representative_rate_lt": 0.10,
            "compatibility_target_coverage": 0.95,
            "minimum_independent_calibration_recipes_per_stratum": 200,
        },
        "paper_evaluation_sets": {
            "test": {
                "minimum_total_clean_parent_design_points": 70_000,
                "target_total_clean_parent_design_points": 140_000,
                "minimum_independent_scramble_replicates": (
                    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
                ),
                "target_independent_scramble_replicates": (
                    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
                ),
                "each_replicate_point_count_is_power_of_two": True,
                "observation_views_per_recipe": 1,
            },
            "calibration": {
                "minimum_total_clean_parent_design_points": 64_000,
                "target_total_clean_parent_design_points": 160_000,
                "sampling_design": "iid_exchangeable_within_frozen_calibration_strata",
                "independent_clean_parent_recipes_are_inferential_units": True,
                "randomized_sobol_or_qmc_points_allowed_for_threshold_fitting": False,
                "reachable_design_strata": PREREGISTERED_DESIGN_STRATUM_COUNT,
                "design_stratum_universe_version": DESIGN_STRATUM_UNIVERSE_VERSION,
                "design_stratum_universe_sha256": DESIGN_STRATUM_UNIVERSE_SHA256,
                "observation_views_per_recipe": 1,
            },
            "reference": {
                "minimum_curves": 500,
                "target_curves": 1_000,
                "reference_search_seed_count": 3,
                "minimum_independent_scramble_replicates": (
                    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
                ),
                "target_independent_scramble_replicates": (
                    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
                ),
                "scramble_replicates_are_inference_units": True,
                "clean_parent_sobol_block_is_disjoint": True,
            },
            "ood": {
                "sets": list(OOD_LABELS),
                "included_in_id_primary_endpoint": False,
                "minimum_independent_scramble_replicates": (
                    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
                ),
                "target_independent_scramble_replicates": (
                    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
                ),
                "each_replicate_point_count_is_power_of_two": True,
            },
            "external_real_case_study": {
                "member_of_synthetic_sobol_plan": False,
                "minimum_cases": 8,
                "target_cases": 20,
                "minimum_independent_blinded_experts": 2,
                "required_case_types": [
                    "in_domain_case",
                    "predeclared_model_misspecification_or_negative_control",
                    "truncated_q_or_weak_overlap_case",
                ],
                "success_only_case_selection_allowed": False,
                "all_observed_pipeline_failures_retained": True,
                "failed_cases_remain_in_declared_case_denominator": True,
                "parameter_accuracy_without_ground_truth": False,
            },
        },
        "model_family_promotion": {
            "order": [
                "v5_2_branch_conditioned_diagonal_logistic_normal_mdn",
                "v5_2_branch_conditioned_low_rank_or_full_covariance_mdn",
                "v5_2_branch_conditioned_normalizing_flow",
                "v5_2_branch_conditioned_diffusion_or_score_model",
            ],
            "paired_budget_recall_auc_gain_gte": 0.02,
            "paired_independent_rqmc_student_t_95pct_lower_bound_gt": 0.0,
            "proposal_latency_ratio_lte": 2.0,
            "end_to_end_p90_latency_ratio_lte": 1.25,
            "test_split_may_select_family": False,
        },
        "paper_checkpoint_selection": {
            "selector_implementation": {
                "module": "PosteriorV8.paper_checkpoint_selector_v5",
                "schema": V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA,
                "version": V5_PAPER_CHECKPOINT_SELECTOR_VERSION,
                "rule_id": V5_PAPER_CHECKPOINT_SELECTION_RULE_ID,
                "tuning_summary_schema": V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
                "selection_split": V5_TUNING_SELECTION_SPLIT,
                "checkpoint_method_binding_schema": (
                    V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA
                ),
                "checkpoint_bound_method_protocol_id": (V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID),
                "selection_receipt_claim": V5_SELECTION_ONLY_STATUS,
            },
            "full_epochs_must_be_greater_than_zero": True,
            "warmup_only_run_is_paper_model_eligible": False,
            "retain_every_completed_full_epoch_candidate": True,
            "candidate_identity_requires_model_and_source_sha256": True,
            "selection_split": "tuning_validation",
            "primary_selection_metric": (
                "maximize_recipe_macro_n16_exact_forward_log2_budget_recall_auc"
            ),
            "secondary_selection_metric": (
                "minimize_median_exact_forward_calls_to_first_exact_compatible_verified_return"
            ),
            "tertiary_selection_metric": (
                "minimize_median_wall_clock_time_to_first_exact_compatible_verified_return"
            ),
            "final_tie_break": "earliest_full_epoch_then_lexicographic_artifact_sha256",
            "training_validation_loss_role": "diagnostic_only_not_final_checkpoint_selector",
            "test_reference_calibration_or_real_data_may_select_checkpoint": False,
            "current_bound_training_promotion_enabled": (
                formal_production["training_promotion_enabled_in_bound_implementation"]
            ),
            "paper_training_blocked_until_candidate_retention_and_selector_are_enforced": (True),
        },
        "forward_generalization_contract": {
            "id_synthetic_train_and_test_share_forward_family": True,
            "same_forward_id_result_scope": (
                "algorithmic_candidate_discovery_under_the_nominal_empirical_forward_only"
            ),
            "same_forward_id_design_removes_inverse_crime": False,
            "alternate_forward_ood_required_before_physics_generalization_claim": True,
            "alternate_forward_ood": (
                "independently_implemented_or_physics_perturbed_forward_with_frozen_"
                "parameter_mapping_and_acquisition_policy"
            ),
            "bornagain_challenge_required_where_parameter_mapping_is_valid": True,
            "alternate_forward_executable_frozen": False,
            "bornagain_challenge_executable_frozen": False,
            "physics_generalization_claim_currently_allowed": False,
            "promotion_requirement": (
                "freeze_executable_generator_parameter_mapping_sample_inventory_"
                "failure_and_abstention_rules_before_opening_challenge_results"
            ),
            "alternate_forward_or_bornagain_used_for_training_calibration_or_selection": (False),
            "report_nominal_alternate_forward_and_real_case_results_separately": True,
            "failure_or_model_discrepancy_cases_must_remain_in_denominator": True,
        },
        "reference_search": {
            "starts": [1, 2, 4, 8, 16, 32],
            "final_budget_doubling_new_representative_fraction_lt": 0.01,
            "three_seed_representative_set_jaccard_gte": 0.98,
            "network_proposals_allowed_in_primary_reference": False,
            "primary_sources": [
                "low_discrepancy_multistart",
                "independent_global_optimizer_gold_subset",
                "gui_consistent_exact_refinement",
            ],
            "joint_gold_optimizer": {
                "required_for_stratified_gold_subset": True,
                "variables": "nonlinear_geometry_plus_jointly_optimized_linear_coefficients",
                "coefficient_polytope_schema": COEFFICIENT_POLYTOPE_SCHEMA,
                "amplitude_constraint_version": GUI_AMPLITUDE_CONSTRAINT_VERSION,
                "canonical_amplitude_gauge": CANONICAL_AMPLITUDE_GAUGE,
                "gui_forward_formula": ("BG+k*(sum_i(Int_i*P_i*S_i)+int_Res*R)"),
                "independent_gui_parameter_ranges": [
                    "BG",
                    "k",
                    "Int_i",
                    "int_Res_when_resolution_present",
                ],
                "auxiliary_projection": (
                    "retain_exists_kappa_in_k_range_and_project_"
                    "Int_low*kappa_lte_a_lte_Int_high*kappa_exactly"
                ),
                "returned_snapshot_persists_explicit_k_witness": True,
                "k_equals_sum_a_i": False,
                "constraints": (
                    "full_coupled_GUI_BG_k_Int_i_and_optional_int_Res_ranges_"
                    "not_independent_coefficient_axis_boxes"
                ),
                "comparison": "paired_against_sequential_profile_refine_from_same_branch_seeds",
            },
            "evaluated_method_union_role": "secondary_leave_one_method_out_sensitivity_only",
            "primary_reference_semantics": (
                "all_exact_compatible_representatives_including_observability_unknown_"
                "and_confirmed_redundant"
            ),
            "strict_minimal_reference_role": "secondary_sensitivity_only",
            "mathematical_completeness_claim_allowed": False,
        },
    }


def protocol_payload() -> dict[str, object]:
    """Return a fresh protocol payload with a self-verifying digest."""

    payload = _protocol_core()
    payload["protocol_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return deepcopy(payload)


def validate_protocol(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate exact schema identity and the embedded canonical digest."""

    if not isinstance(payload, Mapping):
        raise TypeError("study protocol must be a mapping")
    value = deepcopy(dict(payload))
    digest = value.pop("protocol_sha256", None)
    if value != _protocol_core():
        raise ValueError("study protocol content is unsupported or has been modified")
    expected = sha256(canonical_json_bytes(value)).hexdigest()
    if digest != expected:
        raise ValueError("study protocol SHA256 is invalid")
    value["protocol_sha256"] = expected
    return value


def write_protocol(path: str | os.PathLike[str]) -> Path:
    """Atomically create one protocol JSON and never replace an existing path."""

    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"protocol parent directory does not exist: {target.parent}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite study protocol: {target}")
    serialized = canonical_json_bytes(protocol_payload()) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite study protocol: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = write_protocol(args.output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    print(validate_protocol(saved)["protocol_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXACT_FORWARD_BUDGETS",
    "OUTPUT_CAPS",
    "PRIMARY_ENDPOINT_NAME",
    "PRIMARY_OUTPUT_CAP",
    "STUDY_PROTOCOL_SCHEMA",
    "STUDY_PROTOCOL_STATUS",
    "STUDY_PROTOCOL_VERSION",
    "V5_OBJECTIVE_MODULE",
    "V5_OBJECTIVE_SCHEMA",
    "V5_OBJECTIVE_VERSION",
    "protocol_payload",
    "validate_protocol",
    "write_protocol",
]
