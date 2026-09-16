from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import (
    AMPLITUDE_QUERY_EMBEDDING_DIM,
    V5_AMPLITUDE_EMBEDDING_VERSION,
    V5_AMPLITUDE_QUERY_SCHEMA,
    V5_AMPLITUDE_QUERY_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
    V5_AMPLITUDE_RANGE_REGIMES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_V5_SCHEMA,
    CANDIDATE_SUPERVISION_V5_VERSION,
    SEARCH_OUTCOME_STATES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_refinement_contract_v5 import (
    V5_EXACT_REFINEMENT_SCHEMA,
    V5_EXACT_REFINEMENT_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_component_slots import (
    CANONICAL_COMPONENT_SLOTS_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrated_search_threshold_v5 import (
    V5_CALIBRATION_IDENTITY_SCHEMA,
    V5_CALIBRATION_IDENTITY_VERSION,
    V5_OBSERVATION_THRESHOLD_SCHEMA,
    V5_OBSERVATION_THRESHOLD_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    ACQUISITION_POLICY_ID_VERSION,
    ACQUISITION_POLICY_REQUIRED_COMPONENTS,
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    DESIGN_STRATUM_UNIVERSE_FIELDS,
    DESIGN_STRATUM_UNIVERSE_SHA256,
    DESIGN_STRATUM_UNIVERSE_VERSION,
    PREREGISTERED_DESIGN_STRATUM_COUNT,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_branch_catalog import (
    CONTEXTUAL_BRANCH_CATALOG_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_reference_bank_v5 import (
    V5_CONTEXTUAL_DEDUPLICATION_VERSION,
    V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
    V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
    V5_CONTEXTUAL_REFERENCE_CLAIM,
    V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION,
    V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_executor_v5 import (
    V5_EXACT_SEARCH_COMPLETION_RULE,
    V5_EXACT_SEARCH_EVALUATOR_VERSION,
    V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
    V5_EXACT_SEARCH_EXECUTOR_VERSION,
    V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
    V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.exact_search_schedule_v5 import (
    V5_EXACT_SEARCH_OPTIMIZER_SCHEMA,
    V5_EXACT_SEARCH_OPTIMIZER_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gui_amplitude_constraints import (
    CANONICAL_AMPLITUDE_GAUGE,
    COEFFICIENT_POLYTOPE_SCHEMA,
    GUI_AMPLITUDE_CONSTRAINT_SCHEMA,
    GUI_AMPLITUDE_CONSTRAINT_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import (
    MODEL_V5_INPUT_KEYS,
    MODEL_V5_NAME,
    MODEL_V5_OUTPUT_KEYS,
    MODEL_V5_SCHEMA,
    MODEL_V5_SCORE_SEMANTICS,
    MODEL_V5_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_budget_evaluator_v5 import (
    V5_EQUIVALENCE_MATCHING_VERSION,
    V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE,
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_checkpoint_selector_v5 import (
    V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID,
    V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA,
    V5_PAPER_CHECKPOINT_SELECTION_RULE_ID,
    V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA,
    V5_PAPER_CHECKPOINT_SELECTOR_VERSION,
    V5_SELECTION_ONLY_STATUS,
    V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
    V5_TUNING_SELECTION_SPLIT,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    REFERENCE_QUALIFICATION_SCHEMA,
    REFERENCE_QUALIFICATION_STATUSES,
    REFERENCE_QUALIFICATION_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_payload_v5 import (
    V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
    V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_rqmc_design_v5 import (
    V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT,
    V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED,
    V5_PAPER_RQMC_DESIGN_SCHEMA,
    V5_PAPER_RQMC_DESIGN_VERSION,
    V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_INFERENCE_RULE,
    V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES,
    V5_PAPER_RQMC_RANDOMIZATION_UNIT,
    V5_PAPER_RQMC_SPLITS,
    V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_rqmc_evaluator_v5 import (
    V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE,
    V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE,
    V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA,
    V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION,
    V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS,
    V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA,
    V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION,
    V5_PAPER_RQMC_SHARED_BASELINE_POLICY,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_bound_evaluation_v5 import (
    V5_QUERY_BOUND_EVALUATION_SCHEMA,
    V5_QUERY_BOUND_EVALUATION_SHA256,
    V5_QUERY_BOUND_EVALUATION_VERSION,
    V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    MAIN_SPLITS,
    OOD_LABELS,
    V5_SPLIT_PLAN_SCHEMA,
    V5_SPLIT_PLAN_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_amplitude_recipe_v5 import (
    V5_DIRECT_AMPLITUDE_SCHEMA,
    V5_DIRECT_AMPLITUDE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    V5_SOBOL_RECIPE_DIM,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    v5_numeric_policy_payload,
    v5_numeric_policy_sha256,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.study_protocol import (
    EXACT_FORWARD_BUDGETS,
    OUTPUT_CAPS,
    PRIMARY_ENDPOINT_NAME,
    STUDY_PROTOCOL_SCHEMA,
    STUDY_PROTOCOL_STATUS,
    STUDY_PROTOCOL_VERSION,
    V5_OBJECTIVE_SCHEMA,
    V5_OBJECTIVE_VERSION,
    protocol_payload,
    validate_protocol,
    write_protocol,
)
import utils.ML_Fitting_1D_GISAXS.PosteriorV8.study_protocol as study_protocol_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    UNCERTAINTY_KINDS,
    V5_UNCERTAINTY_SCHEMA,
    V5_UNCERTAINTY_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_inference_contract_v5 import (
    V5_UNIVERSAL_BRANCH_RANKING,
    V5_UNIVERSAL_INFERENCE_SCHEMA,
    V5_UNIVERSAL_INFERENCE_VERSION,
    V5_UNIVERSAL_SEED_SCHEDULE,
)


def test_methods_template_binds_v5_2_identity_before_holdout() -> None:
    payload = protocol_payload()
    assert STUDY_PROTOCOL_SCHEMA.endswith("/v20")
    assert STUDY_PROTOCOL_VERSION.endswith("_v20")
    assert STUDY_PROTOCOL_STATUS.startswith("methods_template_")
    status = payload["status_contract"]
    assert status["document_kind"] == "methods_template"
    assert status["design_must_be_frozen_before_holdout_inspection"]
    assert status["external_timestamped_preregistration_claimed"] is False
    assert status["concrete_run_plan_frozen"] is False
    assert status["trained_model_artifact_frozen"] is False
    assert status["performance_claim_allowed_from_template_alone"] is False
    assert "preregister" not in json.dumps(payload, sort_keys=True).lower()
    model = payload["model_contract"]
    assert model["schema"] == MODEL_V5_SCHEMA
    assert model["version"] == MODEL_V5_VERSION
    assert model["name"] == MODEL_V5_NAME
    assert model["input_keys"] == list(MODEL_V5_INPUT_KEYS)
    assert model["output_keys"] == list(MODEL_V5_OUTPUT_KEYS)
    assert model["output_keys"][0] == "proposal_search_yield_logit"
    assert model["score_semantics"] == MODEL_V5_SCORE_SEMANTICS
    assert model["contextual_branch_catalog_version"] == CONTEXTUAL_BRANCH_CATALOG_VERSION
    assert model["candidate_supervision_schema"] == CANDIDATE_SUPERVISION_V5_SCHEMA
    assert model["candidate_supervision_version"] == CANDIDATE_SUPERVISION_V5_VERSION
    assert model["candidate_supervision_outcomes"] == list(SEARCH_OUTCOME_STATES)
    assert model["uncertainty_schema"] == V5_UNCERTAINTY_SCHEMA
    assert model["uncertainty_version"] == V5_UNCERTAINTY_VERSION
    assert model["uncertainty_states"] == list(UNCERTAINTY_KINDS)
    assert model["schema"].endswith("/v5.2")

    identities = payload["implementation_identity"]
    assert identities["proposal_model"] == {
        "schema": MODEL_V5_SCHEMA,
        "version": MODEL_V5_VERSION,
        "name": MODEL_V5_NAME,
    }
    assert identities["gui_amplitude_constraint"] == {
        "schema": GUI_AMPLITUDE_CONSTRAINT_SCHEMA,
        "version": GUI_AMPLITUDE_CONSTRAINT_VERSION,
        "coefficient_polytope_schema": COEFFICIENT_POLYTOPE_SCHEMA,
        "shared_gauge_contract": CANONICAL_AMPLITUDE_GAUGE,
    }
    assert identities["training_objective"] == {
        "module": "PosteriorV8.training_objective_v5",
        "schema": V5_OBJECTIVE_SCHEMA,
        "version": V5_OBJECTIVE_VERSION,
        "imported_at_manifest_runtime": False,
        "reason": "keep_manifest_tensorflow_free",
    }
    assert identities["component_slot_canonicalization"] == {
        "version": CANONICAL_COMPONENT_SLOTS_VERSION,
        "equivalence_source": (
            "contextual_branch_catalog.component_slot_contract_equivalence_classes"
        ),
        "requires_geometry_D_and_component_intensity_range_context": True,
        "missing_component_intensity_context_policy": "singleton_slots_no_permutation",
    }
    assert identities["exact_refinement"] == {
        "schema": V5_EXACT_REFINEMENT_SCHEMA,
        "version": V5_EXACT_REFINEMENT_VERSION,
        "optimization_coordinates": "codec_varying_indices_only",
        "fixed_active_axes_are_template_coordinates": True,
        "zero_varying_axes_policy": "one_exact_profile_no_scipy",
    }
    assert identities["exact_search_optimizer"] == {
        "schema": V5_EXACT_SEARCH_OPTIMIZER_SCHEMA,
        "version": V5_EXACT_SEARCH_OPTIMIZER_VERSION,
        "finite_difference_budget_denominator": "varying_dimension_count_plus_one",
    }

    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective_v5 import (
        CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA,
        CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION,
    )

    assert V5_OBJECTIVE_SCHEMA == CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA
    assert V5_OBJECTIVE_VERSION == CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_execution_policy_v5 import (
        V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    )

    assert identities["proposal_execution_policy"]["sha256"] == (
        V5_PROPOSAL_EXECUTION_POLICY_SHA256
    )
    assert identities["proposal_execution_policy"][
        "bound_by_grouped_training_and_universal_inference"
    ] is True
    assert validate_protocol(payload) == payload


def test_paper_identity_names_the_limited_forward_authority() -> None:
    payload = protocol_payload()
    identity = payload["paper_identity"]
    assert identity["preferred_title"] == (
        "Forward-Model-Verified Amortized Multisolution Inversion for "
        "User-Constrained Empirical 1D GISAXS Cuts"
    )
    assert identity["required_forward_terminology"] == (
        "authoritative non-neural versioned empirical forward model"
    )
    assert identity["disallowed_title_claim"] == "Physics-Verified"
    assert "Physics-Verified" not in identity["preferred_title"]
    assert identity["absolute_first_ever_claim_allowed"] is False
    assert identity["novelty_search_is_mathematically_exhaustive"] is False
    assert "equal_exact_forward_budget" in identity["conservative_novelty_scope"]

    authority = payload["authoritative_forward_contract"]
    assert authority["authority"] == "authoritative_non_neural_versioned_empirical_forward_model"
    assert authority["neural_role"] == "amortized_candidate_proposal_and_search_yield_ranking_only"
    assert authority["forward_version_and_digest_required_in_every_paper_artifact"]
    assert authority["neural_surrogate_may_replace_final_exact_verification"] is False
    assert "not_full_physics" in authority["agreement_scope"]


def test_v5_2_amplitude_range_conditioning_and_independent_gui_scale_are_hard_gates() -> None:
    payload = protocol_payload()
    requirement = payload["model_contract"]["paper_training_v5_2_requirement"]
    assert requirement["status"] == "satisfied_by_bound_v5_2_model_contract"
    assert payload["model_contract"]["schema"].endswith("/v5.2")
    assert requirement["geometry_embedding_input"] == "geometry_bounds_embedding"
    assert requirement["amplitude_embedding_input"] == "amplitude_bounds_embedding"
    assert requirement["geometry_embedding_input"] in payload["model_contract"]["input_keys"]
    assert requirement["amplitude_embedding_input"] in payload["model_contract"]["input_keys"]
    assert requirement["amplitude_query_schema"] == V5_AMPLITUDE_QUERY_SCHEMA
    assert requirement["amplitude_query_version"] == V5_AMPLITUDE_QUERY_VERSION
    assert requirement["amplitude_embedding_version"] == V5_AMPLITUDE_EMBEDDING_VERSION
    assert requirement["amplitude_embedding_dimension"] == AMPLITUDE_QUERY_EMBEDDING_DIM
    assert requirement["paper_training_or_holdout_allowed_without_this_contract"] is False
    assert requirement["paper_holdout_allowed_before_artifact_identity_freeze"] is False
    assert requirement["explicit_query_conditioning"] == [
        "geometry_ranges",
        "BG_range",
        "k_range",
        "Int_i_ranges",
        "int_Res_range_when_resolution_present",
    ]
    assert requirement["new_model_schema_version_and_artifact_digest_required"] is True
    assert requirement["geometry_and_amplitude_graph_dependency_audit_required"] is True
    acceptance = payload["final_acceptance"]
    assert acceptance["paper_model_explicitly_conditions_all_geometry_and_amplitude_ranges"]
    assert acceptance["geometry_and_amplitude_graph_dependency_test_required"]
    assert acceptance["amplitude_range_conditioning_audit_coverage"] == 1.0
    ablations = payload["required_ablations"]
    assert "remove_BG_k_Int_i_int_Res_query_conditioning_but_keep_exact_polytope" in ablations

    generation = payload["implementation_identity"]["scientific_data_generation"]
    assert generation["amplitude_query_sampler_version"] == V5_AMPLITUDE_QUERY_SAMPLER_VERSION
    assert generation["amplitude_range_assignment_schema"] == (V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA)
    assert generation["amplitude_range_assignment_version"] == (
        V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION
    )
    assert generation["amplitude_range_regimes"] == list(V5_AMPLITUDE_RANGE_REGIMES)
    assert generation["active_amplitude_axes_receive_independent_range_assignments"]
    assert generation["direct_amplitude_schema"] == V5_DIRECT_AMPLITUDE_SCHEMA
    assert generation["direct_amplitude_version"] == V5_DIRECT_AMPLITUDE_VERSION
    assert generation["sobol_numeric_policy_version"] == (
        V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
    )
    assert generation["sobol_numeric_policy_contract"] == v5_numeric_policy_payload(
        V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
    )
    assert generation["sobol_numeric_policy_sha256"] == v5_numeric_policy_sha256(
        V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
    )
    assert generation["sobol_coordinate_schema"] == V5_SOBOL_RECIPE_COORDINATE_SCHEMA
    assert generation["sobol_coordinate_version"] == V5_SOBOL_RECIPE_COORDINATE_VERSION
    assert generation["sobol_coordinate_sha256"] == V5_SOBOL_RECIPE_COORDINATE_SHA256
    assert generation["sobol_coordinate_dimension"] == V5_SOBOL_RECIPE_DIM == 168

    forward = payload["authoritative_forward_contract"]
    assert forward["gui_formula"] == ("I(q)=BG+k*(sum_i(Int_i*P_i(q)*S_i(q))+int_Res*R(q))")
    assert forward["gui_BG_k_Int_i_and_int_Res_are_independent_inputs"]
    assert forward["effective_linear_coefficients"].startswith("a_i=k*Int_i")
    assert "exists_kappa" in forward["shared_auxiliary_projection"]
    assert forward["k_equals_sum_particle_coefficients_assumption_allowed"] is False


def test_protocol_binds_reachable_calibration_universe_and_formal_single_view_policy() -> None:
    payload = protocol_payload()
    calibration = payload["compatibility_calibration"]
    assert PREREGISTERED_DESIGN_STRATUM_COUNT == 60
    assert calibration["schema"] == CALIBRATION_SCHEMA
    assert calibration["version"] == CALIBRATION_VERSION
    assert calibration["reachable_design_strata"] == 60
    assert calibration["compatibility_stratum_version"] == COMPATIBILITY_STRATUM_VERSION
    assert calibration["compatibility_stratum_fields"] == list(COMPATIBILITY_STRATUM_FIELDS)
    assert calibration["design_stratum_universe_version"] == DESIGN_STRATUM_UNIVERSE_VERSION
    assert calibration["design_stratum_universe_fields"] == list(DESIGN_STRATUM_UNIVERSE_FIELDS)
    assert calibration["design_stratum_universe_sha256"] == (DESIGN_STRATUM_UNIVERSE_SHA256)
    assert calibration["acquisition_policy_id_version"] == ACQUISITION_POLICY_ID_VERSION
    assert calibration["acquisition_policy_required_components"] == list(
        ACQUISITION_POLICY_REQUIRED_COMPONENTS
    )
    assert calibration["formal_lookup_requires_full_policy_membership_in_selected_stratum"]

    identities = payload["implementation_identity"]
    assert identities["compatibility_calibration"] == {
        "schema": CALIBRATION_SCHEMA,
        "version": CALIBRATION_VERSION,
        "checked_identity_schema": V5_CALIBRATION_IDENTITY_SCHEMA,
        "checked_identity_version": V5_CALIBRATION_IDENTITY_VERSION,
        "observation_threshold_schema": V5_OBSERVATION_THRESHOLD_SCHEMA,
        "observation_threshold_version": V5_OBSERVATION_THRESHOLD_VERSION,
    }
    observation = identities["formal_label_observation"]
    assert observation["schema"] == V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA
    assert observation["version"] == V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION
    assert observation["policy_id"] == V5_FORMAL_LABEL_OBSERVATION_POLICY_ID
    assert observation["policy_sha256"] == V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256
    assert observation["candidate_view_indices"] == list(
        V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES
    )
    assert observation["selected_sigma_present_views_per_recipe"] == 1
    view_contract = payload["observation_view_contract"]
    assert view_contract["formal_search_label_views_per_recipe"] == 1
    assert view_contract["training_augmentation_view_count_changes_formal_label_count"] is False
    for stage in payload["stages"].values():
        assert stage["formal_search_label_views_per_recipe"] == 1
        assert stage["formal_label_candidate_view_indices"] == [0, 1]
        assert stage["training_augmentation_views_per_recipe"] >= 1


def test_formal_production_evaluator_distance_and_scheduler_identities_are_bound() -> None:
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_contract_v5 import (
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
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_membership_v5 import (
        V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_plan_v5 import (
        V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA,
        V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION,
        V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_production_search_runtime_v5 import (
        V5FormalProductionExecutableShard,
    )

    identity = protocol_payload()["implementation_identity"]
    formal = identity["formal_production_search"]
    assert formal["source_schema"] == V5_FORMAL_PRODUCTION_SOURCE_SCHEMA
    assert formal["source_version"] == V5_FORMAL_PRODUCTION_SOURCE_VERSION
    assert formal["stage_schema"] == V5_FORMAL_PRODUCTION_STAGE_SCHEMA
    assert formal["stage_version"] == V5_FORMAL_PRODUCTION_STAGE_VERSION
    assert formal["stage_ids"] == list(V5_FORMAL_PRODUCTION_STAGE_IDS)
    assert formal["shard_schema"] == V5_FORMAL_PRODUCTION_SHARD_SCHEMA
    assert formal["shard_version"] == V5_FORMAL_PRODUCTION_SHARD_VERSION
    assert formal["allowed_splits"] == list(V5_FORMAL_PRODUCTION_ALLOWED_SPLITS)
    assert formal["query_contract_sha256"] == V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256
    assert formal["plan_schema"] == V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA
    assert formal["plan_version"] == V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION
    assert formal["membership_proof_schema"] == (V5_FORMAL_PRODUCTION_MEMBERSHIP_PROOF_SCHEMA)
    assert formal["runtime_module"] == V5FormalProductionExecutableShard.__module__
    assert formal["runtime_type"] == V5FormalProductionExecutableShard.__name__
    assert formal["training_promotion_enabled_in_bound_implementation"] is (
        V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED
    )

    distance = identity["query_parameter_distance"]
    assert distance["schema"] == V5_QUERY_PARAMETER_DISTANCE_SCHEMA
    assert distance["version"] == V5_QUERY_PARAMETER_DISTANCE_VERSION
    assert distance["sha256"] == V5_QUERY_PARAMETER_DISTANCE_SHA256
    executor = identity["exact_search_executor"]
    assert executor == {
        "schema": V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
        "version": V5_EXACT_SEARCH_EXECUTOR_VERSION,
        "evaluator_version": V5_EXACT_SEARCH_EVALUATOR_VERSION,
        "representative_distance_id": V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
        "representative_distance_sha256": (V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256),
        "completion_rule": V5_EXACT_SEARCH_COMPLETION_RULE,
    }
    reference_bank = identity["contextual_reference_bank"]
    assert reference_bank == {
        "schema": V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
        "version": V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
        "distance_version": V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION,
        "slot_equivalence_version": V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION,
        "deduplication_version": V5_CONTEXTUAL_DEDUPLICATION_VERSION,
        "claim": V5_CONTEXTUAL_REFERENCE_CLAIM,
    }
    evaluator = identity["paper_budget_evaluator"]
    assert evaluator["schema"] == V5_PAPER_BUDGET_EVALUATOR_SCHEMA
    assert evaluator["version"] == V5_PAPER_BUDGET_EVALUATOR_VERSION
    assert evaluator["matching_version"] == V5_EQUIVALENCE_MATCHING_VERSION
    assert evaluator["paired_bootstrap_claim_scope"] == V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE
    assert evaluator["actual_emitted_representative_payload_is_primary_match_input"]
    assert evaluator["hidden_cluster_members_may_raise_primary_recall"] is False
    assert evaluator["representative_payload_schema"] == V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA
    assert evaluator["representative_payload_version"] == (
        V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION
    )
    assert evaluator["one_payload_contains_exactly_one_typed_parameter_representative"]
    assert evaluator["cross_query_reference_or_emission_payload_allowed"] is False
    rqmc_design = identity["paper_rqmc_design"]
    assert rqmc_design["schema"] == V5_PAPER_RQMC_DESIGN_SCHEMA
    assert rqmc_design["version"] == V5_PAPER_RQMC_DESIGN_VERSION
    assert rqmc_design["engineering_minimum_independent_replicates"] == (
        V5_PAPER_RQMC_ENGINEERING_MIN_INDEPENDENT_REPLICATES
    )
    assert rqmc_design["engineering_minimum_allows_formal_inference"] is False
    assert rqmc_design["formal_minimum_independent_replicates"] == (
        V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
    )
    assert rqmc_design["paper_target_independent_replicates"] == (
        V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
    )
    assert rqmc_design["randomization_unit"] == V5_PAPER_RQMC_RANDOMIZATION_UNIT
    assert rqmc_design["formal_splits"] == list(V5_PAPER_RQMC_SPLITS)
    assert rqmc_design["finite_sample_conformal_calibration_included"] is (
        V5_PAPER_RQMC_CONFORMAL_CALIBRATION_INCLUDED
    )
    assert rqmc_design["compatibility_calibration_sampling_requirement"] == (
        V5_COMPATIBILITY_CALIBRATION_SAMPLING_REQUIREMENT
    )
    assert rqmc_design["names_only_coordinate_identity_allowed"] is False
    rqmc_evaluator = identity["paper_rqmc_evaluator"]
    assert rqmc_evaluator["schema"] == V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA
    assert rqmc_evaluator["version"] == V5_PAPER_RQMC_PAIRED_EVALUATOR_VERSION
    assert rqmc_evaluator["training_artifact_set_schema"] == (
        V5_FROZEN_TRAINING_ARTIFACT_SET_SCHEMA
    )
    assert rqmc_evaluator["training_artifact_set_version"] == (
        V5_FROZEN_TRAINING_ARTIFACT_SET_VERSION
    )
    assert rqmc_evaluator["formal_training_artifact_set_scope"] == (
        V5_FROZEN_TRAINING_ARTIFACT_SET_FORMAL_SCOPE
    )
    assert rqmc_evaluator["engineering_training_artifact_set_scope"] == (
        V5_FROZEN_TRAINING_ARTIFACT_SET_ENGINEERING_SCOPE
    )
    assert rqmc_evaluator["formal_minimum_predeclared_training_seeds"] == (
        V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS
    )
    assert rqmc_evaluator["engineering_tiny_artifact_set_may_enter_formal_summary"] is False
    assert rqmc_evaluator["primary_solver_baseline_policy"] == (
        V5_PAPER_RQMC_SHARED_BASELINE_POLICY
    )
    assert rqmc_evaluator["point_bootstrap_available"] is False
    inference = identity["one_click_inference"]
    assert inference["schema"] == V5_UNIVERSAL_INFERENCE_SCHEMA
    assert inference["version"] == V5_UNIVERSAL_INFERENCE_VERSION
    assert inference["branch_ranking"] == V5_UNIVERSAL_BRANCH_RANKING
    assert inference["seed_schedule"] == V5_UNIVERSAL_SEED_SCHEDULE
    assert inference["query_bound_evaluator"] == {
        "schema": V5_QUERY_BOUND_EVALUATION_SCHEMA,
        "version": V5_QUERY_BOUND_EVALUATION_VERSION,
        "sha256": V5_QUERY_BOUND_EVALUATION_SHA256,
        "reference_matching": V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
        "actual_emitted_representatives_only": True,
        "legacy_global_evaluator_is_product_endpoint": False,
    }

    selector = protocol_payload()["paper_checkpoint_selection"]["selector_implementation"]
    assert selector == {
        "module": "PosteriorV8.paper_checkpoint_selector_v5",
        "schema": V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA,
        "version": V5_PAPER_CHECKPOINT_SELECTOR_VERSION,
        "rule_id": V5_PAPER_CHECKPOINT_SELECTION_RULE_ID,
        "tuning_summary_schema": V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
        "selection_split": V5_TUNING_SELECTION_SPLIT,
        "checkpoint_method_binding_schema": (V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA),
        "checkpoint_bound_method_protocol_id": V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID,
        "selection_receipt_claim": V5_SELECTION_ONLY_STATUS,
    }


def test_protocol_replaces_global_classifiers_with_v5_search_and_local_metrics() -> None:
    payload = protocol_payload()
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "topology_top1",
        "topology_recall_at",
        "canonical_joint_branch_top1",
        "canonical_joint_branch_recall",
        "proposal_compatibility_logit",
    ):
        assert forbidden not in encoded
    memorization = payload["stages"]["k1_memorization"]["gates"]
    assert memorization == {
        "branch_conditioned_local_mdn_single_draw_local_rms_median_lt": 0.05,
        "branch_conditioned_local_mdn_best_of_32_local_rms_median_lt": 0.01,
        "branch_conditioned_local_mdn_best_of_32_local_rms_p90_lt": 0.03,
        "exact_post_refine_raw_log_rmse_p90_lt": 1.0e-3,
        "exact_post_refine_compatible_rate_gte": 0.99,
    }
    memorization_stage = payload["stages"]["k1_memorization"]
    assert memorization_stage["topology_schedule"] == "single_branch_sphere_pattern0"
    assert memorization_stage["purpose"] == "single_branch_capacity_and_objective_diagnostic"
    assert memorization_stage["metric_population_contract"] == {
        "local_mdn_metrics": (
            "learnable_parents_with_at_least_one_varying_coordinate_only"
        ),
        "fully_fixed_parents": (
            "retained_for_exact_forward_and_excluded_from_local_rms_"
            "without_zero_error_dilution"
        ),
        "exact_forward_metrics": "all_clean_parents",
        "minimum_learnable_parent_count": 1,
    }
    schedules = payload["topology_schedule_contract"]
    assert schedules["single_branch_sphere_pattern0"] == {
        "declared_topology": ["sphere"],
        "branch_pattern_id": 0,
        "d_present": [False],
        "resolution_present": False,
        "purpose": "isolated_representation_and_objective_capacity_diagnostic",
        "full_k1_coverage_claim_allowed": False,
    }
    assert schedules["k1"]["legal_branch_count"] == 12
    assert schedules["k1_k2"]["legal_branch_count"] == 60
    assert schedules["all34"]["legal_branch_count"] == 700
    for stage_name in ("engineering_e1", "engineering_e2", "engineering_e3"):
        gates = payload["stages"][stage_name]["gates"]
        assert any("frozen_search_yield_ranking_recall" in key for key in gates)
        assert any("exact_compatible_candidate_rate" in key for key in gates)
        assert any("reference_representative_recall" in key for key in gates)
    for stage in payload["stages"].values():
        for gate_name in stage["gates"]:
            normalized = gate_name.lower()
            assert "classifier" not in normalized
            assert not (
                "accuracy" in normalized and ("topology" in normalized or "pattern" in normalized)
            )
    comparisons = payload["required_method_comparisons"]
    assert "range_conditioned_single_point_mse_regression_plus_same_exact_polish" in comparisons
    assert "differential_evolution_or_genetic_algorithm_plus_same_exact_solver" in comparisons
    estimand = payload["search_yield_estimand"]
    assert estimand["unverified_enters_bce"] is False
    assert estimand["negative_requires_completed_frozen_search_and_budget_provenance"]
    assert not estimand["mathematical_branch_solvability"]
    assert not estimand["no_solution_certificate"]
    assert not estimand["posterior_probability"]
    ranking = payload["search_yield_ranking_endpoint"]
    assert "every_codec_feasible" in ranking["query_eligibility"]
    assert "count_zero" in ranking["partially_unverified_catalog"]
    assert "count_zero" in ranking["zero_positive_catalog"]
    assert ranking["outcome_label_search_uses_evaluated_model_score"] is False
    assert ranking["outcome_label_search_budget_schedule_frozen_before_scoring"]
    assert ranking["operational_not_branch_solvability"]
    uncertainty = payload["uncertainty_contract"]
    assert uncertainty["continuous_density_is_posterior"] is False
    assert uncertainty["post_refinement_outputs_enter_sbc"] is False
    assert (
        payload["local_mdn_diagnostic_contract"]["fixed_and_inactive_axes_enter_distance"] is False
    )
    mass_status = payload["model_contract"]["multisolution_mass_objective_status"]
    assert mass_status["matched_set_or_mass_aware_objective_implemented"] is True
    assert mass_status["matched_set_or_hungarian_implemented"] is False
    assert mass_status["current_objective_is_posterior_mass_calibrated"] is False
    assert mass_status["explicit_low_weight_ghost_mode_mass_penalty_implemented"] is True
    assert mass_status["mass_aware_performance_hypothesis_enabled"] is True


def test_primary_endpoint_thresholds_and_reference_population_remain_frozen() -> None:
    payload = protocol_payload()
    assert payload["exact_forward_budgets"] == list(EXACT_FORWARD_BUDGETS)
    endpoint = payload["primary_endpoint"]
    assert endpoint["name"] == PRIMARY_ENDPOINT_NAME
    assert endpoint["conditioning"] == "frozen_reference_search_qualification"
    assert endpoint["unconditional_population_performance_claimed"] is False
    assert endpoint["reported_output_caps"] == list(OUTPUT_CAPS)
    assert endpoint["primary_output_cap"] == 16
    assert endpoint["auc_formula_for_frozen_doubling_budgets"] == (
        "(0.5*R256+R512+R1024+R2048+0.5*R4096)/4"
    )
    assert "observability_unknown" in endpoint["reference_population"]
    assert "complete_linkage_diameter_delta" in endpoint["scientific_object"]
    assert "does_not_guarantee_pairwise_distance_gt_delta" in endpoint["cluster_semantics"]
    assert "filter_exact_compatible_and_verified" in endpoint["output_cap_filter_order"]
    assert endpoint["output_cap_counts_incompatible_or_unverified"] is False
    assert endpoint["incompatible_and_unverified_exact_calls_still_consume_B"]
    assert "actual_user_visible_emitted" in endpoint["matching_candidate_payload"]
    assert endpoint["matching_candidate_payload_schema"] == (
        V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA
    )
    assert endpoint["hidden_accepted_cluster_member_may_satisfy_primary_match"] is False
    qualification = payload["reference_qualification_and_missingness"]
    assert qualification["schema"] == REFERENCE_QUALIFICATION_SCHEMA
    assert qualification["version"] == REFERENCE_QUALIFICATION_VERSION
    assert qualification["typed_statuses"] == list(REFERENCE_QUALIFICATION_STATUSES)
    assert "imputed_zero" in qualification["complete_preselected_cohort_lower_bound"]
    assert (
        "plus_one_per_unresolved"
        in (qualification["complete_preselected_cohort_unresolved_universal_upper_bound"])
    )
    assert qualification["headline_blocked_when_frozen_stratified_qualification_gate_fails"]
    thresholds = payload["operational_representative_thresholds"]
    assert thresholds["parameter_complete_linkage_distance_max"] == 0.08
    assert thresholds["reference_match_distance_max"] == 0.10
    assert thresholds["raw_curve_equivalence_log_rmse_max"] == 0.02
    assert thresholds["sensitivity_multipliers"] == [0.5, 1.0, 2.0]
    assert thresholds["interpretation"] == "frozen_design_operational_not_physical_constants"
    assert thresholds["distance_schema"] == V5_QUERY_PARAMETER_DISTANCE_SCHEMA
    assert thresholds["distance_version"] == V5_QUERY_PARAMETER_DISTANCE_VERSION
    assert thresholds["distance_sha256"] == V5_QUERY_PARAMETER_DISTANCE_SHA256
    assert thresholds["distance_retains_absolute_scale"] == [
        "background",
        "effective_particle_total",
    ]
    assert thresholds["distance_quotients_only"] == "exact_shared_k_GUI_gauge"
    assert thresholds["complete_linkage_guarantee"] == "within_cluster_diameter_lte_delta"
    assert thresholds["between_representative_pairwise_gt_delta_guaranteed"] is False
    assert not payload["reference_search"]["network_proposals_allowed_in_primary_reference"]
    gold = payload["reference_search"]["joint_gold_optimizer"]
    assert gold["coefficient_polytope_schema"] == COEFFICIENT_POLYTOPE_SCHEMA
    assert gold["amplitude_constraint_version"] == GUI_AMPLITUDE_CONSTRAINT_VERSION
    assert gold["canonical_amplitude_gauge"] == CANONICAL_AMPLITUDE_GAUGE
    assert gold["variables"] == "nonlinear_geometry_plus_jointly_optimized_linear_coefficients"
    assert "BG_k_Int_i" in gold["constraints"]
    assert "int_Res" in gold["constraints"]
    assert "not_independent_coefficient_axis_boxes" in gold["constraints"]
    assert gold["returned_snapshot_persists_explicit_k_witness"]
    assert gold["k_equals_sum_a_i"] is False


def test_clean_parent_splits_statistics_comparisons_and_real_claims_are_explicit() -> None:
    payload = protocol_payload()
    split = payload["split_contract"]
    assert split["schema"] == V5_SPLIT_PLAN_SCHEMA
    assert split["version"] == V5_SPLIT_PLAN_VERSION
    expected_splits = [
        "train",
        "tuning_validation",
        "calibration",
        "test",
        "reference",
        "ood",
    ]
    assert list(MAIN_SPLITS) == expected_splits
    assert split["legacy_engineering_mutually_exclusive_clean_parent_sobol_blocks"] == (
        expected_splits
    )
    assert split["formal_paper_rqmc_splits"] == list(V5_PAPER_RQMC_SPLITS)
    assert "iid_exchangeable" in split["formal_paper_calibration_partition"]
    assert split["ood_subblocks"] == list(OOD_LABELS)
    assert "clean_parent" in split["group_by"]
    assert set(payload["paper_evaluation_sets"]) == {
        "test",
        "calibration",
        "reference",
        "ood",
        "external_real_case_study",
    }

    comparisons = set(payload["required_method_comparisons"])
    assert {
        "legacy_single_or_fixed_multihead_regression",
        "topology_classifier_then_top_k_topologies_then_exact_solver",
        "solver_only_sobol_or_independent_global_search",
        "pure_sobol_multistart_exact_solver_ensemble",
        "retrieval_seeded_exact_solver",
        "matched_branch_conditioned_conditional_flow_or_cinn",
        "exact_forward_mcmc_or_nuts_stratified_bayesian_reference",
        "v5_2_branch_conditioned_mdn_only_without_exact_refinement",
        "v5_2_search_yield_ranked_mdn_plus_exact_refinement",
        "v5_2_search_yield_ranked_mdn_plus_exact_refinement_and_rescue",
    } <= comparisons
    estimands = payload["method_comparison_estimands"]
    assert estimands["matched_branch_conditioned_conditional_flow_or_cinn"][
        "same_estimand_as_primary"
    ]
    bayesian = estimands["exact_forward_mcmc_or_nuts_stratified_bayesian_reference"]
    assert bayesian["scope"] == "predeclared_stratified_subset"
    assert bayesian["same_estimand_as_primary"] is False
    assert "declared_prior_and_likelihood" in bayesian["estimand"]
    assert bayesian["forward"] == "authoritative_non_neural_versioned_empirical_forward_model"
    inference = payload["statistical_inference"]
    assert inference["minimum_independent_training_seeds"] == (
        V5_PAPER_MINIMUM_PREDECLARED_TRAINING_SEEDS
    )
    assert inference["formal_randomization_unit"] == V5_PAPER_RQMC_RANDOMIZATION_UNIT
    assert inference["minimum_independent_scramble_replicates"] == (
        V5_PAPER_RQMC_MIN_INDEPENDENT_REPLICATES
    )
    assert inference["target_independent_scramble_replicates"] == (
        V5_PAPER_RQMC_TARGET_INDEPENDENT_REPLICATES
    )
    assert inference["engineering_four_scramble_design_may_enter_formal_inference"] is False
    assert inference["confidence_interval"] == V5_PAPER_RQMC_INFERENCE_RULE
    assert inference["within_scramble_points_are_iid"] is False
    assert inference["recipe_or_point_bootstrap_is_formal_population_interval"] is False
    assert inference["formal_design_schema"] == V5_PAPER_RQMC_DESIGN_SCHEMA
    assert inference["formal_evaluator_schema"] == V5_PAPER_RQMC_PAIRED_EVALUATOR_SCHEMA
    assert inference["fixed_cohort_secondary_interval"]["bootstrap_replicates"] == 10_000
    assert inference["fixed_cohort_secondary_interval"]["claim_scope"] == (
        V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE
    )
    assert inference["primary_solver_baseline_model_seed_policy"] == (
        V5_PAPER_RQMC_SHARED_BASELINE_POLICY
    )
    assert inference["training_seed_population_interval_claimed"] is False
    assert inference["formal_artifact_set_constructor_enforces_minimum_training_seeds"]
    assert "within_scramble" in inference["frozen_seed_set_summary_order"]
    assert payload["stages"]["paper_id_train"]["independent_training_seeds"] == 5
    calibration = payload["paper_evaluation_sets"]["calibration"]
    assert "iid_exchangeable" in calibration["sampling_design"]
    assert calibration["randomized_sobol_or_qmc_points_allowed_for_threshold_fitting"] is False
    assert inference["secondary_familywise_plan"].startswith("holm_")
    claims = payload["real_data_claim_contract"]
    assert claims["role"].startswith("blinded_case_study")
    assert claims["parameter_accuracy_without_traceable_ground_truth_allowed"] is False
    assert claims["population_performance_claim_allowed"] is False
    assert split["external_real_cases_are_not_sobol_split_members"]
    compute = payload["compute_reporting"]
    assert {"end_to_end_median_and_p90", "proposal_time", "exact_refinement_time"} <= set(
        compute["latency"]
    )
    assert {"exact_forward_calls", "gpu_hours", "cpu_core_hours"} <= set(compute["compute"])
    assert "kWh" in compute["energy_if_metered"]
    assert "CO2e" in compute["carbon_if_reported"]
    assert compute["missing_energy_meter_is_reported_not_imputed"]

    limits = payload["claim_limits"]
    assert limits["full_dwba_gisaxs"] is False
    assert limits["finite_search_failure_proves_no_solution"] is False
    assert limits["refined_candidates_are_posterior_samples"] is False
    assert limits["versioned_forward_agreement_proves_full_scattering_physics"] is False

    real_cases = payload["paper_evaluation_sets"]["external_real_case_study"]
    assert real_cases["minimum_cases"] == 8
    assert real_cases["target_cases"] == 20
    assert real_cases["minimum_independent_blinded_experts"] >= 2
    assert real_cases["required_case_types"] == [
        "in_domain_case",
        "predeclared_model_misspecification_or_negative_control",
        "truncated_q_or_weak_overlap_case",
    ]
    assert real_cases["success_only_case_selection_allowed"] is False
    assert real_cases["all_observed_pipeline_failures_retained"]
    assert real_cases["failed_cases_remain_in_declared_case_denominator"]
    assert real_cases["parameter_accuracy_without_ground_truth"] is False


def test_paper_checkpoint_selection_and_inverse_crime_limits_are_explicit() -> None:
    payload = protocol_payload()
    checkpoint = payload["paper_checkpoint_selection"]
    assert checkpoint["full_epochs_must_be_greater_than_zero"]
    assert checkpoint["warmup_only_run_is_paper_model_eligible"] is False
    assert checkpoint["retain_every_completed_full_epoch_candidate"]
    assert checkpoint["selection_split"] == "tuning_validation"
    assert "exact_forward_log2_budget_recall_auc" in checkpoint["primary_selection_metric"]
    assert "first_exact_compatible_verified_return" in checkpoint["secondary_selection_metric"]
    assert checkpoint["training_validation_loss_role"].endswith("not_final_checkpoint_selector")
    assert checkpoint["test_reference_calibration_or_real_data_may_select_checkpoint"] is False
    assert checkpoint["paper_training_blocked_until_candidate_retention_and_selector_are_enforced"]
    paper_train_gates = payload["stages"]["paper_id_train"]["gates"]
    assert paper_train_gates["full_epochs_minimum"] == 1
    assert paper_train_gates["all_completed_full_epoch_candidates_retained"]
    assert paper_train_gates["checkpoint_selected_by_tuning_exact_budget_auc_and_ttfc"]

    generalization = payload["forward_generalization_contract"]
    assert generalization["id_synthetic_train_and_test_share_forward_family"]
    assert generalization["same_forward_id_design_removes_inverse_crime"] is False
    assert generalization["alternate_forward_ood_required_before_physics_generalization_claim"]
    assert generalization["bornagain_challenge_required_where_parameter_mapping_is_valid"]
    assert generalization["alternate_forward_executable_frozen"] is False
    assert generalization["bornagain_challenge_executable_frozen"] is False
    assert generalization["physics_generalization_claim_currently_allowed"] is False
    assert (
        generalization["alternate_forward_or_bornagain_used_for_training_calibration_or_selection"]
        is False
    )
    assert generalization["report_nominal_alternate_forward_and_real_case_results_separately"]


def test_output_cap_counts_only_verified_compatible_returns_but_all_calls_consume_budget() -> None:
    fairness = protocol_payload()["comparison_fairness"]
    assert fairness["output_cap_counts"] == (
        "only_exact_compatible_verified_returned_cluster_representatives"
    )
    assert fairness["incompatible_or_unverified_attempts_consume_output_cap"] is False
    assert fairness["attempts_consuming_exact_budget"] == [
        "exact_compatible",
        "exact_incompatible",
        "unverified_after_an_exact_call_started",
    ]


def test_required_v5_ablations_cover_each_scientific_mechanism() -> None:
    assert protocol_payload()["required_ablations"] == [
        "remove_query_bounds_conditioning",
        "replace_local_with_global_coordinates",
        "remove_contextual_candidate_branch_enumeration",
        "remove_frozen_search_yield_ranking",
        (
            "replace_profiled_amplitude_initialization_with_direct_or_joint_only_"
            "initialization_keep_exact_polytope"
        ),
        (
            "replace_coupled_gui_amplitude_polytope_with_axis_box_during_search_"
            "keep_final_exact_gui_range_gate"
        ),
        "remove_BG_k_Int_i_int_Res_query_conditioning_but_keep_exact_polytope",
        "collapse_uncertainty_provenance_states",
        "uncertainty_provenance_strata_measured_vs_simulated_vs_encoder_proxy",
        ("pipeline_stage_proposal_only_vs_exact_scoring_only_vs_refinement_and_final_verification"),
        "remove_exact_refinement",
        "replace_search_yield_ranking_with_uniform_and_seeded_random_branch_order",
        ("explicit_GUI_k_Int_witness_vs_effective_coefficients_with_exact_shared_k_gauge_quotient"),
        (
            "representative_delta_output_cap_N_compatibility_threshold_and_"
            "exact_forward_budget_sensitivity"
        ),
        "remove_rescue_keep_ranked_mdn_and_exact_refinement",
        "remove_ambiguity_bank_mining",
        "identifiable_ambiguity_and_hard_negative_training_contribution",
        "guarded_sobol_blocks_vs_random_record_split_leakage_diagnostic",
        "training_scale_learning_curve",
    ]


def test_oracles_and_ablation_controls_decompose_the_pipeline_without_leakage_claims() -> None:
    payload = protocol_payload()
    oracles = payload["oracle_diagnostics"]
    assert set(oracles) >= {
        "topology_ranking",
        "continuous_proposal",
        "refinement",
    }
    assert oracles["role"].endswith("not_a_deployable_method")
    assert oracles["reference_information_may_train_or_select_product_model"] is False
    assert oracles["topology_ranking"]["generating_topology_is_oracle_truth"] is False
    assert oracles["continuous_proposal"]["generating_parameter_is_unique_truth"] is False
    assert oracles["refinement"]["final_authoritative_forward_verification_required"]

    controls = payload["ablation_reporting_contract"]
    assert controls["pipeline_stages"] == [
        "proposal_only_without_exact_scoring_or_refinement",
        "proposal_plus_authoritative_exact_scoring_without_refinement",
        "proposal_plus_refinement_plus_final_authoritative_exact_verification",
    ]
    assert controls["branch_ranking_controls"] == [
        "uniform_contextual_branch_priority_with_frozen_tie_break",
        "seeded_random_contextual_branch_order",
    ]
    assert controls["uncertainty_strata"] == list(UNCERTAINTY_KINDS)
    assert controls["amplitude_parameterizations"] == [
        "direct_gui_BG_k_Int_i_and_optional_int_Res_coordinates",
        (
            "effective_BG_a_i_optional_a_res_with_exact_shared_auxiliary_"
            "kappa_projection_and_explicit_GUI_witness"
        ),
    ]
    sensitivity = controls["sensitivity_axes"]
    assert sensitivity["output_cap_N"] == list(OUTPUT_CAPS)
    assert sensitivity["exact_forward_budget"] == list(EXACT_FORWARD_BUDGETS)
    assert sensitivity["threshold_multipliers"] == [0.5, 1.0, 2.0]
    assert controls["training_case_contributions"] == [
        "identifiable_cases",
        "ambiguous_multi_representative_cases",
        "completed_search_hard_negative_branches",
    ]
    leakage = controls["split_leakage_diagnostic"]
    assert leakage["valid_design"] == "guarded_disjoint_clean_parent_sobol_blocks"
    assert "random_record_split" in leakage["leaky_control"]
    assert leakage["leaky_control_may_support_headline_results"] is False


def test_protocol_import_remains_tensorflow_free() -> None:
    command = (
        "import sys; "
        "import utils.ML_Fitting_1D_GISAXS.PosteriorV8.study_protocol; "
        "assert 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_formal_cli_imports_do_not_cycle_through_study_protocol() -> None:
    for module in (
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.freeze_formal_production_search_plan_v5",
        "utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_formal_production_search_smoke_v5",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in completed.stdout


def test_study_protocol_source_has_no_duplicate_literal_dict_keys() -> None:
    source = Path(study_protocol_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    duplicates: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen: dict[str, int] = {}
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if key.value in seen:
                duplicates.append((key.value, seen[key.value], key.lineno))
            seen[key.value] = key.lineno
    assert duplicates == []


def test_architecture_document_uses_current_physics_and_methods_template_language() -> None:
    document = (
        Path(__file__).resolve().parents[3] / "docs" / "architecture" / "multisolution-inversion.md"
    ).read_text(encoding="utf-8")
    assert "Methods template" in document
    assert "V5.2" in document
    assert "I(q) = BG + k * (sum_i Int_i * P_i(q) * S_i(q) + int_Res * R(q))" in document
    assert "实际可达 60 个" in document
    assert "complete-linkage diameter-δ" in document
    assert "不保证 pairwise distance `>δ`" in document
    assert "incompatible/unverified 尝试消耗实际 exact calls，但不占 N" in document
    assert "`full_epochs=0`" in document
    assert "BornAgain" in document
    assert "δ-separated" not in document
    assert "V5.1" not in document
    assert "预注册 80 acquisition-only strata" not in document


def test_protocol_tampering_fails_closed() -> None:
    payload = protocol_payload()
    payload["final_acceptance"]["reference_representative_recall_at_n16_b4096_gte"] = 0.1
    with pytest.raises(ValueError, match="modified"):
        validate_protocol(payload)

    legacy_v3 = protocol_payload()
    legacy_v3["schema_version"] = "gisaxs.posterior_v8.multisolution_study_protocol/v3"
    with pytest.raises(ValueError, match="modified"):
        validate_protocol(legacy_v3)


def test_protocol_write_is_atomic_and_non_overwriting(tmp_path) -> None:
    output = tmp_path / "study-protocol.json"
    assert write_protocol(output) == output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert validate_protocol(saved) == protocol_payload()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_protocol(output)
