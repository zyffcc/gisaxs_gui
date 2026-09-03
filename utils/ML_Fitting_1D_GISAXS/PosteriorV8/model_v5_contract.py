"""TensorFlow-free contract for the V5 per-hard-branch proposal model."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Mapping

from .amplitude_query_v5 import (
    AMPLITUDE_QUERY_EMBEDDING_DIM,
    V5_AMPLITUDE_EMBEDDING_VERSION,
    V5_AMPLITUDE_QUERY_VERSION,
)
from .bounds_first_contract import (
    BOUNDS_EMBEDDING_DIM,
    BOUNDS_EMBEDDING_VERSION,
    LOCAL_TARGET_SEMANTICS,
)
from .branch_catalog import BRANCH_CATALOG_VERSION, BRANCH_PATTERN_COUNT
from .branch_codec import UNIT_CUBE_DIMENSIONS
from .contextual_branch_catalog import CONTEXTUAL_BRANCH_CATALOG_VERSION
from .contract import NUM_TOPOLOGIES
from .sobol_numeric_canonicalization_v5 import (
    V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
    V5_FAST_NUMERIC_POLICY_VERSION,
    v5_numeric_policy_sha256,
)
from .uncertainty_provenance_v5 import (
    UNCERTAINTY_FEATURE_DIM,
    UNCERTAINTY_KINDS,
    V5_UNCERTAINTY_SCHEMA,
    V5_UNCERTAINTY_VERSION,
)


MODEL_V5_SCHEMA = "gisaxs.posterior_v8.branch_conditioned_proposal_model/v5.2"
MODEL_V5_VERSION = (
    "posterior_v8_numeric_contract_bound_branch_conditioned_search_yield_mdn_v5_2_r2"
)
MODEL_V5_NAME = "posterior_v8_branch_conditioned_search_yield_proposal_v5_2_r2"

CURVE_POINT_FEATURE_DIM = 3
CURVE_GLOBAL_FEATURE_DIM = 5
UNCERTAINTY_PROVENANCE_STATES = UNCERTAINTY_KINDS
MODEL_V5_UNCERTAINTY_PROVENANCE_SEMANTICS = (
    "one_hot_measured_simulated_or_encoder_proxy_missing_sigma_v1"
)

MODEL_V5_INPUT_KEYS = (
    "x",
    "point_mask",
    "global_features",
    "uncertainty_provenance",
    "branch_topology_id",
    "branch_pattern_id",
    "geometry_bounds_embedding",
    "amplitude_bounds_embedding",
    "available_dimension_mask",
    "active_dimension_mask",
    "varying_dimension_mask",
)
MODEL_V5_OUTPUT_KEYS = (
    "proposal_search_yield_logit",
    "mixture_logits",
    "mixture_loc",
    "mixture_logscale",
)

MODEL_V5_SCORE_SEMANTICS = (
    "uncalibrated_probability_logit_that_frozen_generation_and_refinement_protocol_"
    "finds_at_least_one_exact_compatible_representative_for_query_branch_"
    "not_posterior_not_mathematical_solvability_not_no_solution_certificate"
)
MODEL_V5_AVAILABLE_MASK_SEMANTICS = "physical_axes_available_in_the_user_query_union_26d_v1"
MODEL_V5_ACTIVE_MASK_SEMANTICS = "physical_axes_active_in_the_selected_hard_wire_branch_26d_v1"
MODEL_V5_VARYING_MASK_SEMANTICS = "selected_branch_codec_effectively_varying_local_axes_26d_v1"
MODEL_V5_BRANCH_SEMANTICS = "one_contextual_catalog_wire_branch_per_example_external_enumeration_v1"
MODEL_V5_AMPLITUDE_BOUNDS_SEMANTICS = (
    "independent_physical_BG_k_Int_i_int_Res_user_ranges_with_shared_k_only_"
    "BG_and_k_encoded_relative_to_the_observation_intensity_reference_v2"
)

MODEL_V5_FIXED_DIMENSIONS = MappingProxyType(
    {
        "point_feature_dim": CURVE_POINT_FEATURE_DIM,
        "global_feature_dim": CURVE_GLOBAL_FEATURE_DIM,
        "uncertainty_provenance_dim": UNCERTAINTY_FEATURE_DIM,
        "topology_dim": NUM_TOPOLOGIES,
        "wire_pattern_dim": BRANCH_PATTERN_COUNT,
        "geometry_bounds_embedding_dim": BOUNDS_EMBEDDING_DIM,
        "amplitude_bounds_embedding_dim": AMPLITUDE_QUERY_EMBEDDING_DIM,
        "local_coordinate_dim": UNIT_CUBE_DIMENSIONS,
    }
)
MODEL_V5_COORDINATE_CONTRACT = MappingProxyType(
    {
        "geometry_bounds_embedding": BOUNDS_EMBEDDING_VERSION,
        "amplitude_bounds_embedding": V5_AMPLITUDE_EMBEDDING_VERSION,
        "available_dimension_mask": MODEL_V5_AVAILABLE_MASK_SEMANTICS,
        "active_dimension_mask": MODEL_V5_ACTIVE_MASK_SEMANTICS,
        "varying_dimension_mask": MODEL_V5_VARYING_MASK_SEMANTICS,
        "mdn_output": LOCAL_TARGET_SEMANTICS,
        "fixed_and_inactive_local_coordinate": 0.5,
    }
)
MODEL_V5_NUMERIC_POLICY_CONTRACT = MappingProxyType(
    {
        "gui_fast": {
            "version": V5_FAST_NUMERIC_POLICY_VERSION,
            "sha256": v5_numeric_policy_sha256(V5_FAST_NUMERIC_POLICY_VERSION),
        },
        "frozen_direct_sobol": {
            "version": V5_DETERMINISTIC_NUMERIC_POLICY_VERSION,
            "sha256": v5_numeric_policy_sha256(
                V5_DETERMINISTIC_NUMERIC_POLICY_VERSION
            ),
        },
    }
)

if (
    BOUNDS_EMBEDDING_DIM != 78
    or AMPLITUDE_QUERY_EMBEDDING_DIM != 21
    or UNIT_CUBE_DIMENSIONS != 26
    or BRANCH_PATTERN_COUNT != 32
):  # pragma: no cover
    raise RuntimeError("V5 branch-conditioned tensor dimensions changed unexpectedly")


def model_v5_contract_payload() -> dict[str, object]:
    """Return the exact identity required of a serialized V5 model artifact."""

    return {
        "schema_version": MODEL_V5_SCHEMA,
        "model_version": MODEL_V5_VERSION,
        "model_name": MODEL_V5_NAME,
        "input_keys": list(MODEL_V5_INPUT_KEYS),
        "output_keys": list(MODEL_V5_OUTPUT_KEYS),
        "fixed_dimensions": dict(MODEL_V5_FIXED_DIMENSIONS),
        "coordinate_contract": dict(MODEL_V5_COORDINATE_CONTRACT),
        "numeric_policy_contract": {
            name: dict(value) for name, value in MODEL_V5_NUMERIC_POLICY_CONTRACT.items()
        },
        "score_semantics": MODEL_V5_SCORE_SEMANTICS,
        "uncertainty_provenance": {
            "semantics": MODEL_V5_UNCERTAINTY_PROVENANCE_SEMANTICS,
            "schema_version": V5_UNCERTAINTY_SCHEMA,
            "version": V5_UNCERTAINTY_VERSION,
            "states_in_one_hot_order": list(UNCERTAINTY_PROVENANCE_STATES),
        },
        "branch_contract": {
            "semantics": MODEL_V5_BRANCH_SEMANTICS,
            "wire_catalog_version": BRANCH_CATALOG_VERSION,
            "contextual_catalog_version": CONTEXTUAL_BRANCH_CATALOG_VERSION,
            "topology_ids": "stable_integer_0_through_33",
            "wire_pattern_ids": "stable_integer_0_through_31",
            "contextual_catalog_generation": "external_per_user_query",
            "contextual_policy_eligibility": "guaranteed_by_external_catalog",
            "catalog_cardinality": "context_dependent_not_fixed",
            "static_shape_only_canonicalization_required": False,
        },
        "amplitude_query_contract": {
            "version": V5_AMPLITUDE_QUERY_VERSION,
            "embedding_version": V5_AMPLITUDE_EMBEDDING_VERSION,
            "embedding_semantics": MODEL_V5_AMPLITUDE_BOUNDS_SEMANTICS,
            "same_physical_ranges_define_exact_profile_polytope": True,
            "component_intensities_are_independent_not_simplex": True,
            "observation_reference_is_not_part_of_clean_query_identity": True,
        },
        "claim_limits": {
            "score_is_posterior_probability": False,
            "score_is_calibrated_probability": False,
            "score_is_mathematical_branch_solvability": False,
            "score_is_no_solution_certificate": False,
            "mdn_is_exact_posterior": False,
            "model_performs_exact_forward_verification": False,
        },
    }


def validate_model_v5_contract(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate exact V5 identity; missing, legacy, and extended payloads fail."""

    if not isinstance(payload, Mapping):
        raise TypeError("V5 model contract must be a mapping")
    value = deepcopy(dict(payload))
    expected = model_v5_contract_payload()
    missing = sorted(set(expected) - set(value))
    unexpected = sorted(set(value) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "V5 model contract fields are incomplete or unsupported; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"V5 model contract has incompatible {name}")
    return value


__all__ = [
    "CURVE_GLOBAL_FEATURE_DIM",
    "CURVE_POINT_FEATURE_DIM",
    "MODEL_V5_ACTIVE_MASK_SEMANTICS",
    "MODEL_V5_AMPLITUDE_BOUNDS_SEMANTICS",
    "MODEL_V5_AVAILABLE_MASK_SEMANTICS",
    "MODEL_V5_BRANCH_SEMANTICS",
    "MODEL_V5_COORDINATE_CONTRACT",
    "MODEL_V5_FIXED_DIMENSIONS",
    "MODEL_V5_INPUT_KEYS",
    "MODEL_V5_NAME",
    "MODEL_V5_NUMERIC_POLICY_CONTRACT",
    "MODEL_V5_OUTPUT_KEYS",
    "MODEL_V5_SCHEMA",
    "MODEL_V5_SCORE_SEMANTICS",
    "MODEL_V5_VARYING_MASK_SEMANTICS",
    "MODEL_V5_VERSION",
    "MODEL_V5_UNCERTAINTY_PROVENANCE_SEMANTICS",
    "UNCERTAINTY_PROVENANCE_STATES",
    "model_v5_contract_payload",
    "validate_model_v5_contract",
]
