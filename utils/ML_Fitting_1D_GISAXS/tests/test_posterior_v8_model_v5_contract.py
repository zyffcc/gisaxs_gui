from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import (
    AMPLITUDE_QUERY_EMBEDDING_DIM,
    V5_AMPLITUDE_EMBEDDING_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    BOUNDS_EMBEDDING_VERSION,
    LOCAL_TARGET_SEMANTICS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    BRANCH_CATALOG_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contextual_branch_catalog import (
    CONTEXTUAL_BRANCH_CATALOG_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import (
    MODEL_V5_COORDINATE_CONTRACT,
    MODEL_V5_FIXED_DIMENSIONS,
    MODEL_V5_INPUT_KEYS,
    MODEL_V5_NAME,
    MODEL_V5_NUMERIC_POLICY_CONTRACT,
    MODEL_V5_OUTPUT_KEYS,
    MODEL_V5_SCHEMA,
    MODEL_V5_SCORE_SEMANTICS,
    MODEL_V5_VERSION,
    UNCERTAINTY_PROVENANCE_STATES,
    model_v5_contract_payload,
    validate_model_v5_contract,
)


def test_v5_contract_is_per_candidate_without_curve_only_discrete_heads():
    assert MODEL_V5_SCHEMA.endswith("/v5.2")
    assert MODEL_V5_VERSION.endswith("_v5_2_r2")
    assert MODEL_V5_NAME.endswith("_v5_2_r2")
    assert MODEL_V5_INPUT_KEYS == (
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
    assert MODEL_V5_OUTPUT_KEYS == (
        "proposal_search_yield_logit",
        "mixture_logits",
        "mixture_loc",
        "mixture_logscale",
    )
    assert "topology_logits" not in MODEL_V5_OUTPUT_KEYS
    assert "branch_pattern_logits" not in MODEL_V5_OUTPUT_KEYS
    assert "not_posterior" in MODEL_V5_SCORE_SEMANTICS
    assert "not_mathematical_solvability" in MODEL_V5_SCORE_SEMANTICS
    assert "not_no_solution_certificate" in MODEL_V5_SCORE_SEMANTICS


def test_v5_contract_preserves_78d_physical_bounds_and_three_mask_semantics():
    assert MODEL_V5_FIXED_DIMENSIONS == {
        "point_feature_dim": 3,
        "global_feature_dim": 5,
        "uncertainty_provenance_dim": 3,
        "topology_dim": 34,
        "wire_pattern_dim": 32,
        "geometry_bounds_embedding_dim": 78,
        "amplitude_bounds_embedding_dim": AMPLITUDE_QUERY_EMBEDDING_DIM,
        "local_coordinate_dim": 26,
    }
    assert (
        MODEL_V5_COORDINATE_CONTRACT["geometry_bounds_embedding"]
        == BOUNDS_EMBEDDING_VERSION
    )
    assert (
        MODEL_V5_COORDINATE_CONTRACT["amplitude_bounds_embedding"]
        == V5_AMPLITUDE_EMBEDDING_VERSION
    )
    assert MODEL_V5_COORDINATE_CONTRACT["mdn_output"] == LOCAL_TARGET_SEMANTICS
    assert "user_query_union" in MODEL_V5_COORDINATE_CONTRACT["available_dimension_mask"]
    assert "selected_hard_wire_branch" in MODEL_V5_COORDINATE_CONTRACT["active_dimension_mask"]
    assert "effectively_varying" in MODEL_V5_COORDINATE_CONTRACT["varying_dimension_mask"]
    assert model_v5_contract_payload()["amplitude_query_contract"][
        "component_intensities_are_independent_not_simplex"
    ] is True
    assert model_v5_contract_payload()["numeric_policy_contract"] == {
        name: dict(value) for name, value in MODEL_V5_NUMERIC_POLICY_CONTRACT.items()
    }


def test_v5_contract_separates_uncertainty_source_from_curve_features():
    assert UNCERTAINTY_PROVENANCE_STATES == (
        "measured_sigma",
        "simulated_sigma",
        "encoder_proxy_missing_sigma",
    )
    uncertainty = model_v5_contract_payload()["uncertainty_provenance"]
    assert uncertainty["states_in_one_hot_order"] == list(UNCERTAINTY_PROVENANCE_STATES)
    assert "one_hot" in uncertainty["semantics"]


def test_v5_1_contract_conditions_on_the_same_full_amplitude_query_as_exact_profiling():
    amplitude = model_v5_contract_payload()["amplitude_query_contract"]
    assert amplitude["embedding_version"] == V5_AMPLITUDE_EMBEDDING_VERSION
    assert amplitude["same_physical_ranges_define_exact_profile_polytope"] is True
    assert amplitude["observation_reference_is_not_part_of_clean_query_identity"] is True


def test_v5_contract_binds_wire_and_external_contextual_catalog_without_fixed_count():
    payload = model_v5_contract_payload()
    branch = payload["branch_contract"]
    assert branch["wire_catalog_version"] == BRANCH_CATALOG_VERSION
    assert branch["contextual_catalog_version"] == CONTEXTUAL_BRANCH_CATALOG_VERSION
    assert branch["topology_ids"] == "stable_integer_0_through_33"
    assert branch["wire_pattern_ids"] == "stable_integer_0_through_31"
    assert branch["contextual_catalog_generation"] == "external_per_user_query"
    assert branch["contextual_policy_eligibility"] == "guaranteed_by_external_catalog"
    assert branch["catalog_cardinality"] == "context_dependent_not_fixed"
    assert branch["static_shape_only_canonicalization_required"] is False
    assert "physical_branch_count" not in repr(payload)
    assert "418" not in repr(payload)


def test_v5_contract_validation_rejects_v1_v2_v3_and_extensions_fail_closed():
    expected = model_v5_contract_payload()
    assert validate_model_v5_contract(expected) == expected

    for field, legacy in (
        ("schema_version", "gisaxs.posterior_v8.bounds_proposal_training_run/v3"),
        ("model_version", "posterior_v8_bounds_conditioned_logistic_normal_v3"),
        ("model_name", "posterior_v8_bounds_conditioned_local_proposal_v3"),
    ):
        payload = deepcopy(expected)
        payload[field] = legacy
        with pytest.raises(ValueError, match=field):
            validate_model_v5_contract(payload)

    extended = deepcopy(expected)
    extended["legacy_fallback"] = True
    with pytest.raises(ValueError, match="unsupported"):
        validate_model_v5_contract(extended)
    with pytest.raises(TypeError, match="mapping"):
        validate_model_v5_contract(None)


def test_v5_contract_import_is_tensorflow_free():
    command = (
        "import sys; "
        "import utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract; "
        "assert 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
