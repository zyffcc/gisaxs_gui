from __future__ import annotations

import numpy as np

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import UNIT_CUBE_DIMENSIONS
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_branch_catalog import (
    canonical_branch_pattern_id,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_leakage_features_v5 import (
    V5_CLEAN_PHYSICS_LEAKAGE_DIM,
    V5_LEAKAGE_FEATURE_VERSION,
    V5_QUERY_BOUNDS_LEAKAGE_DIM,
    coefficient_total_from_unit,
    coefficient_total_to_unit,
    v5_clean_leakage_features,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    sample_v5_clean_recipe,
)


def test_clean_leakage_features_include_global_geometry_and_all_coefficients():
    recipe = sample_v5_clean_recipe(
        ("sphere", "cylinder"),
        recipe_seed=903,
        amplitude_regime="weak_particle_coefficient",
    )

    features = v5_clean_leakage_features(recipe)

    assert "scale_quotiented_effective_coefficient_composition" in (
        V5_LEAKAGE_FEATURE_VERSION
    )
    assert len(features.normalized_continuous_parameters) == V5_CLEAN_PHYSICS_LEAKAGE_DIM
    assert len(features.normalized_query_bounds) == V5_QUERY_BOUNDS_LEAKAGE_DIM == 99
    assert features.normalized_query_bounds[:78] == recipe.query.bounds_embedding
    assert features.normalized_query_bounds[78:] == recipe.amplitude_query.model_embedding(1.0)
    amplitude = features.normalized_continuous_parameters[UNIT_CUBE_DIMENSIONS:]
    reconstructed_total = coefficient_total_from_unit(amplitude[0])
    expected_coefficients = np.zeros(6)
    expected_coefficients[0] = recipe.amplitude.background
    expected_coefficients[1:3] = recipe.amplitude.particle_amplitudes
    expected_coefficients[-1] = recipe.amplitude.resolution_amplitude
    assert np.isclose(reconstructed_total, np.sum(expected_coefficients), rtol=1e-12)
    assert np.allclose(
        np.asarray(amplitude[1:]) * reconstructed_total,
        expected_coefficients,
        rtol=2e-12,
        atol=1e-12,
    )


def test_positive_coefficient_scale_mapping_is_deterministic_and_injective():
    values = (1e-12, 1.0, 1e9, 1e100)
    units = tuple(coefficient_total_to_unit(value) for value in values)

    assert all(0.0 < value < 1.0 for value in units)
    assert tuple(sorted(units)) == units
    assert np.allclose(
        [coefficient_total_from_unit(value) for value in units],
        values,
        rtol=5e-12,
    )


def test_repeated_shape_wire_branch_is_quotiented_for_physical_leakage_distance():
    recipe = sample_v5_clean_recipe(("sphere", "sphere"), recipe_seed=0)

    features = v5_clean_leakage_features(recipe)

    assert recipe.target.pattern_id != canonical_branch_pattern_id(
        recipe.query.topology_id,
        recipe.target.pattern_id,
    )
    assert features.branch_pattern_id == canonical_branch_pattern_id(
        recipe.query.topology_id,
        recipe.target.pattern_id,
    )
