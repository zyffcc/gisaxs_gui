from __future__ import annotations

from dataclasses import asdict

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    DESIGN_STRATUM_UNIVERSE_FIELDS,
    DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    DESIGN_STRATUM_UNIVERSE_SHA256,
    DESIGN_STRATUM_UNIVERSE_VERSION,
    PREREGISTERED_DESIGN_STRATUM_COUNT,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.simulation import (
    OBSERVATION_DESIGN_STRATUM_COUNT,
    OBSERVATION_DESIGN_STRATUM_FIELDS,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE_SEMANTICS,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE_SHA256,
    OBSERVATION_DESIGN_STRATUM_UNIVERSE_VERSION,
    MAX_GRID_POINTS,
    reachable_observation_design_strata,
    sample_observation_design_stratum,
    sample_observation_view,
)


def test_fifty_thousand_seed_enumeration_reaches_exact_frozen_sixty_strata():
    observed = {
        sample_observation_design_stratum(clean_recipe_seed=seed, view_index=0)
        for seed in range(50_000)
    }

    assert observed == set(OBSERVATION_DESIGN_STRATUM_UNIVERSE)
    assert reachable_observation_design_strata() == OBSERVATION_DESIGN_STRATUM_UNIVERSE
    assert len(observed) == OBSERVATION_DESIGN_STRATUM_COUNT
    assert OBSERVATION_DESIGN_STRATUM_COUNT == PREREGISTERED_DESIGN_STRATUM_COUNT == 60
    assert all(tuple(asdict(value)) == OBSERVATION_DESIGN_STRATUM_FIELDS for value in observed)


def test_calibration_contract_binds_the_simulation_universe_identity():
    assert DESIGN_STRATUM_UNIVERSE_VERSION == OBSERVATION_DESIGN_STRATUM_UNIVERSE_VERSION
    assert DESIGN_STRATUM_UNIVERSE_FIELDS == OBSERVATION_DESIGN_STRATUM_FIELDS
    assert DESIGN_STRATUM_UNIVERSE_SEMANTICS == (OBSERVATION_DESIGN_STRATUM_UNIVERSE_SEMANTICS)
    assert DESIGN_STRATUM_UNIVERSE_SHA256 == OBSERVATION_DESIGN_STRATUM_UNIVERSE_SHA256
    assert len(DESIGN_STRATUM_UNIVERSE_SHA256) == 64


def test_lightweight_stratum_selector_is_the_observation_generators_source_of_truth():
    point_options = (64, MAX_GRID_POINTS // 4, MAX_GRID_POINTS // 2, MAX_GRID_POINTS)
    for seed in range(25):
        for view_index in range(4):
            coordinate = sample_observation_design_stratum(seed, view_index)
            view = sample_observation_view(seed, view_index)
            assert view.grid.n_points == point_options[coordinate.design_point_count_cycle_slot]
            assert view.q_window_id == coordinate.q_window_id
            assert view.noise_id == coordinate.noise_id
