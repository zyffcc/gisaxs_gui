from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    branch_pattern_id,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.simulation import (
    GridProvenance,
    NoiseProvenance,
    SimulationRecipe,
    simulate_recipe,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    V5_CLEAN_RECIPE_SCHEMA,
    authoritative_gui_parameters,
    evaluate_v5_clean_recipe,
    sample_v5_clean_recipe,
)


def test_clean_recipe_is_replayable_and_uses_disjoint_provenance_seeds():
    first = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903)
    replay = sample_v5_clean_recipe(("sphere",), recipe_seed=20260903)

    assert first == replay
    assert first.sha256 == replay.sha256
    assert first.schema_version == V5_CLEAN_RECIPE_SCHEMA
    assert len(
        {
            first.query_seed,
            first.amplitude_query_seed,
            first.target_seed,
            first.amplitude_seed,
        }
    ) == 4
    assert first.query.canonical_json in first.canonical_json
    assert first.amplitude_query.canonical_json in first.canonical_json
    constraint = first.amplitude_query.constraint_for_branch(
        resolution_present=first.amplitude.resolution_present
    )
    assert constraint.contains(first.amplitude.coefficient_vector)
    assert np.all(evaluate_v5_clean_recipe(first) > 0.0)


def test_clean_recipe_replays_global_high_endpoint_without_roundoff_escape():
    # This exact seed exposed exp(log(100)) == 100.00000000000004 on Maxwell
    # while constructing the 512-parent Phase-A cohort.
    recipe = sample_v5_clean_recipe(
        ("sphere",),
        recipe_seed=20260915,
        amplitude_range_regime="full",
        pattern_id=0,
    )

    assert recipe.target.truth_components[0].R == 100.0
    assert np.all(np.isfinite(evaluate_v5_clean_recipe(recipe)))


def test_clean_recipe_truth_is_the_exact_stored_coordinate_decode_representative():
    # This seed previously retained the pre-encode width value, which differed
    # from decoding its stored local coordinate by one binary64 ULP.
    recipe = sample_v5_clean_recipe(
        ("sphere",),
        recipe_seed=20261003,
        amplitude_range_regime="full",
        pattern_id=0,
    )
    codec = recipe.query.codec_for(recipe.target.pattern_id)
    latent, resolution = codec.decode(recipe.target.local_target_unit)

    assert codec.latent_components_to_gui(latent) == recipe.target.truth_components
    assert resolution == recipe.target.truth_resolution


def test_v5_exact_curve_matches_existing_authoritative_simulation_path_for_k1():
    pattern = branch_pattern_id((True, False, False, False), True)
    grid = GridProvenance(kind="hybrid", q_min=3.0e-4, q_max=2.0, n_points=128)
    recipe = sample_v5_clean_recipe(
        ("sphere",),
        recipe_seed=77,
        amplitude_regime="balanced_particles",
        pattern_id=pattern,
        grid=grid,
    )
    existing = SimulationRecipe(
        topology_id=recipe.query.topology_id,
        components=recipe.target.truth_components,
        effective_amplitudes=recipe.amplitude.particle_amplitudes,
        resolution=recipe.target.truth_resolution,
        resolution_effective_amplitude=recipe.amplitude.resolution_amplitude,
        background=recipe.amplitude.background,
        grid=grid,
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
        seed=recipe.recipe_seed,
        require_characteristic_coverage=False,
    )

    expected = simulate_recipe(existing).clean_intensity
    actual = evaluate_v5_clean_recipe(recipe)
    assert np.allclose(actual, expected, rtol=2.0e-12, atol=1.0e-12)
    assert np.allclose(
        authoritative_gui_parameters(recipe)[-5:],
        (
            recipe.amplitude.background,
            recipe.target.truth_resolution.sigma_res,
            recipe.target.truth_resolution.nu_res,
            recipe.amplitude.int_res,
            recipe.amplitude.k,
        ),
    )


def test_weak_particle_recipe_is_supported_without_claiming_observability():
    recipe = sample_v5_clean_recipe(
        ("sphere", "cylinder", "vertical_cylinder"),
        recipe_seed=91,
        amplitude_regime="weak_particle_coefficient",
    )
    selected = recipe.amplitude.selected_particle_slot

    assert selected is not None
    assert recipe.amplitude.particle_weights[selected] < 0.03
    assert recipe.amplitude.audit_payload()["coefficient_fraction_is_observability"] is False
    assert np.all(np.isfinite(evaluate_v5_clean_recipe(recipe)))


def test_recipe_hash_and_forward_inputs_fail_closed():
    recipe = sample_v5_clean_recipe(("vertical_cylinder",), recipe_seed=19)
    with pytest.raises(ValueError, match="audit hash"):
        replace(recipe, sha256="0" * 64)
    with pytest.raises(ValueError, match="unsupported V5 clean recipe generator"):
        replace(recipe, generator_version="stale-v3")
    with pytest.raises(ValueError, match="unsupported V5 clean recipe schema"):
        replace(recipe, schema_version="stale-v3")
    with pytest.raises(ValueError, match="finite positive"):
        evaluate_v5_clean_recipe(recipe, np.asarray((0.0, 1.0)))
    with pytest.raises(ValueError, match="canonical"):
        sample_v5_clean_recipe(("vertical_cylinder", "sphere"), recipe_seed=3)
