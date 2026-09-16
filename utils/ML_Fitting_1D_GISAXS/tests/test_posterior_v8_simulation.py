from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.contract import (
    CYLINDER,
    NUM_TOPOLOGIES,
    GuiComponentParameters,
    topology_from_id,
    topology_id_for,
)
from PosteriorV8.profiled_forward import ResolutionShape
from PosteriorV8.simulation import (
    HARD_CORE_SPACING_MARGIN,
    MIN_EFFECTIVE_AMPLITUDE_FRACTION,
    OBSERVATION_VIEW_VERSION,
    GridProvenance,
    NoiseProvenance,
    SimulationRecipe,
    apply_observation_noise,
    sample_identifiable_recipe,
    sample_observation_view,
    simulate_recipe,
)
from src.gimap.features.fitting.domain.physical_constraints import exclusion_size
from src.gimap.features.fitting.domain.scattering_model import make_mixed_model


def _authoritative_parameters(recipe):
    total = sum(recipe.effective_amplitudes)
    parameters = []
    for component, amplitude in zip(recipe.components, recipe.effective_amplitudes):
        weight = amplitude / total
        d = component.D or 0.0
        sigma_d = component.sigma_D or 0.0
        if component.shape == CYLINDER:
            parameters.extend(
                [weight, component.R, component.sigma_R, component.h, component.sigma_h, d, sigma_d]
            )
        else:
            parameters.extend([weight, component.R, component.sigma_R, d, sigma_d])
    if recipe.resolution is None:
        parameters.extend([recipe.background, 0.0, 0.0, 0.0, total])
    else:
        parameters.extend(
            [
                recipe.background,
                recipe.resolution.sigma_res,
                recipe.resolution.nu_res,
                recipe.resolution_effective_amplitude / total,
                total,
            ]
        )
    return parameters


def test_recipe_and_noise_are_exactly_reproducible():
    first_recipe = sample_identifiable_recipe(1234, topology_id=33, max_points=96)
    second_recipe = sample_identifiable_recipe(1234, topology_id=33, max_points=96)
    assert first_recipe == second_recipe
    first = simulate_recipe(first_recipe)
    second = simulate_recipe(second_recipe)
    np.testing.assert_array_equal(first.q, second.q)
    np.testing.assert_array_equal(first.clean_intensity, second.clean_intensity)
    np.testing.assert_array_equal(first.intensity, second.intensity)
    np.testing.assert_array_equal(first.sigma, second.sigma)


def test_public_noise_primitive_preserves_pre_extraction_values_pointwise():
    clean = np.geomspace(0.0125, 125.0, 64, dtype=np.float64)
    noise = NoiseProvenance(poisson_count_scale=2.0e4, relative_sigma=0.01)
    intensity, sigma = apply_observation_noise(clean, noise, observation_seed=1234)

    # Replay the former inline implementation pointwise.  Platform libm may
    # produce adjacent binary64 values for geomspace/exp, so a digest frozen on
    # macOS is not a valid Linux regression oracle even with the same NumPy.
    rng = np.random.default_rng(np.random.SeedSequence([1234, 0x5638]))
    reference = max(float(np.median(clean)), np.finfo(np.float64).tiny)
    scale = noise.poisson_count_scale
    assert scale is not None
    counts = rng.poisson(np.clip(clean / reference * scale, 0.0, 1.0e9))
    expected_intensity = counts.astype(np.float64) / scale * reference
    expected_sigma_poisson = np.sqrt(np.maximum(counts, 1.0)) / scale * reference
    expected_intensity *= np.exp(
        rng.normal(0.0, noise.relative_sigma, size=expected_intensity.shape)
    )
    floor = max(noise.sigma_floor_fraction * reference, np.finfo(np.float64).tiny)
    expected_intensity = np.maximum(expected_intensity, floor)
    expected_sigma = np.hypot(
        np.hypot(expected_sigma_poisson, noise.relative_sigma * expected_intensity),
        floor,
    )
    np.testing.assert_array_equal(intensity, expected_intensity)
    np.testing.assert_array_equal(sigma, expected_sigma)

    recipe = sample_identifiable_recipe(1234, topology_id=33, max_points=96)
    simulated = simulate_recipe(recipe)
    expected_intensity, expected_sigma = apply_observation_noise(
        simulated.clean_intensity,
        recipe.noise,
        observation_seed=recipe.seed,
    )
    np.testing.assert_array_equal(simulated.intensity, expected_intensity)
    np.testing.assert_array_equal(simulated.sigma, expected_sigma)


def test_noise_sigma_root_sum_square_stays_positive_at_subnormal_scale():
    clean = np.full(64, 1.0e-300, dtype=np.float64)
    noise = NoiseProvenance(
        poisson_count_scale=None,
        relative_sigma=0.0,
        sigma_floor_fraction=1.0e-6,
    )

    intensity, sigma = apply_observation_noise(clean, noise, observation_seed=8128)

    assert np.all(np.isfinite(intensity))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)
    np.testing.assert_array_equal(sigma, np.full_like(sigma, 1.0e-306))


def test_all_34_topologies_generate_valid_identifiability_candidates():
    for topology_id in range(NUM_TOPOLOGIES):
        recipe = sample_identifiable_recipe(1000 + topology_id, topology_id, max_points=64)
        assert tuple(component.shape for component in recipe.components) == topology_from_id(
            topology_id
        )
        assert len(recipe.components) == len(topology_from_id(topology_id))
        assert np.min(recipe.effective_amplitude_fractions) >= (
            MIN_EFFECTIVE_AMPLITUDE_FRACTION - 1e-12
        )
        for component in recipe.components:
            assert (component.D is None) == (component.sigma_D is None)
        curve = simulate_recipe(recipe)
        assert curve.q.size == 64
        assert np.all(curve.clean_intensity > 0.0)
        assert np.all(curve.intensity > 0.0)
        assert np.all(curve.sigma > 0.0)


def test_mixed_shapes_do_not_leak_canonical_slot_through_size_strata():
    # Topology 13 is sphere/cylinder/vertical cylinder. Every different shape
    # must cover the same full R prior; global-slot strata would make the first
    # shape smaller than the third in every generated recipe.
    components = [
        sample_identifiable_recipe(seed, topology_id=13, max_points=64).components
        for seed in range(64)
    ]
    sphere_minus_vertical = [values[0].R - values[2].R for values in components]
    assert any(value < 0.0 for value in sphere_minus_vertical)
    assert any(value > 0.0 for value in sphere_minus_vertical)

    repeated = sample_identifiable_recipe(91, topology_id=3, max_points=64)
    assert repeated.components[0].R < repeated.components[1].R


def test_all_34_topologies_and_multiple_seeds_obey_authoritative_hard_core_rule():
    present_count = 0
    for topology_id in range(NUM_TOPOLOGIES):
        for seed_offset in range(5):
            recipe = sample_identifiable_recipe(
                10_000 + 101 * topology_id + seed_offset,
                topology_id,
                max_points=64,
            )
            assert recipe.hard_core_spacing_margin == HARD_CORE_SPACING_MARGIN == 1.001
            for component in recipe.components:
                if component.D is None:
                    continue
                present_count += 1
                params = {"R": component.R}
                if component.h is not None:
                    params["h"] = component.h
                required = HARD_CORE_SPACING_MARGIN * exclusion_size(component.shape, params)
                assert component.D > required
    assert present_count > 0


@pytest.mark.parametrize(
    ("component", "violating_d"),
    [
        (GuiComponentParameters("sphere", 10.0, 1.0, D=20.0, sigma_D=2.0), 20.0),
        (
            GuiComponentParameters(
                "cylinder", 10.0, 1.0, h=20.0, sigma_h=2.0, D=28.3, sigma_D=2.83
            ),
            28.3,
        ),
        (
            GuiComponentParameters("vertical_cylinder", 10.0, 0.1, D=20.0, sigma_D=2.0),
            20.0,
        ),
    ],
)
def test_manual_recipe_rejects_hard_core_spacing_violation(component, violating_d):
    assert component.D == violating_d
    with pytest.raises(ValueError, match="violates hard-core spacing"):
        SimulationRecipe(
            topology_id=topology_id_for([component.shape]),
            components=(component,),
            effective_amplitudes=(1.0,),
            resolution=None,
            resolution_effective_amplitude=0.0,
            background=1.0,
            grid=GridProvenance(n_points=64),
            noise=NoiseProvenance(),
            seed=1,
        )


def test_clean_curve_is_exact_authoritative_gui_forward():
    recipe = sample_identifiable_recipe(
        808,
        topology_id=33,
        max_points=128,
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
    )
    curve = simulate_recipe(recipe)
    exact = make_mixed_model([component.shape for component in recipe.components])(
        curve.q, *_authoritative_parameters(recipe)
    )
    np.testing.assert_allclose(curve.clean_intensity, exact, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(curve.intensity, curve.clean_intensity)


def test_noise_is_configurable_without_changing_clean_forward():
    noiseless_recipe = sample_identifiable_recipe(
        42,
        topology_id=4,
        max_points=80,
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
    )
    noisy_recipe = SimulationRecipe(
        topology_id=noiseless_recipe.topology_id,
        components=noiseless_recipe.components,
        effective_amplitudes=noiseless_recipe.effective_amplitudes,
        resolution=noiseless_recipe.resolution,
        resolution_effective_amplitude=noiseless_recipe.resolution_effective_amplitude,
        background=noiseless_recipe.background,
        grid=noiseless_recipe.grid,
        noise=NoiseProvenance(poisson_count_scale=2.0e4, relative_sigma=0.01),
        seed=noiseless_recipe.seed,
    )
    noiseless = simulate_recipe(noiseless_recipe)
    noisy = simulate_recipe(noisy_recipe)
    np.testing.assert_array_equal(noiseless.clean_intensity, noisy.clean_intensity)
    assert not np.array_equal(noisy.intensity, noisy.clean_intensity)


def test_phase2_observation_views_are_deterministic_and_share_clean_physics():
    clean = sample_identifiable_recipe(
        8842,
        topology_id=13,
        max_points=64,
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
    )
    first = sample_observation_view(clean.seed, 0, max_points=512)
    repeated = sample_observation_view(clean.seed, 0, max_points=512)
    second = sample_observation_view(clean.seed, 1, max_points=512)
    assert first == repeated
    assert first.version == OBSERVATION_VIEW_VERSION
    assert first.observation_seed != second.observation_seed
    assert first.grid.kind != second.grid.kind
    assert first.grid.n_points != second.grid.n_points

    observed = first.simulation_recipe(clean)
    assert observed.components == clean.components
    assert observed.effective_amplitudes == clean.effective_amplitudes
    assert observed.resolution == clean.resolution
    assert observed.background == clean.background
    curve = simulate_recipe(observed)
    np.testing.assert_array_equal(first.selection_mask(curve.q), repeated.selection_mask(curve.q))
    assert np.count_nonzero(first.selection_mask(curve.q)) >= 16


def test_observation_policy_never_reads_unknown_physics_and_rejects_noisy_base():
    first_physics = sample_identifiable_recipe(
        42,
        topology_id=0,
        max_points=64,
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
    )
    different_physics = sample_identifiable_recipe(
        42,
        topology_id=13,
        max_points=64,
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
    )
    assert first_physics.components != different_physics.components
    assert sample_observation_view(first_physics.seed, 2) == sample_observation_view(
        different_physics.seed, 2
    )

    noisy = sample_identifiable_recipe(42, topology_id=0, max_points=64)
    with pytest.raises(ValueError, match="noise-free clean recipe"):
        sample_observation_view(noisy.seed, 0).simulation_recipe(noisy)


def test_truth_independent_narrow_window_is_retained_as_ambiguous_or_ood_view():
    clean = SimulationRecipe(
        topology_id=0,
        components=(GuiComponentParameters("sphere", 1.0, 0.1),),
        effective_amplitudes=(100.0,),
        resolution=None,
        resolution_effective_amplitude=0.0,
        background=0.01,
        grid=GridProvenance(q_min=1.0e-3, q_max=5.0, n_points=64),
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
        seed=123,
    )
    view = next(
        sample_observation_view(clean.seed, index)
        for index in range(20)
        if sample_observation_view(clean.seed, index).q_window_id == 0
    )
    assert view.grid.q_max == 0.8 < 4.49 / clean.components[0].R
    observed = view.simulation_recipe(clean)
    assert observed.require_characteristic_coverage is False
    assert np.all(simulate_recipe(observed).intensity > 0.0)


def test_invalid_simulation_configuration_fails_closed():
    base = sample_identifiable_recipe(7, topology_id=1, max_points=64)
    with pytest.raises(ValueError, match="topology"):
        SimulationRecipe(
            topology_id=0,
            components=base.components,
            effective_amplitudes=base.effective_amplitudes,
            resolution=base.resolution,
            resolution_effective_amplitude=base.resolution_effective_amplitude,
            background=base.background,
            grid=base.grid,
            noise=base.noise,
            seed=base.seed,
        )
    with pytest.raises(ValueError, match="at least"):
        SimulationRecipe(
            topology_id=4,
            components=sample_identifiable_recipe(9, 4, 64).components,
            effective_amplitudes=(0.99, 0.01),
            resolution=None,
            resolution_effective_amplitude=0.0,
            background=1.0,
            grid=GridProvenance(n_points=64),
            noise=NoiseProvenance(),
            seed=9,
        )
    with pytest.raises(ValueError, match="zero"):
        SimulationRecipe(
            topology_id=base.topology_id,
            components=base.components,
            effective_amplitudes=base.effective_amplitudes,
            resolution=None,
            resolution_effective_amplitude=1.0,
            background=base.background,
            grid=base.grid,
            noise=base.noise,
            seed=base.seed,
        )
    with pytest.raises(ValueError, match="n_points"):
        GridProvenance(n_points=32)
    with pytest.raises(ValueError, match="relative_sigma"):
        NoiseProvenance(relative_sigma=-0.1)
    with pytest.raises(ValueError, match="fixed"):
        SimulationRecipe(
            topology_id=base.topology_id,
            components=base.components,
            effective_amplitudes=base.effective_amplitudes,
            resolution=base.resolution,
            resolution_effective_amplitude=base.resolution_effective_amplitude,
            background=base.background,
            grid=base.grid,
            noise=base.noise,
            seed=base.seed,
            hard_core_spacing_margin=1.0,
        )
    with pytest.raises((TypeError, ValueError)):
        sample_identifiable_recipe(-1)


def test_grid_coverage_is_validated_for_manual_recipes():
    base = sample_identifiable_recipe(81, topology_id=0, max_points=64)
    with pytest.raises(ValueError, match="does not cover"):
        SimulationRecipe(
            topology_id=base.topology_id,
            components=base.components,
            effective_amplitudes=base.effective_amplitudes,
            resolution=base.resolution,
            resolution_effective_amplitude=base.resolution_effective_amplitude,
            background=base.background,
            grid=GridProvenance(q_min=0.001, q_max=0.01, n_points=64),
            noise=base.noise,
            seed=base.seed,
        )


def test_resolution_shape_must_match_effective_amplitude_presence():
    base = sample_identifiable_recipe(91, topology_id=2, max_points=64)
    with pytest.raises(ValueError, match="positive"):
        SimulationRecipe(
            topology_id=base.topology_id,
            components=base.components,
            effective_amplitudes=base.effective_amplitudes,
            resolution=ResolutionShape(0.01, 5.0),
            resolution_effective_amplitude=0.0,
            background=base.background,
            grid=base.grid,
            noise=base.noise,
            seed=base.seed,
        )
