from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_sampling_v5 import (
    V5_AMPLITUDE_REGIMES,
    V5_AMPLITUDE_SAMPLING_SCHEMA,
    sample_v5_amplitude_composition,
    sample_v5_constrained_amplitude_composition,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gui_amplitude_constraints import (
    GuiAmplitudeConstraint,
)


def test_composition_is_deterministic_and_preserves_independent_gui_k_and_int():
    first = sample_v5_amplitude_composition(
        3, resolution_present=True, seed=17, regime="balanced_particles"
    )
    replay = sample_v5_amplitude_composition(
        3, resolution_present=True, seed=17, regime="balanced_particles"
    )

    assert first == replay
    assert np.isclose(sum(first.effective_particle_fractions), 1.0)
    assert all(
        np.isclose(amplitude, first.k * intensity)
        for amplitude, intensity in zip(first.particle_amplitudes, first.particle_weights)
    )
    assert np.isclose(first.int_res, first.resolution_amplitude / first.k)
    assert np.isclose(sum(first.coefficient_fractions), 1.0)
    assert first.audit_payload()["coefficient_fraction_is_observability"] is False
    assert first.audit_payload()["schema_version"] == V5_AMPLITUDE_SAMPLING_SCHEMA


def test_weak_particle_stratum_removes_the_old_fifteen_percent_floor():
    composition = sample_v5_amplitude_composition(
        4,
        resolution_present=False,
        seed=31,
        regime="weak_particle_coefficient",
    )
    weak_slot = composition.selected_particle_slot

    assert weak_slot is not None
    assert composition.effective_particle_fractions[weak_slot] < 0.03
    assert composition.effective_particle_fractions[weak_slot] < 0.15
    assert composition.resolution_amplitude == 0.0


def test_background_and_resolution_confounding_are_explicit_strata():
    background = sample_v5_amplitude_composition(
        1,
        resolution_present=False,
        seed=41,
        regime="background_confounded",
    )
    resolution = sample_v5_amplitude_composition(
        2,
        resolution_present=True,
        seed=42,
        regime="resolution_confounded",
    )

    assert 0.05 <= background.background / background.k <= 20.0
    assert 0.1 <= resolution.int_res <= 10.0


def test_constrained_sampler_supports_k10_int2_without_canonicalizing_it():
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.5, 0.5),
        component_intensities=(ClosedInterval(2.0, 2.0),),
        k=ClosedInterval(10.0, 10.0),
        resolution_present=False,
    )
    result = sample_v5_constrained_amplitude_composition(
        constraint,
        seed=17,
        regime="balanced_particles",
    )
    assert result.k == 10.0
    assert result.particle_weights == (2.0,)
    assert result.particle_amplitudes == (20.0,)
    assert constraint.contains(result.coefficient_vector, k=result.k)


def test_default_schedule_only_selects_branch_compatible_regimes():
    absent = {
        sample_v5_amplitude_composition(2, resolution_present=False, seed=seed).regime
        for seed in range(12)
    }
    present = {
        sample_v5_amplitude_composition(2, resolution_present=True, seed=seed).regime
        for seed in range(12)
    }

    assert not any(value.startswith("resolution_") for value in absent)
    assert present == set(V5_AMPLITUDE_REGIMES)


def test_amplitude_composition_fails_closed():
    with pytest.raises(ValueError, match="for this branch"):
        sample_v5_amplitude_composition(
            1,
            resolution_present=False,
            seed=1,
            regime="resolution_confounded",
        )
    valid = sample_v5_amplitude_composition(
        2, resolution_present=True, seed=9, regime="balanced_particles"
    )
    assert replace(valid, resolution_amplitude=0.0).resolution_amplitude == 0.0
    with pytest.raises(ValueError, match="at least one positive"):
        replace(valid, particle_amplitudes=(0.0, 0.0))
    with pytest.raises(ValueError, match="zero when Resolution is absent"):
        replace(valid, resolution_present=False)
    with pytest.raises(ValueError, match="unsupported V5 amplitude generator"):
        replace(valid, generator_version="legacy_simplex_v2")
