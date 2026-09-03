from __future__ import annotations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_REGIMES,
    V5_BACKGROUND_DOMAIN,
    V5_COMPONENT_INTENSITY_DOMAIN,
    V5_INT_RES_DOMAIN,
    V5_K_DOMAIN,
    V5AmplitudeRangeRegimes,
    amplitude_axis_range_regimes_for,
    full_range_v5_amplitude_query,
    sample_v5_amplitude_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_sampling_v5 import (
    sample_v5_constrained_amplitude_composition,
)


@pytest.mark.parametrize("count", (1, 2, 3, 4))
@pytest.mark.parametrize("policy", ("absent", "optional", "required"))
@pytest.mark.parametrize("regime", V5_AMPLITUDE_RANGE_REGIMES)
def test_amplitude_query_is_truth_independent_replayable_and_branch_feasible(
    count, policy, regime
):
    query = sample_v5_amplitude_query(
        count,
        resolution_presence_policy=policy,
        query_seed=1701 + count,
        range_regime=regime,
    )
    replay = sample_v5_amplitude_query(
        count,
        resolution_presence_policy=policy,
        query_seed=1701 + count,
        range_regime=regime,
    )

    assert query == replay
    for resolution_present in query.allowed_resolution_states:
        constraint = query.constraint_for_branch(resolution_present=resolution_present)
        composition = sample_v5_constrained_amplitude_composition(
            constraint,
            seed=37,
            regime="balanced_particles",
        )
        assert constraint.contains(composition.coefficient_vector, k=composition.k)


def test_full_range_supports_weak_dominant_and_resolution_confounding_strata():
    query = full_range_v5_amplitude_query(4, resolution_presence_policy="required")
    constraint = query.constraint_for_branch(resolution_present=True)
    for regime in (
        "weak_particle_coefficient",
        "dominant_particle_coefficient",
        "background_confounded",
        "resolution_weak_coefficient",
        "resolution_confounded",
    ):
        result = sample_v5_constrained_amplitude_composition(
            constraint,
            seed=91,
            regime=regime,
        )
        assert constraint.contains(result.coefficient_vector, k=result.k)
    weak = sample_v5_constrained_amplitude_composition(
        constraint,
        seed=92,
        regime="weak_particle_coefficient",
    )
    assert min(weak.particle_weights) < 0.03


def test_fixed_query_produces_the_exact_fixed_independent_intensities():
    query = sample_v5_amplitude_query(
        3,
        resolution_presence_policy="absent",
        query_seed=89,
        range_regime="fixed",
    )
    result = sample_v5_constrained_amplitude_composition(
        query.constraint_for_branch(resolution_present=False),
        seed=11,
        regime="balanced_particles",
    )
    expected = np.asarray([value.low for value in query.component_intensities])
    np.testing.assert_allclose(result.particle_weights, expected, rtol=0.0, atol=2e-12)


def test_infeasible_requested_diagnostic_regime_fails_instead_of_mislabeling():
    query = sample_v5_amplitude_query(
        1,
        resolution_presence_policy="absent",
        query_seed=7,
        range_regime="fixed",
    )
    with pytest.raises(ValueError, match="regime is infeasible"):
        sample_v5_constrained_amplitude_composition(
            query.constraint_for_branch(resolution_present=False),
            seed=3,
            regime="weak_particle_coefficient",
        )


def test_engineering_sampler_accepts_an_explicit_mixed_per_axis_assignment():
    regimes = V5AmplitudeRangeRegimes.create(
        2,
        resolution_presence_policy="required",
        background="fixed",
        k="full",
        component_intensities=("narrow", "edge_high"),
        int_res="edge_low",
    )

    query = sample_v5_amplitude_query(
        2,
        resolution_presence_policy="required",
        query_seed=1701,
        range_regimes=regimes,
    )

    assert query.background.low == query.background.high
    assert query.k == V5_K_DOMAIN
    assert V5_COMPONENT_INTENSITY_DOMAIN.contains(query.component_intensities[0].low)
    assert V5_COMPONENT_INTENSITY_DOMAIN.contains(query.component_intensities[0].high)
    assert query.component_intensities[1].high == V5_COMPONENT_INTENSITY_DOMAIN.high
    assert query.int_res is not None
    assert query.int_res.low == V5_INT_RES_DOMAIN.low
    assert regimes.summary == "mixed"
    assert regimes.inactive_axis_names == ("Int_3", "Int_4")
    assert regimes.audit_payload()["normalization"] == "none"
    assert V5_BACKGROUND_DOMAIN.contains(query.background.low)


def test_default_engineering_schedule_is_independent_per_axis_and_replayable():
    seen = {name: set() for name in ("BG", "k", "Int_1", "Int_2", "int_Res")}
    for seed in range(1024):
        first = amplitude_axis_range_regimes_for(
            2,
            resolution_presence_policy="required",
            query_seed=seed,
        )
        replay = amplitude_axis_range_regimes_for(
            2,
            resolution_presence_policy="required",
            query_seed=seed,
        )
        assert first == replay
        payload = first.audit_payload()["axis_regimes"]
        for name in seen:
            seen[name].add(payload[name])
    assert all(values == set(V5_AMPLITUDE_RANGE_REGIMES) for values in seen.values())


def test_mixed_assignment_validation_and_stale_versions_fail_closed():
    regimes = V5AmplitudeRangeRegimes.create(
        1,
        resolution_presence_policy="absent",
        background="full",
        k="fixed",
        component_intensities=("narrow",),
        int_res=None,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_v5_amplitude_query(
            1,
            resolution_presence_policy="absent",
            query_seed=1,
            range_regime="full",
            range_regimes=regimes,
        )
    with pytest.raises(ValueError, match="component count"):
        sample_v5_amplitude_query(
            2,
            resolution_presence_policy="absent",
            query_seed=1,
            range_regimes=regimes,
        )
    with pytest.raises(ValueError, match="unsupported amplitude range-assignment version"):
        V5AmplitudeRangeRegimes(
            background="full",
            k="full",
            component_intensities=("full", None, None, None),
            int_res=None,
            version="legacy-shared-regime-v0",
        )
