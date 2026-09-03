from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_polish import (
    AmplitudeBounds,
    polish_profiled_amplitudes,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    STANDARDIZED_LOG_RMSE_METRIC,
    natural_log_rmse,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    NonlinearComponent,
    ResolutionShape,
    build_design_matrix,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)


def _sphere_problem():
    q = np.geomspace(0.004, 2.0, 320)
    components = (
        NonlinearComponent(
            "sphere",
            R=11.0,
            sigma_R=1.3,
            D=34.0,
            sigma_D=3.5,
        ),
    )
    resolution = ResolutionShape(sigma_res=0.018, nu_res=6.0)
    coefficients = np.array([0.003, 2.4, 0.15])
    intensity = build_design_matrix(q, components, resolution) @ coefficients
    return q, components, resolution, coefficients, intensity


def test_clean_synthetic_recovers_joint_amplitudes_and_exact_gui_forward():
    q, components, resolution, expected, intensity = _sphere_problem()
    exact_profile = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
    )
    perturbed_start = replace(
        exact_profile,
        background=0.012,
        particle_amplitudes=(0.9,),
        resolution_amplitude=0.32,
    )

    result = polish_profiled_amplitudes(
        q,
        intensity,
        perturbed_start,
        max_nfev=200,
    )

    assert result.success
    assert result.nfev > 0
    assert result.final_metric <= result.initial_metric
    assert result.final_metric < 1e-10
    assert result.returned_source in {"optimizer_terminal", "optimizer_intermediate"}
    assert result.best_residual_call > 0
    assert result.final_coefficients == pytest.approx(tuple(expected), rel=2e-8, abs=1e-12)
    assert result.final_profile.k == pytest.approx(expected[1])
    assert result.final_profile.component_weights == pytest.approx((1.0,))
    assert result.final_profile.int_res == pytest.approx(expected[2] / expected[1])
    np.testing.assert_allclose(result.exact_intensity, intensity, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.exact_intensity,
        evaluate_profiled_forward(q, result.final_profile),
        rtol=2e-12,
        atol=1e-13,
    )
    assert result.exact_intensity.flags.writeable is False


def test_standardized_log_objective_matches_evaluation_and_does_not_worsen():
    q, components, resolution, _, clean = _sphere_problem()
    phase = np.linspace(0.0, 8.0 * np.pi, q.size)
    sigma_log = 0.015 + 0.025 * np.sin(np.linspace(0.0, 2.0 * np.pi, q.size)) ** 2
    observed = clean * np.exp(0.06 * np.sin(phase) + 0.025 * np.cos(0.7 * phase))
    # This is the fast weighted-linear approximation used inside nonlinear
    # refinement.  The polish must optimize the exact evaluation objective.
    initial = profile_linear_amplitudes(
        q,
        observed,
        components,
        resolution=resolution,
        sigma=observed * sigma_log,
    )

    result = polish_profiled_amplitudes(
        q,
        observed,
        initial,
        sigma_log=sigma_log,
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        max_nfev=200,
    )

    assert result.final_metric <= result.initial_metric
    assert result.final_metric < result.initial_metric - 1e-4
    assert result.initial_metric == pytest.approx(
        natural_log_rmse(
            evaluate_profiled_forward(q, result.initial_profile),
            observed,
            sigma_log=sigma_log,
        )
    )
    assert result.final_metric == pytest.approx(
        natural_log_rmse(result.exact_intensity, observed, sigma_log=sigma_log)
    )
    assert result.final_standardized_log_rmse == pytest.approx(result.final_metric)
    assert result.final_profile.weighted_rss == pytest.approx(
        q.size * result.final_metric**2
    )
    assert result.final_raw_log_rmse == pytest.approx(
        natural_log_rmse(result.exact_intensity, observed)
    )


def test_resolution_absent_zero_component_and_fixed_bounds_are_supported():
    q = np.geomspace(0.006, 1.8, 300)
    components = (
        NonlinearComponent("sphere", R=8.0, sigma_R=0.7),
        NonlinearComponent(
            "vertical cylinder",
            R=22.0,
            sigma_R=0.2,
            D=65.0,
            sigma_D=6.0,
        ),
    )
    expected = np.array([0.02, 0.75, 0.0])
    intensity = build_design_matrix(q, components) @ expected
    initial = profile_linear_amplitudes(q, intensity, components)
    initial = replace(
        initial,
        background=0.0,
        particle_amplitudes=(0.6, 0.0),
        resolution_amplitude=0.0,
    )
    bounds = AmplitudeBounds(
        lower=(1e-8, 0.0, 0.0),
        upper=(0.1, 2.0, 0.0),
    )

    result = polish_profiled_amplitudes(q, intensity, initial, bounds=bounds)

    assert result.bounds_satisfied
    assert result.coefficient_bounds.contains(result.final_coefficients)
    assert result.initial_coefficients[0] == pytest.approx(1e-8)
    assert result.final_coefficients == pytest.approx(tuple(expected), rel=2e-8, abs=1e-12)
    assert result.final_profile.resolution is None
    assert result.final_profile.resolution_amplitude == 0.0
    assert result.final_profile.int_res == 0.0
    np.testing.assert_allclose(result.exact_intensity, intensity, rtol=2e-10, atol=1e-12)


def test_maximum_evaluations_reports_failure_without_returning_a_worse_curve():
    q, components, resolution, _, intensity = _sphere_problem()
    initial = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
    )
    initial = replace(
        initial,
        background=0.012,
        particle_amplitudes=(0.9,),
        resolution_amplitude=0.32,
    )

    result = polish_profiled_amplitudes(q, intensity, initial, max_nfev=1)

    assert result.success is False
    assert result.nfev == 1
    assert result.final_metric <= result.initial_metric
    assert result.bounds_satisfied


def test_default_bounds_accept_authoritative_zero_or_tiny_background():
    q, components, resolution, _, intensity = _sphere_problem()
    initial = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
    )
    for background in (0.0, np.finfo(np.float64).tiny):
        candidate = replace(initial, background=background)
        result = polish_profiled_amplitudes(q, intensity, candidate, max_nfev=5)
        assert result.bounds_satisfied
        assert result.final_metric <= result.initial_metric


def test_invalid_inputs_and_infeasible_bounds_fail_closed():
    q, components, resolution, _, intensity = _sphere_problem()
    initial = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
    )

    with pytest.raises(ValueError, match="q must contain finite"):
        polish_profiled_amplitudes(q * np.nan, intensity, initial)
    with pytest.raises(ValueError, match="intensity must contain finite"):
        polish_profiled_amplitudes(q, np.full_like(intensity, np.inf), initial)
    with pytest.raises(ValueError, match="sigma_log must contain finite"):
        polish_profiled_amplitudes(q, intensity, initial, sigma_log=np.zeros_like(q))
    with pytest.raises(ValueError, match="requires sigma_log"):
        polish_profiled_amplitudes(
            q,
            intensity,
            initial,
            metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        )
    zero_background = AmplitudeBounds(
        lower=(0.0, 0.0, 0.0), upper=(1.0, 2.0, 1.0)
    )
    assert zero_background.contains((0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="non-negative"):
        AmplitudeBounds(lower=(-1e-8, 0.0, 0.0), upper=(1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="length does not match"):
        polish_profiled_amplitudes(
            q,
            intensity,
            initial,
            bounds=AmplitudeBounds(lower=(1e-8, 0.0), upper=(1.0, 2.0)),
        )
    with pytest.raises(ValueError, match="outside the requested bounds"):
        polish_profiled_amplitudes(
            q,
            intensity,
            initial,
            bounds=AmplitudeBounds(
                lower=(1e-8, 0.0, 0.0),
                upper=(1.0, 1.0, 1.0),
            ),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        polish_profiled_amplitudes(
            q,
            intensity,
            replace(initial, particle_amplitudes=(-1.0,)),
        )
