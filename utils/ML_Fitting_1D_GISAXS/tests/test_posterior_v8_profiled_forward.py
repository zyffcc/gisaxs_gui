from __future__ import annotations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    NonlinearComponent,
    ResolutionShape,
    build_design_matrix,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
    to_gui_mixed_model_parameters,
)


def _curve(q, components, coefficients, resolution=None):
    design = build_design_matrix(q, components, resolution)
    return design @ np.asarray(coefficients, dtype=np.float64)


def test_weighted_single_sphere_recovers_exact_gui_gauge_and_forward():
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
    expected = np.array([0.003, 2.4, 0.15])  # BG, sphere amplitude, resolution amplitude
    intensity = _curve(q, components, expected, resolution)
    sigma = np.geomspace(0.2, 3.0, q.size)

    result = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
        sigma=sigma,
    )

    assert result.background == pytest.approx(expected[0], rel=1e-9, abs=1e-11)
    assert result.particle_amplitudes == pytest.approx((expected[1],), rel=1e-9)
    assert result.resolution_amplitude == pytest.approx(expected[2], rel=1e-9)
    assert result.k == pytest.approx(expected[1], rel=1e-9)
    assert result.component_weights == pytest.approx((1.0,), rel=1e-12)
    assert result.int_res == pytest.approx(expected[2] / expected[1], rel=1e-9)
    assert result.weighted_rss < 1e-20
    np.testing.assert_allclose(result.fitted_intensity, intensity, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        evaluate_profiled_forward(q, result),
        intensity,
        rtol=1e-10,
        atol=1e-12,
    )


def test_mixed_sphere_cylinder_vertical_recovers_all_amplitudes_and_forward_parity():
    q = np.geomspace(0.003, 2.5, 420)
    components = (
        NonlinearComponent("sphere", R=14.0, sigma_R=1.1),
        NonlinearComponent(
            "cylinder",
            R=5.5,
            sigma_R=0.6,
            h=48.0,
            sigma_h=4.5,
            D=42.0,
            sigma_D=4.0,
        ),
        NonlinearComponent(
            "vertical cylinder",
            R=27.0,
            sigma_R=0.16,
            D=76.0,
            sigma_D=8.0,
        ),
    )
    resolution = ResolutionShape(sigma_res=0.012, nu_res=7.0)
    expected = np.array([0.006, 1.3, 0.45, 3.2, 0.21])
    intensity = _curve(q, components, expected, resolution)

    result = profile_linear_amplitudes(q, intensity, components, resolution=resolution)

    assert result.background == pytest.approx(expected[0], rel=2e-8, abs=1e-10)
    assert result.particle_amplitudes == pytest.approx(tuple(expected[1:4]), rel=2e-8)
    assert result.resolution_amplitude == pytest.approx(expected[4], rel=2e-8)
    expected_k = float(np.sum(expected[1:4]))
    assert result.k == pytest.approx(expected_k, rel=2e-8)
    assert result.component_weights == pytest.approx(
        tuple(expected[1:4] / expected_k), rel=2e-8
    )

    spec, params = to_gui_mixed_model_parameters(result)
    assert spec == ["sphere", "cylinder", "vertical_cylinder"]
    assert params[-1] == pytest.approx(expected_k, rel=2e-8)
    np.testing.assert_allclose(
        evaluate_profiled_forward(q, result),
        result.fitted_intensity,
        rtol=2e-10,
        atol=2e-11,
    )
    np.testing.assert_allclose(result.fitted_intensity, intensity, rtol=2e-8, atol=2e-10)


def test_resolution_absent_omits_basis_and_maps_to_exact_zero_intensity():
    q = np.geomspace(0.01, 1.5, 260)
    components = (
        NonlinearComponent("sphere", R=8.0, sigma_R=0.7),
        NonlinearComponent(
            "vertical_cylinder",
            R=22.0,
            sigma_R=0.2,
            D=65.0,
            sigma_D=6.0,
        ),
    )
    expected = np.array([0.02, 0.75, 2.1])
    intensity = _curve(q, components, expected)

    result = profile_linear_amplitudes(q, intensity, components)

    assert build_design_matrix(q, components).shape == (q.size, 3)
    assert result.resolution_present is False
    assert result.resolution_amplitude == 0.0
    assert result.int_res == 0.0
    assert result.gui_global_parameters()["int_Res"] == 0.0
    np.testing.assert_allclose(
        evaluate_profiled_forward(q, result),
        intensity,
        rtol=2e-9,
        atol=2e-11,
    )


def test_profiled_forward_rejects_invalid_arrays_and_component_semantics():
    q = np.geomspace(0.01, 1.0, 40)
    component = NonlinearComponent("sphere", R=10.0, sigma_R=1.0)
    intensity = _curve(q, (component,), [0.01, 1.0])

    with pytest.raises(ValueError, match="same shape"):
        profile_linear_amplitudes(q, intensity[:-1], (component,))
    with pytest.raises(ValueError, match="strictly positive"):
        profile_linear_amplitudes(q, intensity, (component,), sigma=np.zeros_like(q))
    with pytest.raises(ValueError, match="sigma_D must be finite and positive"):
        NonlinearComponent("sphere", R=10.0, sigma_R=1.0, D=20.0, sigma_D=0.0)
    with pytest.raises(ValueError, match="requires h"):
        NonlinearComponent("cylinder", R=10.0, sigma_R=1.0)
