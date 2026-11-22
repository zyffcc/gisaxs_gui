"""Secondary-standard scale and uncertainty on synthetic data."""

import numpy as np
import pytest

from src.gimap.features.analyze.domain.absolute_scale import fit_standard, water_intensity


def test_poisson_water_recovers_factor_within_one_standard_uncertainty():
    q = np.linspace(0.02, 0.18, 160)
    true_factor = 2.4e-6
    terms = dict(exposure_s=2.0, monitor=5.0, transmission=0.8, thickness_mm=3.0)
    denominator = 2.0 * 5.0 * 0.8 * 0.3
    expected = 0.01632 * denominator / true_factor
    counts = np.random.default_rng(849).poisson(expected, q.size)
    fit = fit_standard(q, counts, np.sqrt(counts), **terms, reference_fractional_sigma=0.001)
    assert abs(fit.factor - true_factor) < fit.sigma_factor
    assert 0.001 < fit.sigma_factor / fit.factor < 0.002
    np.testing.assert_allclose(fit.residual, counts * fit.factor / denominator - 0.01632)


def test_glassy_carbon_shape_interpolated_over_measured_overlap_only():
    nodes = np.linspace(.03, .23, 21)
    curve = 60.0 / (1.0 + (nodes / .07)**2)**1.65
    q = np.arange(1, 28) * .009
    scale = 0.031
    measured = np.interp(q, nodes, curve) * 0.7 / scale  # standard thickness 7 mm
    reference = (nodes[::-1], curve[::-1], .015 * curve[::-1])
    fit = fit_standard(q[::-1], measured[::-1], thickness_mm=7, reference=reference)
    np.testing.assert_array_equal(fit.q, q[(q >= nodes[0]) & (q <= nodes[-1])])
    assert fit.factor == pytest.approx(scale)
    assert fit.sigma_factor > 0
    np.testing.assert_allclose(fit.residual, 0, atol=5e-14)
    fit = fit_standard(q, measured, thickness_mm=7, reference=reference, q_range=(.06, .10))
    np.testing.assert_array_equal(fit.q, q[(q >= .06) & (q <= .10)])


def test_temperature_lookup_and_thickness_conversion():
    assert water_intensity(20) == .01632
    assert water_intensity(5) == .01665
    assert water_intensity(2.5) == pytest.approx((.01680 + .01675) / 2)
    fit = fit_standard([0, 1], [1, 1], thickness_mm=1.0, temperature_c=5)
    assert fit.factor == pytest.approx(.01665 * .1)
    for value in [-1, 101, np.nan]:
        with pytest.raises(ValueError, match="temperature"):
            water_intensity(value)


def test_reference_only_errors_give_wls_on_measured_axis():
    q = np.arange(4)
    x = np.array([1., 2., 3., 4.])
    reference = np.array([2., 5., 6., 7.8])
    sigma = np.array([.2, 1., .4, .1])
    fit = fit_standard(q, x, reference=(q, reference, sigma), thickness_mm=10)
    weights = 1 / sigma**2
    assert fit.factor == pytest.approx((weights * x) @ reference / ((weights * x) @ x))


def test_uncertainty_on_both_axes_matches_through_origin_deming_solution():
    q = np.arange(4)
    x = np.array([1.2, 2.0, 2.9, 4.1])
    ref = np.array([2.3, 4.9, 5.6, 8.4])
    u, v = 0.2, 0.3
    fit = fit_standard(q, x, np.full(4, u), reference=(q, ref, np.full(4, v)), thickness_mm=10)
    ratio = v**2 / u**2
    delta = ref @ ref - ratio * (x @ x)
    cross = x @ ref
    expected = (delta + np.sqrt(delta**2 + 4 * ratio * cross**2)) / (2 * cross)
    assert fit.factor == pytest.approx(expected, rel=1e-10)


def test_interpolation_keeps_shared_node_uncertainties_and_common_scale():
    nodes = np.array([.02, .18])
    reference = (nodes, np.full(2, 20.), np.full(2, 1.))
    # Even a dense exact measurement constrains just two independent certificate values.
    for count in (2, 1001):
        q = np.linspace(*nodes, count)
        fit = fit_standard(q, np.full(count, 5.), thickness_mm=10, reference=reference,
                           reference_fractional_sigma=.03)
        assert fit.factor == pytest.approx(4)
        assert fit.sigma_factor == pytest.approx(4 * np.hypot(1 / (20 * np.sqrt(2)), .03))


def test_scatter_without_quoted_errors_and_excess_scatter_with_errors():
    q = np.arange(5)
    x = np.array([10, 14, 9, 11, 7])
    expected_a = .01632 / x.mean()
    expected_sigma = expected_a * x.std(ddof=1) / (np.sqrt(len(q)) * x.mean())
    for sigma in (None, np.full(5, 0.01)):
        fit = fit_standard(q, x, sigma, thickness_mm=10)
        assert fit.factor == pytest.approx(expected_a)
        assert fit.sigma_factor == pytest.approx(expected_sigma)


def test_fit_rejects_empty_overlap_ambiguous_grid_and_nonpositive_signal():
    with pytest.raises(ValueError, match="shared range"):
        fit_standard([0, 1], [10, 11], thickness_mm=10, q_range=(3, 4))
    with pytest.raises(ValueError, match="distinct"):
        fit_standard([0, 1], [10, 11], thickness_mm=10, reference=([1, 1], [2, 3], [1, 1]))
    with pytest.raises(ValueError, match="positive scale"):
        fit_standard([0, 1], [-10, -1], thickness_mm=10)
