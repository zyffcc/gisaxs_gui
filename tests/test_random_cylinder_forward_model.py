import numpy as np
from scipy.special import j1

from utils.fitting import _gaussian_grid, cylinder_form_factor_pd


def _scalar_reference(q, radius, sigma_radius, height, sigma_height, *, n_r, n_h, n_orient, nsig=4.0):
    radii, radius_weights = _gaussian_grid(radius, sigma_radius, nsig=nsig, n=n_r, clip_min=0.0)
    heights, height_weights = _gaussian_grid(height, sigma_height, nsig=nsig, n=n_h, clip_min=0.0)
    orientations = np.linspace(0.0, np.pi / 2.0, n_orient)
    orientation_weights = np.sin(orientations)
    orientation_weights /= orientation_weights.sum()

    result = np.zeros_like(q, dtype=float)
    for current_radius, radius_weight in zip(radii, radius_weights):
        for current_height, height_weight in zip(heights, height_weights):
            for alpha, orientation_weight in zip(orientations, orientation_weights):
                radial_x = q * current_radius * np.sin(alpha)
                radial = np.ones_like(radial_x)
                regular = np.abs(radial_x) >= 1e-8
                radial[regular] = 2.0 * j1(radial_x[regular]) / radial_x[regular]
                axial_x = q * current_height * np.cos(alpha) / 2.0
                amplitude = radial * np.sinc(axial_x / np.pi)
                result += radius_weight * height_weight * orientation_weight * amplitude**2
    return result


def test_random_cylinder_vectorization_matches_exact_scalar_reference():
    q = np.geomspace(1e-3, 3.0, 61)
    options = {"n_R": 5, "n_h": 4, "n_orient": 7, "nsig": 4.0}
    actual = cylinder_form_factor_pd(q, 12.0, 1.5, 35.0, 4.0, **options)
    expected = _scalar_reference(
        q,
        12.0,
        1.5,
        35.0,
        4.0,
        n_r=options["n_R"],
        n_h=options["n_h"],
        n_orient=options["n_orient"],
        nsig=options["nsig"],
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_random_cylinder_has_normalized_zero_q_limit():
    result = cylinder_form_factor_pd(np.array([0.0]), 12.0, 1.5, 35.0, 4.0)
    np.testing.assert_allclose(result, np.array([1.0]), rtol=0.0, atol=1e-14)
