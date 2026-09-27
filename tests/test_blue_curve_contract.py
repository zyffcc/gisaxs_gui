"""Contracts of the experimental blue-curve branch, not its training quality."""

import numpy as np

from tools.blue_curve_distillation import CENTERS, EDGES, features, forward


def test_feature_bins_preserve_missing_intervals_without_interpolation():
    a, b = 40, 50
    q = np.array([CENTERS[a], CENTERS[b]])
    x = features(q, [10.0, 30.0], [6.0, 6.0])
    n = len(CENTERS)
    assert x[n + a] == x[n + b] == 1
    assert not x[n + a + 1 : n + b].any()
    assert not x[a + 1 : b].any()


def test_feature_bin_mean_uses_valid_pixel_counts_and_excludes_masked_columns():
    i = 42
    q = np.linspace(EDGES[i] + 1e-5, EDGES[i + 1] - 1e-5, 3)
    x = features(q, [2.0, 10.0, 1e8], [2.0, 6.0, 0.0])
    assert np.isclose(np.exp(x[i]), 8.0)
    assert np.isclose(np.expm1(x[3 * len(CENTERS) + i]), 8.0)


def test_physical_amplitudes_do_not_change_when_render_grid_changes():
    u = np.full(11, 0.5)
    q = np.linspace(0.003, 4.2, 120)
    full = forward(q, u)
    selected = np.arange(13, 101, 3)
    np.testing.assert_allclose(forward(q[selected], u), full[selected], rtol=2e-14)
