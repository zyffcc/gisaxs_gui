"""Regression tests for synthetic count noise without peak clipping."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TrainSetBuild.noise import _sample_counts, add_noise


def test_normal_count_path_is_unchanged():
    expected = np.array([0.0, 1.0, 200.0, 1e9])
    np.testing.assert_array_equal(
        _sample_counts(expected, np.random.default_rng(7)),
        np.random.default_rng(7).poisson(expected),
    )


def test_high_count_peak_is_not_clipped():
    clean = np.array([1.0, 1.0, 1.0, 1e12])
    noisy, sigma = add_noise(
        clean, np.random.default_rng(17),
        poisson_scale_min=200, poisson_scale_max=200,
        rel_noise_min=0, rel_noise_max=0,
    )
    assert abs(noisy[-1] / clean[-1] - 1) < 1e-5
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)


@pytest.mark.parametrize("bad", [np.inf, np.nan, -1.0])
def test_invalid_expected_counts_fail_explicitly(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _sample_counts(np.array([bad]), np.random.default_rng(1))
