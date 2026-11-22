"""Errors on reduced means, using synthetic detector lines and histogram bins."""

import numpy as np
import pytest

from src.gimap.features.analyze.domain.binning import BinnedMean, native_profile
from src.gimap.features.analyze.domain.curve_errors import PixelErrors


def test_sparse_lines_use_robust_intensity_dependent_pixel_variance():
    means = np.linspace(0, 70, 71)
    occupancy = np.full(71, 20)
    occupancy[[10, 22, 28]] = [1, 9, 10]
    target = 4 + 1.5 * means
    frame = np.full((20, 71), np.nan)
    for i, n in enumerate(occupancy):
        delta = np.arange(n) - 0.5 * (n - 1)
        if n > 1:
            delta *= np.sqrt(target[i]) / delta.std(ddof=1)
        if i in (22, 28):
            delta *= 0.01  # two very small empirical variances, on either side of the boundary
        if i == 43:
            delta *= 100.0  # outlier in a densely sampled line
        frame[:n, i] = means[i] + delta

    def cut(errors):
        return native_profile(frame, np.isfinite(frame), means, axis=0, errors=errors)

    _, signal, pooled, n = cut(PixelErrors(model="scatter"))
    _, _, independent, _ = cut(PixelErrors(model="scatter", pooled_scatter=False))
    np.testing.assert_allclose(signal, means, atol=1e-12)
    np.testing.assert_allclose(n, occupancy)
    np.testing.assert_allclose(pooled[[10, 22]]**2 * n[[10, 22]], target[[10, 22]], rtol=1e-5)
    np.testing.assert_allclose(pooled[n >= 10], independent[n >= 10])
    assert independent[22] < 0.02 * pooled[22]


def test_pooled_variance_fits_the_bins_after_they_are_merged():
    means = np.arange(12, dtype=float)
    variance = 4 + means
    delta = np.arange(8) - 3.5
    values = means[:, None] + delta / delta.std(ddof=1) * np.sqrt(variance[:, None])
    values[5] = means[5]  # fluctuations in this merged point happened to be zero
    hist = BinnedMean.linear(-0.5, values.size - 0.5, values.size)
    hist.add(np.arange(values.size), values.ravel())
    _, signal, sigma, n = hist.result(counts=PixelErrors(model="scatter"), merge=True)
    np.testing.assert_equal(n, 8)
    np.testing.assert_allclose(signal, means, atol=1e-12)
    np.testing.assert_allclose(sigma**2 * n, variance, rtol=1e-5)


def test_dense_merged_scatter_and_observed_one_count_floors_keep_independent_bin_errors():
    # Two bins of 5 pixels merge to n=10. Their mean intensities differ,
    # but the gradient between bins is not part of the measured within-bin scatter.
    x = np.repeat(np.arange(4), [5, 5, 8, 8])
    baseline = np.repeat([0., 20., 40., 60.], [5, 5, 8, 8])
    data = baseline + np.tile([-1., 1.], 13)
    hist = BinnedMean.linear(-0.5, 3.5, 4)
    hist.add(x, data)
    scatter = PixelErrors(model="scatter")
    _, _, individual, n = hist.result(counts=PixelErrors(model="scatter", pooled_scatter=False))
    for errors in (scatter, PixelErrors(model="scatter", pooled_scatter=False)):
        sigma = hist.result(counts=errors, merge=True)[2]
        assert sigma[0] == pytest.approx(np.hypot(*(individual * n)[:2]) / 10)
    zero = BinnedMean.linear(-0.5, 3.5, 4)
    zero.add(x, np.zeros_like(x))
    assert zero.result(merge=True)[2][0] == pytest.approx(np.sqrt(2)/10)


def test_pooling_two_pixel_bins_recovers_noise_despite_skew_in_sample_variances():
    rng = np.random.default_rng(443)
    intensity = np.linspace(10, 400, 1200)
    expected = 4 + 0.2 * intensity
    frame = intensity + np.sqrt(expected) * rng.normal(size=(2, len(intensity)))
    _, observed, sigma, n = native_profile(frame, np.isfinite(frame), intensity, axis=0,
                                            errors=PixelErrors(model="scatter"))
    recovered = sigma**2 * n / (4 + 0.2 * observed)
    assert np.median(recovered[:400]) == pytest.approx(1, abs=0.14)
    assert np.median(recovered[-400:]) == pytest.approx(1, abs=0.14)


def test_sparse_bins_keep_measured_scatter_where_the_variance_line_crosses_zero():
    # Within-bin texture makes the variance grow faster than linearly. The
    # faint end has measurable noise even though the fitted line is negative.
    means = np.arange(101.)
    variance = 9 + means**2
    offsets = np.arange(6) - 2.5
    frame = means + offsets[:, None] / offsets.std(ddof=1) * np.sqrt(variance)
    _, _, sigma, n = native_profile(frame, np.isfinite(frame), means, axis=0,
                                    errors=PixelErrors(model="scatter"))
    pooled = sigma**2 * n
    assert pooled[0] > 0
    on_floor = np.isclose(pooled, pooled[0])
    assert 1 < on_floor.sum() < len(means)
    assert np.all(np.diff(pooled) >= -1e-10)


@pytest.mark.parametrize("native", [False, True])
def test_smoothed_poisson_respects_gaps_occupancy_and_reference_noise(native):
    frame = np.array([[2., 8., 0., np.nan, 40., 48., np.nan, 3.],
                      [2., np.nan, 0., np.nan, np.nan, np.nan, np.nan, np.nan]])
    valid = np.isfinite(frame)
    x = np.broadcast_to(np.arange(8.), frame.shape)
    scale = 0.5
    # The sample shot term is distinct from the reduced intensity (e.g. after subtraction).
    errors = PixelErrors(scale=scale, variance=frame * scale**2, extra_variance=0.75, poisson_variance="smoothed")
    signal = frame * scale - 3
    if native:
        result = native_profile(signal, valid, x, axis=0, errors=errors)
    else:
        hist = BinnedMean.linear(-0.5, 7.5, 8)
        hist.add(x[valid], signal[valid], errors=errors.subset(valid))
        result = hist.result(counts=errors)
    _, mean, sigma, n = result
    # First bin's neighbours have 8 counts in 3 pixels, second bin's have 4 in 4.
    expected_rate = np.array([8/3, 1, 12/3, 48, 40, 3])
    expected = np.maximum(expected_rate * n, 1) * scale**2 + 0.75 * n
    np.testing.assert_allclose(sigma**2 * n**2, expected)
    np.testing.assert_allclose(mean, np.array([2, 8, 0, 40, 48, 3]) * scale - 3)


def test_smoothed_weights_recover_constant_low_count_intensity():
    observations = np.random.default_rng(38).poisson(3.0, size=6000)
    hist = BinnedMean.linear(-0.5, len(observations) - 0.5, len(observations))
    hist.add(np.arange(len(observations)), observations)
    _, observed_mean, observed_sigma, _ = hist.result()
    _, smoothed_mean, smoothed_sigma, _ = hist.result(counts=PixelErrors(poisson_variance="smoothed"))
    np.testing.assert_array_equal(observed_mean, smoothed_mean)
    np.testing.assert_allclose(observed_sigma**2, np.maximum(observations, 1))
    assert np.average(observed_mean, weights=1/observed_sigma**2) < 2.4
    assert np.average(smoothed_mean, weights=1/smoothed_sigma**2) == pytest.approx(3, abs=0.09)


def test_given_pixel_variance_keeps_small_errors_and_skips_poisson_smoothing():
    hist = BinnedMean.linear(0, 2, 2)
    errors = PixelErrors(model="sigma", variance=np.array([0.01, 0.04, 0.09]), extra_variance=0.001,
                         poisson_variance="smoothed")
    hist.add(np.array([0.3, 0.5, 1.5]), np.array([-3., 2., 4.]), errors=errors)
    _, _, sigma, n = hist.result(counts=errors)
    np.testing.assert_allclose((sigma * n)**2, [0.052, 0.091])
