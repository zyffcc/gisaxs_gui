"""Clean cut curves: bins no finer than pixels, sparse bins joined, defective detector rows left out,
and a log axis that is not stretched by a few near-zero bins."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.analyze.domain import GiwaxsSettings, find_bad_pixels, giwaxs_maps, reduce_giwaxs
from src.gimap.features.analyze.domain.binning import merge_sparse_bins
from src.gimap.shared.geometry import DetectorGeometry
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, WAVELENGTH, giwaxs_frame
from tests.test_assistant_gui import _app

GEOMETRY = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2)


def test_sparse_bins_are_joined_into_points_with_enough_pixels() -> None:
    x = np.arange(10, dtype=float)
    pixels = np.array([2, 3, 1, 50, 60, 55, 52, 58, 4, 3])
    mean = np.array([10, 20, 40, 5, 5, 5, 5, 5, 1, 2], dtype=float)
    sigma = np.sqrt(mean * pixels) / pixels
    mx, mmean, msigma, mpixels = merge_sparse_bins(x, mean, sigma, pixels, min_pixels=6)
    assert mpixels[0] == 6 and mx[0] == pytest.approx((0 * 2 + 1 * 3 + 2 * 1) / 6)
    assert mmean[0] == pytest.approx((10 * 2 + 20 * 3 + 40 * 1) / 6)  # the mean of all its pixels
    assert msigma[0] == pytest.approx(np.sqrt(10 * 2 + 20 * 3 + 40) / 6)
    assert list(mpixels[1:6]) == [50, 60, 55, 52, 58] and mpixels[-1] == 7  # the sparse tail joins
    full = merge_sparse_bins(x, mean, sigma, np.full(10, 50))
    assert full[0] is x or np.array_equal(full[0], x)
    gap = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    _x, _m, _s, merged = merge_sparse_bins(gap, np.ones(5), np.ones(5), np.array([2, 2, 50, 2, 50]), min_pixels=5)
    assert 2 + 2 + 50 in merged.tolist() and 2 + 50 in merged.tolist()  # nothing bridges the gap between 2 and 10


def test_radial_bins_are_no_finer_than_a_pixel() -> None:
    image = giwaxs_frame()
    maps = giwaxs_maps(image.shape, GEOMETRY)
    assert 0 < maps.q_step < 0.01
    radial = reduce_giwaxs(image, np.ones(image.shape, dtype=bool), GEOMETRY, GiwaxsSettings(), maps=maps).curve("radial")
    assert float(np.median(np.diff(radial.x))) >= 0.95 * maps.q_step
    assert int(radial.pixels.min()) >= 8  # no point from a handful of pixels
    chi = reduce_giwaxs(image, np.ones(image.shape, dtype=bool), GEOMETRY, GiwaxsSettings(), maps=maps).curve("azimuthal")
    assert np.allclose((chi.x + 90.0) % 1.0, 0.5)  # I(χ) keeps its 1° bins (the orientation analysis relies on them)


def test_a_defective_row_is_left_out_but_real_rods_and_bands_are_kept() -> None:
    rng = np.random.default_rng(4)
    data = rng.poisson(0.3, (300, 400)).astype(np.float32)  # sparse counts, like the high-q part of a Lambda frame
    columns = np.array([20, 21, 60, 95, 96, 140, 180, 181, 230, 260, 300, 301, 350])
    data[120, columns] = rng.integers(10_000, 400_000, columns.size)  # scattered, singly and in pairs
    data[:, 200] += 400.0  # a real one-pixel-wide rod: continuous, so not a defect
    data[40:43, :] += 300.0  # a band three rows high (a Yoneda-like maximum)
    bad = find_bad_pixels(data, np.ones(data.shape, dtype=bool))
    assert bad.line_count >= 6  # the pairs, which the isolated-pixel test cannot see; the single ones are hot pixels
    assert not bad.mask[:, 200].any() and not bad.mask[40:43, :].any()
    assert bad.mask[120, columns].all()


def test_errors_of_a_frame_that_is_not_photon_counts_come_from_the_pixel_scatter() -> None:
    rng = np.random.default_rng(7)
    image = giwaxs_frame().astype(np.float64)
    maps = giwaxs_maps(image.shape, GEOMETRY)
    valid = np.ones(image.shape, dtype=bool)
    noisy = image + rng.normal(0.0, 25.0, image.shape)  # read noise of a dark-subtracted flat panel
    poisson = reduce_giwaxs(noisy, valid, GEOMETRY, GiwaxsSettings(), maps=maps).curve("radial")
    scatter = reduce_giwaxs(noisy, valid, GEOMETRY, GiwaxsSettings(), maps=maps, counts=False).curve("radial")
    assert np.array_equal(poisson.intensity, scatter.intensity)  # the means do not change, only their errors
    expected = 25.0 / np.sqrt(scatter.pixels)
    flat = scatter.x > 0.8 * scatter.x.max()  # far from the rings the pattern is flat within a bin
    assert np.median(scatter.sigma[flat] / expected[flat]) == pytest.approx(1.0, rel=0.25)
    assert np.median(poisson.sigma[flat] / expected[flat]) < 0.5  # √sum/n does not see read noise
    counts = reduce_giwaxs(image, valid, GEOMETRY, GiwaxsSettings(), maps=maps).curve("radial")
    assert np.allclose(counts.sigma, np.sqrt(np.clip(counts.intensity * counts.pixels, 0, None)) / counts.pixels)


def test_a_native_gisaxs_cut_takes_its_errors_from_the_band() -> None:
    from src.gimap.features.analyze.domain.binning import native_profile

    rng = np.random.default_rng(3)
    band = 100.0 + rng.normal(0.0, 10.0, (40, 500))  # 40 rows of one cut, read noise 10
    x = np.linspace(0.0, 0.2, 500)
    _x, mean, sigma, pixels = native_profile(band, np.ones(band.shape, dtype=bool), x, axis=0, counts_model=False)
    assert np.all(pixels == 40)
    assert np.median(sigma) == pytest.approx(10.0 / np.sqrt(40), rel=0.1)
    poisson = native_profile(band, np.ones(band.shape, dtype=bool), x, axis=0)[2]
    assert np.median(poisson) == pytest.approx(np.sqrt(100.0 * 40) / 40, rel=0.05)


def test_merged_bins_combine_errors_of_either_kind() -> None:
    x = np.arange(4, dtype=float)
    mean = np.array([4.0, 6.0, 5.0, 5.0])
    pixels = np.array([2, 3, 50, 50])
    sigma = np.array([1.0, 2.0, 0.1, 0.1])
    _x, _m, merged_sigma, merged_pixels = merge_sparse_bins(x, mean, sigma, pixels, min_pixels=5)
    assert merged_pixels[0] == 5
    assert merged_sigma[0] == pytest.approx(np.sqrt(2**2 * 1.0**2 + 3**2 * 2.0**2) / 5)


def test_no_line_is_drawn_across_bins_without_pixels() -> None:
    from src.gimap.shared.figures import break_at_gaps

    chi = np.concatenate([np.arange(-30.0, -24.0), np.arange(25.0, 36.0)])  # I(χ) with a masked range in between
    x, y = break_at_gaps(chi, np.ones(chi.size))
    assert np.isnan(x).sum() == 1 and np.isnan(x[6]) and np.array_equal(x[~np.isnan(x)], chi)
    merged = np.array([0.0, 3.0, 6.0, 9.0, 10.0, 11.0, 12.0, 13.0])  # merged sparse bins: wider, but no gap
    assert not np.isnan(break_at_gaps(merged, np.ones(merged.size))[0]).any()


def test_log_axis_labels_do_not_pile_up_on_a_short_plot() -> None:
    from src.gimap.app.presentation.components.curve_plot import readable_log_ticks

    def minor(low, high, size):
        return [value for spacing, value in readable_log_ticks(low, high, size, []) if spacing is None][0]

    short = np.round(10 ** np.array(minor(-1.2, 0.9, 150)), 6)  # 0.06 … 8 on 150 px: 1-2-5 at most
    assert set(np.round(short / 10 ** np.floor(np.log10(short)), 6)) <= {1.0, 2.0, 5.0}
    assert len(minor(0.0, 0.5, 400)) >= 2  # within one decade there are still labels
    assert len(minor(0.0, 1.0, 2000)) == 8  # 2 … 9 when there is room


def test_a_log_plot_is_not_stretched_by_a_few_near_zero_bins() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    plot = CurvePlot("", None, log_y=True)
    x = np.linspace(0.1, 2.0, 400)
    y = 1000.0 * np.exp(-x)  # 1000 … 135
    y[50] = 1e-6  # one bin of dark-subtracted noise
    plot.set_curves([("I(q)", x, y)])
    low, high = plot.plot.getViewBox().viewRange()[1]
    assert low > 1.5 and high >= 2.99  # log10: the range follows the curve (≈ 2.1 … 3.0), not the dip at −6
    plot.dispose()
