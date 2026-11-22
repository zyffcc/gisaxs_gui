"""Invented raw, dark, background, flat and uncertainty frames for Analyze."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.application import AnalyzeFrame, Corrections
from src.gimap.features.analyze.domain import Normalisation
from src.gimap.shared.detector_io.models import DetectorImage


class References:
    def __init__(self, folder):
        self.folder = folder
        self.images = {}

    def add(self, name, data, exposure=None):
        path = self.folder / (name + ".edf")
        path.touch()
        data = np.asarray(data)
        self.images[path] = DetectorImage(data, None, path, metadata={
            "stored_dtype": data.dtype.str, "exposure_time_s": exposure,
        })
        return str(path)

    def load(self, path, index):
        assert index == 0
        return self.images[Path(path)]


def correct(analyze, sample, options, summed=0):
    sample = np.asarray(sample)
    common = {"metadata": {"stored_dtype": sample.dtype.str}, "summed_frames": (None,) * summed}
    data = sample.astype(np.float32)
    valid = np.ones(sample.shape, dtype=bool)
    counts = analyze.correct_pixels(common, data, valid, options, None, None)
    if not options.normalisation.is_identity:
        analyze._normalise(common, data, valid, options, counts, "sample")
    return common


def test_dark_scales_follow_the_reference_headers_and_record_their_source(tmp_path):
    refs = References(tmp_path)
    dark = refs.add("dark", [[7, 8]], 2)
    raw_background = refs.add("background", [[25, 40]], 6)
    raw_flat = refs.add("response", [[100, 180]], 4)
    opts = Corrections(bad_pixels=False, dark_path=dark, dark_scale=1.5, background_path=raw_background,
                       background_scale=0.5, flat_field_path=raw_flat)
    analyze = AnalyzeFrame(refs, None)
    sample = np.array([[65, 83]])
    actual = correct(analyze, sample, opts)
    k = np.array([[7, 8]])
    signal = sample - 1.5 * k - 0.5 * (np.array([[25, 40]]) - 3 * k)
    flat = np.array([[100, 180]]) - 2 * k
    np.testing.assert_allclose(actual["data"], signal * flat.mean() / flat)
    assert actual["reference_dark"]["background_dark_scale"] == 3
    assert actual["reference_dark"]["source"] == "exposure times"
    assert actual["reference_dark"]["background_exposure_time_s"] == 6
    assert actual["reference_dark"]["dark_exposure_time_s"] == 2
    assert actual["flat_field"].dark_scale == 2
    assert actual["flat_field"].dark_scale_source == "exposure times"
    assert actual["flat_field"].to_record()["exposure_time_s"] == 4
    typed = correct(analyze, sample, replace(opts, background_dark_scale=0.25))
    assert typed["reference_dark"]["background_dark_scale"] == 0.25
    assert typed["reference_dark"]["source"] == "entered"
    prepared = correct(analyze, sample, replace(opts, background_already_dark_subtracted=True,
                                                flat_already_dark_subtracted=True))
    prepared_signal = sample - 1.5 * k - 0.5 * np.array([[25, 40]])
    np.testing.assert_allclose(prepared["data"], prepared_signal * 140 / [[100, 180]])
    assert prepared["flat_field"].dark_scale == prepared["reference_dark"]["background_dark_scale"] == 0
    assert prepared["flat_field"].to_record()["already_dark_subtracted"]
    assert prepared["reference_dark"]["background_already_dark_subtracted"]
    missing_header = refs.add("untimed", [[7, 8]])
    fallback = correct(analyze, sample, replace(opts, dark_path=missing_header))
    assert fallback["reference_dark"]["background_dark_scale"] == fallback["flat_field"].dark_scale == 1.5
    assert fallback["reference_dark"]["source"] == fallback["flat_field"].dark_scale_source == "sample dark scale"


def test_no_dark_gives_identical_background_subtraction_and_retains_negative_values(tmp_path):
    refs = References(tmp_path)
    rng = np.random.default_rng(22)
    sample = rng.normal(20, 10, (11, 14)).astype("f4")
    background = rng.uniform(9, 42, sample.shape).astype("f4")
    options = Corrections(bad_pixels=False, background_path=refs.add("background", background, 5),
                          background_scale=1.3, background_dark_scale=8.5)
    analyze = AnalyzeFrame(refs, None)
    expected = sample - np.float32(1.3) * background
    for prepared in (False, True):
        actual = correct(analyze, sample, replace(options, background_already_dark_subtracted=prepared))
        np.testing.assert_array_equal(actual["data"], expected)
        assert actual["valid"].all() and (actual["data"] < 0).any()


def test_changing_floor_recomputes_retained_pixels_and_mean(tmp_path):
    refs = References(tmp_path)
    opts = Corrections(bad_pixels=False, flat_field_path=refs.add("flat", [[1, 10, 39]]))
    analyze = AnalyzeFrame(refs, None)
    initial = correct(analyze, [[12, 12, 12]], opts)
    strict = correct(analyze, [[12, 12, 12]], replace(opts, flat_field_floor=0.2))
    assert initial["valid"].tolist() == [[True, True, True]]
    assert strict["valid"].tolist() == [[False, True, True]]
    assert strict["flat_field"].mean == 24.5


def test_replacing_dark_at_the_same_path_updates_the_background_and_flat(tmp_path):
    refs = References(tmp_path)
    dark = refs.add("dark", [[5, 10]])
    opts = Corrections(bad_pixels=False, dark_path=dark, background_scale=0.3,
                       background_path=refs.add("background", [[25, 50]]),
                       flat_field_path=refs.add("flat", [[90, 110]]))
    analyze = AnalyzeFrame(refs, None)
    first = correct(analyze, [[110, 99]], opts)
    refs.images[Path(dark)].data = np.array([[3, 5]])
    Path(dark).write_bytes(b"updated synthetic reference")  # Different file size even on coarse clocks.
    second = correct(analyze, [[110, 99]], opts)
    assert first["flat_field"].mean == 92.5
    assert second["flat_field"].mean == 96
    signal = np.array([[107, 94]]) - 0.3 * np.array([[22, 45]])
    np.testing.assert_allclose(second["data"], signal * 96 / [[87, 105]])


def test_pixel_variance_matches_independent_repetitions_with_a_shared_dark(tmp_path):
    rng = np.random.default_rng(812)
    # Each pixel is a new replicate with independent raw measurements of a uniform source.
    # Within a replicate the *same* measured K corrects the sample, background and flat.
    shape = (320, 320)
    refs = References(tmp_path)
    sample = rng.poisson(2600, shape)
    k = rng.poisson(300, shape)
    b = rng.poisson(600, shape)
    dff = rng.poisson(3300, shape)
    opts = Corrections(bad_pixels=False, dark_scale=2, dark_path=refs.add("dark", k, 1),
                       background_scale=0.5, background_path=refs.add("background", b, 2),
                       flat_field_path=refs.add("flat", dff, 3), error_model="poisson")
    result = correct(AnalyzeFrame(refs, None), sample, opts)
    assert result["error_model_used"] == "poisson"
    # Partial derivatives evaluated at the true means (m=2400, J=2000, F=1).
    expected = 2600 + 0.5**2 * 600 + (2000 / 2400)**2 * 3300 + (-1 + 3 * 2000 / 2400)**2 * 300
    measured = np.var(result["data"], ddof=1, dtype=float)
    propagated = np.mean(result["pixel_variance"] + result["extra_variance"], dtype=float)
    assert measured == pytest.approx(expected, rel=0.025)
    assert propagated == pytest.approx(measured, rel=0.025)


def test_gain_offset_supplied_errors_and_prepared_references(tmp_path):
    refs = References(tmp_path)
    sigma = np.array([[1, 2], [3, -1]], dtype="f4")
    np.save(tmp_path / "sigma.npy", sigma)
    opts = Corrections(bad_pixels=False, counts_per_adu=2, offset_adu=100, error_model="sigma",
                       sigma_path=str(tmp_path / "sigma.npy"), background_already_dark_subtracted=True,
                       background_path=refs.add("prepared", [[3, 5], [7, 6]]),
                       flat_already_dark_subtracted=True, flat_field_path=refs.add("flat", [[3, 7], [5, 5]]),
                       normalisation=Normalisation(exposure="value", exposure_s=2))
    analyze = AnalyzeFrame(refs, None)
    sample = np.full((2, 2), 214, dtype=np.uint16)
    result = correct(analyze, sample, opts, summed=1)
    factor = np.array([[3, 7], [5, 5]]) / 5
    j = 28 - 2 * np.array([[3, 5], [7, 6]])
    valid = result["valid"]
    assert result["error_model_used"] == "sigma"
    np.testing.assert_array_equal(valid, sigma >= 0)
    np.testing.assert_allclose(result["data"][valid], (0.5 * j / factor)[valid])
    np.testing.assert_allclose(result["pixel_variance"][valid], ((sigma / factor)**2)[valid], rtol=1e-6)
    counts = correct(analyze, sample, replace(opts, error_model="poisson"), summed=1)
    np.testing.assert_allclose(counts["pixel_variance"], 0.25 * 28 / factor**2, rtol=1e-6)
    scatter = correct(analyze, sample, replace(opts, error_model="scatter"), summed=1)
    assert scatter["error_model_used"] == "scatter" and "pixel_variance" not in scatter
    with pytest.raises(ValueError, match="Choose a sigma frame"):
        correct(analyze, sample, replace(opts, sigma_path=None))
    np.save(tmp_path / "small.npy", [[3]])
    with pytest.raises(ValueError, match="same shape"):
        correct(analyze, sample, replace(opts, sigma_path=str(tmp_path / "small.npy")))
    invalid = refs.add("invalid_sigma", [[0, np.nan], [np.inf, -1]])
    marked = correct(analyze, sample, replace(opts, sigma_path=invalid), summed=1)
    np.testing.assert_array_equal(marked["valid"], [[True, False], [False, False]])
