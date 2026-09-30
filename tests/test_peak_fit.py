"""Peaks fitted in a window: the right centre, width and area for each shape, no peak where there is
none, and a drifting peak followed frame after frame from the previous result."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.analyze.application import FitStarts
from src.gimap.features.analyze.domain.peak_fit import fit_peak, shape


@pytest.mark.parametrize("profile, eta", [("gaussian", 0.0), ("lorentzian", 1.0), ("pseudo_voigt", 0.4)])
def test_the_peak_parameters_come_back(profile: str, eta: float) -> None:
    rng = np.random.default_rng(3)
    x = np.linspace(0.9, 1.1, 150)
    truth = 5.0 * shape(x, 1.003, 0.012, eta, profile) + 20 + 30 * (x - 1.0)
    sigma = 0.3 * np.sqrt(truth)
    fit = fit_peak(x, truth + rng.normal(0, sigma), sigma, (0.95, 1.05), profile=profile)
    assert fit.ok and fit.center == pytest.approx(1.003, abs=3 * fit.center_err + 1e-4)
    assert fit.fwhm == pytest.approx(0.012, rel=0.05) and fit.area == pytest.approx(5.0, rel=0.05)
    assert 0.5 < fit.chi2_red < 2.0 and fit.x.size == fit.y.size == fit.fitted.size


def test_no_peak_is_reported_where_there_is_none() -> None:
    x = np.linspace(0.9, 1.1, 120)
    flat = 20 + np.random.default_rng(1).normal(0, 1, x.size)
    fit = fit_peak(x, flat, np.ones_like(x), (0.95, 1.05))
    assert not fit.ok and "no significant peak" in fit.message
    assert not fit_peak(x[:5], flat[:5], None, (0.9, 1.0)).ok  # too few points


def test_a_drifting_peak_is_followed_from_the_previous_frame() -> None:
    rng = np.random.default_rng(5)
    x = np.linspace(0.9, 1.1, 150)
    starts, centres = FitStarts("previous"), []
    for frame in range(8):
        centre = 0.99 + 0.004 * frame  # the peak moves by a third of its width per frame
        y = 4.0 * shape(x, centre, 0.012, 0.3, "pseudo_voigt") + 15 + rng.normal(0, 0.8, x.size)
        fit = fit_peak(x, y, np.full(x.size, 0.8), (0.95, 1.05), start=starts.for_key("ring"))
        assert fit.ok
        starts.remember("ring", fit.as_start())
        centres.append(fit.center)
    assert np.allclose(centres, 0.99 + 0.004 * np.arange(8), atol=0.0015)
    assert FitStarts("first").for_key("ring") is None


def test_frames_are_written_in_other_formats(tmp_path) -> None:
    from src.gimap.shared.detector_io import write_frame

    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(np.load(write_frame(tmp_path / "a.npy", data, "npy")), data)
    import fabio

    for fmt, suffix in (("tiff", ".tif"), ("edf", ".edf")):
        path = write_frame(tmp_path / f"a{suffix}", data, fmt, {"source_file": "x.nxs"})
        assert np.allclose(fabio.open(str(path)).data, data)
    with pytest.raises(ValueError):
        write_frame(tmp_path / "a.bmp", data, "bmp")
