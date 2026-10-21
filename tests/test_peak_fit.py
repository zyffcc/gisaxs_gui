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


def _counted(monkeypatch) -> list[int]:
    """How often the fit evaluates the peak shape: what its time is made of, whatever the machine."""
    from src.gimap.features.analyze.domain import peak_fit

    calls = [0]

    def counting(*args, **kwargs):
        calls[0] += 1
        return shape(*args, **kwargs)

    monkeypatch.setattr(peak_fit, "shape", counting)
    return calls


@pytest.mark.parametrize("profile, eta", [("gaussian", 0.0), ("lorentzian", 1.0)])
def test_a_pure_shape_is_fitted_as_directly_as_the_pseudo_voigt(profile: str, eta: float, monkeypatch) -> None:
    """η is not a parameter of a Gaussian or a Lorentzian. Fitted as one boxed into ±1e-9 it stalled the
    bounded least squares: thousands of evaluations for a good peak (seconds per frame of a series)."""
    rng = np.random.default_rng(3)
    x = np.linspace(0.9, 1.1, 150)
    truth = 5.0 * shape(x, 1.003, 0.012, eta, profile) + 20 + 30 * (x - 1.0)
    sigma = 0.3 * np.sqrt(truth)
    y = truth + rng.normal(0, sigma)
    calls = _counted(monkeypatch)
    for start in (None, {"center": 1.0031, "fwhm": 0.0118, "area": 10.0, "eta": eta},  # the area of another frame
                  {"center": 1.003, "fwhm": 0.012, "area": 5.0, "eta": 0.4}):  # η of a pseudo-Voigt handed on
        calls[0] = 0
        fit = fit_peak(x, y, sigma, (0.95, 1.05), profile=profile, start=start)
        assert fit.ok, fit.message
        assert calls[0] < 300, (start, calls[0])  # the pseudo-Voigt needs about 50 here; it was 2 500 – 40 000
        assert fit.center == pytest.approx(1.003, abs=1e-3) and fit.fwhm == pytest.approx(0.012, rel=0.05)
        assert fit.area == pytest.approx(5.0, rel=0.05) and 0.5 < fit.chi2_red < 2.0
        assert fit.eta == eta and fit.as_start()["eta"] == eta  # exactly the shape's, not a fitted number
        assert np.isfinite([fit.center_err, fit.fwhm_err, fit.area_err]).all() and fit.area_err > 0
        assert fit.background[0] == pytest.approx(20 + 30 * (fit.center - 1.0), abs=1.0)


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_a_pure_shape_says_no_peak_instead_of_not_converging(profile: str, monkeypatch) -> None:
    x = np.linspace(0.9, 1.1, 120)
    flat = 20 + np.random.default_rng(1).normal(0, 1, x.size)
    calls = _counted(monkeypatch)
    fit = fit_peak(x, flat, np.ones_like(x), (0.95, 1.05), profile=profile)
    assert not fit.ok and "did not converge" not in fit.message, fit.message
    assert "no significant peak" in fit.message or "no peak above" in fit.message or "as wide as" in fit.message
    assert calls[0] < 2000, calls[0]  # it ran to the limit of 140 000 evaluations (about 17 s)


def test_a_gaussian_ring_is_followed_from_the_first_frames_result(monkeypatch) -> None:
    """Batch Export ▸ peak shape Gaussian, start values “the first frame's result”: every frame fits."""
    rng = np.random.default_rng(11)
    x = np.linspace(0.9, 1.1, 150)
    starts, calls, fits = FitStarts("first"), _counted(monkeypatch), []
    for frame in range(6):
        y = (4.0 + 0.5 * frame) * shape(x, 0.997 + 0.0005 * frame, 0.011, 0.0, "gaussian") + 15 + 8 * (x - 1.0)
        fit = fit_peak(x, y + rng.normal(0, 0.8, x.size), np.full(x.size, 0.8), (0.96, 1.04), profile="gaussian",
                       start=starts.for_key("ring"))
        assert fit.ok, (frame, fit.message)
        starts.remember("ring", fit.as_start())
        fits.append(fit)
    assert np.allclose([fit.center for fit in fits], 0.997 + 0.0005 * np.arange(6), atol=5e-4)
    assert np.allclose([fit.area for fit in fits], 4.0 + 0.5 * np.arange(6), rtol=0.08)
    assert calls[0] < 6 * 300, calls[0]


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
