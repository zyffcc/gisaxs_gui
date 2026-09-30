"""GIWAXS metrics of the assistant on synthetic profiles with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gimap.features.assistant.domain import (
    Profile,
    compare_sectors,
    find_peaks,
    ring_orientation,
    scherrer_size,
)

PEAKS = ((0.36, 0.020, 60.0), (0.72, 0.024, 20.0), (1.08, 0.030, 9.0), (1.65, 0.12, 25.0))


def _gaussian(q, center, fwhm, height):
    return height * np.exp(-0.5 * ((q - center) / (fwhm / 2.3548200450309493)) ** 2)


def _profile(peaks=PEAKS, *, pixels=400.0, seed=1, q=None) -> Profile:
    rng = np.random.default_rng(seed)
    q = np.linspace(0.1, 2.2, 1400) if q is None else q
    truth = 4.0 / q ** 1.5 + 3.0
    for center, fwhm, height in peaks:
        truth = truth + _gaussian(q, center, fwhm, height)
    counts = rng.poisson(truth * pixels) / pixels
    sigma = np.sqrt(np.clip(truth, 1e-9, None) / pixels)
    return Profile.of(q, counts, sigma, np.full_like(q, pixels))


def test_peaks_have_the_right_positions_widths_and_d_spacings() -> None:
    search = find_peaks(_profile())

    assert search.reason == ""
    found = {round(peak.q, 2): peak for peak in search.peaks}
    for center, fwhm, _height in PEAKS:
        peak = min(search.peaks, key=lambda item: abs(item.q - center))
        assert abs(peak.q - center) / center < 0.005, (center, peak.q)
        assert abs(peak.fwhm - fwhm) / fwhm < 0.15, (center, peak.fwhm)
        assert peak.d == pytest.approx(2 * math.pi / peak.q)
        assert peak.q_err > 0 and peak.snr > 5
    assert len(found) == len(PEAKS)
    assert any("lamellar" in hint and "0.36" in hint for hint in search.series)


def test_background_only_explains_why_there_is_no_peak() -> None:
    search = find_peaks(_profile(peaks=()))

    assert search.peaks == ()
    assert search.reason.startswith("No peak rises 3σ and 2% above the background")
    assert "strongest feature" in search.reason


def test_too_few_bins_are_reported() -> None:
    search = find_peaks(_profile(q=np.linspace(0.3, 0.4, 8)))

    assert search.peaks == ()
    assert "Only 8 measured bins" in search.reason


def test_sectors_tell_out_of_plane_from_in_plane_peaks() -> None:
    lamellar = _profile(peaks=PEAKS[:3], seed=2)
    pi_stack = _profile(peaks=PEAKS[3:], seed=3)
    pairs = [(center, fwhm) for center, fwhm, _height in PEAKS]

    rows = compare_sectors(pairs, in_plane=pi_stack, out_of_plane=lamellar, background_window=0.15)

    assert [row.preference for row in rows[:3]] == ["out-of-plane only"] * 3
    assert rows[3].preference == "in-plane only"


def _ring(shape, *, missing_below=0.0, height=40.0, background=10.0, pixels=300.0, seed=4) -> Profile:
    rng = np.random.default_rng(seed)
    chi = np.arange(-89.5, 90.0, 1.0)
    truth = background + height * shape(np.abs(chi))
    counts = rng.poisson(truth * pixels) / pixels
    measured = np.abs(chi) >= missing_below
    return Profile.of(chi, counts, np.sqrt(truth / pixels), np.where(measured, pixels, 0.0))


def test_isotropic_ring_has_herman_factor_near_zero() -> None:
    result = ring_orientation(_ring(lambda chi: np.ones_like(chi)), q_window=(0.35, 0.37), background=10.0)

    assert result.reason == ""
    assert abs(result.herman) < 0.05
    assert result.texture.startswith("isotropic")
    assert result.coverage == pytest.approx(1.0)


def test_out_of_plane_ring_with_missing_wedge() -> None:
    shape = lambda chi: np.exp(-0.5 * (chi / 12.0) ** 2)  # noqa: E731
    result = ring_orientation(
        _ring(shape, missing_below=4.0), q_window=(0.35, 0.37), background=10.0
    )

    assert result.herman > 0.8 and result.herman_err is not None
    assert result.texture.startswith("oriented along the surface normal")
    assert result.missing[0] == (0.0, 4.0)
    assert any("missing wedge" in note for note in result.notes)
    assert result.maxima and result.maxima[0]["chi"] < 5.0


def test_in_plane_ring_and_a_ring_below_the_noise() -> None:
    in_plane = ring_orientation(
        _ring(lambda chi: np.exp(-0.5 * ((chi - 90.0) / 10.0) ** 2)), q_window=(1.6, 1.7), background=10.0
    )
    faint = ring_orientation(
        _ring(lambda chi: np.ones_like(chi), height=0.0), q_window=(1.6, 1.7), background=10.0
    )

    assert in_plane.herman < -0.35
    assert in_plane.texture.startswith("oriented in the sample plane")
    assert faint.herman is None and faint.reason.startswith("The ring is not above the radial background")


def test_scherrer_size_and_its_limits() -> None:
    plain = scherrer_size(0.36, 0.02, 0.001)
    corrected = scherrer_size(0.36, 0.02, instrumental_fwhm=0.01)
    unresolved = scherrer_size(0.36, 0.02, instrumental_fwhm=0.03)
    binned = scherrer_size(0.36, 0.004, bin_width=0.0015)

    assert plain.size == pytest.approx(2 * math.pi * 0.9 / 0.02)
    assert plain.size_err == pytest.approx(plain.size * 0.05)
    assert plain.lower_bound and "lower bound" in plain.notes[-1]
    assert corrected.size == pytest.approx(2 * math.pi * 0.9 / math.sqrt(0.02 ** 2 - 0.01 ** 2))
    assert not corrected.lower_bound
    assert unresolved.size is None and "not broader" in unresolved.reason
    assert binned.lower_bound and "radial bins" in binned.notes[-1]
