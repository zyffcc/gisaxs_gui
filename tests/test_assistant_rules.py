"""Judgement the code makes so an agent does not have to: found on real GIWAXS data.

Each rule comes from a case where a plausible reading was wrong: a detector
spike taken for a peak, a halo given a crystallite size, a hexagonal series
built from artefacts, a flat halo called oriented, a calibration rejected for
its pixel residual although its lines sit where they should.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gimap.features.assistant.domain import (
    Profile,
    assessment,
    caveat,
    compare_sectors,
    compare_verdict,
    energy_from_notes,
    find_peaks,
    incidence_from_notes,
    module_series,
    name_preference,
    ring_orientation,
    scherrer_size,
    standard_from_name,
)
from src.gimap.features.calibration.domain import STANDARDS, best_by_lines, check_lines, detect_standard_keys


def _gaussian(q, center, fwhm, height):
    return height * np.exp(-0.5 * ((q - center) / (fwhm / 2.3548200450309493)) ** 2)


def _profile(peaks, *, spike_at=None, pixels=400.0, seed=1) -> Profile:
    rng = np.random.default_rng(seed)
    q = np.linspace(0.1, 2.6, 2000)
    truth = 4.0 / q**1.5 + 3.0
    for center, fwhm, height in peaks:
        truth = truth + _gaussian(q, center, fwhm, height)
    counts = rng.poisson(truth * pixels) / pixels
    if spike_at is not None:
        index = int(np.argmin(np.abs(q - spike_at)))
        counts[index : index + 2] += 300.0  # hot pixels on two bins
    sigma = np.sqrt(np.clip(truth, 1e-9, None) / pixels)
    return Profile.of(q, counts, sigma, np.full_like(q, pixels))


def test_a_spike_is_flagged_and_completes_no_series() -> None:
    # A two-bin spike at 2.16 sits exactly where the third lamellar order of 0.72 would be.
    lamellar = ((0.72, 0.024, 20.0), (1.44, 0.03, 8.0))
    search = find_peaks(_profile(lamellar, spike_at=2.16))
    spike = min(search.peaks, key=lambda peak: abs(peak.q - 2.16))
    assert abs(spike.q - 2.16) < 0.01 and "spike" in spike.flags
    assert caveat(spike).startswith("a spike")
    assert not search.series, search.series
    real = find_peaks(_profile((*lamellar, (2.16, 0.035, 5.0))))
    assert any(hint.startswith("q = 0.72") and "lamellar" in hint for hint in real.series), real.series


def test_a_halo_is_flagged_and_a_sharp_peak_on_it_is_kept() -> None:
    halo = find_peaks(_profile(((0.36, 0.02, 60.0), (1.9, 0.45, 12.0))))
    broad = [peak for peak in halo.peaks if "broad" in peak.flags]
    assert len(broad) == 1 and "halo" in caveat(broad[0])
    # Semicrystalline polymers: crystalline reflections sit on the amorphous halo.
    both = find_peaks(_profile(((0.36, 0.02, 60.0), (2.2, 0.4, 12.0), (2.1, 0.02, 20.0))))
    sharp = [peak for peak in both.peaks if abs(peak.q - 2.1) < 0.01 and peak.fwhm < 0.05]
    assert len(sharp) == 1 and not caveat(sharp[0])
    assert any(abs(peak.q - 2.2) < 0.05 for peak in both.peaks)  # the halo is still reported


def test_a_halo_gets_no_crystallite_size() -> None:
    halo = scherrer_size(1.9, 0.42, 0.03)
    assert halo.size is None and "halo" in halo.reason and "correlation length" in halo.reason
    crystal = scherrer_size(2.94, 0.08, 0.003)
    assert crystal.size == pytest.approx(2 * math.pi * 0.9 / 0.08) and crystal.reason == ""


def _ring(shape, *, pixels=300.0, seed=4) -> Profile:
    rng = np.random.default_rng(seed)
    chi = np.arange(-89.5, 90.0, 1.0)
    truth = 10.0 + 40.0 * shape(np.abs(chi))
    counts = rng.poisson(truth * pixels) / pixels
    return Profile.of(chi, counts, np.sqrt(truth / pixels), np.full_like(chi, pixels))


def test_rival_maxima_or_f_near_zero_are_not_called_an_orientation() -> None:
    def two_maxima(chi):
        return 0.5 + np.exp(-0.5 * ((chi - 12.0) / 6.0) ** 2) + np.exp(-0.5 * ((chi - 72.0) / 6.0) ** 2)

    rival = ring_orientation(_ring(two_maxima), q_window=(1.6, 1.8), background=10.0)
    assert rival.texture.startswith("no single preferred orientation"), rival.texture

    def weak_bump(chi):
        return 1.0 + 0.2 * np.exp(-0.5 * ((chi - 4.0) / 6.0) ** 2)

    weak = ring_orientation(_ring(weak_bump), q_window=(1.6, 1.8), background=10.0)
    assert abs(weak.herman) < 0.1
    assert weak.texture.startswith("weak or no preferred orientation"), weak.texture

    def normal(chi):
        return np.exp(-0.5 * (chi / 12.0) ** 2)

    oriented = ring_orientation(_ring(normal), q_window=(0.35, 0.37), background=10.0)
    assert oriented.herman > 0.5 and oriented.texture.startswith("oriented along the surface normal")


def test_a_shadowed_part_of_the_ring_is_unmeasured_and_f_is_not_faked() -> None:
    # The real P03 case: |χ| > 57° is 20–150× below the diffuse background at every q.
    rng = np.random.default_rng(6)
    chi = np.arange(-89.5, 90.0, 1.0)
    truth = np.where(np.abs(chi) < 57.0, 300.0 + 1500.0, 6.0)  # an isotropic ring over a background of 300
    counts = rng.poisson(truth * 300.0) / 300.0
    profile = Profile.of(chi, counts, np.sqrt(truth / 300.0), np.full_like(chi, 300.0))
    shadowed = ring_orientation(profile, q_window=(2.88, 3.0), background=300.0)
    assert shadowed.herman is None and "shadowed" in " ".join(shadowed.notes)
    assert shadowed.weighted_coverage < 0.8 and "sin χ" in shadowed.reason
    assert shadowed.shadowed == ((56.0, 90.0),)  # with its edge bin: what the q map draws in orange
    # Without the rule the dark side would look like orientation along the normal (f ≈ 0.4).

    wedge = ring_orientation(_ring(lambda chi: np.ones_like(chi)), q_window=(1.6, 1.8), background=10.0)
    assert wedge.herman is not None and wedge.weighted_coverage == pytest.approx(1.0)
    missing = _ring(lambda chi: np.ones_like(chi))
    missing = Profile.of(missing.x, missing.y, missing.sigma, np.where(np.abs(missing.x) < 13.0, 0.0, 300.0))
    low = ring_orientation(missing, q_window=(1.6, 1.8), background=10.0)
    assert low.herman is not None and low.weighted_coverage > 0.95  # a missing wedge hardly matters to f
    assert low.herman_isotropic == pytest.approx(-0.026, abs=0.01) and any("random" in note for note in low.notes)


def test_a_shadowed_sector_is_not_compared() -> None:
    q = np.linspace(1.5, 2.3, 400)

    def sector(background, height, seed):
        truth = background + height * np.exp(-0.5 * ((q - 1.93) / 0.17) ** 2)
        counts = np.random.default_rng(seed).poisson(truth * 500.0) / 500.0
        return Profile.of(q, counts, np.sqrt(truth / 500.0), np.full_like(q, 500.0))

    rows = compare_sectors(
        [(1.93, 0.4)], sector(15.0, 16.0, 1), sector(900.0, 650.0, 2),
        background_window=0.15, reference_background=lambda _q: 950.0,
    )
    assert rows[0].ratio is None
    assert rows[0].preference.startswith("only the out-of-plane sector is usable here: the other is shadowed")
    assert "in-plane sector is shadowed" in rows[0].note
    unshadowed = compare_sectors(
        [(1.93, 0.4)], sector(15.0, 16.0, 1), sector(900.0, 650.0, 2), background_window=0.15,
    )
    assert unshadowed[0].ratio is not None  # without the whole ring's background nothing is judged


def test_notes_give_alpha_i_and_the_energy_only_when_unambiguous() -> None:
    assert incidence_from_notes("GIWAXS at P03, alpha_i = 0.4 deg, 11.8 keV") == pytest.approx(0.4)
    assert incidence_from_notes("αi: 0.15°") == pytest.approx(0.15)
    assert incidence_from_notes("入射角 0.3°") == pytest.approx(0.3)
    assert incidence_from_notes("αi 0.4° for PEO, αi 0.2° for the blend") is None  # two values: ask
    assert incidence_from_notes("sample 117, 20 pulses") is None
    assert energy_from_notes("E = 11.8 keV") == pytest.approx(11.8)
    assert energy_from_notes("11.8 keV and later 12.4 keV") is None


def _check(result: dict, relative: float, lines: int = 8) -> dict:
    return {**result, "line_check": {"lines_checked": lines, "mean_relative": relative}}


def test_the_lines_of_the_standard_decide_not_the_pixel_residual() -> None:
    # The real P03 case: the mixture fit with 9 rings has rms 4.2 px but puts its lines 0.06 % off.
    fitted = _check({"standard": "lab6_ceo2", "matched_rings": 9, "rms_residual_px": 4.2, "confidence": "Medium"}, 0.00056)
    assert assessment(fitted).startswith("good")
    assert assessment({"matched_rings": 9, "rms_residual_px": 4.2, "confidence": "Medium"}).startswith("doubtful")
    lab6 = _check({"standard": "lab6", "matched_rings": 7, "rms_residual_px": 1.9}, 0.0027)
    assert assessment(lab6).startswith("usable")
    assert compare_verdict([fitted, lab6]).startswith("clear: lab6_ceo2")
    close = _check({"standard": "ceo2", "matched_rings": 6, "rms_residual_px": 2.0}, 0.0007)
    assert compare_verdict([fitted, close]).startswith("ambiguous")
    assert compare_verdict([_check(fitted, 0.02)]).startswith("none fits well")


def test_check_lines_measures_where_a_standards_lines_land() -> None:
    shape, pixel, distance_mm, center, wavelength = (500, 500), 172e-6, 200.0, (250.0, 250.0), 1.0
    q_lines = (0.6, 0.9, 1.2)
    rows, columns = np.indices(shape)
    radius = np.hypot((columns + 0.5 - center[0]) * pixel, (rows + 0.5 - center[1]) * pixel)
    q = 4 * math.pi / wavelength * np.sin(0.5 * np.arctan2(radius, distance_mm * 1e-3))
    image = 20.0 + sum(300.0 * np.exp(-0.5 * ((q - line) / 0.003) ** 2) for line in q_lines)
    image = np.random.default_rng(2).poisson(image).astype(float)
    valid = np.ones(shape, dtype=bool)

    def run(distance):
        return check_lines(
            image, valid, center_x_px=center[0], center_y_px=center[1], distance_mm=distance,
            pixel_size_x_m=pixel, pixel_size_y_m=pixel, wavelength_angstrom=wavelength, q_lines=q_lines,
        )

    right, wrong = run(distance_mm), run(distance_mm * 1.01)
    assert right["lines_checked"] == 3 and right["mean_relative"] < 5e-4
    assert wrong["mean_relative"] > 5e-3
    assert best_by_lines([wrong, right]) == 1
    assert best_by_lines([{"lines_checked": 2, "mean_relative": 1e-5}, None]) is None


def test_a_lab6_ceo2_mixture_and_nexus_module_series_are_recognised() -> None:
    name = "giwaxs_calib_lab6_ceo2_redone_final_00001_00002_m01.nxs"
    assert standard_from_name(name) == "lab6_ceo2"
    assert detect_standard_keys(name)[0] == "lab6_ceo2"
    mixture = set(STANDARDS["lab6_ceo2"].q_values_inv_angstrom)
    assert set(STANDARDS["lab6"].q_values_inv_angstrom) <= mixture and set(STANDARDS["ceo2"].q_values_inv_angstrom) <= mixture
    assert module_series(name) is not None and module_series("sample_00001.cbf") is None
    assert name_preference("calib_redone_final/x.nxs") > name_preference("calib_redone/x.nxs") > name_preference("calib/x.nxs")
