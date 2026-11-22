"""Image evidence that distinguishes harmonic AgBh distance assignments."""

from pathlib import Path

import numpy as np

from src.gimap.features.calibration.domain.candidate_ranker import score_candidate, score_ring_evidence
from src.gimap.features.calibration.domain.engine import _guided_matches, _rings
from src.gimap.features.calibration.domain.geometry_model import energy_to_wavelength, q_to_ring_radius_m, q_to_ring_radius_px
from src.gimap.features.calibration.domain.guided_rings import search_rings
from src.gimap.features.calibration.domain.models import CalibrationCandidate, CalibrationStandard, DetectorImage
from src.gimap.features.calibration.domain.peak_detector import DetectedPeak, detect_radial_peaks
from src.gimap.features.calibration.domain.peak_matcher import generate_distance_candidates
from src.gimap.features.calibration.domain.preprocessing import preprocess_detector_image
from src.gimap.features.calibration.domain.radial_profile import calculate_azimuthal_profile
from src.gimap.features.calibration.domain.standards import STANDARDS
from src.gimap.features.calibration.domain.tilt_fit import refine_ring_geometry


def test_broad_background_hump_cannot_shift_three_harmonics():
    energy, distance, pixel = 18.0, 1650.0, 120e-6
    standard = STANDARDS["agbh"]
    radii = q_to_ring_radius_px(standard.q_values_inv_angstrom[:5], energy, distance, pixel)
    peaks = [DetectedPeak(radius, 2, 1, 5, 0.5) for radius in radii[:3]]
    peaks.append(DetectedPeak(radii[4] * 0.92, 0.09, 0.06, 170, 0.1))
    candidates = generate_distance_candidates(peaks, standard, energy, pixel, 300, 260, 1.0,
                                              (0.9 * distance, 1.1 * distance))
    assert any([ring.theoretical_index for ring in c.matched_rings] == [0, 1, 2]
               and abs(c.distance_mm - distance) < 0.01 for c in candidates)


def test_anisotropic_detector_compares_radial_peak_to_the_measured_arc():
    standard = STANDARDS["agbh"]
    energy, distance, px, py = 24.0, 1200.0, 100e-6, 200e-6
    cx, cy = 181.4, 176.5
    yy, xx = np.indices((367, 373))
    r_m = np.hypot((xx - cx) * px, (yy - cy) * py)
    wavelength = energy_to_wavelength(energy)
    ring_m = float(q_to_ring_radius_m(standard.q_values_inv_angstrom[0], wavelength, distance))
    counts = 12.0 + 900 * np.exp(-0.5 * ((r_m - ring_m) / (1.8 * px))**2)
    # Only the ends of the major axis are on the active detector.
    mask = abs(yy - cy) > np.tan(np.radians(15)) * abs(xx - cx)
    analysis = preprocess_detector_image(DetectorImage(counts, mask, Path("elliptical_arc.npy")))
    profile = calculate_azimuthal_profile(analysis.signal, analysis.valid, cx, cy)
    peak = max(detect_radial_peaks(profile), key=lambda item: item.prominence)
    candidate = CalibrationCandidate("agbh", cx, cy, distance)
    evidence = search_rings(analysis, candidate, standard, wavelength, px, py)
    first = next(item for item in evidence if item.theoretical_index == 0)
    assert first.match is not None
    assert peak.radius_px > first.match.observed_radius_px + 15

    reference_score = score_ring_evidence(candidate, evidence, standard)
    candidate.strong_peak_radii = (peak.radius_px,)
    assert score_ring_evidence(candidate, evidence, standard) == reference_score
    # The radius at the masked minor axis has no supporting sector.
    candidate.strong_peak_radii += (ring_m / py,)
    assert np.isclose(score_ring_evidence(candidate, evidence, standard), reference_score - 1)


def test_confidence_uses_the_larger_of_point_and_ring_radius_rms():
    candidate = CalibrationCandidate("agbh", 120, 140, 1000, matched_ring_count=4,
                                     rms_residual_px=0.25, azimuthal_coverage=0.8)
    assert score_candidate(candidate).confidence == "High"
    candidate.point_rms_px = 3
    assert score_candidate(candidate).confidence == "Medium"


def test_two_confirmed_orders_keep_a_resolved_close_radial_pair():
    # Narrow peaks 12 pixels apart resolve cleanly, but a two-sided guided background does not fit.
    energy, distance, pixel = 18.0, 380.0, 100e-6
    wavelength = energy_to_wavelength(energy)
    radii = np.array([64.0, 116.0, 165.0, 177.0])
    angles = np.arctan(radii * pixel / (distance * 1e-3))
    standard = CalibrationStandard("mixed", "Mixed powder", tuple(4*np.pi/wavelength * np.sin(angles/2)))
    yy, xx = np.indices((421, 419))
    cx, cy = 206.5, 207.0
    r = np.hypot(xx - cx, yy - cy)
    counts = np.full(r.shape, 15.0)
    for radius in radii:
        counts += 650.0 * np.exp(-0.5 * ((r - radius) / 1.1)**2)
    analysis = preprocess_detector_image(DetectorImage(counts, None, Path("close_reflections.npy")))
    peaks, window, _ = _rings(analysis, cx, cy)
    candidates = generate_distance_candidates(peaks, standard, energy, pixel, cx, cy, 1.0,
                                              (0.98*distance, 1.02*distance), window)
    candidate = candidates[0]
    np.testing.assert_allclose([m.observed_radius_px for m in candidate.matched_rings], radii, atol=1)
    evidence = search_rings(analysis, candidate, standard, wavelength, pixel, pixel)
    assert [e.theoretical_index for e in evidence if e.match] == [0, 1]
    assert [e.theoretical_index for e in evidence if e.resolved_radial_match] == [2, 3]
    _guided_matches(candidate, evidence)
    assert [m.theoretical_index for m in candidate.matched_rings] == [0, 1, 2, 3]
    # Neither close line is marked missing, and each earns the half-ring radial-only weight.
    with_pair = score_ring_evidence(candidate, evidence, standard)
    candidate.matched_rings = candidate.matched_rings[:2]
    # Omitting both lines also leaves two strong radial peaks unexplained.
    assert np.isclose(with_pair - score_ring_evidence(candidate, evidence, standard), 3)


def test_absent_intermediate_orders_resolve_half_distance_alias():
    standard = STANDARDS["agbh"]
    energy, distance, pixel = 24.0, 1050.0, 100e-6
    wavelength = energy_to_wavelength(energy)
    radii = q_to_ring_radius_px(standard.q_values_inv_angstrom[:3], energy, distance, pixel)
    cy, cx = 306.7, 311.2
    yy, xx = np.indices((613, 625))
    r = np.hypot(xx - cx, yy - cy)
    counts = np.full(r.shape, 12.0, dtype=np.float32)
    for order, radius in enumerate(radii, start=1):
        counts += (1200 / order) * np.exp(-0.5 * ((r - radius) / 1.8)**2)
    analysis = preprocess_detector_image(DetectorImage(counts, None, Path("simulated_agbh.npy")))

    peaks = [DetectedPeak(radius, 2, 1, 4, 1) for radius in radii]
    candidates = generate_distance_candidates(peaks, standard, energy, pixel, cx, cy, 1.0,
                                              (distance * 0.45, distance * 1.05))
    solutions = {tuple(ring.theoretical_index + 1 for ring in c.matched_rings): c for c in candidates}
    scores = []
    for orders in ((1, 2, 3), (2, 4, 6)):
        c = solutions[orders]
        refine_ring_geometry(analysis, c, wavelength, pixel, pixel, tilt="auto")
        assert c.tilt_refined and c.point_rms_px < 0.5
        evidence = search_rings(analysis, c, standard, wavelength, pixel, pixel)
        present = [item for item in evidence if item.match is not None]
        assert tuple(item.theoretical_index + 1 for item in present) == orders
        # With only the measured lines the wrong distance estimate favours the alias.
        without_absence = score_ring_evidence(c, present, standard, distance / 2)
        with_absence = score_ring_evidence(c, evidence, standard, distance / 2)
        scores.append((without_absence, with_absence))
    assert scores[0][0] < scores[1][0]
    assert scores[0][1] > scores[1][1]
