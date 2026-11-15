"""Calibration standard-ring matching."""

from __future__ import annotations


import numpy as np

from .geometry_model import distance_from_ring_radius, q_to_ring_radius_px
from .models import CalibrationCandidate, CalibrationStandard, MatchedRing
from .peak_detector import DetectedPeak

HARMONIC_TOLERANCE_PX, HARMONIC_TOLERANCE_REL = 2.5, 0.012
"""A ring of a harmonic standard (AgBH: radii in the ratio 1 : 2 : 3 ...) that is off the fitted distance by more than
this (pixels, or this share of its radius) is not that order: a Yoneda band about half the first order's radius is
then not taken for order 1 of a half-distance solution."""
MAX_EXPECTED_ORDER = 8
"""Highest order (AgBH) or reflection (crystalline standards) expected to be seen where the detector records."""
STRONG_PEAK_SHARE = 0.3
"""A radial peak with at least this share of the strongest peak's prominence is a ring the solution must explain."""


def _fit_distance(matches: list[MatchedRing], energy_kev: float, pixel_size_m: float) -> float:
    unit = q_to_ring_radius_px([item.q_inv_angstrom for item in matches], energy_kev, 1.0, pixel_size_m)
    observed = np.asarray([item.observed_radius_px for item in matches], dtype=float)
    return float(np.dot(unit, observed) / max(np.dot(unit, unit), 1e-12))


def _completeness(matches: list[MatchedRing], predicted: np.ndarray, window_px, peaks: list[DetectedPeak]) -> float | None:
    """Share of the orders (at most MAX_EXPECTED_ORDER) expected to be seen that were found. Expected: the predicted
    radius lies where the detector records a radial profile and not beyond the outermost strong peak (prominence at
    least 30 % of the strongest; with tolerance). Near 1 for the true solution; low for a short-distance alias that pairs
    a few peaks with scattered high orders, and for a half-distance one that leaves a strong outer ring unexplained."""
    if window_px is None or not matches:
        return None
    low, high = window_px
    strongest = max((peak.prominence for peak in peaks), default=0.0)
    outer = max((peak.radius_px for peak in peaks if peak.prominence >= STRONG_PEAK_SHARE * strongest), default=high)
    high = min(high, outer * (1.0 + HARMONIC_TOLERANCE_REL) + HARMONIC_TOLERANCE_PX)
    top = max(max(item.theoretical_index for item in matches), int(np.sum(predicted[:MAX_EXPECTED_ORDER] <= high)) - 1)
    top = min(top, MAX_EXPECTED_ORDER - 1)
    expected = [index for index in range(top + 1) if low <= predicted[index] <= high]
    if not expected:
        return None
    found = {item.theoretical_index for item in matches}
    return float(sum(1 for index in expected if index in found) / len(expected))


def _match_at_distance(
    peaks: list[DetectedPeak],
    standard: CalibrationStandard,
    energy_kev: float,
    pixel_size_m: float,
    distance_mm: float,
    center_x_px: float,
    center_y_px: float,
    valid_fraction: float,
    window_px: tuple[float, float] | None = None,
) -> CalibrationCandidate | None:
    if not peaks:
        return None
    predicted = q_to_ring_radius_px(standard.q_values_inv_angstrom, energy_kev, distance_mm, pixel_size_m)
    pairs: list[tuple[float, int, int]] = []
    for theory_idx, radius in enumerate(predicted):
        for peak_idx, peak in enumerate(peaks):
            tolerance = max(3.0, 0.018 * float(radius), 0.6 * peak.width_px)
            difference = abs(peak.radius_px - float(radius))
            if difference <= tolerance:
                pairs.append((difference, theory_idx, peak_idx))
    used_theory: set[int] = set()
    used_peaks: set[int] = set()
    matches: list[MatchedRing] = []
    for difference, theory_idx, peak_idx in sorted(pairs):
        if theory_idx in used_theory or peak_idx in used_peaks:
            continue
        used_theory.add(theory_idx)
        used_peaks.add(peak_idx)
        peak = peaks[peak_idx]
        matches.append(MatchedRing(
            theoretical_index=theory_idx,
            q_inv_angstrom=float(standard.q_values_inv_angstrom[theory_idx]),
            observed_radius_px=peak.radius_px,
            predicted_radius_px=float(predicted[theory_idx]),
            residual_px=peak.radius_px - float(predicted[theory_idx]),
            prominence=peak.prominence,
            coverage=peak.coverage,
        ))
    if not matches:
        return None
    # With correspondences fixed, solve the one-parameter distance model
    # analytically before joint center refinement.
    unit_radii = q_to_ring_radius_px(
        [item.q_inv_angstrom for item in matches], energy_kev, 1.0, pixel_size_m
    )
    observed_radii = np.asarray([item.observed_radius_px for item in matches], dtype=float)
    fitted_distance = float(np.dot(unit_radii, observed_radii) / max(np.dot(unit_radii, unit_radii), 1e-12))
    if fitted_distance > 0:
        distance_mm = fitted_distance
        for item, unit_radius in zip(matches, unit_radii):
            item.predicted_radius_px = float(unit_radius * distance_mm)
            item.residual_px = item.observed_radius_px - item.predicted_radius_px
    if standard.key == "agbh" and len(matches) > 1:
        for _ in range(2):                                # harmonic: an order far off the common distance is not it
            limit = [max(HARMONIC_TOLERANCE_PX, HARMONIC_TOLERANCE_REL * item.predicted_radius_px) for item in matches]
            kept = [item for item, tol in zip(matches, limit) if abs(item.residual_px) <= tol]
            if not kept or len(kept) == len(matches):
                break
            matches = kept
            distance_mm = _fit_distance(matches, energy_kev, pixel_size_m)
            for item in matches:
                item.predicted_radius_px = float(q_to_ring_radius_px([item.q_inv_angstrom], energy_kev, distance_mm, pixel_size_m)[0])
                item.residual_px = item.observed_radius_px - item.predicted_radius_px
    inferred = [
        distance_from_ring_radius(item.observed_radius_px, item.q_inv_angstrom, 12.398419843320026 / energy_kev, pixel_size_m)
        for item in matches
    ]
    rms = float(np.sqrt(np.mean([item.residual_px ** 2 for item in matches])))
    consistency = float(np.std(inferred) / max(np.mean(inferred), 1e-6)) if len(inferred) > 1 else 1.0
    coverage = float(np.mean([item.coverage for item in matches]))
    candidate = CalibrationCandidate(
        standard_key=standard.key,
        center_x_px=float(center_x_px),
        center_y_px=float(center_y_px),
        distance_mm=float(distance_mm),
        matched_ring_count=len(matches),
        rms_residual_px=rms,
        distance_consistency=consistency,
        azimuthal_coverage=coverage,
        matched_rings=sorted(matches, key=lambda item: item.theoretical_index),
        detected_peak_radii_px=[peak.radius_px for peak in peaks],
        valid_data_fraction=float(valid_fraction),
    )
    # Not a field of the record (a file stays readable by earlier versions): used for ranking only.
    all_predicted = q_to_ring_radius_px(standard.q_values_inv_angstrom, energy_kev, distance_mm, pixel_size_m)
    candidate.order_completeness = _completeness(matches, np.asarray(all_predicted, dtype=float), window_px, peaks)
    return candidate


def generate_distance_candidates(
    peaks: list[DetectedPeak],
    standard: CalibrationStandard,
    energy_kev: float,
    pixel_size_m: float,
    center_x_px: float,
    center_y_px: float,
    valid_fraction: float,
    distance_range_mm: tuple[float, float],
    window_px: tuple[float, float] | None = None,
) -> list[CalibrationCandidate]:
    low, high = distance_range_mm
    wavelength = 12.398419843320026 / energy_kev
    seeds: list[float] = []
    for peak in peaks[:35]:
        for q in standard.q_values_inv_angstrom:
            try:
                distance = distance_from_ring_radius(peak.radius_px, q, wavelength, pixel_size_m)
            except ValueError:
                continue
            if low <= distance <= high:
                seeds.append(distance)
    if not seeds:
        return []
    seeds.sort()
    clustered: list[float] = []
    cluster = [seeds[0]]
    for value in seeds[1:]:
        if abs(value - np.median(cluster)) / max(value, 1.0) < 0.008:
            cluster.append(value)
        else:
            clustered.append(float(np.median(cluster)))
            cluster = [value]
    clustered.append(float(np.median(cluster)))
    candidates = [
        _match_at_distance(peaks, standard, energy_kev, pixel_size_m, distance, center_x_px, center_y_px, valid_fraction,
                           window_px)
        for distance in clustered
    ]
    # The range is a limit, not a starting point: a match whose fitted distance leaves it is not a solution.
    valid_candidates = [candidate for candidate in candidates
                        if candidate is not None and low <= candidate.distance_mm <= high]
    # Keep fundamental-order solutions before truncation. Sorting only by the
    # number of coincidences discards the physically correct long-distance
    # solution whenever noise peaks can be paired with many high AgBH orders.
    if standard.key == "agbh":
        valid_candidates.sort(key=lambda item: (
            min((match.theoretical_index for match in item.matched_rings), default=999),
            -item.matched_ring_count,
            item.rms_residual_px,
            item.distance_consistency,
        ))
    else:
        # Crystalline powder standards are not harmonic sequences. Low-q
        # reflections may simply fall outside a WAXS detector, so retain the
        # solutions supported by the most internally consistent reflections.
        valid_candidates.sort(key=lambda item: (
            -item.matched_ring_count,
            item.rms_residual_px,
            item.distance_consistency,
        ))
    return valid_candidates[:50]


def rematch_candidate(
    candidate: CalibrationCandidate,
    peaks: list[DetectedPeak],
    standard: CalibrationStandard,
    energy_kev: float,
    pixel_size_m: float,
    window_px: tuple[float, float] | None = None,
) -> CalibrationCandidate:
    result = _match_at_distance(
        peaks, standard, energy_kev, pixel_size_m, candidate.distance_mm,
        candidate.center_x_px, candidate.center_y_px, candidate.valid_data_fraction, window_px,
    )
    if result is not None:
        result.center_method = candidate.center_method
        result.center_support = candidate.center_support
        result.warnings = list(candidate.warnings)
    return result or candidate
