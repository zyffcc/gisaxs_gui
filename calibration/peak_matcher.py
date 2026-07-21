from __future__ import annotations

import math

import numpy as np

from .geometry_model import distance_from_ring_radius, q_to_ring_radius_px
from .models import CalibrationCandidate, CalibrationStandard, MatchedRing
from .peak_detector import DetectedPeak


def _match_at_distance(
    peaks: list[DetectedPeak],
    standard: CalibrationStandard,
    energy_kev: float,
    pixel_size_m: float,
    distance_mm: float,
    center_x_px: float,
    center_y_px: float,
    valid_fraction: float,
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
    inferred = [
        distance_from_ring_radius(item.observed_radius_px, item.q_inv_angstrom, 12.398419843320026 / energy_kev, pixel_size_m)
        for item in matches
    ]
    rms = float(np.sqrt(np.mean([item.residual_px ** 2 for item in matches])))
    consistency = float(np.std(inferred) / max(np.mean(inferred), 1e-6)) if len(inferred) > 1 else 1.0
    coverage = float(np.mean([item.coverage for item in matches]))
    return CalibrationCandidate(
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


def generate_distance_candidates(
    peaks: list[DetectedPeak],
    standard: CalibrationStandard,
    energy_kev: float,
    pixel_size_m: float,
    center_x_px: float,
    center_y_px: float,
    valid_fraction: float,
    distance_range_mm: tuple[float, float],
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
        _match_at_distance(peaks, standard, energy_kev, pixel_size_m, distance, center_x_px, center_y_px, valid_fraction)
        for distance in clustered
    ]
    valid_candidates = [candidate for candidate in candidates if candidate is not None]
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
) -> CalibrationCandidate:
    result = _match_at_distance(
        peaks, standard, energy_kev, pixel_size_m, candidate.distance_mm,
        candidate.center_x_px, candidate.center_y_px, candidate.valid_data_fraction,
    )
    if result is not None:
        result.center_method = candidate.center_method
        result.center_support = candidate.center_support
        result.warnings = list(candidate.warnings)
    return result or candidate
