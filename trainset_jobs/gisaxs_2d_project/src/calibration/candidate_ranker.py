from __future__ import annotations

import math

import numpy as np

from .models import CalibrationCandidate


def _add_warning(candidate: CalibrationCandidate, message: str) -> None:
    if message not in candidate.warnings:
        candidate.warnings.append(message)


def score_candidate(candidate: CalibrationCandidate, estimated_distance_mm: float | None = None) -> CalibrationCandidate:
    rings = candidate.matched_ring_count
    prior = 0.0
    if estimated_distance_mm and estimated_distance_mm > 0:
        relative = abs(candidate.distance_mm - estimated_distance_mm) / estimated_distance_mm
        # A stated estimate is intentionally a soft, not hard, constraint. Its
        # weight is nevertheless large enough to resolve common AgBH harmonic
        # aliases (for example distance and half-distance solutions).
        prior = 65.0 * math.exp(-0.5 * (relative / 0.20) ** 2)
    harmonic_standard = candidate.standard_key == "agbh"
    indices = sorted(match.theoretical_index for match in candidate.matched_rings)
    if indices:
        lowest_order = indices[0]
        span = max(1, indices[-1] - indices[0] + 1)
        fundamental_support = 1.0 / (1.0 + lowest_order)
        continuity = min(1.0, rings / span)
        candidate.harmonic_quality = (
            0.55 * fundamental_support + 0.45 * continuity
            if harmonic_standard else continuity
        )
    else:
        lowest_order = 0
        candidate.harmonic_quality = 0.0
    inner_unmatched = 0
    if harmonic_standard and candidate.matched_rings:
        first_matched = min(match.observed_radius_px for match in candidate.matched_rings)
        matched_radii = [match.observed_radius_px for match in candidate.matched_rings]
        inner_unmatched = sum(
            1 for radius in candidate.detected_peak_radii_px
            if 50.0 <= radius < 0.65 * first_matched
            and all(abs(radius - matched) > 5.0 for matched in matched_radii)
        )
    mean_prominence = float(np.mean([
        max(0.0, match.prominence) for match in candidate.matched_rings
    ])) if candidate.matched_rings else 0.0
    ring_score = min(rings, 5) * 22.0 + max(0, rings - 5) * 5.0
    candidate.score = (
        ring_score
        + min(candidate.azimuthal_coverage, 1.0) * 16.0
        + candidate.valid_data_fraction * 5.0
        + min(1.0, candidate.center_support) * 80.0
        + candidate.harmonic_quality * 28.0
        + min(mean_prominence, 0.5) * 40.0
        + prior
        - min(candidate.rms_residual_px, 20.0) * 3.5
        - min(candidate.distance_consistency, 1.0) * 25.0
        - min(inner_unmatched, 2) * 32.0
        # For AgBH-like harmonic standards, assigning the innermost clear
        # experimental peak to a high theoretical order creates many seductive
        # but unphysical short-distance aliases. Missing order 1 is acceptable
        # near a large beamstop; missing several observable inner orders is not.
        - (min(lowest_order, 5) * 32.0 if harmonic_standard else 0.0)
    )
    if rings >= 4 and candidate.rms_residual_px <= 2.0 and candidate.azimuthal_coverage >= 0.18:
        candidate.confidence = "High"
    elif (
        rings >= 2 and candidate.rms_residual_px <= 6.0
        or rings >= 4
        and candidate.center_support >= 0.8
        and candidate.rms_residual_px <= 8.0
        and candidate.azimuthal_coverage >= 0.05
        or rings >= 5
        and candidate.center_support >= 0.8
        and candidate.rms_residual_px <= 10.0
        and candidate.azimuthal_coverage >= 0.015
    ):
        candidate.confidence = "Medium"
    else:
        candidate.confidence = "Low"
    if rings == 1:
        _add_warning(candidate, "Only one ring is matched; center and distance remain ambiguous.")
    coverage_warnings = {
        "Matched rings have very limited azimuthal coverage; verify the overlay before applying.",
        "Matched rings are partial GISAXS arcs; verify the overlay before applying.",
    }
    candidate.warnings[:] = [warning for warning in candidate.warnings if warning not in coverage_warnings]
    if candidate.azimuthal_coverage < 0.08:
        _add_warning(candidate, "Matched rings have very limited azimuthal coverage; verify the overlay before applying.")
    elif candidate.azimuthal_coverage < 0.15:
        _add_warning(candidate, "Matched rings are partial GISAXS arcs; verify the overlay before applying.")
    if inner_unmatched:
        _add_warning(candidate, "A strong inner radial peak is unexplained by this harmonic assignment.")
    return candidate


def rank_candidates(candidates: list[CalibrationCandidate], estimated_distance_mm: float | None = None, limit: int = 5) -> list[CalibrationCandidate]:
    scored = [score_candidate(candidate, estimated_distance_mm) for candidate in candidates]
    scored.sort(key=lambda item: item.score, reverse=True)
    unique: list[CalibrationCandidate] = []
    for item in scored:
        duplicate = any(
            item.standard_key == old.standard_key
            and abs(item.distance_mm - old.distance_mm) / max(item.distance_mm, 1.0) < 0.015
            and math.hypot(item.center_x_px - old.center_x_px, item.center_y_px - old.center_y_px) < 5.0
            for old in unique
        )
        if not duplicate:
            unique.append(item)
        if len(unique) >= limit:
            break
    return unique
