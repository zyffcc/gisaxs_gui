"""Calibration candidate ranking rules."""

from __future__ import annotations

import math

import numpy as np

from .guided_rings import N_SECTORS, RingEvidence
from .models import CalibrationCandidate, CalibrationStandard

RING_PROMINENCE_FULL = 0.1
"""Prominence of a matched ring (in the log1p-scaled radial profile: about its relative contrast, 0.1 = 10 %) from
which it counts as a whole ring in the score; a weaker one counts in proportion. Rings of calibrants stand out by far
more (real frames 0.12-1.7); peaks of a few percent are where noise and stray features meet a harmonic sequence by
chance, and four of them must not outweigh three clear rings (a grazing AgBH with the CI's NumPy 1.26 / SciPy 1.15
gave such a 1683 mm alias of a 2200 mm image)."""


def ring_weight(prominence: float) -> float:
    """How much of a ring a matched peak of this prominence counts for (0-1)."""
    return min(1.0, max(0.0, float(prominence)) / RING_PROMINENCE_FULL)


def _add_warning(candidate: CalibrationCandidate, message: str) -> None:
    if message not in candidate.warnings:
        candidate.warnings.append(message)


def score_candidate(candidate: CalibrationCandidate, estimated_distance_mm: float | None = None,
                    grazing: bool = False) -> CalibrationCandidate:
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
    # Grazing incidence: the Yoneda band and the sample horizon are strong peaks inside the first ring by nature.
    if harmonic_standard and candidate.matched_rings and not grazing:
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
    # the rings counted by how clearly they stand out (RING_PROMINENCE_FULL): a coincidence with a noise-level peak is
    # not a ring of the standard
    counted = sum(ring_weight(match.prominence) for match in candidate.matched_rings) if candidate.matched_rings \
        else float(rings)
    ring_score = min(counted, 5.0) * 22.0 + max(0.0, counted - 5.0) * 5.0
    completeness = getattr(candidate, "order_completeness", None)
    if completeness is not None:
        # The orders a harmonic standard shows where the detector records are all there for the true distance; a
        # short-distance alias pairs a few peaks with scattered high orders and leaves most of its orders unseen.
        # Squared for AgBH: one missing order outweighs one extra coincidence (a Yoneda band taken for order 1 of a
        # half-distance solution leaves its order 3 unseen). A crystalline standard has weak reflections: linear.
        ring_score *= completeness ** 2 if harmonic_standard else completeness
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
    # Median offsets of opposing arcs can cancel even when individual points miss the fitted cone.
    rms = max(candidate.rms_residual_px, candidate.point_rms_px or 0.0)
    if rings >= 4 and rms <= 2.0 and candidate.azimuthal_coverage >= 0.18:
        candidate.confidence = "High"
    elif (
        rings >= 2 and rms <= 6.0
        or rings >= 4
        and candidate.center_support >= 0.8
        and rms <= 8.0
        and candidate.azimuthal_coverage >= 0.05
        or rings >= 5
        and candidate.center_support >= 0.8
        and rms <= 10.0
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


def rank_candidates(candidates: list[CalibrationCandidate], estimated_distance_mm: float | None = None, limit: int = 5,
                    grazing: bool = False) -> list[CalibrationCandidate]:
    scored = [score_candidate(candidate, estimated_distance_mm, grazing) for candidate in candidates]
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


def score_ring_evidence(candidate: CalibrationCandidate, evidence: list[RingEvidence], standard: CalibrationStandard,
                        estimated_distance_mm: float | None = None) -> float:
    """Rank the fitted geometry in units of a confirmed ring.

    A confirmed order contributes 1/2 plus half its supporting fraction of observable sectors.
    Missing reflections cost half their observable fraction. An unexplained strong peak in the
    independent radial scan costs one ring. The point-fit penalty is log(1 + RMS²) in pixel units.
    Distance and tabulated intensity order break close ties.
    """
    observable = [item for item in evidence if item.observable_sectors >= 4]
    confirmed = [item for item in observable if item.match is not None]
    supported = sum(0.5 + 0.5 * item.supporting_sectors / item.observable_sectors for item in confirmed)
    missing = sum(item.observable_sectors / N_SECTORS for item in observable if item.match is None)
    tested_orders = {item.theoretical_index for item in observable}
    # Radial matches with no guided test, including separately resolved close reflections.
    untested = [ring for ring in candidate.matched_rings if ring.theoretical_index not in tested_orders]
    supported += 0.5 * sum(ring_weight(ring.prominence) for ring in untested)

    # The independent profile is circular and may precede the final centre refinement.
    # Its peak can move by at most the centre displacement in addition to the radial match tolerance.
    old_center = getattr(candidate, "radial_profile_center", (candidate.center_x_px, candidate.center_y_px))
    center_shift = math.hypot(old_center[0] - candidate.center_x_px, old_center[1] - candidate.center_y_px)
    intervals = [span for item in confirmed for span in item.detector_radius_ranges_px]
    intervals.extend((ring.observed_radius_px, ring.observed_radius_px) for ring in untested)
    unexplained = 0
    for radius in getattr(candidate, "strong_peak_radii", ()):
        tolerance = center_shift + max(3.0, 0.018 * radius)
        if all(radius < low - tolerance or radius > high + tolerance for low, high in intervals):
            unexplained += 1

    # Pairwise concordance, bounded in [-1,1]. Sample texture and absorption can reverse the intensity order.
    ordering = []
    if standard.relative_intensities:
        for i, outer in enumerate(confirmed):
            for inner in confirmed[:i]:
                expected = (standard.relative_intensities[inner.theoretical_index]
                            - standard.relative_intensities[outer.theoretical_index])
                observed = inner.match.prominence - outer.match.prominence
                ordering.append(float(np.sign(expected * observed)))
    intensity_prior = float(np.mean(ordering)) if ordering else 0.0
    distance_prior = 0.0
    if estimated_distance_mm and estimated_distance_mm > 0:
        relative = (candidate.distance_mm - estimated_distance_mm) / estimated_distance_mm
        distance_prior = math.exp(-0.5 * (relative / 0.20) ** 2)

    # A one-ring geometry has no independent point fit and a zero radial assignment residual by construction.
    rms = candidate.point_rms_px if candidate.point_rms_px is not None else 2.0
    candidate.score = (supported - 0.5 * missing - unexplained - math.log1p(rms**2)
                       + 0.2 * distance_prior + 0.1 * intensity_prior)
    return candidate.score
