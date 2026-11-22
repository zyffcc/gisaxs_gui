"""Search along the predicted cones for diffraction lines with local radial contrast."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import CalibrationCandidate, CalibrationStandard, MatchedRing
from .preprocessing import AnalysisImage
from .tilt_fit import _geometry


N_SECTORS = 72
RAYS_PER_SECTOR = 12


@dataclass(frozen=True)
class RingEvidence:
    theoretical_index: int
    predicted_radius_px: float
    observable_sectors: int
    supporting_sectors: int
    match: MatchedRing | None
    detector_radius_ranges_px: tuple[tuple[float, float], ...]
    """Raw detector-pixel radii sampled at the peak, one interval per supporting sector."""
    resolved_radial_match: MatchedRing | None = None
    """An independently resolved line whose neighbour leaves no room for background flanks."""


def _sector_profiles(analysis, cx, cy, dx, dy, radii, offsets):
    """Mean signal at integer offsets from the predicted ellipse, masking gaps on each ray."""
    rows, cols = analysis.signal.shape
    distance = radii[:, None] + offsets
    x = np.rint(cx + dx[:, None] * distance).astype(np.intp)
    y = np.rint(cy + dy[:, None] * distance).astype(np.intp)
    valid = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
    np.clip(x, 0, cols - 1, out=x)
    np.clip(y, 0, rows - 1, out=y)
    valid &= analysis.valid[y, x]
    shape = (N_SECTORS, RAYS_PER_SECTOR, offsets.size)
    counts = valid.reshape(shape).sum(axis=1)
    sums = np.where(valid, analysis.signal[y, x], 0).reshape(shape).sum(axis=1, dtype=float)
    # A single grazing ray through a corner or a module edge gives an unreliable sector peak.
    profiles = np.divide(sums, counts, out=np.zeros_like(sums), where=counts >= 4)
    return profiles, counts >= 4, valid.reshape(shape)


def _evidence(analysis, cx, cy, dx, dy, radii, nominal, index, q, window, flank):
    offsets = np.arange(-window - flank, window + flank + 1)
    profiles, valid, ray_valid = _sector_profiles(analysis, cx, cy, dx, dy, radii, offsets)
    # Leave a guard band between the search and its reference background. Use both flanks:
    # a step where a module or a beam stop ends is not a peak.
    sides = [offsets <= -window - 3, offsets >= window + 3]
    backgrounds, variances, counts = [], [], []
    for side in sides:
        values, use = profiles[:, side], valid[:, side]
        count = use.sum(axis=1)
        mean = (values * use).sum(axis=1) / np.maximum(count, 1)
        variance = (((values - mean[:, None]) ** 2) * use).sum(axis=1) / np.maximum(count - 1, 1)
        backgrounds.append(mean)
        variances.append(variance)
        counts.append(count)
    search = abs(offsets) <= window
    active = (counts[0] >= 4) & (counts[1] >= 4) & (valid[:, search].sum(axis=1) >= window + 1)
    # Average adjacent radial bins to suppress isolated bright pixels.
    smooth = (profiles[:, :-2] + 2 * profiles[:, 1:-1] + profiles[:, 2:]) / 4
    neighbors = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    peaks = np.where(neighbors, smooth, -np.inf)[:, search[1:-1]]
    argmax = np.argmax(peaks, axis=1)
    height = peaks[np.arange(N_SECTORS), argmax] - np.maximum(*backgrounds)
    noise = np.sqrt((variances[0] + variances[1]) / 2)
    significant = active & (height > np.maximum(4 * noise, 1e-5))
    shifts = argmax - window
    # A real arc has compatible radial offsets in multiple sectors. A search-edge maximum
    # could just be the shoulder of a neighbouring line outside the band.
    significant &= abs(shifts) < window
    if significant.any():
        typical = np.median(shifts[significant])
        significant &= abs(shifts - typical) <= 3
    n_valid, n_found = int(active.sum()), int(significant.sum())
    match = None
    ranges = ()
    if n_found >= max(4, math.ceil(0.2 * n_valid)):
        # Report residuals in the untilted equivalent-radius convention of MatchedRing.
        local_radius = radii.reshape(N_SECTORS, RAYS_PER_SECTOR).mean(axis=1)
        shift = float(np.median(shifts[significant] * nominal / local_radius[significant]))
        match = MatchedRing(index, q, nominal + shift, nominal, shift,
                            float(np.median(height[significant])), n_found / N_SECTORS)
        # A circular profile measures detector pixels, even on a tilted or non-square detector.
        # Keep the span of each measured arc, excluding rays that missed the detector or a module.
        sectors = np.flatnonzero(significant)
        peak_bins = argmax[sectors] + flank
        usable = ray_valid[sectors, :, peak_bins]
        observed = radii.reshape(N_SECTORS, RAYS_PER_SECTOR)[sectors] + shifts[sectors, None]
        lower = np.min(observed, axis=1, where=usable, initial=np.inf)
        upper = np.max(observed, axis=1, where=usable, initial=-np.inf)
        ranges = tuple(zip(lower.tolist(), upper.tolist()))
    return RingEvidence(index, nominal, n_valid, n_found, match, ranges)


def search_rings(
    analysis: AnalysisImage, candidate: CalibrationCandidate, standard: CalibrationStandard,
    wavelength: float, px: float, py: float,
) -> list[RingEvidence]:
    """Test each reachable standard line, including the already assigned lines.

    Along one detector ray one pixel of travel has laboratory components (a, b, c)
    along, right and up. For t = tan(2θ), solve r hypot(b, c) / (D + r a) = t.
    Only orders with observable sectors contribute evidence of presence or absence.
    """
    cx, cy = candidate.center_x_px, candidate.center_y_px
    params = (cx, cy, candidate.distance_mm, *np.radians([candidate.rot1_deg, candidate.rot2_deg]))
    geometry = _geometry(params, wavelength, px, py, math.radians(candidate.rot3_deg))
    angles = (np.arange(N_SECTORS * RAYS_PER_SECTOR) + 0.5) * (2 * np.pi / (N_SECTORS * RAYS_PER_SECTOR))
    dx, dy = np.cos(angles), np.sin(angles)
    columns, up_rows = geometry.in_plane_axes()
    along, right, up = np.asarray(columns)[:, None] * (px * dx) - np.asarray(up_rows)[:, None] * (py * dy)
    transverse = np.hypot(right, up)
    rows, cols = analysis.signal.shape
    reach = np.hypot(max(abs(cx), abs(cols - 1 - cx)), max(abs(cy), abs(rows - 1 - cy)))
    q_values = np.asarray(standard.q_values_inv_angstrom)
    ratios = q_values * wavelength / (4 * np.pi)
    indices = np.flatnonzero((ratios > 0) & (ratios < math.sin(np.pi / 4)))
    tangents = np.tan(2 * np.arcsin(ratios[indices]))
    nominal = geometry.distance_m * tangents / (0.5 * (px + py))
    # A pair of close reflections may be resolved in the radial scan even when two clean
    # background flanks cannot fit between them. Require a measured peak and at least as
    # much azimuth as the four sectors required to accept a guided arc.
    radial_matches = {ring.theoretical_index: ring for ring in candidate.matched_rings
                      if ring.observed_radius_px in candidate.detected_peak_radii_px
                      and ring.coverage >= 4 / N_SECTORS}
    evidence = []
    for i, (index, tangent, radius) in enumerate(zip(indices, tangents, nominal)):
        denominator = transverse - tangent * along
        radii = np.divide(geometry.distance_m * tangent, denominator,
                          out=np.full_like(denominator, reach + 100), where=denominator > 0)
        if radius < 40 or radii.min() > reach + 25:
            continue
        # The flanks must not include the neighbouring standard line.
        spacing = min(radius - nominal[i - 1] if i else np.inf,
                      nominal[i + 1] - radius if i + 1 < nominal.size else np.inf)
        room = min(22, int(0.45 * spacing)) if np.isfinite(spacing) else 22
        window = min(int(np.clip(np.ceil(0.012 * radius), 4, 12)), room - 6)
        if window < 2:
            if index in radial_matches:
                evidence.append(RingEvidence(int(index), float(radius), 0, 0, None, (), radial_matches[index]))
            continue
        item = _evidence(analysis, cx, cy, dx, dy, radii, float(radius), int(index), float(q_values[index]),
                         window, min(10, room - window))
        if item.observable_sectors:
            evidence.append(item)
    return evidence
