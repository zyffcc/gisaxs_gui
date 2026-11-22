"""Joint ring-point fit of the direct beam and sample distance with fixed or fitted detector tilt.

The radial residual in pixels is L (tan 2θ_point − tan 2θ_ring) / mean_pixel_size. Pixel directions
come from the same geometry as Analyze. Each pass finds subpixel centroids in azimuthal sectors around
the current rings. The last pass supplies one set of points for the fit, all RMS values and uncertainties.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from src.gimap.shared.geometry import DetectorGeometry, lab_rays

from .geometry_model import q_to_two_theta
from .models import CalibrationCandidate
from .preprocessing import AnalysisImage

TILT_MIN_RINGS = 3
"""Matched rings needed before the tilt is refined automatically (two rings: only on request)."""
TILT_LIMIT_DEG = 30.0
"""Largest |rot1|, |rot2| the fit may reach: beyond this the ring points no longer constrain a flat detector."""
POINT_PASSES = 4
"""Times the ring points are taken again around the rings of the newest geometry."""
POINTS_PER_RING = 800
PEAK_SHARE = 0.3
"""A sector gives a ring point when the ring stands out there by at least this share of its median height."""
_BLOCK_PIXELS = 1 << 20
NOT_REFINED_FEW_RINGS = "The detector tilt was not refined: it needs at least three matched rings."
NOT_REFINED_POINTS = "The detector tilt was not refined: too few ring points."
NOT_REFINED_CONTRIBUTING_RINGS = "Automatic tilt refinement needs usable points from at least three rings."
_TOO_FEW_GEOMETRY_POINTS = "Too few experimental ring points were available for joint refinement."


def _geometry(params, wavelength: float, px: float, py: float, rot3: float) -> DetectorGeometry:
    """GIMaP's geometry of the fit parameters ``(cx, cy, distance_mm[, rot1, rot2])``; the centre counts pixel
    centres from 0 like the calibration engine (+0.5 px in GIMaP's pixel-corner coordinates)."""
    rot1, rot2 = (float(params[3]), float(params[4])) if len(params) > 3 else (0.0, 0.0)
    return DetectorGeometry(px, py, float(params[2]) * 1e-3, float(params[0]) + 0.5, float(params[1]) + 0.5,
                            wavelength, rot1_rad=rot1, rot2_rad=rot2, rot3_rad=rot3)


def tangent_two_theta(geometry: DetectorGeometry, xs, ys, cx: float, cy: float) -> np.ndarray:
    """``tan 2θ`` of pixel centres at column ``xs``, row ``ys`` (indices) for a beam centre ``(cx, cy)`` (indices)."""
    along, right, up = lab_rays(geometry, (np.asarray(xs, float) - cx) * geometry.pixel_size_x_m,
                                (cy - np.asarray(ys, float)) * geometry.pixel_size_y_m)
    return np.hypot(right, up) / along


def tilt_residual(xs, ys, qs, wavelength: float, px: float, py: float, rot3: float = 0.0, *, fixed_tilt=None):
    """Residual (pixels) of ring points for the parameters ``(cx, cy, distance_mm, rot1_rad, rot2_rad)``."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ring_tangent = np.tan(q_to_two_theta(np.asarray(qs, float), wavelength))
    mean_pixel = 0.5 * (px + py)

    def residual(params) -> np.ndarray:
        geometry = _geometry(params if fixed_tilt is None else np.r_[params, fixed_tilt], wavelength, px, py, rot3)
        tangent = tangent_two_theta(geometry, xs, ys, float(params[0]), float(params[1]))
        return float(params[2]) * 1e-3 * (tangent - ring_tangent) / mean_pixel

    return residual


def tilted_ring_points(
    analysis: AnalysisImage, candidate: CalibrationCandidate, params, wavelength: float,
    px: float, py: float, rot3: float = 0.0, *, widen: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(xs, ys, qs)``: one point per azimuthal sector of each matched ring of the geometry ``params``.

    The valid pixels within a band about the ring (± max(3, min(10, 1.2 % of the radius)) px, times ``widen``) are
    split into sectors about 2 px of arc long (36 to 720 a ring). In a sector, every pixel weighs by how far it
    stands above the sector's lower quartile; the point is the weighted mean position (sub-pixel), kept when the
    sector's brightest pixel stands above that quartile by at least ``PEAK_SHARE`` of the ring's median such height.
    Both radial flanks of the band must have data. Averaging a peak cut by a gap or detector edge displaces the
    centroid towards the illuminated side. Done in blocks of rows."""
    geometry = _geometry(params, wavelength, px, py, rot3)
    cx, cy = float(params[0]), float(params[1])
    distance_m, mean_pixel = float(params[2]) * 1e-3, 0.5 * (px + py)
    rings = []
    for match in candidate.matched_rings:
        try:
            tangent = float(np.tan(q_to_two_theta(match.q_inv_angstrom, wavelength)))
        except ValueError:
            continue
        radius = distance_m * tangent / mean_pixel
        rings.append((match.q_inv_angstrom, tangent, radius, widen * max(3.0, min(10.0, 0.012 * radius))))
    rows, columns = analysis.signal.shape
    found: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in rings]
    step = max(1, _BLOCK_PIXELS // max(1, columns))
    column_index = np.arange(columns, dtype=float)
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        ys, xs = np.meshgrid(np.arange(start, stop, dtype=float), column_index, indexing="ij")
        tangent = tangent_two_theta(geometry, xs, ys, cx, cy)
        valid = analysis.valid[start:stop]
        signal = analysis.signal[start:stop]
        for index, (_q, ring_tangent, _radius, half_width) in enumerate(rings):
            offset = distance_m * (tangent - ring_tangent) / mean_pixel
            band = valid & (np.abs(offset) <= half_width)
            if band.any():
                found[index].append((xs[band], ys[band], signal[band].astype(float), offset[band]))
    points_x, points_y, point_q = [], [], []
    for (q, _tangent, radius, half_width), parts in zip(rings, found):
        if not parts:
            continue
        xs = np.concatenate([part[0] for part in parts])
        ys = np.concatenate([part[1] for part in parts])
        values = np.concatenate([part[2] for part in parts])
        if values.size < 12:
            continue
        sectors = int(np.clip(np.pi * radius, 36, min(720, POINTS_PER_RING)))
        angle = np.arctan2(cy - ys, xs - cx)
        sector = np.minimum(((angle + np.pi) / (2 * np.pi) * sectors).astype(np.int64), sectors - 1)
        order = np.lexsort((values, sector))
        counts = np.bincount(sector, minlength=sectors)
        starts = np.cumsum(counts) - counts
        low = np.zeros(sectors)
        filled = counts >= 3
        low[filled] = values[order][starts[filled] + counts[filled] // 4]
        weight = np.clip(values - low[sector], 0.0, None)
        total = np.bincount(sector, weights=weight, minlength=sectors)
        peak = np.zeros(sectors)
        np.maximum.at(peak, sector, weight)
        offsets = np.concatenate([part[3] for part in parts])
        inner = np.full(sectors, np.inf)
        outer = np.full(sectors, -np.inf)
        np.minimum.at(inner, sector, offsets)
        np.maximum.at(outer, sector, offsets)
        complete = filled & (inner <= -half_width + 1) & (outer >= half_width - 1)
        typical = float(np.median(peak[complete])) if complete.any() else 0.0
        use = complete & (total > 0) & (peak > 0) & (peak >= PEAK_SHARE * typical)
        if not use.any():
            continue
        mean_x = np.bincount(sector, weights=weight * xs, minlength=sectors)[use] / total[use]
        mean_y = np.bincount(sector, weights=weight * ys, minlength=sectors)[use] / total[use]
        points_x.append(mean_x)
        points_y.append(mean_y)
        point_q.append(np.full(mean_x.size, q, dtype=float))
    if not points_x:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(points_x), np.concatenate(points_y), np.concatenate(point_q)


def fit_tilted_geometry(
    xs, ys, qs, wavelength: float, px: float, py: float, initial, *, rot3: float = 0.0, fixed_tilt=None,
    distance_range_mm: Optional[tuple[float, float]] = None, center_window_px: float = 60.0,
) -> np.ndarray:
    """Fit centre and distance, and both tilt angles unless supplied in radians as fixed_tilt.
    The initial and returned vectors contain the free parameters only."""
    start = np.asarray(initial, dtype=float).copy()
    limit = math.radians(TILT_LIMIT_DEG)
    low_mm, high_mm = distance_range_mm or (1.0, np.inf)
    lower = np.array([start[0] - center_window_px, start[1] - center_window_px,
                      max(1.0, low_mm, 0.75 * start[2]), -limit, -limit])
    upper = np.array([start[0] + center_window_px, start[1] + center_window_px,
                      min(high_mm, 1.25 * start[2]), limit, limit])
    if not lower[2] < upper[2]:
        lower[2], upper[2] = start[2] - 1e-6, start[2] + 1e-6
    lower, upper = lower[:start.size], upper[:start.size]
    start = np.clip(start, lower + 1e-9, upper - 1e-9)
    residual = tilt_residual(xs, ys, qs, wavelength, px, py, rot3, fixed_tilt=fixed_tilt)
    fit = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1", f_scale=2.0, max_nfev=200,
                        x_scale=np.array([1.0, 1.0, max(1.0, 1e-3 * start[2]), 1e-3, 1e-3])[:start.size])
    return fit.x


def point_rms_px(xs, ys, qs, params, wavelength: float, px: float, py: float, rot3: float = 0.0) -> float:
    """Root mean square of the ring-point residuals (pixels) of the geometry ``params`` (3 or 5 values)."""
    values = np.asarray(params, dtype=float)
    full = np.r_[values, 0.0, 0.0] if values.size == 3 else values
    residual = tilt_residual(xs, ys, qs, wavelength, px, py, rot3)(full)
    return float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")


def refine_ring_geometry(
    analysis: AnalysisImage, candidate: CalibrationCandidate, wavelength: float, px: float, py: float, *,
    tilt: str, distance_range_mm: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Refine the candidate in place and return the ring points that determined it."""
    minimum = 2 if tilt == "on" else TILT_MIN_RINGS
    can_refine = tilt != "off" and candidate.matched_ring_count >= minimum
    rotations = np.radians([candidate.rot1_deg, candidate.rot2_deg])
    fixed_rotations = rotations if tilt == "off" else np.zeros(2)
    # A guided refit can lose a ring at a detector edge. Its earlier tilt and
    # uncertainties no longer describe the points available to this call.
    candidate.clear_fit_statistics()
    candidate.tilt_refined = False
    candidate.rot1_deg, candidate.rot2_deg = np.degrees(fixed_rotations)
    point_warnings = (NOT_REFINED_FEW_RINGS, NOT_REFINED_POINTS, NOT_REFINED_CONTRIBUTING_RINGS,
                      _TOO_FEW_GEOMETRY_POINTS)
    candidate.warnings[:] = [message for message in candidate.warnings if message not in point_warnings]
    if tilt != "off" and not can_refine:
        candidate.warnings.append(NOT_REFINED_FEW_RINGS)
    if candidate.matched_ring_count < 2:
        return None

    rot3 = math.radians(candidate.rot3_deg)
    initial = np.r_[candidate.center_x_px, candidate.center_y_px, candidate.distance_mm,
                    rotations if can_refine else fixed_rotations]
    params = initial.copy()
    window = max(12.0, min(60.0, 0.04 * max(analysis.signal.shape)))
    options = dict(rot3=rot3, distance_range_mm=distance_range_mm, center_window_px=window)
    for passed in range(POINT_PASSES):
        points = tilted_ring_points(analysis, candidate, params, wavelength, px, py, rot3,
                                    widen=2.0 if passed == 0 else 1.0)
        xs, ys, qs = points
        contributing = np.unique(qs).size
        if xs.size < 30 or contributing < 2:
            candidate.warnings.append(NOT_REFINED_POINTS if can_refine else _TOO_FEW_GEOMETRY_POINTS)
            return None
        free_tilt = can_refine and contributing >= minimum
        free = params if free_tilt else params[:3]
        fit = fit_tilted_geometry(*points, wavelength, px, py, free,
                                  fixed_tilt=None if free_tilt else fixed_rotations, **options)
        updated = fit if free_tilt else np.r_[fit, fixed_rotations]
        moved = np.abs(updated - params)
        params = updated
        if moved[0] < 0.01 and moved[1] < 0.01 and moved[3] < 1e-6 and moved[4] < 1e-6:
            break
    candidate.center_x_px, candidate.center_y_px, candidate.distance_mm = map(float, params[:3])
    candidate.rot1_deg, candidate.rot2_deg = np.degrees(params[3:])
    candidate.tilt_refined = free_tilt
    if can_refine and not free_tilt:
        candidate.warnings.append(NOT_REFINED_CONTRIBUTING_RINGS)
    candidate.initial_point_rms_px = point_rms_px(*points, initial, wavelength, px, py, rot3)
    candidate.point_rms_px = point_rms_px(*points, params, wavelength, px, py, rot3)
    if free_tilt:
        # Isolate the effect of tilt: centre and distance may adjust in both models on these exact points.
        flat = fit_tilted_geometry(*points, wavelength, px, py, params[:3], fixed_tilt=(0.0, 0.0), **options)
        candidate.untilted_point_rms_px = point_rms_px(*points, flat, wavelength, px, py, rot3)
    return points


__all__ = [
    "NOT_REFINED_FEW_RINGS", "NOT_REFINED_POINTS", "TILT_MIN_RINGS", "fit_tilted_geometry", "point_rms_px",
    "refine_ring_geometry", "tangent_two_theta", "tilt_residual", "tilted_ring_points",
]
