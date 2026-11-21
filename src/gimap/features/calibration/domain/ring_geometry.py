"""Pure geometry used to present and manually refine calibration rings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import CalibrationCandidate, CalibrationResult
from .scientific_kernel import STANDARDS, distance_from_ring_radius, q_to_ring_radius_m
from .tilt_fit import _geometry as tilt_geometry, tangent_two_theta


@dataclass(frozen=True)
class TheoreticalRingOverlay:
    width_px: float
    height_px: float
    matched: bool


def theoretical_ring_overlays(
    candidate: CalibrationCandidate,
    result: CalibrationResult,
) -> tuple[TheoreticalRingOverlay, ...]:
    standard = STANDARDS.get(candidate.standard_key)
    if standard is None:
        return ()
    radii_m = q_to_ring_radius_m(
        standard.q_values_inv_angstrom,
        result.wavelength_angstrom,
        candidate.distance_mm,
    )
    matched_indices = {match.theoretical_index for match in candidate.matched_rings}
    return tuple(
        TheoreticalRingOverlay(
            width_px=float(2.0 * radius_m / result.pixel_size_x_m),
            height_px=float(2.0 * radius_m / result.pixel_size_y_m),
            matched=index in matched_indices,
        )
        for index, radius_m in enumerate(radii_m)
    )


@dataclass(frozen=True)
class TiltedRingGrid:
    """``tan 2θ`` of a coarse grid of detector positions (``xs``, ``ys`` in GIMaP's pixel-corner coordinates) for a
    tilted candidate, and the ``tan 2θ`` of each line of its standard: the rings are the contours of ``values``."""

    xs: np.ndarray
    ys: np.ndarray
    values: np.ndarray
    levels: tuple[float, ...]
    matched: tuple[bool, ...]


def tilted_ring_grid(
    candidate: CalibrationCandidate, result: CalibrationResult, shape: tuple[int, int], points: int = 360,
) -> TiltedRingGrid | None:
    """The rings of a tilted candidate as contours (``None`` for an untilted one: its rings are the ellipses of
    ``theoretical_ring_overlays``)."""
    standard = STANDARDS.get(candidate.standard_key)
    if standard is None or not (candidate.rot1_deg or candidate.rot2_deg):
        return None
    params = np.array([candidate.center_x_px, candidate.center_y_px, candidate.distance_mm,
                       math.radians(candidate.rot1_deg), math.radians(candidate.rot2_deg)])
    geometry = tilt_geometry(params, result.wavelength_angstrom, result.pixel_size_x_m, result.pixel_size_y_m,
                             math.radians(candidate.rot3_deg))
    rows, columns = int(shape[0]), int(shape[1])
    step = max(1.0, max(rows, columns) / float(points))
    xs, ys = np.arange(0.0, columns + step, step), np.arange(0.0, rows + step, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    values = tangent_two_theta(geometry, grid_x - 0.5, grid_y - 0.5, candidate.center_x_px, candidate.center_y_px)
    matched_indices = {match.theoretical_index for match in candidate.matched_rings}
    rings = []
    for index, q in enumerate(standard.q_values_inv_angstrom):
        ratio = q * result.wavelength_angstrom / (4 * math.pi)
        if 0 < ratio < math.sin(math.pi / 4):
            rings.append((math.tan(2 * math.asin(ratio)), index in matched_indices))
    rings.sort()
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    kept = []
    for level, matched in rings:  # strictly increasing, and on the frame (contour levels)
        if low < level < high and (not kept or level > kept[-1][0] * (1 + 1e-9)):
            kept.append((level, matched))
    rings = kept
    return TiltedRingGrid(xs, ys, values, tuple(level for level, _ in rings), tuple(flag for _, flag in rings))


def manual_ring_distance(
    result: CalibrationResult,
    experimental_radius_px: float,
    theoretical_q_inv_angstrom: float,
) -> float:
    return distance_from_ring_radius(
        experimental_radius_px,
        theoretical_q_inv_angstrom,
        result.wavelength_angstrom,
        0.5 * (result.pixel_size_x_m + result.pixel_size_y_m),
    )
