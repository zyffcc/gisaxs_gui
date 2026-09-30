"""Decide whether a frame is small-angle (GISAXS) or wide-angle (GIWAXS)."""

from __future__ import annotations

import math

from src.gimap.shared.geometry import DetectorGeometry

from .models import GISAXS, GIWAXS

WIDE_ANGLE_LIMIT_DEG = 20.0
"""A detector reaching beyond this scattering angle is treated as GIWAXS.

GIWAXS set-ups (D of 0.1–0.35 m) reach about 30° and more at the far corner.
GISAXS set-ups stay well below that even with a large detector close by: a
Pilatus 2M at 1.46 m with the beam near the bottom reaches about 10°.  The
user can always choose the mode; this only sets the automatic guess.
"""


def max_scattering_angle_deg(shape: tuple[int, int], geometry: DetectorGeometry) -> float:
    """Largest 2θ over the frame; attained at a corner for a flat detector."""
    rows, columns = int(shape[0]), int(shape[1])
    largest = 0.0
    for corner_x in (0.0, float(columns)):
        for corner_y in (0.0, float(rows)):
            dx = (corner_x - geometry.beam_center_x_px) * geometry.pixel_size_x_m
            dy = (corner_y - geometry.beam_center_y_px) * geometry.pixel_size_y_m
            largest = max(largest, math.degrees(math.atan2(math.hypot(dx, dy), geometry.distance_m)))
    return largest


def classify_measurement(shape: tuple[int, int], geometry: DetectorGeometry) -> str:
    return GIWAXS if max_scattering_angle_deg(shape, geometry) > WIDE_ANGLE_LIMIT_DEG else GISAXS


__all__ = ["WIDE_ANGLE_LIMIT_DEG", "classify_measurement", "max_scattering_angle_deg"]
