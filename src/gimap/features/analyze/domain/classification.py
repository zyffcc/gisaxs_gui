"""Decide whether a frame is small-angle (GISAXS) or wide-angle (GIWAXS), or a calibrant in transmission."""

from __future__ import annotations

import math
import re
from pathlib import Path

from src.gimap.shared.geometry import DetectorGeometry, lab_rays

from .models import GISAXS, GIWAXS

WIDE_ANGLE_LIMIT_DEG = 20.0
"""A detector reaching beyond this scattering angle is treated as GIWAXS.

GIWAXS set-ups (D of 0.1–0.35 m) reach about 30° and more at the far corner.
GISAXS set-ups stay well below that even with a large detector close by: a
Pilatus 2M at 1.46 m with the beam near the bottom reaches about 10°.  The
user can always choose the mode; this only sets the automatic guess.
"""


def max_scattering_angle_deg(shape: tuple[int, int], geometry: DetectorGeometry) -> float:
    """Largest 2θ over the frame's corners (where it lies for a detector perpendicular to the beam)."""
    rows, columns = int(shape[0]), int(shape[1])
    largest = 0.0
    for corner_x in (0.0, float(columns)):
        for corner_y in (0.0, float(rows)):
            dx = (corner_x - geometry.beam_center_x_px) * geometry.pixel_size_x_m
            dy = (geometry.beam_center_y_px - corner_y) * geometry.pixel_size_y_m
            along, right, up = (float(v) for v in lab_rays(geometry, dx, dy))
            largest = max(largest, math.degrees(math.atan2(math.hypot(right, up), along)))
    return largest


def classify_measurement(shape: tuple[int, int], geometry: DetectorGeometry) -> str:
    return GIWAXS if max_scattering_angle_deg(shape, geometry) > WIDE_ANGLE_LIMIT_DEG else GISAXS


CALIBRANT_TOKENS = frozenset({"agbh", "agbe", "behenate", "lab6", "ceo2"})
"""Words in a file name that name a calibration standard measured in transmission."""


def calibrant_name(path) -> bool:
    """Whether the file name names a calibration standard (silver behenate, LaB6, CeO2)."""
    tokens = re.split(r"[^a-z0-9]+", Path(str(path)).name.lower())
    return any(token in CALIBRANT_TOKENS for token in tokens)


__all__ = ["CALIBRANT_TOKENS", "WIDE_ANGLE_LIMIT_DEG", "calibrant_name", "classify_measurement", "max_scattering_angle_deg"]
