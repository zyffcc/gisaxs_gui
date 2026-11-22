"""Position of the specular rod and the default column band for I(qz)."""

from __future__ import annotations

import math

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import positions_of_q, qz_of_alpha_f


SPECULAR_EXCLUSION_PX = 15
"""Clearance in detector columns. The Yoneda search uses the same exclusion beside the rod."""
LIMITED_CLEARANCE = "The frame is too narrow for the default vertical band to clear the specular rod by 15 columns."


def rod_columns(geometry: DetectorGeometry, rows, columns):
    """Horizontal extent of qy = 0 within this rectangle, above the horizon (αc with refraction).

    The plane of incidence intersects a flat detector along a straight line through the direct beam.
    Its visible segment ends at a detector edge or at the lower exit-angle limit.
    """
    top, bottom = rows
    left, right = columns
    cx, cy = geometry.beam_center_x_px, geometry.beam_center_y_px
    if not geometry.grazing_requires_projection:
        if left <= cx <= right and top <= geometry.row_for_exit_angle(geometry.critical_angle_deg):
            return cx, cx
        return None
    along_x, along_up = geometry.rolled_detector_axes()
    # The rolled transverse component vanishes on the rod.
    a = along_x[1] * geometry.pixel_size_x_m
    b = -along_up[1] * geometry.pixel_size_y_m
    ends = []
    if a:
        ends.extend((cx - (y - cy) * b / a, y) for y in (top, bottom))
    if b:
        ends.extend((x, cy - (x - cx) * a / b) for x in (left, right))
    xs = [x for x, y in ends if left <= x <= right and top <= y <= bottom
          and geometry.exit_angle_at(x, y) >= geometry.critical_rad]
    vacuum = geometry.with_critical_angle(0.0)
    x, y = positions_of_q(vacuum, 0.0, qz_of_alpha_f(vacuum, geometry.critical_angle_deg))
    if left <= x <= right and top <= y <= bottom:
        xs.append(float(x))
    return (min(xs), max(xs)) if xs else None


def default_vertical_column(geometry: DetectorGeometry, shape, half_width: float) -> tuple[float, bool]:
    """Centre a full band outside the rod's column range, preferring the right side.

    If neither side fits, choose the edge with more room and report the reduced clearance.
    """
    rows, columns = shape
    rod = rod_columns(geometry, (0, rows), (0, columns))
    low, high = rod or (geometry.beam_center_x_px, geometry.beam_center_x_px)
    right = max(0, math.ceil(high + SPECULAR_EXCLUSION_PX))
    left = min(columns, math.floor(low - SPECULAR_EXCLUSION_PX))
    if right + 2 * half_width <= columns:
        return right + half_width, True
    if left - 2 * half_width >= 0:
        return left - half_width, True
    inset = min(half_width, 0.5 * columns)
    return (columns - inset if columns - high >= low else inset), rod is None
