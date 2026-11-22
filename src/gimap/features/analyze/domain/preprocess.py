"""Preprocessing a person asks for: masks drawn on the image, and gaps filled from the mirror side.

Pixel shapes are drawn in canonical pixel coordinates (x to the right, y down, the
corner of pixel (0, 0) at the origin); a pixel is masked when its centre lies inside a shape.

Mirror filling uses the left–right symmetry of a grazing-incidence pattern about the column
of the direct beam: with a flat detector perpendicular to the beam, x → 2·x_c − x maps q∥ to −q∥
and leaves qz and |q| unchanged. A pixel without data (a module gap, a masked or hot pixel)
takes the value at its mirror position — interpolated between the two columns that bracket it,
both of which must be valid — and is recorded as filled. Nothing is filled whose mirror lies
outside the detector or has no data either.
"""

from __future__ import annotations

import numpy as np

from .mask_edits import POLYGON, RECTANGLE, MaskShape
from .mask_raster import rasterize


def mirror_fill(data: np.ndarray, valid: np.ndarray, center_x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(data, valid, filled)`` with invalid pixels taken from the mirror side (module docstring)."""
    data = np.asarray(data, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    columns = data.shape[1]
    # Pixel centres j + 0.5 mirror to 2·x_c − (j + 0.5); as a fractional column index:
    mirror = 2.0 * float(center_x) - (np.arange(columns) + 0.5) - 0.5
    left = np.floor(mirror).astype(int)
    weight = (mirror - left).astype(np.float32)
    inside = (left >= 0) & (left + 1 <= columns - 1)
    exact = np.isclose(weight, 0.0)
    inside |= exact & (left >= 0) & (left <= columns - 1)
    left_c = np.clip(left, 0, columns - 1)
    right_c = np.clip(left + 1, 0, columns - 1)
    left_valid = valid[:, left_c]
    right_valid = np.where(exact[None, :], True, valid[:, right_c])
    usable = inside[None, :] & left_valid & right_valid
    fill = ~valid & usable
    if not fill.any():
        return data, valid, fill
    mirrored = (1.0 - weight)[None, :] * data[:, left_c] + weight[None, :] * np.where(exact[None, :], 0.0, data[:, right_c])
    filled_data = data.copy()
    filled_data[fill] = mirrored[fill]
    return filled_data, valid | fill, fill


__all__ = ["MaskShape", "POLYGON", "RECTANGLE", "mirror_fill", "rasterize"]
