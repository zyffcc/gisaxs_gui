"""Preprocessing a person asks for: masks drawn on the image, and gaps filled from the mirror side.

Pixel shapes are drawn in canonical pixel coordinates (x to the right, y down, the
corner of pixel (0, 0) at the origin); a pixel is masked when its centre lies inside a shape.

Mirror filling reflects a detector ray in the sample's plane of incidence. This reverses q∥
and preserves qz and |q|. At zero roll on a perpendicular detector the partner lies in the
same row at x → 2·x_c − x. Interpolation requires valid measured neighbours. Partners outside
the frame or without data leave the gap empty.
"""

from __future__ import annotations

import numpy as np

from .mask_edits import POLYGON, RECTANGLE, MaskShape
from .mask_raster import rasterize
from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.detector_geometry import roll_transverse
from src.gimap.shared.geometry.q_inverse import positions_of_rays
from src.gimap.shared.geometry.q_mapping import lab_rays


def mirror_fill(data: np.ndarray, valid: np.ndarray, center_x: float,
                *, geometry: DetectorGeometry | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(data, valid, filled)`` with invalid pixels taken from the mirror side (module docstring)."""
    data = np.asarray(data, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if geometry is not None and (geometry.tilted or geometry.sample_roll_deg):
        return _mirror_rays(data, valid, geometry)
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


def _mirror_rays(data, valid, geometry):
    """Sample at the reflected incident-plane ray, using only originally measured interpolation neighbours."""
    height, width = data.shape
    filled = np.zeros_like(valid)
    result = data.copy()
    for start in range(0, height, 128):
        i, j = np.nonzero(~valid[start:start + 128])
        if not i.size:
            continue
        i += start
        dx = (j + 0.5 - geometry.beam_center_x_px) * geometry.pixel_size_x_m
        dy = (geometry.beam_center_y_px - i - 0.5) * geometry.pixel_size_y_m
        along, right, up = lab_rays(geometry, dx, dy)
        across, rolled_up = roll_transverse(right, up, geometry.sample_roll_rad)
        right, up = roll_transverse(-across, rolled_up, -geometry.sample_roll_rad)
        x, y = positions_of_rays(geometry, np.stack([along, right, up]))
        x, y = x - 0.5, y - 0.5  # fractional indices of the mirror's measured neighbours
        # An exactly centred ray can return a few ulps either side of an integer.
        x = np.where(np.abs(x - np.rint(x)) < 1e-10, np.rint(x), x)
        y = np.where(np.abs(y - np.rint(y)) < 1e-10, np.rint(y), y)
        inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        x, y, i, j = x[inside], y[inside], i[inside], j[inside]
        left, top = np.floor(x).astype(int), np.floor(y).astype(int)
        wx, wy = x - left, y - top
        values = np.zeros(x.shape)
        usable = np.ones(x.shape, dtype=bool)
        for row, ry in ((top, 1 - wy), (np.minimum(top + 1, height - 1), wy)):
            for col, rx in ((left, 1 - wx), (np.minimum(left + 1, width - 1), wx)):
                w = rx * ry
                use = w > 0
                usable &= ~use | valid[row, col]
                values += w * np.where(use, data[row, col], 0)
        filled[i[usable], j[usable]] = True
        result[i[usable], j[usable]] = values[usable]
    return result, valid | filled, filled


__all__ = ["MaskShape", "POLYGON", "RECTANGLE", "mirror_fill", "rasterize"]
