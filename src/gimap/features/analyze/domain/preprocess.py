"""Preprocessing a person asks for: masks drawn on the image, and gaps filled from the mirror side.

Masks are rectangles and polygons in canonical pixel coordinates (x to the right, y down, the
corner of pixel (0, 0) at the origin); a pixel is masked when its centre lies inside a shape.

Mirror filling uses the left–right symmetry of a grazing-incidence pattern about the column
of the direct beam: with a flat detector perpendicular to the beam, x → 2·x_c − x maps q∥ to −q∥
and leaves qz and |q| unchanged. A pixel without data (a module gap, a masked or hot pixel)
takes the value at its mirror position — interpolated between the two columns that bracket it,
both of which must be valid — and is recorded as filled. Nothing is filled whose mirror lies
outside the detector or has no data either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import positions_of_rays
from src.gimap.shared.geometry.q_mapping import lab_rays

RECTANGLE = "rectangle"
POLYGON = "polygon"


@dataclass(frozen=True)
class MaskShape:
    kind: str
    """``rectangle`` (two corners) or ``polygon`` (three or more vertices)."""
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = tuple((float(x), float(y)) for x, y in self.points)
        object.__setattr__(self, "points", points)
        if self.kind == RECTANGLE and len(points) != 2:
            raise ValueError("A rectangle mask needs two corners.")
        if self.kind == POLYGON and len(points) < 3:
            raise ValueError("A polygon mask needs at least three vertices.")
        if self.kind not in (RECTANGLE, POLYGON):
            raise ValueError(f"Unknown mask shape {self.kind!r}.")

    def describe(self) -> str:
        xs = [x for x, _y in self.points]
        ys = [y for _x, y in self.points]
        what = "Rectangle" if self.kind == RECTANGLE else f"Polygon ({len(self.points)} points)"
        return f"{what}: x {min(xs):.0f}–{max(xs):.0f}, y {min(ys):.0f}–{max(ys):.0f}"


def rasterize(shapes, frame_shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask (``frame_shape``) of the pixels whose centre lies in any shape."""
    from matplotlib.path import Path as Polygon

    rows, columns = int(frame_shape[0]), int(frame_shape[1])
    masked = np.zeros((rows, columns), dtype=bool)
    for shape in shapes:
        xs = np.array([x for x, _y in shape.points])
        ys = np.array([y for _x, y in shape.points])
        column0, column1 = max(0, int(np.floor(xs.min() - 0.5))), min(columns, int(np.ceil(xs.max() + 0.5)))
        row0, row1 = max(0, int(np.floor(ys.min() - 0.5))), min(rows, int(np.ceil(ys.max() + 0.5)))
        if column1 <= column0 or row1 <= row0:
            continue
        if shape.kind == RECTANGLE:
            centres_x = np.arange(column0, column1) + 0.5
            centres_y = np.arange(row0, row1) + 0.5
            inside_x = (centres_x >= xs.min()) & (centres_x <= xs.max())
            inside_y = (centres_y >= ys.min()) & (centres_y <= ys.max())
            masked[row0:row1, column0:column1] |= inside_y[:, None] & inside_x[None, :]
            continue
        grid_y, grid_x = np.mgrid[row0:row1, column0:column1]
        centres = np.column_stack([grid_x.ravel() + 0.5, grid_y.ravel() + 0.5])
        inside = Polygon(np.column_stack([xs, ys])).contains_points(centres)
        masked[row0:row1, column0:column1] |= inside.reshape(row1 - row0, column1 - column0)
    return masked


def mirror_fill(data: np.ndarray, valid: np.ndarray, center_x: float,
                *, geometry: DetectorGeometry | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(data, valid, filled)`` with invalid pixels taken from the mirror side (module docstring)."""
    data = np.asarray(data, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if geometry is not None and geometry.tilted:
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
        x, y = positions_of_rays(geometry, np.stack([along, -right, up]))
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
