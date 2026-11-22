"""Evaluate mask edits at detector pixel centres, in the order they were added."""

from __future__ import annotations

import math

import numpy as np

from src.gimap.shared.geometry.conventions import canonical_from_index_center
from src.gimap.shared.geometry.q_mapping import geometry_rays, transmission_q

from .mask_edits import MaskImage, MaskRule, MaskShape

_FIRST_X, _FIRST_Y = canonical_from_index_center(0, 0)


def _pixels(mask, left, right, top, bottom):
    """The image slice and the x/y coordinates of its pixel centres inside inclusive bounds."""
    j0, j1 = max(0, math.ceil(left - _FIRST_X)), min(mask.shape[1], math.floor(right - _FIRST_X) + 1)
    i0, i1 = max(0, math.ceil(top - _FIRST_Y)), min(mask.shape[0], math.floor(bottom - _FIRST_Y) + 1)
    # A negative slice stop would count from the opposite edge.
    pixels = mask[i0:max(i1, i0), j0:max(j1, j0)]
    return pixels, np.arange(j0, j1)[None, :] + _FIRST_X, np.arange(i0, i1)[:, None] + _FIRST_Y


def _stroke(mask, points, radius, exclude):
    """Paint a disk swept along each segment, including both endpoints and single clicks."""
    for start, end in zip(points, points[1:] or points):
        x0, y0 = start
        x1, y1 = end
        pixels, x, y = _pixels(mask, min(x0, x1) - radius, max(x0, x1) + radius,
                               min(y0, y1) - radius, max(y0, y1) + radius)
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / length2, 0, 1) if length2 else 0.0
        pixels[(x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2 <= radius**2] = exclude


def _q_ring(mask, limits, exclude, geometry):
    if geometry is None:
        raise ValueError("A beam-centred q annulus needs detector geometry.")
    low, high = limits
    rows, columns = mask.shape
    for start in range(0, rows, 128):
        stop = min(start + 128, rows)
        rays = geometry_rays(geometry, (start, stop), (0, columns))
        q, _angle = transmission_q(*rays, geometry.wavevector_inv_angstrom)
        mask[start:stop][(low <= q) & (q <= high)] = exclude


def _paint(mask: np.ndarray, shape: MaskShape, geometry) -> None:
    exclude = shape.mode == "exclude"
    points = shape.points
    if shape.kind in ("brush", "eraser"):
        _stroke(mask, points, shape.radius, exclude)
        return
    if shape.kind == "q_annulus":
        _q_ring(mask, points[0], exclude, geometry)
        return
    if shape.kind in ("circle", "annulus"):
        cx, cy = points[0]
        outer2 = (cx - points[-1][0])**2 + (cy - points[-1][1])**2
        radius = math.sqrt(outer2)
        pixels, x, y = _pixels(mask, cx - radius, cx + radius, cy - radius, cy + radius)
        r2 = (x - cx)**2 + (y - cy)**2
        inside = r2 <= outer2
        if shape.kind == "annulus":
            inside &= r2 >= (cx - points[1][0])**2 + (cy - points[1][1])**2
        pixels[inside] = exclude
        return
    xs, ys = zip(*points)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    pixels, x, y = _pixels(mask, left, right, top, bottom)
    if shape.kind == "rectangle":
        pixels[:] = exclude
    elif shape.kind == "ellipse":
        u = (2 * x - left - right) / (right - left)
        v = (2 * y - top - bottom) / (bottom - top)
        pixels[u**2 + v**2 <= 1.0] = exclude
    elif pixels.size:
        from matplotlib.path import Path

        centres = np.column_stack([grid.ravel() for grid in np.broadcast_arrays(x, y)])
        pixels[Path(points).contains_points(centres).reshape(pixels.shape)] = exclude


def rasterize(shapes, frame_shape: tuple[int, int], geometry=None) -> np.ndarray:
    """The pixels excluded by a sequence of drawn shapes on a blank detector mask."""
    mask = np.zeros(frame_shape, dtype=bool)
    for shape in shapes:
        _paint(mask, shape, geometry)
    return mask


def _morphology(mask, kind, radius):
    if radius == 0 or not mask.any():
        return
    from scipy.ndimage import distance_transform_edt

    if kind == "grow":
        mask[:] = distance_transform_edt(~mask) <= radius
    else:
        # The exterior of the image belongs to the background; pad all four edges equally.
        distance = distance_transform_edt(np.pad(mask, 1, constant_values=False))
        mask[:] = distance[1:-1, 1:-1] > radius


def _rule(mask, rule, data, valid):
    if rule.kind in ("grow", "shrink"):
        _morphology(mask, rule.kind, rule.value)
    elif rule.kind == "invert":
        np.logical_not(mask, out=mask)
    elif rule.kind == "clear":
        mask[:] = False
    else:
        value = rule.value
        if rule.relative:
            sample = data[valid & ~mask & np.isfinite(data)]
            if sample.size == 0:
                raise ValueError("No unmasked valid pixels remain for the median.")
            lower, upper = (sample.size - 1) // 2, sample.size // 2
            sample.partition((lower, upper))
            value *= (float(sample[lower]) + float(sample[upper])) / 2
        # NumPy 1.x narrows a scalar threshold to the detector dtype. An array operand
        # preserves both the entered precision and a median between two float32 values.
        threshold = np.array([value], dtype=np.float64) if value is not None else None
        with np.errstate(invalid="ignore"):
            if rule.kind == "above":
                selected = data > threshold
            elif rule.kind == "below":
                selected = data < threshold
            elif rule.kind == "saturated":
                selected = data >= threshold
            else:
                selected = data < 0
        mask |= np.isfinite(data) & selected


def evaluate_mask(edits, data, valid, *, geometry=None, initial=None, read_image=None) -> np.ndarray:
    """Replay user edits on the raw (possibly summed) detector frame. Never mutate a loaded mask."""
    mask = np.zeros(data.shape, dtype=bool) if initial is None else np.array(initial, dtype=bool, copy=True)
    for edit in edits:
        if isinstance(edit, MaskRule):
            _rule(mask, edit, data, valid)
        elif isinstance(edit, MaskImage):
            mask |= read_image(edit.path, mask.shape)
        else:
            _paint(mask, edit, geometry)
    return mask
