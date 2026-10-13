"""How well a geometry puts a standard's powder lines at their q: the quality that matters.

The ring-fit residual in pixels depends on how many rings are fitted and on
what the flat detector model cannot describe (a tilt): on a wide-angle
detector a fit with more rings can have a larger residual and still give the
better q calibration.  This check integrates the calibration image with the
geometry and measures where each line of the standard actually lands.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

BIN_WIDTH = 0.0015
WINDOW = 0.025
MIN_PIXELS_PER_BIN = 20
MIN_SIGNIFICANCE = 5.0


def _gaussian_line(x, amplitude, center, width, offset, slope):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2) + offset + slope * (x - center)


def check_lines(
    data: np.ndarray,
    valid: np.ndarray,
    *,
    center_x_px: float,
    center_y_px: float,
    distance_mm: float,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    wavelength_angstrom: float,
    q_lines: Sequence[float],
    bin_width: float = BIN_WIDTH,
    window: float = WINDOW,
) -> dict:
    """Measured minus expected q of every line visible on the image (canonical centre, flat detector)."""
    rows, columns = data.shape[:2]
    x = ((np.arange(columns, dtype=np.float32) + 0.5 - np.float32(center_x_px)) * np.float32(pixel_size_x_m)) ** 2
    y = ((np.arange(rows, dtype=np.float32) + 0.5 - np.float32(center_y_px)) * np.float32(pixel_size_y_m)) ** 2
    radius = np.sqrt(y[:, None] + x[None, :])
    q = (4.0 * math.pi / wavelength_angstrom) * np.sin(0.5 * np.arctan2(radius, np.float32(distance_mm * 1e-3)))
    keep = valid & np.isfinite(data)
    q_values, intensity = q[keep], np.asarray(data, dtype=np.float64)[keep]
    if q_values.size == 0:
        return {"lines_checked": 0, "lines": []}
    edges = np.arange(float(q_values.min()), float(q_values.max()) + bin_width, bin_width)
    counts, _ = np.histogram(q_values, edges)
    sums, _ = np.histogram(q_values, edges, weights=intensity)
    centers = 0.5 * (edges[1:] + edges[:-1])
    profile = np.where(counts >= MIN_PIXELS_PER_BIN, sums / np.maximum(counts, 1), np.nan)
    measured = []
    for line in q_lines:
        inside = np.isfinite(profile) & (np.abs(centers - line) < window)
        if inside.sum() < 12:
            continue
        xs, ys = centers[inside], profile[inside]
        start = (float(ys.max() - np.median(ys)), float(xs[np.argmax(ys)]), 0.004, float(np.median(ys)), 0.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)  # only the parameters are used, not their errors
                params, _ = curve_fit(_gaussian_line, xs, ys, p0=start, maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        amplitude, center, width = params[0], params[1], abs(params[2])
        outside = np.abs(xs - center) > 3 * width
        noise = float(np.std(ys[outside])) if outside.sum() > 4 else float(np.std(ys))
        if amplitude > MIN_SIGNIFICANCE * noise and abs(center - line) < 0.8 * window and 0.0008 < width < 0.015:
            measured.append((float(line), float(center - line)))
    if not measured:
        return {"lines_checked": 0, "lines": []}
    errors = np.array([abs(delta) for _line, delta in measured])
    relative = np.array([abs(delta) / line for line, delta in measured])
    return {
        "lines_checked": len(measured),
        "mean_abs_dq": float(errors.mean()),
        "max_abs_dq": float(errors.max()),
        "mean_relative": float(relative.mean()),
        "lines": [{"q": line, "dq": delta} for line, delta in measured],
    }


def best_by_lines(checks: Sequence[Optional[dict]], minimum_lines: int = 3) -> Optional[int]:
    """Index of the check with the smallest mean relative q error among those with enough lines."""
    usable = [
        (check["mean_relative"], index)
        for index, check in enumerate(checks)
        if check and check.get("lines_checked", 0) >= minimum_lines
    ]
    return min(usable)[1] if usable else None


__all__ = ["best_by_lines", "check_lines"]
