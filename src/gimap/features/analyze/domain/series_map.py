"""Intensity against frame and q: the curves of a series stacked into one image.

Rows are frames (or groups of summed frames) in list order, columns a common x grid. Frames
of one geometry share their grid exactly; others are interpolated onto the widest range with
the median number of points (never extrapolated: cells outside a frame's range stay NaN).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SeriesMap:
    x: np.ndarray
    """Common x grid of every row (the unit is in ``x_label``)."""
    image: np.ndarray
    """``(rows, len(x))`` intensities; NaN where a frame has no data."""
    labels: tuple[str, ...]
    """What each row is (file, frame number, summed frames)."""
    x_label: str
    curve: str
    """Key of the curve stacked (``radial``, ``horizontal``, ...)."""
    refs: tuple = field(default=())
    """What opens each row again (``(path, frame_index)``), in row order."""

    @property
    def rows(self) -> int:
        return int(self.image.shape[0])

    def profile(self, row: int) -> np.ndarray:
        """The curve of one row on the common grid."""
        return self.image[int(np.clip(row, 0, self.rows - 1))]

    def most_changing_x(self) -> float:
        """The x where the last quarter of the rows differs most from the first, in units of its noise.

        A t statistic per column (mean difference over the pooled standard error), so a bright
        but steady background does not win over a weak line that grows or fades.
        """
        quarter = max(1, self.rows // 4)
        first, last = self.image[:quarter].astype(float), self.image[-quarter:].astype(float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # columns without data: NaN, left out below
            spread = np.nanvar(first, axis=0) / quarter + np.nanvar(last, axis=0) / quarter
            level = np.abs(np.nanmean(np.vstack([first, last]), axis=0))
            noise = np.sqrt(np.maximum(spread, 1e-12 * np.maximum(level, 1.0) ** 2))
            score = np.abs(np.nanmean(last, axis=0) - np.nanmean(first, axis=0)) / noise
        score = np.where(np.isfinite(score), score, -np.inf)
        return float(self.x[int(np.argmax(score))]) if np.isfinite(score).any() else float(self.x[self.x.size // 2])

    def trace(self, x_value: float, half_width: Optional[float] = None) -> np.ndarray:
        """Mean intensity per row over ``x_value ± half_width`` (default: one grid step)."""
        step = float(np.nanmedian(np.abs(np.diff(self.x)))) if self.x.size > 1 else 0.0
        half = max(float(half_width) if half_width is not None else step, 0.5 * step)
        window = np.abs(self.x - float(x_value)) <= half
        if not window.any():
            window[int(np.argmin(np.abs(self.x - float(x_value))))] = True
        with np.errstate(all="ignore"):
            cells = self.image[:, window]
            counts = np.isfinite(cells).sum(axis=1)
            total = np.nansum(cells, axis=1)
            return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


@dataclass(frozen=True)
class PeakTrack:
    """A peak followed through the rows of a series map inside one x window (NaN: no peak there)."""

    window: tuple[float, float]
    position: np.ndarray
    """Centroid of the intensity above the local background (x unit of the map)."""
    fwhm: np.ndarray
    """2.355 × the standard deviation of that intensity (a Gaussian's FWHM)."""
    area: np.ndarray
    """Integrated intensity above the background (intensity × x unit)."""
    height: np.ndarray
    """Largest value above the background."""

    def table(self) -> list[tuple[str, np.ndarray]]:
        return [("position", self.position), ("fwhm", self.fwhm), ("area", self.area), ("height", self.height)]


def track_peak(series: SeriesMap, low: float, high: float) -> PeakTrack:
    """Centroid, width, area and height of the peak in ``[low, high]`` for every row.

    The background is the straight line through the mean of the first and of the last three
    points of the window, so a sloping background does not move the centroid. Rows with less
    than five points in the window, or nothing above the background, give NaN.
    """
    low, high = sorted((float(low), float(high)))
    window = (series.x >= low) & (series.x <= high)
    x = series.x[window]
    rows = series.rows
    position, fwhm, area, height = (np.full(rows, np.nan) for _ in range(4))
    if x.size < 5:
        return PeakTrack((low, high), position, fwhm, area, height)
    edge = max(1, min(3, x.size // 5))
    step = float(np.nanmedian(np.abs(np.diff(x)))) if x.size > 1 else 0.0
    for row in range(rows):
        y = series.image[row, window].astype(float)
        finite = np.isfinite(y)
        if finite.sum() < 5:
            continue
        xs, ys = x[finite], y[finite]
        left_x, left_y = float(np.mean(xs[:edge])), float(np.mean(ys[:edge]))
        right_x, right_y = float(np.mean(xs[-edge:])), float(np.mean(ys[-edge:]))
        slope = (right_y - left_y) / (right_x - left_x) if right_x > left_x else 0.0
        net = ys - (left_y + slope * (xs - left_x))
        positive = np.clip(net, 0.0, None)
        total = float(positive.sum())
        if total <= 0:
            continue
        centre = float((xs * positive).sum() / total)
        spread = float(np.sqrt(max(((xs - centre) ** 2 * positive).sum() / total, 0.0)))
        position[row], fwhm[row] = centre, 2.3548 * spread
        area[row], height[row] = float(net.sum()) * step, float(net.max())
    return PeakTrack((low, high), position, fwhm, area, height)


def _sorted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def stack_curves(curves: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """``(x, image)``: the curves on one grid, one row each (see the module docstring)."""
    if not curves:
        raise ValueError("No curves to stack.")
    first_x = np.asarray(curves[0][0], dtype=float)
    if all(np.asarray(x).shape == first_x.shape and np.allclose(np.asarray(x, dtype=float), first_x, equal_nan=True)
           for x, _y in curves):
        image = np.vstack([np.asarray(y, dtype=np.float32) for _x, y in curves])
        order = np.argsort(first_x, kind="stable")
        return first_x[order], image[:, order]
    sorted_curves = [_sorted(x, y) for x, y in curves]
    sizes = [x.size for x, _y in sorted_curves if x.size]
    if not sizes:
        raise ValueError("The curves have no points.")
    low = min(float(x[0]) for x, _y in sorted_curves if x.size)
    high = max(float(x[-1]) for x, _y in sorted_curves if x.size)
    grid = np.linspace(low, high, int(np.median(sizes)))
    image = np.full((len(curves), grid.size), np.nan, dtype=np.float32)
    for row, (x, y) in enumerate(sorted_curves):
        if x.size >= 2:
            image[row] = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return grid, image


def series_map(
    curves: Sequence[tuple[np.ndarray, np.ndarray]], labels: Sequence[str], *, x_label: str, curve: str, refs=(),
) -> SeriesMap:
    x, image = stack_curves(curves)
    return SeriesMap(x, image, tuple(labels), x_label, curve, tuple(refs))


__all__ = ["PeakTrack", "SeriesMap", "series_map", "stack_curves", "track_peak"]
