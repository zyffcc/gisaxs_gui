"""Mean-per-bin accumulation shared by every Analyze reduction."""

from __future__ import annotations

import math

import numpy as np

from .curve_errors import PixelErrors, curve_sigma, pixel_errors, scatter_sigma


def counts_and_scale(counts, mask) -> tuple[bool, np.ndarray | None]:
    """Return the counting flag and masked scale for a legacy bool or scale-map input."""
    if isinstance(counts, np.ndarray):
        return True, counts[mask]
    return bool(counts), None


class BinnedMean:
    """Mean of ``values`` per bin of ``x`` over fixed edges, fed in chunks.

    Values exactly on the last edge fall into the last bin. Values outside the
    edges and nonfinite values are ignored. Accumulated moments and variances
    support the error models for measured and merged points in Physics Manual §6.
    """

    def __init__(self, edges: np.ndarray):
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("Bin edges must be a strictly increasing 1D array.")
        self.edges = edges
        size = edges.size - 1
        self._sums = np.zeros(size, dtype=np.float64)
        self._squares = np.zeros(size, dtype=np.float64)
        self._counts = np.zeros(size, dtype=np.int64)
        self._variances: np.ndarray | None = None
        """Sum of sample shot variances or of variances from a supplied σ frame."""
        self._extra = np.zeros(size, dtype=np.float64)
        self._scales: np.ndarray | None = None
        """Σ scale per bin (1 for a value added without one), kept with ``_variances``: the variance of one
        count at the bin's mean factor, (Σ scale / n)², is the least variance a bin of counts is given."""

    @classmethod
    def linear(cls, low: float, high: float, bins: int) -> "BinnedMean":
        low, high, bins = float(low), float(high), max(1, int(bins))
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("Bin range must be finite.")
        if high <= low:
            half = max(abs(low) * 1e-9, 1e-12)
            low, high = low - half, low + half
        return cls(np.linspace(low, high, bins + 1))

    def add(self, x, values, scale=None, *, errors: PixelErrors | None = None) -> None:
        """Accumulate this chunk and its variance terms. The model is selected in result()."""
        x = np.asarray(x, dtype=np.float64).ravel()
        values = np.asarray(values, dtype=np.float64).ravel()
        if x.shape != values.shape:
            raise ValueError("x and values must have the same number of elements.")
        errors = errors if errors is not None else PixelErrors(scale=scale)
        size = self._counts.size
        index = np.searchsorted(self.edges, x, side="right") - 1
        index[x == self.edges[-1]] = size - 1
        keep = np.isfinite(x) & np.isfinite(values) & (index >= 0) & (index < size)
        index = index[keep]
        added = np.bincount(index, minlength=size)

        def selected(array):
            return np.broadcast_to(np.asarray(array, dtype=float).ravel(), values.shape)[keep]

        def bin_sum(array):
            return np.bincount(index, weights=array, minlength=size)

        observed = values[keep]
        if errors.scale is not None or errors.variance is not None or self._variances is not None:
            if self._variances is None:
                self._variances = self._sums.copy()
                self._scales = self._counts.astype(float)
            scale = 1.0 if errors.scale is None else selected(errors.scale)
            variance = observed * scale if errors.variance is None else selected(errors.variance)
            self._variances += bin_sum(variance)
            self._scales += added if errors.scale is None else bin_sum(scale)
        if errors.extra_variance is not None:
            self._extra += bin_sum(selected(errors.extra_variance))
        self._sums += bin_sum(observed)
        self._squares += bin_sum(observed ** 2)
        self._counts += added

    def result(self, *, counts=True, merge=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Centres, means, errors and occupancies. Pass the frame's PixelErrors as counts."""
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        ids = np.flatnonzero(self._counts)
        x, pixels = centres[ids], self._counts[ids]
        starts = _sparse_starts(x, pixels) if merge else None
        errors = pixel_errors(counts)

        def totals(array):
            if array is None:
                return None
            filled = array[ids]
            return np.add.reduceat(filled, starts) if starts is not None else filled

        breaks = np.r_[True, np.diff(ids) != 1][:len(ids)]
        independent = None
        if starts is not None:
            # Keep the measured errors of the constituent bins for ordinary
            # scatter and observed Poisson. A radial intensity gradient between
            # two neighbouring bins is not pixel noise inside either bin.
            if errors.model == "scatter":
                raw_sigma = scatter_sigma(self._sums[ids], self._squares[ids], pixels, pooled=False)
            elif errors.model == "poisson" and errors.poisson_variance == "observed":
                raw_sigma = curve_sigma(errors, pixels, self._sums[ids], None,
                                        None if self._variances is None else self._variances[ids],
                                        None if self._scales is None else self._scales[ids], self._extra[ids], breaks)
            else:
                raw_sigma = None
            if raw_sigma is not None:
                independent = np.sqrt(np.add.reduceat((pixels * raw_sigma)**2, starts))
            pixels = totals(self._counts)
            x = np.add.reduceat(x * self._counts[ids], starts) / pixels
            merged_breaks = breaks[starts]
            if errors.model == "poisson" and errors.poisson_variance == "smoothed":
                # Sparse joining can bridge one empty bin. Such a joined point
                # has no contiguous smoothing window on either side.
                internal_gap = np.add.reduceat(breaks, starts) > merged_breaks
                merged_breaks |= internal_gap
                merged_breaks[1:] |= internal_gap[:-1]
            breaks = merged_breaks
        sums = totals(self._sums)
        if independent is None:
            sigma = curve_sigma(errors, pixels, sums, totals(self._squares), totals(self._variances),
                                totals(self._scales), totals(self._extra), breaks)
        else:
            sigma = independent / pixels
            sparse = pixels < 10
            if errors.model == "scatter" and errors.pooled_scatter and sparse.any():
                pooled = scatter_sigma(sums, totals(self._squares), pixels)
                sigma[sparse] = pooled[sparse]
        return x, sums / pixels, sigma, pixels


MIN_POINT_PIXELS = 8
SPARSE_FRACTION = 0.1
"""A bin with fewer pixels than max(MIN_POINT_PIXELS, SPARSE_FRACTION × the curve's median) is joined
with its neighbours: near the beam, at the edge of a sector, a gap or the missing wedge a bin catches
only a few pixels and its mean is mostly noise."""


def _sparse_starts(x, pixels, min_pixels=None):
    """First occupied bin in each merged point, or None if every bin stays separate."""
    if pixels.size < 3:
        return None
    wanted = int(min_pixels) if min_pixels is not None else max(
        MIN_POINT_PIXELS, int(round(SPARSE_FRACTION * float(np.median(pixels))))
    )
    if pixels.min() >= wanted:
        return None
    step = float(np.median(np.diff(x)))
    starts, total = [0], 0
    for i, n in enumerate(pixels):
        gap = i > 0 and step > 0 and x[i] - x[i - 1] > 2.5 * step
        if gap or total >= wanted:
            starts.append(i)
            total = 0
        total += n
    tail = starts[-1]
    if total < wanted and tail > 0 and step > 0 and x[tail] - x[tail - 1] <= 2.5 * step:
        starts.pop()
    return np.asarray(starts) if len(starts) < len(x) else None


def merge_sparse_bins(
    x: np.ndarray, mean: np.ndarray, sigma: np.ndarray, pixels: np.ndarray, *, min_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Join sparse points of an existing curve with pixel weights and propagate their independent errors."""
    x, mean, sigma, pixels = (np.asarray(value) for value in (x, mean, sigma, pixels))
    starts = _sparse_starts(x, pixels, min_pixels)
    if starts is None:
        return x, mean, sigma, pixels
    n = np.add.reduceat(pixels, starts)
    return (np.add.reduceat(x * pixels, starts) / n, np.add.reduceat(mean * pixels, starts) / n,
            np.sqrt(np.add.reduceat((sigma * pixels) ** 2, starts)) / n, n)


def binned_mean(
    x: np.ndarray, values: np.ndarray, *, low: float, high: float, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accumulator = BinnedMean.linear(low, high, bins)
    accumulator.add(x, values)
    return accumulator.result()


def native_profile(
    values: np.ndarray, valid: np.ndarray, x: np.ndarray, *, axis: int, counts_model: bool = True,
    scale: np.ndarray | None = None, errors: PixelErrors | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One point per measured detector line: the mean over ``axis`` of its valid pixels.

    ``axis=0`` averages each column of a band of rows (a horizontal cut),
    ``axis=1`` each row of a band of columns (a vertical cut).  Returns
    ``(x, mean, sigma, pixels)`` sorted by ``x`` for the lines with data;
    ``x`` is the mean coordinate of the line's valid pixels and ``sigma`` the
    standard error of the line mean from the same model as ``BinnedMean``.
    Each line retains its native coordinates, including detector gaps.
    """
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values) & np.isfinite(x)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    x_sums = np.where(valid, x, 0.0).sum(axis=axis)
    keep = counts > 0
    pixels = counts[keep]
    centres = x_sums[keep] / pixels
    mean = sums[keep] / pixels
    errors = errors if errors is not None else PixelErrors(model="poisson" if counts_model else "scatter", scale=scale)
    order = np.argsort(centres, kind="stable")
    ids = np.flatnonzero(keep)[order]
    pixels = pixels[order]
    breaks = np.r_[True, np.abs(np.diff(ids)) != 1][:len(ids)]

    def totals(array):
        return np.where(valid, array, 0.0).sum(axis=axis, dtype=np.float64)[ids]

    squares = totals(values**2) if errors.model == "scatter" else None
    variance = errors.variance
    if variance is None and errors.scale is not None:
        variance = values * errors.scale
    sigma = curve_sigma(errors, pixels, sums[ids], squares,
                        None if variance is None else totals(variance),
                        None if errors.scale is None else totals(errors.scale),
                        0.0 if errors.extra_variance is None else totals(errors.extra_variance), breaks)
    return centres[order], mean[order], sigma, pixels.astype(np.int64)


def binned_mean_2d(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Mean of ``values`` on a regular grid; ``shape`` is ``(ny, nx)``, row 0 = low y.

    Empty cells are NaN.
    """
    ny, nx = (max(1, int(value)) for value in shape)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[finite], y[finite], values[finite]
    x0, x1 = (float(value) for value in x_range)
    y0, y1 = (float(value) for value in y_range)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Grid ranges must be increasing.")
    column = np.floor((x - x0) / (x1 - x0) * nx).astype(np.int64)
    row = np.floor((y - y0) / (y1 - y0) * ny).astype(np.int64)
    column[x == x1] = nx - 1
    row[y == y1] = ny - 1
    keep = (column >= 0) & (column < nx) & (row >= 0) & (row < ny)
    flat = row[keep] * nx + column[keep]
    sums = np.bincount(flat, weights=values[keep], minlength=nx * ny)
    counts = np.bincount(flat, minlength=nx * ny)
    grid = np.full(nx * ny, np.nan)
    filled = counts > 0
    grid[filled] = sums[filled] / counts[filled]
    return grid.reshape(ny, nx)


def pixel_step(values: np.ndarray, stride: int = 4) -> float:
    """The largest change of a per-pixel quantity from one pixel to the next (99th percentile, on every
    ``stride``-th pixel); 0 when the map is too small to tell."""
    sample = np.asarray(values[::stride, ::stride], dtype=np.float64)
    if sample.shape[0] < 2 or sample.shape[1] < 2:
        return 0.0
    rows, columns = np.gradient(sample)
    step = np.hypot(rows, columns) / stride
    finite = step[np.isfinite(step)]
    return float(np.percentile(finite, 99)) if finite.size else 0.0


MIN_MAP_CELLS = 32
"""Fewest cells along one axis of a q map (a strip a few pixels wide still shows as a picture)."""
MAP_STEP_PERCENTILE = 95.0
"""The cell of a q map is the step between neighbouring pixels that this share of the used pixels does not exceed."""
_STEP_SAMPLE = 2_000_000


def neighbour_step(values: np.ndarray, usable: np.ndarray, percentile: float = MAP_STEP_PERCENTILE) -> float:
    """How far ``values`` (one number per pixel) moves from a used pixel to its right and lower neighbours
    (their root sum of squares; ``percentile`` over the used pixels, rows thinned on large frames); 0 when no
    used pixel has both neighbours."""
    values, usable = np.asarray(values), np.asarray(usable, dtype=bool)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return 0.0
    rows = np.arange(0, values.shape[0] - 1, max(1, values.size // _STEP_SAMPLE))
    here = values[rows, :-1].astype(np.float64)
    down = values[rows + 1, :-1] - here
    right = values[rows, 1:] - here
    both = usable[rows, :-1] & usable[rows + 1, :-1] & usable[rows, 1:]
    step = np.hypot(down, right)[both]
    step = step[np.isfinite(step)]
    return float(np.percentile(step, percentile)) if step.size else 0.0


def map_grid_shape(
    x_map: np.ndarray, y_map: np.ndarray, usable: np.ndarray, x_span: float, y_span: float, limit: tuple[int, int],
) -> tuple[int, int]:
    """``(ny, nx)`` of a regular q-map grid that is not finer than the detector samples q: a cell is at least as
    wide as the step of ``x_map`` (as high as that of ``y_map``) between neighbouring used pixels, so the cells
    of the measured region hold a pixel instead of alternating with empty ones; never more cells than ``limit``.

    The grid is a picture and an export of the frame; no curve is computed from it."""
    def cells(values: np.ndarray, span: float, most: int) -> int:
        step = neighbour_step(values, usable)
        if not (step > 0 and span > 0):
            return int(most)
        return int(min(most, max(MIN_MAP_CELLS, math.floor(span / step))))

    return cells(y_map, y_span, int(limit[0])), cells(x_map, x_span, int(limit[1]))


__all__ = ["BinnedMean", "binned_mean", "binned_mean_2d", "counts_and_scale", "map_grid_shape", "native_profile",
           "neighbour_step", "pixel_step"]
