"""Mean-per-bin accumulation shared by every Analyze reduction."""

from __future__ import annotations

import numpy as np


def scatter_sigma(sums, squares, counts) -> np.ndarray:
    """Standard error of each mean from the scatter of its own values, ``s/√n``.

    For frames that are not photon counts (dark- or background-subtracted, flat-panel detectors)
    the Poisson error ``√sum/n`` says nothing: read noise and the subtraction dominate. A mean of
    fewer than two values takes the median scatter of the others.
    """
    sums, squares, counts = (np.asarray(value, dtype=np.float64) for value in (sums, squares, counts))
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (squares - sums**2 / np.maximum(counts, 1)) / np.maximum(counts - 1, 1)
    variance = np.clip(variance, 0.0, None)
    several = counts >= 2
    typical = float(np.median(variance[several])) if several.any() else 0.0
    variance = np.where(several, variance, typical)
    return np.sqrt(variance / np.maximum(counts, 1))


def counts_and_scale(counts, mask) -> tuple[bool, np.ndarray | None]:
    """``counts`` of a reduction — ``True`` (photon counts), ``False`` (not counts) or a per-pixel array
    (photon counts times that factor, after intensity corrections) — as ``(Poisson?, scale of the
    pixels in mask)``."""
    if isinstance(counts, np.ndarray):
        return True, counts[mask]
    return bool(counts), None


class BinnedMean:
    """Mean of ``values`` per bin of ``x`` over fixed edges, fed in chunks.

    Values exactly on the last edge fall into the last bin; values outside the
    edges or non-finite are ignored.  ``sigma`` is the Poisson standard error
    of the mean, ``sqrt(sum)/n``, which is exact for photon counts; with
    ``counts=False`` it is the standard error from the scatter of the bin's
    values (``scatter_sigma``), for frames that are not photon counts.
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
        """Σ scale × value per bin, once values with a ``scale`` were added (else the Poisson sum)."""

    @classmethod
    def linear(cls, low: float, high: float, bins: int) -> "BinnedMean":
        low, high, bins = float(low), float(high), max(1, int(bins))
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("Bin range must be finite.")
        if high <= low:
            half = max(abs(low) * 1e-9, 1e-12)
            low, high = low - half, low + half
        return cls(np.linspace(low, high, bins + 1))

    def add(self, x: np.ndarray, values: np.ndarray, scale: np.ndarray | None = None) -> None:
        """``scale``: per value, the factor the photon counts were multiplied by (intensity corrections);
        the Poisson variance of such a value is ``scale × value``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        values = np.asarray(values, dtype=np.float64).ravel()
        if x.shape != values.shape:
            raise ValueError("x and values must have the same number of elements.")
        finite = np.isfinite(x) & np.isfinite(values)
        x, values = x[finite], values[finite]
        size = self._counts.size
        index = np.searchsorted(self.edges, x, side="right") - 1
        index[x == self.edges[-1]] = size - 1
        keep = (index >= 0) & (index < size)
        self._sums += np.bincount(index[keep], weights=values[keep], minlength=size)
        self._squares += np.bincount(index[keep], weights=values[keep] ** 2, minlength=size)
        self._counts += np.bincount(index[keep], minlength=size)
        if scale is not None:
            scale = np.asarray(scale, dtype=np.float64).ravel()[finite][keep]
            if self._variances is None:
                self._variances = self._sums - np.bincount(index[keep], weights=values[keep], minlength=size)
            self._variances += np.bincount(index[keep], weights=values[keep] * scale, minlength=size)
        elif self._variances is not None:
            self._variances += np.bincount(index[keep], weights=values[keep], minlength=size)

    def result(self, *, counts: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(centres, mean, sigma, pixels)`` for the bins that received data."""
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        filled = self._counts > 0
        pixels = self._counts[filled]
        sums = self._sums[filled]
        mean = sums / pixels
        if counts:
            variance = sums if self._variances is None else self._variances[filled]
            sigma = np.sqrt(np.clip(variance, 0.0, None)) / pixels
        else:
            sigma = scatter_sigma(sums, self._squares[filled], pixels)
        return centres[filled], mean, sigma, pixels


MIN_POINT_PIXELS = 8
SPARSE_FRACTION = 0.1
"""A bin with fewer pixels than max(MIN_POINT_PIXELS, SPARSE_FRACTION × the curve's median) is joined
with its neighbours: near the beam, at the edge of a sector, a gap or the missing wedge a bin catches
only a few pixels and its mean is mostly noise."""


def merge_sparse_bins(
    x: np.ndarray, mean: np.ndarray, sigma: np.ndarray, pixels: np.ndarray, *, min_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Join consecutive sparse bins until each point has enough pixels.

    A point made of several bins has the pixel-weighted mean of their ``x`` and of their means
    (the mean of all its pixels), the Poisson error ``sqrt(sum)/n`` of all its pixels, and their
    pixel count. Bins are only joined with a neighbour no further than two bin steps away, so a
    real gap in the data stays a gap. A curve whose bins all have enough pixels is returned as is.
    """
    x, mean, sigma, pixels = (np.asarray(value) for value in (x, mean, sigma, pixels))
    if pixels.size < 3:
        return x, mean, sigma, pixels
    wanted = int(min_pixels) if min_pixels is not None else max(
        MIN_POINT_PIXELS, int(round(SPARSE_FRACTION * float(np.median(pixels))))
    )
    if int(pixels.min()) >= wanted:
        return x, mean, sigma, pixels
    step = float(np.median(np.diff(x))) if x.size > 1 else 0.0
    groups: list[list[int]] = []
    current: list[int] = []
    total = 0
    for index in range(x.size):
        if current and step > 0 and x[index] - x[current[-1]] > 2.5 * step:
            groups.append(current)
            current, total = [], 0
        current.append(index)
        total += int(pixels[index])
        if total >= wanted:
            groups.append(current)
            current, total = [], 0
    if current:
        if groups and total < wanted and step > 0 and x[current[0]] - x[groups[-1][-1]] <= 2.5 * step:
            groups[-1].extend(current)  # a sparse tail joins the last point
        else:
            groups.append(current)
    if all(len(group) == 1 for group in groups):
        return x, mean, sigma, pixels
    out_x, out_mean, out_sigma, out_pixels = [], [], [], []
    for group in groups:
        n = pixels[group].astype(np.float64)
        count = float(n.sum())
        sums = float((mean[group] * n).sum())
        out_x.append(float((x[group] * n).sum() / count))
        out_mean.append(sums / count)
        # Independent means weighted by their pixels: σ² = Σ n²σ² / (Σn)²; for photon counts Σ n²σ² = Σ sum.
        out_sigma.append(float(np.sqrt(float((n**2 * sigma[group] ** 2).sum())) / count) if count else float("nan"))
        out_pixels.append(int(count))
    return np.array(out_x), np.array(out_mean), np.array(out_sigma), np.array(out_pixels, dtype=np.int64)


def binned_mean(
    x: np.ndarray, values: np.ndarray, *, low: float, high: float, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accumulator = BinnedMean.linear(low, high, bins)
    accumulator.add(x, values)
    return accumulator.result()


def native_profile(
    values: np.ndarray, valid: np.ndarray, x: np.ndarray, *, axis: int, counts_model: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One point per measured detector line: the mean over ``axis`` of its valid pixels.

    ``axis=0`` averages each column of a band of rows (a horizontal cut),
    ``axis=1`` each row of a band of columns (a vertical cut).  Returns
    ``(x, mean, sigma, pixels)`` sorted by ``x`` for the lines with data;
    ``x`` is the mean coordinate of the line's valid pixels and ``sigma`` the
    Poisson standard error of the mean, ``sqrt(max(sum, 1))/n`` (at least one
    count, so a line of zeros never gets σ = 0), or with ``counts=False`` the
    error from the scatter of the line's values (``scatter_sigma``).  Nothing is
    interpolated or rebinned, so detector gaps stay gaps.
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
    if counts_model:
        sigma = np.sqrt(np.maximum(sums[keep], 1.0)) / pixels
    else:
        squares = np.where(valid, values**2, 0.0).sum(axis=axis)
        sigma = scatter_sigma(sums[keep], squares[keep], pixels)
    order = np.argsort(centres, kind="stable")
    return centres[order], mean[order], sigma[order], pixels[order].astype(np.int64)


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


__all__ = ["BinnedMean", "binned_mean", "binned_mean_2d", "counts_and_scale", "native_profile"]
