"""How far each frame of a series lies from the frame before it and from a reference frame: the timeline of a
Series map, where a transition shows as a peak (as Xi-cam's Timeline draws the χ² of neighbouring frames).

The rows of a map are mean intensities without their σ, so the noise of every x column is taken from the series
itself: from the second differences of the column along the frames, ``s = 1.4826 · median|I[i+1] − 2 I[i] +
I[i−1]| / √6`` (a steady rise or fall cancels in them, and a jump of a few frames hardly moves the median), and no
column counts as quieter than the usual one (``column_noise``; below four frames: the square root of the level).
The difference of frames ``a`` and ``b`` is then

    χ² = Σₖ ((I_ak − I_bk) / s_k)² / (2 n)

over the ``n`` columns where both have a value: about 1 for two frames that differ by noise alone (the difference
of two frames has twice the variance of one), large where the sample changed. A *jump* is a frame whose χ² to the
frame before is a local maximum above ``JUMP_LEAST`` and above the median of the series by more than
``JUMP_SPREAD`` robust deviations (at most ``MOST_JUMPS``, the largest).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

JUMP_SPREAD = 5.0
"""A jump stands above the median χ² of neighbouring frames by this many robust deviations (1.4826 × MAD)."""
JUMP_LEAST = 3.0
"""… and is at least this χ² (three times what noise alone gives): a quiet series has no jumps."""
MOST_JUMPS = 10
NOISE_SERIES = "second differences along the frames"
NOISE_LEVEL = "square root of the level (fewer than four frames)"


def column_noise(image: np.ndarray) -> tuple[np.ndarray, str]:
    """``(noise of every column, how it was found)`` (module docstring): never below the usual column's."""
    image = np.asarray(image, dtype=float)
    rows = image.shape[0]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns without data: NaN
        if rows >= 4:
            noise = 1.4826 * np.nanmedian(np.abs(np.diff(image, n=2, axis=0)), axis=0) / math.sqrt(6.0)
            how = NOISE_SERIES
        else:
            noise = np.sqrt(np.abs(np.nanmean(image, axis=0)))
            how = NOISE_LEVEL
    usual = noise[np.isfinite(noise) & (noise > 0)]
    floor = float(np.median(usual)) if usual.size else 1.0
    return np.where(np.isfinite(noise), np.maximum(noise, floor), np.nan), how


def difference_chi2(image: np.ndarray, rows_a: Sequence[int], rows_b: Sequence[int], noise: np.ndarray,
                    columns: Optional[np.ndarray] = None) -> np.ndarray:
    """χ² between row ``rows_a[i]`` and ``rows_b[i]`` of ``image`` for every i (NaN without a common column)."""
    image = np.asarray(image, dtype=float)
    keep = np.isfinite(noise) & (noise > 0)
    if columns is not None:
        keep &= np.asarray(columns, dtype=bool)
    a = image[np.asarray(rows_a, dtype=int)][:, keep]
    b = image[np.asarray(rows_b, dtype=int)][:, keep]
    with np.errstate(all="ignore"):
        terms = ((a - b) / noise[keep]) ** 2
        both = np.isfinite(terms)
        count = both.sum(axis=1)
        total = np.where(both, terms, 0.0).sum(axis=1)
        return np.where(count > 0, total / (2.0 * np.maximum(count, 1)), np.nan)


def jump_rows(values, *, spread: float = JUMP_SPREAD, least: float = JUMP_LEAST, most: int = MOST_JUMPS) -> tuple:
    """The rows whose value is a local maximum standing out of the others (module docstring), in row order."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return ()
    median = float(np.median(finite))
    deviation = 1.4826 * float(np.median(np.abs(finite - median)))
    limit = max(median + spread * deviation, float(least))
    padded = np.concatenate([[-np.inf], np.where(np.isfinite(values), values, -np.inf), [-np.inf]])
    peaks = [row for row in range(values.size)
             if padded[row + 1] > limit and padded[row + 1] >= padded[row] and padded[row + 1] >= padded[row + 2]]
    chosen = sorted(peaks, key=lambda row: -values[row])[: int(most)]
    return tuple(sorted(chosen))


@dataclass(frozen=True)
class Timeline:
    """The χ² of every frame to the frame before it and to the reference frame (module docstring)."""

    neighbour: np.ndarray
    """χ² of row i to row i − 1 (NaN for the first row)."""
    reference: np.ndarray
    """χ² of every row to ``reference_row`` (0 there)."""
    reference_row: int
    jumps: tuple
    """Rows where the series jumps (``jump_rows`` of ``neighbour``)."""
    noise_from: str
    columns: int
    """How many x columns were compared."""


def series_timeline(x: np.ndarray, image: np.ndarray, *, reference_row: int = 0,
                    x_range: Optional[Sequence[float]] = None) -> Timeline:
    """The timeline of the rows of ``image`` (on the columns ``x``; ``x_range``: only those inside it)."""
    image = np.asarray(image, dtype=float)
    rows = image.shape[0]
    if rows < 2:
        raise ValueError("A timeline needs at least two frames.")
    reference_row = int(np.clip(reference_row, 0, rows - 1))
    columns = np.ones(image.shape[1], dtype=bool)
    if x_range is not None:
        low, high = sorted(float(value) for value in x_range)
        x = np.asarray(x, dtype=float)
        columns = (x >= low) & (x <= high)
    noise, how = column_noise(image[:, columns] if columns.any() else image)
    full = np.full(image.shape[1], np.nan)
    full[columns if columns.any() else slice(None)] = noise
    neighbour = np.full(rows, np.nan)
    neighbour[1:] = difference_chi2(image, range(1, rows), range(rows - 1), full)
    reference = difference_chi2(image, range(rows), [reference_row] * rows, full)
    return Timeline(neighbour, reference, reference_row, jump_rows(neighbour), how,
                    int(np.count_nonzero(np.isfinite(full))))


__all__ = ["JUMP_LEAST", "JUMP_SPREAD", "MOST_JUMPS", "NOISE_LEVEL", "NOISE_SERIES", "Timeline", "column_noise",
           "difference_chi2", "jump_rows", "series_timeline"]
