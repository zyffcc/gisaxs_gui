"""Hot and dead pixels found in the frame itself, so they never reach a curve.

Only isolated pixels are flagged; anything that spreads over neighbouring pixels (a Bragg
spot, a streak, the beam-stop edge) is left alone:

* hot — brighter than all eight valid neighbours by a factor ``HOT_FACTOR`` *and* by
  ``HOT_SIGMA`` standard deviations (Poisson for counting detectors; a robust frame-wide
  noise for floating-point frames): a stuck pixel or a zinger.
* dead — reads 0 (or less) while every valid neighbour has at least ``DEAD_MIN_COUNTS``
  counts: at that level a true zero has a probability of about e^-20. Counting detectors only.
* defective line — a detector row (or column) with scattered pixels far brighter than the rows
  just above and below (seen on a Lambda detector: pixels of two rows at up to 4·10⁵ counts where
  the pattern has 0, singly or in pairs, so the isolated-pixel test misses the pairs). See
  ``_defective_lines`` for the rule that keeps real features.

Fast on large frames (a 15-megapixel Lambda frame in a fraction of a second): separable 3 × 3
sums give a necessary condition, and only the few pixels that pass it are compared with their
eight neighbours one by one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

HOT_FACTOR = 5.0
HOT_SIGMA = 10.0
DEAD_MIN_COUNTS = 20.0
LINE_MIN_PIXELS = 10
LINE_MIN_SPREAD = 50
LINE_MAX_RUN = 10
OFFSETS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
"""The eight neighbours (the pixel itself excluded)."""


@dataclass(frozen=True)
class BadPixels:
    hot: np.ndarray
    dead: np.ndarray
    lines: Optional[np.ndarray] = None
    """Pixels of defective rows and columns."""

    @property
    def mask(self) -> np.ndarray:
        mask = self.hot | self.dead
        return mask if self.lines is None else mask | self.lines

    @property
    def line_count(self) -> int:
        return 0 if self.lines is None else int(self.lines.sum())

    @property
    def hot_count(self) -> int:
        return int(self.hot.sum())

    @property
    def dead_count(self) -> int:
        return int(self.dead.sum())


def _neighbours(padded: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``(len(rows), 8)`` values of the eight neighbours in an array padded by one pixel."""
    return np.stack([padded[rows + 1 + dr, columns + 1 + dc] for dr, dc in OFFSETS], axis=1)


def _robust_noise(data: np.ndarray, valid: np.ndarray, neighbour_mean: np.ndarray) -> float:
    """1.4826 × MAD of the pixel-to-neighbour-mean differences (floating-point frames)."""
    difference = (data - neighbour_mean)[valid]
    if difference.size < 16:
        return float("inf")
    sample = difference[:: max(1, difference.size // 200_000)]
    return float(1.4826 * np.median(np.abs(sample - np.median(sample)))) or float("inf")


def find_bad_pixels(data: np.ndarray, valid: np.ndarray, *, counts: bool = True) -> BadPixels:
    """Hot and dead pixels among ``valid`` (see the module docstring); ``counts``: a counting detector."""
    from scipy.ndimage import uniform_filter

    data = np.asarray(data, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(data)
    hot = np.zeros(data.shape, dtype=bool)
    dead = np.zeros(data.shape, dtype=bool)
    if not valid.any():
        return BadPixels(hot, dead)
    values = np.where(valid, data, np.float32(0.0))
    # Sum of the valid neighbours (3 × 3 window minus the pixel): one separable filter.
    window = uniform_filter(values, size=3, mode="constant")
    window *= 9.0
    window -= values
    if counts:
        noise = 1.0
        # A zero among neighbours of ≥ DEAD_MIN_COUNTS needs at least that much around it.
        candidates_dead = valid & (values <= 0) & (window >= DEAD_MIN_COUNTS)
    else:
        noise = _robust_noise(data, valid, window / 8.0)
        candidates_dead = np.zeros(data.shape, dtype=bool)
    # Brighter than HOT_FACTOR × the mean neighbour (≤ the brightest one) is necessary for a hot pixel.
    window *= HOT_FACTOR / 8.0
    np.maximum(window, 0.0, out=window)
    window += noise
    candidates_hot = values > window  # invalid pixels are 0 here and never exceed it
    if candidates_hot.any():
        rows, columns = np.nonzero(candidates_hot)
        low = np.pad(np.where(valid, data, -np.inf), 1, constant_values=-np.inf)
        base = np.max(_neighbours(low, rows, columns), axis=1)
        value = data[rows, columns]
        has_neighbours = np.isfinite(base)
        base = np.where(has_neighbours, base, 0.0)
        sigma = np.sqrt(np.maximum(base, 0.0) + 1.0) if counts else np.full(value.shape, noise)
        flagged = has_neighbours & (value > HOT_FACTOR * np.maximum(base, 0.0) + sigma) & (value - base > HOT_SIGMA * sigma)
        hot[rows[flagged], columns[flagged]] = True
    if candidates_dead.any():
        rows, columns = np.nonzero(candidates_dead)
        high = np.pad(np.where(valid, data, np.inf), 1, constant_values=np.inf)
        lowest = np.min(_neighbours(high, rows, columns), axis=1)
        flagged = np.isfinite(lowest) & (lowest >= DEAD_MIN_COUNTS)
        dead[rows[flagged], columns[flagged]] = True
    return BadPixels(hot=hot, dead=dead, lines=_defective_lines(data, valid, counts=counts, noise=noise) & ~hot)


def _longest_run(positions: np.ndarray) -> int:
    """Length of the longest run of consecutive integers in sorted ``positions``."""
    if positions.size == 0:
        return 0
    breaks = np.flatnonzero(np.diff(positions) != 1)
    edges = np.concatenate([[-1], breaks, [positions.size - 1]])
    return int(np.diff(edges).max())


def _defective_lines(data: np.ndarray, valid: np.ndarray, *, counts: bool, noise: float) -> np.ndarray:
    """Scattered pixels of one row (column) far brighter than the rows (columns) on both sides.

    A pixel is a candidate when it exceeds both neighbours *across* the line by the hot-pixel
    factor and significance. A line is defective when it has at least ``LINE_MIN_PIXELS`` such
    pixels spread over at least ``LINE_MIN_SPREAD`` pixels, none of their runs longer than
    ``LINE_MAX_RUN``: a detector row that reads wrong here and there. A real feature — a rod, the
    Yoneda band, the horizon — is continuous or wider than one pixel and is left alone.
    """
    low = np.where(valid, data, -np.inf).astype(np.float32)
    lines = np.zeros(data.shape, dtype=bool)
    for axis in (0, 1):  # axis 0: compare with the rows above and below → defective rows
        before = np.roll(low, 1, axis=axis)
        after = np.roll(low, -1, axis=axis)
        edge = [slice(None)] * 2
        edge[axis] = 0
        before[tuple(edge)] = -np.inf
        edge[axis] = -1
        after[tuple(edge)] = -np.inf
        across = np.maximum(before, after)
        has_neighbour = np.isfinite(across)
        base = np.where(has_neighbour, np.maximum(across, 0.0), 0.0)
        with np.errstate(invalid="ignore"):
            sigma = np.sqrt(base + 1.0) if counts else np.float32(noise)
            candidate = valid & has_neighbour & (data > HOT_FACTOR * base + sigma) & (data - base > HOT_SIGMA * sigma)
        if not candidate.any():
            continue
        line_index, position = np.nonzero(candidate) if axis == 0 else np.nonzero(candidate.T)
        for line in np.unique(line_index):
            along = np.sort(position[line_index == line])
            if (
                along.size >= LINE_MIN_PIXELS
                and int(along[-1] - along[0]) >= LINE_MIN_SPREAD
                and _longest_run(along) <= LINE_MAX_RUN
            ):
                if axis == 0:
                    lines[line, along] = True
                else:
                    lines[along, line] = True
    return lines


__all__ = ["BadPixels", "DEAD_MIN_COUNTS", "HOT_FACTOR", "HOT_SIGMA", "find_bad_pixels"]
