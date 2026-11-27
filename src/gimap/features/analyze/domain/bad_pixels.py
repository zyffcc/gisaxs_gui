"""Hot and dead pixels found in the frame itself, so they never reach a curve.

Only isolated pixels are flagged; anything that spreads over neighbouring pixels (a Bragg
spot, a streak, the beam-stop edge) is left alone:

* hot — brighter than all eight valid neighbours by a factor ``HOT_FACTOR`` *and* by
  ``HOT_SIGMA`` standard deviations (Poisson for counting detectors; a robust frame-wide
  noise for floating-point frames): a stuck pixel or a zinger.
* dead — reads 0 (or less) while every valid neighbour has at least ``DEAD_MIN_COUNTS``
  counts: at that level a true zero has a probability of about e^-20. Counting detectors only.
* defective line — a detector row (or column) with scattered pixels far brighter than the rows
  just above and below (seen on a counting detector: pixels of single rows reading orders of
  magnitude above an empty pattern, singly or in pairs, so the isolated-pixel test misses the pairs). See
  ``_defective_lines`` for the rule that keeps real features.

Fast on large frames (a 15-megapixel Lambda frame in a fraction of a second): separable 3 × 3
sums give a necessary condition, and only the few pixels that pass it are compared with their
eight neighbours one by one. A sparse counting frame (a short exposure: few pixels above
``HOT_SIGMA`` counts, none below 0) is faster still from those pixels alone (``_find_sparse``):
a hot pixel or a pixel of a defective line exceeds ``HOT_SIGMA`` counts, and a dead pixel lies
next to one of at least ``DEAD_MIN_COUNTS``. Both ways apply the same tests and flag the same
pixels; only the candidates are found differently.
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
SPARSE_SHARE = 0.02
"""A counting frame with at most this share of its pixels above ``HOT_SIGMA`` counts is searched from those
pixels alone (``_find_sparse``); on a fuller frame the sums over the whole frame are faster."""
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


def _gathered(data: np.ndarray, valid: np.ndarray, rows: np.ndarray, columns: np.ndarray, offsets,
              fill: float) -> np.ndarray:
    """``(len(rows), len(offsets))`` values of the pixels at ``offsets`` from each pixel; ``fill`` where that pixel
    is not valid or off the frame (what ``_neighbours`` reads from the padded frame, without making it)."""
    height, width = data.shape
    out = np.full((rows.size, len(offsets)), fill, dtype=data.dtype)
    for k, (dr, dc) in enumerate(offsets):
        r, c = rows + dr, columns + dc
        inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        r, c = r[inside], c[inside]
        out[inside, k] = np.where(valid[r, c], data[r, c], fill)
    return out


def _hot(value: np.ndarray, base: np.ndarray, *, counts: bool, noise: float) -> np.ndarray:
    """The hot-pixel test of pixels ``value`` whose brightest valid neighbour is ``base`` (``-inf``: none)."""
    has_neighbours = np.isfinite(base)
    base = np.where(has_neighbours, base, 0.0)
    sigma = np.sqrt(np.maximum(base, 0.0) + 1.0) if counts else np.full(value.shape, noise)
    return has_neighbours & (value > HOT_FACTOR * np.maximum(base, 0.0) + sigma) & (value - base > HOT_SIGMA * sigma)


def _line_candidates(value: np.ndarray, across: np.ndarray, *, counts: bool, noise: float) -> np.ndarray:
    """The defective-line test of pixels ``value`` whose brighter neighbour across the line is ``across``."""
    has_neighbour = np.isfinite(across)
    base = np.where(has_neighbour, np.maximum(across, 0.0), 0.0)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(base + 1.0) if counts else np.float32(noise)
        return has_neighbour & (value > HOT_FACTOR * base + sigma) & (value - base > HOT_SIGMA * sigma)


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
    if counts:
        bright = _sparse(data, valid)
        if bright is not None:
            return _find_sparse(data, valid, *bright)
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
        flagged = _hot(data[rows, columns], base, counts=counts, noise=noise)
        hot[rows[flagged], columns[flagged]] = True
    if candidates_dead.any():
        rows, columns = np.nonzero(candidates_dead)
        high = np.pad(np.where(valid, data, np.inf), 1, constant_values=np.inf)
        lowest = np.min(_neighbours(high, rows, columns), axis=1)
        flagged = np.isfinite(lowest) & (lowest >= DEAD_MIN_COUNTS)
        dead[rows[flagged], columns[flagged]] = True
    return BadPixels(hot=hot, dead=dead, lines=_defective_lines(data, valid, counts=counts, noise=noise) & ~hot)


def _sparse(data: np.ndarray, valid: np.ndarray):
    """``(rows, columns)`` of the valid pixels above ``HOT_SIGMA`` counts of a sparse counting frame, in row order;
    ``None``: the frame is not sparse (``SPARSE_SHARE``) or has valid pixels below 0 (a neighbour below 0 would
    let a hot pixel be dimmer)."""
    if (valid & (data < 0)).any():
        return None
    rows, columns = np.nonzero(valid & (data > HOT_SIGMA))
    return (rows, columns) if rows.size <= SPARSE_SHARE * data.size else None


def _find_sparse(data: np.ndarray, valid: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> BadPixels:
    """``find_bad_pixels`` of a sparse counting frame from its pixels above ``HOT_SIGMA`` counts (``_sparse``).

    Every hot pixel is among them (it exceeds its brightest neighbour, at least 0, by ``HOT_SIGMA`` σ ≥ 1), every
    pixel of a defective line too (the same against the brighter pixel across the line), and every dead pixel lies
    next to one of at least ``DEAD_MIN_COUNTS``. The tests are those of the whole-frame search."""
    hot = np.zeros(data.shape, dtype=bool)
    dead = np.zeros(data.shape, dtype=bool)
    lines = np.zeros(data.shape, dtype=bool)
    if rows.size:
        value = data[rows, columns]
        flagged = _hot(value, np.max(_gathered(data, valid, rows, columns, OFFSETS, -np.inf), axis=1),
                       counts=True, noise=1.0)
        hot[rows[flagged], columns[flagged]] = True
        for axis in (0, 1):
            offsets = ((-1, 0), (1, 0)) if axis == 0 else ((0, -1), (0, 1))
            across = np.max(_gathered(data, valid, rows, columns, offsets, -np.inf), axis=1)
            candidate = _line_candidates(value, across, counts=True, noise=1.0)
            if axis == 0:  # (already in row order)
                _mark_lines(lines, rows[candidate], columns[candidate], axis)
            else:
                order = np.lexsort((rows[candidate], columns[candidate]))
                _mark_lines(lines, columns[candidate][order], rows[candidate][order], axis)
        _dead_near(data, valid, rows[value >= DEAD_MIN_COUNTS], columns[value >= DEAD_MIN_COUNTS], dead)
    return BadPixels(hot=hot, dead=dead, lines=lines & ~hot)


def _dead_near(data: np.ndarray, valid: np.ndarray, rows: np.ndarray, columns: np.ndarray, dead: np.ndarray) -> None:
    """Mark the dead pixels next to the pixels ``rows, columns`` of at least ``DEAD_MIN_COUNTS`` (the test of the
    whole-frame search)."""
    height, width = data.shape
    near = []
    for dr, dc in OFFSETS:  # the pixels that have one of them at (dr, dc)
        r, c = rows - dr, columns - dc
        inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        near.append(r[inside] * width + c[inside])
    r, c = np.divmod(np.unique(np.concatenate(near)), width)
    keep = valid[r, c] & (data[r, c] <= 0)
    r, c = r[keep], c[keep]
    if r.size:
        lowest = np.min(_gathered(data, valid, r, c, OFFSETS, np.inf), axis=1)
        flagged = np.isfinite(lowest) & (lowest >= DEAD_MIN_COUNTS)
        dead[r[flagged], c[flagged]] = True


def _mark_lines(lines: np.ndarray, line_index: np.ndarray, position: np.ndarray, axis: int) -> None:
    """Mark the defective lines (``_defective_lines``) among candidate pixels sorted by line, then by position along
    it (axis 0: lines are rows)."""
    if line_index.size == 0:
        return
    new_line = np.ones(line_index.size, dtype=bool)
    new_line[1:] = line_index[1:] != line_index[:-1]
    starts = np.flatnonzero(new_line)
    ends = np.append(starts[1:], line_index.size)
    sizes = ends - starts
    spread = position[ends - 1] - position[starts]
    new_run = new_line.copy()
    new_run[1:] |= np.diff(position) != 1
    run_starts = np.flatnonzero(new_run)
    run_lengths = np.diff(np.append(run_starts, line_index.size))
    longest = np.maximum.reduceat(run_lengths, np.searchsorted(run_starts, starts))
    bad = (sizes >= LINE_MIN_PIXELS) & (spread >= LINE_MIN_SPREAD) & (longest <= LINE_MAX_RUN)
    if bad.any():
        member = np.repeat(bad, sizes)
        if axis == 0:
            lines[line_index[member], position[member]] = True
        else:
            lines[position[member], line_index[member]] = True


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
        candidate = valid & _line_candidates(data, np.maximum(before, after), counts=counts, noise=noise)
        if not candidate.any():
            continue
        line_index, position = np.nonzero(candidate) if axis == 0 else np.nonzero(candidate.T)
        _mark_lines(lines, line_index, position, axis)  # (sorted by line, then position: as nonzero gives them)
    return lines


__all__ = ["BadPixels", "DEAD_MIN_COUNTS", "HOT_FACTOR", "HOT_SIGMA", "find_bad_pixels"]
