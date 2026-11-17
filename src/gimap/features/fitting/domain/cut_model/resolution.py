"""A Gaussian q resolution of 1σ = σ_q applied to the particle terms of a symmetric cut.

The terms are made on a uniform grid from max(0, q_min − 5σ) to q_max + 5σ, at most σ/5 apart (and no coarser than
the points), folded at q = 0 (I(−q) = I(q)), convolved with the Gaussian cut at ±5σ and interpolated back to the
points. Points of the fold sit at ±step/2, ±3step/2 …, so q = 0 itself is never evaluated.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SPAN = 5.0
POINTS = 5
"""Grid points per σ."""
GRID_MAX = 6000


def grid(q: np.ndarray, sigma: float) -> tuple[np.ndarray, float, bool]:
    """(grid, step, folded) of the convolution for the points ``q`` (|q|, nm⁻¹)."""
    finite = np.diff(np.unique(q))
    step = sigma / POINTS
    if finite.size:
        step = min(step, float(np.median(finite)))
    low, high = max(0.0, float(q.min()) - SPAN * sigma), float(q.max()) + SPAN * sigma
    count = min(GRID_MAX, int(np.ceil((high - low) / step)) + 1)
    step = (high - low) / (count - 1)
    folded = low == 0.0
    return low + (np.arange(count) + (0.5 if folded else 0.0)) * step, step, folded


def smeared(q: np.ndarray, sigma: float, terms_on: Callable[[np.ndarray], list]) -> list[np.ndarray]:
    """Each curve that ``terms_on(grid)`` returns, convolved with the Gaussian and taken at ``q``."""
    q = np.abs(np.asarray(q, float))
    points, step, folded = grid(q, sigma)
    half = int(np.ceil(SPAN * sigma / step))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sigma) ** 2)
    kernel /= kernel.sum()
    out = []
    for values in terms_on(points):
        values = np.asarray(values, float)
        if folded:
            full = np.concatenate([values[::-1], values])
            result = np.convolve(np.pad(full, half, mode="edge"), kernel, mode="valid")[values.size:]
        else:
            result = np.convolve(np.pad(values, half, mode="edge"), kernel, mode="valid")
        out.append(np.interp(q, points, result))
    return out


__all__ = ["smeared"]
