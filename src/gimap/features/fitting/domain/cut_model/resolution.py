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


def grid(q: np.ndarray, sigma: float, finer: bool = True, per_sigma: int = POINTS) -> tuple[np.ndarray, float, bool]:
    """(grid, step, folded) of the convolution for the points ``q`` (|q|, nm⁻¹), ``per_sigma`` points per σ;
    ``finer``: no coarser than the points either."""
    finite = np.diff(np.unique(q))
    step = sigma / per_sigma
    if finer and finite.size:
        step = min(step, float(np.median(finite)))
    low, high = max(0.0, float(q.min()) - SPAN * sigma), float(q.max()) + SPAN * sigma
    count = min(GRID_MAX, int(np.ceil((high - low) / step)) + 1)
    step = (high - low) / (count - 1)
    folded = low == 0.0
    return low + (np.arange(count) + (0.5 if folded else 0.0)) * step, step, folded


def _convolve(values, step: float, folded: bool, sigma: float) -> np.ndarray:
    half = int(np.ceil(SPAN * sigma / step))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sigma) ** 2)
    kernel /= kernel.sum()
    values = np.asarray(values, float)
    if folded:
        full = np.concatenate([values[::-1], values])
        return np.convolve(np.pad(full, half, mode="edge"), kernel, mode="valid")[values.size:]
    return np.convolve(np.pad(values, half, mode="edge"), kernel, mode="valid")


def smeared(q: np.ndarray, sigma: float, terms_on: Callable[[np.ndarray], list]) -> list[np.ndarray]:
    """Each curve that ``terms_on(grid)`` returns, convolved with the Gaussian and taken at ``q``."""
    q = np.abs(np.asarray(q, float))
    points, step, folded = grid(q, sigma)
    return [np.interp(q, points, _convolve(values, step, folded, sigma)) for values in terms_on(points)]


def tabulated(q_out, table_q, curves, sigma: float) -> np.ndarray:
    """The same resolution for curves already tabulated on a sorted grid ``table_q`` (K, len(table_q)), taken at
    ``q_out``: where σ exceeds half the table's local spacing, the curves are interpolated onto a uniform grid of σ/20
    (``grid``; taken back linearly, which σ/5 would do to 1 %), convolved and taken back; elsewhere they are interpolated. The table has to reach q_out ± SPAN σ (or
    q → 0). Used by the training simulator, which tabulates on a fine logarithmic grid."""
    q_out = np.abs(np.asarray(q_out, float))
    table_q = np.asarray(table_q, float)
    curves = np.atleast_2d(np.asarray(curves, float))
    out = np.stack([np.interp(q_out, table_q, c) for c in curves])
    if sigma <= 0:
        return out
    wide = sigma > 0.5 * np.interp(q_out, table_q, np.gradient(table_q))
    if np.any(wide):
        points, step, folded = grid(q_out[wide], sigma, finer=False, per_sigma=20)
        for k, c in enumerate(curves):
            out[k, wide] = np.interp(q_out[wide], points, _convolve(np.interp(points, table_q, c), step, folded,
                                                                   sigma))
    return out


TABLE_POINTS = 1024
PREDICT_TABLE_POINTS = 256
PREDICT_MAX_RELATIVE = 0.01


def predict_sharp_limit(q):
    """Widths below a quarter of the smallest measured spacing resolve no broadening in Predict."""
    steps = np.diff(np.unique(q))
    return 0.25 * float(steps.min()) if steps.size else 0.0


def predict_grid(q, sigma):
    """Common table for the whole Predict prior, independent of the other samples in a batch."""
    return table_grid(q, max(float(sigma), PREDICT_MAX_RELATIVE * float(np.max(q))), PREDICT_TABLE_POINTS)


def table_grid(q, sigma: float, points: int = TABLE_POINTS) -> np.ndarray:
    """A grid for ``tabulated`` at the points ``q`` (|q|): the points themselves, ``points`` logarithmically spaced
    over their range, and SPAN σ beyond both ends (at σ/4, down to q → 0 for the fold)."""
    q = np.unique(np.abs(np.asarray(q, float)))
    q = q[q > 0]
    low, high = float(q[0]), float(q[-1])
    core = np.geomspace(low, high, points) if high > low else np.array([low])
    reach = SPAN * sigma
    bottom = max(low - reach, 0.125 * sigma, low * 1e-3)
    lower = np.linspace(bottom, low, int(min(256, np.ceil((low - bottom) / (0.25 * sigma)))) + 2)[:-1] if low > bottom         else np.zeros(0)
    upper = np.linspace(high, high + reach, int(min(64, np.ceil(reach / (0.25 * sigma)))) + 2)[1:]
    return np.unique(np.concatenate([lower, core, q, upper]))


__all__ = ["smeared", "table_grid", "tabulated"]
