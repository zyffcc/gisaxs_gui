"""Estimate a horizontal symmetry axis from measured detector rows."""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class CenterSymmetryResult:
    center_x: float
    initial_x: float
    score_before: float
    score_after: float
    paired_samples: int
    radius_px: float
    search_min: float
    search_max: float

    def to_dict(self):
        return asdict(self)


def optimize_horizontal_center(image, pixel_region, initial_x):
    """Robust mirrored-profile loss, fixed support and no reflection-filled data.

    Bounds are inclusive analysis-array row/column indices. Negative CBF pixels
    are invalid. A row median suppresses isolated hot pixels; gaps remain NaN,
    so interpolation cannot fabricate measurements across a detector gap.
    Scores are dimensionless Huber losses in asinh intensity, not probabilities.
    """
    data = np.asarray(image, dtype=float)
    if data.ndim != 2:
        raise ValueError("Load a 2D detector image first")
    r0, r1, x0, x1 = map(int, pixel_region)
    if not (0 <= r0 <= r1 < data.shape[0] and 0 <= x0 < x1 < data.shape[1]):
        raise ValueError("Select a valid horizontal cut band")
    if r1 - r0 > x1 - x0:
        raise ValueError("Center X optimization requires a horizontal cut")
    band = data[r0 : r1 + 1, x0 : x1 + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(np.where(np.isfinite(band) & (band >= 0), band, np.nan), axis=0)
    valid = profile[np.isfinite(profile)]
    if len(valid) < 40 or np.percentile(valid, 90) - np.percentile(valid, 10) <= 1e-8:
        raise ValueError("Too little intensity structure to determine a symmetry center")
    initial = float(initial_x)
    if not x0 + 20 < initial < x1 - 20:
        raise ValueError("Place the cut center inside a band with data on both sides")
    search = min(80.0, (x1 - x0) * 0.15, (initial - x0) * 0.4, (x1 - initial) * 0.4)
    low, high = initial - search, initial + search
    radius = min(low - x0, x1 - high)
    offsets = np.arange(3.0, radius, 0.5)
    scale = max(float(np.percentile(valid[valid > 0], 25)), 1e-12)
    transformed = np.arcsinh(profile / scale)
    columns = np.arange(x0, x1 + 1)

    def evaluate(center):
        left = np.interp(center - offsets, columns, transformed)
        right = np.interp(center + offsets, columns, transformed)
        keep = np.isfinite(left) & np.isfinite(right)
        count = int(keep.sum())
        if count < max(40, len(offsets) * 0.6):
            return float("inf"), count
        residual = np.abs(left[keep] - right[keep])
        loss = np.where(residual <= 1, 0.5 * residual**2, residual - 0.5)
        return float(loss.mean()), count

    grid = np.linspace(low, high, max(3, int(np.ceil(2 * search)) + 1))
    scores = np.array([evaluate(c)[0] for c in grid])
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise ValueError("Insufficient paired pixels outside detector gaps")
    if best in (0, len(grid) - 1):
        raise ValueError("Best center reaches the search boundary; widen or recenter the cut band")
    fine = minimize_scalar(
        lambda c: evaluate(c)[0],
        bounds=(grid[best - 1], grid[best + 1]),
        method="bounded",
        options={"xatol": 0.005},
    )
    candidates = (initial, float(grid[best]), float(fine.x))
    center = min(candidates, key=lambda c: evaluate(c)[0])
    after, pairs = evaluate(center)
    return CenterSymmetryResult(
        center, initial, evaluate(initial)[0], after, pairs, radius, low, high
    )
