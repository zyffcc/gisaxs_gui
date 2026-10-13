"""Beam-centre column from the left–right symmetry of a horizontal GISAXS band.

In GISAXS the direct beam usually hides behind the beam stop, but the diffuse
scattering is symmetric in qy.  Mirroring the band's profile about a trial
column and minimising the difference between the two halves finds the
column of qy = 0.  Only measured pixels are compared: detector gaps stay
gaps, nothing is interpolated across them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from src.gimap.shared.geometry.conventions import canonical_from_index_center, index_center_from_canonical

from .symmetry_coarse import MIN_PAIR_DISTANCE_PX, coarse_center

MIN_PAIRED_SAMPLES = 40
FINE_SEARCH_PX = 4.0
FINE_TOLERANCE_PX = 0.0001
"""Numerical stopping tolerance, not a measured uncertainty of the centre."""
MAX_PAIR_MARGIN_PX = 80.0
"""Retain the established mirrored span; fine-search width must not change the sampled data."""


@dataclass(frozen=True)
class SymmetryCenter:
    """Result of :func:`symmetric_center_x` (canonical x, pixel-corner frame)."""

    x_px: float
    initial_x_px: float
    loss_before: float
    """Log-distance weighted Huber loss of the mirrored asinh column mean."""
    loss_after: float
    paired_samples: int
    search_px: float
    coarse_x_px: float
    tolerance_px: float = FINE_TOLERANCE_PX
    profile_statistic: str = "valid_pixel_mean"
    loss_weighting: str = "inverse_pixel_distance"
    min_pair_distance_px: float = MIN_PAIR_DISTANCE_PX

    @property
    def shift_px(self) -> float:
        return self.x_px - self.initial_x_px

    def to_dict(self) -> dict:
        """Result and numerical settings, for the GUI and provenance records."""
        record = asdict(self)
        for key in ("loss_before", "loss_after"):
            if not math.isfinite(record[key]):
                record[key] = None  # The old centre can lack enough measured pairs for a score.
        return record


def _band_profile(image: np.ndarray, valid: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """Mean of the measured pixels in each column, as in the displayed horizontal cut."""
    start, stop = (int(value) for value in rows)
    band = np.asarray(image[start:stop], dtype=np.float64)
    band_valid = np.asarray(valid[start:stop], dtype=bool) & np.isfinite(band)
    profile = np.full(band.shape[1], np.nan)
    counts = band_valid.sum(axis=0)
    has_data = counts > 0
    if has_data.any():
        totals = np.where(band_valid, band, 0.0).sum(axis=0)
        profile[has_data] = totals[has_data] / counts[has_data]
    return profile


def _stable_offsets(profile: np.ndarray, offsets: np.ndarray, low: float, high: float) -> np.ndarray:
    """Use the same measured interpolation support throughout the search interval.

    Every column that can bracket either mirrored position must be valid.
    Otherwise moving a masked low-distance pair out of the score could look
    like an improvement, especially with log-distance weighting.
    """
    invalid = np.concatenate(([0], np.cumsum(~np.isfinite(profile))))
    last = len(profile) - 1
    keep = np.ones(offsets.size, dtype=bool)
    for sign in (-1, 1):
        first = np.floor(low + sign * offsets).astype(int)
        stop = np.ceil(high + sign * offsets).astype(int)
        inside = (first >= 0) & (stop <= last)
        first, stop = np.clip(first, 0, last), np.clip(stop, 0, last)
        keep &= inside & (invalid[stop + 1] == invalid[first])
    return offsets[keep]


def symmetric_center_x(
    image: np.ndarray,
    valid: np.ndarray,
    rows: tuple[int, int],
    initial_x_px: float,
    *,
    cancelled: Optional[Callable[[], bool]] = None,
) -> SymmetryCenter:
    """Find the band's mirror axis globally, then refine it to subpixel coordinates.

    Positions are canonical: pixel ``j`` covers ``[j, j+1]``, so its centre is
    ``j + 0.5``.  Compare the same column means as the horizontal cut, using
    Huber loss on ``asinh(I / scale)`` and weights ``1 / distance``. This gives
    equal logarithmic distance ranges similar weight, so the many far-away
    pixels cannot overwhelm low-q structure. Distance is measured in pixels,
    approximately proportional to |qy| in GISAXS; this is not exact log-q
    resampling. The score is not a probability or an uncertainty.
    Raises ``ValueError`` when the band has too little structure or data.
    """
    from scipy.optimize import minimize_scalar

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("The frame must be a 2D image.")
    if np.shape(valid) != image.shape:
        raise ValueError("The valid-pixel mask must match the frame shape.")
    if not math.isfinite(float(initial_x_px)):
        raise ValueError("The initial beam centre must be finite.")
    start, stop = (int(value) for value in rows)
    if not 0 <= start < stop <= image.shape[0]:
        raise ValueError("The horizontal cut band lies outside the frame.")
    profile = _band_profile(image, valid, (start, stop))
    measured = profile[np.isfinite(profile)]
    if measured.size < MIN_PAIRED_SAMPLES or (
        np.percentile(measured, 90) - np.percentile(measured, 10) <= 1e-8
    ):
        raise ValueError("The horizontal cut has too little structure to find a symmetry axis.")

    last = image.shape[1] - 1
    initial, _ = index_center_from_canonical(initial_x_px, 0.5)
    coarse = coarse_center(profile, cancelled=cancelled)
    search = min(FINE_SEARCH_PX, coarse * 0.1, (last - coarse) * 0.1)
    low, high = coarse - search, coarse + search
    margin = min(MAX_PAIR_MARGIN_PX, last * 0.15, coarse * 0.4, (last - coarse) * 0.4)
    radius = min(coarse, last - coarse) - margin
    # One pixel is the smallest half-pixel-grid distance whose left/right
    # linear interpolation columns stay disjoint at every subpixel phase.
    offsets = np.arange(MIN_PAIR_DISTANCE_PX, radius, 0.5)
    needed = max(MIN_PAIRED_SAMPLES, int(0.6 * offsets.size))
    offsets = _stable_offsets(profile, offsets, low, high)
    if offsets.size < needed:
        raise ValueError("Too few measured pixels pair up across the beam centre (detector gaps).")
    weights = 1.0 / offsets
    weight_sum = float(weights.sum())
    positive = measured[measured > 0]
    scale = max(float(np.percentile(positive, 25)), 1e-12) if positive.size else 1.0
    transformed = np.arcsinh(profile / scale)
    columns = np.arange(image.shape[1], dtype=np.float64)

    def evaluate(center: float) -> tuple[float, int]:
        if cancelled is not None and cancelled():
            raise ValueError("Centre search cancelled.")
        left = np.interp(center - offsets, columns, transformed, left=np.nan, right=np.nan)
        right = np.interp(center + offsets, columns, transformed, left=np.nan, right=np.nan)
        keep = np.isfinite(left) & np.isfinite(right)
        count = int(keep.sum())
        if count != offsets.size:
            return math.inf, count
        residual = np.abs(left - right)
        loss = np.where(residual <= 1.0, 0.5 * residual**2, residual - 0.5)
        return float(np.dot(weights, loss) / weight_sum), count

    grid = np.linspace(low, high, max(3, int(math.ceil(2 * search)) + 1))
    scores = np.array([evaluate(center)[0] for center in grid])
    best = int(np.argmin(scores))
    if not math.isfinite(scores[best]):
        raise ValueError("Too few measured pixels pair up across the beam centre (detector gaps).")
    if best in (0, len(grid) - 1):
        raise ValueError(
            "The coarse symmetry axis could not be refined reliably; check the cut band and masks."
        )
    fine = minimize_scalar(
        lambda center: evaluate(center)[0],
        bounds=(grid[best - 1], grid[best + 1]),
        method="bounded",
        options={"xatol": FINE_TOLERANCE_PX},
    )
    candidates = [coarse, float(grid[best]), float(fine.x)]
    if low <= initial <= high:
        candidates.append(initial)
    center = min(candidates, key=lambda value: evaluate(value)[0])
    loss_after, pairs = evaluate(center)
    return SymmetryCenter(
        x_px=canonical_from_index_center(center, 0.0)[0],
        initial_x_px=float(initial_x_px),
        loss_before=evaluate(initial)[0],
        loss_after=loss_after,
        paired_samples=pairs,
        search_px=float(search),
        coarse_x_px=canonical_from_index_center(coarse, 0.0)[0],
    )


__all__ = ["SymmetryCenter", "symmetric_center_x"]
