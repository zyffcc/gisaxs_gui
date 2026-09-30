"""Beam-centre column from the left–right symmetry of a horizontal GISAXS band.

In GISAXS the direct beam usually hides behind the beam stop, but the diffuse
scattering is symmetric in qy.  Mirroring the band's profile about a trial
column and minimising the difference between the two halves finds the
column of qy = 0.  Only measured pixels are compared: detector gaps stay
gaps, nothing is interpolated across them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_PAIRED_SAMPLES = 40
MAX_SEARCH_PX = 80.0


@dataclass(frozen=True)
class SymmetryCenter:
    """Result of :func:`symmetric_center_x` (canonical x, pixel-corner frame)."""

    x_px: float
    initial_x_px: float
    loss_before: float
    """Mean Huber loss of the mirrored asinh profile at the initial column."""
    loss_after: float
    paired_samples: int
    search_px: float

    @property
    def shift_px(self) -> float:
        return self.x_px - self.initial_x_px


def _band_profile(image: np.ndarray, valid: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """Per-column median over the band rows (isolated hot pixels do not count)."""
    start, stop = (int(value) for value in rows)
    band = np.asarray(image[start:stop], dtype=np.float64)
    band_valid = np.asarray(valid[start:stop], dtype=bool)
    profile = np.full(band.shape[1], np.nan)
    has_data = band_valid.any(axis=0)
    if has_data.any():
        values = np.where(band_valid, band, np.nan)[:, has_data]
        profile[has_data] = np.nanmedian(values, axis=0)
    return profile


def symmetric_center_x(
    image: np.ndarray,
    valid: np.ndarray,
    rows: tuple[int, int],
    initial_x_px: float,
) -> SymmetryCenter:
    """Column of mirror symmetry of the band ``rows`` (half-open), searched near ``initial_x_px``.

    Positions are canonical: pixel ``j`` covers ``[j, j+1]``, so its centre is
    ``j + 0.5``.  The loss is a Huber loss on ``asinh(I / scale)`` (robust to
    the intensity range); it is a comparison score, not a probability.
    Raises ``ValueError`` when the band has too little structure or data.
    """
    from scipy.optimize import minimize_scalar

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("The frame must be a 2D image.")
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
    initial = float(initial_x_px) - 0.5  # index coordinate of the column centre
    if not 20.0 < initial < last - 20.0:
        raise ValueError("The beam centre must lie inside the frame with data on both sides.")
    search = min(MAX_SEARCH_PX, last * 0.15, initial * 0.4, (last - initial) * 0.4)
    low, high = initial - search, initial + search
    radius = min(low, last - high)
    offsets = np.arange(3.0, radius, 0.5)
    positive = measured[measured > 0]
    scale = max(float(np.percentile(positive, 25)), 1e-12) if positive.size else 1.0
    transformed = np.arcsinh(profile / scale)
    columns = np.arange(image.shape[1], dtype=np.float64)
    needed = max(MIN_PAIRED_SAMPLES, int(0.6 * offsets.size))

    def evaluate(center: float) -> tuple[float, int]:
        left = np.interp(center - offsets, columns, transformed)
        right = np.interp(center + offsets, columns, transformed)
        keep = np.isfinite(left) & np.isfinite(right)
        count = int(keep.sum())
        if count < needed:
            return math.inf, count
        residual = np.abs(left[keep] - right[keep])
        loss = np.where(residual <= 1.0, 0.5 * residual**2, residual - 0.5)
        return float(loss.mean()), count

    grid = np.linspace(low, high, max(3, int(math.ceil(2 * search)) + 1))
    scores = np.array([evaluate(center)[0] for center in grid])
    best = int(np.argmin(scores))
    if not math.isfinite(scores[best]):
        raise ValueError("Too few measured pixels pair up across the beam centre (detector gaps).")
    if best in (0, len(grid) - 1):
        raise ValueError(
            "The symmetry axis lies more than "
            f"{search:.0f} px from the beam centre; set the centre roughly first."
        )
    fine = minimize_scalar(
        lambda center: evaluate(center)[0],
        bounds=(grid[best - 1], grid[best + 1]),
        method="bounded",
        options={"xatol": 0.005},
    )
    center = min((initial, float(grid[best]), float(fine.x)), key=lambda value: evaluate(value)[0])
    loss_after, pairs = evaluate(center)
    return SymmetryCenter(
        x_px=center + 0.5,
        initial_x_px=float(initial_x_px),
        loss_before=evaluate(initial)[0],
        loss_after=loss_after,
        paired_samples=pairs,
        search_px=float(search),
    )


__all__ = ["SymmetryCenter", "symmetric_center_x"]
