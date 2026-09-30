"""The unwrapped (cake) view of a GIWAXS frame: intensity against q and χ.

Rows are χ from −90° (bottom) to +90° (top) in the reduction convention (0° along the surface
normal), columns q from 0 to the largest measured q. Each cell is the mean of the valid pixels
above the horizon that fall in it (NaN where none do, e.g. the missing wedge next to χ = 0°).
A ring is a vertical line, a cut region a rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import binned_mean_2d

CAKE_SHAPE = (360, 800)
"""(χ bins, q bins): 0.5° in χ."""


@dataclass(frozen=True)
class CakeMap:
    image: np.ndarray
    """``(χ bins, q bins)``; row 0 is χ = −90°."""
    q_range: tuple[float, float]
    chi_range: tuple[float, float] = (-90.0, 90.0)

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Bin centres: q (columns) and χ (rows, from −90° up)."""
        rows, columns = self.image.shape
        (q0, q1), (c0, c1) = self.q_range, self.chi_range
        q = q0 + (np.arange(columns) + 0.5) * (q1 - q0) / columns
        chi = c0 + (np.arange(rows) + 0.5) * (c1 - c0) / rows
        return q, chi


def cake_map(image: np.ndarray, usable: np.ndarray, maps, *, shape: tuple[int, int] = CAKE_SHAPE) -> CakeMap:
    """Mean intensity on a regular (χ, q) grid of the ``usable`` pixels."""
    usable = np.asarray(usable, dtype=bool)
    if not usable.any():
        raise ValueError("No valid pixels above the horizon to unwrap.")
    q = maps.q[usable]
    q_high = float(q.max())
    grid = binned_mean_2d(
        q, maps.chi_deg[usable], np.asarray(image)[usable],
        x_range=(0.0, q_high + 1e-12), y_range=(-90.0, 90.0 + 1e-9), shape=shape,
    )
    return CakeMap(grid.astype(np.float32), (0.0, q_high))


__all__ = ["CAKE_SHAPE", "CakeMap", "cake_map"]
