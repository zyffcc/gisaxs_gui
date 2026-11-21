"""Where the measured cells of a gridded picture (a q map) lie, for fitting the view to the data."""

from __future__ import annotations

from typing import Optional

import numpy as np

TRIMMED_SHARE = 0.005
"""Share of the filled cells that may lie outside the fitted view on each side: a thin streak of a few cells far
from the rest (the first detector columns of a tilted frame) does not shrink everything else."""


def filled_extent(
    image: np.ndarray, rect: tuple[float, float, float, float], *, trimmed: float = TRIMMED_SHARE,
) -> Optional[tuple[float, float, float, float]]:
    """``(x, y, width, height)`` holding all but ``trimmed`` of the finite cells of ``image`` on each side, in the
    coordinates of ``rect = (x, y, width, height)`` over the whole picture (row 0 at ``y``); ``None`` when no cell
    is finite. The picture itself is not changed: this only says where to look."""
    filled = np.isfinite(np.asarray(image, dtype=float))
    if filled.ndim != 2 or not filled.any():
        return None
    left, bottom, width, height = (float(value) for value in rect)

    def span(counts: np.ndarray) -> tuple[int, int]:
        total = np.cumsum(counts)
        low = int(np.searchsorted(total, trimmed * total[-1], side="right"))
        high = int(np.searchsorted(total, (1.0 - trimmed) * total[-1], side="left"))
        return min(low, high), max(low, high) + 1

    (c0, c1), (r0, r1) = span(filled.sum(axis=0)), span(filled.sum(axis=1))
    rows, columns = filled.shape
    return (left + width * c0 / columns, bottom + height * r0 / rows,
            width * (c1 - c0) / columns, height * (r1 - r0) / rows)


__all__ = ["TRIMMED_SHARE", "filled_extent"]
