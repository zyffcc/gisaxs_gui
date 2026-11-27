"""Module gaps of a Pilatus detector that a file stores as zero counts.

A Pilatus is tiled from modules of 487 × 195 pixels, 7 pixels apart along a row of modules and 17 pixels apart
from one row of modules to the next. The gaps hold no pixel. DECTRIS writes −1 there, which the readers already
leave out as a negative value; ALS 7.3.3 EDF files (and some converted TIFFs) write 0, which looks like a
measured zero count. Here the gaps of a known detector size are found from its layout, not from the counts
alone: a row of a short exposure can be all zero and still be measured.

Only an integer frame is looked at (counts as stored), and in it only a gap band (the rows between two rows of
modules, or the columns between two modules) that holds nothing above 0: some files fill the gaps with values
(interpolated, or the gap rows of a detector whose modules are packed closer), and a zero among those is a value
like the others. A floating-point frame (dark- or background-subtracted, where 0 is data) is left as it is.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

MODULE = (195, 487)
"""Rows and columns of one Pilatus module."""
GAP = (17, 7)
"""Pixels between two rows of modules, and between two modules of a row."""
LAYOUTS = {
    "Pilatus 300K": (3, 1),
    "Pilatus 300K-W": (1, 3),
    "Pilatus 1M": (5, 2),
    "Pilatus 2M": (8, 3),
    "Pilatus 6M": (12, 5),
}
"""Rows and columns of modules of the Pilatus sizes with gaps (a 100K is one module)."""


def _shape(modules: tuple[int, int]) -> tuple[int, int]:
    return tuple(n * size + (n - 1) * gap for n, size, gap in zip(modules, MODULE, GAP))


SHAPES = {_shape(modules): name for name, modules in LAYOUTS.items()}
"""Frame shape (rows, columns) → detector name."""


def layout_gaps(shape: tuple[int, int]) -> tuple[Optional[str], Optional[np.ndarray]]:
    """``(detector, gaps)``: the gap pixels of a Pilatus frame of ``shape`` (also stored turned by 90°), or
    ``(None, None)`` for any other shape. The array is shared between calls: read it, do not change it."""
    return _layout_gaps(tuple(int(value) for value in shape[:2]))


@lru_cache(maxsize=8)
def _layout_gaps(shape: tuple[int, int]) -> tuple[Optional[str], Optional[np.ndarray]]:
    rows, columns = shape
    turned = (columns, rows) in SHAPES and (rows, columns) not in SHAPES
    name = SHAPES.get((columns, rows) if turned else (rows, columns))
    if name is None:
        return None, None
    module_rows, module_columns = LAYOUTS[name]
    height, width = (columns, rows) if turned else (rows, columns)
    gaps = np.ones((height, width), dtype=bool)
    for r in range(module_rows):
        top = r * (MODULE[0] + GAP[0])
        for c in range(module_columns):
            left = c * (MODULE[1] + GAP[1])
            gaps[top:top + MODULE[0], left:left + MODULE[1]] = False
    gaps = np.ascontiguousarray(gaps.T if turned else gaps)
    gaps.setflags(write=False)
    return name, gaps


def _bands(indices: np.ndarray) -> list[slice]:
    """Runs of consecutive indices as slices."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]] + 1
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def zero_gaps(raw: np.ndarray) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """``(invalid, record)``: the pixels of the gap bands of an integer Pilatus frame that hold nothing above 0 and
    store 0 there, and what was left out (``{"detector": …, "pixels": n}``); ``(None, None)`` when the frame is of
    another size, not integer, or has no such band."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        return None, None
    name, gaps = layout_gaps(raw.shape)
    if gaps is None:
        return None, None
    empty = np.zeros(raw.shape, dtype=bool)
    for axis in (1, 0):  # bands of gap rows (whole rows), then of gap columns (whole columns)
        for band in _bands(np.flatnonzero(gaps.all(axis=axis))):
            part = raw[band, :] if axis == 1 else raw[:, band]
            if not (part > 0).any():
                if axis == 1:
                    empty[band, :] = True
                else:
                    empty[:, band] = True
    invalid = empty & (raw == 0)
    count = int(invalid.sum())
    if count == 0:
        return None, None
    return invalid, {"detector": name, "pixels": count,
                     "rule": "module gap bands of the detector layout that hold only 0 (or negative) values"}


__all__ = ["LAYOUTS", "SHAPES", "layout_gaps", "zero_gaps"]
