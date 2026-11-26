"""The ticks of the frame axis of a Series map (``frame_ticks``): frame numbers, or seconds, in the middle of the rows.

The axis asks for them at every redraw, zoom and pan. Picked row by row, the ticks of a live map took the window
0.15 s at 50 000 rows at each of them (twice: the round steps and the finer ones), and naming the rows by seconds
named every row; they are picked from all the rows at once now, and only the rows that get a tick are named. The
decimals of seconds depend on all the rows: they are worked out once for the values the axis keeps.
"""

from __future__ import annotations

import math

import numpy as np

MAJOR_TICKS = 8
"""About this many numbered ticks on the frame axis; the other rows are numbered once there is room (zoom)."""
MINOR_ROWS = 300
"""Every row in view has a tick while at most this many are in view; more rows get the ticks of a finer round step
instead. (A tick for each of 20 000 rows is 20 000 lines the axis draws again at every zoom, pan and redraw: 0.6 s.)"""
_DECIMALS_KEPT: list = [None, 0]
"""The values whose decimals were worked out last, and those decimals (the axis asks again at every zoom and pan)."""


def nice_step(span: float, most: int = MAJOR_TICKS) -> int:
    """1, 2, 5, 10, 20, 50 …: the smallest of them that leaves at most ``most`` ticks over ``span``."""
    step = 1
    for power in range(0, 9):
        for base in (1, 2, 5):
            step = base * 10**power
            if span / step <= most:
                return step
    return step


def _decimals(values, array: np.ndarray) -> int:
    """The decimals of seconds: those of their usual step, as few as say the same of every row (“0.0, 1.5, 3.0”;
    “0.50, 1.00” says no more than “0.5, 1.0”)."""
    if _DECIMALS_KEPT[0] is values:
        return _DECIMALS_KEPT[1]
    listed = array.tolist()
    if np.isfinite(array).all():
        steps = np.sort(np.abs(array[1:] - array[:-1])[array[1:] != array[:-1]]).tolist()
    else:  # (NaN sorts nowhere in particular: as ever)
        steps = sorted(abs(b - a) for a, b in zip(listed, listed[1:]) if b != a)
    usual = steps[len(steps) // 2] if steps else 1.0
    decimals = int(min(4, max(0, 1 - math.floor(math.log10(usual))))) if usual > 0 else 0
    while decimals > 0 and all(f"{value:.{decimals}f}".endswith("0") for value in listed):
        decimals -= 1
    if isinstance(values, np.ndarray) and not values.flags.writeable:  # (values that cannot change: the axis's)
        _DECIMALS_KEPT[:] = [values, decimals]
    return decimals


def _tick_names(values, array: np.ndarray, whole: bool):
    """The name of a row on the frame axis: its frame number, or its seconds with the decimals of their usual step
    — the same for every row."""
    if whole:
        return lambda row: f"{int(round(float(array[row])))}"
    decimals = _decimals(values, array)
    return lambda row: f"{float(array[row]):.{decimals}f}"


def _on_step(rows: np.ndarray, rounded: np.ndarray, step: int, by_row: bool) -> np.ndarray:
    """Per row of ``rows``: it is on a round ``step`` of the rows, or of its frame number."""
    return rows % step == 0 if by_row else rounded % step == 0


def frame_ticks(values, *, whole: bool = True, shown=None) -> list[list[tuple[float, str]]]:
    """Ticks for the frame axis of a map of ``len(values)`` rows: each in the middle of its row, named by
    ``values`` (frame numbers, or seconds). Two levels: about ``MAJOR_TICKS`` named rows — round frame numbers
    where the numbers are whole —, and the rows between them, which are named when the axis has room (of more than
    ``MINOR_ROWS`` rows, those of a finer round step). ``shown``: the part of the axis in view, ``(low, high)`` in
    rows; the ticks are those of its rows (all rows when ``None``)."""
    array = np.asarray(values, dtype=float)
    rows = int(array.size)
    if not rows:
        return [[], []]
    first, end = 0, rows
    if shown is not None:
        low, high = sorted(float(edge) for edge in shown)
        first = min(rows - 1, max(0, int(math.floor(low))))
        end = max(first + 1, min(rows, int(math.ceil(high))))
    count = end - first
    name = _tick_names(values, array, whole)
    in_view = np.arange(first, end)
    rounded = None
    by_row = not whole
    if whole:
        part = array[first:end]
        if not np.isfinite(part).all():
            raise ValueError("cannot convert float NaN or infinity to an integer frame number")
        rounded = np.rint(part).astype(np.int64)  # (round() of each: half to even)
    span = (float(array[end - 1]) - float(array[first]) + 1) if whole else count
    step = nice_step(span)
    major = _on_step(in_view, rounded, step, by_row)
    if int(np.count_nonzero(major)) < min(count, 2):  # numbers that never are round (every 3rd frame from 1): by row
        by_row, span = True, count
        step = nice_step(count)
        major = in_view % step == 0
    if count <= MINOR_ROWS:
        minor = ~major
    else:
        minor = ~major & _on_step(in_view, rounded, nice_step(span, 5 * MAJOR_TICKS), by_row)
    return [[(row + 0.5, name(row)) for row in in_view[major].tolist()],
            [(row + 0.5, name(row)) for row in in_view[minor].tolist()]]


__all__ = ["MAJOR_TICKS", "MINOR_ROWS", "frame_ticks", "nice_step"]
