"""The curves of many frames on one column of x, for a table with a column per frame.

What a table may do to the numbers, in this order (``common_x``; the first that applies):

1. **Exact** — every frame has the same points: the table is the frames side by side.
2. **Matched** — the frames' points are the same up to rounding (the cut sat on another row, so qy differs
   in the eighth digit) or a frame lacks some of them (masked columns, a detector gap, another range): each
   point is put on the row of its x, and a frame without a point there is blank. Nothing is interpolated, no
   measured point is left out.
3. **Interpolated** — the grids really differ (another geometry or number of bins for each frame): the
   points of the first frame that has some, continued with its step across its own holes and as far as the
   other frames reach, are the common x; every frame is interpolated linearly onto it between two neighbouring points of its own —
   never across a hole (points further apart than ``GAP_STEPS`` of its steps) and never beyond its range. What
   another frame measured where the first has a hole (masked columns) is therefore in the table; a row on which
   no frame has a value is left out.

The third line of the table's header says which it was.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

SAME_POINT = 0.05
"""Points of different frames closer than this part of a step are one point of the detector."""
OTHER_GRID = 0.5
"""Points of different frames between ``SAME_POINT`` and this part of a step apart: the grids differ."""
GAP_STEPS = 1.5
"""Two neighbouring points of a frame further apart than this many of its steps have a hole between them."""

EXACT = "x: exact (every frame has the same points)"
MATCHED = ("x: every frame's own points, matched row by row (they agree within {shift:.3g}); nothing is "
           "interpolated; blank = that frame has no point there")
INTERPOLATED = ("x: a common grid (the points of the first frame that has some, continued with its step across its "
                "own holes and as far as the others reach); the frames have other points and are interpolated "
                "linearly onto it between neighbouring points of their own, never across a gap wider than {gap:g} "
                "steps and never beyond their range")


def _step(x: np.ndarray) -> float:
    """The usual distance between neighbouring points (0 for fewer than two different ones)."""
    steps = np.diff(np.unique(x[np.isfinite(x)]))
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else 0.0


def _matched(frames: Sequence[tuple[np.ndarray, np.ndarray]], step: float):
    """``(x, values, shift)`` with every point on the row of its x (the x of the first frame that has the
    point), or ``None`` when the grids differ."""
    x_all = np.concatenate([x for x, _y in frames])
    y_all = np.concatenate([y for _x, y in frames])
    owner = np.concatenate([np.full(x.size, index) for index, (x, _y) in enumerate(frames)])
    order = np.lexsort((owner, x_all))  # by x; equal x: the earlier frame first
    x_all, y_all, owner = x_all[order], y_all[order], owner[order]
    row = np.cumsum(np.r_[0, np.diff(x_all) > SAME_POINT * step])  # a new row where x leaves the points before
    rows = int(row[-1]) + 1
    first = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
    last = np.r_[first[1:], x_all.size] - 1
    shift = float(np.max(x_all[last] - x_all[first]))
    if shift > SAME_POINT * step or np.unique(row * len(frames) + owner).size != x_all.size:
        return None  # a row as wide as the points drift, or two points of one frame on it: no common points
    held = np.zeros((rows, len(frames)), dtype=bool)
    held[row, owner] = True
    by_frame = np.lexsort((owner, row))
    x = x_all[by_frame[np.r_[0, np.flatnonzero(np.diff(row[by_frame])) + 1]]]
    close = np.flatnonzero(np.diff(x_all[first]) < OTHER_GRID * step)
    if close.size and not (held[close] & held[close + 1]).any(axis=1).all():
        return None  # rows closer than half a step that no single frame holds both of: two grids
    values = np.full((len(frames), rows), np.nan)
    values[owner, row] = y_all
    return x, list(values), shift


def _filled(x: np.ndarray, step: float) -> np.ndarray:
    """``x`` with its holes (neighbours further apart than ``GAP_STEPS`` steps) filled at ``step``: no point
    closer than half a step to the far side of the hole."""
    gaps = np.flatnonzero(np.diff(x) > GAP_STEPS * step)
    if not gaps.size:
        return x
    parts, start = [], 0
    for gap in gaps:
        parts.append(x[start:gap + 1])
        count = int(np.floor((x[gap + 1] - x[gap]) / step - 0.5))
        parts.append(x[gap] + step * np.arange(1, count + 1))
        start = gap + 1
    parts.append(x[start:])
    return np.concatenate(parts)


def _interpolated(frames: Sequence[tuple[np.ndarray, np.ndarray]]):
    # the first frame with points (a frame whose reduction gave nothing must not take every row with it)
    x = np.unique(next((c_x for c_x, _y in frames if np.unique(c_x).size > 1), frames[0][0]))
    step = _step(x)
    if step > 0:  # across its holes and as far as the other frames reach, in whole steps: no point of theirs is left out
        low = min(float(c_x.min()) for c_x, _y in frames if c_x.size)
        high = max(float(c_x.max()) for c_x, _y in frames if c_x.size)
        before, after = int((x[0] - low) // step), int((high - x[-1]) // step)
        x = np.r_[x[0] - step * np.arange(before, 0, -1), _filled(x, step), x[-1] + step * np.arange(1, after + 1)]
    values = []
    for c_x, c_y in frames:
        order = np.argsort(c_x, kind="stable")
        c_x, c_y = c_x[order], c_y[order]
        if c_x.size < 2:  # one point cannot be interpolated: it stands where the grid has it
            column = np.full(x.shape, np.nan)
            if c_x.size:
                column[np.isclose(x, c_x[0], rtol=0, atol=1e-9 * step)] = c_y[0]
            values.append(column)
            continue
        column = np.interp(x, c_x, c_y, left=np.nan, right=np.nan)
        right = np.clip(np.searchsorted(c_x, x, side="left"), 1, c_x.size - 1)
        gap = c_x[right] - c_x[right - 1]
        on_a_point = np.isclose(x, c_x[right], rtol=0, atol=1e-9 * step) | np.isclose(x, c_x[right - 1], rtol=0, atol=1e-9 * step)
        column[(gap > GAP_STEPS * (_step(c_x) or step)) & ~on_a_point] = np.nan  # a hole: nothing is drawn across it
        values.append(column)
    held = np.isfinite(np.vstack(values)).any(axis=0) if values else np.zeros(x.shape, dtype=bool)
    return x[held], [column[held] for column in values]  # a row that no frame has a value on is no row


def common_x(frames: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray], str]:
    """``(x, one array of values per frame, the header line that says how)`` for ``frames`` = ``(x, values)``
    of every frame, in list order (module docstring). A point without a value (NaN) is no point."""
    frames = [(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for x, y in frames]
    first = frames[0][0]
    if all(x.shape == first.shape and np.allclose(x, first, rtol=0, atol=1e-12) for x, _y in frames):
        return first, [y for _x, y in frames], EXACT
    measured = [(x[keep], y[keep]) for x, y in frames for keep in (np.isfinite(x) & np.isfinite(y),)]
    step = min((value for value in (_step(x) for x, _y in measured) if value > 0), default=0.0)
    if step > 0:
        matched = _matched(measured, step)
        if matched is not None:
            x, values, shift = matched
            return x, values, MATCHED.format(shift=shift)
    x, values = _interpolated(measured)
    return x, values, INTERPOLATED.format(gap=GAP_STEPS)


__all__ = ["EXACT", "GAP_STEPS", "INTERPOLATED", "MATCHED", "OTHER_GRID", "SAME_POINT", "common_x"]
