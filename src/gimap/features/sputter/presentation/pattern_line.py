"""A line drawn on the measured or the model image of the Patterns view, and the two images along it (2026-11-22, user:
draw a line anywhere on the left or right image and compare the curves in one place).

The user drags on either image: the line is drawn on both (they share their axes) and the cells it crosses are read in
order, one point per cell (``line_profile``): the measured and the model count rate as drawn, no value interpolated or
changed. A line closer to the q_y direction is plotted against the cells' q_y, one closer to the exit-angle direction
against their αf. The line stays while another window or solution is shown, and its curves follow.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .pattern_view import cell_edges

LINE_SAMPLES = 2000
"""Points read along the line; a cell is read once however many of them fall in it."""
MIN_DRAG_PIXELS = 6.0
"""A shorter drag is a click: it draws no line."""
LINE_COLOUR = "#ff4d4d"


def line_profile(q, af, measured, model, start, end, *, log_q: bool) -> Optional[dict]:
    """The cells crossed by the line from ``start`` to ``end`` ((x, y) as drawn: x is log₁₀ q_y when ``log_q``, y the
    exit angle in degrees), in order from the start: ``{"axis": "q" | "af", "x", "measured", "model", "cells"}``, x
    being the cells' q_y (nm⁻¹) or αf (°) — q_y when the line runs more along the q_y axis than along the αf axis
    (each measured on the drawn image) — and the values those of the cells (NaN where a cell has no data; ``model``
    None without a model). None when the line crosses no cell."""
    q, af = np.asarray(q, dtype=float), np.asarray(af, dtype=float)
    q_edges, af_edges = cell_edges(q, positive=log_q), cell_edges(af)
    xs = np.linspace(float(start[0]), float(end[0]), LINE_SAMPLES)
    ys = np.linspace(float(start[1]), float(end[1]), LINE_SAMPLES)
    columns = np.searchsorted(q_edges, 10.0 ** xs if log_q else xs, side="right") - 1
    rows = np.searchsorted(af_edges, ys, side="right") - 1
    inside = (columns >= 0) & (columns < q.size) & (rows >= 0) & (rows < af.size)
    cells: list = []
    for row, column in zip(rows[inside], columns[inside]):
        if not cells or cells[-1] != (int(row), int(column)):
            cells.append((int(row), int(column)))
    if not cells:
        return None
    x_span = (math.log10(q_edges[-1]) - math.log10(q_edges[0])) if log_q else (q_edges[-1] - q_edges[0])
    along_q = abs(float(end[0]) - float(start[0])) / max(x_span, 1e-12) >= \
        abs(float(end[1]) - float(start[1])) / max(af_edges[-1] - af_edges[0], 1e-12)
    index = tuple(np.array(cells).T)
    x = q[index[1]] if along_q else af[index[0]]
    measured_values = np.asarray(measured, dtype=float)[index]
    model_values = np.asarray(model, dtype=float)[index] if model is not None else None
    return {"axis": "q" if along_q else "af", "x": x, "measured": measured_values, "model": model_values,
            "cells": cells}


class PatternLine:
    """The line on a ``PatternPair``; ``shown(profile)`` is told the curves (None: no line)."""

    def __init__(self, pair, shown: Callable[[Optional[dict]], None]):
        self.pair = pair
        self.shown = shown
        self.line: Optional[tuple] = None
        """((x0, y0), (x1, y1)) as drawn (x: log₁₀ q_y on a logarithmic q axis), or None."""
        self.profile: Optional[dict] = None
        self._start: Optional[tuple] = None
        self._artists: list = []
        if pair.canvas is None:
            return
        pair.on_drawn = self.refresh
        canvas = pair.canvas
        canvas.mpl_connect("button_press_event", self._pressed)
        canvas.mpl_connect("motion_notify_event", self._moved)
        canvas.mpl_connect("button_release_event", self._released)

    # -- the user's drag ----------------------------------------------------------------------------------------
    def _images(self) -> tuple:
        axes = self.pair._axes
        return tuple(axes[:2]) if axes else ()

    def _pressed(self, event) -> None:
        if event.button != 1 or event.inaxes not in self._images() or event.xdata is None:
            return
        self._start = (float(event.xdata), float(event.ydata), float(event.x), float(event.y))

    def _moved(self, event) -> None:
        if self._start is None or event.inaxes not in self._images() or event.xdata is None:
            return
        self._draw_line(((self._start[0], self._start[1]), (float(event.xdata), float(event.ydata))))
        self.pair.canvas.draw_idle()

    def _released(self, event) -> None:
        start, self._start = self._start, None
        if start is None:
            return
        if event.xdata is None or event.inaxes not in self._images() or \
                math.hypot(event.x - start[2], event.y - start[3]) < MIN_DRAG_PIXELS:
            self._draw_line(self.line)  # a click: the line of before stays as it was
            self.pair.canvas.draw_idle()
            return
        self.line = ((start[0], start[1]), (float(event.xdata), float(event.ydata)))
        self.refresh()
        self.pair.canvas.draw_idle()

    # -- the line and its curves --------------------------------------------------------------------------------
    def set_line(self, line: Optional[tuple]) -> None:
        """Draw ``line`` (as drawn coordinates) or none, and tell its curves."""
        self.line = line
        self.refresh()
        if self.pair.canvas is not None:
            self.pair.canvas.draw_idle()

    def clear(self) -> None:
        self.set_line(None)

    def refresh(self) -> None:
        """The line on the images as drawn now and its curves from their values (after every drawing of the pair)."""
        self._draw_line(self.line)
        last = self.pair._last
        profile = None
        if self.line is not None and last is not None and last[0] == "patterns" and last[1] is not None:
            patterns, label = last[1], last[2]
            profile = line_profile(patterns.q, patterns.af, patterns.measured, patterns.model(label), *self.line,
                                   log_q=bool(np.all(patterns.q > 0)))
            if profile is not None:
                profile["name"] = last[3]
        self.profile = profile
        self.shown(profile if self.line is not None else None)

    def _draw_line(self, line: Optional[tuple]) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):  # already gone with a figure drawn anew
                pass
        self._artists = []
        if line is None:
            return
        (x0, y0), (x1, y1) = line
        for axes in self._images():
            self._artists += axes.plot([x0, x1], [y0, y1], color=LINE_COLOUR, lw=1.4, marker="o", ms=3, zorder=5,
                                       scalex=False, scaley=False)


__all__ = ["LINE_SAMPLES", "PatternLine", "line_profile"]
