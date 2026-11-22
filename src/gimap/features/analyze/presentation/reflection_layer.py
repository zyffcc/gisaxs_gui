"""Reflection markers drawn over a pyqtgraph plot: spots with hkl labels, powder arcs and ring lines.

The layer owns its items and takes them off the plot before it draws again; the plot's own items (the image,
the curves, the bands) are not touched. Marker sizes follow √(I / I_max) of the phase (I = m·|F|²), so the weak
reflections stay visible; spots the detector does not see (the missing wedge, below the horizon) and the
reflected channel are hollow."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Z_VALUE = 12
MAX_LABELS = 60
"""hkl labels per phase at most (the strongest)."""
SMALLEST, LARGEST = 5.0, 17.0


@dataclass
class Spots:
    color: str
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    hollow: list = field(default_factory=list)
    tips: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    """(x, y, text, strength) of the labelled spots."""


def marker_size(intensity: float, strongest: float) -> float:
    if not strongest > 0:
        return SMALLEST
    return SMALLEST + (LARGEST - SMALLEST) * math.sqrt(max(intensity, 0.0) / strongest)


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _spread_labels(labels, spots) -> list[tuple[float, float, str]]:
    """The strongest labels (``MAX_LABELS``), leaving out one that would sit on a stronger one: closer than 4 % of
    the spots' x span and 2.5 % of their y span."""
    xs = [x for group in spots for x in group.x] or [0.0, 1.0]
    ys = [y for group in spots for y in group.y] or [0.0, 1.0]
    near_x, near_y = 0.04 * (max(xs) - min(xs) or 1.0), 0.025 * (max(ys) - min(ys) or 1.0)
    kept: list[tuple[float, float, str]] = []
    for x, y, text, _strength in sorted(labels, key=lambda entry: -entry[3]):
        if len(kept) >= MAX_LABELS:
            break
        if all(abs(x - other_x) > near_x or abs(y - other_y) > near_y for other_x, other_y, _text in kept):
            kept.append((x, y, text))
    return kept


class ReflectionLayer:
    def __init__(self, plot):
        """``plot``: a ``pyqtgraph.PlotItem``."""
        self._plot = plot
        self._items: list = []
        self._corner = None
        self.arcs = ()
        self.spots = ()
        self.labels = []
        self.legend = ()

    def clear(self) -> None:
        self.arcs = self.spots = self.legend = ()
        self.labels = []
        if self._corner is not None:
            scene = self._corner.scene()
            if scene is not None:
                scene.removeItem(self._corner)
            self._corner = None
        for item in self._items:
            try:
                self._plot.removeItem(item)
            except RuntimeError:  # the plot is being torn down
                pass
        self._items.clear()

    def _add(self, item):
        item.setZValue(Z_VALUE)
        self._plot.addItem(item, ignoreBounds=True)
        self._items.append(item)
        return item

    def draw(self, spots: list[Spots], *, arcs=(), lines=(), labels: bool = True, legend=()) -> None:
        """``arcs``: ``(color, xs, ys)`` curves; ``lines``: ``(color, x, text)`` vertical lines; ``legend``:
        ``(name, color)`` of the phases, written in the corner."""
        import pyqtgraph as pg
        from PyQt5.QtCore import Qt

        self.clear()
        self.arcs, self.spots, self.legend = arcs, spots, legend
        for color, xs, ys in arcs:
            self._add(pg.PlotDataItem(xs, ys, pen=pg.mkPen(color, width=1.2, style=Qt.DashLine), connect="finite"))
        for color, x, text in lines:
            line = self._add(pg.InfiniteLine(pos=float(x), angle=90, movable=False,
                                             pen=pg.mkPen(color, width=1, style=Qt.DashLine)))
            line.setToolTip(text)
            if labels and text:
                label = pg.InfLineLabel(line, text=text, position=0.92, rotateAxis=(1, 0), anchor=(1, 1))
                label.setColor(color)
        for group in spots:
            if not group.x:
                continue
            brushes = [None if hollow else pg.mkBrush(group.color) for hollow in group.hollow]
            item = pg.ScatterPlotItem(
                x=group.x, y=group.y, size=group.sizes, symbol="o", pen=pg.mkPen(group.color, width=1.5),
                brush=[brush if brush is not None else pg.mkBrush(0, 0, 0, 0) for brush in brushes],
                hoverable=True, data=group.tips, tip=lambda x, y, data: str(data),
            )
            self._add(item)
            if not labels:
                continue
            for x, y, text in _spread_labels(group.labels, spots):
                self.labels.append((x, y, text, group.color))
                label = pg.TextItem(text, color=group.color, anchor=(-0.15, 1.1))
                label.setPos(x, y)
                self._add(label)
        if legend:
            self._legend(legend)

    def _legend(self, entries) -> None:
        """Coloured phase names in the top-left corner of the plot (fixed on the screen, not in data units)."""
        import pyqtgraph as pg

        html = "<br>".join(f'<span style="color:{color}">&#9679; {_escape(name)}</span>' for name, color in entries)
        item = pg.TextItem(html=html, anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 140))
        item.setParentItem(self._plot.getViewBox())
        item.setPos(6, 6)
        item.setZValue(Z_VALUE + 1)
        self._corner = item

    @property
    def shown(self) -> int:
        """Items drawn (for tests)."""
        return len(self._items)


__all__ = ["ReflectionLayer", "Spots", "marker_size"]
