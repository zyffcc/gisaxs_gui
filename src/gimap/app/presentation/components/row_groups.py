"""Groups of consecutive rows on an image view — the stages of a series map.

A colour strip along the right edge of the image (one colour per group), dashed lines where a group
begins, the group's number at its left, and a red arrow at rows that do not belong (odd frames). All
of it is one kind of mark, “Stages”, in the view's Marks menu.
"""

from __future__ import annotations

from typing import Optional, Sequence

STAGE_COLORS = ("#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#ca8a04", "#db2777")
"""Colour of stage 1, 2, …; the same in every view that shows stages."""
ODD_COLOR = "#ef4444"
STRIP = 0.03
"""Width of the strip, as a share of the image width."""


def stage_color(stage: int) -> str:
    """Colour of a 1-based stage."""
    return STAGE_COLORS[(int(stage) - 1) % len(STAGE_COLORS)]


class RowGroups:
    """Draws groups of rows over ``view`` (a ``DetectorView``): ``show(edges, x_range, odd_rows=…)``."""

    def __init__(self, view, key: str = "stages", title: str = "Stages"):
        self.view, self.key = view, key
        view.marks.add_layer(key, title)
        self._items: list = []

    def show(self, edges: Sequence[int], x_range: tuple[float, float], *, odd_rows: Sequence[int] = (),
             tips: Optional[Sequence[str]] = None) -> None:
        """``edges``: ``(0, start of group 2, …, rows)``; rows are image rows (y from 0)."""
        import pyqtgraph as pg
        from PyQt5.QtCore import QRectF, Qt
        from PyQt5.QtGui import QBrush, QColor, QPen
        from PyQt5.QtWidgets import QGraphicsRectItem

        self.clear()
        low, high = sorted(float(value) for value in x_range)
        width = max(high - low, 1e-12)
        left = high - STRIP * width
        for index, (first, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            color = QColor(stage_color(index))
            strip = QGraphicsRectItem(QRectF(left, float(first), high - left, float(end - first)))
            color.setAlpha(220)
            strip.setBrush(QBrush(color))
            strip.setPen(QPen(Qt.NoPen))
            if tips is not None and index - 1 < len(tips):
                strip.setToolTip(tips[index - 1])
            self._add(strip, 12)
            label = pg.TextItem(str(index), color="#ffffff", anchor=(0.0, 0.5))
            label.setPos(low + 0.01 * width, 0.5 * (first + end))
            self._add(label, 13)
            if first > 0:
                line = pg.InfiniteLine(pos=float(first), angle=0, movable=False,
                                       pen=pg.mkPen("#ffffff", width=1.2, style=Qt.DashLine))
                self._add(line, 11)
        if odd_rows:
            arrows = pg.ScatterPlotItem(
                pos=[(left - 0.01 * width, row + 0.5) for row in odd_rows], symbol="t3", size=11,
                pen=pg.mkPen(ODD_COLOR, width=1), brush=pg.mkBrush(ODD_COLOR),
            )
            self._add(arrows, 13)

    def _add(self, item, z: int) -> None:
        item.setZValue(z)
        self.view.plot.addItem(item, ignoreBounds=True)
        self.view.marks.track(self.key, item, True)
        self._items.append(item)

    def clear(self) -> None:
        for item in self._items:
            self.view.marks.forget(self.key, item)
            try:
                self.view.plot.removeItem(item)
            except RuntimeError:  # the scene is gone
                pass
        self._items.clear()

    @property
    def shown(self) -> bool:
        return bool(self._items)


__all__ = ["ODD_COLOR", "RowGroups", "STAGE_COLORS", "stage_color"]
