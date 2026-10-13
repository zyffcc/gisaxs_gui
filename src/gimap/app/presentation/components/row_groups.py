"""Groups of consecutive rows on an image view — the stages of a series map.

A colour strip along the right edge of the image (one colour per group), dashed lines (white over a
dark shadow) where a group begins, the group's number on a dark plate at its left, and a red arrow at
rows that do not belong (odd frames). All of it is one kind of mark, “Stages”, in the view's Marks menu.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

STAGE_COLORS = ("#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#ca8a04", "#db2777")
"""Colour of stage 1, 2, …; the same in every view that shows stages."""
ODD_COLOR = "#ef4444"
STRIP = 0.03
"""Width of the strip, as a share of the image width."""
LABEL_PLATE = (0, 0, 0, 150)
"""RGBA of the plate under a stage number (white text)."""
LINE_SHADOW = (0, 0, 0, 170)
"""RGBA of the 3 px line under each white dashed boundary."""


def stage_color(stage: int) -> str:
    """Colour of a 1-based stage (the image strips, the curves: the same in both themes)."""
    return STAGE_COLORS[(int(stage) - 1) % len(STAGE_COLORS)]


TEXT_CONTRAST = 4.5
"""Least contrast of a name's colour on the light surfaces of lists and tables (WCAG AA for text)."""


def _luminance(color) -> float:
    channels = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
                for value in (color.redF(), color.greenF(), color.blueF())]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(first: str, second: str) -> float:
    """WCAG contrast ratio of two colours (1 … 21)."""
    from PyQt5.QtGui import QColor

    high, low = sorted((_luminance(QColor(first)), _luminance(QColor(second))), reverse=True)
    return (high + 0.05) / (low + 0.05)


@lru_cache(maxsize=64)
def _on_light(color: str) -> str:
    """``color`` darkened in small steps (same hue) until it reaches ``TEXT_CONTRAST`` on the darker of the light
    surfaces (a table's alternate rows); a colour that already does is kept as it is."""
    from PyQt5.QtGui import QColor

    from ..theme import LIGHT

    surface = min((LIGHT["surface"], LIGHT["surface_alt"]), key=lambda name: _luminance(QColor(name)))
    shade = QColor(color)
    for _step in range(60):
        if contrast(shade.name(), surface) >= TEXT_CONTRAST:
            break
        shade = shade.darker(104)
    return shade.name()


def text_color(color: str) -> str:
    """``color`` (a stage's, a curve's) as the colour of a name in a list or a table: lighter on a dark theme,
    where the saturated colours are ~3:1 on the surface (lighter: ≥ 6:1); on a light theme as it is, or darker
    in the same hue where it is under 4.5:1 on white (orange, yellow, green, cyan: 2.8–3.7:1).
    Owners colour their names again when ``theme_manager().changed`` fires."""
    from PyQt5.QtGui import QColor

    from ..theme import theme_manager

    if theme_manager().is_dark:
        return QColor(color).lighter(150).name()
    return _on_light(QColor(color).name())


def stage_text_color(stage: int) -> str:
    """Colour of a stage's name in a list or a table (``text_color`` of ``stage_color``: readable on both themes;
    the image strips and the curves keep ``stage_color``)."""
    return text_color(stage_color(stage))


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
        from PyQt5.QtGui import QBrush, QColor, QFont, QPen
        from PyQt5.QtWidgets import QGraphicsRectItem

        self.clear()
        low, high = sorted(float(value) for value in x_range)
        width = max(high - low, 1e-12)
        left = high - STRIP * width
        bold = QFont()
        bold.setBold(True)
        for index, (first, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            color = QColor(stage_color(index))
            strip = QGraphicsRectItem(QRectF(left, float(first), high - left, float(end - first)))
            color.setAlpha(220)
            strip.setBrush(QBrush(color))
            strip.setPen(QPen(Qt.NoPen))
            if tips is not None and index - 1 < len(tips):
                strip.setToolTip(tips[index - 1])
            self._add(strip, 12)
            # White on a dark plate: readable over the bright and the dark end of every colour map.
            label = pg.TextItem(str(index), color="#ffffff", anchor=(0.0, 0.5), fill=pg.mkBrush(*LABEL_PLATE))
            label.setFont(bold)
            label.setPos(low + 0.01 * width, 0.5 * (first + end))
            self._add(label, 13)
            if first > 0:  # a dark line under the white dashes: seen on bright and dark rows alike
                for pen, z in ((pg.mkPen(LINE_SHADOW, width=3), 10.5),
                               (pg.mkPen("#ffffff", width=1.2, style=Qt.DashLine), 11)):
                    self._add(pg.InfiniteLine(pos=float(first), angle=0, movable=False, pen=pen), z)
        if odd_rows:
            arrows = pg.ScatterPlotItem(
                pos=[(left - 0.01 * width, row + 0.5) for row in odd_rows], symbol="t3", size=11,
                pen=pg.mkPen(ODD_COLOR, width=1), brush=pg.mkBrush(ODD_COLOR),
            )
            self._add(arrows, 13)

    def _add(self, item, z: float) -> None:
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


__all__ = ["ODD_COLOR", "RowGroups", "STAGE_COLORS", "TEXT_CONTRAST", "contrast", "stage_color", "stage_text_color",
           "text_color"]
