"""Parts of ``CurvePlot`` that need no plot: its colours, readable log ticks, the marker of a curve drawn as
points, the title that shortens itself, the halves of a curve with a signed x, and the text of a legend entry."""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget

from ..theme import theme_manager

CURVE_COLORS = ("#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2")
LEGEND_ALPHA = 150
"""Opacity of the legend's background: its text stays readable, the data under it still shows."""
LOG_LABEL_SPACING_PX = 16.0
"""Least distance between two labels of a log axis."""
_MANTISSAS = ((1,), (1, 3), (1, 2, 5), tuple(range(1, 10)))  # sparse → dense


def readable_log_ticks(min_val: float, max_val: float, size: float, std_ticks) -> list:
    """Log-axis ticks (in log10 units) whose labels do not pile up.

    pyqtgraph labels every mantissa 1…9 whenever the range spans few decades, which stacks
    0.6, 0.7, 0.8, 0.9 on a short plot. Here the minor ticks are the densest of 1 / 1-3 / 1-2-5 /
    1…9 per decade whose labels stay ``LOG_LABEL_SPACING_PX`` apart (denser only when fewer than
    two would show).
    """
    ticks = [(spacing, values) for spacing, values in std_ticks if spacing >= 1.0]
    if len(ticks) >= 3:
        return ticks
    per_decade = float(size) / max(float(max_val) - float(min_val), 1e-9)
    first, last = int(np.floor(min_val)), int(np.ceil(max_val))

    def inside(mantissas):
        values = [decade + np.log10(m) for decade in range(first, last) for m in mantissas]
        return [value for value in values if min_val < value < max_val]

    fitting = [m for m in _MANTISSAS if np.diff(np.log10(list(m) + [10])).min() * per_decade >= LOG_LABEL_SPACING_PX]
    choice = fitting[-1] if fitting else _MANTISSAS[0]
    for mantissas in _MANTISSAS[_MANTISSAS.index(choice):]:
        choice = mantissas
        if len(inside(mantissas)) >= 2:
            break
    ticks.append((None, inside(choice)))
    return ticks



@dataclass(frozen=True)
class Marker:
    """A curve drawn as points of one symbol (``set_curves(markers=…)``): a pyqtgraph symbol (``"o"``, ``"s"``,
    ``"x"`` …), its size in pixels, and ``hollow`` for an outline only (e.g. where a path starts)."""

    symbol: str = "o"
    size: float = 7.0
    hollow: bool = False


SIDES = (("±", "both"), ("+", "positive"), ("−", "negative"), ("|x|", "folded"))
SIDE_TIPS = (
    "Both halves, as measured",
    "Only the positive half (x > 0)",
    "Only the negative half (x < 0)",
    "Both halves on |x|, the negative one dashed in a second shade: are they the same?",
)


class _Title(QLabel):
    """The plot's title, shortened with “…” when the header is narrow; ``text()`` is the whole title."""

    def __init__(self, text: str, parent: QWidget):
        super().__init__(parent)
        self._full = ""
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = str(text)
        self._refresh()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(0, self.width())))


def _has_both_signs(x) -> bool:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    return finite.size > 1 and float(finite.min()) < 0 < float(finite.max())


def _halves(name: str, x, y, side: str) -> list:
    """``(name, x, y, half)`` of what to draw of one curve for the chosen half (display only).

    ``half``: ``""`` as measured, ``"+"`` / ``"−"`` one half, ``"|+|"`` / ``"|−|"`` a half drawn on |x|
    (the negative one mirrored: its x is −x, dashed and in a second shade)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if side == "both" or not _has_both_signs(x):
        return [(name, x, y, "")]
    positive, negative = x > 0, x < 0
    if side == "positive":
        return [(name, x[positive], y[positive], "+")]
    if side == "negative":
        return [(name, x[negative], y[negative], "−")]
    order = np.argsort(-x[negative], kind="stable")  # the negative half mirrored, in increasing |x|
    return [
        (f"{name} (+)" if name else "", x[positive], y[positive], "|+|"),
        (f"{name} (−)" if name else "", -x[negative][order], y[negative][order], "|−|"),
    ]


def second_shade(color) -> str:
    """The colour of a folded negative half: the curve's own hue, darker on a light plot and lighter on a
    dark one, so the two halves differ in colour as well as in dash (display only)."""
    import pyqtgraph as pg

    base = pg.mkColor(color)
    return (base.lighter(150) if theme_manager().is_dark else base.darker(170)).name()


def legend_text(name: str) -> str:
    """``name`` as a legend shows it whole: pyqtgraph draws a legend entry as rich text, where “<”, “>” and
    “&” would begin a tag or an entity and take the rest of the name with them (``measured (q < 0)``)."""
    return html.escape(str(name), quote=False)


__all__ = ["CURVE_COLORS", "LEGEND_ALPHA", "Marker", "SIDES", "SIDE_TIPS", "legend_text", "readable_log_ticks",
           "second_shade"]
