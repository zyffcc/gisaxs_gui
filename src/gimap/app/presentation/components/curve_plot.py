"""Embeddable pyqtgraph plot for reduced 1D curves."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from src.gimap.shared.figures import break_at_gaps

from ..theme import theme_manager
from .marks import MarkLayers
from .box_zoom import tool_icon, zoom_button
from .segmented import SegmentedControl

CURVE_COLORS = ("#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2")
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



SIDES = (("±", "both"), ("+", "positive"), ("−", "negative"), ("|x|", "folded"))
SIDE_TIPS = (
    "Both halves, as measured",
    "Only the positive half (x > 0)",
    "Only the negative half (x < 0)",
    "Both halves on |x|, the negative one dashed: are they the same?",
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
    """``(name, x, y, pen style)`` of what to draw of one curve for the chosen half (display only)."""
    from PyQt5.QtCore import Qt

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if side == "both" or not _has_both_signs(x):
        return [(name, x, y, Qt.SolidLine)]
    positive, negative = x > 0, x < 0
    if side == "positive":
        return [(name, x[positive], y[positive], Qt.SolidLine)]
    if side == "negative":
        return [(name, x[negative], y[negative], Qt.SolidLine)]
    order = np.argsort(-x[negative], kind="stable")  # the negative half mirrored, in increasing |x|
    return [
        (f"{name} (+)", x[positive], y[positive], Qt.SolidLine),
        (f"{name} (−)", -x[negative][order], y[negative][order], Qt.DashLine),
    ]


class CurvePlot(QWidget):
    """Titled plot of one or more curves with a log-y toggle and an optional x window."""

    windowChanged = pyqtSignal(float, float)
    curveClicked = pyqtSignal(int)
    """The index (in the last ``set_curves``) of a curve the user clicked."""
    positionClicked = pyqtSignal(float, float)
    """``(x, y)`` in data units (not log10) of a left click on the plot."""
    saveFigureRequested = pyqtSignal()
    saveDataRequested = pyqtSignal()
    """From the header's Save menu (``add_save_menu``)."""

    def __init__(
        self, title: str = "", parent: Optional[QWidget] = None, *, log_y: bool = True, log_x: Optional[bool] = None,
    ):
        """``log_x``: ``None`` keeps x linear (no toggle); ``True``/``False`` adds a "Log q" toggle."""
        import pyqtgraph as pg

        super().__init__(parent)
        self.setObjectName("curvePlot")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = _Title(title, self)
        self.title_label.setObjectName("curvePlotTitle")
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.log_check = QCheckBox("Log I", self)
        self.log_check.setChecked(log_y)
        self.log_x_check = QCheckBox("Log q", self)
        self.log_x_check.setChecked(bool(log_x))
        self.log_x_check.setVisible(log_x is not None)
        header.addWidget(self.title_label, 1)
        header.addWidget(self.log_x_check)
        header.addWidget(self.log_check)
        layout.addLayout(header)
        self.plot_widget = pg.PlotWidget(self)
        self.plot = self.plot_widget.getPlotItem()
        # Which half of a curve with a signed x (qy of a GISAXS cut, χ of a ring) to show; shown only then.
        self.side_control = SegmentedControl(self)
        self.side_control.setObjectName("curvePlotSides")
        for text, key in SIDES:
            self.side_control.addItem(text, key)
        for index, tip in enumerate(SIDE_TIPS):
            self.side_control.setItemToolTip(index, tip)
        self.side_control.hide()
        self._side = "both"
        self.side_control.activated.connect(self._side_chosen)
        header.insertWidget(1, self.side_control)
        icon_color = theme_manager().color("plot_fg")
        self.zoom_button = zoom_button(self.plot.getViewBox(), self, icon_color=icon_color)
        self.reset_button = QToolButton(self)
        self.reset_button.setText("Reset")
        self.reset_button.setIcon(tool_icon("fit", icon_color))
        self.reset_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.reset_button.setAutoRaise(True)
        self.reset_button.setObjectName("curvePlotReset")
        self.reset_button.setToolTip("Show every curve again (after zooming or panning)")
        self.reset_button.clicked.connect(self.reset_view)
        self.marks = MarkLayers(self)
        self.marks.add_layer("window", "Window band")
        self.marks_button = self.marks.menu_button(self, icon=tool_icon("marks", icon_color))
        header.addWidget(self.zoom_button)
        header.addWidget(self.reset_button)
        header.addWidget(self.marks_button)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        for side in ("left", "bottom"):
            # Scientific units are in the label; never rescale them to "x1e+06".
            self.plot.getAxis(side).enableAutoSIPrefix(False)
            self.plot.getAxis(side).logTickValues = readable_log_ticks
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plot_widget.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.plot.setLogMode(x=bool(log_x), y=log_y)
        layout.addWidget(self.plot_widget, 1)
        self.x_window = pg.LinearRegionItem(
            orientation="vertical", brush=(249, 115, 22, 14), pen=pg.mkPen("#f97316", width=1.5)
        )
        self.x_window.setZValue(10)
        self.plot.addItem(self.x_window, ignoreBounds=True)
        self.x_window.hide()
        self._updating_window = False
        self._items = []
        self._source: list[tuple[str, np.ndarray, np.ndarray]] = []
        self._source_colors: Optional[list] = None
        self._source_markers: Optional[list] = None
        self._curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        """The curves as shown (after choosing a half); ``figure_state`` and the log range use them."""
        self._colors: list[str] = []
        self._styles: list = []
        self._origin: list[int] = []
        """For each curve shown, the index of the curve it comes from (``curveClicked``)."""
        self.log_check.toggled.connect(self._log_toggled)
        self.log_x_check.toggled.connect(self._log_toggled)
        self.x_window.sigRegionChangeFinished.connect(self._window_finished)
        self._apply_theme()
        theme_manager().changed.connect(self._apply_theme)

    def _apply_theme(self, *_args) -> None:
        """Plot background and axes follow the light/dark theme."""
        import pyqtgraph as pg

        manager = theme_manager()
        foreground = manager.color("plot_fg")
        self.plot_widget.setBackground(manager.color("plot_bg"))
        for side in ("left", "bottom"):
            axis = self.plot.getAxis(side)
            axis.setPen(pg.mkPen(foreground))
            axis.setTextPen(pg.mkPen(foreground))
        self.legend.setLabelTextColor(foreground)
        if hasattr(self, "reset_button"):
            self.zoom_button.setIcon(tool_icon("zoom", foreground))
            self.reset_button.setIcon(tool_icon("fit", foreground))
            self.marks_button.setIcon(tool_icon("marks", foreground))
        if self._source:
            self.set_curves(self._source, self._source_colors, markers=self._source_markers)  # legend labels take the new colour

    def set_title(self, title: str) -> None:
        from ..i18n import tr

        self.title_label.setText(tr(title))
        self.title_label.setToolTip(tr(title))

    def set_labels(self, x_label: str, y_label: str) -> None:
        self.plot.setLabel("bottom", x_label)  # figure content: not translated
        self.plot.setLabel("left", y_label)
        symbol = str(x_label).split(" (")[0].split(" or ")[0].strip().strip("|") or "x"  # “χ or |χ| (°)” → χ
        self.side_control.button(3).setText(f"|{symbol}|")

    def set_side(self, side: str) -> None:
        """``both``, ``positive``, ``negative`` or ``folded`` (both halves on |x|, the negative one dashed)."""
        index = self.side_control.findData(side)
        if index >= 0:
            self.side_control.setCurrentIndex(index)
            self._side_chosen(index)

    def side(self) -> str:
        return self._side

    def _side_chosen(self, index: int) -> None:
        self._side = self.side_control.itemData(index) or "both"
        if self._source:
            self.set_curves(self._source, self._source_colors, markers=self._source_markers)

    def clear_curves(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items.clear()
        self.legend.clear()
        self.side_control.hide()  # shown again by ``set_curves`` when a curve has both signs

    def _log_toggled(self, _checked: bool) -> None:
        self.plot.setLogMode(x=self.log_x_check.isChecked(), y=self.log_check.isChecked())
        self.set_curves(self._source, self._source_colors, markers=self._source_markers)

    def add_save_menu(self) -> None:
        """A “Save” button in the header: the plot as a figure, or its curves as data."""
        from PyQt5.QtWidgets import QMenu, QToolButton

        button = QToolButton(self)
        button.setObjectName("curvePlotSave")
        button.setText("Save")
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        menu.addAction("Plot as Figure…", self.saveFigureRequested.emit)
        menu.addAction("Curves as Data…", self.saveDataRequested.emit)
        button.setMenu(menu)
        self.header_layout.insertWidget(1, button)
        button.setToolTip("Save this plot as a figure, or its curves as data")
        self.save_button = button

    def set_curves(self, curves: Sequence[tuple[str, np.ndarray, np.ndarray]], colors: Optional[Sequence[str]] = None,
                   *, markers: Optional[Sequence[bool]] = None) -> None:
        """``curves`` is a sequence of ``(name, x, y)``; non-positive y is hidden in log mode.

        ``colors``: one colour per curve (e.g. the colour of the region it comes from); by default
        the plot's own sequence. ``markers``: per curve, points instead of a line (measured data) —
        ``True`` for dots or a pyqtgraph symbol (``"x"``, ``"s"`` …).
        """
        import pyqtgraph as pg

        from ..i18n import tr

        curves = list(curves)
        self.clear_curves()
        self._source = curves
        self._source_colors = None if colors is None else list(colors)
        self._source_markers = None if markers is None else list(markers)
        source_colors = [
            (colors[index] if colors is not None and index < len(colors) and colors[index] else CURVE_COLORS[index % len(CURVE_COLORS)])
            for index in range(len(curves))
        ]
        signed = any(_has_both_signs(x) for _name, x, _y in curves)
        self.side_control.setVisible(signed)
        self._curves, self._colors, self._styles, self._origin = [], [], [], []
        self._markers = [markers[index] if markers is not None and index < len(markers) else False
                         for index in range(len(curves))]
        for index, (name, x, y) in enumerate(curves):
            for shown in _halves(name, x, y, self._side if signed else "both"):
                self._curves.append(shown[:3])
                self._colors.append(source_colors[index])
                self._styles.append(shown[3])
                self._origin.append(index)
        for index, (name, x, y) in enumerate(self._curves):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            if self.log_check.isChecked():
                y = np.where(y > 0, y, np.nan)
            if self.log_x_check.isChecked():
                y = np.where(x > 0, y, np.nan)
                x = np.where(x > 0, x, np.nan)
            marker = self._markers[self._origin[index]]
            if marker:
                keep = np.isfinite(x) & np.isfinite(y)
                symbol = marker if isinstance(marker, str) else "o"
                item = self.plot.plot(x[keep], y[keep], pen=None, symbol=symbol, symbolSize=4 if symbol == "o" else 7,
                                      symbolPen=None if symbol == "o" else pg.mkPen(self._colors[index], width=1.4),
                                      symbolBrush=self._colors[index], name=tr(name))
                self._items.append(item)
                continue
            x, y = break_at_gaps(x, y)  # no line across bins without pixels
            pen = pg.mkPen(self._colors[index], width=1.6, style=self._styles[index])
            item = self.plot.plot(x, y, pen=pen, name=tr(name), connect="finite")
            item.setCurveClickable(True, width=8)
            origin = self._origin[index]
            item.sigClicked.connect(lambda _item, *_args, origin=origin: self.curveClicked.emit(origin))
            self._items.append(item)
        self.plot.enableAutoRange()
        if self.log_check.isChecked():
            self._robust_log_range()

    def reset_view(self) -> None:
        """Every curve in view again (after a zoom or a pan)."""
        self.plot.enableAutoRange()
        if self.log_check.isChecked():
            self._robust_log_range()

    def _robust_log_range(self) -> None:
        """On a log axis, let the bulk of the data set the range: a few bins near zero (noise of
        dark-subtracted data, a bin with one pixel) would otherwise stretch it over many decades.
        The lower limit follows each curve after a 9-point running median, so a smooth tail stays in view."""
        from scipy.ndimage import median_filter

        lows, highs = [], []
        for _name, _x, y in self._curves:
            y = np.asarray(y, dtype=np.float64)
            y = y[np.isfinite(y) & (y > 0)]
            if y.size < 3:
                continue
            smooth = median_filter(np.log10(y), size=min(9, y.size) | 1, mode="nearest")
            lows.append(float(smooth.min()))
            highs.append(float(np.log10(y.max())))
        if not lows:
            return
        low, high = min(lows), max(highs)
        if high <= low:
            return
        margin = 0.05 * (high - low) + 0.05
        self.plot.enableAutoRange(axis="y", enable=False)
        self.plot.setYRange(low - margin, high + margin, padding=0)

    def _mouse_clicked(self, event) -> None:
        from PyQt5.QtCore import Qt

        if event.button() != Qt.LeftButton or not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.getViewBox().mapSceneToView(event.scenePos())
        x, y = float(point.x()), float(point.y())
        if self.log_x_check.isChecked():
            x = 10.0 ** x
        if self.log_check.isChecked():
            y = 10.0 ** y
        self.positionClicked.emit(x, y)

    def curve_count(self) -> int:
        return len(self._items)

    def curve_colors(self) -> list[str]:
        """The colour of each curve shown, in ``set_curves`` order."""
        return list(self._colors)

    def figure_state(self) -> dict:
        """The curves and labels shown, for exporting the plot as a figure."""
        return {
            "curves": list(self._curves),
            "colors": list(self._colors),
            "x_label": self.plot.getAxis("bottom").labelText,
            "y_label": self.plot.getAxis("left").labelText,
            "log_y": self.log_check.isChecked(),
            "log_x": self.log_x_check.isChecked(),
            "title": self.title_label.text(),
        }

    def dispose(self) -> None:
        """Delete the pyqtgraph view before this widget's parent is destroyed."""
        try:
            theme_manager().changed.disconnect(self._apply_theme)
        except TypeError:
            pass
        self._items.clear()
        self._curves = []
        self._colors = []
        if not sip.isdeleted(self.plot_widget):
            self.plot_widget.deleteLater()

    def show_window(self, low: float, high: float) -> None:
        self._updating_window = True
        try:
            self.x_window.setRegion((float(low), float(high)))
            self.marks.track("window", self.x_window, True)
        finally:
            self._updating_window = False

    def hide_window(self) -> None:
        self.marks.track("window", self.x_window, False)

    def _window_finished(self) -> None:
        if not self._updating_window and self.marks.wanted(self.x_window):
            low, high = self.x_window.getRegion()
            self.windowChanged.emit(float(low), float(high))


__all__ = ["CURVE_COLORS", "CurvePlot"]
