"""The parts of ``CurvePlot`` around the plot itself.

* ``CompactHeader`` — a header that fits narrow plots, in two independent steps. Below the width the
  title and every header control need, the title moves to a wrapped row of its own above the header.
  Only below the width the controls alone need do the halves (± + − |x|) and the Log q / Log I toggles
  move into a small “⋯” menu whose entries drive the same controls (one source of truth). Each choice
  depends only on the plot's width (with a little hysteresis), never on what the switch itself changes,
  so it cannot flip back and forth.
* ``PlotMenu`` — the right-click menu, in the interface language (pyqtgraph's own English menu is off):
  Reset View, Log I / Log q, the Save entries, Copy Image and Copy Data. A right click opens it (pyqtgraph's
  click logic), a right drag stays pyqtgraph's zoom; the menu key opens it too.
* ``curves_text`` — the curves as shown, as tab-separated text with a header row (Copy Data); a
  curve drawn on |x| (``folded_label``) says so in its header.
* ``CursorReadout`` — the values under the cursor (“q = 0.0213 Å⁻¹ · I = 1.52e+04”) floating over a bottom
  corner of the plot area (never in a layout: the plot's size and header stay); ``EmptyOverlay`` — a muted
  sentence centred on the plot area while the plot has no curves (``set_empty_text``).
* ``IconTools`` — tool buttons that show a line icon instead of their text in a narrow view.
* ``roomy_ticks`` — axis ticks whose labels never run together on a short axis.
* ``style_window`` — the colour of the x-window band.

The plot's values are never changed here: copying writes the numbers that are drawn, in data units.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QContextMenuEvent, QIcon
from PyQt5.QtWidgets import QActionGroup, QLabel, QMenu, QSizePolicy, QToolButton, QWidget

from ..theme import theme_manager
from ..theme.tokens import font_tokens
from .box_zoom import tool_icon

MORE_TEXT = "⋯"
HYSTERESIS = 24
"""px: a compact header turns full again only this much above the width it needs."""
WINDOW_COLOR = "#f97316"
"""The x-window band's own colour (orange)."""
WINDOW_FILL_ALPHA = 14
WINDOW_HOVER_ALPHA = 28  # pyqtgraph's own hover: twice the fill
READOUT_RATE = 30  # cursor readouts per second at most (pyqtgraph's ``SignalProxy``)
OVERLAY_MARGIN = 4  # px between the readout and the edge of the plot area
READOUT_AVOID = 12  # px: the readout goes to the other bottom corner when the cursor comes this close
OVERLAY_ALPHA = 215  # the readout's and the empty text's background: no grid line runs through the text


def _least_width(widget: QWidget) -> int:
    """The narrowest a layout may make ``widget`` (as Qt's own layouts decide it)."""
    policy = widget.sizePolicy().horizontalPolicy()
    if policy == QSizePolicy.Ignored:
        return widget.minimumWidth()
    hint = widget.minimumSizeHint().width() if int(policy) & int(QSizePolicy.ShrinkFlag) else widget.sizeHint().width()
    return max(hint, widget.minimumWidth())


def axis_symbol(x_label: str) -> str:
    """The quantity of an axis label: “qy (Å⁻¹)” → qy, “χ or |χ| (°)” → χ (``x`` when there is none)."""
    return str(x_label or "").split(" (")[0].split(" or ")[0].strip().strip("|") or "x"


def folded_label(x_label: str) -> str:
    """The label of a curve drawn on |x|: “qy (Å⁻¹)” → “|qy| (Å⁻¹)”, “χ or |χ| (°)” → “|χ| (°)”."""
    text = str(x_label or "")
    unit = text.rfind(" (")
    return f"|{axis_symbol(text)}|" + (text[unit:] if unit >= 0 else "")


class CompactHeader(QObject):
    """The wrapped title row and the “⋯” menu of a ``CurvePlot`` (see the module docstring)."""

    def __init__(self, plot):
        super().__init__(plot)
        self.plot = plot
        self.wrapped = False
        """The title is on a row of its own above the header."""
        self.compact = False
        """The halves and the log toggles are in the “⋯” menu."""
        self._hidden: list[QWidget] = []
        """Controls hidden by the compact header (wanted: shown again when the plot is wide)."""
        title = plot.title_label
        policy = title.sizePolicy()
        policy.setRetainSizeWhenHidden(True)  # hidden, it still pushes the controls to the right
        title.setSizePolicy(policy)
        self.title = QLabel(plot)
        self.title.setObjectName("curvePlotTitleWrapped")
        self.title.setWordWrap(True)
        wrapped = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        wrapped.setHeightForWidth(True)
        self.title.setSizePolicy(wrapped)
        self.title.hide()
        plot.layout().insertWidget(0, self.title)  # above the header row: the header's indices stay
        title.installEventFilter(self)  # its tooltip (often the whole sentence) goes to the wrapped title too
        self._side_group = QActionGroup(self)
        """The halves' entries of the “⋯” menu (one group, refilled each time the menu opens)."""
        self.more_button = QToolButton(plot)
        self.more_button.setObjectName("curvePlotMore")
        self.more_button.setText(MORE_TEXT)
        self.more_button.setAutoRaise(True)
        self.more_button.setPopupMode(QToolButton.InstantPopup)
        self.more_button.setToolTip("Halves of the curve and log axes")
        menu = QMenu(self.more_button)
        menu.aboutToShow.connect(lambda: self.fill_menu(menu))
        self.more_button.setMenu(menu)
        self.more_button.hide()
        header = plot.header_layout
        header.insertWidget(max(0, header.indexOf(plot.zoom_button)), self.more_button)

    # -- what the header holds -------------------------------------------------------------

    def _compactable(self) -> tuple:
        return self.plot.side_control, self.plot.log_x_check, self.plot.log_check

    def _controls(self) -> list[QWidget]:
        layout = self.plot.header_layout
        widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
        return [widget for widget in widgets
                if widget is not None and widget is not self.plot.title_label and widget is not self.more_button]

    def wanted(self, widget: QWidget) -> bool:
        """Whether ``widget`` is in the full header (shown, or hidden only by the compact header)."""
        return widget in self._hidden or not widget.isHidden()

    def controls_width(self) -> int:
        """The width of the header row with every control in it and no title (the halves and log toggles
        included even while they are in the “⋯” menu, so the switch does not change it)."""
        shown = [widget for widget in self._controls() if self.wanted(widget)]
        spacing = max(0, self.plot.header_layout.spacing())
        return sum(widget.sizeHint().width() + spacing for widget in shown)

    def full_width(self) -> int:
        """The width of the header row with the whole title and every control in it."""
        spacing = max(0, self.plot.header_layout.spacing())
        text = self.plot.title_label.text()
        title = self.plot.title_label.fontMetrics().horizontalAdvance(text) + spacing if text else 0
        return title + self.controls_width()

    def least_width(self) -> int:
        """The narrowest the plot can be: the compact header row, or the plot area."""
        compactable = self._compactable()
        shown = [widget for widget in self._controls() if self.wanted(widget) and widget not in compactable]
        if any(self.wanted(widget) for widget in compactable):
            shown.append(self.more_button)
        spacing = max(0, self.plot.header_layout.spacing())
        header = sum(_least_width(widget) for widget in shown) + spacing * max(0, len(shown) - 1)
        margins = self.plot.layout().contentsMargins()
        view = self.plot.plot_widget  # deleted by ``CurvePlot.dispose`` while the plot may still be laid out
        area = 0 if sip.isdeleted(view) else view.minimumSizeHint().width()
        return max(header, area) + margins.left() + margins.right()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.ToolTipChange and watched is self.plot.title_label:
            self.title.setToolTip(watched.toolTip())
        return False

    # -- switching -------------------------------------------------------------------------

    def update(self, width: Optional[int] = None) -> None:
        """Two choices, each with its own threshold: the title on its own row below ``full_width()``, the
        halves and log toggles in the “⋯” menu only below ``controls_width()``. Each turns back from its
        threshold ``+ HYSTERESIS`` (a scroll bar that comes with the taller header cannot flip it back and forth)."""
        width = self.plot.width() if width is None else int(width)
        self.wrapped = width < self.full_width() + (HYSTERESIS if self.wrapped else 0)
        compact = width < self.controls_width() + (HYSTERESIS if self.compact else 0)
        if compact != self.compact:
            self._set_compact(compact)
        text, tip = self.plot.title_label.text(), self.plot.title_label.toolTip()
        if self.title.text() != text:
            self.title.setText(text)
        if self.title.toolTip() != tip:
            self.title.setToolTip(tip)
        self.title.setVisible(self.wrapped and bool(text))
        self.plot.title_label.setVisible(not self.wrapped)
        self.more_button.setVisible(compact and any(self.wanted(widget) for widget in self._compactable()))

    def _set_compact(self, compact: bool) -> None:
        self.compact = compact
        if compact:
            for widget in self._compactable():
                if not widget.isHidden():
                    widget.hide()
                    self._hidden.append(widget)
        else:
            for widget in self._hidden:
                widget.show()
            self._hidden = []

    def show_sides(self, shown: bool) -> None:
        """Whether the curves offer a choice of halves (``set_curves``); in the “⋯” menu while compact."""
        side = self.plot.side_control
        if self.compact:
            side.hide()
            if shown and side not in self._hidden:
                self._hidden.append(side)
            elif not shown and side in self._hidden:
                self._hidden.remove(side)
        else:
            side.setVisible(bool(shown))
        self.update()

    def fill_menu(self, menu: QMenu) -> None:
        """The hidden controls as menu entries that drive the controls themselves."""
        from ..i18n import tr

        menu.clear()
        for check in (self.plot.log_check, self.plot.log_x_check):  # as in the right-click menu
            if check in self._hidden:
                _check_action(menu, tr(check.text()), check)
        side = self.plot.side_control
        if side in self._hidden:
            menu.addSeparator()
            for index in range(side.count()):  # ``menu.clear`` deletes them, which takes them out of the group
                action = menu.addAction(tr(side.button(index).toolTip() or side.itemText(index)))
                action.setCheckable(True)
                action.setChecked(index == side.currentIndex())
                self._side_group.addAction(action)
                action.triggered.connect(lambda _on=False, key=side.itemData(index): self.plot.set_side(key))


def _check_action(menu: QMenu, text: str, check) -> None:
    action = menu.addAction(text)
    action.setCheckable(True)
    action.setChecked(check.isChecked())
    action.toggled.connect(check.setChecked)


class PlotMenu(QObject):
    """The plot's right-click menu (see the module docstring).

    A right *click* opens it, as pyqtgraph decides one (``sigMouseClicked`` comes only when the button was
    not dragged), so a right drag stays pyqtgraph's zoom on every system: Qt's own context-menu request
    comes with the press on macOS and Linux, before anyone knows whether a drag follows, and is dropped
    here. The keyboard's request (the menu key, Shift+F10) still opens the menu (``popup``).
    """

    def __init__(self, plot):
        super().__init__(plot)
        self.plot = plot
        self.menu = QMenu(plot.plot_widget)
        self.menu.setObjectName("curvePlotMenu")
        widget = plot.plot_widget
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(self.popup)  # only the keyboard's request gets there
        widget.installEventFilter(self)
        widget.viewport().installEventFilter(self)
        widget.scene().sigMouseClicked.connect(self._clicked)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        """Drop the mouse's context-menu request (a right click opens the menu in ``_clicked``)."""
        return isinstance(event, QContextMenuEvent) and event.reason() == QContextMenuEvent.Mouse

    def _clicked(self, event) -> None:
        if event.button() != Qt.RightButton or event.double():
            return
        if getattr(self.plot.plot_widget.scene(), "dragItem", None) is not None:
            return  # a right click during another button's drag: that item's (it cancels the drag)
        event.accept()
        self.fill()
        self.menu.popup(event.screenPos().toPoint())

    def popup(self, position: QPoint) -> None:
        """The menu at ``position`` (the plot widget's coordinates; its middle when outside it)."""
        widget = self.plot.plot_widget
        if not widget.rect().contains(position):
            position = widget.rect().center()
        self.fill()
        self.menu.popup(widget.mapToGlobal(position))

    def fill(self) -> QMenu:
        from ..i18n import tr

        plot, menu = self.plot, self.menu
        menu.clear()
        menu.addAction(tr("Reset View"), plot.reset_view)
        checks = [check for check in (plot.log_check, plot.log_x_check) if plot.compact_header.wanted(check)]
        if checks:
            menu.addSeparator()
        for check in checks:
            _check_action(menu, tr(check.text()), check)
        if getattr(plot, "save_button", None) is not None:
            menu.addSeparator()
            menu.addAction(tr("Plot as Figure…"), plot.saveFigureRequested.emit)
            menu.addAction(tr("Curves as Data…"), plot.saveDataRequested.emit)
        menu.addSeparator()
        menu.addAction(tr("Copy Image"), plot.copy_image)
        menu.addAction(tr("Copy Data"), plot.copy_data).setEnabled(bool(plot.figure_state()["curves"]))
        return menu


def curves_text(curves, x_label: str = "", y_label: str = "", *, x_labels=None) -> str:
    """``(name, x, y)`` curves as tab-separated columns (x and y of each) under a header row.

    ``x_labels``: one x label per curve where they differ (a half mirrored onto |x| is “|qy| (Å⁻¹)”, not
    “qy”); ``x_label`` for the others. The numbers are the stored values (shortest exact form of their
    type); a missing value (NaN, or past the end of a shorter curve) is an empty cell.
    """
    header, columns = [], []
    for index, (name, x, y) in enumerate(curves):
        label = str(name) or f"curve {index + 1}"
        own = x_labels[index] if x_labels is not None and index < len(x_labels) and x_labels[index] else x_label
        header += [f"{label}: {own or 'x'}", f"{label}: {y_label or 'y'}"]
        columns += [np.asarray(x).ravel(), np.asarray(y).ravel()]

    def cell(column, row) -> str:
        if row >= column.size:
            return ""
        value = column[row]
        try:
            return str(value) if np.isfinite(value) else ""
        except TypeError:
            return str(value)

    rows = max((column.size for column in columns), default=0)
    lines = ["\t".join(header)] + ["\t".join(cell(column, row) for column in columns) for row in range(rows)]
    return "\n".join(lines) + "\n"


# -- over the plot area: the cursor readout and the text of an empty plot ----------------------------


def view_rect(plot) -> QRect:
    """The plot area (pyqtgraph's view box, without the axes) in the plot widget's coordinates."""
    widget = plot.plot_widget
    return widget.mapFromScene(plot.plot.getViewBox().sceneBoundingRect()).boundingRect().translated(widget.viewport().pos())


def axis_parts(label: str, *, folded: bool = False, fallback: str = "x") -> tuple[str, str]:
    """Quantity and unit of an axis label: “q (Å⁻¹)” → (q, Å⁻¹), “χ or |χ| (°)” → (χ, °) or, while the halves
    are drawn on |x| (``folded``), (|χ|, °); “Intensity” → (I, “”)."""
    text, unit = str(label or "").strip(), ""
    start = text.rfind(" (")
    if start >= 0 and text.endswith(")"):
        text, unit = text[:start], text[start + 2:-1].strip()
    symbol = text.split(" or ")[0].strip() or fallback
    if folded:
        return f"|{symbol.strip('|')}|", unit
    return ("I" if symbol.lower() == "intensity" else symbol), unit


def _quantity(symbol: str, value: float, unit: str) -> str:
    return f"{symbol} = {value:.4g}" + (unit if unit in ("°", "%") else f" {unit}" if unit else "")


def readout_text(plot, x: float, y: float) -> str:
    """“q = 0.0213 Å⁻¹ · I = 1.52e+04 counts/pixel” at the view position ``(x, y)`` of a ``CurvePlot``: an axis
    on log holds log10 of the value (10**v only there); quantities and units from the axis labels."""
    with np.errstate(over="ignore"):
        x = float(np.power(10.0, x)) if plot.log_x_check.isChecked() else x
        y = float(np.power(10.0, y)) if plot.log_check.isChecked() else y
    folded = any(half.startswith("|") for half in plot._shown_halves)
    x_symbol, x_unit = axis_parts(plot.plot.getAxis("bottom").labelText, folded=folded)
    y_symbol, y_unit = axis_parts(plot.plot.getAxis("left").labelText, fallback="y")
    return f"{_quantity(x_symbol, x, x_unit)} · {_quantity(y_symbol, y, y_unit)}"


class _Overlay(QObject):
    """A label over the plot area of a ``CurvePlot``: no mouse (the plot under it gets every event), in no
    layout (the plot's size and header never change), muted text on the plot's background colour."""

    SMALL, BORDER, PADDING = False, False, (10, 6)

    def __init__(self, plot, name: str):
        super().__init__(plot)
        self.plot = plot
        self.label = QLabel(plot.plot_widget, objectName=name, visible=False)
        self.label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.label.setContentsMargins(*self.PADDING, *self.PADDING)
        plot.plot.getViewBox().sigResized.connect(self._view_resized)
        theme_manager().changed.connect(self._style)
        self._style()

    def alive(self) -> bool:  # the label goes with the plot widget (``CurvePlot.dispose``)
        return not sip.isdeleted(self.label)

    @pyqtSlot()
    @pyqtSlot(str)
    def _style(self, *_args) -> None:
        if not self.alive():
            return
        manager = theme_manager()
        fill = manager.color("plot_bg")
        size = font_tokens(manager.font_pt)["font_small_pt" if self.SMALL else "font_pt"]
        border = f"1px solid {manager.color('plot_grid').name()}" if self.BORDER else "none"
        self.label.setStyleSheet(
            f"QLabel {{ color: {manager.color('text_muted').name()}; border: {border}; border-radius: 4px; "
            f"background: rgba({fill.red()}, {fill.green()}, {fill.blue()}, {OVERLAY_ALPHA}); font-size: {size}; }}")
        self._view_resized()

    def _view_resized(self, *_args) -> None:
        if self.alive() and not self.label.isHidden():
            self.place()


class EmptyOverlay(_Overlay):
    """The sentence of an empty plot (``CurvePlot.set_empty_text``), centred on the plot area while it has no
    curves; the English is kept and shown with ``tr`` (when shown, and after each switch of the language)."""

    def __init__(self, plot):
        from ..i18n import language_changed

        super().__init__(plot, "curvePlotEmpty")
        self.text = ""
        self.bare = False  # the plot shows the empty text, without ticks and grid
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        language_changed().connect(self.refresh)

    def set_text(self, text: str) -> None:
        self.text = str(text or "")
        self.refresh()

    @pyqtSlot()
    @pyqtSlot(str)
    def refresh(self, *_args) -> None:
        """Shown while there is a sentence and the plot has no curves, hidden as soon as it has some."""
        from ..i18n import tr

        if not self.alive():
            return
        shown = bool(self.text) and not self.plot.has_curves()
        if shown != self.bare:  # an empty frame: no 0–1 ticks, values or grid that mean nothing yet
            self.bare = shown
            for side in ("left", "bottom"):
                self.plot.plot.getAxis(side).setTicks([] if shown else None)  # ``None``: pyqtgraph's own again
        if shown:
            self.label.setText(tr(self.text))
            self.label.raise_()
            self.place()
        self.label.setVisible(shown)

    def place(self) -> None:
        label, area = self.label, view_rect(self.plot)
        label.ensurePolished()
        natural = label.fontMetrics().horizontalAdvance(label.text()) + 2 * self.PADDING[0] + 4
        width = max(24, min(natural, area.width() - 4 * OVERLAY_MARGIN, 420))
        height = label.heightForWidth(width)
        height = height if height > 0 else label.sizeHint().height()
        label.setGeometry(area.center().x() - width // 2, area.center().y() - height // 2, width, height)


class CursorReadout(_Overlay):
    """The values under the cursor (``readout_text``) in the bottom corner of the plot area away from the cursor,
    while the cursor is over the plot area of a plot with curves. Moves come through a ``SignalProxy`` (at most
    ``READOUT_RATE`` a second); a zoom, a pan or new curves under a still cursor update it too."""

    SMALL, BORDER, PADDING = True, True, (5, 1)

    def __init__(self, plot):
        import pyqtgraph as pg

        super().__init__(plot, "curvePlotReadout")
        self.text = ""  # the whole readout (the label shortens it to the width of the plot area)
        self.right = False  # in the bottom-right corner (the cursor came close to the bottom-left one)
        self._position = None  # the cursor's last scene position over the plot (``None`` once it left)
        scene = plot.plot_widget.scene()
        scene.sigMouseMoved.connect(self._seen)
        self._proxy = pg.SignalProxy(scene.sigMouseMoved, rateLimit=READOUT_RATE, slot=self.refresh)
        plot.plot_widget.viewport().installEventFilter(self)
        plot.plot.getViewBox().sigRangeChanged.connect(self.refresh)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Leave, QEvent.Hide):  # gone, or the page with the plot is (no stale values)
            self.clear()
        return False

    def _seen(self, position) -> None:  # every move, before the proxy: the position ``refresh`` uses
        self._position = position

    def refresh(self, *_args) -> None:
        """At the cursor's last position (a move; new curves, range or log scale under a still cursor); not
        after the cursor left (a move still waiting in the proxy)."""
        if self._position is not None:
            self.show_at(self._position)

    def clear(self) -> None:  # nothing shown until the cursor moves over the plot again
        self._position = None
        if self.alive():
            self.label.hide()

    def show_at(self, position) -> None:
        """The values at the scene ``position``; nothing outside the plot area or on a plot without curves."""
        if not self.alive():
            return
        box = self.plot.plot.getViewBox()
        if not self.plot.has_curves() or not box.sceneBoundingRect().contains(position):
            self.label.hide()
            return
        point = box.mapSceneToView(position)
        self.text = readout_text(self.plot, float(point.x()), float(point.y()))
        self.label.show()
        self.label.raise_()
        self.place(position)

    def place(self, position=None) -> None:
        label, area, widget = self.label, view_rect(self.plot), self.plot.plot_widget
        room = max(0, area.width() - 2 * OVERLAY_MARGIN)
        label.ensurePolished()
        label.setText(self.text)
        chrome = label.sizeHint().width() - label.fontMetrics().horizontalAdvance(self.text)
        if label.sizeHint().width() > room:
            label.setText(label.fontMetrics().elidedText(self.text, Qt.ElideRight, max(0, room - chrome)))
        label.resize(min(label.sizeHint().width(), room), label.sizeHint().height())
        width, height = label.width(), label.height()
        top = area.bottom() - OVERLAY_MARGIN - height + 1
        left = QRect(area.left() + OVERLAY_MARGIN, top, width, height)
        right = QRect(area.right() - OVERLAY_MARGIN - width + 1, top, width, height)
        if position is not None:
            cursor = widget.mapFromScene(position) + widget.viewport().pos()
            near = [rect.adjusted(-READOUT_AVOID, -READOUT_AVOID, READOUT_AVOID, READOUT_AVOID).contains(cursor)
                    for rect in ((right, left) if self.right else (left, right))]
            self.right ^= near[0] and not near[1]  # away from the cursor, and only when the other corner is free
        label.move((right if self.right else left).topLeft())


def style_window(region, color=None) -> None:
    """The x-window band in ``color`` (its lines, and a translucent fill); ``None`` — its own orange."""
    import pyqtgraph as pg

    base = QColor(color) if color else QColor(WINDOW_COLOR)
    fill, hover = QColor(base), QColor(base)
    fill.setAlpha(WINDOW_FILL_ALPHA)
    hover.setAlpha(WINDOW_HOVER_ALPHA)
    region.setBrush(fill)
    region.setHoverBrush(hover)
    for line in region.lines:
        line.setPen(pg.mkPen(base, width=1.5))
    region.update()


def roomy_ticks(axis, *, padding: float = 14.0):
    """A ``tickSpacing`` for a pyqtgraph ``axis`` whose labelled ticks never run together.

    pyqtgraph keeps at least two intervals however short the axis is, so a narrow image shows
    “0500100015002000”. Here the major step grows (1-2-5 × 10ⁿ) until the widest label plus ``padding``
    fits between two ticks; an explicit ``setTickSpacing`` is left alone. Display only.
    """
    import math

    from PyQt5.QtGui import QFontMetricsF

    original = axis.tickSpacing

    def spacing(min_value, max_value, size):
        levels = original(min_value, max_value, size)
        span = abs(float(max_value) - float(min_value))
        if not levels or axis._tickSpacing is not None or not span > 0 or not size > 0:
            return levels
        metrics = QFontMetricsF(axis.style.get("tickFont") or axis.font())
        widest = max(metrics.horizontalAdvance(f"{value:g}") for value in (min_value, max_value, (min_value + max_value) / 2))
        room = widest + padding
        major = float(levels[0][0])
        if major * size / span >= room:
            return levels
        unit = 10.0 ** math.floor(math.log10(major))
        for factor in (2, 5, 10, 20, 50, 100, 200, 500, 1000):
            step = factor * unit
            if step * size / span >= room:
                break
        return [(step, 0), (step / 2, 0)]

    return spacing


class IconTools(QObject):
    """Tool buttons (``{icon kind: button}``) that show a line icon instead of their text while ``on``
    (a narrow view), drawn in the theme's plot colour; the text stays for the tooltip and screen readers."""

    def __init__(self, parent: QObject, buttons: dict):
        super().__init__(parent)
        self._buttons = dict(buttons)
        self._on = False
        theme_manager().changed.connect(self._refresh)

    def set_on(self, on: bool) -> None:
        if bool(on) != self._on:
            self._on = bool(on)
            self._refresh()

    def is_on(self) -> bool:
        return self._on

    @pyqtSlot()
    @pyqtSlot(str)
    def _refresh(self, *_args) -> None:
        color = theme_manager().color("plot_fg")
        for kind, button in self._buttons.items():
            try:
                button.setIcon(tool_icon(kind, color) if self._on else QIcon())
                button.setToolButtonStyle(Qt.ToolButtonIconOnly if self._on else Qt.ToolButtonTextOnly)
            except RuntimeError:  # a button deleted with its view
                continue


__all__ = ["CompactHeader", "CursorReadout", "EmptyOverlay", "IconTools", "MORE_TEXT", "PlotMenu", "WINDOW_COLOR",
           "axis_parts", "axis_symbol", "curves_text", "folded_label", "readout_text", "roomy_ticks", "style_window", "view_rect"]
