"""Zoom into a rectangle on any plot or image.

pyqtgraph views (``DetectorView``, ``CurvePlot`` and every other ``ViewBox``):

* **Shift + drag** with the left button draws a rectangle and zooms into it, in every view of the
  application (``install_box_zoom``, called once; the normal left drag still pans).
* ``zoom_button(view_box)`` — a checkable **Zoom** button: while it is on, a plain left drag zooms
  into a rectangle; the view's own reset (Fit view, Reset) shows everything again.

Matplotlib canvases: ``MplBoxZoom(canvas)`` — drag a rectangle to zoom, double-click to see
everything again. The zoom survives redraws of the data (the limits are put back after each draw,
with a small “zoomed — double-click to reset” note), until the person resets it.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QToolButton, QWidget

_INSTALLED = False
ZOOM_TIP = "Drag a rectangle to zoom into it (Shift + drag works in every view); Reset shows everything again"


def install_box_zoom() -> None:
    """Shift + left drag zooms into a rectangle in every pyqtgraph view (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import pyqtgraph as pg

    original = pg.ViewBox.mouseDragEvent

    def mouse_drag(self, event, axis=None):
        if event.isStart():
            self._gimap_box_drag = (
                event.button() == Qt.LeftButton
                and bool(event.modifiers() & Qt.ShiftModifier)
                and self.state["mouseMode"] == pg.ViewBox.PanMode
            )
        if getattr(self, "_gimap_box_drag", False) and event.button() == Qt.LeftButton:
            self.state["mouseMode"] = pg.ViewBox.RectMode
            try:
                original(self, event, axis)
            finally:
                self.state["mouseMode"] = pg.ViewBox.PanMode
                if event.isFinish():
                    self._gimap_box_drag = False
            return
        original(self, event, axis)

    pg.ViewBox.mouseDragEvent = mouse_drag
    _INSTALLED = True


def tool_icon(kind: str, color) -> QIcon:
    """Small line icons drawn in ``color``: ``zoom`` (a magnifier), ``fit`` (four corners), ``undo`` / ``redo``
    (a curved arrow to the left / right)."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(3.0)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    if kind == "marks":  # an eye
        path = QPainterPath(QPointF(3, 16))
        path.quadTo(QPointF(16, 3), QPointF(29, 16))
        path.quadTo(QPointF(16, 29), QPointF(3, 16))
        painter.drawPath(path)
        painter.drawEllipse(QRectF(12, 12, 8, 8))
    elif kind in ("undo", "redo"):
        if kind == "redo":  # the mirror image of undo
            painter.translate(32, 0)
            painter.scale(-1, 1)
        path = QPainterPath(QPointF(8, 14))
        path.cubicTo(QPointF(12, 6), QPointF(26, 6), QPointF(26, 17))
        path.cubicTo(QPointF(26, 24), QPointF(20, 27), QPointF(14, 27))
        painter.drawPath(path)
        painter.drawLine(QPointF(8, 14), QPointF(7, 6))
        painter.drawLine(QPointF(8, 14), QPointF(16, 14))
    elif kind == "zoom":
        painter.drawEllipse(QRectF(5, 5, 16, 16))
        painter.drawLine(QPointF(19, 19), QPointF(27, 27))
        painter.drawLine(QPointF(13, 9.5), QPointF(13, 16.5))
        painter.drawLine(QPointF(9.5, 13), QPointF(16.5, 13))
    else:
        for x, y, dx, dy in ((5, 5, 1, 1), (27, 5, -1, 1), (5, 27, 1, -1), (27, 27, -1, -1)):
            painter.drawLine(QPointF(x, y), QPointF(x + 7 * dx, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + 7 * dy))
    painter.end()
    return QIcon(pixmap)


def zoom_button(
    view_box, parent: Optional[QWidget] = None, *, text: str = "Zoom", icon_color=None,
) -> QToolButton:
    """A checkable button: on, a left drag on ``view_box`` zooms into a rectangle.

    With ``icon_color`` it shows a magnifier instead of the text (for crowded headers).
    """
    import pyqtgraph as pg

    install_box_zoom()
    button = QToolButton(parent)
    button.setText(text)
    if icon_color is not None:
        button.setIcon(tool_icon("zoom", icon_color))
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setAutoRaise(True)
    button.setCheckable(True)
    button.setObjectName("boxZoomButton")
    button.setToolTip(ZOOM_TIP)
    button.toggled.connect(
        lambda on: view_box.setMouseMode(pg.ViewBox.RectMode if on else pg.ViewBox.PanMode)
    )
    return button


def _close(first, second) -> bool:
    import numpy as np

    return bool(np.allclose(np.asarray(first, dtype=float), np.asarray(second, dtype=float), rtol=1e-9, atol=0.0))


class MplBoxZoom:
    """Rectangle zoom on a Matplotlib canvas (see the module docstring)."""

    NOTE = "zoomed — double-click to reset"
    MIN_DRAG_PX = 6

    def __init__(self, canvas):
        self.canvas = canvas
        self._limits: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._press = None
        self._rect = None
        self._restoring = False
        canvas.mpl_connect("button_press_event", self._pressed)
        canvas.mpl_connect("motion_notify_event", self._moved)
        canvas.mpl_connect("button_release_event", self._released)
        canvas.mpl_connect("draw_event", self._drawn)
        if hasattr(canvas, "setToolTip"):  # a Qt canvas (not an off-screen Agg one)
            canvas.setToolTip("Drag a rectangle to zoom; double-click to see everything again")

    @property
    def zoomed(self) -> bool:
        return bool(self._limits)

    def _axes_index(self, axes) -> Optional[int]:
        figure_axes = list(self.canvas.figure.axes)
        return figure_axes.index(axes) if axes in figure_axes else None

    def _pressed(self, event) -> None:
        if event.inaxes is None or event.button != 1:
            return
        if event.dblclick:
            self.reset()
            return
        self._press = (event.inaxes, event.x, event.y, event.xdata, event.ydata)

    def _moved(self, event) -> None:
        if self._press is None or event.inaxes is not self._press[0] or event.xdata is None:
            return
        from matplotlib.patches import Rectangle

        axes, _px, _py, x0, y0 = self._press
        if self._rect is None:
            self._rect = Rectangle((x0, y0), 0, 0, fill=False, linestyle="--", linewidth=1.0, edgecolor="#f97316")
            axes.add_patch(self._rect)
        self._rect.set_bounds(min(x0, event.xdata), min(y0, event.ydata), abs(event.xdata - x0), abs(event.ydata - y0))
        self.canvas.draw_idle()

    def _released(self, event) -> None:
        press, self._press = self._press, None
        if self._rect is not None:
            self._rect.remove()
            self._rect = None
        if press is None or event.button != 1:
            return
        axes, px, py, x0, y0 = press
        if event.xdata is None or event.inaxes is not axes:
            self.canvas.draw_idle()
            return
        if abs(event.x - px) < self.MIN_DRAG_PX or abs(event.y - py) < self.MIN_DRAG_PX:
            self.canvas.draw_idle()
            return
        xlim = tuple(sorted((x0, event.xdata)))
        ylim = tuple(sorted((y0, event.ydata)))
        if axes.xaxis_inverted():
            xlim = xlim[::-1]
        if axes.yaxis_inverted():
            ylim = ylim[::-1]
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        index = self._axes_index(axes)
        if index is not None:
            self._limits[index] = (xlim, ylim)
        self._note(axes)
        self.canvas.draw_idle()

    def _note(self, axes) -> None:
        if any(getattr(text, "_gimap_zoom_note", False) for text in axes.texts):
            return
        note = axes.text(0.99, 0.99, self.NOTE, transform=axes.transAxes, ha="right", va="top", fontsize=7,
                         color="#6b7280")
        note._gimap_zoom_note = True

    def reset(self) -> None:
        """Everything again: the limits the plot chose itself."""
        self._limits.clear()
        for axes in self.canvas.figure.axes:
            for text in list(axes.texts):
                if getattr(text, "_gimap_zoom_note", False):
                    text.remove()
            axes.relim()
            axes.autoscale()
        self.canvas.draw_idle()

    def _drawn(self, _event) -> None:
        """After the owner redrew its data: put the zoom back (one more draw)."""
        if self._restoring:
            self._restoring = False
            return
        if not self._limits:
            return
        changed = False
        axes_list = list(self.canvas.figure.axes)
        for index, (xlim, ylim) in self._limits.items():
            if index >= len(axes_list):
                continue
            axes = axes_list[index]
            if not (_close(axes.get_xlim(), xlim) and _close(axes.get_ylim(), ylim)):
                axes.set_xlim(xlim)
                axes.set_ylim(ylim)
                self._note(axes)
                changed = True
        if changed:
            self._restoring = True
            self.canvas.draw_idle()


__all__ = ["MplBoxZoom", "ZOOM_TIP", "install_box_zoom", "tool_icon", "zoom_button"]
