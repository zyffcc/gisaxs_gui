"""Shapes drawn over a ``DetectorView``: outlines, editable rectangles, and drawing new masks.

* ``set_outlines([(xs, ys, color), …])``: thin closed lines (for example the border of a cut
  region on the q map).
* ``set_rects([(key, x0, y0, x1, y1, color, editable), …])``: rectangles; an editable one can be
  moved and resized and emits ``rectChanged(key, x0, y0, x1, y1)`` when the drag ends.
* ``start_drawing("rectangle" | "polygon" | "point", purpose=…)``: a rectangle is two clicks
  (opposite corners), a polygon one click per vertex and a double click (or Enter) to close, a
  point one click; Esc cancels, Backspace removes the last vertex. The result is emitted in view
  coordinates as ``shapeDrawn(kind, [(x, y), …])`` for a mask (the default purpose) and as
  ``shapePicked(purpose, kind, [(x, y), …])`` for any other purpose (e.g. placing a cut region).
  While drawing, the view's own clicks (beam centre, double click) are held back.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal

RECTANGLE = "rectangle"
POLYGON = "polygon"
POINT = "point"
MASK_PURPOSE = "mask"
DRAW_COLOR = "#f43f5e"


class ShapeLayer(QObject):
    rectChanged = pyqtSignal(str, float, float, float, float)
    shapeDrawn = pyqtSignal(str, list)
    shapePicked = pyqtSignal(str, str, list)
    """``(purpose, kind, points)`` of a shape drawn for something other than a mask."""
    drawingChanged = pyqtSignal(str)
    """The kind being drawn, or "" when drawing ended."""

    def __init__(self, view):
        import pyqtgraph as pg

        super().__init__(view)
        self.view = view
        self._outlines: list = []
        self._rects: dict[str, object] = {}
        self._updating = False
        self.kind = ""
        self.purpose = ""
        self._points: list[tuple[float, float]] = []
        self._preview = pg.PlotCurveItem(pen=pg.mkPen(DRAW_COLOR, width=2, style=Qt.DashLine))
        self._preview.setZValue(30)
        view.plot.addItem(self._preview, ignoreBounds=True)
        self._preview.hide()
        view.graphics.scene().sigMouseClicked.connect(self._clicked)
        view.graphics.scene().sigMouseMoved.connect(self._moved)
        view.graphics.installEventFilter(self)

    # -- outlines and rectangles ---------------------------------------------------------

    def set_outlines(self, outlines) -> None:
        import pyqtgraph as pg

        marks = getattr(self.view, "marks", None)
        for item in self._outlines:
            self.view.plot.removeItem(item)
            if marks is not None:
                marks.forget("shapes", item)
        self._outlines = []
        for xs, ys, color in outlines:
            item = pg.PlotCurveItem(list(xs), list(ys), pen=pg.mkPen(color, width=2))
            item.setZValue(12)
            self.view.plot.addItem(item, ignoreBounds=True)
            if marks is not None:
                marks.track("shapes", item)
            self._outlines.append(item)

    def set_rects(self, rects) -> None:
        import pyqtgraph as pg

        self._updating = True
        marks = getattr(self.view, "marks", None)
        try:
            for roi in self._rects.values():
                self.view.plot.removeItem(roi)
                if marks is not None:
                    marks.forget("shapes", roi)
            self._rects = {}
            for key, x0, y0, x1, y1, color, editable in rects:
                left, right = sorted((float(x0), float(x1)))
                bottom, top = sorted((float(y0), float(y1)))
                roi = pg.RectROI(
                    (left, bottom), (max(right - left, 1e-9), max(top - bottom, 1e-9)),
                    pen=pg.mkPen(color, width=2), movable=bool(editable), resizable=bool(editable),
                    rotatable=False, removable=False,
                )
                if not editable:
                    for handle in list(roi.getHandles()):
                        roi.removeHandle(handle)
                else:
                    roi.addScaleHandle((0.0, 0.0), (1.0, 1.0))
                roi.setZValue(14)
                roi.sigRegionChangeFinished.connect(lambda item, key=key: self._rect_finished(key, item))
                self.view.plot.addItem(roi, ignoreBounds=True)
                if marks is not None:
                    marks.track("shapes", roi)
                self._rects[key] = roi
        finally:
            self._updating = False

    def rect_keys(self) -> list[str]:
        return list(self._rects)

    def _rect_finished(self, key: str, roi) -> None:
        if self._updating:
            return
        x, y = roi.pos()
        width, height = roi.size()
        self.rectChanged.emit(key, float(x), float(y), float(x + width), float(y + height))

    def clear(self) -> None:
        self.set_outlines([])
        self.set_rects([])

    # -- drawing -------------------------------------------------------------------------

    def start_drawing(self, kind: str, *, purpose: str = MASK_PURPOSE) -> None:
        if kind not in (RECTANGLE, POLYGON, POINT):
            raise ValueError(f"Unknown shape {kind!r}.")
        self.view.set_pick_mode(False)
        if self.kind:
            self._finish(None)
        self.kind = kind
        self.purpose = purpose
        self._points = []
        self._preview.setData([], [])
        self._preview.show()
        self.view._drawing = True
        self.view.graphics.setCursor(Qt.CrossCursor)
        self.view.graphics.setFocus()
        self.drawingChanged.emit(kind)

    def cancel_drawing(self) -> None:
        self._finish(None)

    def drawing(self) -> bool:
        return bool(self.kind)

    def _finish(self, points: Optional[list]) -> None:
        kind, purpose = self.kind, self.purpose
        self.kind = ""
        self.purpose = ""
        self._points = []
        self._preview.hide()
        self.view._drawing = False
        self.view.graphics.setCursor(Qt.ArrowCursor)
        self.drawingChanged.emit("")
        if points and purpose == MASK_PURPOSE:
            self.shapeDrawn.emit(kind, points)
        elif points:
            self.shapePicked.emit(purpose, kind, points)

    def _position(self, scene_position) -> tuple[float, float]:
        point = self.view.view_box.mapSceneToView(scene_position)
        return float(point.x()), float(point.y())

    def _clicked(self, event) -> None:
        if not self.kind or event.button() != Qt.LeftButton:
            return
        event.accept()
        x, y = self._position(event.scenePos())
        if self.kind == POINT:
            self._finish([(x, y)])
            return
        if self.kind == RECTANGLE:
            self._points.append((x, y))
            if len(self._points) == 2:
                self._finish(list(self._points))
            return
        if event.double():
            if len(self._points) >= 3:
                self._finish(list(self._points))
            return
        self._points.append((x, y))
        self._draw_preview(None)

    def _moved(self, scene_position) -> None:
        if not self.kind or not self._points:
            return
        self._draw_preview(self._position(scene_position))

    def _draw_preview(self, cursor: Optional[tuple[float, float]]) -> None:
        points = list(self._points)
        if self.kind == RECTANGLE and cursor is not None and points:
            (x0, y0), (x1, y1) = points[0], cursor
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        elif cursor is not None:
            points = points + [cursor, points[0]]
        self._preview.setData([x for x, _y in points], [y for _x, y in points])

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if self.kind and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Escape:
                self.cancel_drawing()
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter) and self.kind == POLYGON and len(self._points) >= 3:
                self._finish(list(self._points))
                return True
            if key == Qt.Key_Backspace and self._points:
                self._points.pop()
                self._draw_preview(None)
                return True
        return False


__all__ = ["DRAW_COLOR", "MASK_PURPOSE", "POINT", "POLYGON", "RECTANGLE", "ShapeLayer"]
