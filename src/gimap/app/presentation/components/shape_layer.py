"""Shapes drawn over a ``DetectorView``: outlines, editable rectangles, and drawing new masks.

* ``set_outlines([(xs, ys, color), …])``: thin closed lines (for example the border of a cut
  region on the q map).
* ``set_rects([(key, x0, y0, x1, y1, color, editable), …])``: rectangles; an editable one can be
  moved and resized and emits ``rectChanged(key, x0, y0, x1, y1)`` when the drag ends.
* Rectangles and ellipses take two bounding corners, circles a centre and rim, annuli a centre and two rims.
  The brush and eraser take a dragged stroke. A line takes its two ends, a point one click, a polygon one click
  per vertex and a double click or Enter to close. Esc cancels, Backspace removes the last vertex. Results use view
  coordinates as ``shapeDrawn(kind, [(x, y), …])`` for a mask (the default purpose) and as
  ``shapePicked(purpose, kind, [(x, y), …])`` for any other purpose (e.g. placing a cut region).
  While drawing, the view's own clicks (beam centre, double click) are held back.
"""

from __future__ import annotations

from math import cos, hypot, pi, sin
from typing import Optional

import numpy as np
from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal

RECTANGLE = "rectangle"
LINE = "line"
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
        self._radius = 1.0
        self._mask_image = pg.ImageItem(axisOrder="row-major")
        self._mask_image.setLookupTable(np.array([[0, 0, 0, 0], [244, 63, 94, 92]], dtype=np.uint8))
        self._mask_image.setZValue(6)
        view.plot.addItem(self._mask_image, ignoreBounds=True)
        self._preview = pg.PlotCurveItem(pen=pg.mkPen(DRAW_COLOR, width=2, style=Qt.DashLine))
        self._preview.setZValue(30)
        view.plot.addItem(self._preview, ignoreBounds=True)
        self._preview.hide()
        view.graphics.scene().sigMouseClicked.connect(self._clicked)
        view.graphics.scene().sigMouseMoved.connect(self._moved)
        view.graphics.installEventFilter(self)
        view.graphics.viewport().installEventFilter(self)

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
        self.set_mask_image(None)
        self.set_outlines([])
        self.set_rects([])

    def set_mask_image(self, mask) -> None:
        """The resulting user mask, including holes where an edit restored pixels."""
        self.view.marks.forget("shapes", self._mask_image)
        if mask is None:
            self._mask_image.clear()
            self._mask_image.hide()
        else:
            self._mask_image.setImage(np.asarray(mask).astype(np.uint8, copy=False), levels=(0, 1))
            self.view.marks.track("shapes", self._mask_image)

    # -- drawing -------------------------------------------------------------------------

    def start_drawing(self, kind: str, *, purpose: str = MASK_PURPOSE, radius: float = 1.0) -> None:
        if kind not in (RECTANGLE, LINE, POLYGON, POINT, "circle", "annulus", "ellipse", "brush", "eraser"):
            raise ValueError(f"Unknown shape {kind!r}.")
        self.view.set_pick_mode(False)
        if self.kind:
            self._finish(None)
        self.kind = kind
        self._radius = radius
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
        if not self.view.view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        event.accept()
        x, y = self._position(event.scenePos())
        if self.kind == POINT:
            self._finish([(x, y)])
            return
        if self.kind in (RECTANGLE, LINE, "circle", "ellipse", "annulus"):
            self._points.append((x, y))
            if len(self._points) == (3 if self.kind == "annulus" else 2):
                self._finish(list(self._points))
            return
        if event.double():
            if len(self._points) >= 3:
                self._finish(list(self._points))
            return
        self._points.append((x, y))
        self._draw_preview(None)

    def _moved(self, scene_position) -> None:
        if not self.kind or (not self._points and self.kind not in ("brush", "eraser")):
            return
        self._draw_preview(self._position(scene_position))

    def _draw_preview(self, cursor: Optional[tuple[float, float]]) -> None:
        points = list(self._points)
        if self.kind in ("brush", "eraser"):
            if cursor is not None:
                points = points + [(np.nan, np.nan)] + _oval(cursor, self._radius, self._radius)
        elif self.kind in ("circle", "annulus", "ellipse") and cursor is not None and points:
            x, y = points[0]
            dx, dy = cursor[0] - x, cursor[1] - y
            if self.kind == "ellipse":
                points = _oval((x + dx / 2, y + dy / 2), abs(dx / 2), abs(dy / 2))
            else:
                radii = [hypot(px - x, py - y) for px, py in points[1:] + [cursor]]
                points = _oval((x, y), radii[0], radii[0])
                if len(radii) == 2:
                    points += [(np.nan, np.nan)] + _oval((x, y), radii[1], radii[1])
        elif self.kind == RECTANGLE and cursor is not None and points:
            (x0, y0), (x1, y1) = points[0], cursor
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        elif self.kind == LINE and cursor is not None and points:
            points = [points[0], cursor]
        elif cursor is not None:
            points = points + [cursor, points[0]]
        self._preview.setData([x for x, _y in points], [y for _x, y in points])

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if self.kind and event.type() == QEvent.ShortcutOverride and event.key() == Qt.Key_Escape:
            event.accept()
            return True
        if self.kind in ("brush", "eraser") and watched is self.view.graphics.viewport():
            kind = event.type()
            if kind in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
                point = self.view.graphics.mapToScene(event.pos())
                if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    if not self.view.view_box.sceneBoundingRect().contains(point):
                        return False
                    self._points = [self._position(point)]
                elif kind == QEvent.MouseMove and self._points:
                    position = self._position(point)
                    if position != self._points[-1]:
                        self._points.append(position)
                elif kind == QEvent.MouseButtonRelease and self._points and event.button() == Qt.LeftButton:
                    self._finish(self._points + [self._position(point)])
                    return True
                else:
                    return False
                self._draw_preview(self._points[-1])
                return True
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


def _oval(centre, rx, ry):
    return [(centre[0] + rx * cos(i * pi / 48), centre[1] + ry * sin(i * pi / 48)) for i in range(97)]


__all__ = ["DRAW_COLOR", "LINE", "MASK_PURPOSE", "POINT", "POLYGON", "RECTANGLE", "ShapeLayer"]
