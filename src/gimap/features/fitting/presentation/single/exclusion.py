"""Leaving points out of the fit: **Exclude** on the plot, then click a point (again to take it back)
or drag a box around several. Left-out points stay on the plot as grey crosses, are not fitted,
and are listed in the export record; the Curve step says how many and has **Include All**. Each
click, box or Include All is one step of Undo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsRectItem

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import point_key

PICK_RADIUS = 0.03
"""A click takes the nearest point within this fraction of the plot (both axes as shown)."""


class FitExclusionMixin:
    """Needs the page's plot, ``exclude_button``, ``include_all_button``, ``excluded_label``, ``session``."""

    def setup_exclusion(self) -> None:
        self._box_start: Optional[tuple] = None
        self._box_item: Optional[QGraphicsRectItem] = None
        view = self.plot.plot.getViewBox()
        self._default_drag = view.mouseDragEvent
        view.mouseDragEvent = self._drag  # instance override: a box while Exclude is on, else the usual
        self.exclude_button.toggled.connect(self._exclude_mode)
        self.plot.positionClicked.connect(self._clicked)
        self.include_all_button.clicked.connect(self.include_all)

    def _exclude_mode(self, on: bool) -> None:
        self.plot.plot_widget.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        if on:
            self._status(tr("Click a point to leave it out of the fit (click again to take it back), or drag a box "
                            "around several. Exclude again to stop."))

    # -- which point ----------------------------------------------------------------------

    def _shown(self, q, intensity):
        """The points in the coordinates of the plot as shown (log10 on log axes)."""
        x = np.log10(np.maximum(q, 1e-300)) if self.plot.log_x_check.isChecked() else np.asarray(q, float)
        y = np.log10(np.maximum(intensity, 1e-300)) if self.plot.log_check.isChecked() else np.asarray(intensity, float)
        return x, y

    def _clicked(self, x: float, y: float) -> None:
        if not self.exclude_button.isChecked():
            return
        points = self.session.all_points()
        if points is None or not points.q.size:
            return
        xs, ys = self._shown(points.q, points.intensity)
        (x0, x1), (y0, y1) = self.plot.plot.getViewBox().viewRange()
        px, py = self._shown(np.array([x]), np.array([y]))
        distance = np.hypot((xs - px[0]) / max(x1 - x0, 1e-12), (ys - py[0]) / max(y1 - y0, 1e-12))
        index = int(np.nanargmin(distance))
        if distance[index] > PICK_RADIUS:
            return
        key = point_key(points.q[index])
        self._set_excluded(self.session.excluded ^ {key})  # in, or back out

    # -- a box ----------------------------------------------------------------------------

    def _drag(self, event, axis=None) -> None:
        if not self.exclude_button.isChecked() or event.button() != Qt.LeftButton:
            self._default_drag(event, axis)
            return
        event.accept()
        view = self.plot.plot.getViewBox()
        position = view.mapSceneToView(event.scenePos())
        if event.isStart():
            self._box_start = (position.x(), position.y())
            self._box_item = QGraphicsRectItem()
            pen = QPen(QColor("#dc2626"))
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            self._box_item.setPen(pen)
            self._box_item.setBrush(QColor(220, 38, 38, 30))
            view.addItem(self._box_item, ignoreBounds=True)
        if self._box_start is None or self._box_item is None:
            return
        x0, y0 = self._box_start
        rect = QRectF(min(x0, position.x()), min(y0, position.y()), abs(position.x() - x0), abs(position.y() - y0))
        self._box_item.setRect(rect)
        if event.isFinish():
            view.removeItem(self._box_item)
            self._box_item, self._box_start = None, None
            self._exclude_box(rect)

    def _exclude_box(self, rect: QRectF) -> None:
        points = self.session.all_points()
        if points is None or not points.q.size:
            return
        xs, ys = self._shown(points.q, points.intensity)
        inside = (xs >= rect.left()) & (xs <= rect.right()) & (ys >= rect.top()) & (ys <= rect.bottom())
        if inside.any():
            self._set_excluded(self.session.excluded | {point_key(value) for value in points.q[inside]})

    # -- all --------------------------------------------------------------------------------

    def include_all(self) -> None:
        self._set_excluded(set())

    def _set_excluded(self, excluded) -> None:
        """The left-out points, as one step of Undo (a click, a box, Include All)."""
        if self.session.set_excluded(excluded):
            self._exclusions_changed()

    def left_out(self) -> int:
        """How many points of the halves chosen are left out."""
        points = self.session.all_points()
        if points is None or not self.session.excluded:
            return 0
        return int(sum(point_key(value) in self.session.excluded for value in points.q))

    def _exclusions_changed(self) -> None:
        self._remember_curve()
        self._render_all()

    def _show_left_out(self) -> None:
        count = self.left_out()
        self.excluded_label.setText(tr("Points left out of the fit: {count}").format(count=count) if count else "")
        self.excluded_row.setVisible(bool(count))


__all__ = ["FitExclusionMixin"]
