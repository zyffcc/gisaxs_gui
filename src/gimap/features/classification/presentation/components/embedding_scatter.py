"""Interactive, selection-linked embedding scatter view."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
)


class _EmbeddingPoint(QGraphicsEllipseItem):
    def __init__(self, sample_id: str, x: float, y: float, color: str, tooltip: str):
        radius = 6.0
        super().__init__(x - radius, y - radius, radius * 2, radius * 2)
        self.sample_id = sample_id
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#ffffff"), 0.8))
        self.setToolTip(tooltip)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.setPen(
                QPen(QColor("#111827"), 2.2)
                if bool(value)
                else QPen(QColor("#ffffff"), 0.8)
            )
            self.setZValue(2 if bool(value) else 0)
        return super().itemChange(change, value)


class EmbeddingScatterView(QGraphicsView):
    """Render embedding points with click and rubber-band selection."""

    selectedSampleIdsChanged = pyqtSignal(list)
    sampleActivated = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("embeddingScatterView")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.selectionChanged.connect(self._emit_selection)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setMinimumSize(520, 420)
        self.setBackgroundBrush(QBrush(QColor("#f8fafc")))
        self._points: dict[str, _EmbeddingPoint] = {}
        self._has_data = False
        self.show_empty("Run a reduction to explore this data group.")

    def show_empty(self, message: str) -> None:
        self._scene.clear()
        self._points.clear()
        self._has_data = False
        text = self._scene.addText(message)
        text.setDefaultTextColor(QColor("#64748b"))
        text.setPos(24, 24)
        self._scene.setSceneRect(0, 0, 760, 460)

    def set_points(
        self,
        values,
        sample_ids: list[str],
        colors: list[str],
        tooltips: list[str],
    ) -> None:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
            self.show_empty("The reduction did not produce two display dimensions.")
            return
        self._scene.clear()
        self._points.clear()
        finite = np.nan_to_num(array[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        low = np.min(finite, axis=0)
        span = np.ptp(finite, axis=0)
        span[span == 0] = 1.0
        display = 40.0 + (finite - low) / span * np.array([920.0, 620.0])
        for index, sample_id in enumerate(sample_ids):
            point = _EmbeddingPoint(
                sample_id,
                float(display[index, 0]),
                float(660.0 - display[index, 1]),
                colors[index],
                tooltips[index],
            )
            self._scene.addItem(point)
            self._points[sample_id] = point
        self._scene.setSceneRect(0, 0, 1000, 700)
        self._has_data = True
        self.fit_data()

    def selected_sample_ids(self) -> list[str]:
        return [
            item.sample_id
            for item in self._scene.selectedItems()
            if isinstance(item, _EmbeddingPoint)
        ]

    def select_sample_ids(self, sample_ids) -> None:
        selected = set(sample_ids)
        self._scene.blockSignals(True)
        try:
            for sample_id, item in self._points.items():
                item.setSelected(sample_id in selected)
        finally:
            self._scene.blockSignals(False)
        self._emit_selection()

    def select_all_points(self) -> None:
        self.select_sample_ids(self._points)

    def clear_selection(self) -> None:
        self._scene.clearSelection()

    def fit_data(self) -> None:
        if self._has_data:
            self.fitInView(self._scene.sceneRect().adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        if isinstance(item, _EmbeddingPoint):
            self.sampleActivated.emit(item.sample_id)
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.clear_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _emit_selection(self) -> None:
        self.selectedSampleIdsChanged.emit(self.selected_sample_ids())
