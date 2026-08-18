"""Formatting coordination for Classification."""

from __future__ import annotations

import io


from typing import Optional


from PyQt5.QtCore import Qt

from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import (
    QGraphicsScene,
)

from src.gimap.features.classification.application import (
    ClassificationSample,
    ModelEvaluationResult,
)


class FormattingMixin:
    """Own formatting presentation behavior."""

    def _set_graphics_text(self, view, text: str) -> None:
        scene = QGraphicsScene(view)
        scene.addText(text)
        view.setScene(scene)

    def _set_graphics_pixmap(self, view, pixmap: QPixmap) -> None:
        scene = QGraphicsScene(view)
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _figure_to_pixmap(self, fig) -> QPixmap:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image = QImage.fromData(buffer.read(), "PNG")
        return QPixmap.fromImage(image)

    def _fit_preview(self) -> None:
        view = self.page.previewGraphicsView
        if view.scene() is not None:
            view.fitInView(view.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def _file_dialog_filter(self) -> str:
        return "Data files (*.dat *.txt *.csv *.xy *.chi *.tif *.tiff *.png *.jpg *.jpeg *.bmp *.cbf *.edf *.h5 *.hdf5 *.npy);;All files (*.*)"

    def _next_color(self, index: int) -> str:
        colors = [
            "#2563eb",
            "#16a34a",
            "#dc2626",
            "#9333ea",
            "#ca8a04",
            "#0891b2",
            "#db2777",
            "#4b5563",
        ]
        return colors[index % len(colors)]

    def _unique_label(self, label: str, existing: Optional[str] = None) -> str:
        if label == existing:
            return label
        base = label
        counter = 2
        while label in self.sources:
            label = f"{base} {counter}"
            counter += 1
        return label

    def _short_paths(self, paths: list[str]) -> str:
        if not paths:
            return "-"
        if len(paths) == 1:
            return paths[0]
        return f"{paths[0]} (+{len(paths) - 1})"

    def _sample_by_id(self, sample_id) -> Optional[ClassificationSample]:
        if sample_id is None:
            return None
        for sample in self.samples:
            if sample.sample_id == str(sample_id):
                return sample
        return None

    def _shape_text(self, shape) -> str:
        if not shape:
            return "-"
        return "x".join(str(value) for value in shape)

    def _metric_text(self, result: Optional[ModelEvaluationResult], metric: str) -> str:
        if result is None or result.status != "ok":
            return "-"
        return f"{float(result.metrics_mean.get(metric, 0.0)):.3f}"

    def _number_text(self, value) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "-"

    def _optional_float(self, value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "-"

    def _parse_parameter(self, text: str, original):
        stripped = text.strip()
        if isinstance(original, bool):
            return stripped.lower() in {"1", "true", "yes", "on"}
        if isinstance(original, int) and not isinstance(original, bool):
            try:
                return int(stripped)
            except ValueError:
                return original
        if isinstance(original, float):
            try:
                return float(stripped)
            except ValueError:
                return original
        if stripped in {"None", "none", ""}:
            return None if original is None else stripped
        return stripped
