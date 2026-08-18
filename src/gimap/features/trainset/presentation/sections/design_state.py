"""Design State section for the Trainset page."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


from PyQt5.QtCore import QTimer, Qt


from PyQt5.QtWidgets import (
    QComboBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
)


from ..visualization_widgets import ArrayCanvas


class DesignStateMixin:
    """Own the design state section."""

    def _step_selected(self, index: int) -> None:
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        self.back_button.setEnabled(index > 0)
        self.step_changed.emit(index)
        QTimer.singleShot(0, self._apply_responsive_layout)

    def add_mask_shape(self, shape: Dict[str, Any]) -> None:
        row = self.mask_shape_table.rowCount()
        self.mask_shape_table.insertRow(row)
        shape_type = str(shape.get("type", "rectangle"))
        if shape_type in {"ellipse", "roi_ellipse_exterior"}:
            values = (
                shape_type,
                shape.get("cx", 0),
                shape.get("cy", 0),
                shape.get("radius_x", shape.get("radius", 0)),
                shape.get("radius_y", shape.get("radius", 0)),
            )
        elif shape_type == "circle":
            values = ("circle", shape.get("cx", 0), shape.get("cy", 0), shape.get("radius", 0), "")
        else:
            values = (
                "rectangle",
                shape.get("x", 0),
                shape.get("y", 0),
                shape.get("width", 0),
                shape.get("height", 0),
            )
        for column, value in enumerate(values):
            self.mask_shape_table.setItem(row, column, QTableWidgetItem(str(value)))

    def set_plugin_parameters(
        self, table: QTableWidget, definitions, values: Dict[str, Any]
    ) -> None:
        table.setRowCount(0)
        for definition in definitions:
            row = table.rowCount()
            table.insertRow(row)
            key = str(definition["key"])
            spec = values.get(key, {}) if isinstance(values, dict) else {}
            distribution = QComboBox()
            distribution.addItems(("uniform", "log_uniform"))
            distribution.setCurrentText(str(spec.get("distribution", "uniform")))
            distribution.currentTextChanged.connect(self.configuration_edited)
            table.setItem(row, 0, QTableWidgetItem(key))
            table.item(row, 0).setFlags(table.item(row, 0).flags() & ~Qt.ItemIsEditable)
            table.setCellWidget(row, 1, distribution)
            table.setItem(
                row, 2, QTableWidgetItem(str(spec.get("minimum", definition.get("minimum", 0.0))))
            )
            table.setItem(
                row, 3, QTableWidgetItem(str(spec.get("maximum", definition.get("maximum", 1.0))))
            )
            meaning_column = 5 if table.columnCount() >= 6 else 4
            if table.columnCount() >= 6:
                table.setItem(
                    row, 4, QTableWidgetItem(str(max(1, int(spec.get("grid_points", 30)))))
                )
                table.item(row, 4).setToolTip(
                    "BornAgain basis-image count on this shape-parameter axis. "
                    "For example, 30 radius values × 30 height values creates a 30 × 30 matrix."
                )
            meaning = f"{definition.get('label', key)}"
            if definition.get("unit"):
                meaning += f" [{definition['unit']}]"
            table.setItem(row, meaning_column, QTableWidgetItem(meaning))
            table.item(row, meaning_column).setFlags(
                table.item(row, meaning_column).flags() & ~Qt.ItemIsEditable
            )

    @staticmethod
    def plugin_parameters(table: QTableWidget) -> Dict[str, Dict[str, Any]]:
        parameters: Dict[str, Dict[str, Any]] = {}
        for row in range(table.rowCount()):
            name = table.item(row, 0).text().strip()
            distribution = table.cellWidget(row, 1)
            parameters[name] = {
                "distribution": distribution.currentText()
                if isinstance(distribution, QComboBox)
                else "uniform",
                "minimum": float(table.item(row, 2).text()),
                "maximum": float(table.item(row, 3).text()),
            }
            if table.columnCount() >= 6 and table.item(row, 4):
                parameters[name]["grid_points"] = max(1, int(float(table.item(row, 4).text())))
        return parameters

    def update_cache_grid_summary(self, config: Dict[str, Any]) -> None:
        if not hasattr(self, "cache_grid_summary"):
            return
        particle = next(iter(config.get("sample", {}).get("particles", [])), {})
        axes = [
            (name, max(1, int(spec.get("grid_points", 30))))
            for name, spec in particle.get("parameters", {}).items()
        ]
        nodes = int(np.prod([points for _name, points in axes], dtype=np.int64)) if axes else 0
        roi = config.get("roi", {})
        estimated_gib = nodes * int(roi.get("width", 0)) * int(roi.get("height", 0)) * 2 / (1024**3)
        shape_text = " × ".join(str(points) for _name, points in axes) or "no axes"
        names_text = ", ".join(name for name, _points in axes) or "none"
        max_files = int(config.get("simulation", {}).get("grid_cache", {}).get("max_files", 5))
        self.cache_grid_summary.setText(
            f"Matrix: {shape_text} = {nodes:,} BornAgain basis images ({names_text}). "
            f"Estimated float16 cache: {estimated_gib:.2f} GiB/file. "
            f"Least-recently-used retention: {max_files} file(s)."
        )

    def _add_layer_row(self) -> None:
        row = self.layer_table.rowCount()
        self.layer_table.insertRow(row)
        for column, value in enumerate(("1", "Silicon", "10.0", "10.0", "0.0", "0.0")):
            self.layer_table.setItem(row, column, QTableWidgetItem(value))

    def add_model_layer(self, spec: Dict[str, Any], row: Optional[int] = None) -> None:
        row = self.model_layer_table.rowCount() if row is None else row
        self.model_layer_table.insertRow(row)
        kind = QComboBox()
        kind.addItems(
            (
                "conv2d",
                "maxpool2d",
                "batch_normalization",
                "dropout",
                "global_average_pooling2d",
                "flatten",
                "dense",
            )
        )
        kind.setCurrentText(str(spec.get("type", "conv2d")))
        kind.currentTextChanged.connect(self.configuration_edited)
        activation = QComboBox()
        activation.addItems(("relu", "gelu", "tanh", "sigmoid", "linear"))
        activation.setCurrentText(str(spec.get("activation", "relu")))
        activation.currentTextChanged.connect(self.configuration_edited)
        self.model_layer_table.setCellWidget(row, 0, kind)
        self.model_layer_table.setItem(row, 1, QTableWidgetItem(str(spec.get("units", ""))))
        self.model_layer_table.setItem(
            row, 2, QTableWidgetItem(str(spec.get("kernel", spec.get("pool", ""))))
        )
        self.model_layer_table.setCellWidget(row, 3, activation)
        self.model_layer_table.setItem(row, 4, QTableWidgetItem(str(spec.get("rate", ""))))

    def set_model_layers(self, layers) -> None:
        self.model_layer_table.setRowCount(0)
        for layer in layers:
            self.add_model_layer(layer)

    def model_layers(self):
        layers = []
        for row in range(self.model_layer_table.rowCount()):
            kind_widget = self.model_layer_table.cellWidget(row, 0)
            activation_widget = self.model_layer_table.cellWidget(row, 3)
            kind = kind_widget.currentText() if isinstance(kind_widget, QComboBox) else "conv2d"
            units_text = (
                self.model_layer_table.item(row, 1).text().strip()
                if self.model_layer_table.item(row, 1)
                else ""
            )
            size_text = (
                self.model_layer_table.item(row, 2).text().strip()
                if self.model_layer_table.item(row, 2)
                else ""
            )
            rate_text = (
                self.model_layer_table.item(row, 4).text().strip()
                if self.model_layer_table.item(row, 4)
                else ""
            )
            spec: Dict[str, Any] = {"type": kind}
            if kind in {"conv2d", "dense"}:
                spec["units"] = int(float(units_text or 32))
                spec["activation"] = (
                    activation_widget.currentText()
                    if isinstance(activation_widget, QComboBox)
                    else "relu"
                )
            if kind == "conv2d":
                spec["kernel"] = int(float(size_text or 3))
            elif kind == "maxpool2d":
                spec["pool"] = int(float(size_text or 2))
            elif kind == "dropout":
                spec["rate"] = float(rate_text or 0.3)
            layers.append(spec)
        return layers

    def _move_model_layer(self, offset: int) -> None:
        row = self.model_layer_table.currentRow()
        target = row + offset
        if row < 0 or target < 0 or target >= self.model_layer_table.rowCount():
            return
        layers = self.model_layers()
        layers[row], layers[target] = layers[target], layers[row]
        self.set_model_layers(layers)
        self.model_layer_table.selectRow(target)

    @staticmethod
    def _remove_selected_rows(table: QTableWidget) -> None:
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def remove_selected_mask_shapes(self) -> bool:
        before = self.mask_shape_table.rowCount()
        self._remove_selected_rows(self.mask_shape_table)
        return self.mask_shape_table.rowCount() != before

    def remove_mask_shapes_by_type(self, *shape_types: str) -> None:
        wanted = set(shape_types)
        for row in range(self.mask_shape_table.rowCount() - 1, -1, -1):
            item = self.mask_shape_table.item(row, 0)
            if item is not None and item.text() in wanted:
                self.mask_shape_table.removeRow(row)

    def mask_shapes(self):
        shapes = []
        for row in range(self.mask_shape_table.rowCount()):
            values = [
                self.mask_shape_table.item(row, column).text()
                if self.mask_shape_table.item(row, column)
                else "0"
                for column in range(5)
            ]
            if values[0] == "circle":
                shapes.append(
                    {
                        "type": "circle",
                        "cx": int(float(values[1])),
                        "cy": int(float(values[2])),
                        "radius": int(float(values[3])),
                    }
                )
            elif values[0] in {"ellipse", "roi_ellipse_exterior"}:
                shapes.append(
                    {
                        "type": values[0],
                        "cx": float(values[1]),
                        "cy": float(values[2]),
                        "radius_x": max(1e-6, float(values[3])),
                        "radius_y": max(1e-6, float(values[4])),
                    }
                )
            else:
                shapes.append(
                    {
                        "type": "rectangle",
                        "x": int(float(values[1])),
                        "y": int(float(values[2])),
                        "width": int(float(values[3])),
                        "height": int(float(values[4])),
                    }
                )
        return shapes

    def set_simulation_preview(
        self,
        comparison_images: Dict[str, np.ndarray],
        comparison_labels: Dict[str, str],
        stages,
        stats: Dict[str, Any],
        spectrum_x: np.ndarray,
        spectrum_y: np.ndarray,
    ) -> None:
        for key, image in comparison_images.items():
            for canvas in self.impact_canvases.get(key, []):
                canvas.set_data(image)
            for label in self.impact_value_labels.get(key, []):
                label.setText(comparison_labels.get(key, key.title()))
        self.preview_tabs.clear()
        self.preview_canvases.clear()
        for stage in stages:
            name = str(stage["name"])
            key = name.lower()
            canvas = ArrayCanvas(f"{name} simulated stage")
            canvas.set_data(stage["image"], stage.get("mask"))
            self.preview_canvases[key] = canvas
            self.preview_tabs.addTab(canvas, name)
        if not stages:
            empty = QLabel("No enabled preprocessing stages were returned.")
            empty.setAlignment(Qt.AlignCenter)
            self.preview_tabs.addTab(empty, "No stages")
        self.preview_views.setTabText(1, f"Pipeline stages ({len(stages)})")
        self._apply_display_settings("preview")
        self.histogram.set_data(spectrum_x, spectrum_y)
        self.preview_stats.setText(
            "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in stats.items())
        )

    def set_preview_busy(self, busy: bool, progress: int = 0, message: str = "") -> None:
        for button in (
            self.generate_preview_button,
            self.force_simulation_button,
            self.new_realization_button,
        ):
            button.setEnabled(not busy)
        self.preview_job_status.setVisible(busy or bool(message))
        if busy:
            state = "running"
        elif message.lower().startswith("preview failed"):
            state = "failed"
        elif progress >= 100:
            state = "succeeded"
        else:
            state = "idle"
        self.preview_job_status.set_state(state, message, progress=progress / 100.0)
        self.preview_progress.setRange(0, 100)
        self.preview_progress.setValue(max(0, min(100, int(progress))))

    def set_preview_progress(self, progress: int, message: str) -> None:
        self.preview_progress.setValue(max(0, min(100, int(progress))))
        self.preview_activity.setText(message)

    def set_local_job_status(
        self,
        state: str,
        message: str,
        progress: Optional[int] = None,
    ) -> None:
        """Update the shared status view while preserving legacy percent semantics。"""

        normalized_progress = None if progress is None else progress / 100.0
        self.trainset_job_status.set_state(
            state,
            message,
            progress=normalized_progress,
        )
        if progress is not None:
            self.local_progress.setRange(0, 100)
            self.local_progress.setValue(max(0, min(100, int(progress))))

    def set_comparison_details(
        self,
        details: Dict[str, Any],
        parameter_specs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._comparison_details = details
        self._comparison_parameter_specs = parameter_specs or {}
        self._comparison_config = config or {}
        self.preview_parameters_button.setEnabled(bool(details))
