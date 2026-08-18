"""Trend Windows for multi-file prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING


import re


from typing import Dict, List, Optional, Any


import numpy as np


from PyQt5.QtWidgets import QLabel, QDialog


from .views import (
    DistributionHeatmapDialogView,
    ParameterTrendDialogView,
)


from .result_types import (
    PredictResult,
)

if TYPE_CHECKING:
    from .results_widget import MultiFilePredictResultsWidget


class DistributionHeatmapWindow(QDialog, DistributionHeatmapDialogView):
    """Detached heatmap viewer for multi-file predicted h/R distributions."""

    def __init__(self, results_widget: "MultiFilePredictResultsWidget", parent=None):
        super().__init__(parent)
        self.results_widget = results_widget
        self.setupUi(self)
        self._component_map: Dict[str, Dict[str, Any]] = {}
        self._last_component_key = ""

        self.figure = None
        self.canvas = None
        self.ax = None
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            self.figure = Figure(figsize=(9, 5), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = None
            self.plot_host_layout.addWidget(self.canvas, 1)
        except Exception as exc:
            self.plot_host_layout.addWidget(
                QLabel(f"Matplotlib is required for heatmap display: {exc}", self), 1
            )

        self.component_combo.currentIndexChanged.connect(self.refresh_plot)
        self.refresh_btn.clicked.connect(self.refresh_components_and_plot)
        self.refresh_components_and_plot()

    @staticmethod
    def _sort_key(result: PredictResult):
        text = f"{result.file_name} {result.file_path}"
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            try:
                return (float(numbers[0]), text.lower())
            except Exception:
                pass
        return (float("inf"), text.lower())

    @staticmethod
    def _as_component_curves(value: Any) -> List[np.ndarray]:
        if not isinstance(value, np.ndarray):
            return []
        arr = np.asarray(value, dtype=float)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return [arr.reshape(-1)]
        if arr.ndim != 2:
            return []
        if arr.shape[0] <= arr.shape[1]:
            return [arr[i, :].reshape(-1) for i in range(arr.shape[0])]
        return [arr[:, i].reshape(-1) for i in range(arr.shape[1])]

    @staticmethod
    def _distribution_kind(key: str) -> Optional[str]:
        low = str(key).lower()
        if low in {"r", "radius", "r_distribution", "radius_distribution"} or (
            "distribution" in low
            and ("radius" in low or re.search(r"(^|[_\-\s])r($|[_\-\s])", low))
        ):
            return "R"
        if low in {"h", "height", "h_distribution", "height_distribution"} or (
            "distribution" in low
            and ("height" in low or re.search(r"(^|[_\-\s])h($|[_\-\s])", low))
        ):
            return "h"
        if low.startswith("r") and "hr" not in low:
            return "R"
        if low.startswith("h") and "hr" not in low:
            return "h"
        return None

    def _completed_results_sorted(self) -> List[PredictResult]:
        return sorted(self.results_widget.get_completed_results(), key=self._sort_key)

    def _prediction_payload(self, result: PredictResult) -> Dict[str, Any]:
        payload = result.prediction_data if isinstance(result.prediction_data, dict) else {}
        nested = payload.get("prediction_data")
        return nested if isinstance(nested, dict) else payload

    def _discover_components(self) -> Dict[str, Dict[str, Any]]:
        components: Dict[str, Dict[str, Any]] = {}
        for result in self._completed_results_sorted():
            payload = self._prediction_payload(result)
            for key, value in payload.items():
                kind = self._distribution_kind(str(key))
                if kind is None:
                    continue
                curves = self._as_component_curves(value)
                for idx, _curve in enumerate(curves):
                    component_key = f"{kind}|{key}|{idx}"
                    label = (
                        f"{kind}: {key}"
                        if len(curves) == 1
                        else f"{kind}: {key} component {idx + 1}"
                    )
                    components.setdefault(
                        component_key,
                        {"kind": kind, "source_key": key, "component_index": idx, "label": label},
                    )
        return components

    def refresh_components_and_plot(self) -> None:
        previous = self.component_combo.currentData() or self._last_component_key
        self._component_map = self._discover_components()
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        for key, meta in sorted(self._component_map.items(), key=lambda item: item[1]["label"]):
            self.component_combo.addItem(meta["label"], key)
        if previous:
            idx = self.component_combo.findData(previous)
            if idx >= 0:
                self.component_combo.setCurrentIndex(idx)
        self.component_combo.blockSignals(False)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self.canvas is None or self.ax is None:
            if self.canvas is None or self.figure is None:
                return
        if self.figure is not None:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.figure.subplots_adjust(left=0.16, right=0.88, top=0.9, bottom=0.12)
        component_key = self.component_combo.currentData()
        if not component_key or component_key not in self._component_map:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "No completed R/h distribution results yet.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            self.status_label.setText("No completed R/h distribution results yet.")
            return
        self._last_component_key = component_key
        meta = self._component_map[component_key]
        rows = []
        labels = []
        for result in self._completed_results_sorted():
            payload = self._prediction_payload(result)
            curves = self._as_component_curves(payload.get(meta["source_key"]))
            idx = int(meta["component_index"])
            if idx >= len(curves):
                continue
            curve = np.asarray(curves[idx], dtype=float).reshape(-1)
            if curve.size == 0:
                continue
            rows.append(curve)
            labels.append(result.file_name)

        if not rows:
            self.status_label.setText("No data available for the selected component.")
            return

        width = max(row.size for row in rows)
        matrix = np.full((len(rows), width), np.nan, dtype=float)
        for row_idx, row in enumerate(rows):
            matrix[row_idx, : row.size] = row

        self.ax.clear()
        image = self.ax.imshow(matrix, aspect="auto", interpolation="nearest", origin="lower")
        self.ax.set_title(f"{meta['label']} heatmap")
        self.ax.set_xlabel("Distribution bin")
        self.ax.set_ylabel("Input file / stack (low to high)")
        if len(labels) <= 25:
            self.ax.set_yticks(np.arange(len(labels)))
            self.ax.set_yticklabels(labels, fontsize=7)
        else:
            tick_idx = np.linspace(0, len(labels) - 1, min(12, len(labels)), dtype=int)
            self.ax.set_yticks(tick_idx)
            self.ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=7)
        self._colorbar = self.figure.colorbar(image, ax=self.ax, fraction=0.046, pad=0.04)
        self._colorbar.set_label("Predicted intensity / probability")
        self.canvas.draw_idle()
        self.status_label.setText(f"Showing {len(rows)} completed result(s), {width} bins.")


class ParameterTrendWindow(QDialog, ParameterTrendDialogView):
    """Detached viewer for multi-file scalar/parameter predictions."""

    def __init__(self, results_widget: "MultiFilePredictResultsWidget", parent=None):
        super().__init__(parent)
        self.results_widget = results_widget
        self.setupUi(self)

        self.figure = None
        self.canvas = None
        self.ax = None
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            self.figure = Figure(figsize=(9, 5), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.plot_host_layout.addWidget(self.canvas, 1)
        except Exception as exc:
            self.plot_host_layout.addWidget(
                QLabel(f"Matplotlib is required for parameter trend display: {exc}", self), 1
            )

        self.parameter_combo.currentIndexChanged.connect(self.refresh_plot)
        self.refresh_btn.clicked.connect(self.refresh_parameters_and_plot)
        self.refresh_parameters_and_plot()

    @staticmethod
    def _sort_key(result: PredictResult):
        text = f"{result.file_name} {result.file_path}"
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            try:
                return (float(numbers[0]), text.lower())
            except Exception:
                pass
        return (float("inf"), text.lower())

    def _completed_results_sorted(self) -> List[PredictResult]:
        return sorted(self.results_widget.get_completed_results(), key=self._sort_key)

    def _prediction_payload(self, result: PredictResult) -> Dict[str, Any]:
        payload = result.prediction_data if isinstance(result.prediction_data, dict) else {}
        nested = payload.get("prediction_data")
        return nested if isinstance(nested, dict) else payload

    def _discover_parameter_names(self) -> List[str]:
        names: List[str] = []
        for result in self._completed_results_sorted():
            payload = self._prediction_payload(result)
            values = payload.get("parameters")
            if values is None:
                values = payload.get("scalars")
            if values is None:
                continue
            arr = np.asarray(values).reshape(-1)
            raw_names = payload.get("parameter_names")
            if isinstance(raw_names, list) and len(raw_names) >= arr.size:
                candidate = [str(name) for name in raw_names[: arr.size]]
            else:
                candidate = [f"p{i + 1}" for i in range(arr.size)]
            for name in candidate:
                if name not in names:
                    names.append(name)
        return names

    def refresh_parameters_and_plot(self) -> None:
        previous = self.parameter_combo.currentText()
        names = self._discover_parameter_names()
        self.parameter_combo.blockSignals(True)
        self.parameter_combo.clear()
        self.parameter_combo.addItem("All parameters")
        for name in names:
            self.parameter_combo.addItem(name)
        if previous:
            idx = self.parameter_combo.findText(previous)
            if idx >= 0:
                self.parameter_combo.setCurrentIndex(idx)
        self.parameter_combo.blockSignals(False)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        if self.canvas is None or self.figure is None:
            return
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        rows = []
        labels = []
        names: List[str] = []
        for result in self._completed_results_sorted():
            payload = self._prediction_payload(result)
            values = payload.get("parameters")
            if values is None:
                values = payload.get("scalars")
            if values is None:
                continue
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 0:
                continue
            raw_names = payload.get("parameter_names")
            if isinstance(raw_names, list) and len(raw_names) >= arr.size:
                row_names = [str(name) for name in raw_names[: arr.size]]
            else:
                row_names = [f"p{i + 1}" for i in range(arr.size)]
            if not names:
                names = row_names
            rows.append(arr)
            labels.append(result.file_name)

        if not rows:
            self.ax.text(
                0.5,
                0.5,
                "No completed parameter predictions yet.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.ax.set_axis_off()
            self.status_label.setText("No completed parameter predictions yet.")
            self.canvas.draw_idle()
            return

        selected = self.parameter_combo.currentText()
        x = np.arange(len(rows))
        matrix = np.full((len(rows), max(row.size for row in rows)), np.nan, dtype=float)
        for i, row in enumerate(rows):
            matrix[i, : row.size] = row

        if selected and selected != "All parameters" and selected in names:
            idx = names.index(selected)
            self.ax.scatter(x, matrix[:, idx], s=34)
            self.ax.plot(x, matrix[:, idx], linewidth=1.0, alpha=0.75)
            self.ax.set_ylabel(selected)
            self.ax.set_title(f"{selected} trend")
        else:
            for idx, name in enumerate(names[: matrix.shape[1]]):
                self.ax.scatter(x, matrix[:, idx], s=24, label=name)
                self.ax.plot(x, matrix[:, idx], linewidth=1.0, alpha=0.65)
            self.ax.set_ylabel("Predicted value")
            self.ax.set_title("SF parameter trends")
            self.ax.legend(loc="best")

        self.ax.set_xlabel("Input file / stack (low to high)")
        if len(labels) <= 25:
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        else:
            tick_idx = np.linspace(0, len(labels) - 1, min(12, len(labels)), dtype=int)
            self.ax.set_xticks(tick_idx)
            self.ax.set_xticklabels(
                [labels[i] for i in tick_idx], rotation=65, ha="right", fontsize=7
            )
        self.ax.grid(alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText(f"Showing {len(rows)} completed result(s).")
