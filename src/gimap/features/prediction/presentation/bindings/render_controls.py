"""Render Controls coordination for Prediction."""

from __future__ import annotations

import os

import datetime


from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QRectF

from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import (
    QMessageBox,
    QGraphicsScene,
    QLabel,
    QGridLayout,
    QCheckBox,
    QDoubleSpinBox,
)


from src.gimap.features.prediction.application import (
    PredictionArrayExportRequest,
)


class RenderControlsMixin:
    """Own render controls presentation behavior."""

    def _render_step_snapshot(self, idx: int) -> None:
        steps = getattr(self, "_step_snapshots", None)
        if not isinstance(steps, list) or idx < 0 or idx >= len(steps):
            return
        snap = steps[idx].get("image") if isinstance(steps[idx], dict) else None
        if not isinstance(snap, np.ndarray):
            return
        self._current_step_index = idx
        self._predict_current_image = snap
        # update buttons state
        for i, b in enumerate(getattr(self, "_step_buttons", []) or []):
            try:
                b.setChecked(i == idx)
            except Exception:
                pass
        display, vmin, vmax = self._prepare_predict_image(snap)
        cmap = self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0])
        pix = self._create_pixmap_from_array(display, vmin, vmax, cmap)
        self._show_pixmap_in_predict_view(pix)

    def _render_parameters_figure(
        self, values: np.ndarray, names: Optional[List[str]] = None
    ) -> Optional[QPixmap]:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            vals = np.asarray(values, dtype=np.float32).reshape(-1)
            labels = (
                names
                if names and len(names) >= vals.size
                else [f"p{i + 1}" for i in range(vals.size)]
            )
            fig = Figure(figsize=(7.2, 3.8), dpi=120)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            x = np.arange(vals.size)
            bars = ax.bar(x, vals, color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"][: vals.size])
            ax.set_xticks(x)
            ax.set_xticklabels(labels[: vals.size])
            ax.set_ylabel("Predicted value")
            ax.set_title("SF Predicted Parameters")
            ax.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{float(value):.5g}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            fig.tight_layout()
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            height, width = buf.shape[:2]
            qimg = QImage(buf.data, width, height, buf.strides[0], QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg.copy())
        except Exception as exc:
            self._append_status_message(f"Parameter plot failed: {exc}", level="ERROR")
            return None

    def _refresh_predict_controls(self, kind: str) -> None:
        param_widget = getattr(self.ui, "predict2dParameterWidget", None)
        if param_widget is None:
            return
        two_d_widgets = [
            "predict2dColorScaleLabel",
            "predict2dAutoScaleCheckBox",
            "predict2dAutoScaleResetButton",
            "predict2dVminLabel",
            "predict2dVmaxLabel",
            "predict2dVminValue",
            "predict2dVmaxValue",
            "predict2dColormapLabel",
            "predict2dLabelCombox",
            "predict2dLogScaleCheckBox",
            "predict2dCountourCheckBox",
            "predict2dCountourLevelsLabel",
            "predict2dCountourLevelsValue",
        ]
        is_curve = kind == "curve"
        for name in two_d_widgets:
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setVisible(not is_curve)

        controls = self._ensure_predict_curve_controls()
        if not controls:
            return

        # 显示/隐藏整个1D参数部分
        curve_widget = getattr(self.ui, "predict2dParameter1dpartWidget", None)
        if curve_widget is not None:
            curve_widget.setVisible(is_curve)

        if not is_curve:
            return

        self._ui_updating = True
        try:
            controls["logx"].setChecked(
                bool(self.current_parameters.get("predict_curve_logx", False))
            )
            controls["logy"].setChecked(
                bool(self.current_parameters.get("predict_curve_logy", False))
            )
            autoscale = bool(self.current_parameters.get("predict_curve_autoscale", True))
            controls["autoscale"].setChecked(autoscale)
            for key in ("xmin", "xmax", "ymin", "ymax"):
                val = self.current_parameters.get(f"predict_curve_{key}")
                box = controls.get(key)
                if isinstance(box, QDoubleSpinBox):
                    if val is None:
                        box.setValue(0.0)
                    else:
                        box.setValue(float(val))
                    box.setEnabled(not autoscale)
        finally:
            self._ui_updating = False

    def _ensure_predict_curve_controls(self) -> Dict[str, object]:
        if self._predict_curve_controls:
            return self._predict_curve_controls
        parent = getattr(self.ui, "predict2dParameter1dpartWidget", None)
        if parent is None:
            return {}

        # 检查是否已经有布局，如果没有则创建一个
        grid = parent.layout()
        if grid is None:
            grid = QGridLayout(parent)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)

        logx = QCheckBox("Log X")
        logy = QCheckBox("Log Y")
        autoscale = QCheckBox("AutoScale")

        xmin = QDoubleSpinBox()
        xmax = QDoubleSpinBox()
        ymin = QDoubleSpinBox()
        ymax = QDoubleSpinBox()
        for box in (xmin, xmax, ymin, ymax):
            box.setRange(-1e9, 1e9)
            box.setDecimals(6)
            box.setSingleStep(0.1)

        grid.addWidget(logx, 0, 0)
        grid.addWidget(logy, 0, 1)
        grid.addWidget(autoscale, 0, 2)
        grid.addWidget(QLabel("X min"), 1, 0)
        grid.addWidget(xmin, 1, 1)
        grid.addWidget(QLabel("X max"), 1, 2)
        grid.addWidget(xmax, 1, 3)
        grid.addWidget(QLabel("Y min"), 2, 0)
        grid.addWidget(ymin, 2, 1)
        grid.addWidget(QLabel("Y max"), 2, 2)
        grid.addWidget(ymax, 2, 3)

        logx.toggled.connect(self._on_predict_curve_control_changed)
        logy.toggled.connect(self._on_predict_curve_control_changed)
        autoscale.toggled.connect(self._on_predict_curve_control_changed)
        for box in (xmin, xmax, ymin, ymax):
            box.editingFinished.connect(self._on_predict_curve_control_changed)

        self._predict_curve_controls = {
            "logx": logx,
            "logy": logy,
            "autoscale": autoscale,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }
        # Hide initially until a curve is shown
        parent.setVisible(False)
        return self._predict_curve_controls

    def _on_predict_curve_control_changed(self) -> None:
        if self._ui_updating:
            return
        controls = self._ensure_predict_curve_controls()
        if not controls:
            return
        self.current_parameters["predict_curve_logx"] = (
            bool(controls.get("logx").isChecked()) if controls.get("logx") else False
        )
        self.current_parameters["predict_curve_logy"] = (
            bool(controls.get("logy").isChecked()) if controls.get("logy") else False
        )
        autoscale = (
            bool(controls.get("autoscale").isChecked()) if controls.get("autoscale") else True
        )
        self.current_parameters["predict_curve_autoscale"] = autoscale
        for key in ("xmin", "xmax", "ymin", "ymax"):
            box = controls.get(key)
            if isinstance(box, QDoubleSpinBox):
                box.setEnabled(not autoscale)
                if not autoscale:
                    self.current_parameters[f"predict_curve_{key}"] = float(box.value())
                else:
                    self.current_parameters[f"predict_curve_{key}"] = None
        self._persist_parameters()
        if self._predict_current_kind == "curve":
            self._rerender_predict_view()

    def _get_curve_xlim(self) -> Optional[Tuple[float, float]]:
        if self.current_parameters.get("predict_curve_autoscale", True):
            return None
        xmin = self.current_parameters.get("predict_curve_xmin")
        xmax = self.current_parameters.get("predict_curve_xmax")
        if xmin is None or xmax is None:
            return None
        return float(xmin), float(xmax)

    def _get_curve_ylim(self) -> Optional[Tuple[float, float]]:
        if self.current_parameters.get("predict_curve_autoscale", True):
            return None
        ymin = self.current_parameters.get("predict_curve_ymin")
        ymax = self.current_parameters.get("predict_curve_ymax")
        if ymin is None or ymax is None:
            return None
        return float(ymin), float(ymax)

    def _show_pixmap_in_predict_view(self, pix: Optional[QPixmap]) -> None:
        if pix is None:
            return
        pview = getattr(self.ui, "predict2dGraphicsView", None)
        if pview is None:
            return
        if self._predict_scene is None:
            self._predict_scene = QGraphicsScene(pview)
            pview.setScene(self._predict_scene)
            pview.setTransformationAnchor(pview.AnchorUnderMouse)
            pview.setDragMode(pview.ScrollHandDrag)
        self._predict_scene.clear()
        self._predict_scene.addPixmap(pix)
        self._predict_scene.setSceneRect(QRectF(pix.rect()))
        self._predict_pixmap = pix
        self._predict_zoom_steps = 0
        self._apply_predict_zoom(reset=True)

    def _rerender_predict_view(self) -> None:
        tabs = getattr(self, "_predict_tabs", None)
        idx = 0
        try:
            if tabs is not None:
                idx = max(0, tabs.currentIndex())
        except Exception:
            idx = 0
        self._render_predict_tab_by_index(idx)

    def _on_predict_log_scale_toggled(self, checked: bool) -> None:
        if self._ui_updating:
            return
        self.current_parameters["predict_log_scale"] = bool(checked)
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_export_clicked(self) -> None:
        """Export prediction results for single-file or multi-file mode."""

        # 检查当前模式
        mode = self.current_parameters.get("mode", "single_file")

        if mode == "multi_files" and self._multifile_results_widget:
            # 多文件模式：触发多文件导出界面
            self._multifile_results_widget.onExportClicked()
            return

        if not self.prediction_results:
            QMessageBox.information(
                self.main_window, "Export", "Run a prediction before exporting the current result."
            )
            self._append_status_message("No prediction result to export", level="WARN")
            return

        # 单文件模式：使用原有逻辑
        spec = None
        tabs = getattr(self, "_predict_tabs", None)
        try:
            if tabs is not None and 0 <= tabs.currentIndex() < len(self._predict_tab_specs):
                spec = self._predict_tab_specs[tabs.currentIndex()]
        except Exception:
            spec = None
        if spec is None and self._predict_tab_specs:
            spec = self._predict_tab_specs[0]
        if spec is None:
            self._append_status_message("No prediction output to export", level="WARN")
            return

        kind = self._predict_current_kind
        if kind is None and isinstance(spec, dict):
            kind = spec.get("kind")

        dialog = QMessageBox(self.main_window)
        dialog.setWindowTitle("Export Predict-2D")
        dialog.setText("Select what to export")
        btn_img = dialog.addButton("Image (JPG)", QMessageBox.AcceptRole)
        btn_data = dialog.addButton("Data (ASCII)", QMessageBox.AcceptRole)
        btn_both = dialog.addButton("Both", QMessageBox.AcceptRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.exec_()
        clicked = dialog.clickedButton()
        if clicked is None or clicked == dialog.button(QMessageBox.Cancel):
            return
        export_image = clicked in (btn_img, btn_both)
        export_data = clicked in (btn_data, btn_both)

        export_path = self._prompt_export_folder("Save Prediction Output To")
        if not export_path:
            return
        if not os.path.isdir(export_path):
            QMessageBox.warning(
                self.main_window, "Export Path", f"Export folder not found: {export_path}"
            )
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_image:
            if self._predict_pixmap is None:
                self._append_status_message("No predict view image to export", level="WARN")
            else:
                img_path = os.path.join(export_path, f"predict_{kind or 'view'}_{timestamp}.jpg")
                try:
                    if not self._predict_pixmap.save(img_path, "JPG"):
                        raise IOError("Save returned False")
                    self._append_status_message(f"Predict image exported: {img_path}")
                except Exception as exc:
                    self._append_status_message(
                        f"Predict image export failed: {exc}", level="ERROR"
                    )

        if export_data:
            try:
                if kind == "curve" and isinstance(self._predict_current_curve, np.ndarray):
                    curve = np.array(self._predict_current_curve, dtype=np.float32)
                    x = np.arange(len(curve), dtype=np.float32)
                    data = np.column_stack([x, curve])
                    data_path = os.path.join(export_path, f"predict_curve_{timestamp}.txt")
                    exported = self.prediction_view_model.export_array(
                        PredictionArrayExportRequest(
                            Path(data_path), data, fmt="%.6g", header="x y", comments=""
                        )
                    )
                    if exported is None:
                        raise OSError(
                            self.prediction_view_model.state.error_message
                            or "Curve data export failed"
                        )
                    self._append_status_message(f"Curve data exported: {exported}")
                elif kind in ("hr", "array", "steps") and isinstance(
                    self._predict_current_image, np.ndarray
                ):
                    arr = np.array(self._predict_current_image, dtype=np.float32)
                    step_suffix = ""
                    if kind == "steps" and isinstance(getattr(self, "_step_snapshots", None), list):
                        try:
                            lbl = self._step_snapshots[self._current_step_index].get("label")
                            if lbl:
                                step_suffix = f"_{str(lbl)}"
                        except Exception:
                            step_suffix = ""
                    data_path = os.path.join(
                        export_path, f"predict_{kind}{step_suffix}_{timestamp}.txt"
                    )
                    exported = self.prediction_view_model.export_array(
                        PredictionArrayExportRequest(Path(data_path), arr, fmt="%.6g")
                    )
                    if exported is None:
                        raise OSError(
                            self.prediction_view_model.state.error_message
                            or "Matrix data export failed"
                        )
                    self._append_status_message(f"Matrix data exported: {exported}")
                else:
                    self._append_status_message("No data available to export", level="WARN")
            except Exception as exc:
                self._append_status_message(f"Predict data export failed: {exc}", level="ERROR")
