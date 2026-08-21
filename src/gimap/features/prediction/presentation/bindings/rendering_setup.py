"""Rendering Setup coordination for Prediction."""

from __future__ import annotations


from pathlib import Path

from typing import Dict, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QSignalBlocker


from PyQt5.QtWidgets import (
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QSizePolicy,
)


class RenderingSetupMixin:
    """Own rendering setup presentation behavior."""

    def _preprocess_for_module(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Ensure a module is selected; fall back to saved name or first available
        if not self._current_module:
            try:
                name = (
                    self.current_parameters.get("module_name", "")
                    if isinstance(self.current_parameters, dict)
                    else ""
                )
                if not name and self._modules_by_name:
                    name = sorted(self._modules_by_name.keys())[0]
                if name and name in self._modules_by_name:
                    self._current_module = self._modules_by_name.get(name)
            except Exception:
                pass
        if image is None:
            return None
        typed_module = (
            self._current_module.get("_prediction_module")
            if isinstance(self._current_module, dict)
            else None
        )
        if typed_module is None:
            self._append_status_message(
                "Selected module has no typed prediction contract",
                level="ERROR",
            )
            return None
        prepared = self.prediction_view_model.prepare_input(image, typed_module)
        if prepared is None:
            self._append_status_message(
                self.prediction_view_model.state.error_message or "Module preprocessing failed",
                level="ERROR",
            )
            return None
        self._latest_preprocess_steps = list(prepared.steps)
        self._append_status_message(f"Module preprocess output shape {prepared.values.shape}")
        return prepared.values

    def _predict_with_current_model(self, inp: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if self._current_model is None or inp is None:
            return None
        typed_module = (
            self._current_module.get("_prediction_module")
            if isinstance(self._current_module, dict)
            else None
        )
        model_path = str(self.current_parameters.get("module_model_path") or "")
        if typed_module is None or not model_path:
            self._append_status_message(
                "Selected module has no typed prediction contract or model path",
                level="ERROR",
            )
            return None
        result = self.prediction_view_model.predict_prepared(
            inp,
            typed_module,
            Path(model_path),
            getattr(self, "_latest_preprocess_steps", ()),
        )
        if result is None:
            self._append_status_message(
                self.prediction_view_model.state.error_message or "Isolated prediction failed",
                level="ERROR",
            )
            return None
        return dict(result.outputs)

    def _get_or_create_predict2d_tabs(self) -> Optional[QTabWidget]:
        # Embed inner tabs inside the existing Predict-2D tab of the main tab widget
        main_tabs = getattr(self.ui, "gisaxsPredictImageShowTabWidget", None)
        if main_tabs is None:
            return None
        pred_index = -1
        try:
            for i in range(main_tabs.count()):
                try:
                    label = main_tabs.tabText(i)
                    if isinstance(label, str) and label.lower().strip() in (
                        "predict-2d",
                        "predict 2d",
                        "predict",
                    ):
                        pred_index = i
                        break
                except Exception:
                    pass
        except Exception:
            pass
        if pred_index < 0:
            # fallback to current tab
            try:
                pred_index = main_tabs.currentIndex()
            except Exception:
                pred_index = 0
        pred_page = main_tabs.widget(pred_index)
        if pred_page is None:
            return None
        layout = pred_page.layout()
        if layout is None:
            layout = QVBoxLayout(pred_page)
        # Reuse existing inner tabs if present
        try:
            inner_tabs = next(iter(pred_page.findChildren(QTabWidget)), None)
        except Exception:
            inner_tabs = None
        if inner_tabs is None:
            inner_tabs = QTabWidget(pred_page)
            # 允许横向扩展，不限制最大宽度，避免挤压父容器
            try:
                inner_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            except Exception:
                pass
            layout.addWidget(inner_tabs)
        self._predict_tabs = inner_tabs
        return inner_tabs

    def _rebuild_predict_tabs(self, tabs: QTabWidget) -> None:
        blocker = QSignalBlocker(tabs)
        try:
            while tabs.count() > 0:
                w = tabs.widget(0)
                tabs.removeTab(0)
                if w:
                    w.deleteLater()
            for spec in self._predict_tab_specs:
                page = QWidget()
                # 不要将页面最大高度设为0，保持可扩展的尺寸策略
                try:
                    page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                except Exception:
                    pass
                tabs.addTab(page, str(spec.get("title", "Panel")))
        finally:
            del blocker
        try:
            tabs.currentChanged.disconnect(self._on_predict_tab_changed)
        except Exception:
            pass
        tabs.currentChanged.connect(self._on_predict_tab_changed)
        if self._predict_tab_specs:
            tabs.setCurrentIndex(0)

    def _on_predict_tab_changed(self, index: int) -> None:
        self._render_predict_tab_by_index(index)

    def _render_predict_tab_by_index(self, index: int) -> None:
        if index < 0 or index >= len(self._predict_tab_specs):
            return
        spec = self._predict_tab_specs[index]
        self._render_predict_panel(spec)

    def _render_predict_panel(self, spec: Dict[str, object]) -> None:
        # Clear any step buttons when switching kinds
        if getattr(self, "_step_buttons", None):
            try:
                for b in self._step_buttons:
                    if b and hasattr(b, "deleteLater"):
                        b.deleteLater()
            except Exception:
                pass
        self._step_buttons = []

        kind = spec.get("kind") if isinstance(spec, dict) else None
        data = spec.get("data") if isinstance(spec, dict) else None
        self._predict_current_kind = kind if isinstance(kind, str) else None
        self._predict_current_curve = None
        if kind == "hr" and isinstance(data, np.ndarray):
            self._render_predict2d_into_view(data)
            self._refresh_predict_controls("hr")
            return
        if kind == "array" and isinstance(data, np.ndarray):
            self._predict_current_image = data
            disp, vmin, vmax = self._prepare_predict_image(data)
            cmap = (
                spec.get("colormap")
                if isinstance(spec.get("colormap"), str)
                else self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0])
            )
            pix = self._create_pixmap_from_array(disp, vmin, vmax, cmap)
            self._show_pixmap_in_predict_view(pix)
            self._refresh_predict_controls("array")
            return
        if kind == "curve" and isinstance(data, np.ndarray):
            title = spec.get("title", "Curve")
            xlabel = spec.get("xlabel", "Index")
            self._predict_current_curve = data
            pix = self._render_curve_figure(
                data,
                x_label=str(xlabel),
                title=str(title),
                log_x=bool(self.current_parameters.get("predict_curve_logx", False)),
                log_y=bool(self.current_parameters.get("predict_curve_logy", False)),
                xlim=self._get_curve_xlim(),
                ylim=self._get_curve_ylim(),
            )
            self._show_pixmap_in_predict_view(pix)
            self._refresh_predict_controls("curve")
            return
        if kind == "parameters" and isinstance(data, np.ndarray):
            names = spec.get("names") if isinstance(spec.get("names"), list) else None
            pix = self._render_parameters_figure(
                data, [str(name) for name in names] if names else None
            )
            if pix is not None:
                self._show_pixmap_in_predict_view(pix)
            self._refresh_predict_controls("parameters")
            return
        if kind == "steps":
            steps = spec.get("steps") if isinstance(spec.get("steps"), list) else []
            if not steps:
                return
            self._step_snapshots = steps
            # Show the final model input by default when the preprocess panel provides it.
            default_idx = spec.get("default_index") if isinstance(spec, dict) else None
            if isinstance(default_idx, int) and 0 <= default_idx < len(steps):
                start_idx = default_idx
            else:
                start_idx = (
                    self._current_step_index if 0 <= self._current_step_index < len(steps) else 0
                )
            self._render_step_snapshot(start_idx)
            self._refresh_predict_controls("steps")
            # Build buttons under the tabs page to switch steps
            tabs = getattr(self, "_predict_tabs", None)
            page = tabs.currentWidget() if tabs else None
            if page is None:
                return
            layout = page.layout()
            if layout is None:
                layout = QVBoxLayout(page)
            # Clear existing items in page layout
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            # Estimate columns based on viewport width to avoid stretching right side
            cols = 4
            try:
                pview = getattr(self.ui, "predict2dGraphicsView", None)
                if pview is not None:
                    vw = max(1, pview.viewport().size().width())
                    cols = max(1, vw // 120)
            except Exception:
                pass
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            btns = []
            for idx, st in enumerate(steps):
                lbl = st.get("label") or st.get("step") or f"Step {idx + 1}"
                btn = QPushButton(str(lbl))
                btn.setCheckable(True)
                btn.setChecked(idx == start_idx)
                btn.clicked.connect(lambda checked, i=idx: self._render_step_snapshot(i))
                r, c = divmod(idx, cols)
                grid.addWidget(btn, r, c)
                btns.append(btn)
            layout.addLayout(grid)
            try:
                row_count = (len(btns) + cols - 1) // cols
                row_h = btns[0].sizeHint().height() if btns else 24
                # 仅设置最小高度，允许父布局根据可用空间扩展
                page.setMinimumHeight(row_count * (row_h + 6) + 4)
                try:
                    page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                except Exception:
                    pass
            except Exception:
                pass
            self._step_buttons = btns
            return

    def _predict_viewport_pixels(self) -> Optional[Tuple[int, int]]:
        pview = getattr(self.ui, "predict2dGraphicsView", None)
        if pview is None:
            return None
        viewport = pview.viewport().size()
        return (max(400, viewport.width()), max(320, viewport.height()))
