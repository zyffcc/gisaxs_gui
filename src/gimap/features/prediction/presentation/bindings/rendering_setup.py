"""Rendering Setup coordination for Prediction."""

from __future__ import annotations


from pathlib import Path

from typing import Dict, Optional, Tuple

import numpy as np

from PyQt5.QtCore import QSize, Qt, QSignalBlocker

from PyQt5.QtGui import QIcon


from PyQt5.QtWidgets import (
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QToolButton,
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
        self._latest_model_input = prepared.values
        self._latest_preprocess_source = image
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
        # Use widget identity because the visible label is presentation-owned
        # (currently "Prediction result") and may be translated or renamed.
        main_tabs = getattr(self.ui, "gisaxsPredictImageShowTabWidget", None)
        if main_tabs is None:
            return None
        pred_page = getattr(self.ui, "predict2dImageTab", None)
        if pred_page is None:
            pred_index = next(
                (
                    index
                    for index in range(main_tabs.count())
                    if main_tabs.tabText(index).strip().casefold()
                    in {"prediction result", "predict-2d", "predict 2d", "predict"}
                ),
                -1,
            )
            pred_page = main_tabs.widget(pred_index) if pred_index >= 0 else None
        if pred_page is None:
            return None
        layout = pred_page.layout()
        if layout is None:
            layout = QVBoxLayout(pred_page)
        # Reuse existing inner tabs if present
        try:
            inner_tabs = pred_page.findChild(QTabWidget, "predictionOutputTabs")
        except Exception:
            inner_tabs = None
        if inner_tabs is None:
            inner_tabs = QTabWidget(pred_page)
            inner_tabs.setObjectName("predictionOutputTabs")
            inner_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            if isinstance(layout, QGridLayout):
                # Keep the output selector and step gallery above the large
                # canvas, where they remain visible without scrolling past the
                # image and inspector.
                pview = getattr(self.ui, "predict2dGraphicsView", None)
                inspector = getattr(self.ui, "predict2dParameterWidget", None)
                if pview is not None:
                    layout.removeWidget(pview)
                if inspector is not None:
                    layout.removeWidget(inspector)
                layout.addWidget(inner_tabs, 0, 0, 1, 2)
                if pview is not None:
                    layout.addWidget(pview, 1, 0)
                if inspector is not None:
                    layout.addWidget(inspector, 1, 1)
                layout.setRowStretch(0, 0)
                layout.setRowStretch(1, 1)
            else:
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
        self._predict_current_curve_x = None
        tabs = getattr(self, "_predict_tabs", None)
        if tabs is not None and kind != "steps":
            compact_height = max(34, tabs.tabBar().sizeHint().height() + 8)
            tabs.setMinimumHeight(compact_height)
            tabs.setMaximumHeight(compact_height)
        if kind == "hr" and isinstance(data, np.ndarray):
            axes = spec.get("axes") if isinstance(spec.get("axes"), dict) else None
            self._render_predict2d_into_view(data, axes=axes)
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
            curve_x = spec.get("x")
            self._predict_current_curve_x = curve_x if isinstance(curve_x, np.ndarray) else None
            pix = self._render_curve_figure(
                data,
                x_label=str(xlabel),
                title=str(title),
                x=self._predict_current_curve_x,
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
            # Build a scrollable image gallery under the main preview. Plain
            # text buttons hid the actual intermediate images and made this
            # tab look empty.
            tabs = getattr(self, "_predict_tabs", None)
            page = tabs.currentWidget() if tabs else None
            if page is None:
                return
            tabs.setMinimumHeight(210)
            tabs.setMaximumHeight(260)
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
                    vw = max(600, pview.viewport().size().width())
                    cols = max(1, vw // 150)
            except Exception:
                pass
            scroll = QScrollArea(page)
            scroll.setObjectName("preprocessStepGallery")
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(160)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            gallery = QWidget(scroll)
            grid = QGridLayout(gallery)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            btns = []
            for idx, st in enumerate(steps):
                lbl = st.get("label") or st.get("step") or f"Step {idx + 1}"
                btn = QToolButton(gallery)
                btn.setObjectName(f"preprocessStepThumbnail{idx + 1}")
                btn.setProperty("stepIndex", idx)
                btn.setText(str(lbl))
                btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                btn.setIconSize(QSize(132, 88))
                btn.setMinimumSize(QSize(144, 116))
                preview = self._preprocess_step_display_array(st)
                if isinstance(preview, np.ndarray):
                    vmin, vmax = self._auto_scale_percentiles(preview, 1, 99.8)
                    pixmap = self._create_pixmap_from_array(
                        preview,
                        vmin,
                        vmax,
                        self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0]),
                    )
                    if pixmap is not None:
                        btn.setIcon(QIcon(pixmap))
                btn.setCheckable(True)
                btn.setChecked(idx == start_idx)
                btn.clicked.connect(lambda checked, i=idx: self._render_step_snapshot(i))
                r, c = divmod(idx, cols)
                grid.addWidget(btn, r, c)
                btns.append(btn)
            scroll.setWidget(gallery)
            layout.addWidget(scroll)
            try:
                row_count = (len(btns) + cols - 1) // cols
                gallery.setMinimumHeight(row_count * 122)
                page.setMinimumHeight(0)
                try:
                    page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                except Exception:
                    pass
            except Exception:
                pass
            self._step_buttons = btns
            self._step_gallery_scroll = scroll
            return

    def _predict_viewport_pixels(self) -> Optional[Tuple[int, int]]:
        pview = getattr(self.ui, "predict2dGraphicsView", None)
        if pview is None:
            return None
        viewport = pview.viewport().size()
        return (max(400, viewport.width()), max(320, viewport.height()))
