"""GISAXS projection into the application-owned interactive pixel viewer."""

from types import SimpleNamespace

from PyQt5.QtWidgets import QMessageBox
from src.gimap.app.presentation import ScientificImageViewer
from ..detector_data_access import analysis_image_for


class InteractiveDetectorMixin:
    def _open_interactive_detector(self):
        window = getattr(self, "_interactive_detector", None)
        if window is None:
            try:
                window = ScientificImageViewer(self.ui.gisaxsInputGraphicsView)
            except ImportError:
                QMessageBox.information(
                    self.ui.gisaxsInputGraphicsView,
                    "Interactive detector",
                    "Install pyqtgraph to enable this viewer.",
                )
                return
            self._interactive_detector = window
            window.log_changed.connect(self.ui.gisaxsInputIntLogCheckBox.setChecked)
            window.levels_changed.connect(self._interactive_levels_changed)
            window.region_selected.connect(self._interactive_region_selected)
        window.show()
        window.raise_()
        image = analysis_image_for(self)
        if image is not None:
            self._try_update_cached_preview(image)

    def _interactive_levels_changed(self, low, high):
        # Reuse the existing shared state commit rather than maintaining local levels.
        for widget, value in (
            (self.ui.gisaxsInputVminValue, low),
            (self.ui.gisaxsInputVmaxValue, high),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._on_color_scale_value_committed()

    def _interactive_region_selected(self, bounds):
        if self._should_show_q_axis():
            return
        x0, x1, y0, y1 = bounds
        self._main_preview_tool = "region"
        self._main_preview_selection_start = (x0, y0)
        self._on_main_preview_release(
            SimpleNamespace(inaxes=self._preview_ax, xdata=x1, ydata=y1, button=1)
        )
        self._set_main_preview_tool(None)

    def _sync_interactive_detector(
        self, preview, intensity, extent, vmin, vmax, show_q_axis, selection_info
    ):
        window = getattr(self, "_interactive_detector", None)
        if window is None or not window.isVisible():
            return
        if show_q_axis:
            window.set_unavailable(
                "Q-space is available in the main detector view. Switch to Pixel to inspect here."
            )
            return
        window.set_frame(
            preview,
            intensity=intensity,
            extent=extent,
            origin="lower",
            levels=(vmin, vmax),
            colormap=self._image_colormap,
            log_scale=self._is_log_mode_enabled(),
        )
        bounds = None
        if selection_info and self._show_cut_region:
            b = selection_info.get("bounds", {})
            bounds = tuple(b.get(key, 0.0) for key in ("x_min", "x_max", "y_min", "y_max"))
        center = self._get_detector_center_for_controller_axis() if self._show_center else None
        window.set_overlay(bounds=bounds, center=center)
