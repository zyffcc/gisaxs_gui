"""Integration Export coordination for WAXS."""

from __future__ import annotations


from pathlib import Path


import numpy as np


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from src.gimap.features.waxs.application import (
    IntegrateWaxsImageRequest,
)


class IntegrationExportMixin:
    """Own integration export presentation behavior."""

    def integrate_current_image(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Integrate", "No image loaded.")
            return
        try:
            cut_kind = "full"
            selection = None
            if self.cut_type_combo.currentText() == "Line Cut":
                cut_kind = "line"
                selection = {
                    "center_x": self.line_center_x_spin.value(),
                    "center_y": self.line_center_y_spin.value(),
                    "width": self.line_width_spin.value(),
                    "height": self.line_height_spin.value(),
                }
            elif self.cut_type_combo.currentText() == "Circle Cut":
                cut_kind = "circle"
                selection = {
                    "center_x": self.circle_center_x_spin.value(),
                    "center_y": self.circle_center_y_spin.value(),
                    "inner_radius": self.circle_inner_spin.value(),
                    "outer_radius": self.circle_outer_spin.value(),
                    "start_angle": self.circle_start_spin.value(),
                    "end_angle": self.circle_end_spin.value(),
                }
            integration = self._integration_settings()
            integration["smooth"] = self.smooth_curve_check.isChecked()
            curve = self.view_model.integrate(
                IntegrateWaxsImageRequest(
                    image=self.current_image,
                    geometry=self._geometry_settings(),
                    integration=integration,
                    mask_min=self._mask_limits()[0],
                    mask_max=self._mask_limits()[1],
                    cut_kind=cut_kind,
                    selection=selection,
                )
            )
            if curve is None:
                raise RuntimeError(self.view_model.state.error_message or "Integration failed.")
            x, y = curve.x, curve.intensity
            self._last_curve = (x, y)
            self._show_1d_view()
            self._plot_curve(x, y)
            self.integration_status.setText(f"Curve calculated: {len(x)} points.")
            self._set_status("1D integration completed")
        except Exception as exc:
            QMessageBox.warning(self, "Integration Failed", f"Failed to integrate:\n{exc}")

    def _plot_curve(self, x: np.ndarray, y: np.ndarray) -> None:
        self.viewer.figure.clear()
        self.viewer.colorbar = None
        self.viewer.cax = None
        self.viewer._preview_cache_key = None
        self.viewer._preview_cache_array = None
        self.viewer._preview_cache_extent = None
        ax = self.viewer.figure.add_subplot(111)
        self.viewer.ax = ax
        ax.plot(x, y)
        ax.set_xlabel(self.x_axis_mode.currentText())
        ax.set_ylabel("Intensity")
        ax.set_title("1D Integration")
        ax.grid(True, alpha=0.25)
        self.viewer.canvas.draw_idle()

    def export_current_curve(self) -> None:
        if self._last_curve is None:
            QMessageBox.information(self, "Export 1D", "No curve calculated.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export 1D Curve", "curve.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        exported = self.view_model.export_curve(Path(self.view_model.normalize_path(path)))
        if exported is None:
            QMessageBox.warning(
                self,
                "Export Failed",
                self.view_model.state.error_message or "Failed to export curve.",
            )
            return
        self._set_status("1D export completed")

    def export_current_image(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Export Image", "No image loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "detector.png", "PNG Image (*.png)"
        )
        if not path:
            return
        image = self.current_image
        if self._current_view_is_cut:
            image, _extent = self._cut_image_by_q_range(image)
        mask_min, mask_max = self._display_mask_limits()
        exported = self.view_model.export_image(
            Path(self.view_model.normalize_path(path)),
            image,
            {
                "log_scale": self.display_log.isChecked(),
                "colormap": self.display_cmap.currentText(),
                "auto_scale": self.display_auto_scale.isChecked(),
                "vmin": self.vmin_spin.value(),
                "vmax": self.vmax_spin.value(),
                "mask_min": mask_min,
                "mask_max": mask_max,
            },
        )
        if exported is None:
            QMessageBox.warning(
                self,
                "Export Failed",
                self.view_model.state.error_message or "Failed to export image.",
            )
            return
        self._set_status("Export completed")
