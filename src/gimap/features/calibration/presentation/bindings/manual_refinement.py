"""Manual Refinement behavior for Calibration."""

from __future__ import annotations

import logging


from PyQt5.QtWidgets import (
    QMessageBox,
)


LOGGER = logging.getLogger(__name__)


class ManualRefinementMixin:
    """Own manual refinement presentation behavior."""

    def fit_selected_ring(self) -> None:
        if (
            self.result is None
            or self.experimental_ring_combo.currentData() is None
            or self.theory_ring_combo.currentData() is None
        ):
            return
        try:
            distance = self.view_model.manual_ring_distance(
                float(self.experimental_ring_combo.currentData()),
                float(self.theory_ring_combo.currentData()),
            )
            self.manual_distance.setValue(distance)
            self.stage_label.setText(
                "Manual distance updated from the selected experimental/theoretical ring pair."
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Manual Refinement", str(exc))

    def _preview_press(self, event) -> None:
        if (
            self.manual_group.isChecked()
            and event.inaxes is self.axes
            and event.xdata is not None
            and event.ydata is not None
        ):
            self._dragging_center = True
            self.manual_x.setValue(event.xdata)
            self.manual_y.setValue(event.ydata)

    def _preview_move(self, event) -> None:
        if (
            self._dragging_center
            and event.inaxes is self.axes
            and event.xdata is not None
            and event.ydata is not None
        ):
            self.manual_x.setValue(event.xdata)
            self.manual_y.setValue(event.ydata)

    def _preview_release(self, _event) -> None:
        self._dragging_center = False

    def _commit_manual_values(self) -> None:
        if self.result is not None:
            self.view_model.commit_manual_refinement(
                manual_enabled=self.manual_group.isChecked(),
                center_x_px=self.manual_x.value(),
                center_y_px=self.manual_y.value(),
                distance_mm=self.manual_distance.value(),
            )

    def _sync_main_window_geometry(self) -> None:
        """把已保存的 calibration 结果反映到现有 PyQt controls。"""
        if self.result is None or self.main_window is None:
            return
        candidate = self.result.selected_candidate
        page = getattr(getattr(self.main_window, "components", None), "waxs_page", None)
        if page is not None:
            controls = {
                "center_x_spin": candidate.center_x_px,
                "center_y_spin": candidate.center_y_px,
                "distance_spin": candidate.distance_mm,
                "pixel_x_spin": self.result.pixel_size_x_m * 1e6,
                "pixel_y_spin": self.result.pixel_size_y_m * 1e6,
                "wavelength_spin": self.result.wavelength_angstrom,
            }
            for name, value in controls.items():
                widget = getattr(page, name, None)
                if widget is not None:
                    widget.setValue(float(value))
            if hasattr(page, "refresh_view"):
                page.refresh_view()
        if hasattr(self.main_window, "statusbar"):
            self.main_window.statusbar.showMessage(
                "Geometry calibration applied: center "
                f"({candidate.center_x_px:.2f}, {candidate.center_y_px:.2f}), "
                f"distance {candidate.distance_mm:.2f} mm"
            )
