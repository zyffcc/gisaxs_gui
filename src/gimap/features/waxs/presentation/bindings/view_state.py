"""View State coordination for WAXS."""

from __future__ import annotations


from pathlib import Path


import numpy as np


class ViewStateMixin:
    """Own view state presentation behavior."""

    def _set_frame_controls_enabled(self, enabled: bool) -> None:
        self.frame_label.setVisible(enabled)
        self.frame_spin.setVisible(enabled)
        self.frame_spin.setEnabled(enabled)

    def _mask_limits(self) -> tuple[float, float]:
        if not self.apply_mask_check.isChecked():
            return -1e12, 1e12
        return self.mask_min_spin.value(), self.mask_max_spin.value()

    def _display_mask_limits(self) -> tuple[float, float]:
        """Mask thresholds are defined in linear intensity space only."""
        if self.display_log.isChecked():
            return -1e12, 1e12
        return self._mask_limits()

    def _geometry_settings(self) -> dict:
        return {
            "incidence": self.incidence_spin.value(),
            "center_x": self.center_x_spin.value(),
            "center_y": self.center_y_spin.value(),
            "distance": self.distance_spin.value(),
            "pixel_x": self.pixel_x_spin.value(),
            "pixel_y": self.pixel_y_spin.value(),
            "wavelength": self.wavelength_spin.value(),
            "qr_min": self.qr_min_spin.value(),
            "qr_max": self.qr_max_spin.value(),
            "qz_min": self.qz_min_spin.value(),
            "qz_max": self.qz_max_spin.value(),
        }

    def _integration_settings(self) -> dict:
        return {
            "mode": self.integration_mode.currentText().lower(),
            "bins": self.bin_spin.value(),
            "x_axis": self.x_axis_mode.currentText().lower(),
        }

    def _cut_image_by_q_range(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        result = self.view_model.cut_image(image, self._geometry_settings())
        return result.image, result.extent

    def _update_auto_colorbar_limits(self) -> None:
        if self.current_image is None:
            return
        limits = self.viewer.display_limits(
            self.current_image,
            log_scale=self.display_log.isChecked(),
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            flip_vertical=False,
        )
        if limits is None:
            return
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(limits[0])
        self.vmax_spin.setValue(limits[1])
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

    def _update_metadata(self, image: np.ndarray) -> None:
        arr = np.asarray(image, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            minmax = f"min/max: {np.nanmin(finite):.6g} / {np.nanmax(finite):.6g}"
        else:
            minmax = "min/max: n/a"
        name = Path(self.current_file).name if self.current_file else "No file"
        self.meta_label.setText(
            f"File: {name} | size: {arr.shape[1]} × {arr.shape[0]} | "
            f"frame: {self.frame_spin.value()} / {self.current_frame_count} | {minmax}"
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(message)

    def set_job_state(
        self,
        state: str,
        message: str,
        *,
        progress: int | None = None,
    ) -> None:
        """Update shared status presentation while retaining 0–100 aliases。"""

        normalized_progress = None if progress is None else progress / 100.0
        self.waxs_job_status.set_state(
            state,
            message,
            progress=normalized_progress,
        )
        if progress is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(progress))))
        self.statusChanged.emit(message)
