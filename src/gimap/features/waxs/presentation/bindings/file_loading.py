"""File Loading coordination for WAXS."""

from __future__ import annotations


from pathlib import Path


from PyQt5.QtCore import QThread

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from ..file_types import SCATTERING_FILTER, SUPPORTED_EXTENSIONS
from ..workers import ImageLoadResult, ImageLoadWorker


class FileLoadingMixin:
    """Own file loading presentation behavior."""

    def open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Scattering File", "", SCATTERING_FILTER
        )
        if file_path:
            self.load_file(self.view_model.normalize_path(file_path))

    def load_file(self, file_path: str, frame_index: int = 0) -> None:
        suffix = Path(file_path).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(
                self,
                "Unsupported File Type",
                "Unsupported file type. Please select .nxs, .tif, or .tiff.",
            )
            return
        self.current_file = self.view_model.normalize_path(file_path)
        self._start_loader(self.current_file, frame_index)

    def reload_current_file(self) -> None:
        if not self.current_file:
            QMessageBox.information(self, "Reload", "No image loaded.")
            return
        self._start_loader(self.current_file, self.frame_spin.value() - 1)

    def _start_loader(self, file_path: str, frame_index: int) -> None:
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._set_status("A file is already loading...")
            return

        self.set_job_state(
            "running",
            f"Loading {Path(file_path).name}...",
        )
        self._loader_thread = QThread(self)
        self._loader_worker = ImageLoadWorker(
            file_path,
            frame_index,
            self.view_model,
        )
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.finished.connect(self._on_image_loaded)
        self._loader_worker.failed.connect(self._on_image_load_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._cleanup_loader)
        self._loader_thread.start()

    def _on_image_loaded(self, result: ImageLoadResult) -> None:
        self.current_file = result.file_path
        self.current_image = result.image
        self.current_frame_count = max(1, result.frame_count)
        self._current_view_is_cut = False
        self._cut_extent = None
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(self.current_frame_count)
        self.frame_spin.setValue(result.frame_index + 1)
        self.frame_spin.blockSignals(False)
        self._set_frame_controls_enabled(Path(result.file_path).suffix.lower() == ".nxs")

        self._sync_selection_defaults_to_image()
        self._update_auto_colorbar_limits()
        self._show_2d_view()
        self.refresh_view()
        self.set_job_state(
            "succeeded",
            f"Loaded {Path(result.file_path).name}",
            progress=100,
        )

    def _on_image_load_failed(self, message: str) -> None:
        self.set_job_state("failed", "Failed to load file", progress=0)
        QMessageBox.warning(self, "Failed to Load File", f"Failed to load file:\n{message}")

    def _cleanup_loader(self) -> None:
        self._loader_worker = None
        if self._loader_thread is not None:
            self._loader_thread.deleteLater()
        self._loader_thread = None

    def _on_frame_changed(self, value: int) -> None:
        if self.current_file and Path(self.current_file).suffix.lower() == ".nxs":
            self._start_loader(self.current_file, value - 1)

    def refresh_view(self) -> None:
        if self.current_image is None:
            return
        if self._active_view != "2d":
            return
        image = self.current_image
        extent = None
        xlabel = "X (pixel)"
        ylabel = "Y (pixel)"
        title = Path(self.current_file).name if self.current_file else "Detector Image"
        if self._current_view_is_cut:
            image, extent = self._cut_image_by_q_range(image)
            xlabel = "Qr (Å⁻¹)"
            ylabel = "Qz (Å⁻¹)"
            title = f"{title} - Cut"
        mask_min, mask_max = self._display_mask_limits()
        self.viewer.show_image(
            image,
            log_scale=self.display_log.isChecked(),
            colormap=self.display_cmap.currentText(),
            auto_scale=self.display_auto_scale.isChecked(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            mask_min=mask_min,
            mask_max=mask_max,
            flip_vertical=self.display_flip.isChecked(),
            title=title,
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        self._draw_overlays()
        self._update_metadata(image)

    def _on_view_tab_changed(self, index: int) -> None:
        if index == 1:
            self._active_view = "1d"
            if self._last_curve is not None:
                self._plot_curve(*self._last_curve)
            else:
                self.viewer.figure.clear()
                self.viewer.ax = self.viewer.figure.add_subplot(111)
                self.viewer.cax = None
                self.viewer.colorbar = None
                self.viewer.ax.text(
                    0.5,
                    0.5,
                    "No 1D curve calculated",
                    ha="center",
                    va="center",
                    transform=self.viewer.ax.transAxes,
                )
                self.viewer.ax.set_axis_off()
                self.viewer.canvas.draw_idle()
            return
        self._active_view = "2d"
        self.refresh_view()

    def _show_2d_view(self) -> None:
        self._active_view = "2d"
        self.view_tabs.blockSignals(True)
        self.view_tabs.setCurrentIndex(0)
        self.view_tabs.blockSignals(False)

    def _show_1d_view(self) -> None:
        self._active_view = "1d"
        self.view_tabs.blockSignals(True)
        self.view_tabs.setCurrentIndex(1)
        self.view_tabs.blockSignals(False)

    def _on_log_intensity_toggled(self, checked: bool) -> None:
        self.vmin_spin.setToolTip(
            "Colorbar minimum in log10(intensity) units."
            if checked
            else "Colorbar minimum in linear intensity units."
        )
        self.vmax_spin.setToolTip(
            "Colorbar maximum in log10(intensity) units."
            if checked
            else "Colorbar maximum in linear intensity units."
        )
        self.mask_min_spin.setEnabled(not checked)
        self.mask_max_spin.setEnabled(not checked)
        self.apply_mask_check.setEnabled(not checked)
        self._update_auto_colorbar_limits()
        self.refresh_view()
