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

    def select_background_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Background File", "", SCATTERING_FILTER
        )
        if not file_path:
            return
        normalized = self.view_model.normalize_path(file_path)
        self.background_path_edit.setText(normalized)
        self.background_enable.setChecked(True)
        self._update_background_frame_range(Path(normalized))
        self._reload_current_for_background()

    def clear_background(self) -> None:
        self.background_path_edit.clear()
        self.background_enable.setChecked(False)
        self._update_background_frame_range(None)
        self._reload_current_for_background()

    def _on_background_changed(self, *_args) -> None:
        self._update_background_frame_range_from_edit()
        self._reload_current_for_background()

    def _update_background_frame_range_from_edit(self) -> None:
        text = self.background_path_edit.text().strip()
        if text:
            self._update_background_frame_range(Path(text))
        else:
            self._update_background_frame_range(None)

    def _update_background_frame_range(self, path: Path | None) -> None:
        """Set the background frame spin range based on file type/frame count."""
        if getattr(self, "background_frame_spin", None) is None:
            return
        if path is None:
            self.background_frame_spin.setEnabled(False)
            self.background_frame_spin.setMaximum(1)
            self.background_frame_spin.setValue(1)
            return
        suffix = path.suffix.lower()
        if suffix != ".nxs":
            self.background_frame_spin.setEnabled(False)
            self.background_frame_spin.setMaximum(1)
            self.background_frame_spin.setValue(1)
            return
        try:
            count = self.view_model.frame_count(path)
        except Exception:
            count = 1
        count = max(1, int(count))
        self.background_frame_spin.setMaximum(count)
        self.background_frame_spin.setEnabled(count > 1)
        if self.background_frame_spin.value() > count:
            self.background_frame_spin.setValue(count)

    def preview_background_dialog(self) -> None:
        text = self.background_path_edit.text().strip()
        if not text:
            QMessageBox.information(
                self, "Preview Background", "No background file selected."
            )
            return
        from PyQt5.QtWidgets import (
            QDialog,
            QHBoxLayout,
            QLabel,
            QSpinBox,
            QVBoxLayout,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Background Preview")
        dialog.resize(560, 480)
        layout = QVBoxLayout(dialog)
        header = QHBoxLayout()
        header_label = QLabel(Path(text).name)
        header.addWidget(header_label)
        header.addStretch(1)
        layout.addLayout(header)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        frame_edit = QSpinBox(dialog)
        frame_edit.setMinimum(1)
        frame_edit.setMaximum(
            max(1, int(self._background_preview_frame_count(Path(text))))
        )
        frame_edit.setValue(self.background_frame_spin.value())
        frame_row.addWidget(frame_edit)
        frame_row.addStretch(1)
        layout.addLayout(frame_row)
        from .image_viewer import ScatteringImageViewer

        viewer = ScatteringImageViewer(view_model=self.view_model)
        layout.addWidget(viewer, 1)

        def render() -> None:
            image = self.view_model.preview_background(
                Path(text), int(frame_edit.value() - 1)
            )
            if image is None:
                QMessageBox.warning(
                    dialog, "Preview Background", "Could not load background frame."
                )
                return
            viewer.show_image(
                image,
                log_scale=self.display_log.isChecked(),
                colormap=self.display_cmap.currentText(),
                auto_scale=True,
                vmin=self.vmin_spin.value(),
                vmax=self.vmax_spin.value(),
                mask_min=0.0,
                mask_max=0.0,
                flip_vertical=self.display_flip.isChecked(),
                title=f"Background {Path(text).name} - frame {frame_edit.value()}",
            )

        frame_edit.valueChanged.connect(lambda _v: render())
        render()
        dialog.exec_()

    def _background_preview_frame_count(self, path: Path) -> int:
        try:
            return self.view_model.frame_count(path)
        except Exception:
            return 1

    def _reload_current_for_background(self) -> None:
        """Re-apply background settings to the currently displayed image."""
        if self.current_file:
            self._start_loader(self.current_file, self.frame_spin.value() - 1)

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

    def _background_settings(
        self,
    ) -> tuple[str | None, float, int]:
        """Return current background path, coefficient and frame from the UI.

        Returns ``(None, coefficient, frame_index)`` when background subtraction
        is disabled or no background path has been chosen, so the loader applies
        no change.
        """
        if not getattr(self, "background_enable", None) or not getattr(
            self, "background_path_edit", None
        ):
            return None, 1.0, 0
        if not self.background_enable.isChecked():
            return None, 1.0, 0
        path = self.background_path_edit.text().strip()
        coefficient = (
            float(self.background_coefficient_spin.value())
            if getattr(self, "background_coefficient_spin", None)
            else 1.0
        )
        frame_index = (
            int(self.background_frame_spin.value() - 1)
            if getattr(self, "background_frame_spin", None)
            else 0
        )
        if not path:
            return None, coefficient, 0
        return path, coefficient, frame_index

    def _start_loader(self, file_path: str, frame_index: int) -> None:
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._set_status("A file is already loading...")
            return

        background_path, background_coefficient, background_frame_index = (
            self._background_settings()
        )
        self.set_job_state(
            "running",
            f"Loading {Path(file_path).name}...",
        )
        self._loader_thread = QThread(self)
        self._loader_worker = ImageLoadWorker(
            file_path,
            frame_index,
            self.view_model,
            background_path=background_path,
            background_coefficient=background_coefficient,
            background_frame_index=background_frame_index,
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
        self._update_geometry_summaries()
        self._update_auto_colorbar_limits()
        self._show_2d_view()
        self.refresh_view()
        self.set_job_state(
            "succeeded",
            f"Loaded {Path(result.file_path).name}",
            progress=100,
        )

    def _on_image_load_failed(self, message: str) -> None:
        window = getattr(self.viewer, "interactive_window", None)
        if window is not None:
            window.stop_playback()
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
        q_coordinates = None
        title = Path(self.current_file).name if self.current_file else "Detector Image"
        if self._current_view_is_cut:
            image, _legacy_extent = self._cut_image_by_q_range(image)
        if self.coordinate_mode_combo.currentText() == "q space":
            q_coordinates = self.view_model.compute_q_maps(
                self.current_image.shape,
                self._geometry_settings(),
            )
            xlabel = "Qr (Å⁻¹)"
            ylabel = "Qz (Å⁻¹)"
            title = f"{title} - q space"
        elif self._current_view_is_cut:
            title = f"{title} - Cut mask"
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
            q_coordinates=q_coordinates,
            no_data_color=self.display_no_data_color.currentData(),
        )
        if q_coordinates is not None and self._current_view_is_cut:
            geometry = self._geometry_settings()
            if geometry["qr_min"] != -121.0 and geometry["qr_max"] != -121.0:
                self.viewer.ax.set_xlim(geometry["qr_min"], geometry["qr_max"])
            if geometry["qz_min"] != -121.0 and geometry["qz_max"] != -121.0:
                self.viewer.ax.set_ylim(geometry["qz_min"], geometry["qz_max"])
        self._draw_overlays()
        self._sync_interactive_detector_overlays()
        self._update_metadata(image)

    def _on_view_tab_changed(self, index: int) -> None:
        if index == 1:
            window = getattr(self.viewer, "interactive_window", None)
            if window is not None:
                window.set_unavailable("Select the 2D detector tab to inspect pixels.")
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
        window = getattr(self.viewer, "interactive_window", None)
        if window is not None:
            window.set_unavailable("Select the 2D detector tab to inspect pixels.")
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
