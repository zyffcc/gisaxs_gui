"""Batch Processing coordination for WAXS."""

from __future__ import annotations


from pathlib import Path


from PyQt5.QtCore import QThread

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QTableWidgetItem,
)


from src.gimap.features.waxs.application import WaxsBatchRequest, WaxsBatchSource


from ..workers import BatchWorker


class BatchProcessingMixin:
    """Own batch processing presentation behavior."""

    def select_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            normalized = self.view_model.normalize_path(folder)
            row = self.batch_sources_table.rowCount()
            self.batch_sources_table.insertRow(row)
            self.batch_sources_table.setItem(row, 0, QTableWidgetItem(normalized))
            self.batch_sources_table.setItem(
                row,
                1,
                QTableWidgetItem(self.batch_pattern_edit.text().strip() or "*.tif"),
            )
            self.batch_sources_table.setItem(
                row, 2, QTableWidgetItem(Path(normalized).name)
            )
            self.batch_sources_table.selectRow(row)

    def remove_selected_batch_folder(self) -> None:
        row = self.batch_sources_table.currentRow()
        if row >= 0:
            self.batch_sources_table.removeRow(row)

    def select_batch_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.batch_output_edit.setText(self.view_model.normalize_path(folder))

    def _update_batch_export_limits_enabled(self, auto_scale: bool) -> None:
        self.batch_export_vmin.setEnabled(not auto_scale)
        self.batch_export_vmax.setEnabled(not auto_scale)

    def _update_batch_preprocessing_enabled(self, *_args) -> None:
        calibration = self.batch_calibration_enabled.isChecked()
        normalization = self.batch_normalization_enabled.isChecked()
        self.batch_calibration_target.setEnabled(calibration)
        self.batch_calibration_window.setEnabled(calibration)
        self.batch_normalization_target.setEnabled(normalization)
        self.batch_normalization_window.setEnabled(normalization)
        self.batch_normalization_intensity.setEnabled(normalization)
        self.batch_normalization_mode.setEnabled(normalization)

    def copy_current_preview_style(self) -> None:
        self.batch_export_cmap.setCurrentText(self.display_cmap.currentText())
        self.batch_export_log.setChecked(self.display_log.isChecked())
        self.batch_export_auto_scale.setChecked(self.display_auto_scale.isChecked())
        self.batch_export_vmin.setValue(self.vmin_spin.value())
        self.batch_export_vmax.setValue(self.vmax_spin.value())

    def preview_batch_export_style(self) -> None:
        if self.current_image is None:
            QMessageBox.information(
                self, "Export Preview", "Load an image before previewing export style."
            )
            return
        self.display_cmap.setCurrentText(self.batch_export_cmap.currentText())
        self.display_log.setChecked(self.batch_export_log.isChecked())
        self.display_auto_scale.setChecked(
            self.batch_export_auto_scale.isChecked()
        )
        self.vmin_spin.setValue(self.batch_export_vmin.value())
        self.vmax_spin.setValue(self.batch_export_vmax.value())
        if self.batch_export_q_images.isChecked():
            self.coordinate_mode_combo.setCurrentText("q space")
        self._show_2d_view()
        self.refresh_view()
        if self.batch_export_q_images.isChecked() and self.batch_limit_q_range.isChecked():
            self.viewer.ax.set_xlim(
                self.batch_qr_min.value(), self.batch_qr_max.value()
            )
            self.viewer.ax.set_ylim(
                self.batch_qz_min.value(), self.batch_qz_max.value()
            )
            self.viewer.canvas.draw_idle()

    def start_batch(self) -> None:
        if self._batch_thread is not None and self._batch_thread.isRunning():
            return
        sources = self._batch_sources()
        if not sources:
            QMessageBox.warning(
                self, "Batch Processing", "Add at least one valid input folder."
            )
            return
        q_range = self._batch_q_range()
        if q_range is False:
            return
        integration = self._integration_settings()
        if (
            self.batch_calibration_enabled.isChecked()
            or self.batch_normalization_enabled.isChecked()
        ) and (
            integration.get("x_axis") != "q"
            or integration.get("mode") != "radial"
        ):
            QMessageBox.warning(
                self,
                "Batch Preprocessing",
                "Calibration and normalization require Radial mode with the q axis.",
            )
            return
        request = self._build_batch_request(sources, q_range, integration)
        if not (
            request.export_images
            or request.export_q_images
            or request.export_curves
            or request.export_curve_images
        ):
            QMessageBox.information(self, "Batch Processing", "Select at least one export option.")
            return

        self.set_job_state(
            "running",
            "Batch processing started...",
            progress=0,
        )
        self.batch_start_button.setEnabled(False)
        self.batch_pause_button.setEnabled(True)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(True)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(request, self.view_model)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch)
        self._batch_thread.start()

    def _build_batch_request(self, sources, q_range, integration) -> WaxsBatchRequest:
        output_folder = self.batch_output_edit.text().strip() or self.view_model.working_directory()
        return WaxsBatchRequest(
            folder=sources[0].folder,
            pattern=sources[0].pattern,
            output_folder=Path(output_folder),
            export_images=self.batch_export_pixel_images.isChecked(),
            export_curves=self.batch_export_curves.isChecked(),
            export_background_subtracted=False,
            display={
                "log_scale": self.batch_export_log.isChecked(),
                "colormap": self.batch_export_cmap.currentText(),
                "auto_scale": self.batch_export_auto_scale.isChecked(),
                "vmin": self.batch_export_vmin.value(),
                "vmax": self.batch_export_vmax.value(),
                "mask_min": self._display_mask_limits()[0],
                "mask_max": self._display_mask_limits()[1],
            },
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            geometry=self._geometry_settings(),
            integration=integration,
            continue_on_error=True,
            sources=tuple(sources),
            export_q_images=self.batch_export_q_images.isChecked(),
            export_curve_images=self.batch_export_curve_images.isChecked(),
            q_range=q_range,
            calibration_enabled=self.batch_calibration_enabled.isChecked(),
            calibration_target_q=self.batch_calibration_target.value(),
            calibration_half_width=self.batch_calibration_window.value(),
            normalization_enabled=self.batch_normalization_enabled.isChecked(),
            normalization_target_q=self.batch_normalization_target.value(),
            normalization_half_width=self.batch_normalization_window.value(),
            normalization_target_intensity=self.batch_normalization_intensity.value(),
            normalization_mode=str(self.batch_normalization_mode.currentData()),
        )

    def _batch_sources(self) -> list[WaxsBatchSource]:
        sources = []
        output_names = set()
        for row in range(self.batch_sources_table.rowCount()):
            folder_item = self.batch_sources_table.item(row, 0)
            pattern_item = self.batch_sources_table.item(row, 1)
            output_item = self.batch_sources_table.item(row, 2)
            folder = folder_item.text().strip() if folder_item else ""
            pattern = pattern_item.text().strip() if pattern_item else ""
            output_name = output_item.text().strip() if output_item else ""
            if not self.view_model.is_directory(folder):
                QMessageBox.warning(
                    self,
                    "Batch Processing",
                    f"Row {row + 1} has an invalid input folder.",
                )
                return []
            if not output_name:
                output_name = Path(folder).name
            if Path(output_name).name != output_name or output_name in {".", ".."}:
                QMessageBox.warning(
                    self,
                    "Batch Processing",
                    f"Row {row + 1} needs a simple output subfolder name.",
                )
                return []
            if output_name.casefold() in output_names:
                QMessageBox.warning(
                    self,
                    "Batch Processing",
                    f"Output subfolder '{output_name}' is used more than once.",
                )
                return []
            output_names.add(output_name.casefold())
            sources.append(
                WaxsBatchSource(
                    Path(folder), pattern or "*.tif", output_name
                )
            )
        return sources

    def _batch_q_range(self):
        if not self.batch_limit_q_range.isChecked():
            return None
        q_range = {
            "qr_min": self.batch_qr_min.value(),
            "qr_max": self.batch_qr_max.value(),
            "qz_min": self.batch_qz_min.value(),
            "qz_max": self.batch_qz_max.value(),
        }
        if q_range["qr_min"] >= q_range["qr_max"] or q_range["qz_min"] >= q_range["qz_max"]:
            QMessageBox.warning(
                self,
                "Batch Processing",
                "The q range requires min < max for both qr and qz.",
            )
            return False
        return q_range

    def stop_batch(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.stop()
            self.set_job_state(
                "running",
                "Stopping batch processing...",
                progress=self.progress.value(),
            )

    def toggle_batch_pause(self) -> None:
        if self._batch_worker is None:
            return
        paused = self.batch_pause_button.text() == "Pause"
        self._batch_worker.set_paused(paused)
        self.batch_pause_button.setText("Resume" if paused else "Pause")
        self.set_job_state(
            "paused" if paused else "running",
            "Batch processing paused." if paused else "Batch processing resumed.",
            progress=self.progress.value(),
        )

    def _on_batch_progress(self, value: int, message: str) -> None:
        self.set_job_state("running", message, progress=value)

    def _on_batch_finished(self, message: str) -> None:
        self.batch_start_button.setEnabled(True)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(False)
        completed = "completed" in message.lower()
        self.set_job_state(
            "succeeded" if completed else "cancelled",
            message,
            progress=100 if completed else 0,
        )
        QMessageBox.information(self, "Batch Processing", message)

    def _on_batch_failed(self, message: str) -> None:
        self.batch_start_button.setEnabled(True)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(False)
        self.set_job_state("failed", "Batch processing failed", progress=0)
        QMessageBox.warning(self, "Batch Processing Failed", message)

    def _cleanup_batch(self) -> None:
        self._batch_worker = None
        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
        self._batch_thread = None
