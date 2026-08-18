"""Batch Processing coordination for WAXS."""

from __future__ import annotations


from pathlib import Path


from PyQt5.QtCore import QThread

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from src.gimap.features.waxs.application import (
    WaxsBatchRequest,
)


from ..workers import BatchWorker


class BatchProcessingMixin:
    """Own batch processing presentation behavior."""

    def select_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.batch_folder_edit.setText(self.view_model.normalize_path(folder))

    def select_batch_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.batch_output_edit.setText(self.view_model.normalize_path(folder))

    def start_batch(self) -> None:
        if self._batch_thread is not None and self._batch_thread.isRunning():
            return
        folder = self.batch_folder_edit.text().strip()
        if not self.view_model.is_directory(folder):
            QMessageBox.warning(self, "Batch Processing", "Please select a valid input folder.")
            return
        output_folder = self.batch_output_edit.text().strip() or self.view_model.working_directory()
        request = WaxsBatchRequest(
            folder=Path(folder),
            pattern=self.batch_pattern_edit.text().strip() or "*.tif",
            output_folder=Path(output_folder),
            export_images=self.batch_export_images.isChecked(),
            export_curves=self.batch_export_curves.isChecked(),
            export_background_subtracted=self.batch_export_subbg.isChecked(),
            display={
                "log_scale": self.display_log.isChecked(),
                "colormap": self.display_cmap.currentText(),
                "auto_scale": self.display_auto_scale.isChecked(),
                "vmin": self.vmin_spin.value(),
                "vmax": self.vmax_spin.value(),
                "mask_min": self._display_mask_limits()[0],
                "mask_max": self._display_mask_limits()[1],
            },
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            geometry=self._geometry_settings(),
            integration=self._integration_settings(),
            continue_on_error=False,
        )
        if not (
            request.export_images or request.export_curves or request.export_background_subtracted
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
