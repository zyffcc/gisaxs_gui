"""Selected-group preprocessing preview coordination for WAXS batch."""

from __future__ import annotations

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

from src.gimap.features.waxs.application import WaxsBatchPreviewRequest

from ..workers import BatchPreviewWorker


class BatchPreviewMixin:
    """Render one selected batch item after production preprocessing."""

    def _on_batch_group_selection_changed(self, *_args) -> None:
        self.batch_preview_item_spin.setMaximum(999999)
        self.batch_preview_item_spin.setValue(1)

    def preview_batch_preprocessing(self) -> None:
        if self._batch_preview_thread is not None and self._batch_preview_thread.isRunning():
            return
        selected_row = self.batch_sources_table.currentRow()
        if selected_row < 0:
            QMessageBox.information(
                self, "Batch Preview", "Select a data-source row to preview."
            )
            return
        sources = self._batch_sources()
        if not sources or selected_row >= len(sources):
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
        batch = self._build_batch_request(sources, q_range, integration)
        request = WaxsBatchPreviewRequest(
            sources[selected_row], self.batch_preview_item_spin.value() - 1, batch
        )
        self.batch_preview_preprocessing_button.setEnabled(False)
        self.set_job_state("running", "Preparing batch preview...", progress=0)
        self._batch_preview_thread = QThread(self)
        self._batch_preview_worker = BatchPreviewWorker(request, self.view_model)
        self._batch_preview_worker.moveToThread(self._batch_preview_thread)
        self._batch_preview_thread.started.connect(self._batch_preview_worker.run)
        self._batch_preview_worker.finished.connect(self._on_batch_preview_ready)
        self._batch_preview_worker.failed.connect(self._on_batch_preview_failed)
        self._batch_preview_worker.finished.connect(self._batch_preview_thread.quit)
        self._batch_preview_worker.failed.connect(self._batch_preview_thread.quit)
        self._batch_preview_thread.finished.connect(self._cleanup_batch_preview)
        self._batch_preview_thread.start()

    def _on_batch_preview_ready(self, result) -> None:
        frame = result.frame
        use_q = self.batch_export_q_images.isChecked()
        q_coordinates = (
            self.view_model.compute_q_maps(frame.image.shape, frame.geometry)
            if use_q
            else None
        )
        self._show_2d_view()
        self.viewer.show_image(
            frame.image,
            log_scale=self.batch_export_log.isChecked(),
            colormap=self.batch_export_cmap.currentText(),
            auto_scale=self.batch_export_auto_scale.isChecked(),
            vmin=self.batch_export_vmin.value(),
            vmax=self.batch_export_vmax.value(),
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            flip_vertical=False,
            title=f"{result.path.name} - preprocessed preview",
            xlabel="Qr (Å⁻¹)" if use_q else "X (pixel)",
            ylabel="Qz (Å⁻¹)" if use_q else "Y (pixel)",
            q_coordinates=q_coordinates,
            no_data_color=self.display_no_data_color.currentData(),
        )
        if use_q and self.batch_limit_q_range.isChecked():
            self.viewer.ax.set_xlim(self.batch_qr_min.value(), self.batch_qr_max.value())
            self.viewer.ax.set_ylim(self.batch_qz_min.value(), self.batch_qz_max.value())
        self.viewer.canvas.draw_idle()
        self.batch_preview_item_spin.blockSignals(True)
        self.batch_preview_item_spin.setMaximum(result.item_count)
        self.batch_preview_item_spin.setValue(result.item_index + 1)
        self.batch_preview_item_spin.blockSignals(False)
        details = [f"Preview {result.item_index + 1}/{result.item_count}"]
        if self.batch_calibration_enabled.isChecked():
            details.append(f"SDD {frame.geometry['distance']:.6g} mm")
        if frame.normalization_factor is not None:
            details.append(f"normalization ×{frame.normalization_factor:.6g}")
        self.set_job_state("succeeded", " · ".join(details), progress=100)

    def _on_batch_preview_failed(self, message: str) -> None:
        self.set_job_state("failed", "Batch preprocessing preview failed", progress=0)
        QMessageBox.warning(self, "Batch Preview", message)

    def _cleanup_batch_preview(self) -> None:
        self.batch_preview_preprocessing_button.setEnabled(True)
        self._batch_preview_worker = None
        if self._batch_preview_thread is not None:
            self._batch_preview_thread.deleteLater()
        self._batch_preview_thread = None
