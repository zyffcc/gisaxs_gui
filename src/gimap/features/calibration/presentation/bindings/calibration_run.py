"""Calibration Run behavior for Calibration."""

from __future__ import annotations

import logging


from PyQt5.QtCore import QSignalBlocker, QThread, QTimer

from PyQt5.QtWidgets import (
    QMessageBox,
    QTableWidgetItem,
)

from ...application import (
    CalibrationCancelledError,
    CalibrationResult,
)


from ..workers import CalibrationWorker

LOGGER = logging.getLogger(__name__)


class CalibrationRunMixin:
    """Own calibration run presentation behavior."""

    def start_calibration(self) -> None:
        if self.image is None or (self._cal_thread is not None and self._cal_thread.isRunning()):
            return
        try:
            options = {
                "energy_kev": self.energy_spin.value(),
                "standard_key": self.standard_combo.currentData(),
                "estimated_distance_mm": self.estimated_distance_spin.value() or None,
                "distance_range_mm": self._distance_range(),
                "pixel_size_x_m": self.pixel_x_spin.value() * 1e-6,
                "pixel_size_y_m": self.pixel_y_spin.value() * 1e-6,
                "subtract_background": self.background_check.isChecked(),
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration Input", str(exc))
            return
        self.job_status.set_state(
            "running",
            "Starting calibration...",
            progress=0.0,
        )
        self._set_running(True)
        self._cal_thread = QThread(self)
        self._cal_worker = CalibrationWorker(self.view_model, options)
        self._cal_worker.moveToThread(self._cal_thread)
        self._cal_thread.started.connect(self._cal_worker.run)
        self._cal_worker.progress.connect(self._calibration_progress)
        self._cal_worker.finished.connect(self._calibration_finished)
        self._cal_worker.failed.connect(self._calibration_failed)
        self._cal_worker.finished.connect(self._cal_thread.quit)
        self._cal_worker.failed.connect(self._cal_thread.quit)
        self._cal_thread.finished.connect(self._cleanup_calibration)
        self._cal_thread.start()

    def cancel_calibration(self) -> None:
        if self._cal_worker is not None:
            self._cal_worker.cancel()
            self.job_status.set_state(
                "running",
                "Cancelling after the current numerical step...",
                progress=self.progress.value() / max(1, self.progress.maximum()),
            )
            self.cancel_button.setEnabled(False)

    def _calibration_progress(self, value: int, stage: str) -> None:
        self.job_status.set_state("running", stage, progress=value / 100.0)

    def _calibration_finished(self, result: CalibrationResult) -> None:
        self.result = result
        self.job_status.set_state(
            "succeeded",
            "Calibration complete. Review the selected candidate, then Apply.",
            progress=1.0,
        )
        candidate_blocker = QSignalBlocker(self.candidate_table)
        self._populate_candidates()
        self.candidate_table.selectRow(0)
        del candidate_blocker
        self._show_candidate(result.selected_candidate)
        self._set_running(False)
        self.manual_group.setChecked(True)

    def _calibration_failed(self, exc: Exception) -> None:
        if isinstance(exc, CalibrationCancelledError):
            self.job_status.set_state("cancelled", "Calibration cancelled.", progress=0.0)
        else:
            self.job_status.set_state(
                "failed",
                "Calibration failed. Adjust the inputs and try again.",
                progress=0.0,
            )
            QMessageBox.warning(self, "Geometry Calibration", str(exc))
        self._set_running(False)

    def _cleanup_calibration(self) -> None:
        self._cal_worker = None
        if self._cal_thread is not None:
            self._cal_thread.deleteLater()
        self._cal_thread = None
        if self._close_when_idle and self._load_thread is None:
            QTimer.singleShot(0, self.close)

    def _populate_candidates(self) -> None:
        if self.result is None:
            return
        self.candidate_table.setRowCount(len(self.result.candidates))
        for row, candidate in enumerate(self.result.candidates):
            values = (
                self.view_model.standard_display_name(candidate.standard_key),
                f"{candidate.distance_mm:.2f} mm",
                f"{candidate.center_x_px:.1f}, {candidate.center_y_px:.1f}",
                str(candidate.matched_ring_count),
                f"{candidate.rms_residual_px:.2f} px",
                candidate.confidence,
            )
            for column, value in enumerate(values):
                self.candidate_table.setItem(row, column, QTableWidgetItem(value))

    def _candidate_selected(self) -> None:
        if self.result is None:
            return
        rows = self.candidate_table.selectionModel().selectedRows()
        if not rows:
            return
        candidate = self.view_model.select_candidate(rows[0].row())
        self._show_candidate(candidate)
