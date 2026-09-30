"""Persistence behavior for Calibration."""

from __future__ import annotations

import logging


from PyQt5.QtCore import QSignalBlocker

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


LOGGER = logging.getLogger(__name__)


class PersistenceMixin:
    """Own persistence presentation behavior."""

    def apply_result(self) -> None:
        if self.result is None:
            return
        self._commit_manual_values()
        changed = self.view_model.significantly_changed_profile()
        if changed is not None:
            answer = QMessageBox.question(
                self,
                "Apply Geometry",
                f"This calibration moves the beam centre or distance of the saved instrument "
                f"profile \u201c{changed.name}\u201d significantly. Overwrite the profile?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.view_model.apply_result()
        self._sync_main_window_geometry()
        self.calibrationApplied.emit(self.result)
        message = "The calibrated geometry was applied."
        profile = self.view_model.last_profile
        if profile is not None:
            message = (
                f"Saved as instrument profile \u201c{profile.name}\u201d; Analyze uses it "
                "automatically for frames from this detector."
            )
        QMessageBox.information(self, "Geometry Calibration", message)

    def export_result(self) -> None:
        if self.result is None:
            return
        self._commit_manual_values()
        default = self.view_model.default_export_path(self.result.source_image)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Calibration", default, "JSON Files (*.json)"
        )
        if path:
            try:
                self.view_model.export_result(path)
                self.stage_label.setText(f"Calibration exported to {path}")
            except Exception as exc:
                LOGGER.exception("Failed to export calibration")
                QMessageBox.warning(self, "Export Calibration", str(exc))

    def import_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Calibration", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            previous_image = self.image
            self.result = self.view_model.import_result(path)
            self.path_edit.setText(self.result.source_image)
            if self.image is not previous_image:
                self._preview_cache.clear()
            self.energy_spin.setValue(self.result.energy_kev)
            self.pixel_x_spin.setValue(self.result.pixel_size_x_m * 1e6)
            self.pixel_y_spin.setValue(self.result.pixel_size_y_m * 1e6)
            candidate_blocker = QSignalBlocker(self.candidate_table)
            self._populate_candidates()
            self.candidate_table.selectRow(0)
            del candidate_blocker
            self._show_candidate(self.result.selected_candidate)
            self.stage_label.setText(
                f"Imported calibration from {self.view_model.source_name(path)}"
            )
            self._set_running(False)
            self.manual_group.setChecked(True)
        except Exception as exc:
            LOGGER.exception("Failed to import calibration")
            QMessageBox.warning(self, "Import Calibration", str(exc))

    def closeEvent(self, event) -> None:
        if self._cal_thread is not None and self._cal_thread.isRunning():
            answer = QMessageBox.question(
                self,
                "Calibration Running",
                "Cancel calibration and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel_calibration()
            self._close_when_idle = True
            self.hide()
            event.ignore()
            return
        if self._load_thread is not None and self._load_thread.isRunning():
            self._close_when_idle = True
            self.hide()
            event.ignore()
            return
        event.accept()
