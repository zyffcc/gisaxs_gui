"""Persistence behavior for Calibration."""

from __future__ import annotations

import logging


from PyQt5.QtCore import QSignalBlocker, Qt

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf


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
                tr("Apply Geometry"),
                trf(
                    "This calibration moves the beam centre or distance of the saved instrument "
                    "profile \u201c{name}\u201d significantly. Overwrite the profile?",
                    name=changed.name,
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.view_model.apply_result()
        self._sync_main_window_geometry()
        self.calibrationApplied.emit(self.result)
        # One confirmation: a toast that never blocks (no message box after the overwrite question).
        show_toast(self._toast_host(), self._applied_text(), level="ok")

    def _applied_text(self) -> str:
        candidate = self.result.selected_candidate
        text = trf(
            "Geometry calibration applied: centre ({x}, {y}) px, distance {distance} mm.",
            x=f"{candidate.center_x_px:.2f}",
            y=f"{candidate.center_y_px:.2f}",
            distance=f"{candidate.distance_mm:.2f}",
        )
        profile = self.view_model.last_profile
        if profile is not None:
            text += " " + trf(
                "Saved as instrument profile \u201c{name}\u201d; Analyze uses it automatically "
                "for frames from this detector.",
                name=profile.name,
            )
        return text

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
            except Exception as exc:
                LOGGER.exception("Failed to export calibration")
                QMessageBox.warning(self, "Export Calibration", str(exc))
                return
            # The name only: the full path is one click away (Open Folder in the toast).
            text = trf("Calibration exported: {name}", name=self.view_model.source_name(path))
            self.stage_label.setText(text)
            self._show_saved_toast(text, path)

    def import_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Calibration", self._start_folder(), "JSON Files (*.json)"
        )
        if path:
            self.import_result_from(path)

    def import_result_from(self, path: str) -> None:
        """Read an exported calibration (Import Calibration…, or a JSON file dropped on the window)."""
        try:
            previous_image = self.image
            self.result = self.view_model.import_result(path)
            self._remember_fitted_geometry()
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
            # Manual mode starts only from 'Manual refine'.
            self.manual_group.setChecked(False)
            self._show_candidate(self.result.selected_candidate)
            self.stage_label.setText(
                trf("Imported calibration from {name}", name=self.view_model.source_name(path))
            )
            self._set_running(False)
        except Exception as exc:
            LOGGER.exception("Failed to import calibration")
            QMessageBox.warning(self, "Import Calibration", str(exc))

    def dispose_when_idle(self) -> None:
        """Delete this window once nothing runs in it any more.

        For a window made for one use (Analyze opens one per calibration with ``exec_()``):
        call it after ``exec_()`` returned, never before, since a close during ``exec_()`` would
        then delete the window under its own event loop. With no image read or run pending the
        window closes and is deleted now. A close that already waits for a thread
        (``_close_when_idle``) closes it when that thread ends, and the window is deleted then.
        """
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        if not self._close_when_idle:
            self.close()

    def reject(self) -> None:
        """Esc goes through ``closeEvent``, which asks before cancelling a running calibration.

        QDialog's own reject() would hide the window without a close event. ``closeEvent`` does
        not call ``QDialog.closeEvent``, so this cannot recurse.
        """
        self.close()

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
        # The Tools menu reuses this window: a deferred close must not carry over to the next
        # time it is shown.
        self._close_when_idle = False
        event.accept()
