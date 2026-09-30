"""Session cache of the in-situ series: records, CSV export, cache folder."""

from __future__ import annotations

import datetime
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class InsituPersistencePreviewMixin:
    """Own the persistent in-situ session records."""

    def _insitu_cache_dir(self) -> Path:
        return self.fitting_view_model.storage.insitu_cache_directory()

    def _insitu_session_cache_path(self) -> Path:
        return self.fitting_view_model.storage.insitu_session_path()

    def _reset_insitu_session_cache(self):
        self._insitu_workflow_processed_count = 0
        self._insitu_workflow_failed_count = 0
        self._insitu_workflow_results = []
        self.fitting_view_model.storage.reset_insitu_records()
        self._refresh_insitu_workflow_status()

    def _append_insitu_session_cache(self, record: dict):
        try:
            self.fitting_view_model.storage.append_insitu_record(record)
        except Exception as exc:
            self._log_insitu_workflow(f"Session cache write failed: {exc}", "ERROR")

    def _load_insitu_session_records(self) -> list[dict]:
        if self._insitu_workflow_results:
            rows = list(self._insitu_workflow_results)
            current = getattr(self, "_insitu_workflow_current_record", None)
            if isinstance(current, dict):
                rows.append(current.copy())
            return rows
        try:
            rows = self.fitting_view_model.storage.load_insitu_records()
        except Exception as exc:
            rows = []
            self._log_insitu_workflow(f"Session cache read failed: {exc}", "ERROR")
        current = getattr(self, "_insitu_workflow_current_record", None)
        if isinstance(current, dict):
            rows.append(current.copy())
        return rows

    def _export_insitu_records_to_csv(self, path: Path, rows: list[dict]):
        try:
            if not rows:
                return
            self.fitting_view_model.storage.export_insitu_records(path, rows)
        except Exception as exc:
            self._log_insitu_workflow(f"CSV export failed: {exc}", "ERROR")

    def _export_insitu_workflow_results(self):
        rows = self._load_insitu_session_records()
        if not rows:
            QMessageBox.information(
                self._insitu_workflow_parent_widget(),
                "Export Results",
                "No cached in-situ results to export.",
            )
            return
        default_name = f"in_situ_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self._insitu_workflow_parent_widget(),
            "Export In-situ Results",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filename:
            return
        self._export_insitu_records_to_csv(Path(filename), rows)
        self._log_insitu_workflow(f"Exported results to {filename}", "SUCCESS")

    def _clear_insitu_session_cache(self):
        if self._insitu_workflow_busy:
            QMessageBox.information(
                self._insitu_workflow_parent_widget(),
                "Clear Session Cache",
                "Stop the workflow before clearing the cache.",
            )
            return
        self._reset_insitu_session_cache()
        self._insitu_workflow_last_fit_params = None
        self._insitu_workflow_last_fit_status = "-"
        self._insitu_workflow_last_chi_square = None
        self._reset_insitu_heatmap_data()
        self._log_insitu_workflow("Session cache cleared")

    def _open_insitu_cache_folder(self):
        try:
            directory = self.fitting_view_model.storage.ensure_insitu_cache_directory()
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        except Exception as exc:
            self._log_insitu_workflow(f"Open cache folder failed: {exc}", "ERROR")
