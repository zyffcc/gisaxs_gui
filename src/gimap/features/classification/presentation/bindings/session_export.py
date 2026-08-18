"""Session Export coordination for Classification."""

from __future__ import annotations


import os


from pathlib import Path


from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
)


class SessionExportMixin:
    """Own session export presentation behavior."""

    def _save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Classification Session", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.classification_view_model.save_session(Path(path), self.get_parameters())
            self.log(f"[Session] Saved session to {path}")
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Save Session", str(exc))

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load Classification Session", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.set_parameters(self.classification_view_model.load_session(Path(path)))
            self.log(f"[Session] Loaded session from {path}")
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Load Session", str(exc))

    def _export_results_csv(self) -> None:
        if self.experiment_result is None:
            QMessageBox.information(self.main_window, "Export Results", "No results are available.")
            return
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Export Results", "", "CSV (*.csv)")
        if not path:
            return
        columns = (
            "rank",
            "algorithm",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "status",
            "error",
        )
        rows = tuple(
            (
                rank,
                result.display_name,
                result.metrics_mean.get("accuracy", ""),
                result.metrics_mean.get("balanced_accuracy", ""),
                result.metrics_mean.get("macro_f1", ""),
                result.metrics_mean.get("weighted_f1", ""),
                result.status,
                result.error_message or "",
            )
            for rank, result in enumerate(self.experiment_result.results, start=1)
        )
        self.classification_view_model.export_csv(Path(path), columns, rows)
        self.log(f"[Export] Results CSV written to {path}")

    def _export_predictions_csv(self) -> None:
        if not self.prediction_results:
            QMessageBox.information(
                self.main_window, "Export Predictions", "No prediction results are available."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Predictions", "", "CSV (*.csv)"
        )
        if not path:
            return
        columns = ("file", "predicted_label", "confidence", "decision_score", "status", "message")
        rows = tuple(
            (
                result.file_path,
                result.predicted_label or "",
                result.confidence if result.confidence is not None else "",
                result.decision_score if result.decision_score is not None else "",
                result.status,
                result.message,
            )
            for result in self.prediction_results
        )
        self.classification_view_model.export_csv(Path(path), columns, rows)
        self.log(f"[Export] Prediction CSV written to {path}")

    def _export_selected_file_list(self) -> None:
        selected = [self._sample_by_id(sample_id) for sample_id in self._selected_sample_ids()]
        selected = [sample for sample in selected if sample is not None]
        if not selected:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Selected Files", "", "CSV (*.csv)"
        )
        if not path:
            return
        columns = ("included", "class", "file", "path", "load_status", "qc_status")
        rows = tuple(
            (
                sample.included,
                sample.label,
                sample.file_name,
                sample.file_path,
                sample.load_status,
                sample.qc_status,
            )
            for sample in selected
        )
        self.classification_view_model.export_csv(Path(path), columns, rows)
        self.log(f"[Export] Selected file list written to {path}")

    def _set_selected_included(self, included: bool) -> None:
        selected_ids = set(self._selected_sample_ids())
        for sample in self.samples:
            if sample.sample_id in selected_ids:
                sample.included = included
        self.summary = self.classification_view_model.validate_dataset(self.samples)
        self._mark_results_outdated()
        self._refresh_everything()

    def _remove_selected_samples(self) -> None:
        selected_ids = set(self._selected_sample_ids())
        if not selected_ids:
            return
        self.samples = [sample for sample in self.samples if sample.sample_id not in selected_ids]
        self._mark_results_outdated()
        self._refresh_everything()

    def _open_selected_location(self) -> None:
        sample = self._sample_by_id(self.current_preview_sample_id)
        if sample is None:
            ids = self._selected_sample_ids()
            sample = self._sample_by_id(ids[0]) if ids else None
        if sample is None:
            return
        folder = os.path.dirname(sample.file_path)
        if os.path.isdir(folder):
            os.startfile(folder)

    def _copy_selected_paths(self) -> None:
        selected = [self._sample_by_id(sample_id) for sample_id in self._selected_sample_ids()]
        paths = [sample.file_path for sample in selected if sample is not None]
        if paths:
            QApplication.clipboard().setText("\n".join(paths))
            self.log(f"[Dataset] Copied {len(paths)} path(s).")

    def _move_preview(self, delta: int) -> None:
        loaded = [sample for sample in self.samples if sample.load_status == "loaded"]
        if not loaded:
            return
        current = self._sample_by_id(self.current_preview_sample_id)
        try:
            index = loaded.index(current)
        except ValueError:
            index = 0
        index = max(0, min(len(loaded) - 1, index + delta))
        self.current_preview_sample_id = loaded[index].sample_id
        self._render_sample_preview(loaded[index])
