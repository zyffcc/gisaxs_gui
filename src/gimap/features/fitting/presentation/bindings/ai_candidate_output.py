"""Ai Candidate Output for fitting presentation."""

from __future__ import annotations


import datetime

from pathlib import Path

import numpy as np

from PyQt5.QtCore import QUrl

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)


from PyQt5.QtGui import QDesktopServices


class AiCandidateOutputMixin:
    """Own ai candidate output behavior."""

    def _export_ai_prediction_output(self) -> None:
        out_dir = Path(getattr(self, "_ai_output_dir", "") or self._ai_current_prediction_dir())
        if not self.fitting_view_model.storage.has_ai_output(out_dir):
            QMessageBox.information(
                self.main_window or self.ui,
                "Export AI Output",
                "No AI prediction output is available yet. Run a prediction first.",
            )
            return

        settings = self._ai_fitting_settings()
        start_dir = str(settings.get("last_export_parent") or Path.cwd())
        parent = QFileDialog.getExistingDirectory(
            self.main_window or self.ui,
            "Choose Folder for Exported AI Output",
            start_dir,
        )
        if not parent:
            return

        parent_path = Path(parent)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            dest = self.fitting_view_model.storage.export_ai_output(
                out_dir,
                parent_path,
                timestamp,
            )
        except Exception as exc:
            QMessageBox.warning(
                self.main_window or self.ui,
                "Export AI Output",
                f"Failed to export AI prediction output:\n{exc}",
            )
            return

        self._save_ai_fitting_settings(last_export_parent=str(parent_path))
        self._set_ai_workspace_status(f"Exported AI output to: {dest}", None)
        self._append_ai_log(f"Exported AI output to: {dest}")
        QMessageBox.information(
            self.main_window or self.ui,
            "Export AI Output",
            f"AI prediction output exported to:\n{dest}",
        )

    def _show_ai_candidate_table(self, output_dir: Path | None = None, rows=None) -> None:
        output_dir = Path(output_dir or getattr(self, "_ai_output_dir", "") or "")
        if rows is None:
            rows = self.fitting_view_model.load_candidate_results(output_dir)
        if not rows:
            self._set_ai_workspace_status("AI fitting produced no candidates.", None)
            return
        rows = list(
            self.fitting_view_model.review_candidates(
                rows,
                self._ai_run_settings().get("constraint_set"),
            )
        )
        self._ai_candidate_rows = rows

        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("AI Fitting Candidates")
        dialog.resize(900, 520)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(rows), 8, dialog)
        table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Combination",
                "Score Prob.",
                "Posterior",
                "logRMSE",
                "Chi2",
                "Constraints",
                "Source",
            ]
        )
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row_idx, row in enumerate(rows):
            values = [
                row.get("rank", row_idx + 1),
                row.get("combination", ""),
                f"{float(row.get('score_weighted_probability', 0.0)) * 100:.2f}%",
                f"{float(row.get('posterior_frequency', 0.0)) * 100:.2f}%",
                f"{float(row.get('best_log_rmse', np.nan)):.5g}",
                f"{float(row.get('best_chi2_weighted', np.nan)):.5g}",
                "Valid"
                if not row.get("constraint_violations")
                else "; ".join(row["constraint_violations"]),
                row.get("best_source", ""),
            ]
            for col, value in enumerate(values):
                table.setItem(row_idx, col, QTableWidgetItem(str(value)))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, 8):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)

        preview_hint = QLabel(
            "Selecting a row automatically loads its parameters and refreshes the fitting plot.",
            dialog,
        )
        preview_hint.setWordWrap(True)
        layout.addWidget(preview_hint)

        button_row = QHBoxLayout()
        load_btn = QPushButton("Load Selected Params", dialog)
        open_btn = QPushButton("Open Output Folder", dialog)
        close_btn = QPushButton("Close", dialog)
        load_btn.clicked.connect(
            lambda: self._load_selected_ai_candidate_from_table(table, rows, dialog)
        )
        table.doubleClicked.connect(
            lambda _index: self._load_selected_ai_candidate_from_table(table, rows, dialog)
        )
        table.currentCellChanged.connect(
            lambda current_row, _current_column, _previous_row, _previous_column: (
                self._preview_ai_candidate_from_table(current_row, rows)
            )
        )
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
        )
        close_btn.clicked.connect(dialog.close)
        button_row.addWidget(load_btn)
        button_row.addWidget(open_btn)
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        self._ai_results_dialog = dialog
        table.selectRow(0)
        dialog.show()

    def _preview_ai_candidate_from_table(self, selected: int, rows: list) -> None:
        """Load and render the currently selected candidate without closing the table."""
        if selected < 0 or selected >= len(rows):
            return
        self._load_ai_candidate_params(rows[selected], refresh_plot=True)

    def _load_selected_ai_candidate_from_table(
        self, table: QTableWidget, rows: list, dialog: QDialog | None = None
    ) -> None:
        selected = table.currentRow()
        if selected < 0 or selected >= len(rows):
            return
        if self._load_ai_candidate_params(rows[selected], refresh_plot=True):
            if dialog is not None:
                dialog.accept()
