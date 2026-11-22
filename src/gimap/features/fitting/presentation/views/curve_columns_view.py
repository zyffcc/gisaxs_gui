"""Static layout of Choose Columns (behaviour in ``curve_columns_dialog.py``): the table's first rows as Import Curves
shows them, and which column is q, I and σ, and the unit of q."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QTableWidget,
    QVBoxLayout,
)


class CurveColumnsView:
    def setup_curve_columns_view(self, dialog) -> None:
        dialog.setObjectName("fitCurveColumnsDialog")
        dialog.setWindowTitle("Choose Columns")
        dialog.resize(640, 460)
        layout = QVBoxLayout(dialog)
        self.columns_file_label = QLabel("", dialog)
        self.columns_file_label.setObjectName("fitColumnsFile")
        self.columns_file_label.setWordWrap(True)
        self.columns_note = QLabel("", dialog)
        self.columns_note.setObjectName("fitColumnsNote")
        self.columns_note.setWordWrap(True)
        self.columns_note.setProperty("gimapRole", "muted")
        self.columns_preview = QTableWidget(dialog)
        self.columns_preview.setObjectName("fitColumnsPreview")
        self.columns_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.columns_preview.setSelectionMode(QAbstractItemView.NoSelection)
        form = QFormLayout()
        self.q_column_combo = QComboBox(dialog)
        self.q_column_combo.setObjectName("fitColumnQ")
        self.intensity_column_combo = QComboBox(dialog)
        self.intensity_column_combo.setObjectName("fitColumnI")
        self.sigma_column_combo = QComboBox(dialog)
        self.sigma_column_combo.setObjectName("fitColumnSigma")
        self.columns_unit_combo = QComboBox(dialog)
        self.columns_unit_combo.setObjectName("fitColumnsUnit")
        self.columns_unit_combo.addItem("Å⁻¹", "angstrom")
        self.columns_unit_combo.addItem("nm⁻¹", "nm")
        form.addRow("q", self.q_column_combo)
        form.addRow("I", self.intensity_column_combo)
        form.addRow("σ", self.sigma_column_combo)
        form.addRow("Unit of q", self.columns_unit_combo)
        self.columns_problem = QLabel("", dialog)
        self.columns_problem.setObjectName("fitColumnsProblem")
        self.columns_problem.setWordWrap(True)
        self.columns_problem.setProperty("gimapRole", "warning")
        self.columns_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        self.columns_buttons.button(QDialogButtonBox.Ok).setText("Open")
        for widget in (self.columns_file_label, self.columns_note, self.columns_preview):
            layout.addWidget(widget)
        layout.addLayout(form)
        layout.addWidget(self.columns_problem)
        layout.addWidget(self.columns_buttons)


__all__ = ["CurveColumnsView"]
