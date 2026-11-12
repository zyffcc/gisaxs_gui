"""Static layout for reviewing text curve tables before importing them."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QPlainTextEdit, QPushButton, QSplitter,
    QTableWidget, QVBoxLayout, QWidget,
)

from src.gimap.app.presentation.components.feedback import JobStatus
from src.gimap.app.presentation.i18n import DATA_HEADERS


class CurveImportView:
    def setup_curve_import_view(self):
        self.setWindowTitle("Import Curve Tables")
        self.resize(940, 720)
        self.setMinimumSize(720, 580)
        root = QVBoxLayout(self)
        self.explanation = QLabel(
            "Review the columns before importing. The preview reads only the beginning of each file; "
            "the complete curves are read when needed.", self)
        self.explanation.setWordWrap(True)
        root.addWidget(self.explanation)

        self.inspect_job = JobStatus(self)
        self.inspect_job.set_actions_visible(pause=False, details=False)
        root.addWidget(self.inspect_job)
        splitter = QSplitter(Qt.Horizontal, self)
        self.file_list = QListWidget(splitter)
        self.file_list.setObjectName("curveImportFiles")
        self.file_list.setMinimumWidth(180)
        self.file_list.setWordWrap(True)
        detail = QWidget(splitter)
        column = QVBoxLayout(detail)
        column.setContentsMargins(8, 0, 0, 0)
        self.file_path_label = QLabel("", detail)
        self.file_path_label.setTextFormat(Qt.PlainText)
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.file_path_label)
        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("Column layout", detail))
        self.layout_combo = QComboBox(detail)
        self.layout_combo.setObjectName("curveImportLayout")
        self.layout_combo.addItem("Single curve: q, I, optional σ", "single")
        self.layout_combo.addItem("Shared q: q, I₁, I₂, …", "shared_q")
        layout_row.addWidget(self.layout_combo, 1)
        self.apply_all_button = QPushButton("Apply layout to all", detail)
        layout_row.addWidget(self.apply_all_button)
        column.addLayout(layout_row)
        self.layout_note = QLabel("", detail)
        self.layout_note.setWordWrap(True)
        self.layout_note.setTextFormat(Qt.PlainText)
        self.layout_note.setProperty("gimapRole", "warning")
        column.addWidget(self.layout_note)
        self.preview_label = QLabel("First 10 data rows · column roles are shown in the headings", detail)
        self.preview_label.setWordWrap(True)
        column.addWidget(self.preview_label)
        self.preview_table = QTableWidget(detail)
        self.preview_table.setObjectName("curveImportPreview")
        self.preview_table.setProperty(DATA_HEADERS, True)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.verticalHeader().setDefaultSectionSize(24)
        column.addWidget(self.preview_table, 1)
        self.unit_note = QLabel("", detail)
        self.unit_note.setWordWrap(True)
        column.addWidget(self.unit_note)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        form = QFormLayout()
        self.mode_combo = QComboBox(self)
        self.mode_combo.setObjectName("curveImportMode")
        self.mode_combo.addItem("Independent samples (fit each curve)", True)
        self.mode_combo.addItem("Time series (file and column order)", False)
        form.addRow("Use curves as", self.mode_combo)
        self.q_unit_combo = QComboBox(self)
        self.q_unit_combo.setObjectName("curveImportUnit")
        self.q_unit_combo.addItem("Å⁻¹", "A^-1")
        self.q_unit_combo.addItem("nm⁻¹", "nm^-1")
        form.addRow("q unit when absent from the header", self.q_unit_combo)
        root.addLayout(form)
        self.order_note = QLabel(
            "Files keep their selected order; shared-q curves keep their left-to-right column order. "
            "A unit written in a file header takes priority.", self)
        self.order_note.setWordWrap(True)
        root.addWidget(self.order_note)
        self.error_details = QPlainTextEdit(self)
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(86)
        self.error_details.setProperty("gimapRole", "warning")
        self.error_details.hide()
        root.addWidget(self.error_details)
        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.import_button = self.buttons.button(QDialogButtonBox.Ok)
        self.import_button.setEnabled(False)
        root.addWidget(self.buttons)
