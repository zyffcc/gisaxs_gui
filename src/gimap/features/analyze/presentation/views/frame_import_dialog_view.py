"""Frame import choices: all, a range, or particular frame numbers. Behaviour is frame_import_dialog.py."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox, QVBoxLayout,
)


class FrameImportDialogView:
    def setup_import_view(self):
        self.setWindowTitle("Import Frames")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        self.explanation = QLabel(
            "Frames are listed without loading every image. Only the frame you open is read. "
            "Batch Export and Build Map process the imported frames when you start them.", self)
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        form = QFormLayout()
        self.source_combo = QComboBox(self)
        self.source_combo.setObjectName("frameImportSource")
        form.addRow("Source", self.source_combo)
        self.import_mode_combo = QComboBox(self)
        self.import_mode_combo.setObjectName("frameImportMode")
        for text, key in (("All Frames", "all"), ("Frame Range", "range"), ("Specific Frames", "specific"),
                          ("Keep Current Selection", "current")):
            self.import_mode_combo.addItem(text, key)
        form.addRow("Import", self.import_mode_combo)
        self.range_row = QHBoxLayout()
        self.import_first_spin = QSpinBox(self)
        self.import_first_spin.setObjectName("frameImportFirst")
        self.import_last_spin = QSpinBox(self)
        self.import_last_spin.setObjectName("frameImportLast")
        self.import_step_spin = QSpinBox(self)
        self.import_step_spin.setObjectName("frameImportEvery")
        self.range_row.addWidget(self.import_first_spin)
        self.range_row.addWidget(QLabel("to", self))
        self.range_row.addWidget(self.import_last_spin)
        self.range_row.addWidget(QLabel("every", self))
        self.range_row.addWidget(self.import_step_spin)
        form.addRow("Frames", self.range_row)
        self.import_numbers_edit = QLineEdit(self)
        self.import_numbers_edit.setObjectName("frameImportNumbers")
        self.import_numbers_edit.setPlaceholderText("1, 5, 20")
        self.import_numbers_edit.setMaxLength(4096)
        form.addRow("Frame numbers", self.import_numbers_edit)
        layout.addLayout(form)
        self.import_summary_label = QLabel("", self)
        self.import_summary_label.setWordWrap(True)
        self.import_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.import_summary_label)
        self.import_error_label = QLabel("", self)
        self.import_error_label.setWordWrap(True)
        self.import_error_label.setProperty("gimapRole", "warning")
        layout.addWidget(self.import_error_label)
        self.import_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.import_buttons.button(QDialogButtonBox.Ok).setText("Import")
        layout.addWidget(self.import_buttons)
