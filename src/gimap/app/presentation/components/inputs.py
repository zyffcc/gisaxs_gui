"""Generic path input presentation component。"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

from ..styles import apply_design_system


class FilePicker(QWidget):
    """Path editor that emits intent signals without opening file dialogs。"""

    pathChanged = pyqtSignal(str)
    browseRequested = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        placeholder: str = "Choose a file or folder…",
        browse_text: str = "Browse…",
        clearable: bool = True,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.path_edit = QLineEdit(self)
        self.path_edit.setProperty("gimapFilePath", True)
        self.path_edit.setPlaceholderText(placeholder)
        self.browse_button = QPushButton(browse_text, self)
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.setVisible(clearable)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.clear_button)
        self.path_edit.textChanged.connect(self.pathChanged)
        self.browse_button.clicked.connect(self.browseRequested)
        self.clear_button.clicked.connect(self._clear)
        apply_design_system(self)

    def path(self) -> str:
        return self.path_edit.text().strip()

    def set_path(self, path: str) -> None:
        self.path_edit.setText(str(path))

    def set_read_only(self, read_only: bool) -> None:
        self.path_edit.setReadOnly(bool(read_only))

    def _clear(self) -> None:
        self.path_edit.clear()
        self.clearRequested.emit()
