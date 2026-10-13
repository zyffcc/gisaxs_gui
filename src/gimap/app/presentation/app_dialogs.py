"""Small standard dialogs used by the application shell (file choice, notices, a list of details)."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .i18n import tr

JSON_FILTER = "JSON Files (*.json);;All Files (*)"


def ask_open_json(parent: Optional[QWidget], title: str, folder: str = "", file_filter: str = JSON_FILTER) -> str:
    path, _ = QFileDialog.getOpenFileName(parent, title, folder, file_filter)
    return path


def ask_save_json(parent: Optional[QWidget], title: str, suggested: str, file_filter: str = JSON_FILTER) -> str:
    path, _ = QFileDialog.getSaveFileName(parent, title, suggested, file_filter)
    return path


def inform(parent: Optional[QWidget], title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


def warn(parent: Optional[QWidget], title: str, text: str) -> None:
    QMessageBox.warning(parent, title, text)


def show_details(parent: Optional[QWidget], title: str, summary: str, details: str) -> QDialog:
    """A long list behind a short notice (the files a project no longer finds): ``summary`` in a sentence or
    two, ``details`` as text that can be selected and copied. The window does not block work and deletes
    itself when closed; the texts come in the interface language already. One at a time: the list shown
    before (of another project) is closed (``close_details``)."""
    close_details(parent)
    dialog = QDialog(parent)
    dialog.setObjectName("detailsDialog")
    dialog.setWindowTitle(title)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    layout = QVBoxLayout(dialog)
    label = QLabel(summary, dialog)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(label)
    text = QPlainTextEdit(details, dialog)
    text.setObjectName("detailsText")
    text.setReadOnly(True)  # wrapped: a long folder is read whole, without a scroll bar under the list
    layout.addWidget(text, 1)
    buttons = QDialogButtonBox(dialog)
    copy = buttons.addButton(tr("Copy"), QDialogButtonBox.ActionRole)
    copy.setObjectName("detailsCopy")
    copy.clicked.connect(lambda: QApplication.clipboard().setText(details))
    close = buttons.addButton(tr("Close"), QDialogButtonBox.RejectRole)
    close.setDefault(True)
    buttons.rejected.connect(dialog.close)
    layout.addWidget(buttons)
    dialog.resize(680, 380)
    dialog.show()
    return dialog


def close_details(parent: Optional[QWidget]) -> int:
    """Close the details windows of ``parent`` (what they list is no longer what is open); how many went."""
    if parent is None:
        return 0
    shown = parent.findChildren(QDialog, "detailsDialog", Qt.FindDirectChildrenOnly)
    for dialog in shown:
        dialog.close()
    return len(shown)


__all__ = ["JSON_FILTER", "ask_open_json", "ask_save_json", "close_details", "inform", "show_details", "warn"]
