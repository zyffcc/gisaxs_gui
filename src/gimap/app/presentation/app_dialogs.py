"""Small standard dialogs used by the application shell (file choice, notices)."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget

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


__all__ = ["JSON_FILTER", "ask_open_json", "ask_save_json", "inform", "warn"]
