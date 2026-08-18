"""Small application-shell feedback dialogs."""

from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox, QWidget


def show_workspace_unavailable(
    parent: QWidget | None,
    workspace_name: str,
    message: str,
) -> None:
    QMessageBox.warning(parent, workspace_name, message)


__all__ = ["show_workspace_unavailable"]
