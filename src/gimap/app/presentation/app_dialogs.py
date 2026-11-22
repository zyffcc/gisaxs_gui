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

from .question_box import ACCEPT, ACTION, DESTRUCTIVE, REJECT, ask_choice
from .i18n import tr, trf

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


def _choose(parent: Optional[QWidget], title: str, text: str, buttons, default: int, escape: int,
            informative: str = "") -> int:
    """A question with the buttons ``(text, role)`` given; returns the index of the one chosen (``escape`` for
    Esc or the window's close button). A ``QuestionBox``, not a ``QMessageBox`` (which kills the process on the
    offscreen platform: ``question_box.py``)."""
    return ask_choice(parent, title, text, buttons, default=default, escape=escape, informative=informative)


def ask_quit_with_jobs(parent: Optional[QWidget], jobs: str) -> bool:
    """While ``jobs`` (their names, one phrase) run: stop them and quit?"""
    answer = QMessageBox.question(parent, tr("Quit GIMaP"), trf("Still running: {jobs}. Stop them and quit?", jobs=jobs),
                                  QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
    return answer == QMessageBox.Yes


def ask_save_changes(parent: Optional[QWidget], name: str) -> str:
    """``"save"``, ``"discard"`` or ``"cancel"``: what to do with changes not saved in the project ``name``
    (``""``: work that is in no project yet)."""
    text = (trf("Save the changes to the project {name}?", name=name) if name
            else tr("Save the current work as a project?"))
    answer = _choose(parent, tr("Unsaved Changes"), text,
                     ((tr("Save"), ACCEPT), (tr("Don't Save"), DESTRUCTIVE),
                      (tr("Cancel"), REJECT)), default=0, escape=2,
                     informative=tr("Changes that are not saved are lost."))
    return ("save", "discard", "cancel")[answer]


def ask_restore_session(parent: Optional[QWidget], when: str) -> bool:
    """After GIMaP was not closed properly: restore the work it autosaved at ``when``?"""
    answer = _choose(parent, tr("Restore the Session"),
                     trf("GIMaP was not closed properly. Restore the work autosaved at {time}?", time=when),
                     ((tr("Restore"), ACCEPT), (tr("Discard"), DESTRUCTIVE)),
                     default=0, escape=1)
    return answer == 0


def ask_already_running(parent: Optional[QWidget]) -> str:
    """A second start: ``"switch"`` to the GIMaP that runs, ``"read_only"`` (start anyway, settings not saved)
    or ``"cancel"``."""
    answer = _choose(parent, "GIMaP", tr("GIMaP is already running."),
                     ((tr("Switch to It"), ACCEPT),
                      (tr("Start Anyway (Read-only Settings)"), ACTION),
                      (tr("Cancel"), REJECT)), default=0, escape=2,
                     informative=tr("Two windows on one user folder would overwrite each other's settings and "
                                    "session. A second one started anyway keeps its settings in memory only."))
    return ("switch", "read_only", "cancel")[answer]


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


__all__ = ["JSON_FILTER", "ask_already_running", "ask_open_json", "ask_quit_with_jobs", "ask_restore_session", "ask_save_changes",
           "ask_save_json", "close_details", "inform", "show_details", "warn"]
