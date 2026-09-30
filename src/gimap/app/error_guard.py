"""Unexpected errors do not close GIMaP: they are written to a log and shown, and the work goes on.

PyQt5 ends the whole process (``qFatal``) when a Python exception escapes a slot and
``sys.excepthook`` is the default one — one bug in a button handler would lose every unsaved
result. ``ErrorGuard.install`` replaces ``sys.excepthook`` (the GUI thread) and
``threading.excepthook`` (worker threads) with a handler that

* appends the traceback, the time and the version to ``<user data>/logs/errors.log``;
* shows one non-modal window (“Something went wrong … GIMaP keeps running”) with the details, a
  button to copy them and one to open the log folder — errors of a burst are counted in the same
  window instead of opening one each (a ``QDialog``: a ``QMessageBox`` cannot be shown on the
  offscreen test platform);
* lets ``KeyboardInterrupt`` / ``SystemExit`` through to the default handler.

Worker-thread errors reach the GUI thread through a queued signal.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QObject, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

LOG_NAME = "errors.log"
MAX_LOG_BYTES = 2_000_000


class ErrorGuard(QObject):
    caught = pyqtSignal(str, str)
    """``(summary, details)`` of an error, delivered on the GUI thread."""

    def __init__(self, log_dir: Path, *, version: str = "", notify: Optional[Callable[[str, str], None]] = None,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self.log_dir = Path(log_dir)
        self.version = version
        self._notify = notify or self._show
        self._box: Optional[ErrorDialog] = None
        self._count = 0
        self._previous_hook = sys.excepthook
        self._previous_thread_hook = getattr(threading, "excepthook", None)
        self.caught.connect(self._deliver, Qt.QueuedConnection)

    @property
    def log_path(self) -> Path:
        return self.log_dir / LOG_NAME

    def install(self) -> "ErrorGuard":
        sys.excepthook = self._handle
        if self._previous_thread_hook is not None:
            threading.excepthook = lambda args: self._handle(args.exc_type, args.exc_value, args.exc_traceback,
                                                             where=f"thread {args.thread.name if args.thread else '?'}")
        return self

    def uninstall(self) -> None:
        sys.excepthook = self._previous_hook
        if self._previous_thread_hook is not None:
            threading.excepthook = self._previous_thread_hook

    # -- handling ------------------------------------------------------------------------

    def _handle(self, exc_type, exc, tb, *, where: str = "") -> None:
        if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            self._previous_hook(exc_type, exc, tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        summary = f"{getattr(exc_type, '__name__', 'Error')}: {exc}"
        self._write(details, where)
        try:
            self.caught.emit(summary, details)
        except RuntimeError:  # the guard is being deleted at exit
            sys.__excepthook__(exc_type, exc, tb)

    def _write(self, details: str, where: str) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists() and self.log_path.stat().st_size > MAX_LOG_BYTES:
                self.log_path.replace(self.log_path.with_suffix(".old.log"))
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            head = f"==== {stamp} GIMaP {self.version} {where}".rstrip()
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{head}\n{details}\n")
        except OSError:
            pass

    def _deliver(self, summary: str, details: str) -> None:
        self._notify(summary, details)

    def _show(self, summary: str, details: str) -> None:
        from src.gimap.app.presentation.i18n import tr

        self._count += 1
        if self._box is not None and self._box.isVisible():
            self._box.headline.setText(tr("{count} unexpected errors — GIMaP keeps running.").format(count=self._count))
            self._box.details.setPlainText(details)
            return
        self._count = 1
        box = ErrorDialog(self.log_dir, QApplication.activeWindow())
        box.headline.setText(tr("An unexpected error — GIMaP keeps running."))
        box.message.setText(tr(
            "What you were doing may not have finished; your data and the other results are unchanged.\n\n{summary}\n\n"
            "The details are saved in {log}."
        ).format(summary=summary, log=self.log_path))
        box.details.setPlainText(details)
        box.destroyed.connect(self._forget_box)
        self._box = box
        box.show()

    def _forget_box(self, *_args) -> None:
        self._box = None


class ErrorDialog(QDialog):
    """Non-modal: the headline, what it means, the traceback behind “Show Details”, copy and log folder."""

    def __init__(self, log_dir: Path, parent: Optional[QWidget] = None):
        from src.gimap.app.presentation.i18n import tr
        from src.gimap.app.presentation.theme import set_role

        super().__init__(parent)
        self.setObjectName("gimapErrorDialog")
        self.setWindowTitle(tr("Something went wrong"))
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        icon = QLabel(self)
        icon.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxWarning).pixmap(32, 32))
        icon.setAlignment(Qt.AlignTop)
        top.addWidget(icon)
        text = QVBoxLayout()
        self.headline = QLabel(self)
        set_role(self.headline, "strong")
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text.addWidget(self.headline)
        text.addWidget(self.message)
        top.addLayout(text, 1)
        layout.addLayout(top)
        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.details.setMinimumHeight(160)
        self.details.hide()
        layout.addWidget(self.details, 1)
        buttons = QHBoxLayout()
        self.details_button = QPushButton(tr("Show Details"), self)
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.details.setVisible)
        copy = QPushButton(tr("Copy Details"), self)
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.details.toPlainText()))
        folder = QPushButton(tr("Open Log Folder"), self)
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir))))
        close = QPushButton(tr("Close"), self)
        close.setDefault(True)
        close.clicked.connect(self.close)
        for button in (self.details_button, copy, folder):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)


__all__ = ["ErrorDialog", "ErrorGuard", "LOG_NAME"]
