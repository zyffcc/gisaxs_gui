"""Choose a record and run its replay on a worker, with cancellation and a comparison notice."""

import re
from pathlib import Path
from string import Formatter
from threading import Event

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDialog, QFileDialog, QVBoxLayout

from .app_dialogs import JSON_FILTER, ask_open_json, warn
from .components import JobStatus, show_toast
from .i18n import language_changed, tr, trf
from .i18n.zh_reproduction import REPLAY_ERROR_TEMPLATES
from .task_runner import TaskRunner


def reproduction_error_text(message):
    for template in REPLAY_ERROR_TEMPLATES:
        parts = [re.escape(literal) + (f"(?P<{field}>.+?)" if field else "")
                 for literal, field, _, _ in Formatter().parse(template)]
        match = re.fullmatch("".join(parts), message, flags=re.DOTALL)
        if match:
            values = match.groupdict()
            if "error" in values:
                values["error"] = tr(values["error"])
            text = trf(template, **values)
            if template == "Missing referenced file: {path}":
                text += "\n" + tr("Place a copy with its original filename beside the record and try again.")
            return text
    return tr(message)


_error_text = reproduction_error_text


class ReproductionProgress(QDialog):
    def __init__(self, parent, record, *, title="Reproduce from Record", operation="Reproducing {name}"):
        super().__init__(parent)
        self._title, self._operation = title, operation
        self.record_name = Path(record).name
        self.cancelled = Event()
        self.result = None
        self.error = ""
        self.runner = TaskRunner(self)
        layout = QVBoxLayout(self)
        self.status = JobStatus(self)
        self.status.set_actions_visible(pause=False, details=False)
        self.status.message_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.setMinimumWidth(380)
        self.status.cancelRequested.connect(self.reject)
        language_changed().connect(self.refresh_language)
        self.refresh_language()

    def refresh_language(self):
        self.setWindowTitle(tr(self._title))
        message = (tr("Stopping after the current step…") if self.cancelled.is_set()
                   else trf(self._operation, name=self.record_name))
        self.status.set_state("running", message)
        self.status.cancel_button.setEnabled(not self.cancelled.is_set())

    def reject(self):
        # Escape, Cancel and the title-bar close button all wait for the worker to stop.
        if self.runner.is_busy():
            self.cancelled.set()
            self.refresh_language()
        else:
            super().reject()

    def finished_run(self, result):
        self.result = result
        self.accept()

    def failed_run(self, message, _traceback):
        self.error = message
        super().reject()


def choose_record(parent, reproduce):
    record = ask_open_json(parent, tr("Reproduce from Record"), file_filter=tr(JSON_FILTER))
    if not record:
        return
    folder = QFileDialog.getExistingDirectory(
        parent, tr("Choose a Folder for Reproduced Results"), str(Path(record).parent))
    if not folder:
        return
    dialog = ReproductionProgress(parent, record)
    dialog.runner.submit("reproduce", lambda: reproduce(record, folder, should_cancel=dialog.cancelled.is_set),
                         on_done=dialog.finished_run, on_error=dialog.failed_run)
    dialog.exec_()
    dialog.deleteLater()
    if dialog.result is None:
        if dialog.cancelled.is_set():
            show_toast(parent, tr("Reproduction cancelled."))
        else:
            warn(parent, tr("Reproduce from Record"), _error_text(dialog.error))
        return
    result = dialog.result
    identical = result["status"] == "identical"
    text = ("The reproduced values are bitwise identical." if identical
            else "The reproduced values differ. See comparison.json for details.")
    show_toast(parent, tr(text), level="ok" if identical else "warning",
               action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(result["output_dir"])))))
