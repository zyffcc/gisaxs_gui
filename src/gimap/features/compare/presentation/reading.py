"""Queued, cancellable Compare imports. Only callbacks on the GUI thread touch widgets."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from src.gimap.app.presentation.i18n import tr, trf

from ..application import SeriesData
from .errors import error_text

READING = "Reading curve files: {done} of {total} …"
SCANNING = "Looking for curve files …"
CANCELLED = "Import cancelled. Unfinished series were not added."


class _ReadSignals(QObject):
    progress = pyqtSignal(object, int, int)


class _ReadCancellation:
    """Cancellation surviving QWidget destruction without retaining or reading the widget itself.

    A destroyed callback closing over its QWidget can run while cyclic GC is clearing that same closure.
    Keep only the current pure-Python Event in the callback's receiver; renewed imports still get distinct tokens.
    """

    def __init__(self):
        self.token = Event()

    def renew(self):
        self.token = Event()
        return self.token

    def cancel(self):
        self.token.set()


@dataclass
class _Request:
    folder: Optional[Path] = None
    paths: tuple = ()
    kind: Optional[str] = None
    prefer: Optional[str] = None
    group: Optional[dict] = None
    entry: Optional[dict] = None
    sidecar: Optional[Path] = None


class CompareReadingMixin:
    def _init_reading(self):
        self._reading = 0
        self._read_queue = deque()
        self._read_cancellation = _ReadCancellation()
        self._read_cancel = self._read_cancellation.token
        self._disposed = False
        self._read_message = lambda: tr(SCANNING)
        self._read_progress_value = None
        self._read_signals = _ReadSignals(self)
        self._read_signals.progress.connect(self._read_progress)
        self.import_job.cancelRequested.connect(self.cancel_import)
        self.destroyed.connect(self._read_cancellation.cancel)

    def _enqueue_import(self, **values):
        if self._disposed:
            return
        self._read_queue.append(_Request(**values))
        if not self._reading:
            self._read_cancel = self._read_cancellation.renew()
            self._read_errors = []
            self._reading = 1
            self._timer.stop()
            self.import_job.show()
            self.refresh()
            self._render()
            self._next_import()

    def _next_import(self):
        if not self._read_queue:
            self._reading = 0
            self.import_job.hide()
            self.refresh()
            self._render()
            if self.series and not self.comparison_current():
                self._timer.start()
            else:
                self._timer.stop()
            if self._read_errors:
                self._status(lambda errors=tuple(self._read_errors): self._import_error_text(errors), "error")
            return
        request = self._read_queue.popleft()
        token = self._read_cancel
        if request.folder is not None and request.kind is None:
            self._set_reading_message(lambda: tr(SCANNING))
            service = self.service
            self.tasks.submit("compare-read", lambda: service.folder_kinds(request.folder),
                              on_done=lambda kinds: self._folder_ready(request, token, kinds),
                              on_error=lambda message, _trace: self._import_failed(token, message, request))
        else:
            self._start_import(request, token)

    def _folder_ready(self, request, token, kinds):
        if not self._read_current(token):
            return
        prefer = request.prefer or (request.group or {}).get("kind")
        request.kind = self._choose_kind(request.folder, kinds, prefer)
        # A modal kind chooser can process Cancel / Close events.
        if not self._read_current(token):
            return
        if request.kind is False:
            self._next_import()
            return
        if request.group is not None and request.kind:
            request.group["kind"] = request.kind
        self._start_import(request, token)

    def _start_import(self, request, token):
        service, signals = self.service, self._read_signals
        self._set_reading_message(lambda: tr("Reading curve files …"))

        def progress(done, total):
            if not token.is_set():
                signals.progress.emit(token, done, total)

        def read():
            if token.is_set():
                return None
            if request.entry is not None:
                return _restore_entry(service, request.entry, request.sidecar, progress, token)
            if request.folder is not None:
                return service.series_from_folder(request.folder, kind=request.kind or None,
                                                  progress=progress, cancelled=token.is_set)
            return service.series_from_files(request.paths, progress=progress, cancelled=token.is_set)

        self.tasks.submit("compare-read", read,
                          on_done=lambda data: self._import_ready(token, data),
                          on_error=lambda message, _trace: self._import_failed(token, message, request))

    def _read_current(self, token):
        return not self._disposed and token is self._read_cancel and not token.is_set()

    def _read_progress(self, token, done, total):
        if self._read_current(token):
            self._status(lambda: trf(READING, done=done, total=total))
            self._set_reading_message(lambda: trf(READING, done=done, total=total),
                                      progress=done / total if total else None)

    def _set_reading_message(self, message, progress=None):
        self._read_message, self._read_progress_value = message, progress
        self._refresh_reading_language()

    def _refresh_reading_language(self):
        if self._reading:
            self.import_job.set_state("running", self._read_message(), progress=self._read_progress_value)

    def _import_ready(self, token, data):
        if not self._read_current(token):
            return
        if data is not None:
            self.add_series(data)
        self._next_import()

    def _import_failed(self, token, message, request):
        if not self._read_current(token):
            return
        name = (request.entry or {}).get("name", "")
        self._read_errors.append((name, str(message)))
        self._next_import()

    def _import_error_text(self, errors):
        name, message = errors[0]
        reason = error_text(message)
        if name:
            reason = trf("Compare: {name} could not be read ({error})", name=name, error=reason)
        if len(errors) > 1:
            return trf("{n} series could not be imported. First error: {error}",
                       n=len(errors), error=reason)
        return reason

    def cancel_import(self, *, announce=True):
        active = bool(self._reading)
        self._read_cancel.set()
        self.tasks.cancel("compare-read")
        self._read_queue.clear()
        self._reading = 0
        self.import_job.hide()
        if active:
            self._status((lambda: tr(CANCELLED)) if announce else "")
        if active and not self._disposed:
            self.refresh()
            self._render()
            if self.series and not self.comparison_current():
                self._timer.start()

    def closeEvent(self, event):
        self.cancel_import(announce=False)
        super().closeEvent(event)


def _restore_entry(service, entry, sidecar, progress, token):
    """Restore a project's series on a worker, closing NPZ handles before publishing arrays."""
    if "files" in entry:
        return service.series_from_files(tuple(Path(path) for path in entry["files"]), entry.get("name"),
                                         progress=progress, cancelled=token.is_set)
    if sidecar is None:
        raise ValueError("The project's Compare array file is missing.")
    with np.load(sidecar) as stored:
        index = int(entry["arrays"])
        return SeriesData(entry.get("name", "series"), stored[f"x_{index}"], stored[f"image_{index}"],
                          tuple(entry.get("labels") or ()), entry.get("x_label", "q (Å⁻¹)"),
                          entry.get("source", "Analyze"))
