"""One background task runner for the GUI.

``TaskRunner.submit(key, fn, on_done=..., on_error=...)`` runs ``fn()`` on a
thread pool and calls back on the GUI thread.  A newer submission with the
same key supersedes the older ones: their results are discarded, so a quick
succession of edits (dragging a cut band, stepping through files) only ever
shows the latest outcome.  The work itself is not interrupted; callers that
need cooperative cancellation pass a flag into ``fn``.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable, Optional

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot


class _Signals(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str, str)


class _Task(QRunnable):
    def __init__(self, ticket: int, fn: Callable[[], Any], signals: _Signals):
        super().__init__()
        self.setAutoDelete(True)
        self._ticket = ticket
        self._fn = fn
        self._signals = signals

    def run(self) -> None:  # executed on a pool thread
        try:
            result = self._fn()
        except Exception as exc:  # delivered to the GUI thread, never swallowed
            message = str(exc) or type(exc).__name__
            self._signals.failed.emit(self._ticket, message, traceback.format_exc())
        else:
            self._signals.done.emit(self._ticket, result)


class TaskRunner(QObject):
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None, *, max_threads: Optional[int] = None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        if max_threads is not None:
            self._pool.setMaxThreadCount(max(1, int(max_threads)))
        self._signals = _Signals(self)
        self._signals.done.connect(self._deliver)
        self._signals.failed.connect(self._deliver_error)
        self._next_ticket = 0
        self._latest: dict[str, int] = {}
        self._pending: dict[int, tuple[str, Optional[Callable], Optional[Callable]]] = {}

    def submit(
        self,
        key: str,
        fn: Callable[[], Any],
        *,
        on_done: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> int:
        self._next_ticket += 1
        ticket = self._next_ticket
        was_busy = self.is_busy()
        self._latest[key] = ticket
        self._pending[ticket] = (key, on_done, on_error)
        self._pool.start(_Task(ticket, fn, self._signals))
        if not was_busy:
            self.busy_changed.emit(True)
        return ticket

    def cancel(self, key: str) -> None:
        """Forget the running task for ``key``; its result will be ignored."""
        self._latest.pop(key, None)

    def is_busy(self) -> bool:
        return bool(self._pending)

    def is_current(self, key: str, ticket: int) -> bool:
        return self._latest.get(key) == ticket

    def wait(self, timeout_s: float = 30.0) -> bool:
        """Block (processing events) until every submitted task was delivered."""
        deadline = time.monotonic() + float(timeout_s)
        while self._pending and time.monotonic() < deadline:
            self._pool.waitForDone(20)
            QCoreApplication.processEvents()
        return not self._pending

    def shutdown(self, timeout_ms: int = 10000) -> None:
        self._pool.clear()
        self._pool.waitForDone(int(timeout_ms))
        self._latest.clear()
        self._pending.clear()

    def _finish(self, ticket: int):
        entry = self._pending.pop(ticket, None)
        if not self._pending:
            self.busy_changed.emit(False)
        if entry is None:
            return None
        key, on_done, on_error = entry
        if self._latest.get(key) != ticket:
            return None
        self._latest.pop(key, None)
        return on_done, on_error

    @pyqtSlot(int, object)
    def _deliver(self, ticket: int, result: Any) -> None:
        callbacks = self._finish(ticket)
        if callbacks and callbacks[0] is not None:
            _call(callbacks[0], result)

    @pyqtSlot(int, str, str)
    def _deliver_error(self, ticket: int, message: str, details: str) -> None:
        callbacks = self._finish(ticket)
        if callbacks and callbacks[1] is not None:
            _call(callbacks[1], message, details)


def _call(callback: Callable, *values) -> None:
    """Call back, unless the widget the result was for has been deleted meanwhile (a dialog closed while its
    task ran): then the result has nowhere to go and is dropped."""
    receiver = getattr(callback, "__self__", None)
    if isinstance(receiver, QObject) and sip.isdeleted(receiver):
        return
    try:
        callback(*values)
    except RuntimeError as exc:
        if "has been deleted" not in str(exc):
            raise
        logging.getLogger(__name__).debug("A task's result was dropped: %s", exc)


__all__ = ["TaskRunner"]
