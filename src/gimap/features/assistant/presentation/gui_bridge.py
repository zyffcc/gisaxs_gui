"""Hand work from the run's worker thread to the GUI thread and wait for the answer.

The model loop runs on a worker thread so the window stays responsive; every
action on the Analyze page must still happen on the GUI thread.  ``call_async``
passes ``start(done)`` to the GUI thread and blocks the worker until ``done``
is called there, which allows actions that finish later (a re-analysis).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from PyQt5.QtCore import QCoreApplication, QObject, Qt, QThread, pyqtSignal, pyqtSlot

DEFAULT_TIMEOUT_S = 600.0


class RunCancelled(RuntimeError):
    """The user stopped the run while it waited for the GUI."""


class _Box:
    def __init__(self):
        self.event = threading.Event()
        self.value: Any = None
        self.error: Optional[BaseException] = None


class GuiBridge(QObject):
    _request = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._request.connect(self._execute, Qt.QueuedConnection)

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Run ``fn(*args)`` on the GUI thread and return its result."""
        return self.call_async(lambda done: done(fn(*args)), **kwargs)

    def call_async(
        self,
        start: Callable[[Callable[[Any], None]], None],
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> Any:
        box = _Box()
        if QThread.currentThread() is self.thread():
            # Already on the GUI thread (tests): run here and keep the event loop turning.
            self._execute((start, box))
            deadline = time.monotonic() + timeout
            while not box.event.is_set():
                QCoreApplication.processEvents()
                time.sleep(0.005)
                if time.monotonic() > deadline:
                    raise TimeoutError("The GUI did not finish the action in time.")
        else:
            self._request.emit((start, box))
            deadline = time.monotonic() + timeout
            while not box.event.wait(0.05):
                if cancelled is not None and cancelled():
                    raise RunCancelled("Stopped by the user.")
                if time.monotonic() > deadline:
                    raise TimeoutError("The GUI did not finish the action in time.")
        if box.error is not None:
            raise box.error
        return box.value

    @pyqtSlot(object)
    def _execute(self, item) -> None:
        start, box = item

        def done(value: Any = None) -> None:
            box.value = value
            box.event.set()

        try:
            start(done)
        except BaseException as exc:  # reported to the waiting worker
            box.error = exc
            box.event.set()


__all__ = ["GuiBridge", "RunCancelled"]
