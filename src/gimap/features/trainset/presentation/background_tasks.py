"""Qt worker primitives for short Trainset presentation tasks."""

from __future__ import annotations


from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


class _WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class _FunctionWorker(QRunnable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.with_progress = bool(kwargs.pop("_with_progress", False))
        self.kwargs = kwargs
        self.signals = _WorkerSignals()

    def run(self):
        try:
            if self.with_progress:
                result = self.function(self.signals.progress.emit, *self.args, **self.kwargs)
            else:
                result = self.function(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
