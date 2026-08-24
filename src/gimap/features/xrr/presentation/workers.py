"""Qt thread bridges for XRR application commands."""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class XrrInspectWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, view_model, spec):
        super().__init__()
        self.view_model = view_model
        self.spec = spec

    def run(self) -> None:
        try:
            self.finished.emit(self.view_model.inspect(self.spec))
        except Exception as exc:
            self.failed.emit(str(exc))


class XrrExtractionWorker(QObject):
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, view_model, request):
        super().__init__()
        self.view_model = view_model
        self.request = request

    def run(self) -> None:
        try:
            result = self.view_model.extract(
                self.request,
                on_progress=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.view_model.cancel()


__all__ = ["XrrExtractionWorker", "XrrInspectWorker"]
