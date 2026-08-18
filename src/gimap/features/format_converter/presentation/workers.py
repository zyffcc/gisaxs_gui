"""Workers for Format Converter."""

from __future__ import annotations


from PyQt5.QtCore import QObject, pyqtSignal


class _PreviewWorker(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id: int, source, view_model):
        super().__init__()
        self.request_id = request_id
        self.source = source
        self.view_model = view_model

    def run(self) -> None:
        try:
            payload = self.view_model.load_preview(self.source)
            self.finished.emit(self.request_id, payload)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class _ConversionWorker(QObject):
    progress = pyqtSignal(int, int, str, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, options, view_model):
        super().__init__()
        self.options = options
        self.view_model = view_model

    def run(self) -> None:
        try:
            report = self.view_model.convert(self.options, self.progress.emit)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.view_model.cancel()

    def set_paused(self, paused: bool) -> None:
        self.view_model.set_paused(paused)
