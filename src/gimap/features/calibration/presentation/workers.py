"""Qt worker bridges for Calibration use cases."""

from __future__ import annotations

import logging

from typing import Optional


from PyQt5.QtCore import QObject, pyqtSignal


from ..application import (
    CalibrationCancelledError,
)


LOGGER = logging.getLogger(__name__)


class ImageLoaderWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, path: str, view_model, dataset_path: Optional[str] = None):
        super().__init__()
        self.path = path
        self.view_model = view_model
        self.dataset_path = dataset_path

    def run(self) -> None:
        try:
            self.finished.emit(self.view_model.load_image(self.path, self.dataset_path))
        except Exception as exc:
            LOGGER.exception("Failed to load calibration image")
            self.failed.emit(exc)


class CalibrationWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, view_model, options: dict):
        super().__init__()
        self.view_model = view_model
        self.options = options
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            self.finished.emit(
                self.view_model.run_calibration(
                    self.options,
                    progress=lambda value, stage: self.progress.emit(value, stage),
                    cancelled=lambda: self.cancel_requested,
                )
            )
        except Exception as exc:
            if not isinstance(exc, CalibrationCancelledError):
                LOGGER.exception("Geometry calibration failed")
            self.failed.emit(exc)
