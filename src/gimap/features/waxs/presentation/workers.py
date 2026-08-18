"""Qt worker objects for WAXS loading and batch jobs."""

from __future__ import annotations

from dataclasses import dataclass


from pathlib import Path


import numpy as np


from PyQt5.QtCore import QObject, pyqtSignal


from src.gimap.features.waxs.application import (
    WaxsBatchRequest,
)


@dataclass
class ImageLoadResult:
    file_path: str
    frame_index: int
    frame_count: int
    image: np.ndarray


class ImageLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, frame_index: int, view_model):
        super().__init__()
        self.file_path = file_path
        self.frame_index = int(frame_index)
        self.view_model = view_model

    def run(self) -> None:
        try:
            loaded = self.view_model.load_image(Path(self.file_path), self.frame_index)
            if loaded is None:
                raise RuntimeError(self.view_model.state.error_message or "Failed to load image.")
            self.finished.emit(
                ImageLoadResult(
                    file_path=str(loaded.path),
                    frame_index=loaded.frame_index,
                    frame_count=loaded.frame_count,
                    image=loaded.image,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, request: WaxsBatchRequest, view_model):
        super().__init__()
        self.request = request
        self.view_model = view_model

    def stop(self) -> None:
        self.view_model.cancel_batch()

    def set_paused(self, paused: bool) -> None:
        self.view_model.set_batch_paused(paused)

    def run(self) -> None:
        try:

            def report(value) -> None:
                total = max(1, int(value.total))
                percent = int(round(int(value.completed) * 100 / total))
                self.progress.emit(percent, f"Processed {value.name}")

            result = self.view_model.run_batch(self.request, on_progress=report)
            if result is None:
                raise RuntimeError(
                    self.view_model.state.error_message or "Batch processing failed."
                )
            if result.cancelled:
                self.finished.emit("Batch stopped by user.")
            elif result.failed_count:
                failures = [
                    item.error_message or item.name
                    for item in result.items
                    if item.status == "failed"
                ]
                self.failed.emit("; ".join(failures))
            else:
                self.finished.emit("Batch processing completed.")
        except Exception as exc:
            self.failed.emit(str(exc))
