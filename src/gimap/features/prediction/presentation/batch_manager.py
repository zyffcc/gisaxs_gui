"""The run of a folder batch: its frames are predicted one after another on a worker thread.

``start_batch_prediction(paths, predict)`` predicts ``paths`` in order; row ``i`` of the results is ``paths[i]``.
With ``follow`` (Watch for new frames) the run does not end when they are done: ``append(rows_and_paths)`` gives it
more while it runs, and it waits for them until ``stop_following()`` or Stop (``cancel_prediction``), which marks the
frames not started as cancelled.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List

from PyQt5.QtCore import QObject, pyqtSignal

from .result_types import PredictStatus

WAIT_S = 0.2
"""How often a waiting run looks whether it was stopped."""


class MultiFilePredictManager(QObject):
    """Runs a folder batch (one frame at a time, in a single worker thread)."""

    prediction_started = pyqtSignal()
    prediction_completed = pyqtSignal()
    result_updated = pyqtSignal(int, dict)  # row, update
    progress_updated = pyqtSignal(int, int)  # predicted, given so far

    def __init__(self, parent=None):
        super().__init__(parent)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_futures = []
        self.is_running = False
        self.cancelled = False
        self.follow = False
        self._lock = threading.Lock()
        self._frames: queue.Queue = queue.Queue()
        self._given = 0
        self._done = 0
        self._closed = True

    def start_batch_prediction(self, file_paths: List[str], predict_func: Callable, follow: bool = False) -> None:
        """Predict ``file_paths`` (rows 0, 1, …); with ``follow``, wait for more (``append``) once they are done."""
        if self.is_running:
            return
        self.is_running = True
        self.cancelled = False
        self.follow = bool(follow)
        self.current_futures.clear()
        with self._lock:
            self._frames = queue.Queue()
            self._given = self._done = 0
            self._closed = False
        self._put(enumerate(file_paths))
        self.prediction_started.emit()
        future = self.executor.submit(self._batch_predict_worker, self._frames, predict_func)
        self.current_futures.append(future)

    def append(self, rows_and_paths: Iterable[tuple[int, str]]) -> bool:
        """Give the run more frames (row, path); ``False`` when it has ended or was stopped and takes none."""
        if self.cancelled:
            return False
        return self._put(rows_and_paths)

    def _put(self, rows_and_paths) -> bool:
        with self._lock:
            if self._closed:
                return False
            for row, path in rows_and_paths:
                self._frames.put((row, path))
                self._given += 1
        return True

    def idle(self) -> bool:
        """Every frame given so far is predicted (a run that follows waits for more)."""
        with self._lock:
            return self._done >= self._given

    def can_take(self) -> bool:
        """Nothing waits in the queue (the frame being predicted aside): Keep up gives the next stack now."""
        with self._lock:
            return self._frames.qsize() == 0

    def stop_following(self) -> None:
        """End the run once the frames given so far are predicted."""
        self.follow = False

    def cancel_prediction(self) -> None:
        """Stop after the current frame; the frames not started are marked cancelled."""
        self.cancelled = True
        self.follow = False
        for future in self.current_futures:
            future.cancel()

    def _next(self, frames: queue.Queue):
        """The next frame, ``None`` while one may still come, or ``False`` at the end of the run (closed)."""
        try:
            return frames.get(timeout=WAIT_S) if self.follow else frames.get_nowait()
        except queue.Empty:
            with self._lock:
                if self.follow and not self.cancelled:
                    return None
                if frames.empty():
                    self._closed = True
                    return False
            return None

    def _batch_predict_worker(self, frames: queue.Queue, predict_func: Callable) -> None:
        completed = 0
        try:
            while True:
                if self.cancelled:
                    with self._lock:
                        self._closed = True
                    while not frames.empty():
                        row, _path = frames.get_nowait()
                        self.result_updated.emit(row, {"status": PredictStatus.CANCELLED})
                    break
                item = self._next(frames)
                if item is False:
                    break
                if item is None:
                    continue
                row, file_path = item
                self.result_updated.emit(row, {"status": PredictStatus.RUNNING})
                try:
                    result_data = predict_func(file_path)
                    self.result_updated.emit(row, {"status": PredictStatus.COMPLETED, "prediction_data": result_data})
                    completed += 1
                except Exception as e:
                    self.result_updated.emit(row, {"status": PredictStatus.FAILED, "error_message": str(e)})
                with self._lock:
                    self._done += 1
                    given = self._given
                self.progress_updated.emit(completed, given)
        finally:
            with self._lock:
                self._closed = True
            self.is_running = False
            self.follow = False
            self.prediction_completed.emit()
