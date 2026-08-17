"""AI use case 的 Qt thread bridge；不管理进程。"""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class AiCandidateWorker(QObject):
    progress = pyqtSignal(object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    finished = pyqtSignal()

    def __init__(self, view_model, request, *, refine: bool):
        super().__init__()
        self._view_model = view_model
        self._request = request
        self._refine = bool(refine)

    @pyqtSlot()
    def run(self):
        try:
            result = self._view_model.run_ai_candidates(
                self._request,
                refine=self._refine,
                on_progress=self.progress.emit,
            )
            if result is None:
                self.failed.emit(
                    self._view_model.state.ai_error_code or "ai_job_failed",
                    self._view_model.state.error_message or "AI fitting failed",
                )
            else:
                self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(type(exc).__name__, str(exc))
        finally:
            self.finished.emit()
