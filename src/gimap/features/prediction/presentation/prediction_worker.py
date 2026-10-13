"""Single-file prediction off the GUI thread.

The worker only calls the ViewModel (module preprocessing, then the isolated model process);
it never touches a widget. The binding receives ``(run_id, prepared, result, error)`` through
``prediction_finished`` on the GUI thread and does all logging and display there.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

STOPPED_TEXT = "Prediction stopped before the model process started."


class PredictionWorker(QThread):
    """Run ``prepare_input`` and ``predict_prepared`` for one image (same path as before)."""

    # run id, prepared input (or None), prediction result (or None), error message ("" when ok)
    prediction_finished = pyqtSignal(int, object, object, str)

    def __init__(self, view_model, image, typed_module, model_path, run_id: int, parent=None):
        super().__init__(parent)
        self._view_model = view_model
        self._image = image
        self._module = typed_module
        self._model_path = Path(model_path)
        self.run_id = int(run_id)
        self.discard = False

    # What the run was started with: the binding discards the result if any of them changed.
    @property
    def image(self):
        return self._image

    @property
    def module(self):
        return self._module

    @property
    def model_path(self) -> Path:
        return self._model_path

    def run(self) -> None:
        prepared = None
        try:
            prepared = self._view_model.prepare_input(self._image, self._module)
            if prepared is None:
                message = self._view_model.state.error_message or "Module preprocessing failed"
                self.prediction_finished.emit(self.run_id, None, None, str(message))
                return
            if self.discard:
                # Stopped (or the application quits) during preprocessing: start no model process.
                self.prediction_finished.emit(self.run_id, prepared, None, STOPPED_TEXT)
                return
            result = self._view_model.predict_prepared(
                prepared.values, self._module, self._model_path, tuple(prepared.steps)
            )
            if result is None:
                message = self._view_model.state.error_message or "Isolated prediction failed"
                self.prediction_finished.emit(self.run_id, prepared, None, str(message))
                return
            self.prediction_finished.emit(self.run_id, prepared, result, "")
        except Exception as exc:  # noqa: BLE001 - reported on the GUI thread
            self.prediction_finished.emit(self.run_id, prepared, None, str(exc) or type(exc).__name__)


__all__ = ["PredictionWorker"]
