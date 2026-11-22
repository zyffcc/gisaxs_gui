"""In-situ series ▸ Predict each frame: the method choice of the Start step, and the prediction of one frame.

Every frame is predicted with Predict parameters (as in Single analysis), with the particles of Single analysis'
model. With “Prefer the solution that continues the previous frame” (and a start that follows the previous frame:
not “The model in Single analysis”, not the first frame of a new stage) the solution kept is the one closest to the
previous frame's among those about as probable (``closest``). Stop cancels the frame's job.
"""

from __future__ import annotations

import math

from src.gimap.app.presentation.i18n import tr

from ...application.single_predict import MAX_PARTICLES, predict_request, predicted

METHODS = ("local", "global", "predict")
NOT_PREDICT = "Predict each frame needs 1–3 particles in Single analysis' model."
CLOSE_ENOUGH = 0.2
"""Following the previous frame, Predict keeps the solution closest to it among those at least this share of the most
probable one's probability."""


def closest(solutions, previous_model):
    """The most probable solution, or with a previous frame's model the closest of those about as probable (sizes and
    distances compared in ln, the components by radius)."""
    if not solutions:
        raise RuntimeError("Predict parameters found no solution for this frame")
    if previous_model is None:
        return solutions[0]
    top = max(s.probability for s in solutions)

    def distance(model):
        own = sorted(model.components, key=lambda c: c.value("R"))
        other = sorted(previous_model.components, key=lambda c: c.value("R"))
        if len(own) != len(other):
            return float("inf")
        total = 0.0
        for a, b in zip(own, other):
            for key in ("R", "D"):
                if key in a.keys() and key in b.keys():
                    total += abs(math.log(a.value(key) / b.value(key)))
        return total

    near = [s for s in solutions if s.probability >= CLOSE_ENOUGH * top]
    return min(near, key=lambda s: distance(s.model))


class FitSeriesPredictMixin:
    """Needs the page's widgets (``method_local``, ``method_global``, ``method_predict``, ``follow_check``),
    ``single`` (Single analysis), ``view_model`` and ``_stop``."""

    def setup_series_predict(self) -> None:
        self._series_job = None
        self.method_predict.toggled.connect(lambda _on: self.follow_check.setEnabled(self.predicts_frames()
                                                                                     and not self.independent_samples()))
        self.follow_check.setEnabled(self.predicts_frames())

    def predicts_frames(self) -> bool:
        return self.method_predict.isChecked()

    def series_method(self) -> str:
        if self.method_predict.isChecked():
            return "predict"
        return "global" if self.method_global.isChecked() else "local"

    def set_series_method(self, method) -> None:
        {"global": self.method_global, "predict": self.method_predict}.get(method, self.method_local).setChecked(True)

    def predict_can_start(self, model) -> bool:
        """Whether the model can be predicted frame by frame; says why not."""
        if self.series_method() != "predict" or 1 <= len(model.components) <= MAX_PARTICLES:
            return True
        self._status(tr(NOT_PREDICT), "warning")
        return False

    def predict_solutions(self):
        """``solutions(data, start)``: every solution of Predict parameters for one frame's points, most probable first
        (the particles of ``start``; thread-safe, Stop cancels the job)."""
        jobs, stop = self.view_model.context.jobs, self._stop

        def solutions(data, start):
            request = predict_request(data, start, "model")
            self._series_job = request.job_id
            try:
                result = jobs.run(request, lambda _progress: jobs.cancel(request.job_id) if stop.is_set() else None)
            finally:
                self._series_job = None
            if not result.succeeded:
                raise RuntimeError(result.error.message if result.error else "Predict parameters failed")
            return predicted(result.value)

        return solutions

    def predictor(self, settings, previous, new_stage: bool):
        """``predict(data, start)`` for ``fit_frame``: Predict parameters on one frame (the particles of ``start``),
        the solution ``closest`` to the previous frame's model when asked to follow it."""
        follows = settings.follow and settings.start != "same" and not new_stage
        previous_model = previous.result.model if follows and previous is not None and previous.ok else None
        solutions = self.predict_solutions()
        return lambda data, start: closest(solutions(data, start), previous_model).model

    def cancel_series_job(self) -> None:
        if self._series_job:
            self.view_model.context.jobs.cancel(self._series_job)


__all__ = ["CLOSE_ENOUGH", "FitSeriesPredictMixin", "METHODS", "NOT_PREDICT", "closest"]
