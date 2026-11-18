"""In-situ series ▸ Predict each frame (fit1d): the method choice of the Start step, and the prediction of one frame.

Every frame is predicted with fit1d's job (as Predict in Single analysis): the particles, fixed values and ranges are
those of Single analysis' model, the settings those of its Fit step. With “Prefer the solution that continues the
previous frame” (and a start that follows the previous frame: not “The model in Single analysis”, not the first frame
of a new stage) the job also fits the frame as a continuation of the previous one and keeps that when it fits as
well (``domain.fit1d_follow``) — among equivalent solutions the smoothest series. Stop cancels the frame's job.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit1d import SHAPES, fit1d_request
from ...application.single_predict import predict_request, predicted
from ...domain.posterior import encoding as posterior_encoding

METHODS = ("local", "global", "fit1d", "predict")
NOT_FIT1D = "Predict each frame needs Sphere / Vertical cylinder particles in Single analysis' model."
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
                    total += abs(__import__("math").log(a.value(key) / b.value(key)))
        return total

    near = [s for s in solutions if s.probability >= CLOSE_ENOUGH * top]
    return min(near, key=lambda s: distance(s.model))


class FitSeriesFit1DMixin:
    """Needs the page's widgets (``method_local``, ``method_global``, ``method_fit1d``, ``follow_check``),
    ``single`` (Single analysis), ``view_model`` and ``_stop``."""

    def setup_series_fit1d(self) -> None:
        self._series_job = None
        for button in (self.method_fit1d, self.method_predict):
            button.toggled.connect(lambda _on: self.follow_check.setEnabled(self.predicts_frames()
                                                                           and not self.independent_samples()))
        self.follow_check.setEnabled(self.predicts_frames())

    def predicts_frames(self) -> bool:
        return self.method_fit1d.isChecked() or self.method_predict.isChecked()

    def series_method(self) -> str:
        if self.method_predict.isChecked():
            return "predict"
        if self.method_fit1d.isChecked():
            return "fit1d"
        return "global" if self.method_global.isChecked() else "local"

    def set_series_method(self, method) -> None:
        {"global": self.method_global, "fit1d": self.method_fit1d,
         "predict": self.method_predict}.get(method, self.method_local).setChecked(True)

    def fit1d_can_start(self, model) -> bool:
        """Whether the model can be predicted frame by frame; says why not."""
        if self.series_method() == "predict":
            if 1 <= len(model.components) <= posterior_encoding.SLOTS:
                return True
            self._status(tr(NOT_PREDICT), "warning")
            return False
        if self.series_method() != "fit1d":
            return True
        if model.components and all(component.family in SHAPES for component in model.components):
            return True
        self._status(tr(NOT_FIT1D), "warning")
        return False

    def predictor(self, settings, previous, new_stage: bool):
        """``predict(data, start)`` for ``fit_frame``: fit1d's job on one frame, following ``previous`` when asked."""
        follows = settings.follow and settings.start != "same" and not new_stage
        if settings.method == "predict":
            return self._posterior_predictor(previous.result.model if follows and previous is not None and previous.ok
                                             else None)
        populations = None
        if follows and previous is not None and previous.ok and previous.native:
            populations = previous.native.get("populations")
        jobs, stop = self.view_model.context.jobs, self._stop

        def predict(data, start):
            request = fit1d_request(data, start, dict(settings.fit1d), resolve_edges=settings.resolve_edges,
                                    previous=populations)
            self._series_job = request.job_id
            try:
                result = jobs.run(request, lambda _progress: jobs.cancel(request.job_id) if stop.is_set() else None)
            finally:
                self._series_job = None
            if not result.succeeded:
                raise RuntimeError(result.error.message if result.error else "fit1d failed")
            return result.value

        return predict

    def _posterior_predictor(self, previous_model):
        """``predict(data, start)``: Predict parameters on one frame (the particles of ``start``), the solution
        ``closest`` to the previous frame's model."""
        jobs, stop = self.view_model.context.jobs, self._stop

        def predict(data, start):
            request = predict_request(data, start, "model")
            self._series_job = request.job_id
            try:
                result = jobs.run(request, lambda _progress: jobs.cancel(request.job_id) if stop.is_set() else None)
            finally:
                self._series_job = None
            if not result.succeeded:
                raise RuntimeError(result.error.message if result.error else "Predict parameters failed")
            return closest(predicted(result.value), previous_model).model

        return predict

    def cancel_series_job(self) -> None:
        if self._series_job:
            self.view_model.context.jobs.cancel(self._series_job)


__all__ = ["CLOSE_ENOUGH", "FitSeriesFit1DMixin", "METHODS", "NOT_FIT1D", "NOT_PREDICT", "closest"]
