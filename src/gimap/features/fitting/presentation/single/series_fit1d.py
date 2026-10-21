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

METHODS = ("local", "global", "fit1d")
NOT_FIT1D = "Predict each frame needs Sphere / Vertical cylinder particles in Single analysis' model."


class FitSeriesFit1DMixin:
    """Needs the page's widgets (``method_local``, ``method_global``, ``method_fit1d``, ``follow_check``),
    ``single`` (Single analysis), ``view_model`` and ``_stop``."""

    def setup_series_fit1d(self) -> None:
        self._series_job = None
        self.method_fit1d.toggled.connect(lambda on: self.follow_check.setEnabled(on))
        self.follow_check.setEnabled(self.method_fit1d.isChecked())

    def series_method(self) -> str:
        if self.method_fit1d.isChecked():
            return "fit1d"
        return "global" if self.method_global.isChecked() else "local"

    def set_series_method(self, method) -> None:
        {"global": self.method_global, "fit1d": self.method_fit1d}.get(method, self.method_local).setChecked(True)

    def fit1d_can_start(self, model) -> bool:
        """Whether the model can be predicted frame by frame; says why not."""
        if self.series_method() != "fit1d":
            return True
        if model.components and all(component.family in SHAPES for component in model.components):
            return True
        self._status(tr(NOT_FIT1D), "warning")
        return False

    def predictor(self, settings, previous, new_stage: bool):
        """``predict(data, start)`` for ``fit_frame``: fit1d's job on one frame, following ``previous`` when asked."""
        follows = settings.follow and settings.start != "same" and not new_stage
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

    def cancel_series_job(self) -> None:
        if self._series_job:
            self.view_model.context.jobs.cancel(self._series_job)


__all__ = ["FitSeriesFit1DMixin", "METHODS", "NOT_FIT1D"]
