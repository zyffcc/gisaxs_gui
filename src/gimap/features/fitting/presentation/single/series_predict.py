"""In-situ series ▸ Predict each frame: the method choice of the Start step, and the prediction of one frame.

Every frame is predicted with Predict parameters (as in Single analysis), with the particles of Single analysis'
model. With “Prefer the solution that continues the previous frame” (and a start that follows the previous frame:
not “The model in Single analysis”, not the first frame of a new stage) the solution kept is the one closest to the
previous frame's among those about as probable (``closest``). Stop cancels the frame's job.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf

from ...application.series_selection import CLOSE_ENOUGH, FramePredictor, closest_index, ranked, smoothed_fits
from ...application.single_predict import MAX_PARTICLES, predict_request, predicted

METHODS = ("local", "global", "predict")
NOT_PREDICT = "Predict each frame needs 1–3 particles in Single analysis' model."


def closest(solutions, previous_model):
    """Most probable without a reference, otherwise the closest eligible mode in log coordinates."""
    ordered = ranked(solutions)
    return ordered[closest_index(ordered, previous_model, CLOSE_ENOUGH)]


class FitSeriesPredictMixin:
    """Needs the page's widgets (``method_local``, ``method_global``, ``method_predict``, ``follow_check``),
    ``single`` (Single analysis), ``view_model`` and ``_stop``."""

    def setup_series_predict(self) -> None:
        self.reset_smoothing()
        self._series_job = None
        self.follow_check.toggled.connect(lambda on: self.smooth_check.setChecked(False) if on else None)
        self.smooth_check.toggled.connect(lambda on: self.follow_check.setChecked(False) if on else None)
        self.smooth_check.toggled.connect(lambda _: self._remember_choices())
        self.smooth_scale.valueChanged.connect(lambda _: self._remember_choices())

    def _smooth_active(self) -> bool:
        return self.predicts_frames() and self.smooth_check.isChecked() and not self.independent_samples()

    def _update_continuity_controls(self) -> None:
        ordered = not self.independent_samples()
        self.follow_check.setEnabled(self.predicts_frames() and ordered and not self.start_same.isChecked())
        self.smooth_check.setVisible(self.predicts_frames())
        self.smooth_check.setEnabled(ordered)
        self.scale_row.setVisible(self._smooth_active())
        self.backward_check.setEnabled(ordered and not self._smooth_active())

    def _restore_continuity(self, state) -> None:
        self.follow_check.setChecked(bool(state.get("follow", True)))
        self.smooth_check.setChecked(bool(state.get("smooth", False)) and not self.independent_samples())
        self.smooth_scale.setValue(float(state.get("smooth_scale", 0.1)))

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
        follows = settings.follow and not settings.smooth and settings.start != "same" and not new_stage
        return FramePredictor(self.predict_solutions(), previous if follows and previous is not None and previous.ok else None)

    def reset_smoothing(self):
        self._frame_modes = {}
        self._independent_fits = {}
        self._smooth_dirty = False
        self._smoothing_error = ""

    def remember_modes(self, frame, predictor):
        """The GUI thread receives a completed worker's modes, including replacement files in Watch."""
        if self._settings.smooth and predictor is not None:
            self._frame_modes[frame.index] = predictor.ordered
            self._independent_fits[frame.index] = frame
            self._smooth_dirty = True

    def _publish_modes(self, updates):
        if all(self.fits.get(index) is frame for index, frame in updates.items()):
            return
        self.fits.update(updates)
        self._unsaved = self._unsaved or any(frame.fitted for frame in updates.values())
        self._rebuild_results()
        self._mark_stages()
        if self._selected is not None:
            self._select(self._selected)
        self._draw_trend()
        self.refresh()

    def begin_smoothing(self):
        """Apply a joint selection when the queue drains, and again when Watch adds or replaces frames."""
        if not self._smooth_dirty:
            return False
        self._smooth_dirty = False
        fits, modes, curves = dict(self.fits), dict(self._frame_modes), dict(self._curves)
        settings, epoch, stop = self._settings, self._loading_epoch, self._stop.is_set
        boundaries = set()
        if settings.start == "stages":
            ordered = sorted(fits)
            boundaries = {b for a, b in zip(ordered, ordered[1:]) if self.stage_of_frame(a) != self.stage_of_frame(b)}

        def done(updates):
            if self._loading_current(epoch) and self.running:
                self._smoothing_error = ""
                self._publish_modes(self._independent_fits if stop() else updates)
                self._next()

        def failed(message, _details):
            if self._loading_current(epoch) and self.running:
                self._publish_modes(self._independent_fits)
                self._smoothing_error = message
                self._log(trf("Series smoothing failed: {error}", error=tr(message)))
                self._next()

        self._status(lambda: tr("Smoothing the Predict solutions across frames…"))
        self.tasks.submit("series-hmm", lambda: smoothed_fits(fits, modes, curves, settings, boundaries, stop),
                          on_done=done, on_error=failed)
        return True

    def cancel_series_job(self) -> None:
        if self._series_job:
            self.view_model.context.jobs.cancel(self._series_job)


__all__ = ["CLOSE_ENOUGH", "FitSeriesPredictMixin", "METHODS", "NOT_PREDICT", "closest"]
