"""Predict parameters on the single-curve page: GIMaP's proposal network and the posterior it starts, in a job process
(``application/single_predict.py``, ``infrastructure/adapters/posterior_job.py``).

The particles to try (those of Model, or a choice of families for the network to choose among) are shown under the
method; every solution is refined by least squares (``domain.posterior.refine.predict``, one path). The solutions go
to Results with their probability, the most probable into Model; the refined values of the solution in Model with
their 68 % intervals, the points its χ²ᵣ is of and its flags are listed in Fit details (``posterior_lines``; the
words: ``predict_texts``, the names and sentences of ``shared/fit_names``).
"""

from __future__ import annotations

import logging

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_predict import PARTICLES, network_available, predict_request, predicted
from ...application.substrate_level import prediction_model
from ..predict_substrate import ON_SUBSTRATE, PredictSubstrateDialog
from .predict_progress import PredictProgressMixin, as_solutions
from .predict_texts import solution_lines

LOG = logging.getLogger(__name__)

MISSING = "Predict parameters needs GIMaP's proposal network, which is missing in this installation ({name})."


class PredictPageMixin(PredictProgressMixin):
    def setup_predict(self) -> None:
        self._predict_job_id = None
        self._predict_selection = None
        self.setup_predict_progress()  # the stages, the time, and what Stop keeps (predict_progress.py)
        self.predict_substrate_button.clicked.connect(self.edit_predict_cut)
        for text, value in PARTICLES:
            self.predict_particles.addItem(tr(text), value)
        stored = self._remembered("predict_settings")
        self.apply_predict_settings(stored if isinstance(stored, dict) else {})
        self.predict_particles.currentIndexChanged.connect(
            lambda _index: self._remember(predict_settings=self.predict_settings()))

    def predict_settings(self) -> dict:
        return {"particles": self.predict_particles.currentData()}

    def apply_predict_settings(self, settings: dict) -> None:
        for key, combo, default in (("particles", self.predict_particles, "model"),):
            blocker = QSignalBlocker(combo)
            index = combo.findData(settings.get(key, default))
            combo.setCurrentIndex(index if index >= 0 else combo.findData(default))
            del blocker

    def predict_available(self) -> bool:
        return network_available()

    def edit_predict_cut(self) -> None:
        model = self.session.model
        cut = dict(grazing={key: model.globals[key].value for key in ("wavelength", "alpha_i", "alpha_f")},
                   substrate=None if model.substrate is None else model.substrate.to_dict())
        dialog = PredictSubstrateDialog(cut, self)
        if dialog.exec_():
            self.set_model(prediction_model(model, dialog.cut()), label=("an edit in Model", {}))
        dialog.deleteLater()

    def refresh_predict_cut(self) -> None:
        model = self.session.model
        active = model.substrate is not None and all(model.globals[key].value > 0 for key in ("alpha_i", "alpha_f"))
        self.predict_substrate_note.setText(trf(ON_SUBSTRATE, substrate=tr(model.substrate.label)) if active else "")
        self.predict_substrate_note.setVisible(active)
        self.reflected_beam.setVisible(not active)
        self.predict_qz_row.setVisible(not active and self.reflected_beam.isChecked())

    def _start_predict(self, data, model, selection, curve) -> None:
        settings = self.predict_settings()
        particles = settings["particles"]
        if particles == "model" and not model.components:
            particles = "spheres_cylinders"  # nothing in Model yet: the network chooses
        request = predict_request(data, model, particles)
        if self.reflected_beam.isChecked() and "substrate" not in request.payload:
            request.payload["reflected_beam"] = True
            if not request.payload["qz"] and self.measured_qz.value():
                request.payload["measured_qz"] = self.measured_qz.value()
        self._predict_job_id = request.job_id
        self._predict_selection = selection
        given = request.payload.get("compositions")
        self._predict_started(None if given is None else len(given))
        self.tasks.submit("fit", lambda: self._run_predict_job(request),
                          on_done=lambda value: self._predict_finished(value, curve, selection),
                          on_error=lambda message, details="": self._failed(message, details, curve))

    def _run_predict_job(self, request):
        if self._stop_event.is_set():
            raise RuntimeError("Fitting cancelled")
        result = self.view_model.context.jobs.run(request, self._predict_progress)
        if not result.succeeded:
            error = result.error
            if error is not None and error.message != "Fitting cancelled":
                LOG.warning("Predict parameters failed (%s): %s\n%s", error.exception_type or error.code,
                            error.message, error.traceback)
            raise RuntimeError(error.message if error else "Predict parameters failed")
        return result.value

    def _predict_progress(self, progress) -> None:
        if self._stop_event.is_set():
            self._cancel_predict()
        else:
            self._predict_report(progress)

    def _cancel_predict(self) -> None:
        if self._predict_job_id:
            self.view_model.context.jobs.cancel(self._predict_job_id)

    def _predict_finished(self, value, curve, selection) -> None:
        self._predict_job_id = None
        self._predict_clock.stop()
        self._predict_partial = None
        solutions = as_solutions(predicted(value))
        if curve is None or curve is self.session.curve:
            self.session.predict_run = {"stopped": False, "network": value.get("network"),
                                        "seconds": value.get("seconds"), "reproduction": value.get("reproduction")}
        self._solutions_found(solutions, curve, selection)

    def posterior_lines(self) -> list[str]:
        """Fit details of the solution in Model: its refined values with their 68 % intervals, its probability, the
        points of its χ²ᵣ and its flags (``predict_texts.solution_lines``)."""
        solution = next((s for s in self.session.solutions if s.model == self.session.model and s.intervals), None)
        if solution is None:
            return []
        return solution_lines(solution, self.session.solutions, len(self.session.model.components) > 1)


__all__ = ["MISSING", "PredictPageMixin"]
