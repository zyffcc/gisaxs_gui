"""Predict parameters on the single-curve page: GIMaP's proposal network and the posterior it starts, in a job process
(``application/single_predict.py``, ``infrastructure/adapters/posterior_job.py``).

The particles to try (those of Model, or a choice of families for the network to choose among) are shown under the
method; every solution is refined by least squares (``domain.posterior.refine.predict``, one path). The solutions go
to Results with their probability, the most probable into Model; the posterior's 68 % intervals of the solution in
Model are listed in Fit details (``posterior_lines``).
"""

from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_predict import PARTICLES, network_available, predict_request, predicted
from .session import Solution

UNITS = {"R": "nm", "h": "nm", "D": "nm", "tail_width": "nm⁻¹", "q_resolution": "nm⁻¹"}
SHOWN = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn")
LABELS = {"R": "R", "sigma_R": "σR/R", "h": "h", "sigma_h": "σh/h", "D": "D", "sigma_nn": "σnn/D"}
FLAG_TEXTS = {
    "unconstrained": "the curve says little about {name} (its 68 % interval spans most of the prior)",
    "at_bound": "{name} stopped at a bound of its range",
    "correlated": "{name} are strongly correlated",
    "low_ess": "few effective samples: the intervals are rough",
    "bic": "the posterior's sampling did not converge: the solutions are compared by χ² and their number of values (BIC)",
    "large": "{name}: its Guinier region lies at or below the curve's lowest q, so the curve shows only its decay, which roughness or a rise at low q produce as well",
    "small": "{name}: smaller than the curve resolves, it acts as a smooth background",
}
MISSING = "Predict parameters needs GIMaP's proposal network, which is missing in this installation ({name})."


class PredictPageMixin:
    def setup_predict(self) -> None:
        self._predict_job_id = None
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

    def _start_predict(self, data, model, selection, curve) -> None:
        settings = self.predict_settings()
        particles = settings["particles"]
        if particles == "model" and not model.components:
            particles = "spheres_cylinders"  # nothing in Model yet: the network chooses
        request = predict_request(data, model, particles)
        self._predict_job_id = request.job_id
        self.tasks.submit("fit", lambda: self._run_predict_job(request),
                          on_done=lambda value: self._predict_finished(value, curve, selection),
                          on_error=lambda message, details="": self._failed(message, details, curve))

    def _run_predict_job(self, request):
        if self._stop_event.is_set():
            raise RuntimeError("Fitting cancelled")
        result = self.view_model.context.jobs.run(request, self._predict_progress)
        if not result.succeeded:
            raise RuntimeError(result.error.message if result.error else "Predict parameters failed")
        return result.value

    def _predict_progress(self, progress) -> None:
        if self._stop_event.is_set():
            self._cancel_predict()
        else:
            self._progress.changed.emit(float(progress.fraction), str(progress.message))

    def _cancel_predict(self) -> None:
        if self._predict_job_id:
            self.view_model.context.jobs.cancel(self._predict_job_id)

    def _predict_finished(self, value, curve, selection) -> None:
        self._predict_job_id = None
        solutions = [Solution(p.model, p.chi2, f"Predict parameters · {p.name}", 0.0, probability=p.probability,
                              intervals=tuple(p.intervals.items()), flags=p.flags) for p in predicted(value)]
        self._solutions_found(solutions, curve, selection)

    def posterior_lines(self) -> list[str]:
        """The posterior of the solution in Model: each value's median and 68 % interval, and its flags."""
        solution = next((s for s in self.session.solutions if s.model == self.session.model and s.intervals), None)
        if solution is None:
            return []
        parts = []
        for name, values in solution.intervals:
            key, _, slot = name.rpartition("_")
            if key not in SHOWN or values[1] is None:
                continue
            label = LABELS[key] + (slot if len(self.session.model.components) > 1 else "")
            unit = UNITS.get(key, "")
            parts.append(f"{label} {values[1]:.3g} [{values[2]:.3g}, {values[3]:.3g}]{(' ' + unit) if unit else ''}")
        lines = [trf("Posterior of this solution (median [68 %]): {values}; probability {p}.",
                     values="; ".join(parts), p=f"{solution.probability:.2f}")] if parts else []
        for flag in solution.flags:
            kind, _, name = flag.partition(":")
            if kind in FLAG_TEXTS:
                lines.append("⚠ " + trf(FLAG_TEXTS[kind], name=name.replace("~", " ↔ ")))
        return lines


__all__ = ["MISSING", "PredictPageMixin"]
