"""The Fit step: one button, four methods, progress and Stop (the work runs in the background).

* Refine the current values / Search the ranges, then refine — the fitting engine on the model;
* Find the particle shape (no AI) — the quick physical fit (sphere, random and vertical cylinder
  from several starts, injected by the composition root); its solutions go to Results, the best
  into Model;
* AI proposal — 1D Predict (V5) with its numerical correction; its solutions likewise.

A fit replaces the model (Undo brings the previous one back). A stopped fit keeps the best values
so far.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from src.gimap.app.presentation.i18n import tr

from ...application import CandidateGenerationRequest
from ...application.single_fit import NM_PER_A, fit, model_from_solution, residuals
from ...application.workflow_v5 import bundled_workflow, default_options
from .session import Solution, model_label

FAMILY_IDS = {"sphere": 1, "cylinder": 2, "vertical_cylinder": 3}
METHOD_NAMES = {"local": "Refine", "global": "Search the ranges", "shapes": "Find the particle shape",
                "ai": "AI proposal"}


class _Progress(QObject):
    changed = pyqtSignal(float, str)


class FitRunMixin:
    """Needs the page's widgets, ``session``, ``tasks``, ``quick_fit`` and ``view_model``."""

    def setup_run(self) -> None:
        self._progress = _Progress(self)
        self._progress.changed.connect(self._show_progress)
        self._stop_event = threading.Event()
        self._running = None
        self.fit_button.clicked.connect(self.run_fit)
        self.fit_step_button.clicked.connect(self.run_fit)
        self.stop_button.clicked.connect(self.stop_fit)
        self.stop_step_button.clicked.connect(self.stop_fit)
        self.method_group.buttonToggled.connect(lambda _button, _on: self._method_chosen())
        self.method_buttons["shapes"].setEnabled(self.quick_fit is not None)
        self._method_chosen()

    def method(self) -> str:
        return next((key for key, button in self.method_buttons.items() if button.isChecked()), "local")

    def _method_chosen(self) -> None:
        method = self.method()
        self.families_row.setVisible(method == "shapes")
        self.fit_step_button.setText(tr(METHOD_NAMES[method]))
        self.fit_step_button.setToolTip(self.method_buttons[method].toolTip())
        self._remember(method=method)

    def fit_running(self) -> bool:
        return self._running is not None

    # -- run ------------------------------------------------------------------------------

    def run_fit(self) -> None:
        if self.fit_running():
            return
        data = self._data()
        if self.session.curve is None:
            self._status(tr("Open a curve first."), "warning")
            return
        if data is None or data.q.size < 3:
            self._status(tr("Too few points in the fitting range: widen it in the Curve step."), "warning")
            return
        method = self.method()
        if method == "shapes" and self.quick_fit is None:
            self._status(tr("Finding the particle shape is not available here: choose another method."), "warning")
            return
        if method in ("local", "global") and not self.session.model.components:
            self._status(tr("Add a particle in Model, or use Find the particle shape."), "warning")
            return
        self._stop_event.clear()
        self._running = method
        self._busy(True)
        budget = self.budget_spin.value() or None
        model = self.session.model
        selection = self.session.selection()  # the points this fit is made on (a later change makes it stale)
        curve = self.session.curve  # the curve it is of: when another one is open by the end, nothing is kept
        failed = lambda message, details="": self._failed(message, details, curve)  # noqa: E731
        self._log(tr("{method}: {points} points, {free} free parameters").format(
            method=tr(METHOD_NAMES[method]), points=data.q.size,
            free=sum(1 for _path, parameter in model.parameters() if parameter.free)))
        if method in ("local", "global"):
            job = lambda: fit(model, data, method=method, max_evaluations=budget,  # noqa: E731
                              progress=self._progress.changed.emit, stop=self._stop_event.is_set)
            self.tasks.submit("fit", job, on_done=lambda result: self._fitted(result, selection, curve), on_error=failed)
        elif method == "shapes":
            families, edited = self.families_combo.currentData(), self.session.edited  # read here, not in the worker
            self.tasks.submit("fit", lambda: self._find_shapes(data, model, families, edited),
                              on_done=lambda solutions: self._solutions_found(solutions, curve, selection),
                              on_error=failed)
        else:
            self.tasks.submit("fit", lambda: self._ai_solutions(data, model),
                              on_done=lambda solutions: self._solutions_found(solutions, curve, selection),
                              on_error=failed)

    def stop_fit(self) -> None:
        if not self.fit_running():
            return
        self._halt_fit()
        self.fit_progress_text.setText(tr("Stopping … the best values so far are kept."))

    def _halt_fit(self) -> None:
        """Ask the running fit to stop (also when another curve is opened: its result is then not kept)."""
        if not self.fit_running():
            return
        self._stop_event.set()
        if self._running == "ai":
            try:
                self.view_model.cancel_ai_candidates()
            except (AttributeError, RuntimeError):
                pass

    def _busy(self, on: bool) -> None:
        for widget in (self.stop_button, self.stop_step_button, self.fit_progress, self.status_progress):
            widget.setVisible(on)
        for widget in (self.fit_progress, self.status_progress):
            widget.setValue(0)
        if not on:
            self.fit_progress_text.setText("")
        self._refresh_steps()

    def _show_progress(self, fraction: float, text: str) -> None:
        if not self.fit_running():
            return
        value = int(100 * max(0.0, min(1.0, float(fraction))))
        self.fit_progress.setValue(value)
        self.status_progress.setValue(value)
        self.fit_progress_text.setText(text)
        self.status_label.setText(text)
        self._status_text = None  # the fit's own progress line, not one to make again

    # -- outcomes ---------------------------------------------------------------------------

    def _finish(self) -> None:
        self._running = None
        self._busy(False)

    def _of_another_curve(self, curve) -> bool:
        """The fit ended after another curve was opened: it ends here, and nothing of it is kept (its
        values, χ² and solutions belong to the curve before)."""
        if curve is None or curve is self.session.curve:
            return False
        self._finish()
        self._status(lambda: tr("The fit of {name} ended after another curve was opened; its result was not kept.").format(
            name=curve.name), "warning")
        return True

    def _failed(self, message: str, _details: str = "", curve=None) -> None:
        if self._of_another_curve(curve):
            return
        self._finish()
        self._status(lambda: tr("The fit failed: {reason}").format(reason=message), "error")

    def _fitted(self, result, selection=None, curve=None) -> None:
        if self._of_another_curve(curve):
            return
        self._finish()
        self.session.result = result
        self.session.result_selection = self.session.selection() if selection is None else selection
        self.session.set_model(result.model)
        self._model_changed(editor=True)
        if result.stopped:
            self._status(lambda: tr("Stopped: the best values so far are kept ({quality}).").format(
                quality=self._quality_text(result)), "warning")
        else:
            self._status(lambda: tr("Fitted: {quality}.").format(quality=self._quality_text(result)),
                         "ok" if result.converged else "warning")
        self.show_step("results")

    def _solutions_found(self, solutions, curve=None, selection=None) -> None:
        if self._of_another_curve(curve):
            return
        method = self._running
        self._finish()
        if not solutions:
            self._status(tr("No solution: the curve may be too short or too noisy for these families."), "warning")
            return
        self.session.solutions = list(solutions)
        # The points their χ² is of: those when the search started (the band may be dragged meanwhile).
        self.session.solutions_selection = self.session.selection() if selection is None else selection
        best = solutions[0]
        self.session.result = None
        self.set_model(best.model)
        self.method_buttons["local"].setChecked(True)  # next: refine the chosen solution here
        self._status(lambda: tr("{count} solutions; the best, {model} (χ²ᵣ {chi2}), is in Model. Fit again to refine "
                                "it, or choose another in Results.").format(count=len(solutions), model=best.label,
                                                                         chi2=f"{best.chi2:.3g}"), "ok")
        self._log(tr("{method}: {count} solutions").format(method=tr(METHOD_NAMES.get(method, "")), count=len(solutions)))
        self.show_step("results")

    # -- the solutions of the other methods (background thread) -----------------------------

    def _as_solutions(self, rows, data, base, source: str) -> list[Solution]:
        solutions = []
        for row in rows or ():
            try:
                model, deviation = model_from_solution(row, base)
            except (ValueError, KeyError, TypeError):
                continue
            values = residuals(model, data)
            free = sum(1 for _path, parameter in model.parameters() if parameter.free)
            dof = max(1, data.q.size - free)
            solutions.append(Solution(model_label(model), model, float(values @ values / dof), source, deviation))
        return sorted(solutions, key=lambda item: item.chi2 if math.isfinite(item.chi2) else math.inf)

    def _find_shapes(self, data, base, families: str, edited: bool) -> list[Solution]:
        components = ()
        if families == "model" and base.components:
            components = tuple(FAMILY_IDS[component.family] for component in base.components)
        distance = None
        if edited and base.components and base.components[0].structure:
            distance = base.components[0].value("D")

        def report(done, total, message="", _extra=None):
            self._progress.changed.emit(float(done) / max(1.0, float(total)), str(message))

        rows = self.quick_fit(data.q / NM_PER_A, data.intensity, data.sigma, components=components,
                              distance_nm=distance, report=report, cancelled=self._stop_event.is_set)
        return self._as_solutions(rows, data, base, "shape search")

    def _ai_solutions(self, data, base) -> list[Solution]:
        from ..bindings.ai_job_execution import workflow_options_for_mode

        catalog = self.view_model.storage.ai_catalog
        if catalog is None:
            raise RuntimeError(tr("1D Predict is not available in this installation"))
        profile = catalog.profile(catalog.default_profile_name)
        sigma = data.sigma
        if sigma is None:
            floor = max(float(np.max(np.abs(data.intensity))) * 0.001, 1e-12)
            sigma = np.hypot(0.10 * np.abs(data.intensity), floor)
        options = workflow_options_for_mode({**default_options(), "side": "positive"}, "full")
        output = Path.cwd() / "AI_Fitting_Output" / ("v5_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        request = CandidateGenerationRequest(
            model_path=bundled_workflow(), output_dir=output, q=data.q, intensity=data.intensity, sigma=sigma,
            profile=profile.to_dict(),
            constraints={"workflow_v5": options, "sigma_estimated": data.sigma is None, "observation_metadata": {}},
        )
        result = self.view_model.run_ai_candidates(
            request, refine=False, on_progress=lambda p: self._progress.changed.emit(float(p.fraction), str(p.message)))
        if result is None:
            raise RuntimeError(self.view_model.state.error_message or tr("1D Predict failed"))
        return self._as_solutions(result.candidates, data, base, "1D Predict")


__all__ = ["FitRunMixin"]
