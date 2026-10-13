"""The Fit step: one button, four methods, progress and Stop (the work runs in the background).

* Refine the current values / Search the ranges, then refine — the fitting engine on the model;
* Find the particle shape (no AI) — the quick physical fit (sphere, random and vertical cylinder
  from several starts, injected by the composition root); its solutions go to Results, the best
  into Model;
* AI proposal — 1D Predict (V5) with its numerical correction; its solutions likewise.

A fit replaces the model (Undo brings the previous one back). A stopped fit of the engine keeps the best
values so far; a stopped search for solutions (the last two methods) brings none.

While a fit runs, the bars show how far it is — busy (no number) until there is something to count — and
the line under them what it is doing, in the interface language (``progress_line``). The shape search
counts its starts and, inside one, the evaluations of its budget (``start_fraction``).
"""

from __future__ import annotations

import math
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from src.gimap.app.presentation.i18n import tr, trf

from ...application import CandidateGenerationRequest
from ...application.single_fit import FAMILIES, NM_PER_A, PROGRESS_LINE, fit, model_from_solution, residuals
from ...application.workflow_v5 import bundled_workflow, default_options
from .session import Solution

FAMILY_IDS = {"sphere": 1, "cylinder": 2, "vertical_cylinder": 3}
METHOD_NAMES = {"local": "Refine", "global": "Search the ranges", "shapes": "Find the particle shape",
                "ai": "AI proposal"}
START, START_OF = "Trying start {n} of {total}", "Trying start {n} of {total}: {model}"
"""One start of the shape search (each family is tried from several starting sizes)."""
EVALUATED = "{line} — {evaluations} evaluations"
"""… and how many times this start has evaluated the model so far (one start can take most of the search)."""
HEARTBEAT = 0.25
"""Seconds between two lines of one start (as the fitting engine reports)."""
_ENGINE_LINE = re.compile("".join(
    re.escape(part) if index % 2 == 0 else f"(?P<{part}>.+?)"
    for index, part in enumerate(re.split(r"\{(\w+)\}", PROGRESS_LINE))), re.DOTALL)


def progress_line(what) -> str:
    """What a running fit is doing, in the interface language. ``what``: the engine's English line
    (``PROGRESS_LINE``: its phase is translated, its numbers stay), the details of one start of the shape
    search (``{"start": 2, "starts": 6, "types": [3]}``, with ``evaluations`` once it has some), or any
    other line — shown as it comes unless the table knows it."""
    if isinstance(what, dict):
        names = {number: FAMILIES[key][0] for key, number in FAMILY_IDS.items()}
        model = " + ".join(tr(names[number]) for number in what.get("types") or () if number in names)
        line = trf(START_OF if model else START, n=what.get("start", 1), total=what["starts"], model=model)
        return trf(EVALUATED, line=line, evaluations=what["evaluations"]) if what.get("evaluations") else line
    found = _ENGINE_LINE.fullmatch(str(what))
    if found is None:
        return tr(str(what))
    return trf(PROGRESS_LINE, **{**found.groupdict(), "phase": tr(found["phase"])})


def start_fraction(start: dict) -> float:
    """How far a search of several starts is: the starts done, plus the share of its evaluation budget this
    one has used (a start that converges early ends before its share is full, as a fit of the engine does;
    never the whole of it while it still runs)."""
    used = min(0.99, float(start.get("evaluations") or 0) / max(1.0, float(start.get("budget") or 1)))
    return (float(start.get("start", 1)) - 1.0 + used) / float(start["starts"])


class _Progress(QObject):
    changed = pyqtSignal(float, object)
    """How far (0–1) and what ``progress_line`` takes; sent from the fit's thread, shown on the window's."""


class FitRunMixin:
    """Needs the page's widgets, ``session``, ``tasks``, ``quick_fit`` and ``view_model``."""

    def setup_run(self) -> None:
        self._progress = _Progress(self)
        self._progress.changed.connect(self._show_progress)
        self._stop_event = threading.Event()
        self._running = None
        self._progress_shown = None
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
        keeps = self._running in ("local", "global")  # a search for solutions keeps nothing of a run cut short
        self._halt_fit()
        self.fit_progress_text.setText(tr("Stopping … the best values so far are kept.") if keeps else tr("Stopping…"))

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
        self._progress_shown = None
        self._show_bars(0, busy=on)
        if not on:
            self.fit_progress_text.setText("")
        self._refresh_steps()

    def _show_bars(self, percent: int, *, busy: bool) -> None:
        """``busy``: a moving bar without a number — the fit runs, but there is nothing to count yet (its
        first evaluations, the first start of the shape search, the stages of the AI proposal)."""
        for bar in (self.fit_progress, self.status_progress):
            bar.setRange(0, 0 if busy else 100)
            bar.setValue(percent)

    def _show_progress(self, fraction: float, what) -> None:
        if not self.fit_running() or self._stop_event.is_set():  # a line sent before Stop: not over “Stopping…”
            return
        self._progress_shown = (fraction, what)
        percent = int(100 * max(0.0, min(1.0, float(fraction))))
        self._show_bars(percent, busy=percent == 0)
        text = progress_line(what)
        self.fit_progress_text.setText(text)
        self.status_label.setText(text)
        self._status_text = None  # the fit's own progress line: ``refresh_progress`` makes it again

    def refresh_progress(self) -> None:
        """After a switch of the interface language: the line of the fit that runs, said again (not over
        “Stopping…”)."""
        if self.fit_running() and self._progress_shown is not None and not self._stop_event.is_set():
            self._show_progress(*self._progress_shown)

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
        # Stop during a search for solutions ends as an error of its fitter (“Fitting cancelled”): not a failure.
        stopped = self._stop_event.is_set() and self._running in ("shapes", "ai")
        self._finish()
        if stopped:
            self._status(lambda: tr("Stopped before the search finished: no new solutions."), "warning")
        else:
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
            solutions.append(Solution(model, float(values @ values / dof), source, deviation))
        return sorted(solutions, key=lambda item: item.chi2 if math.isfinite(item.chi2) else math.inf)

    def _find_shapes(self, data, base, families: str, edited: bool) -> list[Solution]:
        components = ()
        if families == "model" and base.components:
            components = tuple(FAMILY_IDS[component.family] for component in base.components)
        distance = None
        if edited and base.components and base.components[0].structure:
            distance = base.components[0].value("D")

        said = [0.0]  # when the last line of a start was sent

        def report(done, total, message="", extra=None):
            # The quick fit counts its starts in the details: (done, total) stay (0, 1) there, because the
            # same callback counts the files of a 1D Predict job (``experimental_fit.fit_candidates``).
            start = extra if isinstance(extra, dict) and extra.get("starts") else None
            if start is None:
                self._progress.changed.emit(float(done) / max(1.0, float(total)), str(message))
            else:
                said[0] = time.monotonic()
                self._progress.changed.emit(start_fraction(start), dict(start))

        def evaluated(start):
            """Every evaluation of a start: a line now and then, so a long start is seen to be at work."""
            now = time.monotonic()
            if now - said[0] >= HEARTBEAT and isinstance(start, dict) and start.get("starts"):
                said[0] = now
                self._progress.changed.emit(start_fraction(start), dict(start))

        rows = self.quick_fit(data.q / NM_PER_A, data.intensity, data.sigma, components=components,
                              distance_nm=distance, report=report, cancelled=self._stop_event.is_set,
                              evaluated=evaluated)
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


__all__ = ["FitRunMixin", "progress_line", "start_fraction"]
