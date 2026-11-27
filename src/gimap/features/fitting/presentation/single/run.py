"""The Fit step: one method — Predict parameters (``predict.py``), Refine or Search — one button, progress and
Stop (the work runs in the background).

* Refine / Search — the fitting engine under Model's hard limits, with boundary diagnostics;
* Predict parameters — the proposal network and its posterior (``predict.py``); its solutions go to Results, the most
  probable into Model.

A fit replaces the model (Undo brings the previous one back). A stopped fit of the engine keeps the best
values so far; a stopped Predict keeps the solutions it finished, marked incomplete (``predict_progress.py``).
A fit that starts from a solution (Refine These Values) says so: ``FitSession.refined_from``.

While a fit runs, the bars show how far it is — busy (no number) until there is something to count — and
the line under them what it is doing, in the interface language (``progress_line``; Predict's stages:
``predict_progress.stage_line``).
"""

from __future__ import annotations

import re
import threading
from typing import Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import (
    BOTH_USES_MEAN,
    ONE_CURVE_METHODS,
    PROGRESS_LINE,
    RESTART_TOLERANCE,
    fit_resolving,
    stopped_result,
)
from .method_hint import fit_error_text
from .predict_progress import stage_line
from .predict_texts import count_solutions, points_text
from .solution_texts import numbered_labels

METHOD_NAMES = {"local": "Refine", "global": "Search the ranges", "predict": "Predict parameters"}
RESTARTED = "The least squares went on from its own result {count} times (each time χ²ᵣ fell by more than {share} %)."
"""The Run Log after a manual fit that ran again (``single_fit.converged_again``)."""
_ENGINE_LINE = re.compile("".join(
    re.escape(part) if index % 2 == 0 else f"(?P<{part}>.+?)"
    for index, part in enumerate(re.split(r"\{(\w+)\}", PROGRESS_LINE))), re.DOTALL)


def progress_line(what) -> str:
    """What a running fit is doing, in the interface language. ``what``: the engine's English line
    (``PROGRESS_LINE``: its phase is translated, its numbers stay) or any other line — shown as it comes unless the
    table knows it."""
    if isinstance(what, dict):  # a stage of Predict parameters (``predict_progress``)
        return stage_line(what)
    found = _ENGINE_LINE.fullmatch(str(what))
    if found is None:
        return tr(str(what))
    return trf(PROGRESS_LINE, **{**found.groupdict(), "phase": tr(found["phase"])})


class _Progress(QObject):
    changed = pyqtSignal(float, object)
    """How far (0–1) and what ``progress_line`` takes; sent from the fit's thread, shown on the window's."""


class FitRunMixin:
    """Needs the page's widgets, ``session``, ``tasks`` and ``view_model``."""

    def setup_run(self) -> None:
        self._progress = _Progress(self)
        self._progress.changed.connect(self._show_progress)
        self._stop_event = threading.Event()
        self._running = None
        self._progress_shown = None
        self._fit_notes = None
        """What the last Refine / Search did by itself and what is still at an edge (``fit_resolving``)."""
        self.fit_button.clicked.connect(lambda: self.run_fit())
        self.fit_step_button.clicked.connect(lambda: self.run_fit())
        self.stop_button.clicked.connect(self.stop_fit)
        self.stop_step_button.clicked.connect(self.stop_fit)
        self.method_combo.currentIndexChanged.connect(lambda _index: self._method_chosen())
        self._method_chosen(remember=False)

    def method(self) -> str:
        return self.method_combo.currentData() or "local"

    def set_method(self, method, *, remember: bool = True) -> None:
        """Choose ``method`` (a key of ``METHOD_NAMES``) in the Fit step; anything else leaves the choice as it is."""
        index = self.method_combo.findData(method)
        if index < 0:
            return
        if index == self.method_combo.currentIndex():
            self._method_chosen(remember=remember)
            return
        self.method_combo.blockSignals(True)
        self.method_combo.setCurrentIndex(index)
        self.method_combo.blockSignals(False)
        self._method_chosen(remember=remember)

    def _method_chosen(self, *, remember=True) -> None:
        method = self.method()
        self.predict_row.setVisible(method == "predict")
        self._show_method_buttons()
        self.fit_step_button.setToolTip(tr(self.method_combo.itemData(self.method_combo.currentIndex(), Qt.ToolTipRole)
                                           or ""))
        self._fill_curve_card()
        self._refresh_method_hint()
        if remember:
            self._remember(method=method)

    def _show_method_buttons(self) -> None:
        name = tr(METHOD_NAMES[self.method()])
        self.fit_button.setText(name)
        self.fit_step_button.setText(name)

    def fit_running(self) -> bool:
        return self._running is not None

    # -- run ------------------------------------------------------------------------------

    def run_fit(self, method: Optional[str] = None) -> None:
        """``method``: one of ``METHOD_NAMES``; the one chosen in the Fit step if None."""
        if self.fit_running():
            return
        if self.session.curve is None:
            self._status(tr("Open a curve first."), "warning")
            return
        method = method or self.method()
        if method not in METHOD_NAMES:
            method = self.method()
        if method == "predict" and not self.predict_available():
            from ...application.single_predict import network_file
            from .predict import MISSING
            self._status(trf(MISSING, name=network_file()), "warning")
            return
        if method in ("local", "global") and not self.session.model.components:
            self._status(tr("Add a particle in Model, or use Predict parameters."), "warning")
            return
        if method in ONE_CURVE_METHODS and self.session.side == "both" and self.session.curve.signed:
            # These methods take one measurement per |q|. The page goes over to the mean of the halves and says
            # so: what it shows, the points a result is of and the record of an export are the fitted points.
            self.set_side("mean")
            self._status(lambda: tr(BOTH_USES_MEAN), "warning")
        try:
            data = self.session.data(predict=method == "predict")
        except ValueError as exc:
            self._status(fit_error_text(str(exc)), "warning")
            return
        if data is None or data.q.size < 3:
            self._status(tr("Too few points in the fitting range: widen it in the Curve step."), "warning")
            return
        # Where the fit starts is taken before the tail's ranges are tied: ranges set from the curve do not make a
        # solution "edited".
        self._refining_from = self._start_of_fit(self.session.model, method) if method in ("local", "global") else None
        if method in ("local", "global"):
            self.tie_tail_before_fit(data)  # generic beam-tail ranges: set from this curve, in Model (tail_ranges.py)
        self._stop_event.clear()
        self._running = self._last_method = method
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
            job = lambda: fit_resolving(model, data, method=method, max_evaluations=budget,  # noqa: E731
                                        progress=self._progress.changed.emit, stop=self._stop_event.is_set,
                                        resolve=True)
            self.tasks.submit("fit", job, on_done=lambda done: self._fitted(*done, selection=selection, curve=curve),
                              on_error=failed)
        elif method == "predict":
            try:
                self._start_predict(data, model, selection, curve)
            except (ValueError, TypeError, RuntimeError) as exc:
                failed(str(exc))

    def _start_of_fit(self, model, method: str):
        """Where a Refine / Search starts (``FitSession.refined_from``): a solution as it is, a fit of one, or one
        edited in Model; ``None`` otherwise."""
        session = self.session
        number = next((index for index, solution in enumerate(session.solutions) if solution.model == model), None)
        if number is not None:
            return {"solution": number + 1, "method": method, "edited": False}
        if session.refined_from and session.result is not None and session.result.model == model:
            return {**session.refined_from, "method": method}  # a fit of a fit of the solution
        if session.model_from is not None and session.model_from < len(session.solutions):
            return {"solution": session.model_from + 1, "method": method, "edited": True}
        return None

    def stop_fit(self) -> None:
        if not self.fit_running():
            return
        keeps = self._running in ("local", "global")  # Predict keeps the solutions it finished (predict_progress)
        self._halt_fit()
        self.fit_progress_text.setText(tr("Stopping … the best values so far are kept.") if keeps
                                       else self.stopping_line())

    def _halt_fit(self) -> None:
        """Ask the running fit to stop (also when another curve is opened: its result is then not kept)."""
        if not self.fit_running():
            return
        self._stop_event.set()
        if self._running == "predict":
            self._cancel_predict()

    def _busy(self, on: bool) -> None:
        for widget in (self.stop_button, self.stop_step_button, self.fit_progress, self.status_progress):
            widget.setVisible(on)
        self._progress_shown = None
        self._show_bars(0, busy=on)
        if not on:
            self.fit_progress_text.setText("")
        self._refresh_steps()
        self.refresh_equivalent_selector()

    def _show_bars(self, percent: int, *, busy: bool) -> None:
        """``busy``: a moving bar without a number — the fit runs, but there is nothing to count yet (its
        first evaluations, the stages of Predict)."""
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
        if hasattr(self, "clear_predict_preview"):
            self.clear_predict_preview()  # the fits drawn while Predict ran

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
        stopped = self._stop_event.is_set() and self._running == "predict"
        if stopped:
            self._predict_stopped(curve, self._predict_selection)  # what it finished, marked incomplete, or nothing
        else:
            self._finish()
            self._status(lambda: tr("The fit failed: {reason}").format(reason=fit_error_text(message)), "error")

    def _fitted(self, result, actions=(), remaining=(), *, selection=None, curve=None) -> None:
        if self._of_another_curve(curve):
            return
        if self._stop_event.is_set():
            result, remaining = stopped_result(result), ()
        self._finish()
        self._fit_notes = dict(result=result, actions=list(actions), remaining=list(remaining))
        self.session.result = result
        self.session.result_selection = self.session.selection() if selection is None else selection
        self.session.refined_from = getattr(self, "_refining_from", None)
        self.session.set_model(result.model, label=("the fit ({method})", {"method": METHOD_NAMES[result.method]
                                                                             if result.method in METHOD_NAMES else "Refine"}))
        self._model_changed(editor=True)
        if getattr(result, "restarts", 0):
            self._log(trf(RESTARTED, count=result.restarts, share=f"{100 * RESTART_TOLERANCE:g}"))
        if result.stopped:
            self._status(lambda: tr("Stopped: the best values so far are kept ({quality}).").format(
                quality=self._quality_text(result)), "warning")
        else:
            self._status(lambda: trf("Fit finished: {state}.", state=self._state_text(result)),  # the rest: Results
                         "ok" if result.converged else "warning")
        self.show_step("results")

    def _solutions_found(self, solutions, curve=None, selection=None, message=None) -> None:
        """``message``: the status line (a function), when not the one of a search that finished."""
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
        self.session.refined_from = None
        self.set_model(best.model, label=("Predict parameters: solution 1 in Model", {}))
        self.session.model_from = 0
        self._refresh_steps()  # also when the best is the model already (a run after one that was stopped)
        self.render_results()
        self.set_method("local")  # next: refine the chosen solution here
        def found() -> str:  # the same χ²ᵣ as the table and Fit details, with the points it is of
            over = points_text(best.points)
            chi2 = trf("{value} over {points}", value=f"{best.chi2:.3g}", points=over) if over else f"{best.chi2:.3g}"
            return trf("{solutions}; the best, {model} (χ²ᵣ {chi2}), is in Model. Fit again to refine it, or choose "
                       "another in Results.", solutions=count_solutions(len(solutions)),
                       model=numbered_labels(solutions)[0], chi2=chi2)

        self._status(message or found, "warning" if message else "ok")
        self._log(trf("{method}: {solutions}", method=tr(METHOD_NAMES.get(method, "")),
                      solutions=count_solutions(len(solutions))))
        self.show_step("results")


__all__ = ["FitRunMixin", "progress_line"]
