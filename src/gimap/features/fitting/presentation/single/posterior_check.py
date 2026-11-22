"""The Posterior check tab of Results ▸ Uncertainty and Report (``views/posterior_check_view.py``; application:
``application/posterior_check.py``; Physics and Methods 9.19).

A short MCMC run on the posterior of the solution in Model (Predict parameters: its own posterior, the model error
and the scales integrated out) or of the last Refine or Search (the fit's likelihood). It runs in the background with
its progress and Stop (the steps done are kept and shown, marked as stopped). The table sets the run's median and
68 % / 95 % intervals beside the intervals checked; the verdict says whether the run is long enough (autocorrelation
time and R̂). A check only: it never changes the model, the Results or Predict's values.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from ...application.posterior_check import CHANGED, CHECK_ONLY, NOTHING, record, rows, run, target_for, trace, verdict
from ..views.posterior_check_view import build_check_tab

WHAT = {"predict": ("Checks the solution of Predict parameters in Model: its posterior (the model error and the scales "
                    "integrated out, the network's prior), against the median and intervals Predict gave."),
        "fit": ("Checks the last Refine or Search: its likelihood (Σr² scaled by its χ²ᵣ, the scales solved), "
                "against its value ± 1σ (± 1.96σ for 95 %).")}
RUNNING = "Running {steps} steps of {walkers} walkers …"
DONE = "{steps} steps of {walkers} walkers in {seconds:.1f} s, acceptance {acceptance:.2f}."
STOPPED = "Stopped after {steps} steps: the steps done are shown."
DIFFERS = "⚠ The run's 68 % interval does not overlap the one checked: {names}."
TRACE_WALKER, TRACE_BAND, TRACE_MEDIAN = "#9aa5b1", "#e07b00", "#3b82d6"
"""Data colours of the trace (readable in both themes): single walkers grey, the 16–84 % band orange, the median blue."""


class _CheckProgress(QObject):
    changed = pyqtSignal(int, int)


def _interval(low: float, high: float) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return "—"
    return f"{low:.5g} – {high:.5g}"


class PosteriorCheckMixin:
    """Needs ``uncertainty_tabs``, ``session``, ``tasks``, ``_data``, ``_current_result`` and ``_save_path``."""

    def setup_posterior_check(self) -> None:
        build_check_tab(self, self.uncertainty_tabs)
        self._check_stop = threading.Event()
        self._check_signal = _CheckProgress(self)
        self._check_signal.changed.connect(self._check_progress)
        self._check = None
        self._disposers = getattr(self, "_disposers", [])
        self.check_run.clicked.connect(self.run_check)
        self.check_stop.clicked.connect(self._check_stop.set)
        self.check_trace_value.currentIndexChanged.connect(lambda _i: self._plot_trace())
        self.check_save.clicked.connect(lambda: self.save_check_dialog())
        self.uncertainty_tabs.currentChanged.connect(lambda _i: self.render_check())
        self.uncertainty_button.clicked.connect(lambda: self.render_check())
        self.render_check()

    def add_disposer(self, function) -> None:
        """Something to stop when the page is disposed (the window of a fit of several curves)."""
        self._disposers = getattr(self, "_disposers", [])
        self._disposers.append(function)

    def dispose_checks(self) -> None:
        if hasattr(self, "_check_stop"):
            self._check_stop.set()
        for function in getattr(self, "_disposers", []):
            try:
                function()
            except RuntimeError:
                continue

    def _checked(self):
        """What a run would check now (``application.posterior_check.target_for``)."""
        return target_for(self.session.model, self._data(), self.session.solutions, self._current_result())

    def render_check(self) -> None:
        busy = self.tasks.is_busy()
        try:
            checked = self._checked()
        except ValueError as exc:
            self.check_what.setText(tr(str(exc)) if str(exc) == NOTHING else str(exc))
            self.check_run.setEnabled(False)
        else:
            self.check_what.setText(tr(WHAT[checked.kind]) + " " + tr(CHECK_ONLY))
            self.check_run.setEnabled(not busy)
        if self._check is not None and self._check[2] != self.session.model:
            self.check_verdict.setText(tr(CHANGED) + " " + self._verdict_text())
        self.check_save.setEnabled(self._check is not None)

    def run_check(self) -> None:
        if self.tasks.is_busy():
            return
        try:
            checked = self._checked()
        except ValueError as exc:
            self.check_what.setText(tr(str(exc)))
            return
        steps, walkers, seed = self.check_steps.value(), self.check_walkers.value() or None, self.check_seed.value()
        if walkers is not None and walkers % 2:
            walkers += 1
        self._check_stop.clear()
        self.check_run.setEnabled(False)
        self.check_stop.show()
        self.check_progress.setRange(0, steps)
        self.check_progress.setValue(0)
        self.check_progress.show()
        self.check_verdict.setText(trf(RUNNING, steps=steps, walkers=walkers or tr("auto")))
        model, signal = self.session.model, self._check_signal
        job = lambda: run(checked, steps=steps, walkers=walkers, seed=seed,  # noqa: E731
                          progress=signal.changed.emit, stop=self._check_stop.is_set)
        self.tasks.submit("check", job, on_done=lambda result: self._check_done(result, checked, model),
                          on_error=lambda message, _details="": self._check_failed(message))

    def _check_progress(self, done: int, total: int) -> None:
        self.check_progress.setMaximum(total)
        self.check_progress.setValue(done)

    def _check_finish(self) -> None:
        self.check_stop.hide()
        self.check_progress.hide()

    def _check_failed(self, message: str) -> None:
        self._check_finish()
        self.check_verdict.setText(trf("The check failed: {reason}", reason=tr(message)))
        self.render_check()

    def _check_done(self, result, checked, model) -> None:
        self._check_finish()
        self._check = (result, checked, model)
        table_rows = rows(result, checked.reference)
        table = self.check_table
        table.setRowCount(len(table_rows))
        for line, row in enumerate(table_rows):
            ref = row["reference"]
            cells = (row["name"], f"{row['checked']:.5g}", "—" if ref is None else _interval(ref[1], ref[2]),
                     f"{row['median']:.5g}", _interval(row["low68"], row["high68"]),
                     _interval(row["low95"], row["high95"]), f"{row['tau']:.0f}", f"{row['rhat']:.3f}")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 4 and row["overlap"] is False:
                    item.setText(text + " ⚠")
                table.setItem(line, column, item)
        current = self.check_trace_value.currentText()
        self.check_trace_value.blockSignals(True)
        self.check_trace_value.clear()
        self.check_trace_value.addItems(result.names)
        index = self.check_trace_value.findText(current)
        self.check_trace_value.setCurrentIndex(max(0, index))
        self.check_trace_value.blockSignals(False)
        self._plot_trace()
        self.check_verdict.setText(self._verdict_text())
        self.render_check()

    def _verdict_text(self) -> str:
        if self._check is None:
            return ""
        result, checked, _model = self._check
        d = result.diagnostics
        lines = [trf(DONE, steps=result.steps, walkers=result.walkers, seconds=result.seconds, acceptance=d.acceptance)]
        if result.stopped:
            lines.append(trf(STOPPED, steps=result.steps))
        template, values = verdict(result)
        lines.append(trf(template, **values))
        apart = [row["name"] for row in rows(result, checked.reference) if row["overlap"] is False]
        if apart:
            lines.append(trf(DIFFERS, names=", ".join(apart)))
        lines.append(tr(CHECK_ONLY))
        return " ".join(lines)

    def _plot_trace(self) -> None:
        if self._check is None or not self.check_trace_value.currentText():
            self.check_trace.clear_curves()
            self.check_trace.set_empty_text(tr("Run the check: the walkers' values step by step are drawn here."))
            return
        result = self._check[0]
        name = self.check_trace_value.currentText()
        steps, values = trace(result, name)
        x = np.asarray(steps, float)
        walkers = min(3, values.shape[1])
        curves = [(trf("walkers 1–{count}", count=walkers) if k == 0 else trf("walker {number}", number=k + 1), x,
                   values[:, k]) for k in range(walkers)]
        curves += [(tr("16 % and 84 % of the walkers"), x, np.quantile(values, 0.16, axis=1)),
                   ("84 %", x, np.quantile(values, 0.84, axis=1)),
                   (tr("median of the walkers"), x, np.median(values, axis=1))]
        self.check_trace.set_labels(tr("step"), name)
        self.check_trace.set_curves(curves, colors=[TRACE_WALKER] * walkers + [TRACE_BAND, TRACE_BAND, TRACE_MEDIAN],
                                    legend=[k == 0 for k in range(walkers)] + [True, False, True])

    def save_check_dialog(self, path=None):
        if self._check is None:
            return None
        path = path or self._save_path("Save Check", f"{self._stem()}_posterior_check.json", "JSON (*.json)")
        if not path:
            return None
        result, checked, _model = self._check
        try:
            Path(path).write_text(json.dumps(record(result, checked), indent=1, ensure_ascii=False, allow_nan=False),
                                  encoding="utf-8")
        except (OSError, ValueError) as exc:
            self._status(trf("Could not save {name}: {reason}", name=Path(path).name, reason=str(exc)), "error")
            return None
        self._status(trf("Saved {name}.", name=Path(path).name), "ok")
        return path


__all__ = ["PosteriorCheckMixin"]
