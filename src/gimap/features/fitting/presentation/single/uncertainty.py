"""The Uncertainty and Report window (``views/fit_uncertainty_view.py``) and the flags of the Results step.

* Errors and correlations: every free value with its 1σ error from the covariance of the last Refine or Search (and
  the linked values, propagated), the correlation matrix as a heat map with its values, the pairs with |r| > 0.9;
* χ² profile: one parameter fixed at a grid of values, the others fitted again, Δχ² with the Δχ² = 1 interval
  (``application.fit_uncertainty.chi2_profile``; in the background, Stop keeps what is done);
* Fit report: Markdown, HTML or text (``application.fit_report``), to copy or save.

The Results step's table and warnings name the values the curve does not determine and a beam tail at an edge.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QApplication, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from ...application.fit_report import FORMATS, format_of, report, suffix
from ...application.fit_uncertainty import (
    SHOWN_CORRELATION,
    chi2_profile,
    correlation,
    parameter_name,
    profile_values,
    strong_pairs,
    tail_flags,
    undetermined,
)
from ..views.fit_uncertainty_view import build_uncertainty_dialog
from .session import path_name

NO_FIT = ("Fit first (Refine or Search): the errors and the correlations are those of the last fit, of the values it "
          "found. Predict parameters gives its intervals in Fit details.")
SUMMARY = ("{chi} {value} over {points} points, {free} free values. Errors: 1σ from the covariance (JᵀJ)⁻¹·s², "
           "s² = Σr²/(N − p); linked values by propagation.")
PAIRS = "Strongly correlated (|r| > {limit}): {pairs}"
NO_PAIRS = "No pair of free values is correlated beyond |r| = {limit}."
PROFILE_RESULT = ("68 % interval from Δχ² = 1 (Δχ² in units of the best fit's χ²ᵣ): {low} – {high} "
                  "(best {best}; ± from the covariance {error}).")
OPEN_SIDE = "open (Δχ² stays below 1 on the grid)"
FOUND_BETTER = "A point of the grid fits better than the fit itself (Δχ² below 0): Refine again from there."
BETTER = 0.1
UNDETERMINED_TIP = "Not determined by the curve: the error is not finite or above 10× the value"
TAIL_LINE = "⚠ Beam tail: {name} {what}. Narrow the tail's range, fix it, or switch it off (A = 0, fixed)."
UNDETERMINED_LINE = "⚠ Not determined by the curve (error not finite or above 10× the value): {names}."
REPORT_FILTER = "Markdown (*.md);;HTML (*.html);;Text (*.txt)"


class _ProfileProgress(QObject):
    changed = pyqtSignal(float, str)


def _heat(value: float) -> tuple[QColor, QColor]:
    """(background, text) of a correlation: blue for −1, white for 0, red for +1; grey when not defined."""
    if not math.isfinite(value):
        return QColor(170, 170, 170), QColor(0, 0, 0)
    strength = min(abs(value), 1.0)
    target = (178, 24, 43) if value > 0 else (33, 102, 172)
    colour = QColor(*(int(round(255 + (channel - 255) * strength)) for channel in target))
    return colour, QColor(255, 255, 255) if strength > 0.6 else QColor(0, 0, 0)


class UncertaintyMixin:
    """Needs the page's widgets, ``session``, ``tasks``, ``_data``, ``_status`` and ``_save_path``."""

    def setup_uncertainty(self) -> None:
        build_uncertainty_dialog(self, self)
        self._profile_stop = threading.Event()
        self._profile_signal = _ProfileProgress(self)
        self._profile_signal.changed.connect(self._profile_progress)
        for key in FORMATS:
            self.report_format.addItem({"markdown": "Markdown", "html": "HTML", "text": tr("Plain text")}[key], key)
        self.uncertainty_button.clicked.connect(self.show_uncertainty)
        self.report_format.currentIndexChanged.connect(lambda _i: self._render_report())
        self.report_copy.clicked.connect(lambda: QApplication.clipboard().setText(self._report_text()))
        self.report_save.clicked.connect(lambda: self.save_report_dialog())
        self.profile_run.clicked.connect(self.run_profile)
        self.profile_stop.clicked.connect(self._profile_stop.set)

    def _current_result(self):
        result = self.session.result
        return result if result is not None and self.session.result_is_current() else None

    def refresh_uncertainty(self) -> None:
        self.uncertainty_button.setVisible(self.session.curve is not None)
        if self.uncertainty_dialog.isVisible():
            self.render_uncertainty()

    def show_uncertainty(self) -> None:
        self.render_uncertainty()
        self.uncertainty_dialog.show()
        self.uncertainty_dialog.raise_()

    # -- errors and correlations ------------------------------------------------------------------

    def render_uncertainty(self) -> None:
        result = self._current_result()
        model = self.session.model
        self.profile_parameter.clear()
        if result is None:
            self.uncertainty_summary.setText(tr(NO_FIT))
            for table in (self.uncertainty_values, self.correlation_table):
                table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            self.correlation_pairs.setText("")
            self.profile_run.setEnabled(False)
        else:
            chi = "χ²ᵣ" if result.weighting == "sigma" else "var ln I"
            self.uncertainty_summary.setText(trf(SUMMARY, chi=chi, value=f"{result.chi2_reduced:.4g}",
                                                 points=result.points, free=result.free))
            self._fill_values(model, result)
            self._fill_matrix(model, result)
            for path in result.covariance_paths:
                self.profile_parameter.addItem(f"{parameter_name(path)} · {path_name(model, path, full=True)}", path)
            self.profile_run.setEnabled(bool(result.covariance_paths) and not self.tasks.is_busy())
        self._render_report()

    def _fill_values(self, model, result) -> None:
        rows = [(path, p) for path, p in model.parameters() if (p.free and path in result.errors) or p.tie]
        table = self.uncertainty_values
        table.setRowCount(len(rows))
        loose = set(undetermined(result))
        for line, (path, parameter) in enumerate(rows):
            error = result.errors.get(path)
            relative = (f"{100 * error / abs(parameter.value):.3g} %" if error is not None and math.isfinite(error)
                        and parameter.value else "—")
            mark = " ⚠" if path in loose or path in result.at_bounds else ""
            cells = (path_name(model, path, full=True, unit=True), parameter_name(path) + (f" = {parameter.tie}"
                     if parameter.tie else ""), f"{parameter.value:.6g}",
                     "—" if error is None or not math.isfinite(error) else f"{error:.3g}" + mark, relative)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if mark and column == 3:
                    item.setToolTip(tr(UNDETERMINED_TIP) if path in loose else tr("at a bound"))
                table.setItem(line, column, item)

    def _fill_matrix(self, model, result) -> None:
        paths, matrix = correlation(result)
        table = self.correlation_table
        names = [parameter_name(path) for path in paths]
        table.setRowCount(len(paths))
        table.setColumnCount(len(paths))
        table.setHorizontalHeaderLabels(names)
        table.setVerticalHeaderLabels(names)
        for i in range(len(paths)):
            for j in range(len(paths)):
                value = float(matrix[i, j])
                item = QTableWidgetItem("—" if not math.isfinite(value) else f"{value:+.2f}")
                background, text = _heat(value)
                item.setBackground(QBrush(background))
                item.setForeground(QBrush(text))
                item.setToolTip(f"r({names[i]}, {names[j]}) = {value:+.4f}" if math.isfinite(value) else
                                tr("not defined: the curve does not see this direction"))
                table.setItem(i, j, item)
        pairs = strong_pairs(result)
        self.correlation_pairs.setText(
            trf(PAIRS, limit=f"{SHOWN_CORRELATION:g}", pairs="; ".join(
                f"{parameter_name(a)} ↔ {parameter_name(b)} ({r:+.3f})" for a, b, r in pairs)) if pairs
            else trf(NO_PAIRS, limit=f"{SHOWN_CORRELATION:g}"))

    # -- the χ² profile ----------------------------------------------------------------------------

    def run_profile(self) -> None:
        result, data = self._current_result(), self._data()
        path = self.profile_parameter.currentData()
        if result is None or data is None or path is None or self.tasks.is_busy():
            return
        path = tuple(path)
        try:
            values = profile_values(result.model, path, result.errors.get(path), self.profile_points.value())
        except ValueError as exc:
            self.profile_result.setText(str(exc))
            return
        self._profile_stop.clear()
        self.profile_run.setEnabled(False)
        self.profile_stop.show()
        self.profile_progress.setRange(0, 100)
        self.profile_progress.setValue(0)
        self.profile_progress.show()
        self.profile_result.setText(tr("Fitting the other values at each point of the grid …"))
        model, signal = result.model, self._profile_signal
        job = lambda: chi2_profile(model, data, path, values, progress=signal.changed.emit,  # noqa: E731
                                   stop=self._profile_stop.is_set)
        self.tasks.submit("profile", job, on_done=lambda done: self._profile_done(done, result),
                          on_error=lambda message, _details="": self._profile_failed(message))

    def _profile_progress(self, fraction: float, _line: str) -> None:
        self.profile_progress.setValue(int(100 * fraction))

    def _profile_finish(self) -> None:
        self.profile_stop.hide()
        self.profile_progress.hide()
        self.profile_run.setEnabled(self._current_result() is not None)

    def _profile_failed(self, message: str) -> None:
        self._profile_finish()
        self.profile_result.setText(trf("The profile failed: {reason}", reason=message))

    def _profile_done(self, profile, result) -> None:
        self._profile_finish()
        x, delta = np.asarray(profile.values, float), profile.delta
        keep = np.isfinite(delta)
        name = parameter_name(profile.path)
        self.profile_plot.set_labels(name, "Δχ²")
        self.profile_plot.set_curves([("Δχ²", x[keep], delta[keep]), ("Δχ² = 1", np.array([x.min(), x.max()]),
                                                                       np.ones(2))], markers=[True, None])
        low, high = profile.interval()
        error = result.errors.get(profile.path)
        text = trf(PROFILE_RESULT, low=tr(OPEN_SIDE) if low is None else f"{low:.4g}",
                   high=tr(OPEN_SIDE) if high is None else f"{high:.4g}", best=f"{profile.best_value:.4g}",
                   error="—" if error is None or not math.isfinite(error) else f"{error:.3g}")
        if np.nanmin(delta) < -BETTER:  # a point of the grid, the others fitted again, beat the fit
            text += " " + tr(FOUND_BETTER)
        if profile.stopped:
            text += " " + tr("Stopped: the values done so far are shown.")
        self.profile_result.setText(text)

    # -- the report ---------------------------------------------------------------------------------

    def _report_text(self, fmt=None) -> str:
        data = self._data()
        if data is None or not data.q.size:
            return ""
        return report(self.session.model, data, self.session.curve, side=self.session.side, q_range=self.session.q_range,
                      result=self._current_result(), excluded=self.session.excluded,
                      solutions=self.session.solutions if self.session.solutions_are_current() else (),
                      check_blocks=self.check_report_blocks(), fmt=fmt or self.report_format.currentData() or "markdown")

    def _render_report(self) -> None:
        fmt = self.report_format.currentData() or "markdown"
        text = self._report_text(fmt)
        if fmt == "html":
            self.report_view.setHtml(text)
        else:
            self.report_view.setPlainText(text)
        self.report_save.setEnabled(bool(text))

    def save_report_dialog(self, path=None):
        from .results import folder_action, write_text_whole

        fmt = self.report_format.currentData() or "markdown"
        path = path or self._save_path("Save Report", f"{self._stem()}_fit_report{suffix(fmt)}", REPORT_FILTER)
        if not path:
            return None
        text = self._report_text(format_of(path, fmt))
        if not text:
            self._status(tr("Open a curve first."), "warning")
            return None
        try:
            write_text_whole(path, text)
        except OSError as exc:
            return self._save_failed(exc)
        self._status(lambda: tr("Saved {name}.").format(name=Path(path).name), "ok", action=folder_action(path))
        return path

    # -- flags in the Results step ------------------------------------------------------------------

    def flag_lines(self, result) -> list[str]:
        """Warnings of the values the curve does not determine and of a beam tail at an edge (``beam_tail``)."""
        lines = []
        tail = tail_flags(result)
        for path, why in tail:
            lines.append(trf(TAIL_LINE, name=self._path_name(path),
                             what=tr("is at an edge of its range") if why == "bound" else tr("is not determined")))
        named = {path for path, _why in tail}
        loose = [self._path_name(path) for path in undetermined(result) if path not in named]
        if loose:
            lines.append(trf(UNDETERMINED_LINE, names=", ".join(loose)))
        return lines

    def flag_cell(self, path, result, plus: str, why: str) -> tuple[str, str]:
        """Mark uncertainty cells for unconstrained values and values at a fit bound."""
        if result is not None and path in undetermined(result):
            return (plus or "–") + " ⚠", tr(UNDETERMINED_TIP)
        if result is not None and path in result.at_bounds:
            label = plus + " ⚠" if plus.startswith("±") else "⚠ " + tr("at a bound")
            return label, why or tr("At a bound of the fit range.")
        return plus, why


__all__ = ["UncertaintyMixin"]
