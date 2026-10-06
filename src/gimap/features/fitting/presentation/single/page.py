"""The single-curve page of Fitting: Curve → Model → Fit → Results, the plot and the residuals.

Built like Analyze: the steps on the left (each says where it stands), the plot on the right — the
points (those outside the fitting range paler), the model over the whole curve and, on demand,
each of its terms; the orange band is the fitting range and can be dragged — and the residuals
below. Fitting itself (four methods, one Fit button) is in ``run.py``; results and export in
``results.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QFileDialog, QMenu, QShortcut, QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.components.curve_plot import CURVE_COLORS
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import CURVE_SUFFIXES, LoadCurveRequest
from ...application.single_fit import (
    ANALYZE_SIDES,
    FAMILIES,
    NM_PER_A,
    SIDES,
    Curve,
    evaluate,
    fit_scales,
    model_from_dict,
    model_from_solution,
    model_to_dict,
    point_key,
    residuals,
)
from ..views.fit_page_view import FitPageView
from .drops import CurveDropMixin, SplitterMemoryMixin
from .exclusion import FitExclusionMixin
from .model_editor import FitModelEditor
from .results import FitResultsMixin
from .run import FitRunMixin
from .session import FitSession, Solution, model_label

CURVE_FILTER = "Curves ({});;All files (*)".format(" ".join(f"*{suffix}" for suffix in CURVE_SUFFIXES))
"""The curve files the reader takes (``application.CURVE_SUFFIXES``: .dat, .txt)."""
MODEL_POINTS = 600
PREFERENCES_KEY = "fitting_single_page"
DATA_COLOR, OUTSIDE_COLOR, MODEL_COLOR, LEFT_OUT_COLOR = "#2563eb", "#94a3b8", "#f97316", "#dc2626"


class FitPage(CurveDropMixin, SplitterMemoryMixin, QWidget, FitPageView, FitRunMixin, FitResultsMixin,
              FitExclusionMixin):
    curveOpened = pyqtSignal(str)
    """The file of the curve now on the page (In-situ series takes its set-up from it)."""
    batchRequested = pyqtSignal()
    """Open 1D Predict for a list of curves."""

    def __init__(self, view_model, *, quick_fit: Optional[Callable] = None, preferences=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.view_model = view_model
        self.quick_fit = quick_fit
        self.preferences = preferences
        self.session = FitSession()
        self._status_text: Optional[Callable[[], str]] = None
        """Makes the status line again in another interface language (``refresh_language``)."""
        self.tasks = TaskRunner(self)
        self.setup_ui(self)
        self.model_editor = FitModelEditor(self)
        self.model_host.addWidget(self.model_editor)
        family_menu = QMenu(self.add_component_button)
        for key, (name, _keys) in FAMILIES.items():
            family_menu.addAction(tr(name), lambda family=key: self.model_editor.add_component(family))
        self.add_component_button.setMenu(family_menu)
        for key, text in SIDES:
            self.side_combo.addItem(tr(text), key)
        self.residual_plot.plot.setXLink(self.plot.plot)
        for button in (self.residual_plot.zoom_button, self.residual_plot.marks_button, self.residual_plot.reset_button):
            button.hide()
        self.residual_plot.legend.hide()
        self.setAcceptDrops(True)  # a curve file dropped here opens (``drops.py``)
        self._connect()
        self.setup_run()
        self.setup_results()
        self.setup_exclusion()
        self._restore()
        self._render_all()

    def _connect(self) -> None:
        self.open_button.clicked.connect(self.open_curve_dialog)
        self.open_curve_step_button.clicked.connect(self.open_curve_dialog)
        self.load_model_action.triggered.connect(self.load_model_dialog)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.step_rail.stepChosen.connect(self.show_step)
        self.side_combo.activated.connect(lambda _index: self.set_side(self.side_combo.currentData()))
        self.range_min_spin.valueChanged.connect(lambda _value: self._range_typed())
        self.range_max_spin.valueChanged.connect(lambda _value: self._range_typed())
        self.whole_range_button.clicked.connect(lambda: self.set_range(None))
        self.unit_combo.activated.connect(lambda _index: self._reload_with_unit())
        self.plot.windowChanged.connect(self._band_dragged)
        self.plot.log_x_check.toggled.connect(self._log_x_toggled)
        self.plot.log_check.toggled.connect(lambda _on: self._redraw())
        self.terms_check.toggled.connect(lambda _on: self._redraw())
        self.show_ranges_check.toggled.connect(self.model_editor.set_show_ranges)
        self.model_editor.edited.connect(self._model_edited)
        self.batch_button.clicked.connect(self.batchRequested)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Shift+Z", self.redo), ("Ctrl+Y", self.redo),
                           ("Ctrl+Return", self.run_fit)):  # File ▸ Open Data opens a curve while Fitting is shown
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # -- steps ----------------------------------------------------------------------------

    def show_step(self, key: str) -> None:
        self.step_rail.set_current(key)
        self.step_stack.setCurrentWidget(self.step_pages[key])

    def _refresh_steps(self) -> None:
        session, data = self.session, self._data()
        curve = session.curve
        if curve is None:
            self.step_rail.set_state("curve", "pending", tr("Open a curve"))
            self.step_intro["curve"].setText(tr("Open a curve (q, I, σ), or send a cut from Analyze ▸ Send to Fitting."))
        else:
            detail = tr("{name} · {points} points fitted").format(name=curve.name, points=0 if data is None else data.q.size)
            self.step_rail.set_state("curve", "ok", detail)
            self.step_intro["curve"].setText(detail + ".")
        count = len(session.model.components)
        names = model_label(session.model)
        free = sum(1 for _path, parameter in session.model.parameters() if parameter.free)
        self.step_rail.set_state("model", "ok" if count else "warn", f"{names} · " + tr("{free} free").format(free=free))
        self.step_intro["model"].setText(
            tr("{names}; {free} values are fitted.").format(names=names, free=free) if session.edited else
            tr("A starting model: set values you know, or let Fit ▸ Find the particle shape choose the family.")
        )
        result, solutions = session.result, session.solutions
        # Solutions count as a fit only when searched here, and only on the points they were searched on.
        searched, current = session.solutions_searched(), session.solutions_are_current()
        stale = tr("Changed after the fit: Fit again")
        if self.fit_running():
            self.step_rail.set_state("fit", "busy", tr("Fitting …"))
        elif result is not None and not session.result_is_current():
            self.step_rail.set_state("fit", "warn", stale)
        elif result is not None and result.stopped:
            self.step_rail.set_state("fit", "warn", tr("Stopped: the best values so far are kept"))
        elif result is not None:
            self.step_rail.set_state("fit", "ok" if result.converged else "warn", self._quality_text(result))
        elif searched and not current:
            self.step_rail.set_state("fit", "warn", stale)
        elif searched:
            best = solutions[0]
            self.step_rail.set_state("fit", "ok", tr("{count} solutions · best {model} {chi} {value}").format(
                count=len(solutions), model=best.label, chi="var ln I" if curve is not None and curve.sigma is None
                else "χ²ᵣ", value=f"{best.chi2:.3g}" if np.isfinite(best.chi2) else "—"))
        elif solutions:  # a solution of Analyze: its χ² is Analyze's, not of a fit here
            self.step_rail.set_state("fit", "pending", tr("From Analyze: Fit to refine it"))
        else:
            self.step_rail.set_state("fit", "pending", tr("Choose a method, then Fit"))
        if result is not None:
            current_result = session.result_is_current()
            self.step_rail.set_state("results", "ok" if current_result else "warn",
                                     tr("Errors, solutions, Save") if current_result else stale)
        elif searched:
            self.step_rail.set_state("results", "ok" if current else "warn",
                                     tr("{count} solutions to compare").format(count=len(solutions)) if current else stale)
        else:
            self.step_rail.set_state("results", "pending", tr("After a fit"))
        self.undo_button.setEnabled(session.can_undo())
        self.redo_button.setEnabled(session.can_redo())
        self.fit_button.setEnabled(curve is not None and not self.fit_running())
        self.fit_step_button.setEnabled(curve is not None and not self.fit_running())

    # -- the curve ------------------------------------------------------------------------

    def open_curve_dialog(self) -> None:
        start = str(self._remembered("folder") or "")
        path, _ = QFileDialog.getOpenFileName(self, tr("Open Curve"), start, CURVE_FILTER)
        if path:
            self.open_curve(path)

    def open_curve(self, path, side: Optional[str] = None, *, unit: Optional[str] = None) -> bool:
        """Load a curve file (Analyze's cuts: q in Å⁻¹, signed); ``side`` as Analyze names it."""
        path = Path(path)
        unit = unit or self.unit_combo.currentData() or "angstrom"
        outcome = self.view_model.load_curve(LoadCurveRequest(path, unit))
        if outcome.error is not None:
            self._status(lambda: tr("Could not open {name}: {reason}").format(name=path.name,
                                                                              reason=tr(outcome.error.message)), "error")
            return False
        loaded = outcome.value
        curve = Curve.from_arrays(loaded.q, loaded.intensity, loaded.error, name=path.name, path=str(path), unit=unit)
        self._halt_fit()  # a fit of the curve before stops; what it brings back is not kept (run.py)
        self.session.set_curve(curve)
        if side is not None:
            self.session.side = ANALYZE_SIDES.get(side, side if side in dict(SIDES) else "mean")
        data = self._data()
        if not self.session.edited and data is not None and data.q.size >= 3:
            self.session.model = fit_scales(self.session.model, data)  # the starting model at the curve's level
        self.unit_combo.setCurrentIndex(max(0, self.unit_combo.findData(unit)))
        self._remember(folder=str(path.parent), curve={"path": str(path), "side": self.session.side, "unit": unit})
        self._render_all()
        self._log(tr("Opened {name}: {points} points").format(name=path.name, points=curve.q.size))
        if not self.session.edited and self.quick_fit is not None:
            self.method_buttons["shapes"].setChecked(True)  # a first curve: find the family first
            self.show_step("fit")
        else:
            self.show_step("model")
        self._status(lambda: tr("Opened {name}.").format(name=path.name), "ok")
        self.curveOpened.emit(str(path))
        return True

    def _reload_with_unit(self) -> None:
        curve = self.session.curve
        if curve is not None and curve.path and self.unit_combo.currentData() != curve.source_unit:
            self.open_curve(curve.path, self.session.side, unit=self.unit_combo.currentData())

    def set_side(self, side: str) -> None:
        self.session.side = side
        self.session.result = None
        self._remember_curve()
        self._render_all()

    def _remember_curve(self) -> None:
        curve = self.session.curve
        if curve is not None and curve.path:
            self._remember(curve={"path": curve.path, "side": self.session.side, "unit": curve.source_unit,
                                  "q_range": None if self.session.q_range is None else list(self.session.q_range),
                                  "excluded": sorted(self.session.excluded)})

    def set_range(self, q_range, *, record: bool = True, coalesce: bool = False) -> None:
        """``record=False``: set again from a project or the last session (not a step of Undo); ``coalesce``:
        a drag or typing, one step of Undo with the range changes just before it."""
        self.session.set_range(q_range, record=record, coalesce=coalesce)
        self._remember_curve()
        self._render_all()

    def _range_typed(self) -> None:
        low, high = self.range_min_spin.value(), self.range_max_spin.value()
        if high > low:
            self.set_range((low, high), coalesce=True)

    def _band_dragged(self, low: float, high: float) -> None:
        if self.plot.log_x_check.isChecked():
            low, high = 10.0 ** low, 10.0 ** high
        self.set_range((max(low, 0.0), high), coalesce=True)

    def _log_x_toggled(self, on: bool) -> None:
        self.residual_plot.log_x_check.setChecked(on)
        self._redraw()

    def _data(self):
        try:
            return self.session.data()
        except ValueError:
            return None

    # -- the model ------------------------------------------------------------------------

    def _model_edited(self, model) -> None:
        if self.session.set_model(model):
            self._model_changed()

    def _model_changed(self, *, editor: bool = False) -> None:
        if editor:
            result = self.session.result if self.session.result_is_current() else None
            self.model_editor.set_model(self.session.model, result.errors if result else {},
                                        result.at_bounds if result else ())
        self._remember(model=model_to_dict(self.session.model))
        self._redraw()
        self._refresh_steps()
        self.render_results()

    def set_model(self, model, *, message=None) -> None:
        """``message``: the status line (or a function that makes it), when one is wanted."""
        if self.session.set_model(model):
            self._model_changed(editor=True)
        if message:
            self._status(message, "ok")

    def undo(self) -> None:
        """The model, the fitting range and the left-out points before the last change; back at a fit's
        own, the fit holds again (its quality and errors)."""
        if self.session.undo():
            self._history_moved()

    def redo(self) -> None:
        if self.session.redo():
            self._history_moved()

    def _history_moved(self) -> None:
        self._remember(model=model_to_dict(self.session.model))
        self._remember_curve()
        self._render_all()

    def show_solution(self, row: dict) -> bool:
        """A solution of Analyze's automatic analysis (Results ▸ Show in Fitting) as the model."""
        try:
            model, deviation = model_from_solution(row, self.session.model)
        except (ValueError, KeyError, TypeError) as exc:
            self._status(lambda reason=str(exc): tr("Could not use this solution: {reason}").format(reason=reason), "error")
            return False
        label = model_label(model)
        self.session.solutions = [Solution(label, model, float(row.get("best_chi2_weighted") or np.nan), "Analyze", deviation)]
        self.session.solutions_selection = None  # not searched here: its χ² is Analyze's, of Analyze's points
        self.session.result = None  # as when a solution is chosen in Results: a new start
        self.set_model(model)
        self._refresh_steps()  # also when the model was already this one
        self.render_results()
        self.show_step("model")
        self._status(lambda: self._deviation_text(label, deviation), "ok" if deviation <= 0.01 else "warning")
        return True

    @staticmethod
    def _deviation_text(label: str, deviation: float) -> str:
        if not np.isfinite(deviation) or deviation <= 0.01:
            return tr("{model} is in Model; this model draws the same curve.").format(model=label)
        return tr("{model} is in Model as a start: this model differs from the solution by up to {percent} % "
                  "(the Vertical Cylinder here weights radii by R⁴). Fit to refine it.").format(
            model=label, percent=f"{100 * deviation:.3g}")

    def load_model_dialog(self) -> None:
        import json

        path, _ = QFileDialog.getOpenFileName(self, tr("Load Model"), str(self._remembered("folder") or ""),
                                              "Fit model (*.json);;All files (*)")
        if not path:
            return
        try:
            model = model_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._status(lambda reason=str(exc): tr("Could not load the model: {reason}").format(reason=reason), "error")
            return
        self.set_model(model, message=lambda: tr("Loaded the model from {name}.").format(name=Path(path).name))

    # -- the plot -------------------------------------------------------------------------

    def _render_all(self) -> None:
        session = self.session
        curve = session.curve
        self.curve_chip.setText(curve.name if curve is not None else tr("No curve yet — open one, or send a cut from Analyze"))
        self._show_left_out()
        self.side_combo.setCurrentIndex(max(0, self.side_combo.findData(session.side)))
        signed = curve is not None and curve.signed
        self.open_curve_step_button.setVisible(curve is None)  # afterwards: Open Curve… in the command bar
        self.side_combo.setVisible(signed)
        self.side_label.setVisible(signed)
        self._fill_curve_card()
        result = session.result if session.result_is_current() else None  # errors only of a fit that still holds
        self.model_editor.set_model(session.model, result.errors if result else {}, result.at_bounds if result else ())
        self._redraw()
        self._refresh_steps()
        self.render_results()

    def _fill_curve_card(self) -> None:
        curve, data = self.session.curve, self._data()
        full = None if curve is None else self.session.all_points()
        if curve is None or full is None or not full.q.size:
            self.curve_info.setText(tr("No curve yet."))
            self.sigma_note.setText("")
            return
        low, high = float(full.q.min()), float(full.q.max())
        for spin, value in ((self.range_min_spin, low), (self.range_max_spin, high)):
            spin.blockSignals(True)
            spin.setRange(0.0, high * 10)
        chosen = self.session.q_range or (low, high)
        self.range_min_spin.setValue(chosen[0])
        self.range_max_spin.setValue(chosen[1])
        for spin in (self.range_min_spin, self.range_max_spin):
            spin.blockSignals(False)
        fitted = 0 if data is None else data.q.size
        self.curve_info.setText(tr(
            "{name}\n{points} points, |q| {low}–{high} nm⁻¹ ({low_a}–{high_a} Å⁻¹); {fitted} in the fitting range."
        ).format(name=curve.name, points=full.q.size, low=f"{low:.4g}", high=f"{high:.4g}",
                 low_a=f"{low / NM_PER_A:.4g}", high_a=f"{high / NM_PER_A:.4g}", fitted=fitted))
        self.sigma_note.setText(tr("σ from the file: the fit weights each point by 1/σ.") if curve.sigma is not None else
                                tr("No σ in the file: every point gets the same relative weight (ln I is fitted)."))
        self.weighting_note.setText(self.sigma_note.text())

    def _redraw(self) -> None:
        session = self.session
        full, data = (None, None) if session.curve is None else (session.all_points(), self._data())
        curves, colors, markers = [], [], []
        if full is not None and full.q.size:
            inside = np.zeros(full.q.size, bool) if data is None else np.isin(full.q, data.q)
            left_out = np.array([point_key(value) in session.excluded for value in full.q]) if session.excluded \
                else np.zeros(full.q.size, bool)
            outside = ~inside & ~left_out
            if outside.any():
                curves.append((tr("outside the range"), full.q[outside], full.intensity[outside]))
                colors.append(OUTSIDE_COLOR)
                markers.append(True)
            if left_out.any():
                curves.append((tr("left out"), full.q[left_out], full.intensity[left_out]))
                colors.append(LEFT_OUT_COLOR)
                markers.append("x")
            if data is not None and data.q.size:
                curves.append((tr("measured"), data.q, data.intensity))
                colors.append(DATA_COLOR)
                markers.append(True)
            q_model = np.geomspace(max(full.q.min(), 1e-6), full.q.max(), MODEL_POINTS)
            try:
                total, parts = evaluate(session.model, q_model, parts=True)
            except (ValueError, FloatingPointError):
                total, parts = None, {}
            if total is not None:
                curves.append((tr("model"), q_model, total))
                colors.append(MODEL_COLOR)
                markers.append(False)
                if self.terms_check.isChecked():
                    for index, (name, values) in enumerate(parts.items()):
                        curves.append((tr(name), q_model, values))
                        colors.append(CURVE_COLORS[(index + 2) % len(CURVE_COLORS)])
                        markers.append(False)
        self.plot.set_curves(curves, colors, markers=markers)
        self._draw_band(full)
        self._draw_residuals(data)

    def _draw_band(self, full) -> None:
        if full is None or not full.q.size:
            self.plot.hide_window()
            return
        low, high = self.session.q_range or (float(full.q.min()), float(full.q.max()))
        if self.plot.log_x_check.isChecked():
            low, high = np.log10(max(low, 1e-9)), np.log10(max(high, 1e-9))
        self.plot.show_window(low, high)

    def _draw_residuals(self, data) -> None:
        if data is None or not data.q.size:
            self.residual_plot.set_curves([])
            return
        try:
            values = residuals(self.session.model, data)
        except (ValueError, FloatingPointError):
            values = np.full(data.q.size, np.nan)
        weighted = data.sigma is not None
        self.residual_plot.set_labels("|q| (nm⁻¹)", "Δ/σ" if weighted else "Δ ln I")  # short: the plot is low
        tip = tr("Residuals: {formula}").format(formula="Δ/σ = (I − model)/σ" if weighted else "Δ ln I = ln(I / model)")
        self.residual_plot.setToolTip(tip)
        self.residual_plot.plot.getAxis("left").setToolTip(tip)
        self.residual_plot.set_curves([("0", data.q[[0, -1]], np.zeros(2)), (tr("residual"), data.q, values)],
                                      [OUTSIDE_COLOR, DATA_COLOR], markers=[False, True])

    # -- status, log, preferences ---------------------------------------------------------

    def _status(self, text, level: str = "info", action=None) -> None:
        """The status line and the log; a toast for outcomes (``action``: e.g. Open Folder after a save).
        ``text``: the line, or a function that makes it — made again after a switch of the interface language."""
        self._status_text = text if callable(text) else None
        text = text() if callable(text) else text
        self.status_label.setText(text)
        self._log(text)
        if level in ("ok", "warning", "error") and self.isVisible():
            show_toast(self.window(), text, level=level, action=action)

    def _log(self, text: str) -> None:
        from datetime import datetime

        self.log_view.appendPlainText(f"[{datetime.now():%H:%M:%S}] {text}")

    def _remembered(self, key: str):
        stored = self.preferences.get(PREFERENCES_KEY, {}) if self.preferences is not None else {}
        return stored.get(key) if isinstance(stored, dict) else None

    def _remember(self, **values) -> None:
        if self.preferences is None:
            return
        stored = self.preferences.get(PREFERENCES_KEY, {})
        stored = dict(stored) if isinstance(stored, dict) else {}
        stored.update(values)
        self.preferences.set(PREFERENCES_KEY, stored)

    def _restore(self) -> None:
        """The model, method, choices and panel widths of last time; the last curve is opened again once shown."""
        self._keep_splitter()
        stored = self._remembered("model")
        if isinstance(stored, dict):
            try:
                model = model_from_dict(stored)
            except (ValueError, KeyError, TypeError):
                model = None
            if model is not None and model != self.session.model:
                self.session.model = model
                self.session.edited = True  # the person's model of last time, not a starting guess
        method = self._remembered("method")
        if method in self.method_buttons:
            self.method_buttons[method].setChecked(True)
        self.show_ranges_check.setChecked(bool(self._remembered("show_ranges")))
        self.show_ranges_check.toggled.connect(lambda on: self._remember(show_ranges=bool(on)))
        last = self._remembered("curve")
        if isinstance(last, dict) and last.get("path") and Path(str(last["path"])).is_file():
            QTimer.singleShot(0, lambda: self._reopen(last))

    def _reopen(self, last: dict) -> None:
        if self.session.curve is not None:  # a curve arrived meanwhile (Send to Fitting)
            return
        if self.open_curve(last["path"], last.get("side"), unit=last.get("unit")):
            self.session.excluded = {float(value) for value in last.get("excluded") or ()}
            q_range = last.get("q_range")
            if isinstance(q_range, list) and len(q_range) == 2:
                self.set_range(q_range, record=False)  # as last time: not a step of Undo
            else:
                self._exclusions_changed()

    # -- a project -------------------------------------------------------------------------

    def project_state(self) -> dict:
        session, curve = self.session, self.session.curve
        return {
            "curve": None if curve is None or not curve.path else {
                "path": curve.path, "unit": curve.source_unit, "side": session.side,
                "q_range": None if session.q_range is None else list(session.q_range),
                "excluded": sorted(session.excluded)},
            "model": model_to_dict(session.model), "method": self.method(),
        }

    def apply_project_state(self, data: dict) -> list[str]:
        notes = []
        if isinstance(data.get("model"), dict):
            try:
                self.session.model = model_from_dict(data["model"])
                self.session.edited = True
            except (ValueError, KeyError, TypeError) as exc:
                notes.append(tr("the Fitting model could not be read: {reason}").format(reason=exc))
        if data.get("method") in self.method_buttons:
            self.method_buttons[data["method"]].setChecked(True)
        curve = data.get("curve") or {}
        if curve.get("path") and Path(curve["path"]).is_file():
            if self.open_curve(curve["path"], curve.get("side"), unit=curve.get("unit")):
                self.session.excluded = {float(value) for value in curve.get("excluded") or ()}
                if curve.get("q_range"):
                    self.set_range(curve["q_range"], record=False)
        elif curve.get("path"):
            notes.append(tr("the Fitting curve {name} is no longer there").format(name=Path(curve["path"]).name))
        self.session.clear_history()  # a new start: Undo does not bring back the work before the project
        self._model_changed(editor=True)
        self._render_all()
        return notes

    def refresh_language(self) -> None:
        """After a switch of the interface language (``i18n.language_changed``): what this page composed with
        ``tr`` at run time — the steps, the curve card, the model's cards, the legends, the results — again."""
        self.model_editor.refresh_language()
        self._render_all()
        if self._status_text is not None:
            self.status_label.setText(self._status_text())

    def dispose(self) -> None:
        self.stop_fit()
        self.tasks.shutdown(2000)
        self.plot.dispose()
        self.residual_plot.dispose()


__all__ = ["FitPage"]
