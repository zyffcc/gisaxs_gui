"""Predict parameters (fit1d) on the single-curve page: the kit's v2 curve, scores and errors, plus a manual
starting model. Candidates and parameters at an edge are in ``fit1d_review.py``."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import FitModel, model_from_solution
from ...application.single_fit1d import (
    COMPOSITIONS,
    fit1d_request,
    in_model_order,
    is_auto,
    model_path,
    native_export,
)
from .fit1d_review import note_text, number
from ..views.step_widgets import fit_to_rows

FINISHED = ("fit1d finished: χ²/point {chi2}. Its original v2 curve and errors are in Results; Model is a starting "
            "point for manual refinement.")
FINISHED_AUTO = ("fit1d tried {count} compositions; the best, {composition} (χ²/point {chi2}), is in Results and in "
                 "Model. Compare the candidates in Results.")
STATE_RANGE = "{state}: {low}–{high} {unit}"
STATUS_WORDS = {"one_sided": "one-sided bound"}
"""States of the kit's table written here (``domain.fit1d_edges.mark_bounds``), in words."""


LABELS = {"sigma_R / R": "σR/R", "w / D": "w/D", "sigma_h / h": "σh/h", "intensity share": "share",
          "background B": "background", "low-q amplitude C": "low-q C", "low-q exponent nu": "low-q ν"}
"""The kit's parameter labels, as Fitting writes them."""


KIT_PREFIXES = ("Another solution fits equally well: ", "Not pinned down by this curve: ",
                "Populations without contribution: ")
"""The kit's sentences that end with names of parameters: the beginning is translated, the names stay."""


def kit_message(text: str) -> str:
    prefix = next((item for item in KIT_PREFIXES if text.startswith(item)), None)
    return tr(text) if prefix is None else tr(prefix.strip()) + " " + text[len(prefix):]


def short_name(row) -> str:
    """``pop1 R`` / ``global low-q width`` (the population's shape is in the tooltip)."""
    group = str(row.get("group") or "").split(" (")[0]
    return f"{group} {LABELS.get(row.get('parameter'), row.get('parameter'))}"


class Fit1DPageMixin:
    def setup_fit1d(self):
        self._fit1d_job_id = None
        self._fit1d_candidates = []
        self._gisaxs_cut = False
        for value, text in COMPOSITIONS:
            self.fit1d_composition.addItem(tr(text), value)
        stored = self._remembered("fit1d_settings") or {}
        for key, widget in (("fit1d_model", self.fit1d_network), ("fit1d_policy", self.fit1d_policy),
                            ("fit1d_mode", self.fit1d_mode), ("fit1d_composition", self.fit1d_composition)):
            index = widget.findData(stored.get(key))
            if index >= 0:
                widget.setCurrentIndex(index)
        qz = stored.get("fit1d_qz")
        self.fit1d_use_qz.setChecked(qz is not None)
        self.fit1d_qz.setValue(float(qz or 0))
        self.fit1d_sigma_floor.setValue(100 * float(stored.get("fit1d_sigma_floor") or 0))
        for widget in (self.fit1d_network, self.fit1d_policy, self.fit1d_mode, self.fit1d_composition):
            widget.currentIndexChanged.connect(self._fit1d_settings_changed)
        self.fit1d_use_qz.toggled.connect(self._fit1d_settings_changed)
        self.fit1d_qz.valueChanged.connect(self._fit1d_settings_changed)
        self.fit1d_sigma_floor.valueChanged.connect(self._fit1d_settings_changed)
        self.setup_fit1d_review()
        self._fit1d_settings_changed()

    def fit1d_settings(self):
        return dict(fit1d_model=self.fit1d_network.currentData(), fit1d_policy=self.fit1d_policy.currentData(),
                    fit1d_mode=self.fit1d_mode.currentData(),
                    fit1d_qz=self.fit1d_qz.value() if self.fit1d_use_qz.isChecked() else None,
                    fit1d_sigma_floor=self.fit1d_sigma_floor.value() / 100,
                    fit1d_composition=self.fit1d_composition.currentData())

    def _fit1d_settings_changed(self, *_args):
        gisaxs = self.fit1d_mode.currentData() == "gisaxs"
        self.fit1d_use_qz.setEnabled(gisaxs)
        self.fit1d_qz.setEnabled(gisaxs and self.fit1d_use_qz.isChecked())
        self.session.native_settings = self.fit1d_settings()
        self._remember(fit1d_settings=self.session.native_settings)
        if self.session.native_result is not None:
            self._render_all()
        else:
            self._refresh_fit1d_action()

    def _start_fit1d(self, data, model, selection, curve):
        request = fit1d_request(data, model, self.fit1d_settings(), resolve_edges=self.resolve_edges_check.isChecked())
        self._fit1d_job_id = request.job_id
        self.tasks.submit("fit", lambda: self._run_fit1d_job(request),
                          on_done=lambda result: self._fit1d_finished(result, selection, curve),
                          on_error=lambda message, details="": self._failed(message, details, curve))

    def _run_fit1d_job(self, request):
        if self._stop_event.is_set():
            raise RuntimeError("Fitting cancelled")
        result = self.view_model.context.jobs.run(
            request, self._fit1d_progress)
        if not result.succeeded:
            raise RuntimeError(result.error.message if result.error else "fit1d failed")
        return result.value

    def _fit1d_progress(self, progress):
        if self._stop_event.is_set():
            self._cancel_fit1d()
        else:
            self._progress.changed.emit(float(progress.fraction), str(progress.message))

    def _cancel_fit1d(self):
        if self._fit1d_job_id:
            self.view_model.context.jobs.cancel(self._fit1d_job_id)

    def _fit1d_finished(self, value, selection, curve):
        self._fit1d_job_id = None
        if self._of_another_curve(curve):
            return
        stopped = self._stop_event.is_set()
        self._finish()
        if stopped:
            self._status(lambda: tr("Stopped before the search finished: no new solutions."), "warning")
            return
        native = value["native"]
        auto = is_auto(native.get("settings") or {})
        if value.get("candidates"):  # a fit of one composition keeps the list of the compositions tried before
            self._fit1d_candidates = list(value["candidates"])
        row = value["mapping_row"] if auto else in_model_order(value["mapping_row"], self.session.model)
        self._take_fit1d(native, row, FitModel() if auto else self.session.model, selection)
        self.set_method("local")  # next: refine these values with Fitting's own model
        self.show_step("results")
        summary, bounds = native["summary"], native.get("fit1d_bounds") or {}
        if auto and len(value.get("candidates") or ()) > 1:
            message = lambda: trf(FINISHED_AUTO, count=len(self._fit1d_candidates),  # noqa: E731
                                  composition=summary["composition"], chi2=f"{summary['chi2_per_point']:.3g}")
        else:
            message = lambda: trf(FINISHED, chi2=f"{summary['chi2_per_point']:.3g}")  # noqa: E731
        self._status(message, "ok" if summary["status"] == "ok" and not bounds.get("edges") else "warning")

    def _take_fit1d(self, native, row, base, selection=None):
        """Show ``native`` (fit1d's own result) and put its values into Model as a starting point."""
        try:
            mapped, _deviation = model_from_solution(row, base)
            self.session.set_model(mapped)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            native.setdefault("warnings", []).append(f"Manual starting model could not be made: {exc}")
        self.session.result = None
        self.session.solutions = []
        self.session.native_result = native
        if selection is not None:
            self.session.native_selection = selection
        self.session.native_model = self.session.model
        self._model_changed(editor=True)
        self._render_all()

    def _render_fit1d_results(self):
        native = self.session.native_result
        if native is None or self.session.result is not None:
            return False
        summary = native["summary"]
        current = self.session.native_is_current()
        quality = tr("fit1d · χ²/point {chi2} · {points} points · {model}").format(
            chi2=f"{summary['chi2_per_point']:.3g}", points=summary["n_points"], model=summary["model"])
        if summary.get("composition"):
            quality += f" · {summary['composition']}"
        if not current:
            quality = tr("{quality} (previous fit1d result)").format(quality=quality)
        intro = tr("The original fit1d v2 curve is shown. Model is only a manual starting point; "
                   "its physics and parameter widths differ.") if current else tr(
                       "The plot shows the manual starting model. The saved fit1d result is stale: fit again.")
        self.quality_label.setText(quality + "\n" + intro)
        warnings = list(native.get("warnings") or summary.get("messages") or [])
        shown = [kit_message(message) for message in summary.get("messages") or []]
        shown.extend(tr(message) for message in warnings if message.startswith("No qz"))
        shown.extend("⚠ " + note_text(edge["note"]) for edge in (native.get("fit1d_bounds") or {}).get("edges") or ())
        self.warnings_label.setText("\n".join(shown))
        self.warnings_label.setToolTip("\n".join(warnings))
        self.step_intro["results"].setText(tr("Predicted by fit1d.") if current else
                                           tr("Changed after the fit: Fit again"))
        self.parameters_table.setHorizontalHeaderLabels(["", tr("value"), tr("state")])
        header = self.parameters_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # short names; the state takes the rest
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        rows = native.get("table") or []
        self.parameters_table.setRowCount(len(rows))
        self._row_paths = [model_path(self.session.model, row.get("key", "")) for row in rows]
        for line, row in enumerate(rows):
            for column, text in enumerate(self.fit1d_row_cells(row)):
                item = QTableWidgetItem(str(text))
                item.setToolTip(self.fit1d_row_tip(row))
                self.parameters_table.setItem(line, column, item)
        fit_to_rows(self.parameters_table)
        self._fill_solutions()
        return True

    @staticmethod
    def fit1d_row_cells(row) -> tuple:
        """name, value ± error, state (with the range of equivalent solutions of an ambiguous R or D)."""
        status = row.get("status", "ok")
        bounds = row.get("range")
        state = tr(STATUS_WORDS.get(status, status))
        if status == "ambiguous" and bounds:
            state = trf(STATE_RANGE, state=state, low=number(bounds[0]), high=number(bounds[1]),
                        unit=row.get("unit") or "").strip()
        return (short_name(row),
                f"{row.get('text') or row.get('value')} {row.get('unit') or ''}".strip(), state)

    @staticmethod
    def fit1d_row_tip(row) -> str:
        bounds = row.get("range")
        return (f"{row.get('group')} · {row.get('parameter')}: {row.get('status', 'ok')}"
                + (f"; equivalent solutions: {bounds[0]:.4g}–{bounds[1]:.4g}" if bounds else ""))

    def _fit1d_export(self):
        native = self.session.native_result
        names, table = native_export(native)
        curve = self.session.curve
        record = dict(backend="fit1d_physics_v2", native_result=native,
                      curve=None if curve is None else dict(name=curve.name, path=curve.path,
                                                           q_unit_of_file=curve.source_unit),
                      selection=dict(side=self.session.side, q_range_nm_inv=self.session.q_range,
                                     excluded_q_nm_inv=sorted(self.session.excluded)))
        return names, table, record

    def _draw_fit1d_residuals(self):
        if not self.session.native_is_current():
            return False
        curves = self.session.native_result["curves"]
        q = np.asarray(curves["q_nm_inv"], float)
        self.residual_plot.set_labels("|q| (nm⁻¹)", "Δ/σ")
        self.residual_plot.setToolTip(tr("fit1d residual: (I − model) / effective σ"))
        self.residual_plot.set_curves([("0", q[[0, -1]], np.zeros(2)),
                                       (tr("residual"), q, np.asarray(curves["residual"], float))],
                                      ["#94a3b8", "#2563eb"], markers=[False, True])
        return True
