"""The single-page v2 fit: original curves, scores and errors, plus a manual starting model."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ...application.single_fit import model_from_solution
from ...application.single_fit1d import fit1d_request, native_export
from ..views.fit_steps_view import fit_to_rows


class Fit1DPageMixin:
    def setup_fit1d(self):
        self._fit1d_job_id = None
        stored = self._remembered("fit1d_settings") or {}
        for key, widget in (("fit1d_model", self.fit1d_network), ("fit1d_policy", self.fit1d_policy),
                            ("fit1d_mode", self.fit1d_mode)):
            index = widget.findData(stored.get(key))
            if index >= 0:
                widget.setCurrentIndex(index)
        qz = stored.get("fit1d_qz")
        self.fit1d_use_qz.setChecked(qz is not None)
        self.fit1d_qz.setValue(float(qz or 0))
        self.fit1d_sigma_floor.setValue(100 * float(stored.get("fit1d_sigma_floor") or 0))
        for widget in (self.fit1d_network, self.fit1d_policy, self.fit1d_mode):
            widget.currentIndexChanged.connect(self._fit1d_settings_changed)
        self.fit1d_use_qz.toggled.connect(self._fit1d_settings_changed)
        self.fit1d_qz.valueChanged.connect(self._fit1d_settings_changed)
        self.fit1d_sigma_floor.valueChanged.connect(self._fit1d_settings_changed)
        self._fit1d_settings_changed()

    def fit1d_settings(self):
        return dict(fit1d_model=self.fit1d_network.currentData(), fit1d_policy=self.fit1d_policy.currentData(),
                    fit1d_mode=self.fit1d_mode.currentData(),
                    fit1d_qz=self.fit1d_qz.value() if self.fit1d_use_qz.isChecked() else None,
                    fit1d_sigma_floor=self.fit1d_sigma_floor.value() / 100)

    def _fit1d_settings_changed(self, *_args):
        gisaxs = self.fit1d_mode.currentData() == "gisaxs"
        self.fit1d_use_qz.setEnabled(gisaxs)
        self.fit1d_qz.setEnabled(gisaxs and self.fit1d_use_qz.isChecked())
        self.session.native_settings = self.fit1d_settings()
        self._remember(fit1d_settings=self.session.native_settings)
        if self.session.native_result is not None:
            self._render_all()

    def _start_fit1d(self, data, model, selection, curve):
        request = fit1d_request(data, model, self.fit1d_settings())
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
        try:
            mapped, _deviation = model_from_solution(value["mapping_row"], self.session.model)
            self.session.set_model(mapped)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            native.setdefault("warnings", []).append(f"Manual starting model could not be made: {exc}")
        self.session.result = None
        self.session.solutions = []
        self.session.native_result = native
        self.session.native_selection = selection
        self.session.native_model = self.session.model
        self._model_changed(editor=True)
        self._render_all()
        self.show_step("results")
        self._status(lambda: tr("fit1d finished: χ²/point {chi2}. Its original v2 curve and errors are in Results; "
                                "Model is a starting point for manual refinement.").format(
                                    chi2=f"{native['summary']['chi2_per_point']:.3g}"),
                     "ok" if native["summary"]["status"] == "ok" else "warning")

    def _render_fit1d_results(self):
        native = self.session.native_result
        if native is None or self.session.result is not None:
            return False
        summary = native["summary"]
        current = self.session.native_is_current()
        quality = tr("fit1d · χ²/point {chi2} · {points} points · {model}").format(
            chi2=f"{summary['chi2_per_point']:.3g}", points=summary["n_points"], model=summary["model"])
        if not current:
            quality = tr("{quality} (previous fit1d result)").format(quality=quality)
        self.quality_label.setText(quality)
        warnings = list(native.get("warnings") or summary.get("messages") or [])
        intro = tr("The original fit1d v2 curve is shown. Model is only a manual starting point; "
                   "its physics and parameter widths differ.") if current else tr(
                       "The plot shows the manual starting model. The saved fit1d result is stale: fit again.")
        shown = [intro] + list(summary.get("messages") or [])
        shown.extend(message for message in warnings if message.startswith("No qz"))
        if len(warnings) > len(shown) - 1:
            shown.append(tr("Parameter warnings are in the State column; hover for ranges. All warnings are "
                            "kept in the saved JSON."))
        self.warnings_label.setText("\n".join(shown))
        self.warnings_label.setToolTip("\n".join(warnings))
        self.step_intro["results"].setText(quality)
        self.parameters_table.setHorizontalHeaderLabels(["", tr("value"), tr("state")])
        rows = native.get("table") or []
        self.parameters_table.setRowCount(len(rows))
        for line, row in enumerate(rows):
            status = row.get("status", "ok")
            bounds = row.get("range")
            tip = status + (f"; equivalent solutions: {bounds}" if bounds else "")
            values = (f"{row['group']} · {row['parameter']}",
                      f"{row.get('text') or row.get('value')} {row.get('unit') or ''}".strip(), tr(status))
            for column, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                item.setToolTip(tip)
                self.parameters_table.setItem(line, column, item)
        fit_to_rows(self.parameters_table)
        self._fill_solutions()
        return True

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
