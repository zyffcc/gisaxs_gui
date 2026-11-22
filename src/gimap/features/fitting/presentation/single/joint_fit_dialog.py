"""Fit Several Curves Together (Fitting ▸ Fit Several Curves Together…; Analyze ▸ Send Both Cuts to Fitting): the
behaviour of ``views/joint_fit_view.py`` (application: ``application/joint_fit.py``; Physics and Methods 9.18).

The model is the one in Model of Single analysis when the fit starts: its particles, their values, freedom and ranges.
The curves are added from files, from the page, or by Analyze (both cuts of a frame with their geometry). Each
curve's kind and geometry are edited under the list; the substrate is the sample's, the same for all. A free value is
shared (fitted once) or each curve's own; the scales are always each curve's own. The fit runs in the background with
its progress and Stop (the best values so far are kept); Use in Model puts the values of the first horizontal cut into
Model, Save Record… writes the whole fit as JSON.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

from PyQt5.QtCore import QObject, QSignalBlocker, Qt, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import CURVE_SUFFIXES, LoadCurveRequest
from ...application.joint_fit import (
    KINDS,
    CurveSpec,
    default_shared,
    hints,
    joint_record,
    model_for_page,
    plot_points,
    run_joint_fit,
    shareable,
    substrate_of,
    values_of,
    warnings,
    with_heights,
)
from ...application.single_fit import Curve
from ...application.substrate_level import CUSTOM, SUBSTRATE_CHOICES
from ..views.joint_fit_view import build_joint_dialog
from .run import progress_line
from .session import path_name

CURVE_FILTER = "Curves ({});;All files (*)".format(" ".join(f"*{suffix}" for suffix in CURVE_SUFFIXES))
METHODS = (("local", "Refine: from the values in Model"), ("global", "Search the ranges, then refine"))
SUMMARY = ("χ²ᵣ {chi2} over {points} points of {curves} curves and {free} free values · {outcome} · "
           "{evaluations} evaluations, {seconds:.1f} s")
CURVE_LINE = "{name}: Σr²/N = {value} ({points} points, {weighting})"
BOUNDS = "At a bound of its range: {names}"
MIXED = "The curves are weighted differently (some by σ, some by ln I): their χ² are not comparable."
NOT_ADDED = "{name} was not added: {reason}"
GEOMETRY = {"horizontal": "αi {ai}°, αf {af}°, λ {wl} nm", "vertical": "αi {ai}°, q∥ {qp} nm⁻¹, λ {wl} nm",
            "other": "q_z = 0"}


class _Progress(QObject):
    changed = pyqtSignal(float, str)


class JointFitDialog:
    """The window and what it does; ``page``: Single analysis (its model, curve and ``set_model``)."""

    def __init__(self, page, view_model, parent):
        self.page, self.view_model = page, view_model
        self.specs: list[CurveSpec] = []
        self.result = None
        self.result_specs: list[CurveSpec] = []
        self.result_substrate = None
        self.tasks = TaskRunner(parent)
        self._stop = threading.Event()
        self._signal = _Progress(parent)
        self._signal.changed.connect(self._progress)
        build_joint_dialog(self, parent)
        for key, text in KINDS:
            self.joint_kind.addItem(tr(text), key)
        for key, text in SUBSTRATE_CHOICES:
            self.joint_substrate.addItem(tr(text), key)
        for key, text in METHODS:
            self.joint_method.addItem(tr(text), key)
        self._sync_substrate()
        self._connect()
        language_changed().connect(self.refresh_language)
        self._render()

    def _connect(self) -> None:
        self.joint_add_files.clicked.connect(self.add_files_dialog)
        self.joint_add_current.clicked.connect(self.add_page_curve)
        self.joint_remove.clicked.connect(self.remove_selected)
        self.joint_curves.itemSelectionChanged.connect(self._selected)
        self.joint_kind.currentIndexChanged.connect(lambda _i: self._edited())
        for box in (self.joint_alpha_i, self.joint_alpha_f, self.joint_wavelength, self.joint_q_parallel,
                    self.joint_critical):
            box.valueChanged.connect(lambda _v: self._edited())
        self.joint_substrate.currentIndexChanged.connect(lambda _i: self._sync_substrate())
        self.joint_run.clicked.connect(self.run)
        self.joint_stop.clicked.connect(self._stop.set)
        self.joint_shown.currentIndexChanged.connect(lambda _i: self._plot())
        self.joint_use.clicked.connect(self.use_in_model)
        self.joint_save.clicked.connect(lambda: self.save_record_dialog())

    def show(self) -> None:
        self._render()
        self.joint_dialog.show()
        self.joint_dialog.raise_()

    # -- curves -------------------------------------------------------------------------------------------------

    def _page_globals(self) -> dict:
        g = self.page.session.model.globals
        return {key: float(g[name].value) for key, name in (("alpha_i", "alpha_i"), ("alpha_f", "alpha_f"),
                                                             ("wavelength", "wavelength"))}

    def add_spec(self, spec: CurveSpec) -> None:
        self.specs.append(spec)
        self._render()
        self.joint_curves.selectRow(len(self.specs) - 1)
        self._warn()

    def add_page_curve(self) -> bool:
        curve = self.page.session.curve
        if curve is None:
            self._say(tr("Single analysis has no curve open."), "warning")
            return False
        self.add_spec(CurveSpec(curve.name, curve.q, curve.intensity, curve.sigma, "horizontal", path=curve.path,
                                dq=curve.dq, **self._page_globals()))
        return True

    def add_file(self, path, geometry=None) -> bool:
        """A curve file: its kind and geometry from the file (Analyze's vertical cut) or ``geometry``, else those of
        the page's Model."""
        path = Path(path)
        outcome = self.view_model.load_curve(LoadCurveRequest(path, self.page.unit_setting(), keep_missing_sigma=True))
        if outcome.error is not None:
            self._say(trf(NOT_ADDED, name=path.name, reason=tr(outcome.error.message)), "error")
            return False
        loaded = outcome.value
        curve = Curve.from_arrays(loaded.q, loaded.intensity, loaded.error, name=path.name, path=str(path),
                                  unit=loaded.q_source_unit, dq=loaded.dq)
        found = {**self._page_globals(), **hints(getattr(loaded, "observation", None)), **dict(geometry or {})}
        kind = found.pop("kind", "horizontal")
        alpha_f = found.pop("horizontal_alpha_f", None)
        if kind == "horizontal" and alpha_f is not None:
            found["alpha_f"] = alpha_f
        allowed = {"alpha_i", "alpha_f", "wavelength", "q_parallel", "critical_angle", "band_crosses_specular"}
        self.add_spec(CurveSpec(path.name, curve.q, curve.intensity, curve.sigma, kind, path=str(path),
                                dq=curve.dq, **{k: v for k, v in found.items() if k in allowed}))
        return True

    def add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self.joint_dialog, tr("Add Curves"), "", CURVE_FILTER)
        for path in paths:
            self.add_file(path)

    def remove_selected(self) -> None:
        row = self.joint_curves.currentRow()
        if 0 <= row < len(self.specs):
            del self.specs[row]
            self._render()
            self._warn()

    def _selected(self) -> None:
        row = self.joint_curves.currentRow()
        spec = self.specs[row] if 0 <= row < len(self.specs) else None
        widgets = (self.joint_kind, self.joint_alpha_i, self.joint_alpha_f, self.joint_wavelength,
                   self.joint_q_parallel, self.joint_critical)
        blockers = [QSignalBlocker(w) for w in widgets]
        for widget in widgets:
            widget.setEnabled(spec is not None)
        if spec is not None:
            self.joint_kind.setCurrentIndex(max(0, self.joint_kind.findData(spec.kind)))
            for box, value in ((self.joint_alpha_i, spec.alpha_i), (self.joint_alpha_f, spec.alpha_f),
                               (self.joint_wavelength, spec.wavelength), (self.joint_q_parallel, spec.q_parallel),
                               (self.joint_critical, spec.critical_angle)):
                box.setValue(value)
            self._kind_fields(spec.kind)
        del blockers

    def _kind_fields(self, kind: str) -> None:
        self.joint_alpha_f.setEnabled(kind == "horizontal")
        self.joint_q_parallel.setEnabled(kind == "vertical")
        self.joint_critical.setEnabled(kind == "vertical")
        self.joint_alpha_i.setEnabled(kind != "other")

    def _edited(self) -> None:
        row = self.joint_curves.currentRow()
        if not 0 <= row < len(self.specs):
            return
        kind = str(self.joint_kind.currentData())
        self.specs[row] = self.specs[row].with_geometry(
            kind=kind, alpha_i=self.joint_alpha_i.value(), alpha_f=self.joint_alpha_f.value(),
            wavelength=self.joint_wavelength.value(), q_parallel=self.joint_q_parallel.value(),
            critical_angle=self.joint_critical.value())
        self._kind_fields(kind)
        self._fill_curves()
        self._fill_parameters()
        self._warn()

    def _warn(self) -> None:
        found = warnings(self.specs)
        if found:
            self._say(" ".join("⚠ " + trf(template, **values) for template, values in found), "warning")
        elif self.joint_status.text().startswith("⚠ "):
            self._say("", "info")

    def _sync_substrate(self) -> None:
        custom = self.joint_substrate.currentData() == CUSTOM
        for widget in (self.joint_delta_label, self.joint_delta, self.joint_beta_label, self.joint_beta):
            widget.setVisible(custom)

    def substrate(self):
        def number(field):
            try:
                return float(field.text().replace(",", "."))
            except ValueError:
                return 0.0

        return substrate_of(str(self.joint_substrate.currentData() or ""), number(self.joint_delta),
                            number(self.joint_beta))

    def set_substrate(self, material: str, delta: float = 0.0, beta: float = 0.0) -> None:
        self.joint_substrate.setCurrentIndex(max(0, self.joint_substrate.findData(material)))
        if material == CUSTOM:
            self.joint_delta.setText(f"{delta:.6g}")
            self.joint_beta.setText(f"{beta:.6g}")

    # -- the parameters -----------------------------------------------------------------------------------------

    def _model(self):
        model = self.page.session.model
        return with_heights(model) if any(s.kind == "vertical" for s in self.specs) else model

    def shared(self) -> set:
        out = set()
        for line in range(self.joint_parameters.rowCount()):
            box, item = self.joint_parameters.item(line, 1), self.joint_parameters.item(line, 0)
            if box is not None and item is not None and box.checkState() == Qt.Checked:
                out.add(tuple(item.data(Qt.UserRole)))
        return out

    def _fill_parameters(self) -> None:
        model = self._model()
        before = {tuple(self.joint_parameters.item(i, 0).data(Qt.UserRole)):
                  self.joint_parameters.item(i, 1).checkState() == Qt.Checked
                  for i in range(self.joint_parameters.rowCount())
                  if self.joint_parameters.item(i, 0) is not None and self.joint_parameters.item(i, 1) is not None}
        paths = shareable(model)
        default = default_shared(model)
        table = self.joint_parameters
        table.setRowCount(len(paths))
        result = self.result if self.result is not None and len(self.result.models) == len(self.result_specs) else None
        for line, path in enumerate(paths):
            item = QTableWidgetItem(path_name(model, path, unit=True, full=True))
            item.setData(Qt.UserRole, list(path))
            table.setItem(line, 0, item)
            box = QTableWidgetItem("")
            height = path[1] in ("h", "sigma_h") and model.components[path[0]].heights
            box.setFlags(Qt.ItemIsUserCheckable | (Qt.NoItemFlags if height else Qt.ItemIsEnabled))
            box.setCheckState(Qt.Checked if (height or before.get(path, path in default)) else Qt.Unchecked)
            table.setItem(line, 1, box)
            value, error = f"{model.get(path).value:.5g}", "—"
            if result is not None:
                found = values_of(result, path)
                value = " | ".join(f"{v:.5g}" for v, _e in found) if len({v for v, _e in found}) > 1 else f"{found[0][0]:.5g}"
                error = " | ".join("—" if not math.isfinite(e) else f"{e:.3g}" for _v, e in found[:1 if len(
                    {v for v, _e in found}) == 1 else None])
            table.setItem(line, 2, QTableWidgetItem(value))
            table.setItem(line, 3, QTableWidgetItem(error))

    # -- rendering ----------------------------------------------------------------------------------------------

    def _fill_curves(self) -> None:
        table = self.joint_curves
        kinds = dict(KINDS)
        table.setRowCount(len(self.specs))
        for line, spec in enumerate(self.specs):
            geometry = trf(GEOMETRY[spec.kind], ai=f"{spec.alpha_i:g}", af=f"{spec.alpha_f:g}",
                           wl=f"{spec.wavelength:g}", qp=f"{spec.q_parallel:g}")
            for column, text in enumerate((spec.name, tr(kinds[spec.kind]), str(len(spec.q)), geometry)):
                table.setItem(line, column, QTableWidgetItem(text))

    def _render(self) -> None:
        row = self.joint_curves.currentRow()
        self._fill_curves()
        if 0 <= row < len(self.specs):
            self.joint_curves.selectRow(row)
        self._selected()
        self._fill_parameters()
        busy = self.tasks.is_busy()
        self.joint_run.setEnabled(bool(self.specs) and not busy)
        self.joint_add_current.setEnabled(self.page.session.curve is not None)
        self.joint_use.setEnabled(self.result is not None and not busy)
        self.joint_save.setEnabled(self.result is not None)
        blocker = QSignalBlocker(self.joint_shown)
        self.joint_shown.clear()
        for spec in (self.result_specs if self.result is not None else self.specs):
            self.joint_shown.addItem(spec.name)
        del blocker
        self._plot()

    def _plot(self) -> None:
        index = self.joint_shown.currentIndex()
        if self.result is None or not 0 <= index < len(self.result_specs):
            self.joint_plot.clear_curves()
            self.joint_plot.set_empty_text(tr("Add the curves, then Fit Together: each curve is shown here with its "
                                              "model."))
            return
        spec = self.result_specs[index]
        try:
            x, y, fitted = plot_points(spec, self.result.models[index])
        except ValueError as exc:
            self.joint_plot.set_empty_text(str(exc))
            return
        self.joint_plot.set_labels("q_z (nm⁻¹)" if spec.kind == "vertical" else "|q| (nm⁻¹)", "I")
        self.joint_plot.set_curves([(spec.name, x, y), (tr("Model"), x, fitted)], markers=[True, None])

    # -- the fit ------------------------------------------------------------------------------------------------

    def run(self) -> None:
        if not self.specs or self.tasks.is_busy():
            return
        try:
            substrate = self.substrate()
        except ValueError as exc:
            self._say(str(exc), "error")
            return
        specs, model, shared = list(self.specs), self.page.session.model, self.shared()
        method = str(self.joint_method.currentData())
        self._stop.clear()
        self.joint_run.setEnabled(False)
        self.joint_stop.show()
        self.joint_progress.setRange(0, 100)
        self.joint_progress.setValue(0)
        self.joint_progress.show()
        self._say(tr("Fitting the curves together …"), "info")
        job = lambda: run_joint_fit(specs, model, substrate, shared, method=method,  # noqa: E731
                                    progress=self._signal.changed.emit, stop=self._stop.is_set)
        self.tasks.submit("joint", job, on_done=lambda result: self._done(result, specs, substrate),
                          on_error=lambda message, _details="": self._failed(message))

    def _progress(self, fraction: float, line: str) -> None:
        self.joint_progress.setValue(int(100 * fraction))
        self.joint_status.setText(progress_line(line))

    def _finish(self) -> None:
        self.joint_stop.hide()
        self.joint_progress.hide()

    def _failed(self, message: str) -> None:
        self._finish()
        self._say(trf("The fit failed: {reason}", reason=tr(message)), "error")
        self._render()

    def _done(self, result, specs, substrate) -> None:
        self._finish()
        self.result, self.result_specs, self.result_substrate = result, specs, substrate
        self._render()
        self._summary()
        self._say(tr("Stopped: the best values so far are kept.") if result.stopped else tr("The curves were fitted "
                                                                                             "together."), "ok")

    def _summary(self) -> None:
        result = self.result
        if result is None:
            self.joint_summary.setText("")
            return
        outcome = tr("stopped") if result.stopped else tr("converged") if result.converged else tr("not converged")
        lines = [trf(SUMMARY, chi2=f"{result.chi2_reduced:.4g}", points=sum(result.points), curves=len(result.points),
                     free=result.free, outcome=outcome, evaluations=result.evaluations, seconds=result.seconds)]
        for spec, value, points, weighting in zip(self.result_specs, result.curve_chi2, result.points, result.weightings):
            lines.append(trf(CURVE_LINE, name=spec.name, value=f"{value:.4g}", points=points,
                             weighting=tr("σ") if weighting == "sigma" else tr("ln I")))
        if len(set(result.weightings)) > 1:
            lines.append("⚠ " + tr(MIXED))
        lines.extend("⚠ " + trf(template, **values) for template, values in warnings(self.result_specs))
        if result.at_bounds:
            from ...application.joint_fit import key_name

            lines.append("⚠ " + trf(BOUNDS, names=", ".join(key_name(key) for key in result.at_bounds)))
        self.joint_summary.setText("\n".join(lines))

    def use_in_model(self) -> None:
        if self.result is None:
            return
        number = next((i for i, s in enumerate(self.result_specs) if s.kind == "horizontal"), 0)
        model = model_for_page(self.page.session.model, self.result, number)
        self.page.set_model(model, message=lambda: tr("The values of the joint fit are in Model."),
                            label=("a joint fit", {}))

    def save_record_dialog(self, path=None):
        if self.result is None:
            return None
        path = path or QFileDialog.getSaveFileName(self.joint_dialog, tr("Save Record"), "joint_fit.json",
                                                   "JSON (*.json)")[0]
        if not path:
            return None
        text = json.dumps(joint_record(self.result, self.result_specs, self.result_substrate), indent=1,
                          ensure_ascii=False, allow_nan=False)
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as exc:
            self._say(trf("Could not save {name}: {reason}", name=Path(path).name, reason=str(exc)), "error")
            return None
        self._say(trf("Saved {name}.", name=Path(path).name), "ok")
        return path

    def _say(self, text: str, level: str) -> None:
        self.joint_status.setText(text)
        self.joint_status.setProperty("gimapRole", {"error": "danger", "warning": "warning"}.get(level, "muted"))
        self.joint_status.style().unpolish(self.joint_status)
        self.joint_status.style().polish(self.joint_status)

    def refresh_language(self) -> None:
        for combo, items in ((self.joint_kind, KINDS), (self.joint_substrate, SUBSTRATE_CHOICES),
                             (self.joint_method, METHODS)):
            for index, (_key, text) in enumerate(items):
                combo.setItemText(index, tr(text))
        self._render()
        self._summary()
        self._warn()

    def dispose(self) -> None:
        self._stop.set()
        self.tasks.shutdown(2000)
        try:
            language_changed().disconnect(self.refresh_language)
        except TypeError:  # not connected
            pass


__all__ = ["JointFitDialog", "METHODS"]
