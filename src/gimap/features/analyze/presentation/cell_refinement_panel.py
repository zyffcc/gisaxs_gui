"""Display the cell fitted from indexed peaks and the measurements supporting it."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QSignalBlocker, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.components.toast import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.numbers import format_number

from ..application.cell_refinement import frame_cell_refinement
from ..application.export_record import analysis_metadata, export_stem
from .indexing_export import _finite
from .indexing_texts import percent_text


def _table(table, rows):
    table.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            table.setItem(i, j, item)


def _units(name):
    return {"alpha": "α (°)", "beta": "β (°)", "gamma": "γ (°)", "V": "V (Å³)"}.get(name, name + " (Å)")


def _fibre_text(fit):
    if not fit:
        return tr("Fit one resolved fibre axis in Pole Figure to measure directional strain from the sectors.")
    if not fit["ok"]:
        return trf("Directional strain unavailable. {reason}", reason=tr(fit["reason"]))
    return trf("Fibre ({axis}) · in-plane strain {ip}% · out-of-plane strain {op}%",
               axis=" ".join(map(str, fit["axis_hkl"])),
               ip=percent_text(fit["values"]["in_plane"], fit["errors"]["in_plane"]),
               op=percent_text(fit["values"]["out_of_plane"], fit["errors"]["out_of_plane"]))


def _sensitivity(fit):
    lines = []
    sensitivity = fit.get("assignment_sensitivity", {})
    if sensitivity:
        bounds = ", ".join(f"{k} = {lo:.6g}–{hi:.6g} Å" for k, (lo, hi) in sensitivity["range_A"].items())
        lines.append(trf("Omit one centre: {values} ({n}/{total} subsets determined).",
                         values=bounds or "—", n=sum(not r["reason"] for r in sensitivity["fits"]),
                         total=len(sensitivity["fits"])))
    optics = fit.get("refraction_sensitivity", {})
    if optics.get("available"):
        for key, angle in (("off", 0.), ("on", optics["critical_angle_deg"])):
            trial = optics[key]
            values = ", ".join(f"{k} = {v:.6g} Å" for k, v in trial["values"].items() if k in ("a", "b", "c"))
            lines.append(trf("αc = {angle}° · {n} centres · {values}", angle=format_number(angle),
                             n=len(trial["seeds"]), values=values or tr("Cell undetermined")))
    return lines


class CellRefinementMixin:
    def _setup_cell(self):
        self._cell_key = self._cell_cancel = None
        self._cells = []
        self._cell_message = ""
        self.tabs.currentChanged.connect(self._refresh_cell)
        self.cell_phase_combo.currentIndexChanged.connect(self._show_cell)
        self.cell_export_button.clicked.connect(self._export_cell)
        self.cell_cancel_button.clicked.connect(self._stop_or_restart_cell)
        self.cell_series_button.clicked.connect(self._start_cell_series)
        self.finished.connect(self._cancel_cell)

    def _stop_or_restart_cell(self):
        if self._cell_cancel is None:
            self._refresh_cell()
        else:
            self._cancel_cell()
            self._cell_message = "Cancelled"
            self._show_cell()

    def _cancel_cell(self, *_args):
        if self._cell_cancel is not None:
            self._cell_cancel.set()
            self._page.tasks.cancel("cell_refinement")
            self._cell_cancel = self._cell_key = None

    def _cell_source(self):
        analysis = self._page.view_model.state.analysis
        assignment, orientations = self.model.cell_inputs(analysis)
        return (id(analysis), tuple(self.model.phases), self.model.settings,
                id(self._crystal_result or orientations), id(assignment))

    def _refresh_cell(self, *_args):
        if self.tabs.currentWidget() is not self.cell_tab:
            return
        key = self._cell_source()
        if key == self._cell_key:
            self._show_cell()
            return
        self._cancel_cell()
        self._cell_key = key
        self._cells = []
        self.cell_phase_combo.clear()
        analysis = self._page.view_model.state.analysis
        if not self.model.active or analysis is None or analysis.kind != "giwaxs" or analysis.geometry is None:
            self._cell_message = "Open a GIWAXS frame and show a reference phase."
            self._show_cell()
            return
        cancelled = self._cell_cancel = Event()
        self._cell_message = "Refining unit cells"
        self._show_cell()
        phases, settings = key[1], key[2]
        assignment, automatic_axes = self.model.cell_inputs(analysis)
        orientations = self._crystal_result or automatic_axes

        def check_cancel():
            if cancelled.is_set():
                raise InterruptedError("Cancelled")

        self._page.tasks.submit(
            "cell_refinement",
            lambda: frame_cell_refinement(analysis, phases, settings, assignment=assignment,
                                          orientations=orientations, check_cancel=check_cancel),
            on_done=self._cell_ready, on_error=self._cell_failed)

    def _cell_ready(self, results):
        self._cell_cancel = None
        if self._cell_key != self._cell_source():
            self._refresh_cell()
            return
        self._cells = results
        self._cell_message = ""
        with QSignalBlocker(self.cell_phase_combo):
            self.cell_phase_combo.clear()
            self.cell_phase_combo.addItems([r["phase"] for r in results])
        self._show_cell()

    def _cell_failed(self, message, _details):
        self._cell_key = self._cell_cancel = None
        self._cell_message = message
        self._show_cell()

    def _show_cell(self, *_args):
        i = self.cell_phase_combo.currentIndex()
        fit = self._cells[i] if 0 <= i < len(self._cells) else None
        self.cell_export_button.setEnabled(bool(fit))
        self.cell_series_button.setEnabled(any(r["seeds"] for r in self._cells)
                                           and self._page.view_model.listed_frames() > 1)
        self.cell_cancel_button.setVisible(self._cell_cancel is not None or self._cell_key is None)
        self.cell_cancel_button.setText(tr("Cancel") if self._cell_cancel is not None else tr("Refine Cell"))
        self.cell_summary.setText(tr(self._cell_message))
        self.cell_fibre_label.clear()
        parameters, peaks = [], []
        if fit:
            self.cell_summary.setText(tr(fit["reason"]))
            if fit["ok"]:
                shift = (format_number(fit["zero_shift"], fit["zero_shift_error"])
                         if fit["zero_shift_error"] is not None else tr("0 (fixed)"))
                self.cell_summary.setText(trf("{system} · {n} peaks · χ²red {chi} · q shift {shift} Å⁻¹",
                                              system=tr(fit["system"]), n=len(fit["residuals"]),
                                              chi=format_number(fit["chi2_red"]), shift=shift))
                for key, value in fit["values"].items():
                    parameters.append((_units(key), format_number(value, fit["errors"][key]),
                                       format_number(fit["reference"][key], fit.get("reference_errors", {}).get(key),
                                                     compact=True, digits=5),
                                       percent_text(fit["strain"].get(key), fit["strain_errors"].get(key))))
                for row in fit["residuals"]:
                    residual = format_number(row["residual"], row["sigma"], value_only=True)
                    peaks.append((" ".join(map(str, row["hkl"])), format_number(row["q"], row["sigma"]),
                                  residual, tr("Yes")))
            else:
                for row in fit["seeds"]:
                    peaks.append((" ".join(map(str, row["hkl"])), format_number(row["q"], row["sigma"]),
                                  "—", tr("Available")))
            critical = fit["q_axis"].get("critical_angle_deg")
            basis = trf("Transmitted q (αc = {angle}°)", angle=f"{critical:g}") if critical else tr("Kinematic q")
            rank = trf("metric rank {rank}/{total}", rank=fit["metric_rank"], total=fit["metric_parameters"])
            self.cell_summary.setText(" · ".join((self.cell_summary.text(), rank, basis)))
            for row in fit.get("excluded", ()):
                peaks.append((str(row.get("candidate_hkl") or "—"), format_number(row["q"], row.get("sigma")),
                              "—", tr(row["reason"])))
            self.cell_fibre_label.setText("\n".join([*_sensitivity(fit), _fibre_text(fit.get("fibre_strain"))]))
        _table(self.cell_values, parameters)
        self.cell_values.setVisible(bool(parameters))
        height = self.cell_values.horizontalHeader().sizeHint().height()
        self.cell_values.setMaximumHeight(height + sum(self.cell_values.rowHeight(i) for i in range(len(parameters))) + 4)
        _table(self.cell_peaks, peaks)

    def _cell_provenance(self):
        return {"analysis": analysis_metadata(self._page.view_model.state.analysis),
                "settings": self.model.settings.record(), "q_axis": self._cells[0]["q_axis"]}

    def _start_cell_series(self):
        from .cell_series_dialog import open_cell_series

        open_cell_series(self, tuple(self.model.phases), self._cells, self._cell_provenance())

    def _export_cell(self):
        analysis = self._page.view_model.state.analysis
        folder = self._page.view_model.default_export_dir() or Path.home()
        name, _ = QFileDialog.getSaveFileName(self, tr("Export Refined Cells"),
                                              str(folder / (export_stem(analysis) + "_cells.json")),
                                              tr("JSON files (*.json)"))
        if not name:
            return
        path = Path(name)
        payload = {**self._cell_provenance(), "cell_refinement": self._cells}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_finite(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except OSError as error:
            self.cell_summary.setText(trf("Could not write {file}. {error}", file=path.name, error=str(error)))
            return
        show_toast(self._page, trf("Wrote {files}.", files=path.name), level="ok",
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))))
