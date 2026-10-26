"""Showing a result: the explanation chosen, the line under Predict, the trend, the table, the report and the
sketch (a mixin of ``SputterPage``).

The default explanation is the folder's first (``StructureFolderReader.available``): the literature method when a
deposition rate makes it possible, else the network's physical path; the other is offered second (More options ▸
Explanation, with the per-window choice and the exploratory ones); the Literature tab compares the series with
published growth. A result knows the scan it was predicted from (``gimap_request.json``); the deposition fields
(rate, t0, end) say what it was drawn with only while that scan is the one in the picker (a result of unknown scan:
until another folder is chosen). Windows where the beam may be off the sample (the toolkit's ``beam_off_sample``: an
intensity dip of the back-and-forth stage translation; without a modulation only an extreme dip, where the beam may
also have been interrupted, and the line says so) are counted in the line under Predict and have a column of their
own in the Windows table. The island height H from the q_z fringes (``fringe_flow.py``) adds a panel to Size Trend, two
columns to Windows (the time group nearest to each window) and a paragraph to Reliability once it is measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ..application import (GRADES, StructureResult, beam_off_count, beam_off_record, film_onset_established,
                           folder_key, modulation_period, onset_of, percolation_short, reliability_report)
from .fringe_panel import height_cells
from .texts import message_text, note_text, stage_name, text
from .views.sputter_page_view import REPORT_EMPTY

CHOICES = {"physical": "Physical path (network)", "literature": "Literature method (mass balance)",
           "window": "Per window (network / refined)", "selected": "Learned selector",
           "sequence": "Time-consistent path (exploratory)"}
THICKNESS_VARIANTS = ("physical", "literature")
"""The explanations drawn from ``structure_physical*``: the deposition fields say what they were drawn with."""
TABLE_COLUMNS = (
    ("time_s", "Time (s)"), ("stage", "Stage (rule)"), ("stage_path", "Stage (path)"), ("module", "Model"),
    ("module_path", "Model (path)"), ("beam_off_sample", "Beam may be off the sample"), ("R_nm", "R (nm)"),
    ("R_grade", "R grade"),
    ("R_typical_rel_error", "R typ. error"), ("R_trend_nm", "R trend (nm)"), ("D_nm", "D (nm)"),
    ("D_grade", "D grade"), ("D_typical_rel_error", "D typ. error"), ("D_trend_nm", "D trend (nm)"),
    ("D_corr_nm", "2π/q* (nm)"), ("D_over_direct", "D ÷ 2π/q*"), ("D_direct_check", "D check"),
    ("t_eq_nm", "t_eq (nm)"), ("t_eq_range", "t_eq range (nm)"), ("nominal_thickness_nm", "Nominal thickness (nm)"),
    ("lit_status", "Literature method"), ("R_lit_nm", "R literature (nm)"), ("D_lit_nm", "D literature (nm)"),
    ("two_R_over_D_lit", "2R/D literature"), ("coverage_lit", "Coverage literature"),
    ("two_R_over_D_net", "2R/D network"), ("coverage_net", "Coverage network"), ("eta_net", "η network"),
    ("film_thickness_status", "Film thickness"), ("L_TS_nm", "TS length L (nm)"), ("refined_R_nm", "Refined R (nm)"),
    ("refined_D_nm", "Refined D (nm)"), ("refine_status", "Refinement"), ("coequal_alternatives", "Alternatives"),
    ("path_changed", "Changed by the path"), ("sequence_status", "Path"),
    ("sequence_data_cost_nats", "Data cost (nats)"), ("selector_probability", "Selector P"),
    ("selector_choice", "Component"), ("R_reason", "R note"), ("D_reason", "D note"), ("lit_reason", "Literature note"),
    ("beam_off_reason", "Beam note"),
)
FRINGE_COLUMNS = (("H_fringes_nm", "H fringes (nm)"), ("H_fringes_range", "H fringes 68% (nm)"))
"""The island height from the q_z fringes of the time group nearest to each window (when H is measured)."""
FRINGES_AFTER = ("D_trend_nm", "D_nm", "R_nm", "time_s")
"""The fringe columns come after the first of these the table has."""
FRINGE_KEYS = {key for key, _label in FRINGE_COLUMNS}
LITERATURE_HIDDEN = {"R_lit_nm", "D_lit_nm", "D_direct_check", "D_over_direct"}
"""Columns the literature explanation shows as R and D already (or that do not apply to it)."""
LIT_STATUS = {"islands": "islands", "after_percolation": "after percolation", "host_peak": "host peak",
              "outlier": "outlier", "late_peak": "late peak", "no_peak": "no peak", "no_data": "no data",
              "before_deposition": "before deposition", "beam_off": "beam may be off the sample"}
"""The toolkit's ``lit_status`` words as the page says them."""
REASONS = ("R_reason", "D_reason", "lit_reason", "beam_off_reason")
D_CHECKS = {"consistent": "consistent"}
"""The toolkit's words of the D check the page says in its language (any other: outside the simulated range)."""
D_OUTSIDE = "outside the simulated 90% range"


def thickness_values(rate, t0, t_end) -> tuple:
    """What the physical files record, as the toolkit writes it: (rate, t0, end of the deposition), t0 = 0 when a rate
    is given without one, None when it is not given (the end: the deposition lasts to the end of the scan). Without a
    rate t0 and the end are recorded too (the end is marked in Size Trend): a change of them rewrites the files, so the
    fields and the folder's memory always say what the result was drawn with."""
    def number(value):
        return None if value is None else float(value)

    try:
        t0 = number(t0)
        return number(rate), (t0 or 0.0) if rate is not None else t0, number(t_end)
    except (TypeError, ValueError):
        return None, None, None


def _cell(row: dict, key: str) -> tuple[str, str]:
    """(what the cell shows, its tooltip: the toolkit's own words where the cell says them shorter)."""
    if key == "t_eq_range":
        low, high = row.get("t_eq_lo_nm"), row.get("t_eq_hi_nm")
        return (f"{low:.3g}–{high:.3g}" if low is not None and high is not None else ""), ""
    value = row.get(key)
    if key == "beam_off_sample" and value:  # the reason (the toolkit's words) as its tooltip
        return tr("yes"), note_text(row.get("beam_off_reason") or "")
    if isinstance(value, bool):
        return (tr("yes") if value else ""), ""
    if isinstance(value, float):
        return (f"±{100 * value:.0f}%" if key.endswith("typical_rel_error") else f"{value:.3g}"), ""
    if value is None:
        return "", ""
    value = str(value)
    if key in ("stage", "stage_path"):
        return stage_name(value), value
    if key == "film_thickness_status":
        return tr("not determined"), value
    if key == "lit_status":
        return (tr(LIT_STATUS[value]) if value in LIT_STATUS else value), value
    if key in REASONS:
        return note_text(value), value
    if key == "D_direct_check":
        return tr(D_CHECKS.get(value, D_OUTSIDE)), value
    return value, ""


def table_columns(result: StructureResult, fringes=None) -> list:
    """The columns that hold a value in this result (the time always), with H from the q_z fringes when it is
    measured."""
    def present(key):
        if key == "t_eq_range":
            return result.has("t_eq_lo_nm")
        if result.variant == "literature" and key in LITERATURE_HIDDEN:
            return False
        return result.has(key)
    columns = [(key, label) for key, label in TABLE_COLUMNS if key == "time_s" or present(key)]
    if fringes is not None and fringes.measured():
        keys = [key for key, _label in columns]
        after = next(key for key in FRINGES_AFTER if key in keys)
        index = keys.index(after) + 1
        columns[index:index] = list(FRINGE_COLUMNS)
    return columns


def _fringe_cell(fringes, row: dict, key: str) -> tuple:
    height, interval = height_cells(fringes, row.get("time_s"))
    return height if key == "H_fringes_nm" else interval


class SputterResultsMixin:
    def _connect_results(self) -> None:
        self.open_button.clicked.connect(self._choose_results)
        self.choice_box.currentIndexChanged.connect(self._choice_changed)
        self.log_scale.toggled.connect(lambda _on: self._draw_trend())
        self.trend_details.toggled.connect(lambda _on: self._draw_trend())

    def _choose_results(self) -> None:
        start = str(self.result.folder) if self.result is not None else str(self.service.results_root())
        folder = QFileDialog.getExistingDirectory(self, tr("Open Sputter Results Folder"), start)
        if folder:
            self.open_results(Path(folder))

    def open_results(self, folder: Path, variant: Optional[str] = None, *, quiet: bool = False) -> None:
        try:
            available = self.service.variants(folder)
            self._fill_choices(available, variant)
            self.show_result(self.service.open(folder, self.choice_box.currentData()))
            self._remember_result(Path(folder))
        except (OSError, ValueError) as exc:
            if not quiet:
                self._notify(message_text(exc), error=True)

    def _fill_choices(self, available: list, wanted: Optional[str]) -> None:
        self.choice_box.blockSignals(True)
        self.choice_box.clear()
        for key in available:
            self.choice_box.addItem(tr(CHOICES.get(key, key)), key)
        index = self.choice_box.findData(wanted)
        self.choice_box.setCurrentIndex(index if index >= 0 else 0)
        self.choice_box.setEnabled(len(available) > 1 and self._busy is None)
        self.choice_box.blockSignals(False)

    def _choice_changed(self, _index: int) -> None:
        if self.result is not None and self._busy is None:
            try:
                self.show_result(self.service.open(self.result.folder, self.choice_box.currentData()))
            except (OSError, ValueError) as exc:
                self._notify(message_text(exc), error=True)

    # -- whose result ------------------------------------------------------------------------------------------
    def _picker_key(self) -> str:
        path = self.scan_picker.path()
        return folder_key(path) if path else ""

    def _result_is_for_picker(self) -> bool:
        """The result shown is of the scan in the picker (or no folder is chosen): the rate fields are its."""
        if self.result is None:
            return False
        picker = self._picker_key()
        return not picker or picker == self._result_owner

    def _show_thickness_fields(self) -> None:
        """The rate, t0 and end fields say what the result shown was drawn with (when it is the picker's scan)."""
        result = self.result
        thickness = result.summary.get("thickness") if result is not None else None
        if result is None or result.variant not in THICKNESS_VARIANTS or not isinstance(thickness, dict) \
                or not self._result_is_for_picker():
            return
        for field, key in ((self.rate, "rate_nm_s"), (self.t0, "t0_s"), (self.t_end, "t_end_s")):
            value = thickness.get(key)
            field.setText("" if value is None else f"{float(value):.6g}")

    # -- showing ---------------------------------------------------------------------------------------------
    def show_result(self, result: StructureResult) -> None:
        previous, self.result = self.result, result
        new_folder = previous is None or Path(previous.folder) != Path(result.folder)
        if result.scan_folder is not None:
            self._result_owner = folder_key(result.scan_folder)
        elif new_folder:  # a result of unknown scan belongs to the folder chosen when it was opened
            self._result_owner = self._picker_key()
        thickness = result.summary.get("thickness")
        if result.variant in THICKNESS_VARIANTS and isinstance(thickness, dict):
            self._drawn = thickness_values(thickness.get("rate_nm_s"), thickness.get("t0_s"), thickness.get("t_end_s"))
        elif new_folder:
            self._drawn = None
        self.merge_button.setEnabled(self._busy is None)
        self._fill_choices(self.service.variants(result.folder) or [result.variant], result.variant)
        self._show_thickness_fields()
        self._show_result_line()
        self._update_fringes(result)
        self._draw_trend()
        self._fill_table()
        self._show_report()
        self.sketch.set_result(result)
        self.literature.set_result(result)

    def _draw_trend(self) -> None:
        self.trend.draw(self.result, log_scale=self.log_scale.isChecked(), detailed=self.trend_details.isChecked(),
                        fringes=self.fringes)

    def _show_report(self) -> None:
        """The Reliability tab: the model's report and the paragraph on H from the q_z fringes (when measured)."""
        if self.result is not None:
            self.report.setHtml(reliability_report(self.result, text, note=note_text) + self.fringe_report())

    def _show_result_line(self) -> None:
        result = self.result
        if result is None:
            return
        counts_R, counts_D = result.counts("R"), result.counts("D")
        parts = [f"<b>{result.folder.name}</b> ({result.metal or '?'})",
                 text("{windows} windows · R in {R} · D in {D}", windows=len(result.rows),
                      R=sum(counts_R[g] for g in GRADES), D=sum(counts_D[g] for g in GRADES))]
        onset, interval, _kind = onset_of(result)
        if result.variant == "literature":
            parts.append(percolation_short(result.summary, text) or tr("no percolation found"))
        elif result.variant == "physical":
            if onset is not None:
                parts.append(text("film from {onset} s", onset=f"{onset:.1f}") + (
                    text(" ({low}–{high} s)", low=f"{interval[0]:.1f}", high=f"{interval[1]:.1f}") if interval else "")
                    + ("" if film_onset_established(result.summary) else tr(", not established: no film fits about as well")))
            elif interval is not None:
                parts.append(text("no film window; an onset from {time} s or later is not excluded",
                                  time=f"{interval[0]:.1f}"))
            else:
                parts.append(tr("no film window"))
        else:
            parts.append(text("refined A+B: R {R}, D {D}", R=counts_R["A"] + counts_R["B"],
                              D=counts_D["A"] + counts_D["B"]))
        period = modulation_period(result.summary)
        if period is not None:
            parts.append(text("intensity modulation period ≈ {period} s (sample stage translation?)",
                              period=f"{period:.3g}"))
        off = beam_off_count(result)
        if off and beam_off_record(result.summary).get("rule") != "extreme_dip_only":
            parts.append(tr("beam may be off the sample in 1 window (stage translation)") if off == 1 else
                         text("beam may be off the sample in {count} windows (stage translation)", count=off))
        elif off:  # without a modulation only extreme dips are flagged: the beam may also have been interrupted
            parts.append(tr("beam may be off the sample or interrupted in 1 window") if off == 1 else
                         text("beam may be off the sample or interrupted in {count} windows", count=off))
        line = " · ".join(parts)
        if result.metal == "Cu":
            line += "<br><b>" + tr("Cu: approximated with the Ag model, for reference only.") + "</b>"
        self.result_label.setText(line)
        self.result_label.setToolTip(str(result.folder))

    def _fill_table(self) -> None:
        result = self.result
        rows = result.rows if result is not None else []
        columns = table_columns(result, self.fringes) if result is not None else []
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels([tr(label) for _key, label in columns])
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _label) in enumerate(columns):
                shown, tip = _fringe_cell(self.fringes, row, key) if key in FRINGE_KEYS else _cell(row, key)
                item = QTableWidgetItem(shown)
                if tip and tip != shown:
                    item.setToolTip(tip)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

    def refresh_results_language(self) -> None:
        for index in range(self.choice_box.count()):
            key = self.choice_box.itemData(index)
            self.choice_box.setItemText(index, tr(CHOICES.get(key, key)))
        if self.result is not None:
            self.show_result(self.result)
        else:
            self.report.setHtml(tr(REPORT_EMPTY))


__all__ = ["CHOICES", "FRINGE_COLUMNS", "SputterResultsMixin", "TABLE_COLUMNS", "table_columns", "thickness_values"]
