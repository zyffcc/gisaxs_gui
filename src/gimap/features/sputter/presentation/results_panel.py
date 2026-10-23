"""Showing a result: the explanation chosen, the line under Predict, the trend, the table, the report and the
sketch (a mixin of ``SputterPage``).

The default explanation is the folder's first (``StructureFolderReader.available``): the literature method when a
deposition rate makes it possible, else the physical path; the others stay one choice away (More options ▸
Explanation). The Literature tab compares the series with published growth. A result knows the scan it was predicted from
(``gimap_request.json``); the deposition-rate fields say what it was drawn with only while that scan is the
one in the picker (a result of unknown scan: until another folder is chosen).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ..application import (GRADES, StructureResult, folder_key, onset_of, percolation_short, reliability_report,
                           rotation_period)
from .texts import message_text, note_text, stage_name, text

CHOICES = {"literature": "Literature method (mass balance)", "physical": "Physical path (network)",
           "window": "Per window (network / refined)", "selected": "Learned selector",
           "sequence": "Time-consistent path (exploratory)"}
THICKNESS_VARIANTS = ("physical", "literature")
"""The explanations drawn from ``structure_physical*``: the deposition-rate fields say what they were drawn with."""
TABLE_COLUMNS = (
    ("time_s", "Time (s)"), ("stage", "Stage (rule)"), ("stage_path", "Stage (path)"), ("module", "Model"),
    ("module_path", "Model (path)"), ("R_nm", "R (nm)"), ("R_grade", "R grade"),
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
)
LITERATURE_HIDDEN = {"R_lit_nm", "D_lit_nm", "D_direct_check", "D_over_direct"}
"""Columns the literature explanation shows as R and D already (or that do not apply to it)."""
LIT_STATUS = {"islands": "islands", "after_percolation": "after percolation", "host_peak": "host peak",
              "outlier": "outlier", "late_peak": "late peak", "no_peak": "no peak", "no_data": "no data",
              "before_deposition": "before deposition"}
"""The toolkit's ``lit_status`` words as the page says them."""
REASONS = ("R_reason", "D_reason", "lit_reason")
D_CHECKS = {"consistent": "consistent"}
"""The toolkit's words of the D check the page says in its language (any other: outside the simulated range)."""
D_OUTSIDE = "outside the simulated 90% range"


def thickness_pair(rate, t0) -> tuple:
    """What the physical files depend on: (rate, t0), t0 = 0 when a rate is given without one, nothing without
    a rate (t0 alone changes nothing)."""
    try:
        return (None, None) if rate is None else (float(rate), float(t0 or 0.0))
    except (TypeError, ValueError):
        return None, None


def _cell(row: dict, key: str) -> tuple[str, str]:
    """(what the cell shows, its tooltip: the toolkit's own words where the cell says them shorter)."""
    if key == "t_eq_range":
        low, high = row.get("t_eq_lo_nm"), row.get("t_eq_hi_nm")
        return (f"{low:.3g}–{high:.3g}" if low is not None and high is not None else ""), ""
    value = row.get(key)
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


def table_columns(result: StructureResult) -> list:
    """The columns that hold a value in this result (the time always)."""
    def present(key):
        if key == "t_eq_range":
            return result.has("t_eq_lo_nm")
        if result.variant == "literature" and key in LITERATURE_HIDDEN:
            return False
        return result.has(key)
    return [(key, label) for key, label in TABLE_COLUMNS if key == "time_s" or present(key)]


class SputterResultsMixin:
    def _connect_results(self) -> None:
        self.open_button.clicked.connect(self._choose_results)
        self.choice_box.currentIndexChanged.connect(self._choice_changed)
        self.log_scale.toggled.connect(lambda _on: self._draw_trend())

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
        """The rate and t0 fields say what the result shown was drawn with (when it is the picker's scan)."""
        result = self.result
        thickness = result.summary.get("thickness") if result is not None else None
        if result is None or result.variant not in THICKNESS_VARIANTS or not isinstance(thickness, dict) \
                or not self._result_is_for_picker():
            return
        for field, key in ((self.rate, "rate_nm_s"), (self.t0, "t0_s")):
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
            self._drawn = thickness_pair(thickness.get("rate_nm_s"), thickness.get("t0_s"))
        elif new_folder:
            self._drawn = None
        self.merge_button.setEnabled(self._busy is None)
        self._fill_choices(self.service.variants(result.folder) or [result.variant], result.variant)
        self._show_thickness_fields()
        self._show_result_line()
        self._draw_trend()
        self._fill_table()
        self.report.setHtml(reliability_report(result, text, note=note_text))
        self.sketch.set_result(result)
        self.literature.set_result(result)

    def _draw_trend(self) -> None:
        self.trend.draw(self.result, log_scale=self.log_scale.isChecked())

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
                    text(" ({low}–{high} s)", low=f"{interval[0]:.1f}", high=f"{interval[1]:.1f}") if interval else ""))
            else:
                parts.append(tr("no film window"))
        else:
            parts.append(text("refined A+B: R {R}, D {D}", R=counts_R["A"] + counts_R["B"],
                              D=counts_D["A"] + counts_D["B"]))
        period = rotation_period(result.summary)
        if period is not None:
            parts.append(text("intensity modulation period ≈ {period} s (sample rotation?)", period=f"{period:.3g}"))
        line = " · ".join(parts)
        if result.metal == "Cu":
            line += "<br><b>" + tr("Cu: approximated with the Ag model, for reference only.") + "</b>"
        self.result_label.setText(line)
        self.result_label.setToolTip(str(result.folder))

    def _fill_table(self) -> None:
        result = self.result
        rows = result.rows if result is not None else []
        columns = table_columns(result) if result is not None else []
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels([tr(label) for _key, label in columns])
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _label) in enumerate(columns):
                shown, tip = _cell(row, key)
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


__all__ = ["CHOICES", "SputterResultsMixin", "TABLE_COLUMNS", "table_columns", "thickness_pair"]
