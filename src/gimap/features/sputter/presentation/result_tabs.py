"""Details ▸ Windows, Report and Logs: the result the main screen shows, from the files of its folder (a mixin of
``SputterPage``).

**Windows**: one row per time window (a series of static samples: per sample) with the solution the main screen shows
for it — the time (or the sample), the nominal thickness, the values of the tiles the result has, the fit deviation
and which solution it is; a window without a solution says why. The rows follow a choice on the main screen; Ctrl+C
copies the selected rows (``enable_table_copy``).

**Report**: the facts of ``analysis.json`` — the network used and the networks tried with their fit deviation, the
mode, the pixel mask, the run time of the steps —; *Time per step*, a small table of every step with its time (m:ss),
the import with the frames read of how many and the disk's seconds per frame and share of the time (``reading``), the
total and the processes the steps ran with (``workers``); a note when reading the frames from disk was the slow part;
then ``REPORT.md`` rendered as Markdown (its images are found in the result folder). An older result without these
keys has the rows it can. **Logs**: the ``analyze_*.log`` of every step, one at a time.

Without a result each tab says in one line what to do; texts are read again when another result is shown.
"""

from __future__ import annotations

import html
from typing import Optional

from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.i18n import tr

from ..application import ResultTexts, disk_slow
from .client_panel import TILE_DIGITS, mask_text
from .client_texts import entry_name, shown_number
from .evolution_view import RELATIVE_RADIUS
from .live_panel import REUSED_LINE, STEP_NAMES
from .live_static import STATIC_STEP_NAMES
from .reading_texts import PROCESSES, SLOW_DISK_NOTE, clock_time, disk_words
from .texts import message_text, text
from .views.sputter_main_view import TILES

COLUMN_TIME, COLUMN_SAMPLE, COLUMN_NOMINAL = "Time (s)", "Sample", "Nominal thickness (nm)"
COLUMN_DEVIATION, COLUMN_SOLUTION = "Fit deviation", "Solution"
NO_SOLUTION_CELL = "no solution: {reason}"
FACT_NETWORK = "Network used: {family}"
FACT_TRIED = "Networks tried (fit deviation): {tried}"
FACT_MODE = "Mode: {mode}"
FACT_STATUS = "Status: {status}"
FACT_TIME = "Run time: {time}"
MINUTES = "{minutes} min"
SECONDS = "{seconds} s"
NO_REPORT_TEXT = "This result has no report (REPORT.md)."
NO_LOGS = "This result has no log files."
STEPS_TITLE = "Time per step"
COLUMN_STEP, COLUMN_STEP_TIME, TOTAL_ROW = "Step", "Time (m:ss)", "All steps"
FRAMES_READ = "{read} of {total} frames read"
FRAMES_READ_ONLY = "{read} frames read"
TEMPLATES = (COLUMN_TIME, COLUMN_SAMPLE, COLUMN_NOMINAL, COLUMN_DEVIATION, COLUMN_SOLUTION, NO_SOLUTION_CELL,
             FACT_NETWORK, FACT_TRIED, FACT_MODE, FACT_STATUS, FACT_TIME, MINUTES, NO_REPORT_TEXT, NO_LOGS,
             STEPS_TITLE, COLUMN_STEP, COLUMN_STEP_TIME, TOTAL_ROW, FRAMES_READ, FRAMES_READ_ONLY)
"""The texts of the three tabs that need Chinese."""


def run_time(seconds: float) -> str:
    return text(MINUTES, minutes=f"{seconds / 60:.1f}") if seconds >= 120 else text(SECONDS, seconds=f"{seconds:.0f}")


def facts_html(analysis: dict, mask_words: str) -> str:
    """The facts of ``analysis.json`` as an HTML list ("" when there are none)."""
    lines = []
    if analysis.get("family"):
        lines.append(text(FACT_NETWORK, family=html.escape(str(analysis["family"]))))
    tried = analysis.get("families_tried")
    if isinstance(tried, dict) and tried:
        parts = [f"{html.escape(str(name))} ({shown_number(value) if isinstance(value, (int, float)) else '–'})"
                 for name, value in sorted(tried.items(), key=lambda item: _order(item[1]))]
        lines.append(text(FACT_TRIED, tried=", ".join(parts)))
    if analysis.get("mode"):
        lines.append(text(FACT_MODE, mode=html.escape(str(analysis["mode"]))))
    if analysis.get("status") and analysis["status"] != "ok":
        lines.append(text(FACT_STATUS, status=html.escape(str(analysis["status"]))))
    if mask_words:
        lines.append(html.escape(mask_words))
    steps = [step for step in analysis.get("steps") or () if isinstance(step, dict)]
    seconds = sum(float(step.get("seconds") or 0) for step in steps if isinstance(step.get("seconds"), (int, float)))
    if steps:
        lines.append(text(FACT_TIME, time=run_time(seconds)))
    return "<ul style='margin-left:0px'>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>" if lines else ""


def _order(value) -> float:
    return float(value) if isinstance(value, (int, float)) else float("inf")


def _number(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _step_name(step: str, static: bool) -> str:
    """The name of a step of ``analysis.json`` in the interface language: ``predict (refine)`` is *Find the
    solutions (refine)* (the part in brackets — a network, ``refine``, ``main`` — is the package's own word)."""
    base, _bracket, tag = str(step).partition(" (")
    names = STATIC_STEP_NAMES if static else STEP_NAMES
    name = tr(names[base]) if base in names else base
    return html.escape(f"{name} ({tag}" if tag else name)


def _import_words(step: dict, reading: dict, workers: Optional[int]) -> str:
    """What the import row adds: read before, frames read of how many, the disk's seconds per frame and share, and
    the processes of the import when they differ from those of the steps."""
    parts = []
    if not step.get("log") and "reused" in str(step.get("note") or ""):
        parts.append(tr(REUSED_LINE))
    read, total = _number(reading.get("frames_read")), _number(reading.get("frames_total"))
    if read is not None:
        parts.append(text(FRAMES_READ, read=f"{int(read):,}", total=f"{int(total):,}") if total is not None else
                     text(FRAMES_READ_ONLY, read=f"{int(read):,}"))
    disk = disk_words(_number(reading.get("read_s_per_frame")), _number(reading.get("read_share")))
    if disk:
        parts.append(disk)
    processes = _number(reading.get("workers"))
    if processes and int(processes) != workers:
        parts.append(text(PROCESSES, count=int(processes)))
    return " · ".join(parts)


def _reading(analysis: dict) -> dict:
    return analysis.get("reading") if isinstance(analysis.get("reading"), dict) else {}


def _workers(analysis: dict) -> Optional[int]:
    found = _number(analysis.get("workers"))
    return int(found) if found else None


def steps_html(analysis: dict, static: bool = False) -> str:
    """Time per step: each step of ``analysis.json`` with its time (m:ss) — the import with how its frames were
    read —, the total and the processes the steps ran with, as HTML ("" when the analysis lists no step and says
    nothing of its processes)."""
    steps = [step for step in analysis.get("steps") or () if isinstance(step, dict) and step.get("step")]
    reading, workers = _reading(analysis), _workers(analysis)
    rows, total = [], 0.0
    for step in steps:
        seconds = _number(step.get("seconds"))
        total += seconds or 0.0
        notes = _import_words(step, reading, workers) if str(step["step"]) == "intake" else ""
        rows.append((_step_name(step["step"], static), clock_time(seconds) if seconds is not None else "–",
                     html.escape(notes)))
    if rows:
        rows.append((f"<b>{html.escape(tr(TOTAL_ROW))}</b>", f"<b>{clock_time(total)}</b>", ""))
    lines = []
    if rows:
        gap = "&nbsp;&nbsp;"
        cells = "".join(f"<tr><td>{name}</td><td align='right'>{gap}{time_}</td><td>{gap}{notes}</td></tr>"
                        for name, time_, notes in rows)
        lines.append(f"<b>{html.escape(tr(STEPS_TITLE))}</b><table cellspacing='0' cellpadding='1'>"
                     f"<tr><th align='left'>{html.escape(tr(COLUMN_STEP))}</th>"
                     f"<th align='right'>&nbsp;&nbsp;{html.escape(tr(COLUMN_STEP_TIME))}</th><th></th></tr>{cells}"
                     "</table>")
    processes = workers or (int(_number(reading.get("workers")) or 0) or None)
    if processes:
        lines.append(html.escape(text(PROCESSES, count=processes)))
    return "<br>".join(lines)


def reading_was_slow(analysis: dict) -> bool:
    """Reading the frames from disk was the slow part of the import that made the result's scan."""
    reading = _reading(analysis)
    return disk_slow(_number(reading.get("read_s_per_frame")), _number(reading.get("read_share")),
                     reading.get("slow"))


class ResultTabsMixin:
    """Needs the tabs of ``views/sputter_page_view.py``, ``service``, ``client_result`` and ``client_choices``."""

    def _init_result_tabs(self) -> None:
        self._result_texts: Optional[ResultTexts] = None
        self.log_choice.currentIndexChanged.connect(self._show_log)
        self.show_result_details()

    def show_result_details(self) -> None:
        """The three tabs for the result shown (read again from its folder)."""
        result = self.client_result
        self._result_texts = None
        if result is not None and self.service.client is not None:
            try:
                self._result_texts = self.service.client.texts(result)
            except (OSError, ValueError):
                self._result_texts = ResultTexts()
        for tab in (self.tab_windows, self.tab_report, self.tab_log):
            tab.setCurrentIndex(0 if result is None else 1)
        self.refresh_windows_table()
        self._show_report_tab()
        self._fill_logs()

    # -- Windows -------------------------------------------------------------------------------------------------
    def _windows_columns(self) -> list:
        """(key, header) of the table: the position, the tiles the result has a value for, the fit and its name."""
        result = self.client_result
        first = [("sample", tr(COLUMN_SAMPLE))] if result.static else [("time", tr(COLUMN_TIME))]
        columns = first + [("nominal", tr(COLUMN_NOMINAL))]
        chosen = [window.chosen(self.client_choices) for window in result.windows]
        for key, caption, unit, _tip in TILES:
            if any(entry is not None and entry.value(key) is not None for entry in chosen):
                name = tr(RELATIVE_RADIUS if result.porous and key == "R_nm" else caption)
                columns.append((key, f"{name} ({unit})" if unit else name))
        return columns + [("deviation", tr(COLUMN_DEVIATION)), ("solution", tr(COLUMN_SOLUTION))]

    def refresh_windows_table(self) -> None:
        table, result = self.windows_table, self.client_result
        table.clear()
        if result is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        columns = self._windows_columns()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([header for _key, header in columns])
        table.setRowCount(len(result.windows))
        for row, window in enumerate(result.windows):
            entry = window.chosen(self.client_choices)
            for column, (key, _header) in enumerate(columns):
                if key == "time":
                    words = shown_number(window.time_s, 4)
                elif key == "sample":
                    words = window.label or shown_number(window.time_s)
                elif key == "nominal":
                    words = shown_number(window.nominal_nm)
                elif entry is None:
                    reason = message_text(window.error) if window.error else ""
                    words = text(NO_SOLUTION_CELL, reason=reason) if key == "solution" else ""
                elif key == "deviation":
                    words = shown_number(entry.fit_deviation)
                elif key == "solution":
                    words = entry_name(entry.label)
                else:
                    words = shown_number(entry.value(key), TILE_DIGITS.get(key, 3))
                table.setItem(row, column, QTableWidgetItem(words))
        table.resizeColumnsToContents()

    # -- Report ----------------------------------------------------------------------------------------------------
    def _show_report_tab(self) -> None:
        found, result = self._result_texts, self.client_result
        if found is None or result is None:
            self.report_facts.clear()
            self.report_steps.clear()
            self.report_disk_note.setVisible(False)
            self.report_view.clear()
            return
        self.report_facts.setText(facts_html(found.analysis, mask_text(result.mask)))
        self.report_facts.setVisible(bool(self.report_facts.text()))
        self.report_steps.setText(steps_html(found.analysis, result.static))
        self.report_steps.setVisible(bool(self.report_steps.text()))
        slow = reading_was_slow(found.analysis)
        if slow:
            self.report_disk_note.message_label.setText(tr(SLOW_DISK_NOTE))
        self.report_disk_note.setVisible(slow)
        self.report_view.setSearchPaths([str(result.folder)])
        if found.report:
            self.report_view.setMarkdown(found.report)
        else:
            self.report_view.setPlainText(tr(NO_REPORT_TEXT))

    # -- Log -------------------------------------------------------------------------------------------------------
    def _fill_logs(self) -> None:
        logs = self._result_texts.logs if self._result_texts is not None else ()
        self.log_choice.blockSignals(True)
        self.log_choice.clear()
        for name, _words in logs:
            self.log_choice.addItem(name, name)
        self.log_choice.blockSignals(False)
        self.log_choice.setVisible(bool(logs))
        self._show_log()

    def _show_log(self, _index: int = 0) -> None:
        logs = dict(self._result_texts.logs) if self._result_texts is not None else {}
        if self.client_result is not None and not logs:
            self.log_view.setPlainText(tr(NO_LOGS))
            return
        self.log_view.setPlainText(logs.get(self.log_choice.currentData(), ""))

    def refresh_result_tabs_language(self) -> None:
        self.refresh_windows_table()
        self._show_report_tab()
        if self.client_result is not None and not (self._result_texts and self._result_texts.logs):
            self._show_log()


__all__ = ["ResultTabsMixin", "TEMPLATES", "facts_html", "reading_was_slow", "run_time", "steps_html"]
