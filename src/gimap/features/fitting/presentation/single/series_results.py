"""The Results step of the In-situ series: the table of every frame, what the run came to, and saving.

The table gets one row when a frame is done (filling the whole table again at every frame, with columns
that measured themselves at every cell, froze the window for minutes) and keeps its columns as wide as
their text: the frame, χ²ᵣ, every fitted value ± 1σ, then the points the frame was fitted on and what they
are. A frame without a fit says why across its row; a frame not to be trusted as it is (not converged, a
poor fit, stopped) is in the warning colour, with the reason on hover. The card above says how many frames
are in each state and what they were fitted on. Saving writes the table with a JSON record of the run, or a
plot with a title that says what it shows; results that were not saved are asked about before they go.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, QTimer
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

from src.gimap.app.presentation.i18n import current_language, tr, trf
from src.gimap.app.presentation.theme import theme_manager

from ...application.single_fit import FAMILIES
from ...application.series_fit import (
    POOR_CHI2,
    file_name,
    parameter_columns,
    poor_fit,
    series_counts,
    series_record,
    series_table,
)
from ..views.fit_series_view import RESULTS_WIDTH
from .results import folder_action, start_folder, write_pair, write_plot
from .series_stages import CHI2
from .series_text import HALF_TEXT, STATE_COLOR, STATE_MARK
from .session import path_name

STALE = "Changed after the fit: Start again"
"""The halves, the fitting range or the left-out points are no longer those of the run."""
UNTOUCHED = "The model was the starting guess of Single analysis: it was never set up or fitted there."
PLOTS_MIN_WIDTH = 460
"""The plots keep at least this width when the steps panel widens for the table."""
NAMED_FRAMES = 8
"""How many frame numbers a sentence lists before it says “…”."""


def frame_numbers(indices) -> str:
    """“3, 7, 11”: frame numbers for a sentence (the first few of many)."""
    numbers = [str(index + 1) for index in sorted(indices)]
    return ", ".join(numbers[:NAMED_FRAMES]) + (" …" if len(numbers) > NAMED_FRAMES else "")


class FitSeriesResultsMixin:
    """Needs the series page: ``fits``, ``paths``, ``folder``, ``_start_model``, ``_settings``, ``_unit``,
    ``_frame_state()``, ``_select()``, ``_status()``, the widgets of the Results step."""

    def _connect_results(self) -> None:
        self._table_frames: list[int] = []
        """The frame of every row of the table, in ascending order."""
        self._table_model = None
        self._table_language = None
        self._unsaved = False
        """Frames were fitted since the table was last saved."""
        self._untouched_start = False
        """The run started from a model nobody set up (the starting guess of Single analysis)."""
        self._side_from = ""
        self._run_listing: Optional[dict] = None
        self._narrow_sizes = None
        """The splitter as the person left it, while the Results step has widened the steps panel."""
        self._width_timer = QTimer(self)
        self._width_timer.setSingleShot(True)
        self._width_timer.setInterval(0)
        self._width_timer.timeout.connect(self._refit_steps_width)
        self._moved_on_results = False
        """The person set the splitter while Results was shown: a resized window leaves it alone."""
        self._asking = False
        self.results_table.currentCellChanged.connect(lambda row, *_: self._select_table_row(row))
        self.trend_combo.currentIndexChanged.connect(lambda _index: self._show_trend_column())
        self.to_single_button.clicked.connect(self._open_in_single)
        self.save_table_action.triggered.connect(lambda: self.save_table())
        self.save_trend_action.triggered.connect(lambda: self._save_plot(self.trend_plot, "trend"))
        self.save_frame_action.triggered.connect(lambda: self._save_plot(self.frame_plot, "frame"))
        self.splitter.splitterMoved.connect(self._splitter_moved)

    # -- the table --------------------------------------------------------------------------

    def _table_is_stale(self) -> bool:
        """The table is not of these fits (another run, a project, a language): to be filled again."""
        rows = len(self.fits) if self._start_model is not None else 0
        return (self._table_model is not self._start_model or len(self._table_frames) != rows
                or self._table_language != current_language())

    def _rebuild_results(self) -> None:
        """The whole table again: at Start, after a project or a switch of the language or the theme."""
        table, model = self.results_table, self._start_model
        table.setUpdatesEnabled(False)
        blocker = QSignalBlocker(table)
        try:
            table.clearSpans()
            table.setRowCount(0)
            self._table_frames = []
            if model is not None:
                columns = parameter_columns(model)
                table.setColumnCount(len(columns) + 4)
                table.setHorizontalHeaderLabels(["#", "χ²ᵣ"] + [path_name(model, path, unit=True) for path, _name in columns]
                                                + [tr("Points"), tr("Halves")])
                for column, (path, _name) in enumerate(columns, start=2):  # the whole name (“1·Sphere R (nm)”) on hover
                    table.horizontalHeaderItem(column).setToolTip(path_name(model, path, unit=True, full=True))
                table.horizontalHeaderItem(len(columns) + 2).setToolTip(tr("The points the frame was fitted on"))
                table.horizontalHeaderItem(len(columns) + 3).setToolTip(
                    tr("What those points are: the mean of both halves of the cut, or one half"))
                header = table.horizontalHeader()
                for column in range(table.columnCount()):
                    table.setColumnWidth(column, header.sectionSizeHint(column))
                self._table_frames = sorted(self.fits)
                table.setRowCount(len(self._table_frames))
                for row, index in enumerate(self._table_frames):
                    self._fill_row(row, self.fits[index], columns)
        finally:
            del blocker
            table.setUpdatesEnabled(True)
        self._table_model, self._table_language = model, current_language()
        self._mark_table_row(self._selected)

    def _show_result_row(self, frame) -> None:
        """One frame's row: added where it belongs, or made again (a curve fitted again by Watch)."""
        rows = len(self._table_frames)
        if self._table_model is not self._start_model or rows not in (len(self.fits), len(self.fits) - 1) \
                or self._table_language != current_language():
            self._rebuild_results()
            return
        table = self.results_table
        row = bisect_left(self._table_frames, frame.index)
        blocker = QSignalBlocker(table)
        if row == rows or self._table_frames[row] != frame.index:
            table.insertRow(row)
            self._table_frames.insert(row, frame.index)
        self._fill_row(row, frame, parameter_columns(self._start_model))
        del blocker

    def _fill_row(self, row: int, frame, columns) -> None:
        table = self.results_table
        last = table.columnCount() - 1
        if table.columnSpan(row, 1) > 1:
            table.setSpan(row, 1, 1, 1)
        if not frame.ok:  # no values: why, across the row
            cells = [str(frame.index + 1), f"{STATE_MARK['failed']} {self._reason_text(frame.error)}"]
        else:
            result = frame.result
            cells = [str(frame.index + 1), f"{result.chi2_reduced:.3g}" if np.isfinite(result.chi2_reduced) else "—"]
            for path, _name in columns:
                value, error = result.model.get(path).value, result.errors.get(path, np.nan)
                cells.append("—" if not np.isfinite(value) else
                             f"{value:.4g}" + ("" if not np.isfinite(error) else f" ± {error:.2g}"))
            cells += [str(frame.points), tr(HALF_TEXT.get(frame.halves_used, frame.halves_used))]
        token = STATE_COLOR.get(self._frame_state(frame.index)[0])
        brush = None if token is None else QBrush(theme_manager().color(token))
        tip = self._frame_tip(frame)
        for column in range(last + 1):
            item = QTableWidgetItem(cells[column] if column < len(cells) else "")
            item.setToolTip(tip)
            if brush is not None:
                item.setForeground(brush)
            table.setItem(row, column, item)
        if not frame.ok:
            table.setSpan(row, 1, 1, last)
        model = table.model()
        for column in range(last + 1) if frame.ok else (0,):  # a reason spans the row: it widens no column
            width = table.sizeHintForIndex(model.index(row, column)).width() + 2
            if width > table.columnWidth(column):
                table.setColumnWidth(column, width)

    def _frame_tip(self, frame) -> str:
        """What a row is, on hover: the file, the points and what they are, and why it is not a plain result."""
        lines = [f"{frame.index + 1} · {file_name(frame.path, self.folder)}"]
        if frame.points:
            made_of = trf("{points} points", points=frame.points) + " · " + tr(HALF_TEXT.get(frame.halves_used, frame.halves_used))
            if frame.halves_used == "mean" and frame.averaged < frame.points:
                made_of += " (" + trf("{single} from one half alone", single=frame.points - frame.averaged) + ")"
            if frame.left_out:
                made_of += " · " + trf("{count} left out", count=frame.left_out)
            lines.append(made_of)
        state, detail = self._frame_state(frame.index)
        if state != "done" and detail:
            lines.append(detail)
        if frame.ok and frame.state != "ok" and frame.reason:
            lines.append(tr(frame.reason))
        return "\n".join(lines)

    def _select_table_row(self, row: int) -> None:
        if 0 <= row < len(self._table_frames):
            self._select(self._table_frames[row], follow=True)

    def _mark_table_row(self, index) -> None:
        """The table's current row follows the frame that is shown (selected in the list, or just fitted)."""
        table = self.results_table
        row = bisect_left(self._table_frames, index) if index is not None else len(self._table_frames)
        if row < len(self._table_frames) and self._table_frames[row] == index and table.currentRow() != row:
            blocker, bar = QSignalBlocker(table), table.horizontalScrollBar()
            across = bar.value()
            table.setCurrentCell(row, max(table.currentColumn(), 0))
            bar.setValue(across)  # the row comes into view; the columns stay where the person scrolled them
            del blocker

    def _show_trend_column(self) -> None:
        """The column of the value the trend follows comes into view."""
        key, model, table = self.trend_combo.currentData(), self._start_model, self.results_table
        if model is None or not table.rowCount():
            return
        column = 1 if key == CHI2 else next(
            (position for position, (path, _name) in enumerate(parameter_columns(model), start=2) if path == key), None)
        if column is None:
            return
        # By the column's place, not by a cell: the cell of a row without a fit spans the table.
        left, bar = table.columnViewportPosition(column), table.horizontalScrollBar()
        beyond = left + table.columnWidth(column) - table.viewport().width()
        if left < 0 or beyond > 0:
            bar.setValue(bar.value() + (left if left < 0 else min(beyond, left)))

    def _fit_steps_width(self, key: str) -> None:
        """The steps panel is wide enough for the table while Results is shown — half of a wide window — and
        as the person left it on the other steps (also when there is no room: the plots keep theirs)."""
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if len(sizes) != 2 or total <= 0:
            return
        own = self._narrow_sizes[0] if self._narrow_sizes is not None else sizes[0]
        wide = max(RESULTS_WIDTH, total // 2)
        if key == "results" and own < wide and total - wide >= PLOTS_MIN_WIDTH:
            if self._narrow_sizes is None:
                self._narrow_sizes = sizes
            self.splitter.setSizes([wide, total - wide])
        elif self._narrow_sizes is not None:
            self.splitter.setSizes([own, total - own])
            self._narrow_sizes = None

    def _splitter_moved(self, *_args) -> None:
        """The person's own width: it holds on every step, and is not widened again while Results stays shown."""
        self._narrow_sizes = None
        self._moved_on_results = self.step_rail.current() == "results"

    def _refit_steps_width(self) -> None:
        """After the window was resized (once the splitter has its new width)."""
        if not self._moved_on_results:
            self._fit_steps_width(self.step_rail.current())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        timer = getattr(self, "_width_timer", None)  # (none yet while the page is being built)
        if timer is not None and (self._narrow_sizes is not None or self.step_rail.current() == "results"):
            timer.start()

    # -- what the run came to ---------------------------------------------------------------

    def _results_stale(self) -> bool:
        """A run is done, and the points it fitted are no longer those the page would fit now."""
        if self.running or self._start_model is None or not self.fits:
            return False
        now, run = self._stage_settings(), self._settings
        return (now.side, now.q_range, now.excluded) != (run.side, run.q_range, run.excluded)

    def _poor_frames(self) -> list[int]:
        return [index for index, frame in self.fits.items() if poor_fit(frame)]

    def _counts_text(self) -> str:
        """“17 fitted · 1 failed”: the frames by what became of them (only the states that occur)."""
        counts = series_counts(list(self.fits.values()))
        parts = [trf("{count} fitted", count=counts["ok"] + counts["not_converged"])]
        for key, template in (("failed", "{count} failed"), ("not_converged", "{count} not converged"),
                              ("stopped", "{count} stopped")):
            if counts[key]:
                parts.append(trf(template, count=counts[key]))
        poor = self._poor_frames()
        if poor:
            parts.append(trf("{count} poor fits", count=len(poor)))
        return " · ".join(parts)

    def _points_text(self, settings, source: str) -> str:
        """“q < 0 half (as in Single analysis); |q| 0.1–1.5 nm⁻¹ · 3 left out”: the points of a run."""
        return trf("{side}; {range}", side=self.halves_text(settings.side, source), range=self._range_text(settings))

    def _fitted_on_text(self) -> str:
        """The halves, range and left-out points of the run, and how many points its frames were fitted on."""
        text = self._points_text(self._settings, self._side_from)
        points = [frame.points for frame in self.fits.values() if frame.ok]
        if points:
            low, high = min(points), max(points)
            text += " · " + (trf("{points} points per frame", points=low) if low == high else
                             trf("{low}–{high} points per frame", low=low, high=high))
        return text

    def _refresh_results(self, listed: list) -> None:
        """The Results step's state in the rail, its card, and the table when it is of other fits."""
        if self._table_is_stale():
            self._rebuild_results()
        frames = list(self.fits.values())
        started = self._start_model is not None
        self.save_button.setVisible(bool(self.fits))
        self.to_single_button.setVisible(bool(self.fits))
        if not started:
            self.results_summary.setText(tr("Choose the curves, check the start, then Start."))
            self.step_rail.set_state("results", "pending", tr("After Start"))
            return
        counts = series_counts(frames)
        poor, stale = self._poor_frames(), self._results_stale()
        lines = [self._counts_text(), self._fitted_on_text()]
        if poor:
            lines.append(trf("Poor fit (χ²ᵣ above {limit}): frames {frames}", limit=f"{POOR_CHI2:g}", frames=frame_numbers(poor)))
        if self._untouched_start:
            lines.append(tr(UNTOUCHED))
        if stale:
            lines.append(tr(STALE))
        self.results_summary.setText("\n".join(lines))
        if self.running:
            waiting = self._waiting_for_curves()
            self.step_rail.set_state("results", "busy", trf("{done} frames · waiting for new curves", done=len(frames))
                                     if waiting else trf("{done} of {total} frames", done=self._frames_done,
                                                         total=self._frames_to_do()))
        elif not frames:
            self.step_rail.set_state("results", "pending", tr("After Start"))
        else:
            trouble = counts["failed"] or counts["not_converged"] or counts["stopped"] or poor or self._untouched_start
            self.step_rail.set_state("results", "warn" if trouble or stale else "ok", tr(STALE) if stale else lines[0])

    def _open_in_single(self) -> None:
        frame = self.fits.get(self._selected) if self._selected is not None else None
        if frame is None:
            self._status(tr("Select a fitted frame first."), "warning")
            return
        self.openFrameRequested.emit(frame.path, frame.result.model if frame.ok else None)

    # -- results that were not saved --------------------------------------------------------

    def _may_drop_results(self) -> bool:
        """Before the fits go (a new Start, other curves): asks when frames were fitted and not saved.
        ``False``: keep them (Cancel, or Save that did not write)."""
        if self._asking:
            return False
        if not self._unsaved or not any(frame.fitted for frame in self.fits.values()):
            return True  # nothing fitted to the end: where a stopped fit was is no result to save
        self._asking = True
        try:
            answer = QMessageBox.question(
                self, tr("In-situ series"),
                tr("The results of the last run are not saved.\nSave the table of every frame first?"),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        finally:
            self._asking = False
        if answer == QMessageBox.Save:
            return self.save_table() is not None
        return answer == QMessageBox.Discard

    # -- saving -------------------------------------------------------------------------------

    def save_table(self, path=None):
        if not self.fits or self._start_model is None:
            return None
        path = path or QFileDialog.getSaveFileName(self, tr("Save Table of Every Frame"),
                                                   str(start_folder(self.folder) / "series_fit.csv"), "CSV (*.csv)")[0]
        if not path:
            return None
        frames = list(self.fits.values())
        header, rows = series_table(frames, self._start_model, details=True, folder=self.folder)

        def write(target) -> None:
            with open(target, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)

        listing = dict(self._run_listing or {})
        if self._untouched_start:
            listing["model"] = UNTOUCHED
        record = series_record(self._start_model, self._settings, self.folder, self.pattern_edit.text(), frames,
                               q_unit=self._unit, side_from=self._side_from, listing=listing)
        try:
            write_pair(path, write, record)  # both, or neither
        except OSError as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._unsaved = False
        self._status(lambda name=Path(path).name: tr("Saved {name} and its record.").format(name=name), "ok",
                     action=folder_action(path))
        return path

    def _plot_words(self, name: str) -> tuple[str, str]:
        """``(file name, title)`` of a saved plot: which value through which series, or which frame — and on
        which points. The words of the title are English, as the axes of a figure are."""
        folder = self._folder_name()
        if name == "trend":
            key, model = self.trend_combo.currentData(), self._start_model
            if model is None or not any(frame.ok for frame in self.fits.values()) or key is None:
                return "series_trend", ""
            column = "chi2_reduced" if key == CHI2 else dict(parameter_columns(model)).get(key, "value")
            value = "χ²ᵣ" if key == CHI2 else path_name(model, key, unit=True, translate=False, full=True)
            return f"series_trend_{column}", f"{value} · {folder} · {model_label_plain(model)}"
        index = self._selected
        if index is None or not 0 <= index < len(self.paths) or not self.frame_plot.has_curves():
            return "series_frame", ""
        return f"series_frame_{index + 1:05d}", f"{index + 1} · {file_name(self.paths[index], self.folder)}"

    def _save_plot(self, plot, name: str, path=None):
        start = start_folder(self.folder, self._remembered("folder"))  # no series yet: not the program's folder
        stem, title = self._plot_words(name)
        path = path or QFileDialog.getSaveFileName(self, tr("Save Plot"), str(start / f"{stem}.png"),
                                                   "PNG image (*.png);;SVG vector (*.svg)")[0]
        if not path:
            return None
        try:
            if title:
                plot.plot.setTitle(title)  # in the picture only: on the page the plot's header says it
                plot.plot.layout.activate()  # its row now, not at the next turn of the event loop (half a title)
            write_plot(plot.plot, path)
        except OSError as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        finally:
            if title:
                plot.plot.setTitle(None)
        self._status(lambda name=Path(path).name: tr("Saved {name}.").format(name=name), "ok", action=folder_action(path))
        return path


def model_label_plain(model) -> str:
    """The families of a model in English (the title of a figure), whatever the interface language."""
    return " + ".join(FAMILIES[component.family][0] for component in model.components) or "no particle"


__all__ = ["FitSeriesResultsMixin", "PLOTS_MIN_WIDTH", "STALE", "UNTOUCHED", "frame_numbers", "model_label_plain"]
