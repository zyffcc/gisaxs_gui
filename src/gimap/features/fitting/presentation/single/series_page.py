"""The In-situ series page of Fitting: every curve of a folder fitted with the model of Single analysis.

Curves → Start → Results. The model, the halves, the fitting range and the left-out points are those
of Single analysis (one representative curve is set up there); each frame starts from the previous
frame's result, or every frame from that model. Frames are fitted one after another in the
background — the list shows each one's state, the right side the selected frame with its fit and
the trend of a parameter through the series — with Pause and Stop, and new curves can be fitted as
they appear (Watch). Save: the table of every frame (values, errors, χ²ᵣ) with a JSON record.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QTableWidgetItem, QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import DiscoverInSituFramesRequest, LoadCurveRequest
from ...application.series_fit import (
    FrameFit,
    SeriesSettings,
    fit_frame,
    next_start,
    parameter_columns,
    series_record,
    series_table,
)
from ...application.single_fit import Curve
from ..views.fit_series_view import FitSeriesView
from .drops import FolderDropMixin, SplitterMemoryMixin
from .results import folder_action, write_pair, write_plot
from .series_preview import FitSeriesPreviewMixin
from .series_stages import FitSeriesStagesMixin
from .session import model_label, path_name

PREFERENCES_KEY = "fitting_series_page"
WATCH_MS = 2000
STATE_MARK = {"waiting": "·", "running": "▸", "done": "✓", "warn": "!", "failed": "✗"}
FIGURE_SPACE = "\u2007"
"""Pads the frame numbers: as wide as a digit (a space is narrower)."""


class FitSeriesPage(FolderDropMixin, SplitterMemoryMixin, FitSeriesStagesMixin, FitSeriesPreviewMixin, QWidget,
                    FitSeriesView):
    editModelRequested = pyqtSignal()
    """Show Single analysis (the model is set up there)."""
    openFrameRequested = pyqtSignal(str, object)
    """A frame's file and fitted model, to look at in Single analysis."""

    def __init__(self, view_model, single_page, *, preferences=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.view_model = view_model
        self.single = single_page
        self.preferences = preferences
        self.tasks = TaskRunner(self)
        self.setup_ui(self)
        self.folder = ""
        self.paths: list[str] = []
        self.fits: dict[int, FrameFit] = {}
        self.queue: list[int] = []
        self.running = False
        self._stop = threading.Event()
        self._start_model = None
        self._settings = SeriesSettings()
        self._unit = "angstrom"
        """The unit of q in the files, as Single analysis read its curve at Start (fixed for the run)."""
        self._previous: Optional[FrameFit] = None
        self._status_text = None
        """Makes the status line again in another interface language (``refresh_language``)."""
        self._curves: dict = {}
        self._preview_curves: dict = {}
        self._current = -1
        self._selected: Optional[int] = None
        self._last_done: Optional[int] = None
        self._watch = QTimer(self)
        self._watch.setInterval(WATCH_MS)
        self._watch.timeout.connect(self._look_for_new)
        self.setAcceptDrops(True)  # a folder dropped here is listed (``drops.py``)
        self._connect()
        self._connect_stages()
        self._restore()
        self.refresh()

    def _connect(self) -> None:
        self.step_rail.stepChosen.connect(self.show_step)
        self.folder_button.clicked.connect(self.choose_folder)
        self.pattern_edit.editingFinished.connect(self._list_curves)
        self.subfolders_check.toggled.connect(lambda _on: self._list_curves())
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.valueChanged.connect(lambda _value: self._frames_changed())
        self.start_button.clicked.connect(self.start)
        self.pause_button.toggled.connect(self._pause_toggled)
        self.stop_button.clicked.connect(self.stop)
        self.frame_list.currentRowChanged.connect(lambda row: self._select(self._listed()[row] if 0 <= row < len(self._listed()) else None))
        self.results_table.currentCellChanged.connect(lambda row, *_: self._select_table_row(row))
        self.trend_combo.currentIndexChanged.connect(lambda _index: self._draw_trend())
        self.trend_plot.log_check.toggled.connect(lambda _checked: self._draw_trend())  # the bars again
        self.edit_model_button.clicked.connect(self.editModelRequested)
        self.to_single_button.clicked.connect(self._open_in_single)
        self.save_table_action.triggered.connect(lambda: self.save_table())
        self.save_trend_action.triggered.connect(lambda: self._save_plot(self.trend_plot, "trend"))
        self.save_frame_action.triggered.connect(lambda: self._save_plot(self.frame_plot, "frame"))
        self.start_group.buttonToggled.connect(lambda *_: self._remember_choices())
        self.method_group.buttonToggled.connect(lambda *_: self._remember_choices())

    # -- steps --------------------------------------------------------------------------------

    def show_step(self, key: str) -> None:
        self.step_rail.set_current(key)
        self.step_stack.setCurrentWidget(self.step_pages[key])

    def refresh(self) -> None:
        """Where each step stands, and what the Start step will use."""
        listed = self._listed()
        self._schedule_stages()
        if not self.paths:
            self.step_rail.set_state("curves", "pending", tr("Choose the folder of the curves"))
            self.step_intro["curves"].setText(tr("A folder of curves, e.g. the gimap_analysis folder of Analyze's "
                                                 "Batch Export or Send Series to Fitting."))
        else:
            detail = tr("{listed} of {total} curves").format(listed=len(listed), total=len(self.paths))
            self.step_rail.set_state("curves", "ok", detail)
            self.step_intro["curves"].setText(detail + " · " + Path(self.folder).name)  # the whole path: in the bar
        model = self.single.session.model
        free = sum(1 for _path, parameter in model.parameters() if parameter.free)
        single = self.single.session
        range_text = tr("every point") if single.q_range is None else f"|q| {single.q_range[0]:.4g}–{single.q_range[1]:.4g} nm⁻¹"
        left_out = f" · {len(single.excluded)} " + tr("left out") if single.excluded else ""
        self.model_summary.setText(tr("{model} · {free} values fitted\n{side}; {range}{left_out}").format(
            model=model_label(model), free=free, side=tr(dict(SIDE_TEXT).get(single.side, single.side)),
            range=range_text, left_out=left_out))
        start = {"previous": tr("from the previous frame"), "same": tr("from the same model"),
                 "stages": tr("from the previous frame, afresh at each stage")}[self._start_choice()]
        self.step_rail.set_state("start", "ok" if model.components else "warn",
                                 f"{model_label(model)} · {start}" if model.components else tr("No model: set one in Single analysis"))
        self.step_intro["start"].setText(tr("The model of Single analysis, fitted to every frame."))
        self.step_intro["results"].setText(tr("Every frame: the quality of its fit and each value ± 1σ. Click a row "
                                              "to see that frame; the plot below follows a value through the series."))
        done = [frame for frame in self.fits.values()]
        failed = sum(1 for frame in done if not frame.ok)
        if self.running:
            self.step_rail.set_state("results", "busy", tr("{done} of {total} frames").format(done=len(done), total=len(listed)))
        elif done:
            self.step_rail.set_state("results", "warn" if failed else "ok",
                                     tr("{done} frames, {failed} failed").format(done=len(done), failed=failed))
        else:
            self.step_rail.set_state("results", "pending", tr("After Start"))
        self.start_button.setVisible(not self.running)
        self.start_button.setEnabled(bool(listed) and bool(model.components))
        self.start_button.setToolTip(
            tr("Fit every listed curve with the model of Single analysis") if self.start_button.isEnabled() else
            tr("Choose a folder of curves first") if not listed else tr("Set up a model in Single analysis first"))
        self.pause_button.setVisible(self.running)
        self.stop_button.setVisible(self.running)
        self.save_button.setVisible(bool(self.fits))
        self.to_single_button.setVisible(bool(self.fits))
        self._fill_results()
        self._preview_trend()
        self._redraw_preview()

    # -- the curves ---------------------------------------------------------------------------

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Folder of the Curves"), self.folder or str(self._remembered("folder") or ""))
        if folder:
            self.open_series(folder)

    def open_series(self, folder, pattern: Optional[str] = None) -> bool:
        """List the curves of ``folder`` (Analyze ▸ Send Series to Fitting)."""
        if self.running:
            self._status(tr("Stop the running series first."), "warning")
            return False
        self.folder = str(folder)
        if pattern:
            self.pattern_edit.setText(pattern)
        parts = Path(self.folder).parts
        self.folder_chip.setText(self.folder if len(parts) <= 4 else str(Path("…", *parts[-3:])))
        self.folder_chip.setToolTip(self.folder)
        self._remember(folder=self.folder, pattern=self.pattern_edit.text())
        return self._list_curves()

    def _list_curves(self) -> bool:
        if not self.folder or self.running:
            return False
        try:
            frames = self.view_model.storage.discover_insitu_frames(DiscoverInSituFramesRequest(
                Path(self.folder), self.pattern_edit.text().strip() or "*_fit_input.dat", self.subfolders_check.isChecked()))
        except (OSError, ValueError, RuntimeError) as exc:
            self._status(lambda reason=str(exc): tr("Could not list the curves: {reason}").format(reason=reason), "error")
            return False
        paths = [str(frame.path) for frame in frames]
        if paths != self.paths:
            self._forget_stages()  # those of the curves listed before (Start would leave out their odd frames)
        self.paths = paths
        self.fits, self._curves, self._selected, self._start_model = {}, {}, None, None
        self._preview_curves = {}
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(True)
            spin.setRange(1, max(1, len(self.paths)))
        self.first_spin.setValue(1)
        self.last_spin.setValue(max(1, len(self.paths)))
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(False)
        self._frames_changed()
        self._select_first()
        if not self.paths:
            self._status(lambda pattern=self.pattern_edit.text(), folder=self.folder: tr(
                "No curves matching {pattern} in {folder}.").format(pattern=pattern, folder=folder), "warning")
        else:
            self._status(lambda count=len(self.paths): tr("{count} curves listed.").format(count=count))
        self._schedule_stages()
        return bool(self.paths)

    def _listed(self) -> list[int]:
        """The frames to fit (indices into ``paths``): first to last, every n-th."""
        if not self.paths:
            return []
        first, last = self.first_spin.value() - 1, min(self.last_spin.value(), len(self.paths))
        return list(range(first, last, max(1, self.every_spin.value())))

    def _frames_changed(self) -> None:
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for index in self._listed():
            self.frame_list.addItem(QListWidgetItem(self._item_text(index)))
        self._mark_stages()
        self._keep_selection()
        self.frame_list.blockSignals(False)
        self.refresh()

    def _frame_state(self, index: int) -> tuple[str, str]:
        """``(state, detail)``: waiting, running, done, warn (not converged) or failed, and χ²ᵣ or the error."""
        frame = self.fits.get(index)
        if frame is None:
            return ("running" if self.running and index == self._current else "waiting"), ""
        if not frame.ok:
            return "failed", tr(frame.error)  # shown in the interface language; the record keeps it as it came
        return ("done" if frame.result.converged else "warn"), f"χ²ᵣ {frame.result.chi2_reduced:.3g}"

    def _item_text(self, index: int) -> str:
        state, detail = self._frame_state(index)
        number = str(index + 1).rjust(len(str(len(self.paths))), FIGURE_SPACE)  # as wide as a digit: names line up
        return f"{STATE_MARK[state]} {number}  {Path(self.paths[index]).name}" + (f"  —  {detail}" if detail else "")

    # -- the run ------------------------------------------------------------------------------

    def start(self) -> None:
        listed = self._without_odd(self._listed())
        model = self.single.session.model
        if not listed or not model.components:
            return
        self._settings = SeriesSettings(
            side=self.single.session.side, q_range=self.single.session.q_range,
            excluded=frozenset(self.single.session.excluded),
            method="global" if self.method_global.isChecked() else "local",
            start=self._start_choice())
        curve = self.single.session.curve  # the unit too is read once: another curve opened meanwhile changes nothing
        self._unit = curve.source_unit if curve is not None else "angstrom"
        self._start_model = model
        self.fits, self.queue, self._previous, self._curves = {}, list(listed), None, {}
        self._selected = self._last_done = None
        self._stop.clear()
        self.running = True
        self.pause_button.setChecked(False)
        self._fill_trend_choices()
        self.status_progress.setRange(0, len(listed))
        self.status_progress.setValue(0)
        self.status_progress.show()
        if self.watch_check.isChecked():
            self._watch.start()
        self._log(tr("Started: {count} frames, {model}.").format(count=len(listed), model=model_label(model)))
        self.show_step("results")
        self._frames_changed()
        self._next()

    def _next(self) -> None:
        if not self.running:
            return
        if self._stop.is_set() or (not self.queue and not self._watch.isActive()):
            self._finish()
            return
        if self.pause_button.isChecked() or not self.queue:
            return  # paused, or waiting for new curves (Watch)
        index = self.queue.pop(0)
        self._current = index
        path = self.paths[index]
        outcome = self.view_model.load_curve(LoadCurveRequest(Path(path), self._unit))  # the unit at Start
        if outcome.error is not None:
            self._frame_done(FrameFit(index, path, error=outcome.error.message))
            return
        loaded = outcome.value
        curve = Curve.from_arrays(loaded.q, loaded.intensity, loaded.error, name=Path(path).name, path=path,
                                  unit=loaded.q_source_unit)
        start = next_start(self._previous, self._start_model, self._settings, new_stage=self._new_stage(index))
        settings, stop = self._settings, self._stop.is_set
        self._set_item(index)
        self.tasks.submit("series", lambda: (fit_frame(index, path, curve, start, settings, stop), curve),
                          on_done=lambda value: self._frame_done(value[0], value[1]),
                          on_error=lambda message, _details: self._frame_done(FrameFit(index, path, error=message)))

    def _frame_done(self, frame: FrameFit, curve: Optional[Curve] = None) -> None:
        self.fits[frame.index] = frame
        if frame.ok:
            self._previous = frame
        self._curves[frame.index] = curve
        self._set_item(frame.index)
        self.status_progress.setValue(len(self.fits))
        self._status(lambda: tr("Frame {frame}: {state}").format(
            frame=frame.index + 1, state=f"χ²ᵣ {frame.result.chi2_reduced:.3g}" if frame.ok else tr(frame.error)))
        if not frame.ok:
            self._log(tr("Frame {frame} ({name}) failed: {reason}").format(
                frame=frame.index + 1, name=Path(frame.path).name, reason=tr(frame.error)))
        if self._selected is None or self._selected == self._last_done:  # following the run
            self._select(frame.index, follow=True)
        self._last_done = frame.index
        self._draw_trend()
        self.refresh()
        QTimer.singleShot(0, self._next)

    def _pause_toggled(self, paused: bool) -> None:
        self.pause_button.setText(tr("Continue") if paused else tr("Pause"))
        if not paused and self.running and not self.tasks.is_busy():
            self._next()
        self._status(tr("Paused after this frame.") if paused else tr("Going on."))

    def stop(self) -> None:
        if not self.running:
            return
        self._stop.set()
        self.queue.clear()
        self._watch.stop()
        self._status(tr("Stopping after this frame …"))
        if not self.tasks.is_busy():
            self._finish()

    def _finish(self) -> None:
        self.running = False
        self._watch.stop()
        self.status_progress.hide()
        failed, count = sum(1 for frame in self.fits.values() if not frame.ok), len(self.fits)

        def done() -> str:
            return tr("Series done: {count} frames fitted, {failed} failed.").format(count=count, failed=failed)

        self._log(done())
        self._status(done, "warning" if failed else "ok")
        self._frames_changed()

    def _look_for_new(self) -> None:
        """Watch: curves that appeared since the start are fitted too."""
        if not self.running or not self.folder:
            return
        try:
            frames = self.view_model.storage.discover_insitu_frames(DiscoverInSituFramesRequest(
                Path(self.folder), self.pattern_edit.text().strip() or "*_fit_input.dat", self.subfolders_check.isChecked()))
        except (OSError, ValueError, RuntimeError):
            return
        known = set(self.paths)
        new = [str(frame.path) for frame in frames if str(frame.path) not in known]
        if not new:
            return
        for path in new:
            self.paths.append(path)
            self.queue.append(len(self.paths) - 1)
            item = QListWidgetItem()
            self.frame_list.addItem(item)
            self._style_item(item, len(self.paths) - 1)
        self.last_spin.blockSignals(True)
        self.last_spin.setRange(1, len(self.paths))
        self.last_spin.setValue(len(self.paths))
        self.last_spin.blockSignals(False)
        self.every_spin.setMaximum(len(self.paths))
        self.status_progress.setMaximum(self.status_progress.maximum() + len(new))
        self._log(tr("{count} new curves.").format(count=len(new)))
        if not self.tasks.is_busy():
            self._next()

    def _set_item(self, index: int) -> None:
        listed = self._listed()
        if index in listed:
            item = self.frame_list.item(listed.index(index))
            if item is not None:
                self._style_item(item, index)  # its state's mark and colour; an odd frame keeps its “· odd”

    # -- showing (the selected frame and the trend's choices: ``series_preview.py``) -----------

    def _select_table_row(self, row: int) -> None:
        indices = sorted(self.fits)
        if 0 <= row < len(indices):
            self._select(indices[row], follow=True)

    def _fill_results(self) -> None:
        if self._start_model is None:
            self.results_summary.setText(tr("Choose the curves, check the start, then Start."))
            self.results_table.setRowCount(0)
            return
        header, rows = series_table(list(self.fits.values()), self._start_model)
        failed = sum(1 for frame in self.fits.values() if not frame.ok)
        loose = sum(1 for frame in self.fits.values() if frame.ok and not frame.result.converged)
        self.results_summary.setText(tr("{done} frames fitted · {failed} failed · {loose} not converged").format(
            done=len(self.fits), failed=failed, loose=loose))
        shown = ["frame", "chi2_reduced"] + [name for name in header[5::2]]
        self.results_table.setColumnCount(len(shown))
        columns = parameter_columns(self._start_model)
        self.results_table.setHorizontalHeaderLabels(
            ["#", "χ²ᵣ"] + [path_name(self._start_model, path, unit=True) for path, _name in columns])
        for column, (path, _name) in enumerate(columns, start=2):  # the whole name (“1·Sphere R (nm)”) on hover
            self.results_table.horizontalHeaderItem(column).setToolTip(path_name(self._start_model, path, unit=True, full=True))
        self.results_table.setRowCount(len(rows))
        for line, row in enumerate(rows):
            record = dict(zip(header, row))
            cells = [str(record["frame"]), "—" if not np.isfinite(record["chi2_reduced"]) else f"{record['chi2_reduced']:.3g}"]
            for name in shown[2:]:
                value, error = record[name], record[f"{name}_error"]
                cells.append("—" if not np.isfinite(value) else
                             f"{value:.4g}" + ("" if not np.isfinite(error) else f" ± {error:.2g}"))
            for column, text in enumerate(cells):
                self.results_table.setItem(line, column, QTableWidgetItem(text))

    def _open_in_single(self) -> None:
        frame = self.fits.get(self._selected) if self._selected is not None else None
        if frame is None:
            self._status(tr("Select a fitted frame first."), "warning")
            return
        self.openFrameRequested.emit(frame.path, frame.result.model if frame.ok else None)

    # -- saving -------------------------------------------------------------------------------

    def save_table(self, path=None):
        if not self.fits or self._start_model is None:
            return None
        path = path or QFileDialog.getSaveFileName(self, tr("Save Table of Every Frame"),
                                                   str(Path(self.folder or ".") / "series_fit.csv"), "CSV (*.csv)")[0]
        if not path:
            return None
        header, rows = series_table(list(self.fits.values()), self._start_model)

        def write(target) -> None:
            with open(target, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)

        record = series_record(self._start_model, self._settings, self.folder, self.pattern_edit.text(), list(self.fits.values()))
        try:
            write_pair(path, write, record)  # both, or neither
        except OSError as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._status(lambda name=Path(path).name: tr("Saved {name} and its record.").format(name=name), "ok",
                     action=folder_action(path))
        return path

    def _save_plot(self, plot, name: str, path=None):
        path = path or QFileDialog.getSaveFileName(self, tr("Save Plot"), str(Path(self.folder or ".") / f"series_{name}.png"),
                                                   "PNG image (*.png);;SVG vector (*.svg)")[0]
        if not path:
            return None
        try:
            write_plot(plot.plot, path)
        except OSError as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._status(lambda name=Path(path).name: tr("Saved {name}.").format(name=name), "ok", action=folder_action(path))
        return path

    # -- status, log, preferences -------------------------------------------------------------

    def _status(self, text, level: str = "info", action=None) -> None:
        """The status line; a toast for outcomes (``action``: e.g. Open Folder after a save). ``text``: the line,
        or a function that makes it — made again after a switch of the interface language."""
        self._status_text = text if callable(text) else None
        text = text() if callable(text) else text
        self.status_label.setText(text)
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

    def _remember_choices(self) -> None:
        self._remember(start=self._start_choice(),
                       method="global" if self.method_global.isChecked() else "local")
        self.refresh()

    def _restore(self) -> None:
        self._keep_splitter()
        pattern = self._remembered("pattern")
        if pattern:
            self.pattern_edit.setText(str(pattern))
        self._set_start(self._remembered("start"))
        if self._remembered("method") == "global":
            self.method_global.setChecked(True)

    # -- a project -------------------------------------------------------------------------

    def project_state(self) -> dict:
        return {"folder": self.folder, "pattern": self.pattern_edit.text(), "subfolders": self.subfolders_check.isChecked(),
                "first": self.first_spin.value(), "last": self.last_spin.value(), "every": self.every_spin.value(),
                "start": self._start_choice(),
                "method": "global" if self.method_global.isChecked() else "local"}

    def apply_project_state(self, data: dict) -> list[str]:
        if self.running:  # nothing changes under a running series (its folder, frames and choices stay)
            return [tr("the In-situ series was not restored: a series is running")]
        self._set_start(data.get("start"))
        (self.method_global if data.get("method") == "global" else self.method_local).setChecked(True)
        self.subfolders_check.blockSignals(True)
        self.subfolders_check.setChecked(bool(data.get("subfolders")))
        self.subfolders_check.blockSignals(False)
        folder = str(data.get("folder") or "")
        if not folder:
            return []
        if not Path(folder).is_dir():
            return [tr("the series folder {folder} is no longer there").format(folder=folder)]
        if self.open_series(folder, str(data.get("pattern") or "*_fit_input.dat")):
            for spin, key in ((self.first_spin, "first"), (self.last_spin, "last"), (self.every_spin, "every")):
                if data.get(key):
                    spin.setValue(int(data[key]))
        return []

    def refresh_language(self) -> None:
        """After a switch of the interface language (``i18n.language_changed``): what this page composed with
        ``tr`` at run time — the steps, the stages line, the frame list, the trend's values and legend, the
        selected frame's legend and the table's headers — again, the chosen trend value kept."""
        if self.trend_combo.count():
            chosen = self.trend_combo.currentData()
            self._fill_trend_choices(self._start_model if self._start_model is not None else self.single.session.model)
            index = next((i for i in range(self.trend_combo.count()) if self.trend_combo.itemData(i) == chosen), -1)
            if index >= 0:
                self.trend_combo.blockSignals(True)
                self.trend_combo.setCurrentIndex(index)
                self.trend_combo.blockSignals(False)
        self.refresh()
        self._refresh_stages_language()
        if self._selected is not None:
            self._select(self._selected)
        if self._status_text is not None:
            self.status_label.setText(self._status_text())

    def dispose(self) -> None:
        self._dispose_stages()
        self.stop()
        self.tasks.shutdown(2000)
        self.frame_plot.dispose()
        self.trend_plot.dispose()


SIDE_TEXT = (("mean", "Mean of both halves"), ("both", "Both halves on |q|"), ("positive", "q > 0 half"),
             ("negative", "q < 0 half"))

__all__ = ["FitSeriesPage"]
