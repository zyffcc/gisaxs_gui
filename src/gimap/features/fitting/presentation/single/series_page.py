"""The In-situ series page of Fitting: every curve of a folder fitted with the model of Single analysis.

Curves → Start → Results. The model, the fitting range and the left-out points are those of Single analysis
(one representative curve is set up there); the halves of the cut are the series' own choice in the Curves
step (``series_halves.py``). Each frame starts from the previous frame's result, or every frame from that
model. Frames are fitted one after another in the background — the list shows each one's state, the right
side the selected frame with its fit and the trend of a parameter through the series — with Pause and Stop,
and new curves can be fitted as they appear (Watch, ``series_watch.py``). The table of every frame, what the
run came to and saving are in ``series_results.py``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import LoadCurveRequest
from ...application.series_fit import (
    POOR_CHI2,
    FrameFit,
    SeriesSettings,
    fit_frame,
    next_start,
    poor_fit,
    series_counts,
)
from ...application.single_fit import Curve
from ..views.fit_series_view import FitSeriesView
from .drops import FolderDropMixin, SplitterMemoryMixin
from .series_fit1d import FitSeriesFit1DMixin
from .series_frames import FitSeriesFramesMixin
from .series_halves import FitSeriesHalvesMixin
from .series_listing import CURVES_HINT, FitSeriesListingMixin
from .series_preview import FitSeriesPreviewMixin
from .series_results import UNTOUCHED, FitSeriesResultsMixin
from .series_stages import NO_STAGES_AT_START, FitSeriesStagesMixin
from .series_text import FIGURE_SPACE, ONE_HALF, STATE_MARK
from .series_units import FitSeriesUnitsMixin
from .series_watch import WATCH_MS, FitSeriesWatchMixin, file_stamp
from .session import model_label

PREFERENCES_KEY = "fitting_series_page"
NOT_SET_UP = "Model not set up yet: fit one frame in Single analysis"
NOT_SET_UP_NOTE = ("The model is still the starting guess: nothing was set up or fitted in Single analysis. "
                   "Start would fit this guess to every frame.")
END_SENTENCES = {
    (False, False): "Series done: {count} frames fitted, {failed} failed.",
    (True, False): "Series stopped: {count} frames fitted, {failed} failed.",
    (False, True): "Series done: {count} frames fitted, {poor} of them poor fits (χ²ᵣ above {limit}), {failed} failed.",
    (True, True): "Series stopped: {count} frames fitted, {poor} of them poor fits (χ²ᵣ above {limit}), {failed} failed.",
}
"""The end of a run, ``(stopped, with poor fits)``: a run whose fits converged far from the curves (``poor_fit``) is
not said like a success (said at the level of a warning)."""


class FitSeriesPage(FolderDropMixin, SplitterMemoryMixin, FitSeriesListingMixin, FitSeriesStagesMixin, FitSeriesFit1DMixin,
                    FitSeriesFramesMixin, FitSeriesUnitsMixin,
                    FitSeriesPreviewMixin, FitSeriesHalvesMixin, FitSeriesResultsMixin, FitSeriesWatchMixin, QWidget,
                    FitSeriesView):
    editModelRequested = pyqtSignal()
    """Show Single analysis (the model is set up there)."""
    openFrameRequested = pyqtSignal(str, object)
    """A frame's file and fitted model, to look at in Single analysis."""

    def __init__(self, view_model, single_page, *, preferences=None, frames=None, parent: Optional[QWidget] = None):
        """``frames``: the cutter of detector frames (``app/series_frames.py``); ``None``: curves only."""
        super().__init__(parent)
        self.view_model = view_model
        self.single = single_page
        self.preferences = preferences
        self.tasks = TaskRunner(self)
        self.setup_ui(self)
        self.setup_series_frames(frames)
        self.folder = ""
        self.paths: list[str] = []
        self.fits: dict[int, FrameFit] = {}
        self.queue: list[int] = []
        self.running = False
        self._stop = threading.Event()
        self._start_model = None
        self._settings = SeriesSettings()
        self._unit = "angstrom"
        """The unit of q of the files whose header states none: Single analysis' setting at Start, fixed for the run
        (``series_units.py``; detector frames are cut in Å⁻¹)."""
        self._previous: Optional[FrameFit] = None
        self._status_text = None
        """Makes the status line again in another interface language (``refresh_language``)."""
        self._curves: dict = {}
        self._preview_curves: dict = {}
        self._current = -1
        self._selected: Optional[int] = None
        self._last_done: Optional[int] = None
        self._frames_done = 0
        """Fits that ended in this run (a curve fitted again counts again): the progress bar."""
        self._names: Optional[list] = None
        self._listing = (self.pattern_edit.text(), False)
        """The pattern and “Also in subfolders” of the curves that are listed (set again when a change is cancelled)."""
        self._below = ""
        self.setAcceptDrops(True)  # a folder dropped here is listed (``drops.py``)
        self._connect()
        self._connect_halves()
        self._connect_results()
        self._connect_watch()
        self._connect_stages()
        self._restore()
        self.refresh()

    def _connect(self) -> None:
        self.step_rail.stepChosen.connect(self.show_step)
        self.folder_button.clicked.connect(self.choose_folder)
        self.pattern_edit.editingFinished.connect(self._list_curves)
        self.subfolders_check.toggled.connect(lambda _on: self._list_curves())
        self.first_spin.valueChanged.connect(self.last_spin.setMinimum)  # never a range without a frame
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.valueChanged.connect(lambda _value: self._frames_changed())
        self.start_button.clicked.connect(self.start)
        self.pause_button.toggled.connect(self._pause_toggled)
        self.stop_button.clicked.connect(self.stop)
        self.frame_list.currentRowChanged.connect(lambda row: self._select(self._listed()[row] if 0 <= row < len(self._listed()) else None))
        self.trend_combo.currentIndexChanged.connect(lambda _index: self._draw_trend())
        self.trend_plot.log_check.toggled.connect(lambda _checked: self._draw_trend())  # the bars again
        self.edit_model_button.clicked.connect(self._edit_or_set_up)
        self.start_group.buttonToggled.connect(lambda *_: self._remember_choices())
        self.method_group.buttonToggled.connect(lambda *_: self._remember_choices())
        self.follow_check.toggled.connect(lambda _on: self._remember_choices())
        self.setup_series_fit1d()

    # -- steps --------------------------------------------------------------------------------

    def show_step(self, key: str) -> None:
        self.step_rail.set_current(key)
        self.step_stack.setCurrentWidget(self.step_pages[key])
        self._moved_on_results = False
        self._fit_steps_width(key)  # room for the table of every frame while Results is shown

    def _range_text(self, settings) -> str:
        """“every point”, or the fitting range and how many points are left out by name."""
        text = tr("every point") if settings.q_range is None else f"|q| {settings.q_range[0]:.4g}–{settings.q_range[1]:.4g} nm⁻¹"
        return text + (f" · {len(settings.excluded)} " + tr("left out") if settings.excluded else "")

    def refresh(self) -> None:
        """Where each step stands, and what the Start step will use."""
        listed = self._listed()
        self.refresh_frames_card()
        self._show_halves()
        self._show_units()
        self._schedule_stages()
        if not self.paths:
            self.step_rail.set_state("curves", "pending", tr("Choose the folder of the curves"))
            self.step_intro["curves"].setText(tr(CURVES_HINT))
        else:
            detail = tr("{listed} of {total} curves").format(listed=len(listed), total=len(self.paths))
            self.step_rail.set_state("curves", "ok", detail)
            self.step_intro["curves"].setText(detail + " · " + self._folder_name())  # the whole path: on the chip
        session = self.single.session
        model = session.model
        free = sum(1 for _path, parameter in model.parameters() if parameter.free)
        self.model_summary.setText(tr("{model} · {free} values fitted\n{side}; {range}{left_out}").format(
            model=model_label(model), free=free, side=self.halves_text(), range=self._range_text(self._stage_settings()),
            left_out=""))
        start = {"previous": tr("from the previous frame"), "same": tr("from the same model"),
                 "stages": tr("from the previous frame, afresh at each stage")}[self._start_choice()]
        untouched = bool(model.components) and not session.edited  # the starting guess: nobody set it up
        self.step_rail.set_state("start", "ok" if model.components and not untouched else "warn",
                                 tr("No model: set one in Single analysis") if not model.components else
                                 tr(NOT_SET_UP) if untouched else f"{model_label(model)} · {start}")
        self.setup_note.setText(tr(NOT_SET_UP_NOTE) if untouched else "")
        self.setup_note.setVisible(untouched)
        self._show_set_up()
        self.step_intro["start"].setText(tr("The model of Single analysis, fitted to every frame."))
        self.step_intro["results"].setText(tr("Every frame: the quality of its fit and each value ± 1σ. Click a row "
                                              "to see that frame; the plot below follows a value through the series."))
        self._refresh_results(listed)
        self.start_button.setVisible(not self.running)
        self.start_button.setEnabled(bool(listed) and bool(model.components))
        self.start_button.setToolTip(
            tr("Fit every listed curve with the model of Single analysis") + ("\n" + tr(NOT_SET_UP) if untouched else "")
            if self.start_button.isEnabled() else
            tr("Choose a folder of curves first") if not listed else tr("Set up a model in Single analysis first"))
        self.pause_button.setVisible(self.running)
        self.stop_button.setVisible(self.running)
        self._preview_trend()
        self._redraw_preview()

    def _set_up_frame(self) -> Optional[int]:
        """The listed frame Single analysis opens to set the model up — while it holds no curve of its own."""
        if self.single.session.curve is not None or self.running or not self.paths:
            return None
        listed = self._listed()
        return self._selected if self._selected in listed else (listed[0] if listed else None)

    def _show_set_up(self) -> None:
        """The Start step's button: to Single analysis — with the selected frame while it holds no curve."""
        frame = self._set_up_frame()
        self.edit_model_button.setText(tr("Edit in Single Analysis") if frame is None else
                                       trf("Set Up on Frame {n} in Single Analysis", n=frame + 1))

    def _edit_or_set_up(self) -> None:
        frame = self._set_up_frame()
        if frame is None:
            self.editModelRequested.emit()
            return
        if not self._started():
            self._settings = self._stage_settings()  # opened there on the points this series would fit
        self.openFrameRequested.emit(self.paths[frame], None)

    # -- the frames (listing them: ``series_listing.py``) ---------------------------------------

    def _frame_state(self, index: int) -> tuple[str, str]:
        """``(state, detail)``: waiting, running, done, warn (not converged, a poor fit), stopped or failed —
        before the run also for a curve that cannot be used as the series stands — and χ²ᵣ or the reason."""
        frame = self.fits.get(index)
        if frame is None:
            if self.running:
                return ("running" if index == self._current and self.tasks.is_busy() else "waiting"), ""
            reason = self._skipped.get(index)
            return ("failed", self._reason_text(reason)) if reason else ("waiting", "")
        if not frame.ok:
            return "failed", self._reason_text(frame.error)  # in the interface language; the record keeps it as it came
        result = frame.result
        detail = f"χ²ᵣ {result.chi2_reduced:.3g}"
        if getattr(result, "stopped", False):
            return "stopped", detail + " · " + tr("stopped")
        if not result.converged:
            return "warn", detail + " · " + tr("not converged")
        if poor_fit(frame):
            return "warn", detail + " · " + tr("poor fit")
        half = ONE_HALF.get(getattr(frame, "halves_used", "")) if self._settings.side == "mean" else None
        return "done", detail + (" · " + tr(half) if half and self.curves_signed() else "")

    def _item_text(self, index: int) -> str:
        state, detail = self._frame_state(index)
        number = str(index + 1).rjust(len(str(len(self.paths))), FIGURE_SPACE)  # as wide as a digit: names line up
        return f"{STATE_MARK[state]} {number}  {self._short_name(index)}" + (f"  {detail}" if detail else "")

    # -- the run ------------------------------------------------------------------------------

    def start(self) -> None:
        session = self.single.session
        model = session.model
        if self.running or not self._listed() or not model.components or not self.fit1d_can_start(model) \
                or not self.ready_to_cut() or not self._may_drop_results():
            return
        self._probe_signed()  # a half is only chosen for curves that have two (other frames chosen since the listing)
        listed = self._without_odd(self._listed())
        self._settings = SeriesSettings(
            side=self.series_side(), q_range=session.q_range, excluded=frozenset(session.excluded),
            method=self.series_method(), start=self._start_choice(),
            resolve_edges=self.single.resolve_edges_check.isChecked(),  # as in Single analysis' Fit step
            follow=self.follow_check.isChecked(), fit1d={**self.single.fit1d_settings(), "fit1d_composition": "model"})
        self._side_from, self._untouched_start = self.halves_source(), not session.edited
        self._unit = "angstrom" if self.frames_mode() else self.single.unit_setting()  # for files without a unit
        self._start_model = model
        self._run_listing = self._listing_record(listed)
        self.fits, self.queue, self._previous, self._curves = {}, list(listed), None, {}
        self._selected = self._last_done = None
        self._frames_done, self._unsaved, self._read_stamps, self._seen = 0, False, {}, {}
        self._stop.clear()
        self.pause_button.setChecked(False)  # (before the run is on: un-pausing a run starts its next frame)
        self.running = True
        self._fill_trend_choices()
        self._rebuild_results()
        self._show_progress()
        self.status_progress.show()
        if self.watch_check.isChecked():
            self._watch.start()
        self._log(tr("Started: {count} frames, {model}.").format(count=len(listed), model=model_label(model)))
        self._log(self._points_text(self._settings, self._side_from))
        if self._settings.start == "stages" and self._series_stages is None:  # not found (yet): none steer this run
            self._log(tr(NO_STAGES_AT_START))
            self._status(lambda: tr(NO_STAGES_AT_START), "warning")
        if self._untouched_start:
            self._status(lambda: tr(UNTOUCHED), "warning")
        self.show_step("results")
        self._frames_changed()
        self._next()

    def _frames_to_do(self) -> int:
        return self._frames_done + len(self.queue) + (1 if self.tasks.is_busy() else 0)

    def _show_progress(self) -> None:
        self.status_progress.setRange(0, max(1, self._frames_to_do()))
        self.status_progress.setValue(self._frames_done)

    def _next(self) -> None:
        if not self.running or self.tasks.is_busy():
            return  # a fit is under way and goes on from its end: never two at once (Watch and a finished fit)
        if self._stop.is_set() or (not self.queue and not self._watch.isActive()):
            self._finish()
            return
        if self.pause_button.isChecked():
            return
        if not self.queue:
            self._say_waiting()  # Watch: every curve there is has been fitted
            return
        index = self.queue.pop(0)
        self._current = index
        path = self.paths[index]
        self._read_stamps[index] = file_stamp(path)  # Watch reads it again if the file changes after this
        outcome = self.load_series_curve(LoadCurveRequest(Path(path), self._unit))  # the unit at Start; a frame is cut
        if outcome.error is not None:
            self._frame_done(FrameFit(index, path, error=outcome.error.message))
            return
        loaded = outcome.value
        self._note_curve(index, loaded.q, loaded.observation)
        curve = Curve.from_arrays(loaded.q, loaded.intensity, loaded.error, name=Path(path).name, path=path,
                                  unit=loaded.q_source_unit, unit_from=loaded.q_unit_from)
        new_stage = self._new_stage(index)
        start = next_start(self._previous, self._start_model, self._settings, new_stage=new_stage)
        settings, stop = self._settings, self._stop.is_set
        predict = self.predictor(settings, self._previous, new_stage) if settings.method == "fit1d" else None
        self.tasks.submit(f"series-{index}", lambda: (fit_frame(index, path, curve, start, settings, stop, predict), curve),
                          on_done=lambda value: self._frame_done(value[0], value[1]),
                          on_error=lambda message, _details: self._frame_done(FrameFit(index, path, error=message)))
        self._set_item(index)

    def _frame_done(self, frame: FrameFit, curve: Optional[Curve] = None) -> None:
        self.fits[frame.index] = frame
        self._frames_done += 1
        if frame.ok:
            self._previous = frame
            self._unsaved = self._unsaved or frame.fitted  # where a stopped fit was is no result to save
        self._curves[frame.index] = curve
        self._show_result_row(frame)
        self._set_item(frame.index)
        self._show_progress()
        self._status(lambda: tr("Frame {frame}: {state}").format(frame=frame.index + 1, state=self._frame_state(frame.index)[1]))
        if not frame.ok:
            self._log(tr("Frame {frame} ({name}) failed: {reason}").format(
                frame=frame.index + 1, name=Path(frame.path).name, reason=self._reason_text(frame.error)))
        if self._selected is None or self._selected == self._last_done:  # following the run
            self._select(frame.index, follow=True)
        self._last_done = frame.index
        self._draw_trend()
        self.refresh()
        QTimer.singleShot(0, self._next)

    def _pause_toggled(self, paused: bool) -> None:
        self.pause_button.setText(tr("Continue") if paused else tr("Pause"))
        self._status(tr("Paused after this frame.") if paused else tr("Going on."))
        if not paused and self.running:
            self._next()

    def stop(self) -> None:
        if not self.running:
            return
        self._stop.set()
        self.cancel_series_job()  # Predict each frame: its job ends now
        self.queue.clear()
        self._watch.stop()
        self._status(tr("Stopping after this frame …"))
        if not self.tasks.is_busy():
            self._finish()

    def _finish(self) -> None:
        stopped = self._stop.is_set()
        self.running = False
        self._watch.stop()
        self.status_progress.hide()
        counts = series_counts(list(self.fits.values()))
        fitted, failed, poor = counts["ok"] + counts["not_converged"], counts["failed"], len(self._poor_frames())
        trouble = failed or counts["not_converged"] or counts["stopped"] or poor or self._untouched_start

        def done() -> str:  # (poor fits — χ²ᵣ far above 1, e.g. a sphere on a diffraction peak — are counted)
            template = END_SENTENCES[(bool(stopped), bool(poor))]
            return trf(template, count=fitted, failed=failed, poor=poor, limit=f"{POOR_CHI2:g}")

        self._log(done())
        self._status(done, "warning" if trouble else "ok")
        self._frames_changed()

    def _set_item(self, index: int) -> None:
        listed = self._listed()
        if index in listed:
            item = self.frame_list.item(listed.index(index))
            if item is not None:
                self._style_item(item, index)  # its state's mark and colour; an odd frame keeps its “· odd”

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
        self._remember(start=self._start_choice(), method=self.series_method(), follow=self.follow_check.isChecked())
        self.refresh()

    def _restore(self) -> None:
        self._keep_splitter()
        pattern = self._remembered("pattern")
        if pattern:
            self.pattern_edit.setText(str(pattern))
        self._set_start(self._remembered("start"))
        self.set_series_method(self._remembered("method"))
        if self._remembered("follow") is not None:
            self.follow_check.setChecked(bool(self._remembered("follow")))

    # -- a project -------------------------------------------------------------------------

    def project_state(self) -> dict:
        return {"folder": self.folder, "pattern": self.pattern_edit.text(), "subfolders": self.subfolders_check.isChecked(),
                "first": self.first_spin.value(), "last": self.last_spin.value(), "every": self.every_spin.value(),
                "start": self._start_choice(),
                "method": self.series_method(), "follow": self.follow_check.isChecked(), **self.halves_state()}

    def apply_project_state(self, data: dict) -> list[str]:
        if self.running:  # nothing changes under a running series (its folder, frames and choices stay)
            return [tr("the In-situ series was not restored: a series is running")]
        self._set_start(data.get("start"))
        self.set_series_method(data.get("method"))
        self.follow_check.setChecked(bool(data.get("follow", True)))
        self.subfolders_check.blockSignals(True)
        self.subfolders_check.setChecked(bool(data.get("subfolders")))
        self.subfolders_check.blockSignals(False)
        folder = str(data.get("folder") or "")
        if not folder:
            return []
        if not Path(folder).is_dir():
            return [tr("the series folder {folder} is no longer there").format(folder=folder)]
        if self._open(folder, str(data.get("pattern") or "*_fit_input.dat"), ask=False):
            for spin, key in ((self.first_spin, "first"), (self.last_spin, "last"), (self.every_spin, "every")):
                if data.get(key):
                    spin.setValue(int(data[key]))
            self.apply_halves_state(data)  # the halves the project kept for this series
            self.refresh()
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
        self._watch.stop()
        self.stop()
        self.tasks.shutdown(2000)
        self.frame_plot.dispose()
        self.trend_plot.dispose()


__all__ = ["FIGURE_SPACE", "FitSeriesPage", "PREFERENCES_KEY", "STATE_MARK", "WATCH_MS"]
