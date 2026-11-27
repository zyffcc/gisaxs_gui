"""Watch for new frames: a folder batch of 2D Prediction that goes on with the frames written into its folder.

Ticked when Predict is pressed, the batch predicts the frames of the folder and then keeps looking at it: as soon as
the folder changes (``QFileSystemWatcher``; while it keeps changing, every ``SOON_MS``, or as long after the look
before as that look took: a folder of tens of thousands of frames takes a while to list, and back to back the looks
kept a thread busy all the time) and at least every ``LOOK_MS``, the listing made off the window's thread. A new frame is taken once it is completely written (``FrameWatch``), gets its row in the
results and is predicted after the ones before. Keep up with the detector (``KeepUp``): at most two stacks wait for the
model; when more arrive, the oldest waiting are left out and counted. Unticking Watch ends the run once the frames
taken are predicted; Stop ends it at once. An open trend or heatmap window is drawn again at most every ``REFRESH_MS``.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Optional, Sequence

from PyQt5.QtCore import QFileSystemWatcher, QTimer

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import FRAME_SUFFIXES, FrameWatch, KeepUp

LOOK_MS = 1000
SOON_MS = 150
REFRESH_MS = 1000

WATCHING = "Watching {folder} for new frames … (Stop to end)"
WAITING = "{done} predicted · waiting for new frames"
LEFT_OUT = "{count} frames left out to keep up with the detector (a folder batch predicts them later)"
OUTSIDE = "{count} new frames outside the Range: not predicted"
FOLDER_GONE = "The watched folder cannot be read: waiting for it … (Stop to end)"
WATCH_ENDED = "Watch ended: {taken} new frames taken, {left_out} left out to keep up."


class FrameWatchMixin:
    """Needs the page binding: ``ui`` (``gisaxsPredictWatchCheck``, ``gisaxsPredictKeepUpCheck``),
    ``_multifile_manager``, ``_multifile_results_widget``, ``_multifile_batch_map``, ``_add_batch_row``,
    ``prediction_view_model``, ``_append_status_message`` and ``status_updated``."""

    def _setup_frame_watch(self) -> None:
        self._watch: Optional[FrameWatch] = None
        self._keep_up: Optional[KeepUp] = None
        self._watch_folder = ""
        self._watch_folder_of_run = ""
        self._watch_listed_of_run: tuple = ()
        self._watch_every_of_run: tuple = (1, ())
        self._watch_taken = 0
        self._watch_predicted = 0
        self._watch_gone = False
        self._watch_busy = False
        self._watch_tasks = TaskRunner(self)
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(LOOK_MS)
        self._watch_timer.timeout.connect(self._look_at_folder)
        self._watch_soon = QTimer(self)
        self._watch_soon.setSingleShot(True)
        self._watch_soon.setInterval(SOON_MS)
        self._watch_soon.timeout.connect(self._look_at_folder)
        self._watch_changes = QFileSystemWatcher(self)
        self._watch_changes.directoryChanged.connect(self._folder_changed)
        self._watch_refresh = QTimer(self)
        self._watch_refresh.setSingleShot(True)
        self._watch_refresh.setInterval(REFRESH_MS)
        self._watch_refresh.timeout.connect(self._refresh_watch_windows)
        watch = getattr(self.ui, "gisaxsPredictWatchCheck", None)
        keep_up = getattr(self.ui, "gisaxsPredictKeepUpCheck", None)
        if watch is not None:
            watch.toggled.connect(self._frame_watch_toggled)
            if keep_up is not None:
                keep_up.setEnabled(watch.isChecked())
        if keep_up is not None:
            keep_up.toggled.connect(self._keep_up_toggled)

    def _folder_changed(self, _folder: str) -> None:
        """A look ``SOON_MS`` after the first change not looked at yet. Not restarted by the changes after it: a
        detector writing ten frames a second changes the folder all the time, and a look that waited for a quiet
        moment would never come."""
        if not self._watch_soon.isActive():
            self._watch_soon.start()

    # -- the boxes ---------------------------------------------------------------------------------

    def _frame_watch_wanted(self) -> bool:
        box = getattr(self.ui, "gisaxsPredictWatchCheck", None)
        return bool(box is not None and box.isChecked())

    def _keep_up_wanted(self) -> bool:
        box = getattr(self.ui, "gisaxsPredictKeepUpCheck", None)
        return bool(box is not None and box.isChecked())

    def _frame_watch_toggled(self, on: bool) -> None:
        """Watch ticked or unticked: Keep up follows it; unticked during a watched run, the run ends once the frames
        taken are predicted (ticked during a run of the frames listed, that run goes on with the new ones)."""
        keep_up = getattr(self.ui, "gisaxsPredictKeepUpCheck", None)
        if keep_up is not None:
            keep_up.setEnabled(on)
        manager = self._multifile_manager
        if manager is None or not manager.is_running:
            self._refresh_batch_plan_summary()  # an empty folder can be watched: the plan and Predict say so
            self._refresh_predict_readiness()
            return
        if on and self._watch is None and self._watch_folder_of_run:
            self._begin_frame_watch(self._watch_folder_of_run, self._watch_listed_of_run, self._watch_every_of_run)
            manager.follow = True
        elif not on and self._watch is not None:
            manager.stop_following()
            self._watch_timer.stop()
            self._watch_soon.stop()

    def _keep_up_toggled(self, on: bool) -> None:
        if self._keep_up is not None:
            self._keep_up.on = bool(on)
            if not on:
                self._queue_stacks(self._keep_up.release_all())  # those waiting: predicted in turn

    # -- the run -----------------------------------------------------------------------------------

    def _prepare_frame_watch(self, folder: str, listed: Sequence[str], every: int, waiting: Sequence[str] = ()) -> bool:
        """At Predict of a folder batch: remember what the run is (so Watch may be ticked during it) and begin the
        watch when Watch is ticked. ``listed``: every frame in the folder now; ``waiting``: trailing frames that do
        not fill a stack of Every. ``True`` when the run follows the folder."""
        self._watch_folder_of_run = folder
        self._watch_listed_of_run = tuple(listed)
        self._watch_every_of_run = (every, tuple(waiting))
        self._watch = None
        if not self._frame_watch_wanted():
            return False
        self._begin_frame_watch(folder, listed, self._watch_every_of_run)
        return True

    def _begin_frame_watch(self, folder: str, listed: Iterable[str], every_and_waiting) -> None:
        every, waiting = every_and_waiting
        files = self.prediction_view_model.files
        indices = None
        range_text = (self.current_parameters.get("range_value") or "").strip()
        if range_text:
            try:
                indices = set(files.index_range(range_text)) or None
            except Exception:  # noqa: BLE001 - the batch already said what is wrong with the Range
                indices = None
        self._watch = FrameWatch(listed, every, waiting=waiting, indices=indices,
                                 index_of=lambda path: files.file_index(os.path.basename(path)))
        self._keep_up = KeepUp(self._keep_up_wanted())
        self._watch_folder = folder
        self._watch_taken = 0
        self._watch_predicted = 0
        self._watch_gone = False
        if folder not in self._watch_changes.directories():
            self._watch_changes.addPath(folder)
        self._watch_timer.start()
        self._show_watch_summary()
        self._append_status_message(trf(WATCHING, folder=folder), level="INFO")

    def _look_at_folder(self) -> None:
        """List the folder off the window's thread (one look at a time)."""
        if self._watch is None or self._watch_busy:
            return
        self._watch_busy = True
        self._watch_look_began = time.monotonic()
        folder, files, skip = self._watch_folder, self.prediction_view_model.files, self._watch.known

        def failed(_message: str, _details: str) -> None:
            self._watch_busy = False
            if self._watch is not None and not self._watch_gone:
                self._watch_gone = True
                self.status_updated.emit(tr(FOLDER_GONE))

        def done(entries) -> None:
            self._watch_busy = False
            took = time.monotonic() - self._watch_look_began
            self._watch_soon.setInterval(max(SOON_MS, round(1000 * took)))  # (the next look as long after)
            self._apply_look(entries)

        self._watch_tasks.submit("look", lambda: files.look_folder(folder, FRAME_SUFFIXES, skip), on_done=done,
                                 on_error=failed)

    def _apply_look(self, entries) -> None:
        if self._watch is None:
            return
        if self._watch_gone:  # a share that came back
            self._watch_gone = False
            if self._watch_folder not in self._watch_changes.directories():
                self._watch_changes.addPath(self._watch_folder)
        before_left, before_outside = self._keep_up.left_out, self._watch.outside
        stacks = self._watch.take(entries)
        self._watch_taken += sum(len(stack) for stack in stacks)
        manager = self._multifile_manager
        self._queue_stacks(self._keep_up.offer(stacks, manager is not None and manager.can_take()))
        if self._keep_up.left_out > before_left:
            self.status_updated.emit(trf(LEFT_OUT, count=self._keep_up.left_out))
        if self._watch.outside > before_outside:
            self._append_status_message(trf(OUTSIDE, count=self._watch.outside), level="WARN")
        if manager is not None and manager.idle() and not stacks:
            self.status_updated.emit(trf(WAITING, done=self._watch_predicted))

    def _queue_stacks(self, stacks) -> None:
        """Give the run the stacks, each in a row of its own after the rows there are."""
        manager = self._multifile_manager
        if not stacks or manager is None:
            return
        items = []
        for stack in stacks:
            row = self._add_batch_row(stack)
            if row is not None:
                items.append((row, stack[0]))
        if items and not manager.append(items):
            self._end_frame_watch()

    def _frame_watch_feed(self) -> None:
        """The model took a frame from its queue (it started one, or finished one): the next stack waiting to keep up
        goes into the queue, so the model never waits for the window between two frames."""
        if getattr(self, "_keep_up", None) is not None and self._multifile_manager is not None:
            self._queue_stacks(self._keep_up.release(self._multifile_manager.can_take()))

    def _frame_watch_after_frame(self, predicted: int) -> None:
        """A frame is predicted: the queue is fed; open windows drawn again soon."""
        if getattr(self, "_watch_timer", None) is None:
            return
        self._watch_predicted = predicted
        self._frame_watch_feed()
        if self._watch is not None and not self._watch_refresh.isActive():
            self._watch_refresh.start()

    def _show_watch_summary(self) -> None:
        panel = getattr(self.ui, "predictionInputModePanel", None)
        if panel is not None and hasattr(panel, "set_watch_summary") and self._keep_up is not None:
            panel.set_watch_summary(predicted=self._watch_predicted, left_out=self._keep_up.left_out)

    def _refresh_watch_windows(self) -> None:
        if self._watch is not None:
            self._show_watch_summary()
        widget = self._multifile_results_widget
        if widget is not None:
            widget.refreshParameterTrend()
            widget.refreshDistributionHeatmap()

    def _end_frame_watch(self) -> None:
        """The run ended (all predicted after Watch was unticked, Stop, or the window closes)."""
        if getattr(self, "_watch_timer", None) is None:
            return
        self._watch_folder_of_run = ""
        if self._watch is None:
            return
        self._watch_timer.stop()
        self._watch_soon.stop()
        self._watch_tasks.cancel("look")
        folders = self._watch_changes.directories()
        if folders:
            self._watch_changes.removePaths(folders)
        left_out = self._keep_up.left_out if self._keep_up else 0
        self._append_status_message(trf(WATCH_ENDED, taken=self._watch_taken, left_out=left_out), level="INFO")
        self._watch = None
        self._keep_up = None
        self._refresh_watch_windows()
        if os.path.isdir(self._watch_folder):
            self._scan_directory_for_cbf(self._watch_folder)  # the plan of the folder as it is now, new frames too
        else:
            self._refresh_batch_plan_summary()

    def _watching_frames(self) -> bool:
        return getattr(self, "_watch", None) is not None


__all__ = ["FrameWatchMixin"]
