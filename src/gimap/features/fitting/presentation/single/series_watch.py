"""Watch of the In-situ series: curves that appear in the folder while the series runs are fitted too.

Every ``WATCH_MS`` the folder is listed again. A new file is taken once it has settled — the same size and
time at two looks in a row — so a curve that is still being written is not fitted on its first rows; and a
file that changed after its frame was read (it was read while it was written) is read and fitted again
when it has settled. The box can be ticked and unticked while the series runs. A run that has fitted
everything there is says that it is waiting, and for what; it ends with Stop, or when the box is unticked.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QListWidgetItem

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from .series_loading import discover_sources, read_units

from ...application.series_frames import file_stamp

WATCH_MS = 2000
FOLDER_GONE = "The folder of the curves cannot be read: waiting for it … (Stop to end)"


class FitSeriesWatchMixin:
    """Needs the series page: ``paths``, ``queue``, ``fits``, ``tasks``, ``running``, ``_stop``, ``_next()``,
    ``_finish()``, ``_listed()``, ``_style_item()``, ``_run_listing``, the widgets of the Curves step."""

    def _connect_watch(self) -> None:
        self.watch_tasks = TaskRunner(self)
        self._watch_busy = False
        self._watch_scan = 0
        self._watch = QTimer(self)
        self._watch.setInterval(WATCH_MS)
        self._watch.timeout.connect(self._look_for_new)
        self._seen: dict[str, Optional[tuple]] = {}
        """How every curve file of the folder looked at the last look (``file_stamp``)."""
        self._read_stamps: dict[int, Optional[tuple]] = {}
        """How a frame's file looked when it was read for its fit."""
        self._folder_gone = False
        """The last look could not list the folder (a share that dropped, a folder moved away)."""
        self.watch_check.toggled.connect(self._watch_toggled)

    def _watch_toggled(self, on: bool) -> None:
        """Ticked or unticked while the series runs: from now on, or no longer — a run that only waited ends."""
        self._watch_scan += 1
        self._watch_busy = False
        if not self.running or self._stop.is_set():
            return
        if on:
            self._seen = {}  # nothing has been looked at twice yet
            self._watch.start()
            if self._run_listing is not None:
                self._run_listing["watch"] = True  # the record: new curves were watched for in this run
            self._next()  # with nothing left to fit: says what it waits for
        else:
            self._watch.stop()
            if not self.queue and not self.tasks.is_busy():
                self._finish()
        if self.running:
            self.refresh()

    def _waiting_for_curves(self) -> bool:
        """The run has fitted every curve there is and stays for those to come."""
        return (self.running and not self.queue and not self.tasks.is_busy() and self._watch.isActive()
                and not self._stop.is_set())

    def _say_waiting(self) -> None:
        self._status(lambda folder=self._folder_name(): trf(
            "Waiting for new curves in {folder} … (Stop to end)", folder=folder))

    def _look_for_new(self) -> None:
        """A look at the folder: settled new curves are listed and queued, changed ones queued again."""
        if not self.running or not self.folder or self._stop.is_set() or self._watch_busy:
            return
        self._watch_busy = True
        epoch, scan, stop = self._loading_epoch, self._watch_scan, self._stop
        storage, bridge, recipe = self.view_model.storage, self.frames_bridge, self._recipe
        folder, pattern = self.folder, self.pattern_edit.text().strip() or "*_fit_input.dat"
        recursive = self.subfolders_check.isChecked()
        stated = getattr(self.view_model, "stated_q_unit", None)
        frame_mode = self.frames_mode()

        def work():
            found, _pattern, _below, _places, items = discover_sources(
                storage, bridge, recipe, folder, pattern, recursive, stop, fallback=False)
            units = {} if frame_mode else read_units(found, stated, stop)
            return found, items, {path: file_stamp(path) for path in found if not stop.is_set()}, units

        def done(result):
            if not self._loading_current(epoch) or scan != self._watch_scan or not self.running or stop.is_set():
                return
            self._watch_busy = False
            found, items, stamps, units = result
            self._stated_units = units
            self._frame_items.update(items)
            self._accept_new_curves(found, stamps)

        def failed(_message, _details):
            if not self._loading_current(epoch) or scan != self._watch_scan or not self.running or stop.is_set():
                return
            self._watch_busy = False
            self._folder_gone = True
            if self._waiting_for_curves():
                self._status(lambda: tr(FOLDER_GONE))

        self.watch_tasks.submit("watch", work, on_done=done, on_error=failed)

    def _accept_new_curves(self, found, stamps):
        if self._folder_gone:  # it is back
            self._folder_gone = False
            if self._waiting_for_curves():
                self._say_waiting()
        known = {path: index for index, path in enumerate(self.paths)}
        queued, fitting = set(self.queue), self._current if self.tasks.is_busy() else None
        seen, new, again = {}, [], []
        for path in found:
            stamp = seen[path] = stamps.get(path)
            settled = stamp is not None and stamp[0] > 0 and self._seen.get(path) == stamp
            index = known.get(path)
            if index is None:
                if settled:
                    new.append(path)
            elif settled and index not in queued and index != fitting and self._read_stamps.get(index, stamp) != stamp:
                again.append(index)
        self._seen = seen
        if new:
            self._add_curves(new)
        if again:  # read while they were written: what was fitted is not the curve
            for index in again:
                self._read_stamps.pop(index, None)
                self._curves.pop(index, None)
                self._invalidate_curve(self.paths[index])
                self.queue.append(index)
            self._log(trf("{count} curves changed after they were read: fitted again.", count=len(again)))
        if new or again:
            self._show_progress()
            if not self.tasks.is_busy():
                self._next()

    def _listing_grew(self, fresh: list) -> None:
        """The record's ``listing`` was made at Start: the curves that arrived since are part of what the run
        fitted (the last frame, how many curves the folder holds and how many of them were to be fitted)."""
        listing = self._run_listing
        if listing is None:
            return
        listing["curves_in_folder"] = len(self.paths)
        if fresh:
            listing["last"] = max(listing["last"], fresh[-1] + 1)
            listing["frames_to_fit"] += len(fresh)
            listing["added_by_watch"] = listing.get("added_by_watch", 0) + len(fresh)

    def _add_curves(self, new: list) -> None:
        """New curves at the end of the list; those among the chosen frames are queued."""
        count = len(self.paths)
        at_end = self.last_spin.value() >= count
        first_name = self._short_name(0) if count else None
        self.paths.extend(new)
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(True)
            spin.setMaximum(len(self.paths))
        if at_end:
            self.last_spin.setValue(len(self.paths))
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(False)
        fresh = [index for index in self._listed() if index >= count]
        self.queue.extend(fresh)
        self._listing_grew(fresh)
        for index in fresh:
            item = QListWidgetItem()
            self.frame_list.addItem(item)
            self._style_item(item, index)
        if count and (self._short_name(0) != first_name or len(str(count)) != len(str(len(self.paths)))):
            self._mark_stages()  # the names share less than before, or the numbers got a digit: every line again
        self._log(tr("{count} new curves.").format(count=len(new)))
        if len(fresh) < len(new):
            self._log(trf("{count} of them are outside the chosen frames and are not fitted.", count=len(new) - len(fresh)))


__all__ = ["FOLDER_GONE", "FitSeriesWatchMixin", "WATCH_MS", "file_stamp"]
