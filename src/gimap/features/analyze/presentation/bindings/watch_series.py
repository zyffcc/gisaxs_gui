"""Append complete watched frames to the Series map, using the existing watch request queue.

Auto-exports return their native row from the same reduction. Without exporting, an existing map uses
its ordinary series-row use case on a worker. The map's cut, settings and subset remain fixed; no raw
frame is kept, and batch/watch completions share one deduplicated native-row store.
"""
from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker, QTimer
from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import trf

from ..frames_model import with_cut
from ..texts import message_text
from ..series_texts import SUM_SWITCHED_OFF

CORRECTED_MAP = ("This map uses batch series corrections. Watching cannot append uncorrected frames: "
                 "Build Map again or rerun Batch Export with the same corrections.")


class WatchSeriesMixin:
    def _stop_watch_series(self) -> None:
        self._watch_series_epoch = object()
        self._watch_series_waiting = []
        self._watch_series_built = None
        for job in getattr(self, "_watch_series_jobs", ()):
            self.tasks.cancel(job)
        self._watch_series_jobs = set()
        self._watch_series_rereading = False
        self._watch_series_correction_warned = None
        self.series_watch_follow_check.hide()
        timer = getattr(self, "_watch_series_timer", None)
        if timer is not None:
            timer.stop()
        self.tasks.cancel("watch-series")

    def _watch_series_start(self) -> None:
        self._stop_watch_series()
        if not hasattr(self, "_watch_series_timer"):
            self._watch_series_timer = QTimer(self)
            self._watch_series_timer.setSingleShot(True)
            self._watch_series_timer.setInterval(1000)
            self._watch_series_timer.timeout.connect(self._watch_series_redraw)
            self.analysisShown.connect(self._watch_series_ready)
        self._watch_series_followed = self._series_row
        if self._series_map is not None and self._series_row != self._series_map.rows - 1:
            with QSignalBlocker(self.batch_panel.follow_check):
                self.batch_panel.follow_check.setChecked(False)
        with QSignalBlocker(self.series_watch_follow_check):
            self.series_watch_follow_check.setChecked(self.batch_panel.follow_check.isChecked())
        self.series_watch_follow_check.show()

    def _watch_series_ready(self, _analysis) -> None:
        if self.view_model.watch.active and self._watch_series_waiting:
            self._watch_dispatch_requests(())

    def _watch_series_follow_toggled(self, follow: bool) -> None:
        with QSignalBlocker(self.batch_panel.follow_check):
            self.batch_panel.follow_check.setChecked(follow)
        if follow and self.view_model.watch.active:
            self._watch_series_followed = self._series_row
            self._watch_series_redraw()

    def _watch_series_align_groups(self) -> None:
        """Complete a map's open final sum at its original origin, including its already measured tail.

        The ordinary watch grouping queue owns these references. A worker rereads the short group's
        earlier frames when it becomes complete; a complete sum is never exported again.
        """
        built = getattr(self, "_series_built", {})
        model = self.view_model
        if (self._watch_series_corrected() or (self._series_map is None and not self.batch_running())
                or model.series_settings_changed(built) or built.get("last")):
            return
        model._watch_cut = built.get("cut")
        if int(built.get("group", 1)) <= 1:
            return
        refs = [*built.get("watch_tail", ()), *model._watch_pending]
        unique = {(str(path).casefold(), int(index)): (path, index) for path, index in refs}
        numbers = model.frame_numbers()
        model._watch_pending = [unique[key] for key in sorted(unique, key=lambda key: numbers.get(key, float("inf")))]

    def _watch_series_corrected(self) -> bool:
        correction = getattr(getattr(self, "_series_run", None), "series", None)
        return correction is not None and not correction.is_identity

    def _watch_series_sum(self, *, exporting: bool) -> None:
        model = self.view_model
        if not exporting or self._series_map is not None or model.sum_set_by != "automatic" or model.sum_count <= 1:
            return
        previous = model.sum_count
        pending = list(model._watch_pending)
        self._watch_series_rereading = True
        self.sum_spin.setValue(1)
        if model.sum_count != 1:
            model.set_sum_count(1)
        self.run_analysis()  # supersede an in-flight summed frame even when the spin binding changed the model
        model._watch_pending = pending
        text = trf(SUM_SWITCHED_OFF, n=previous)
        self._status(text, "warning")
        show_toast(self.window(), text, level="warning")

    def _watch_series_prepare(self, requests, *, exporting: bool):
        if not hasattr(self, "_watch_series_epoch"):
            self._watch_series_start()
        model = self.view_model
        requests = [*self._watch_series_waiting, *requests]
        self._watch_series_waiting = []
        analysis = model.state.analysis
        if analysis is None or (self._watch_series_rereading and len(analysis.summed_frames) > 0):
            self._watch_series_waiting = requests
            return []
        self._watch_series_rereading = False
        has_map = (self._series_map is not None or self.batch_running()
                   or (self._watch_series_built is not None and self._watch_series_built is self._series_built))
        stored_key = getattr(self, "_series_key", None)
        live = analysis.reduction is not None and (has_map or exporting)
        if has_map and self._watch_series_corrected():
            live = False
            self._series_say(CORRECTED_MAP)
            if self._watch_series_correction_warned is not self._series_built:
                self._watch_series_correction_warned = self._series_built
                show_toast(self.window(), trf(CORRECTED_MAP), level="warning")
        if has_map and (model.series_settings_changed(self._series_built)
                        or stored_key is None or self.series_curve_combo.currentData() != stored_key):
            live = False
        key = stored_key if has_map else self.series_curve_combo.currentData()
        if live and not has_map:
            key = key or ("horizontal" if analysis.kind == "gisaxs" else "radial")
            self._series_keep_curve(key)
            self._series_run = None
            self._series_built["every"] = max(1, self.series_step_spin.value())
            if requests:
                earliest = min(model.frame_numbers().get((str(request.path).casefold(), int(request.frame_index)), 1)
                               for request in requests)
                self._series_built["first"] = max(self._series_built["first"], earliest)
            self._watch_series_followed = self._series_row
        built = self._series_built if live else None
        self._watch_series_built = built
        token = self._watch_series_epoch
        prepared = []
        for request in requests:
            row_key = key if live else None
            if built is not None:
                request = with_cut(request, built.get("cut"))
                with model.only_frames(built.get("first", 1), built.get("last") or None):
                    group = model.batch_group_index(request.path, request.frame_index)
                if group is None or group % max(1, built.get("every", 1)):
                    row_key = None
            prepared.append((request, row_key, token, built))
        return prepared

    def _watch_series_done(self, value, key, token, built, job: str = "") -> None:
        self._watch_series_jobs.discard(job)
        model = self.view_model
        if token is not getattr(self, "_watch_series_epoch", None) or not model.watch.active or key is None:
            return
        if (self._watch_series_corrected() or built is not self._series_built or model.series_settings_changed(built)
                or self.series_curve_combo.currentData() != key):
            return
        row = getattr(value, "live_row", value if isinstance(value, tuple) else None)
        if getattr(value, "live_error", ""):
            self._status(trf("Could not append a frame to the Series map: {error}",
                             error=message_text(value.live_error)), "warning")
        if row is None or len(row) != 5:
            return
        numbers = model.frame_numbers()
        identity = (str(row[4][0]).casefold(), int(row[4][1]))
        if identity not in numbers:
            return
        existing = next((i for i, old in enumerate(self._series_rows)
                         if (str(old[4][0]).casefold(), int(old[4][1])) == identity), None)
        if existing is not None:
            old = self._series_rows[existing]
            old_sum, new_sum = getattr(old[3], "summed", 1), getattr(row[3], "summed", 1)
            if old_sum >= int(built.get("group", 1)) or new_sum <= old_sum:
                return
            self._series_rows[existing] = row
        else:
            self._series_rows.append(row)
        self._watch_series_normalize()
        if not self._watch_series_timer.isActive():
            self._watch_series_timer.start()

    def _watch_series_failed(self, message, name, token, job, *, exporting: bool) -> None:
        self._watch_series_jobs.discard(job)
        if token is not self._watch_series_epoch or not self.view_model.watch.active:
            return
        template = ("Auto-export of {name} failed: {error}" if exporting
                    else "Could not append {name} to the Series map: {error}")
        self._status(trf(template, name=name, error=message_text(message)), "error")

    def _watch_series_normalize(self) -> None:
        if not self.view_model.watch.active:
            return
        numbers = self.view_model.frame_numbers()
        unique = {}
        selected = None
        if self._series_rows and 0 <= self._series_row < len(self._series_rows):
            ref = self._series_rows[self._series_row][4]
            selected = (str(ref[0]).casefold(), int(ref[1]))
        for row in self._series_rows:
            ref = row[4]
            identity = (str(ref[0]).casefold(), int(ref[1]))
            if identity in numbers:
                old = unique.get(identity)
                if old is None or getattr(row[3], "summed", 1) > getattr(old[3], "summed", 1):
                    unique[identity] = row
        identities = sorted(unique, key=numbers.get)
        self._series_rows = [unique[key] for key in identities]
        if selected in unique:
            self._series_row = identities.index(selected)
        self._series_built["numbers"] = numbers
        self._series_built["listed"] = self.view_model.listed_frames()

    def _watch_series_redraw(self) -> None:
        if not self.view_model.watch.active or not self._series_rows:
            return
        if self._watch_series_corrected() or self.view_model.series_settings_changed(self._series_built):
            return
        self._watch_series_normalize()
        if self.batch_running():
            return  # BatchLive redraws the shared rows while its workers run.
        had_map = self._series_map is not None
        follow = self.batch_panel.follow_check.isChecked()
        with QSignalBlocker(self.series_watch_follow_check):
            self.series_watch_follow_check.setChecked(self.batch_panel.follow_check.isChecked())
        if follow:
            self._series_row = len(self._series_rows) - 1
        self._watch_series_followed = self._series_row
        self._clear_stages()
        self._show_series_map(keep_view=had_map and not follow, keep_q=had_map)
        self._find_stages()
