"""The panel above the tabs while a batch runs and when it has ended (layout: ``views/batch_progress_view.py``).

* **Running** — which frames are being reduced, how many are done, the speed and the time left, the latest
  fit. The time left is counted from the frames that are finished: it goes down between two frames and is
  set again when a frame is done, instead of growing while a slow frame is being reduced.
* **Ended** — a title that says how it ended (done / stopped / frames failed), how many frames in how long
  and where, every frame that failed with its reason, what was written, **Open Folder** and **Close**.

Part of ``BatchRunMixin`` (``bindings/batch_run.py``), which runs the batch.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import QSignalBlocker, QUrl
from PyQt5.QtGui import QDesktopServices

from src.gimap.app.presentation.i18n import tr, trf

from ...application import frame_name_of
from ..batch_texts import end_title, failed_lines, latest_fit_text, short_folder, with_fits, written_text

TEXT_S = 0.4


def eta_text(seconds: float) -> str:
    if seconds < 90:
        return tr("about {n} s left").format(n=max(1, int(round(seconds))))
    return tr("about {n} min left").format(n=int(round(seconds / 60.0)))


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds < 3600 else f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class BatchPanelMixin:
    """Needs ``view_model``, ``batch_panel``, the status widgets and the state of ``BatchRunMixin``."""

    def _frame_name(self, request) -> str:
        """A frame that is being reduced, as its files and tables will name it (``frame 4 of 12`` of a source
        that holds several; no file is read for it)."""
        model = self.view_model
        return frame_name_of(request.path, model.describe(request.path), frame_index=request.frame_index,
                             frame_count=model.known_frame_count(request.path), summed=len(request.summed_frames))

    def _batch_panel_title(self) -> None:
        panel, run = self.batch_panel, self._batch_run
        if self._batch_map_only:
            panel.title_label.setText(tr("Series map: {curve}").format(curve=self._series_curve_name()))
            panel.title_label.setToolTip("")
        else:
            panel.title_label.setText(tr("Batch Export → {folder}").format(folder=run.destination.name))
            panel.title_label.setToolTip(str(run.destination))

    def _batch_language(self) -> None:
        """After a switch of the interface language: the panel of the batch running, or of the one that ended."""
        ended = getattr(self, "_batch_ended", None)
        if self.batch_running() and self._batch_run is not None:
            self._batch_panel_title()
            self._batch_panel_update(force=True)
        elif ended is not None and self._batch_run is ended[0] and not self.batch_panel.isHidden():
            self._batch_panel_finish(*ended[1:3], live=ended[3], map_only=ended[4])

    def _batch_panel_start(self) -> None:
        panel, run = self.batch_panel, self._batch_run
        self._batch_panel_title()
        with QSignalBlocker(panel.at_once_combo):
            panel.at_once_combo.clear()
            for count in range(1, self._batch_capacity + 1):
                panel.at_once_combo.addItem(
                    tr("1 frame at a time") if count == 1 else tr("{n} frames at once").format(n=count), count)
            panel.at_once_combo.setCurrentIndex(self._batch_capacity - 1)
        panel.at_once_combo.setVisible(self._batch_capacity > 1)
        with QSignalBlocker(panel.pause_button):
            panel.pause_button.setChecked(False)
        panel.pause_button.setText(tr("Pause"))
        for button in (panel.pause_button, panel.stop_button):
            button.setEnabled(True)
            button.show()
        panel.open_button.hide()
        panel.close_button.hide()
        panel.fit_label.setVisible(False)
        panel.fit_label.setText("")
        panel.failed_label.setVisible(False)
        panel.failed_label.setText("")
        panel.detail_label.show()
        panel.follow_check.setVisible(bool(run.live_key) and self._batch_total > 1)
        panel.bar.setRange(0, max(1, self._batch_total))
        panel.bar.setValue(0)
        self._batch_rate: Optional[float] = None
        self._batch_rate_done, self._batch_rate_at = 0, time.monotonic()
        panel.show()

    def _batch_left(self, done: int, total: int, now: float) -> Optional[float]:
        """Seconds until the end, from the frames that are finished (``None``: none is yet). The speed is taken
        when a frame is done; between two frames the time only goes down."""
        if done != self._batch_rate_done and done:
            self._batch_rate, self._batch_rate_done, self._batch_rate_at = (now - self._batch_started) / done, done, now
        if self._batch_rate is None:
            return None
        return max(0.0, self._batch_rate * (total - done) - (now - self._batch_rate_at))

    def _batch_panel_update(self, *, force: bool = False) -> None:
        run = self._batch_run
        if run is None or not self._batch_timer.isActive():
            return
        now = time.monotonic()
        if not force and now - self._batch_text_time < TEXT_S:
            return
        self._batch_text_time = now
        panel = self.batch_panel
        done = len(run.frames) + len(run.failures)
        total = max(self._batch_total, done)
        elapsed = now - self._batch_started
        panel.bar.setRange(0, max(1, total))
        panel.bar.setValue(done)
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        panel.elapsed_label.setText(_clock(elapsed))
        running = sorted(self._batch_inflight)
        left = self._batch_left(done, total, now)
        per_frame = self._batch_rate
        if self._batch_stopping:
            wait = f" ({tr('a frame takes about {s} s').format(s=f'{per_frame * self._batch_at_once:.0f}')})" if per_frame else ""
            now_text = tr("Stopping — waiting for the {n} frame(s) being reduced").format(n=len(running)) + wait + " …"
        elif self._batch_paused:
            now_text = (tr("Paused — the {n} frame(s) being reduced finish first").format(n=len(running))
                        if running else tr("Paused."))
        elif len(running) == 1:
            index = running[0]
            now_text = tr("Frame {i} of {total}: {name}").format(
                i=index + 1, total=total, name=self._frame_name(self._batch_inflight[index][0]))
        elif running:
            now_text = tr("Frames {first}–{last} of {total} ({n} at once): {name} …").format(
                first=running[0] + 1, last=running[-1] + 1, total=total, n=min(len(running), self._batch_at_once),
                name=self._frame_name(self._batch_inflight[running[0]][0]))
        else:
            now_text = tr("Writing …")
        panel.now_label.setText(now_text)
        parts = [tr("{done} of {total} done").format(done=done, total=total)]
        if run.failures:
            parts.append(tr("{n} failed").format(n=len(run.failures)))
        if per_frame is not None:
            parts.append(tr("{s} s per frame").format(s=f"{per_frame:.1f}"))
            if total > done and not self._batch_paused:
                parts.append(eta_text(left))
        if self._batch_note:
            parts.append(self._batch_note)
        panel.detail_label.setText(" · ".join(parts))
        panel.fit_label.setText(latest_fit_text(run))
        panel.fit_label.setVisible(bool(run.last_fit))
        name = self._frame_name(self._batch_inflight[running[0]][0]) if running else ""
        if self._batch_map_only:  # Series ▸ Build Map writes nothing: it is no batch export
            text = trf("Series map {done}/{total}: {name} …", done=done, total=total, name=name) if name else trf(
                "Series map {done}/{total}", done=done, total=total)
        else:
            text = trf("Batch {done}/{total}: {name} …", done=done, total=total, name=name) if name else trf(
                "Batch {done}/{total}", done=done, total=total)
        if left is not None and total > done and not self._batch_paused:  # no time left is promised while paused
            text += "  " + eta_text(left)
        self._status(text)

    def _batch_panel_finish(self, done: int, seconds: float, *, live: bool = False,
                            map_only: Optional[bool] = None) -> None:
        panel, run = self.batch_panel, self._batch_run
        stopped = self._batch_stopping
        map_only = self._batch_map_only if map_only is None else map_only
        self._batch_ended = (run, done, seconds, live, map_only)  # said again after a language switch
        requested = run.requested or done + len(run.failures)
        whole = done == requested
        if whole:
            said = tr("1 frame in {time}") if done == 1 else tr("{done} frames in {time}")
        else:
            said = tr("{done} of {total} frames in {time}")
        frames = said.format(done=done, total=requested, time=_clock(seconds))
        if map_only:
            panel.title_label.setText(tr("Series map stopped") if stopped else tr("Series map done"))
            panel.title_label.setToolTip("")
            text = frames
            if run.failures:
                text += "; " + tr("{n} failed: {names}").format(n=len(run.failures), names=", ".join(run.failures[:3]))
        else:
            panel.title_label.setText(end_title(run, stopped=stopped))
            panel.title_label.setToolTip(str(run.destination))
            text = with_fits(f"{frames} → {short_folder(run.destination)}", run)
        panel.now_label.setText(text)
        panel.now_label.setToolTip("" if map_only else str(run.destination))
        failed = "" if map_only else failed_lines(run)  # every frame that failed, with its reason
        panel.failed_label.setText(failed)
        panel.failed_label.setVisible(bool(failed))
        details = [] if map_only else [written_text(run)]
        if live:
            details.append(tr("The map of every frame is in the Series tab (Export ▸ Map as CSV Table…)."))
        panel.detail_label.setText("\n".join(line for line in details if line))
        panel.detail_label.setVisible(bool(panel.detail_label.text()))
        panel.fit_label.setText(latest_fit_text(run))  # of the last frame; and again after a switch of the language
        panel.fit_label.setVisible(bool(run.last_fit))
        panel.bar.setRange(0, max(1, requested))
        panel.bar.setValue(done)
        panel.elapsed_label.setText(_clock(seconds))
        for widget in (panel.pause_button, panel.stop_button, panel.at_once_combo, panel.follow_check):
            widget.hide()
        panel.open_button.setVisible(not map_only)
        panel.close_button.show()

    def _batch_open_folder(self) -> None:
        if self._batch_run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._batch_run.destination)))


__all__ = ["BatchPanelMixin", "TEXT_S", "eta_text"]
