"""Running a batch: frames reduced in the background, several at once when the computer allows, shown live.

* **Where** — with *Gentle* (or a batch of one or two frames) one frame at a time in the application,
  as before; otherwise in worker processes at below-normal priority (``view_model.open_workers``),
  as many as ``frames_at_once`` allows for the cores and the free memory. Only one frame more than
  there are workers waits in line, so memory stays bounded. If a worker process dies (out of
  memory …) the rest of the batch goes on in the application, one frame at a time.
* **Order** — outcomes may arrive in any order; they are folded into the batch in frame order
  (tables, fits started from the previous frame's result, the live map). Fits run off the GUI thread.
* **Live** — the panel above the tabs says which frames are being reduced, how many are done, the
  speed and the time left, the latest fit; the Series tab shows the map of every frame done so far
  (redrawn at most once a second) and the newest frame's curve.
* **Control** — *Pause* starts no new frames; *Stop* drops the frames not started and waits for
  those being reduced (the panel says so — a frame cannot be interrupted); fewer frames at once can
  be chosen while running. What is written stays; the tables are written for the frames done.
"""

from __future__ import annotations

import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QSignalBlocker, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices

from src.gimap.app.presentation.i18n import tr, trf

from ...application import BALANCED, FAST, FOLDERS, GENTLE, GISAXS, BatchChoices, frames_at_once

POLL_MS = 100
REDRAW_S = 1.0
"""The live map is redrawn at most this often."""
TEXT_S = 0.4
DEFAULT_PIXELS = 4_000_000
LIVE_CURVE = {GISAXS: "horizontal"}


def eta_text(seconds: float) -> str:
    if seconds < 90:
        return tr("about {n} s left").format(n=max(1, int(round(seconds))))
    return tr("about {n} min left").format(n=int(round(seconds / 60.0)))


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds < 3600 else f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _frame_name(request) -> str:
    frame = f" #{request.frame_index + 1}" if request.frame_index else ""
    return f"{request.path.name}{frame}"


class BatchRunMixin:
    """Needs ``view_model``, ``tasks``, ``batch_panel``, the Series tab and the status widgets."""

    def _connect_batch_run(self) -> None:
        self._batch_timer = QTimer(self)
        self._batch_timer.setInterval(POLL_MS)
        self._batch_timer.timeout.connect(self._batch_poll)
        self._batch_workers = None
        self._batch_inflight: dict = {}
        self._batch_ready: dict = {}
        self._batch_live = False
        self._batch_folding = False
        self._batch_map_only = False
        self._batch_run = None
        panel = self.batch_panel
        panel.pause_button.toggled.connect(self._batch_pause_toggled)
        panel.stop_button.clicked.connect(self.cancel_batch)
        panel.close_button.clicked.connect(panel.hide)
        panel.open_button.clicked.connect(self._batch_open_folder)
        panel.at_once_combo.currentIndexChanged.connect(self._batch_at_once_changed)
        panel.follow_check.toggled.connect(lambda _on: self._batch_live_redraw(force=True))
        self.series_map_view.positionClicked.connect(lambda *_args: self._batch_unfollow())

    # -- how fast ------------------------------------------------------------------------

    def batch_running(self) -> bool:
        return bool(self._batch) or bool(self._batch_inflight) or self._batch_timer.isActive()

    def batch_kind(self) -> str:
        """What the batch is (meaningful while ``batch_running()``): ``series_map`` (Series ▸ Build Map, nothing
        written) or ``export`` (Batch Export, also when it fills the map)."""
        return "series_map" if self._batch_map_only else "export"

    def _batch_pixels(self) -> int:
        analysis = self.view_model.state.analysis
        if analysis is None:
            return DEFAULT_PIXELS
        rows, columns = analysis.shape
        return max(1, int(rows) * int(columns))

    def batch_speed_options(self, frames: int) -> list[tuple[str, str]]:
        """``(speed, text)`` for the Batch Export dialog, with the frames at once this computer allows."""
        if self.view_model.frame_workers is None:
            return [(GENTLE, tr("One frame at a time"))]
        cores, free = self.view_model.resources()
        count = {speed: frames_at_once(speed, cores=cores, free_bytes=free, frame_pixels=self._batch_pixels(),
                                       frames=int(frames))
                 for speed in (BALANCED, FAST)}
        if count[FAST] < 2:
            return [(GENTLE, tr("One frame at a time (a short batch: more at once would not be faster)"))]
        return [
            (GENTLE, tr("Gentle — one frame at a time; the computer stays free")),
            (BALANCED, tr("Balanced — {n} frames at once (recommended)").format(n=count[BALANCED])),
            (FAST, tr("Fast — {n} frames at once; other programs get slower").format(n=count[FAST])),
        ]

    # -- running -------------------------------------------------------------------------

    def run_batch(self, target: Path, choices: BatchChoices, *, series=None, stem: str = "batch",
                  then: Optional[Callable[[Path], None]] = None, map_only: bool = False,
                  live_key: Optional[str] = None) -> None:
        """Every listed frame (every n-th) with ``choices``; ``map_only``: nothing written, the Series map only."""
        requests = self.view_model.batch_requests()[:: max(1, int(choices.every))]
        if not map_only:
            requests = self._without_odd_frames(requests)
        if not requests or self.batch_running():
            return
        if not map_only:
            try:
                Path(target).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._status(tr("Cannot write to {folder}: {error}").format(folder=target, error=exc.strerror or exc), "error")
                return
        self._batch_map_only = bool(map_only)
        live_key = live_key or self._batch_live_key()
        self._batch_run = self.view_model.start_batch(Path(target), choices, stem, series, live_key=live_key)
        self._batch_run.display = self.batch_display(choices)
        self._batch = list(enumerate(requests))
        self._batch_total = len(requests)
        self._batch_then = then
        self._batch_started = time.monotonic()
        self._batch_inflight, self._batch_ready = {}, {}
        self._batch_next_fold = 0
        self._batch_folding = self._batch_stopping = self._batch_paused = False
        self._batch_note = ""
        self._batch_text_time = 0.0
        cores, free = self.view_model.resources()
        count = frames_at_once(choices.speed, cores=cores, free_bytes=free, frame_pixels=self._batch_pixels(),
                               frames=len(requests))
        self._batch_workers = None
        if count > 1:
            try:
                self._batch_workers = self.view_model.open_workers(count)
            except (OSError, RuntimeError, ValueError) as exc:
                self._batch_note = tr("Worker processes could not start ({error}): one frame at a time.").format(error=exc)
        self._batch_capacity = count if self._batch_workers is not None else 1
        self._batch_at_once = self._batch_capacity
        self.batch_button.setEnabled(False)
        self.export_all_button.setEnabled(False)
        self.series_build_button.setEnabled(False)
        self.progress_bar.setRange(0, self._batch_total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.cancel_button.show()
        try:
            self.cancel_button.clicked.disconnect()
        except TypeError:
            pass
        self.cancel_button.clicked.connect(self.cancel_batch)
        self._batch_panel_start()
        self._batch_live_start(live_key)
        self._batch_timer.start()
        self._batch_fill()
        self._batch_panel_update(force=True)

    def cancel_batch(self) -> None:
        """Stop: the frames not started are dropped; those being reduced finish (their files stay)."""
        if not self.batch_running():
            return
        self._batch_total -= len(self._batch)
        self._batch = []
        self._batch_stopping = True
        for index, (_request, future) in list(self._batch_inflight.items()):
            if future is not None and future.cancel():  # still waiting in line
                self._batch_inflight.pop(index)
                self._batch_total -= 1
                self._batch_ready[index] = ("skipped", "")
        self.batch_panel.stop_button.setEnabled(False)
        self.batch_panel.pause_button.setEnabled(False)
        self._batch_fold_next()
        self._batch_panel_update(force=True)
        self._batch_check_done()

    def shutdown_batch(self) -> None:
        """The page closes: no new frames, worker processes ended."""
        self._batch = []
        self._batch_timer.stop()
        if self._batch_workers is not None:
            self._batch_workers.shutdown(wait=False)
            self._batch_workers = None
        self._batch_inflight = {}

    def _batch_pause_toggled(self, paused: bool) -> None:
        self._batch_paused = bool(paused)
        self.batch_panel.pause_button.setText(tr("Resume") if paused else tr("Pause"))
        if not paused:
            self._batch_fill()
        self._batch_panel_update(force=True)

    def _batch_at_once_changed(self, _index: int) -> None:
        value = self.batch_panel.at_once_combo.currentData()
        if value:
            self._batch_at_once = int(value)
            self._batch_fill()

    def _batch_fill(self) -> None:
        """Start frames until as many run as allowed (plus one waiting in line for the workers)."""
        if self._batch_paused or self._batch_stopping or self._batch_run is None:
            return
        workers = self._batch_workers
        limit = self._batch_at_once + (1 if workers is not None and self._batch_at_once > 1 else 0)
        while self._batch and len(self._batch_inflight) < limit:
            if self.view_model.needs_first_frame_alone(self._batch_run) and (
                self._batch_inflight or self._batch_ready or self._batch_folding
            ):
                break  # the first frame gives the normalisation factor of the others
            index, request = self._batch.pop(0)
            job = self.view_model.frame_job(index, request, self._batch_run)
            if workers is not None:
                try:
                    self._batch_inflight[index] = (request, workers.submit(job))
                    continue
                except (RuntimeError, OSError, BrokenProcessPool) as exc:
                    self._batch_lost_workers(str(exc))
                    workers = None
            self._batch_inflight[index] = (request, None)
            self.tasks.submit(
                f"batch-{index}", lambda job=job: self.view_model.run_job(job),
                on_done=lambda outcome, index=index: self._batch_arrived(index, outcome),
                on_error=lambda message, _details, index=index: self._batch_arrived(index, None, message),
            )

    def _batch_lost_workers(self, message: str) -> None:
        """A worker process died: go on in the application, one frame at a time.

        Frames the workers finished are kept; the others given to them are reduced again, here."""
        finished = []
        for index, (request, future) in list(self._batch_inflight.items()):
            if future is None:
                continue
            self._batch_inflight.pop(index)
            if future.done() and not future.cancelled() and future.exception() is None:
                finished.append((index, future.result()))
            else:
                self._batch.append((index, request))
        self._batch.sort(key=lambda item: item[0])
        if self._batch_workers is not None:
            self._batch_workers.shutdown(wait=False)
        self._batch_workers = None
        for index, outcome in finished:
            self._batch_ready[index] = outcome
        self._batch_capacity = self._batch_at_once = 1
        self._batch_note = tr("A worker process stopped ({error}); the batch goes on one frame at a time.").format(
            error=message or "out of memory?")
        with QSignalBlocker(self.batch_panel.at_once_combo):
            self.batch_panel.at_once_combo.hide()

    def _batch_poll(self) -> None:
        for index, (request, future) in list(self._batch_inflight.items()):
            if future is None or not future.done():
                continue
            if future.cancelled():
                self._batch_inflight.pop(index)
                self._batch_total -= 1
                self._batch_ready[index] = ("skipped", "")
                continue
            try:
                outcome = future.result()
            except BrokenProcessPool as exc:
                self._batch_lost_workers(str(exc))  # this frame and the others waiting: again, here
                break
            except Exception as exc:  # noqa: BLE001 - the reason is shown with the frame
                self._batch_arrived(index, None, str(exc) or type(exc).__name__)
                continue
            self._batch_arrived(index, outcome)
            if not self._batch_timer.isActive():
                return  # that was the last frame: the batch is finished
        self._batch_fold_next()
        self._batch_fill()
        self._batch_live_redraw()
        self._batch_panel_update()
        self._batch_check_done()

    def _batch_arrived(self, index: int, outcome, error: str = "") -> None:
        entry = self._batch_inflight.pop(index, None)
        if outcome is None:
            name = _frame_name(entry[0]) if entry else f"frame {index + 1}"
            self._batch_ready[index] = ("failed", f"{name} ({error})")
        else:
            self._batch_ready[index] = outcome
        self._batch_fold_next()
        self._batch_fill()
        self._batch_check_done()

    def _batch_fold_next(self) -> None:
        """Fold the outcomes that are next in frame order (fits off the GUI thread, one at a time)."""
        run = self._batch_run
        while run is not None and not self._batch_folding and self._batch_next_fold in self._batch_ready:
            index = self._batch_next_fold
            item = self._batch_ready.pop(index)
            self._batch_next_fold += 1
            if isinstance(item, tuple):
                if item[0] == "failed":
                    run.failures.append(item[1])
                continue
            if run.fit_table is None:
                self.view_model.fold_outcome(run, item)
                self._batch_folded(item)
                continue
            self._batch_folding = True
            self.tasks.submit(
                f"batch-fold-{index}", lambda item=item: self.view_model.fold_outcome(run, item),
                on_done=lambda _written, item=item: self._batch_folded(item, later=True),
                on_error=lambda message, _details, item=item: self._batch_fold_failed(item, message),
            )

    def _batch_folded(self, outcome, *, later: bool = False) -> None:
        if self._batch_live and outcome.live_row is not None:
            self._series_rows.append(outcome.live_row)
        if later:
            self._batch_folding = False
            self._batch_fold_next()
            self._batch_fill()
            self._batch_check_done()

    def _batch_fold_failed(self, outcome, message: str) -> None:
        self._batch_run.failures.append(f"{outcome.label} ({message})")
        self._batch_folding = False
        self._batch_fold_next()
        self._batch_fill()
        self._batch_check_done()

    def _batch_check_done(self) -> None:
        if self._batch_run is None or not self._batch_timer.isActive():
            return
        if self._batch or self._batch_inflight or self._batch_ready or self._batch_folding:
            return
        self._finish_batch()

    def _finish_batch(self) -> None:
        self._batch_timer.stop()
        if self._batch_workers is not None:
            self._batch_workers.shutdown(wait=False)
            self._batch_workers = None
        run = self._batch_run
        destination = run.destination
        done = len(run.frames)
        if done and not self._batch_map_only:
            try:
                self.view_model.finish_batch(run)
            except (OSError, ValueError) as exc:
                run.failures.append(f"tables ({exc})")
        self.batch_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(self.tasks.is_busy())
        self.cancel_button.hide()
        seconds = time.monotonic() - self._batch_started
        live = self._batch_live
        if self._batch_map_only:
            self._batch_live = False
            self._series_finished(list(run.failures))
            self._batch_panel_finish(done, seconds, live=False)
            self._batch_map_only = False
            if done and not run.failures and not self._batch_stopping:
                self.batch_panel.hide()  # the map says it (with a toast and the status line): its room goes to the map
            return
        level = "warning" if run.failures else "ok"
        values = dict(done=done, total=self._batch_total, folder=destination, seconds=f"{seconds:.0f}")
        text = (trf("Exported {done}/{total} frames to {folder} in {seconds} s; failed: {names}",
                    names=", ".join(run.failures[:5]), **values) if run.failures else
                trf("Exported {done}/{total} frames to {folder} in {seconds} s", **values))
        if level == "ok":
            self.notify_written(text, destination)
        else:
            self._status(text, level)
        self._batch_live_finish()
        self._batch_panel_finish(done, seconds, live=live)
        then, self._batch_then = self._batch_then, None
        if then is not None and done:
            try:
                then(destination / FOLDERS["fit_input"] if run.choices.fit_input else destination)
            except (RuntimeError, ValueError, OSError) as exc:
                self._status(trf("Could not open the series in Fitting: {error}", error=exc), "error")

    # -- the panel -----------------------------------------------------------------------

    def _batch_panel_start(self) -> None:
        panel, run = self.batch_panel, self._batch_run
        if self._batch_map_only:
            panel.title_label.setText(tr("Series map: {curve}").format(curve=self.series_curve_combo.currentText()))
            panel.title_label.setToolTip("")
        else:
            panel.title_label.setText(tr("Batch Export → {folder}").format(folder=run.destination.name))
            panel.title_label.setToolTip(str(run.destination))
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
        panel.detail_label.show()
        panel.follow_check.setVisible(bool(run.live_key) and self._batch_total > 1)
        panel.bar.setRange(0, max(1, self._batch_total))
        panel.bar.setValue(0)
        panel.show()

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
        per_frame = elapsed / done if done else None
        if self._batch_stopping:
            wait = f" ({tr('a frame takes about {s} s').format(s=f'{per_frame * self._batch_at_once:.0f}')})" if per_frame else ""
            now_text = tr("Stopping — waiting for the {n} frame(s) being reduced").format(n=len(running)) + wait + " …"
        elif self._batch_paused:
            now_text = (tr("Paused — the {n} frame(s) being reduced finish first").format(n=len(running))
                        if running else tr("Paused."))
        elif len(running) == 1:
            index = running[0]
            now_text = tr("Frame {i} of {total}: {name}").format(
                i=index + 1, total=total, name=_frame_name(self._batch_inflight[index][0]))
        elif running:
            now_text = tr("Frames {first}–{last} of {total} ({n} at once): {name} …").format(
                first=running[0] + 1, last=running[-1] + 1, total=total, n=min(len(running), self._batch_at_once),
                name=_frame_name(self._batch_inflight[running[0]][0]))
        else:
            now_text = tr("Writing …")
        panel.now_label.setText(now_text)
        parts = [tr("{done} of {total} done").format(done=done, total=total)]
        if run.failures:
            parts.append(tr("{n} failed").format(n=len(run.failures)))
        if per_frame is not None:
            parts.append(tr("{s} s per frame").format(s=f"{per_frame:.1f}"))
            if total > done and not self._batch_paused:
                parts.append(eta_text(per_frame * (total - done)))
        if self._batch_note:
            parts.append(self._batch_note)
        panel.detail_label.setText(" · ".join(parts))
        if run.last_fit:
            panel.fit_label.show()
            panel.fit_label.setText(tr("Latest fit ({frame}): {fit}").format(frame=run.frames[-1] if run.frames else "", fit=run.last_fit))
        name = _frame_name(self._batch_inflight[running[0]][0]) if running else ""
        text = trf("Batch {done}/{total}: {name} …", done=done, total=total, name=name) if name else             trf("Batch {done}/{total}", done=done, total=total)
        if per_frame is not None and total > done:
            text += "  " + eta_text(per_frame * (total - done))
        self._status(text)

    def _batch_panel_finish(self, done: int, seconds: float, *, live: bool = False) -> None:
        panel, run = self.batch_panel, self._batch_run
        stopped = self._batch_stopping
        if self._batch_map_only:
            panel.title_label.setText(tr("Series map stopped") if stopped else tr("Series map done"))
            text = tr("{done} frames in {time}").format(done=done, time=_clock(seconds))
        else:
            panel.title_label.setText(tr("Batch Export stopped") if stopped else tr("Batch Export done"))
            text = tr("{done} frames in {time} → {folder}").format(done=done, time=_clock(seconds), folder=run.destination)
        if run.failures:
            text += "; " + tr("{n} failed: {names}").format(n=len(run.failures), names=", ".join(run.failures[:3]))
        panel.now_label.setText(text)
        panel.detail_label.setText(
            tr("The map of every frame is in the Series tab (Export ▸ Map as CSV Table…).") if live else "")
        panel.detail_label.setVisible(live)
        panel.fit_label.setVisible(bool(run.last_fit))
        panel.bar.setRange(0, max(1, done))
        panel.bar.setValue(done)
        panel.elapsed_label.setText(_clock(seconds))
        for widget in (panel.pause_button, panel.stop_button, panel.at_once_combo, panel.follow_check):
            widget.hide()
        panel.open_button.setVisible(not self._batch_map_only)
        panel.close_button.show()

    def _batch_open_folder(self) -> None:
        if self._batch_run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._batch_run.destination)))

    # -- live in the Series tab ------------------------------------------------------------

    def _batch_live_key(self) -> Optional[str]:
        analysis = self.view_model.state.analysis
        reduction = analysis.reduction if analysis is not None else None
        keys = [curve.key for curve in (reduction.curves if reduction is not None else ()) if not curve.is_empty]
        current = self.series_curve_combo.currentData()
        if current in keys:
            return current
        wanted = LIVE_CURVE.get(getattr(analysis, "kind", None), "radial")
        return wanted if wanted in keys else (keys[0] if keys else None)

    def _batch_live_start(self, key: Optional[str]) -> None:
        self._batch_live = bool(key) and (self._batch_total > 1 or self._batch_map_only)
        if not self._batch_live:
            return
        self._series_rows, self._series_failures = [], []
        self._series_key = key
        self._series_map = None
        self._clear_stages()
        self._series_row = 0
        self._batch_last_draw, self._batch_rows_drawn = 0.0, 0
        index = self.series_curve_combo.findData(key)
        if index >= 0:
            with QSignalBlocker(self.series_curve_combo):
                self.series_curve_combo.setCurrentIndex(index)
        with QSignalBlocker(self.batch_panel.follow_check):
            self.batch_panel.follow_check.setChecked(True)
        if self._batch_map_only:
            self._series_say("Reducing {n} frames: each appears here as soon as it is done.", n=self._batch_total)
        else:
            self._series_say("Batch Export: every frame appears here as soon as it is done.")
        self.show_right("series")

    def _batch_live_redraw(self, *, force: bool = False) -> None:
        if not self._batch_live or not self._series_rows:
            return
        if not force and len(self._series_rows) == self._batch_rows_drawn:
            return
        now = time.monotonic()
        if not force and now - self._batch_last_draw < REDRAW_S:
            return
        self._batch_last_draw, self._batch_rows_drawn = now, len(self._series_rows)
        follow = self.batch_panel.follow_check.isChecked()
        if follow:
            self._series_row = len(self._series_rows) - 1
        had_map = self._series_map is not None
        self._show_series_map(keep_view=not follow and had_map, keep_q=had_map)

    def _batch_unfollow(self) -> None:
        if self._batch_live and self.batch_running() and self.batch_panel.follow_check.isChecked():
            with QSignalBlocker(self.batch_panel.follow_check):
                self.batch_panel.follow_check.setChecked(False)

    def _batch_live_finish(self) -> None:
        if self._batch_live:
            self._batch_live_redraw(force=True)
            self._find_stages()
        self._batch_live = False
        self.series_build_button.setEnabled(self.series_curve_combo.count() > 0)


__all__ = ["BatchRunMixin", "eta_text"]
