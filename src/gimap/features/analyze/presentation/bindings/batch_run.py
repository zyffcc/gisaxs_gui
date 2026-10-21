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
* **Before the first frame** — the folder is tried (a file is made in it and taken away): a folder that
  takes no file is said at once, with the way to another one, not after an hour of reduction. An earlier
  export of the same data in it is replaced when that was chosen (``run_batch(replace_earlier=True)``),
  else kept and named in the record. If the folder stops taking files while the batch runs (disk full),
  the batch ends there.
* **The end** — a batch that was stopped, or had frames that failed, is not a success: status line, notice,
  panel, ``README.txt`` and the record say what happened and how far it got (``batch_texts.py``).

The panel is ``bindings/batch_panel.py``, the live map ``bindings/batch_live.py``.
"""

from __future__ import annotations

import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QSignalBlocker, QTimer

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

from ...application import (
    BALANCED,
    FAST,
    GENTLE,
    BatchChoices,
    cannot_write,
    earlier_export,
    frames_at_once,
    remove_export,
    source_name,
)
from ..batch_texts import failure_reason, short_folder
from ..pending_frames import LazyBatchNumbers, PendingFrames
from .batch_finish import BatchFinishMixin
from .batch_live import LIVE_CURVE, REDRAW_S, BatchLiveMixin
from .batch_panel import TEXT_S, BatchPanelMixin, eta_text

POLL_MS = 100
DEFAULT_PIXELS = 4_000_000


class BatchRunMixin(BatchPanelMixin, BatchLiveMixin, BatchFinishMixin):
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
        self._batch_finishing = False
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
        return bool(self._batch) or bool(self._batch_inflight) or self._batch_timer.isActive() or self._batch_finishing

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

    def _batch_frames_at_once(self, speed: str, frames: int) -> int:
        """How many frames a batch of ``frames`` would reduce at once with ``speed`` on this computer."""
        if self.view_model.frame_workers is None:
            return 1
        cores, free = self.view_model.resources()
        return frames_at_once(speed, cores=cores, free_bytes=free, frame_pixels=self._batch_pixels(), frames=int(frames))

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
                  live_key: Optional[str] = None, replace_earlier: bool = False) -> None:
        """Every listed frame (every n-th) with ``choices``; ``map_only``: nothing written, the Series map only.
        ``replace_earlier``: an earlier export of the same data in ``target`` is taken away first (its files as
        its record lists them, and the curves for Fitting when this batch writes some); else it stays, and the
        record of this batch says that the folder held one."""
        listed = self.view_model.batch_requests()
        requests = listed[:: max(1, int(choices.every))]
        if not map_only:
            requests = self._without_odd_frames(requests)
        if not requests or self.batch_running():
            return
        earlier = None
        if not map_only:
            error = cannot_write(Path(target))  # before the first frame: a folder that takes no file is said now
            if error is not None:
                self._batch_cannot_write(Path(target), error)
                return
            earlier = earlier_export(Path(target), stem, fit_inputs=choices.fit_input)
            if earlier is not None and replace_earlier:
                try:
                    removed = remove_export(earlier)
                except OSError as exc:
                    self._status(trf("The earlier export in {folder} could not be removed: {error}",
                                     folder=Path(target).name, error=exc.strerror or exc), "error")
                    return
                earlier = {"action": "replaced", "when": earlier.when, "removed": removed, "complete": earlier.complete,
                           "record": earlier.record.name if earlier.record is not None else ""}
            elif earlier is not None:
                earlier = {"action": "kept", "when": earlier.when,
                           "record": earlier.record.name if earlier.record is not None else ""}
        self._batch_map_only = bool(map_only)
        live_key = live_key or self._batch_live_key()
        self._batch_run = run = self.view_model.start_batch(Path(target), choices, stem, series, live_key=live_key)
        run.display = self.batch_display(choices)
        run.requested, run.earlier = len(requests), earlier or {}
        self._series_fit_start(run)
        if not map_only:
            self._batch_describe(run)
        self._batch = PendingFrames(requests)
        self._batch_numbers = LazyBatchNumbers(requests, self.view_model.frame_numbers())
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
        self._batch_live_start(live_key)  # first: the panel's title names the curve the map holds
        self._batch_panel_start()
        self._batch_timer.start()
        self._batch_fill()
        self._batch_panel_update(force=True)

    def _batch_cannot_write(self, target: Path, error: str) -> None:
        """The folder of a batch takes no file: one sentence (the folder by its last names: the line is one line),
        and the dialog again to choose another one."""
        text = trf("Cannot write to {folder}: {error}", folder=short_folder(target), error=error)
        self._status(text, "error")
        again = getattr(self, "_batch_again", None)
        show_toast(self, text, level="error", action=(tr("Choose Another Folder…"), again) if again is not None else None)

    def _batch_describe(self, run) -> None:
        """What the record of the batch says of its set-up: the cut held for every frame (the view model says
        it when it holds one) and every listed source that consists of several files."""
        model = self.view_model
        held, said = getattr(model, "series_cut", None), getattr(model, "series_cut_text", None)
        try:
            run.series_cut = held() if held is not None else None
            if said is not None:
                template, values = said()
                run.series_cut_text = str(template).format(**values)
        except Exception:  # noqa: BLE001 - a sentence for the record, never a reason not to export
            run.series_cut, run.series_cut_text = None, ""
        for path in model.state.files:
            layout = model.describe(path)
            if layout is not None and (len(layout.members) > 1 or layout.warnings):
                run.sources.append({
                    "name": source_name(path, layout), "summary": layout.summary_text(), "frames": int(layout.frames),
                    "files": [member.name for member in layout.members], "complete": bool(layout.complete),
                    "warnings": list(layout.warning_texts()),
                })

    def cancel_batch(self) -> None:
        """Stop: the frames not started are dropped; those being reduced finish (their files stay)."""
        if not self.batch_running():
            return
        if self._batch_finishing:
            return  # the finished frames' tables and record are being saved
        run = self._batch_run
        dropped = self._batch.names(self._frame_name) if isinstance(self._batch, PendingFrames) else [
            self._frame_name(request) for _index, request in self._batch]
        self._batch_total -= len(self._batch)
        self._batch = []
        self._batch_stopping = True
        for index, (request, future) in list(self._batch_inflight.items()):
            if future is not None and future.cancel():  # still waiting in line
                self._batch_inflight.pop(index)
                self._batch_total -= 1
                self._batch_ready[index] = ("skipped", "")
                dropped.extend([self._frame_name(request)])
        if run is not None and dropped:  # the record says that it was stopped, and which frames are missing
            run.stopped = True
            run.not_started = dropped
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
            job = self.view_model.frame_job(index, request, self._batch_run, self._batch_numbers.get(index))
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
                self._batch_run.stopped = True
                self._batch_run.not_started.extend([self._frame_name(request)])
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
            request = entry[0] if entry else None
            name = self._frame_name(request) if request is not None else f"frame {index + 1}"
            folder = self._batch_run.destination if self._batch_run is not None else None
            reason, file = failure_reason(request, error, folder)
            ref = (Path(request.path), request.frame_index) if request is not None else (Path(file or "."), 0)
            self._batch_ready[index] = ("failed", name, reason, file, self._batch_numbers.get(index), ref)
            self._batch_folder_lost()
        else:
            self._batch_ready[index] = outcome
        self._batch_fold_next()
        self._batch_fill()
        self._batch_check_done()

    def _batch_folder_lost(self) -> None:
        """A frame failed: when the folder takes no file any more (disk full, a drive that went away), every
        further frame would fail the same way after being reduced — the batch ends here and says why."""
        run = self._batch_run
        if run is None or self._batch_map_only or run.write_error or self._batch_stopping:
            return
        error = cannot_write(run.destination)
        if error is not None:
            run.write_error = error
            self.cancel_batch()

    def _batch_fold_next(self) -> None:
        """Fold the outcomes that are next in frame order (fits off the GUI thread, one at a time)."""
        run = self._batch_run
        while run is not None and not self._batch_folding and self._batch_next_fold in self._batch_ready:
            index = self._batch_next_fold
            item = self._batch_ready.pop(index)
            self._batch_next_fold += 1
            if isinstance(item, tuple):
                if item[0] == "failed":
                    run.fail(*item[1:4])
                    self.view_model.failed_batch_preview(run, item[1], item[4], item[5], item[2])
                    self._series_fit_changed()
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
        self._series_fit_changed()
        if self._batch_live and outcome.live_row is not None:
            self._series_rows.append(outcome.live_row)
        if later:
            self._batch_folding = False
            self._batch_fold_next()
            self._batch_fill()
            self._batch_check_done()

    def _batch_fold_failed(self, outcome, message: str) -> None:
        self._batch_run.fail(outcome.label, message)
        ref = (Path(outcome.metadata.get("source_file") or "."), outcome.metadata.get("frame_index", 0))
        self.view_model.failed_batch_preview(self._batch_run, outcome.label, outcome.number, ref, message)
        self._series_fit_changed()
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

__all__ = ["BatchRunMixin", "LIVE_CURVE", "POLL_MS", "REDRAW_S", "TEXT_S", "eta_text"]
