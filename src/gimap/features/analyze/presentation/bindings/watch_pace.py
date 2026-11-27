"""The new frames a folder watch reduces at the same time: a few, not as many as the window has threads; large
frames in worker processes.

A watch hands every new frame to the page's tasks (``batch_watch._watch_dispatch_requests``). A detector writes ten
frames a second and more. Started all at once, the frames of a 2-megapixel detector, some 100 MB each while being
reduced, took all the memory the computer had left (frames failed: "Unable to allocate …"), and the threads, taking
turns at the interpreter with the window, kept it from answering for seconds. At most ``watch_frames_at_once`` frames
(2 to 4 by the cores; fewer when free memory is short) are reduced at once; the others wait in line, in order, and
start as frames finish. Stopping the watch clears the line of the frames for the map; frames waiting to be exported
stay.

The frame on screen is not in the line. While a watch is on, though, its analysis waits for the one under way
(``_analysis_waits``): the watch shows the newest frame at every look that finds new files, and a look came every
second or less while an analysis of a 2-megapixel frame took 1.5–3 s of a busy computer. Each analysis replaced the
one before, which went on to its end unseen; several ran at once, and with every new one the result shown was put
off again, while a detector wrote. Now the newest frame asked for meanwhile is analysed next, once the one under way
is shown: a frame every analysis, and one at a time.

A frame of at least ``WORKER_PIXELS`` pixels that only gives its row of the map is reduced in a worker process of the
watch (``view_model.open_workers``, as Batch Export's: below-normal priority, one core each), when the computer has
them: in the application its threads took turns at the interpreter with the window, the look at the folder and the
map, and a long run slowed down as these grew (a frame of a Pilatus 2M: 0.13 s in a process, several times that in a
thread late in a run). A row the worker gives is the row ``series_row`` gives here. A frame the worker gives no row
(no curve for the map) is reduced here again, which says why; if the workers stop (out of memory …), the watch goes
on with threads.
"""

from __future__ import annotations

import traceback
from collections import deque
from concurrent.futures.process import BrokenProcessPool

from PyQt5.QtCore import QTimer

from ...application.batch_job import watch_frames_at_once

LOOK_AGAIN_MS = 250
"""While frames wait in line: how often the line is looked at besides when a frame finishes (a frame whose task was
superseded finishes without saying so)."""
POLL_MS = 50
"""While worker processes reduce frames: how often their frames are looked for."""
UNKNOWN_PIXELS = 4_000_000
"""The frame size assumed while no frame is on screen."""
WORKER_PIXELS = 1_000_000
"""Frames of at least this many pixels go to worker processes (a smaller frame is reduced quickly enough here)."""
ANALYSIS_CHECK_MS = 200
"""While an analysis of the frame on screen waits: how often it looks whether the one under way ended unseen (its
frame was removed from the list: no result comes)."""


class WatchPaceMixin:
    """Needs ``tasks`` (``TaskRunner``) and ``view_model`` (``resources``, ``state``, ``open_workers``)."""

    def _watch_pace_reset(self) -> None:
        """The frames waiting for a live map leave the line (the watch stopped or began again), and the workers of
        the watch close; frames waiting to be exported stay, as they did when every frame was started at once. The
        frames at once are chosen again at the next frame (for the free memory then)."""
        line = getattr(self, "_watch_line", None) or ()
        self._watch_line = deque(item for item in line if item[0].startswith("export:"))
        if getattr(self, "_watch_out", None) is None:
            self._watch_out = set()  # (frames being reduced count until they finish, also those of a watch before)
        self._watch_at_once = None
        self._watch_pace_close()
        timer = getattr(self, "_watch_pace_timer", None)
        if timer is not None and not self._watch_line:
            timer.stop()
        self._watch_pump()

    def _watch_pace_close(self) -> None:
        """Close the worker processes of the watch; the frames they reduce are not waited for."""
        workers = getattr(self, "_watch_workers", None)
        self._watch_workers, self._watch_workers_tried, self._watch_futures = None, False, []
        if workers is not None:
            workers.shutdown(wait=False)
        for name in ("_watch_poll_timer", "_analysis_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()

    def _watch_submit(self, job: str, fn, *, on_done, on_error, frame_job=None) -> None:
        """Reduce a new frame (``tasks.submit``; ``frame_job``: in a worker process when the watch has them) when fewer
        than ``_watch_limit()`` frames are being reduced, else when its turn comes."""
        if getattr(self, "_watch_line", None) is None:
            self._watch_pace_reset()
        self._watch_line.append((job, fn, on_done, on_error, frame_job))
        self._watch_pump()

    def _watch_pixels(self) -> int:
        analysis = self.view_model.state.analysis
        data = getattr(analysis, "raw_data", None)
        if data is None:
            data = getattr(analysis, "data", None)
        return int(getattr(data, "size", 0) or UNKNOWN_PIXELS)

    def _watch_limit(self) -> int:
        if getattr(self, "_watch_at_once", None) is None:
            cores, free = self.view_model.resources()
            self._watch_at_once = watch_frames_at_once(cores=cores, free_bytes=free, frame_pixels=self._watch_pixels())
        return self._watch_at_once

    def _watch_pool(self):
        """The worker processes of the watch, opened at its first large frame; ``None``: frames are reduced here."""
        workers = getattr(self, "_watch_workers", None)
        if workers is not None or getattr(self, "_watch_workers_tried", False):
            return workers
        self._watch_workers_tried = True
        if self._watch_pixels() < WORKER_PIXELS or self._watch_limit() < 2:
            return None
        try:
            self._watch_workers = self.view_model.open_workers(self._watch_limit())
        except (OSError, RuntimeError, ValueError):
            self._watch_workers = None
        return self._watch_workers

    def _watch_lose_workers(self) -> None:
        """The workers stopped: the watch goes on here (the frames they had are reduced again, ``_watch_poll``)."""
        workers = getattr(self, "_watch_workers", None)
        self._watch_workers, self._watch_workers_tried = None, True
        if workers is not None:
            workers.shutdown(wait=False)

    def _watch_pump(self) -> None:
        """Start frames from the line until ``_watch_limit()`` are being reduced."""
        line = getattr(self, "_watch_line", None)
        if not line:
            return
        futures = getattr(self, "_watch_futures", None)
        if futures is None:
            futures = self._watch_futures = []
        out = {ticket for ticket in self._watch_out if self.tasks.is_pending(ticket)}
        limit = self._watch_limit()
        while line and len(out) + len(futures) < limit:
            job, fn, on_done, on_error, frame_job = line.popleft()
            workers = self._watch_pool() if frame_job is not None else None
            if workers is not None:
                try:
                    futures.append((workers.submit(frame_job), (job, fn, on_done, on_error)))
                    continue
                except (RuntimeError, OSError, BrokenProcessPool):
                    self._watch_lose_workers()
            out.add(self.tasks.submit(job, fn, on_done=self._watch_then(on_done), on_error=self._watch_then(on_error)))
        self._watch_out = out
        if futures:
            self._watch_timer_of("_watch_poll_timer", POLL_MS, self._watch_poll).start()
        if line:
            self._watch_timer_of("_watch_pace_timer", LOOK_AGAIN_MS, self._watch_pump).start()

    def _watch_timer_of(self, name: str, interval_ms: int, slot) -> QTimer:
        timer = getattr(self, name, None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(interval_ms)
            timer.timeout.connect(slot)
            setattr(self, name, timer)
        return timer

    def _watch_poll(self) -> None:
        """Deliver the frames the worker processes finished, then start the next ones."""
        futures = getattr(self, "_watch_futures", None) or []
        done = [item for item in futures if item[0].done()]
        self._watch_futures = [item for item in futures if not item[0].done()]
        again = []
        for future, (job, fn, on_done, on_error) in done:
            try:
                outcome = future.result()
            except BrokenProcessPool:
                self._watch_lose_workers()
                again.append((job, fn, on_done, on_error, None))
                continue
            except Exception as exc:  # noqa: BLE001 - what reducing the frame raised, said as the thread would
                on_error(str(exc) or type(exc).__name__, "".join(traceback.format_exception(exc)))
                continue
            if outcome.live_row is None:  # no curve for the map: reduced here, which says why
                again.append((job, fn, on_done, on_error, None))
            else:
                on_done(outcome)
        if again:
            self._watch_line.extendleft(reversed(again))
        self._watch_pump()
        if self._watch_futures:
            self._watch_timer_of("_watch_poll_timer", POLL_MS, self._watch_poll).start()

    def _watch_then(self, callback):
        """``callback``, then the next frame of the line."""
        def call(*values):
            try:
                callback(*values)
            finally:
                self._watch_pump()
        return call

    def watch_frames_waiting(self) -> int:
        """New frames waiting in line to be reduced."""
        return len(getattr(self, "_watch_line", ()) or ())

    # -- the frame on screen ---------------------------------------------------------------------------------------

    def _analysis_waits(self) -> bool:
        """``True``: the analysis of the frame on screen asked for now waits for the one under way (a watch is on);
        the newest asked for is made once that one is shown (``_analysis_next``)."""
        if not self.view_model.watch.active:
            return False
        ticket = getattr(self, "_analysis_ticket", None)
        if ticket is None or not self.tasks.is_pending(ticket):
            return False
        self._analysis_wanted = True
        self._watch_timer_of("_analysis_timer", ANALYSIS_CHECK_MS, self._analysis_next).start()
        return True

    def _analysis_done(self, analysis) -> None:
        self.show_analysis(analysis)
        self._analysis_next()

    def _analysis_error(self, message: str, details: str) -> None:
        self._analysis_failed(message, details)
        self._analysis_next()

    def _analysis_next(self) -> None:
        """The analysis that waited, of the frame asked for last, once the one before is done."""
        if not getattr(self, "_analysis_wanted", False):
            return
        ticket = getattr(self, "_analysis_ticket", None)
        if ticket is not None and self.tasks.is_pending(ticket):
            self._watch_timer_of("_analysis_timer", ANALYSIS_CHECK_MS, self._analysis_next).start()
            return
        self._analysis_wanted = False
        self.run_analysis()


__all__ = ["ANALYSIS_CHECK_MS", "LOOK_AGAIN_MS", "POLL_MS", "WORKER_PIXELS", "WatchPaceMixin"]
