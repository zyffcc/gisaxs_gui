"""JobRunner that keeps one worker process from job to job, so what a handler loaded stays loaded.

``LocalProcessJobRunner`` starts a process for every job: isolated, but a TensorFlow model is imported
and loaded again for every frame (about 10 s, while the model itself answers in milliseconds). This
runner starts one spawned process at the first job and hands it the next jobs in turn; a handler may
keep what it loaded in its own module (``integrations/tensorflow/worker.py`` keeps the last models).

The process stays as isolated as before: a native crash ends it, not the GUI. A job that is cancelled
or runs past its timeout ends the process (a running model cannot be interrupted), and so does a crash;
the next job starts a fresh one. Jobs run one at a time, in the order they ask. After ``idle_seconds``
without a job the process ends, so a model nobody uses does not hold its memory.
"""

from __future__ import annotations

import queue
import threading
import time

from ...app.jobs import JobError, JobRequest, JobResult, ProgressObserver
from .light_main import light_main
from .local_process import LocalProcessJobRunner, _worker_main

IDLE_SECONDS = 900.0
"""A kept process with no job for this long ends (a TensorFlow model holds about 1 GB)."""


def _kept_main(requests, messages, cancelled) -> None:
    """The kept process: run each request with the per-job worker's code until ``None`` arrives."""
    while True:
        request_data = requests.get()
        if request_data is None:
            return
        cancelled.clear()
        _worker_main(request_data, messages, cancelled)


class _Kept:
    """One kept process with its queues (replaced as a whole when the process ends)."""

    def __init__(self, context):
        self.requests = context.Queue()
        self.messages = context.Queue()
        self.cancelled = context.Event()
        self.process = context.Process(
            target=_kept_main,
            args=(self.requests, self.messages, self.cancelled),
            name="gimap-kept-worker",
            daemon=True,
        )

    def end(self, *, politely: bool) -> None:
        if politely and self.process.is_alive():
            try:
                self.requests.put(None)
                self.process.join(timeout=1.0)
            except (OSError, ValueError):
                pass
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1.0)
        for channel in (self.requests, self.messages):
            channel.cancel_join_thread()  # never wait on data nobody will read
            channel.close()


class KeptProcessJobRunner(LocalProcessJobRunner):
    """``JobRunner`` whose worker process lasts between jobs (message handling as the per-job runner)."""

    def __init__(self, poll_interval_seconds: float = 0.01, idle_seconds: float = IDLE_SECONDS):
        super().__init__(poll_interval_seconds)
        self.idle_seconds = float(idle_seconds)
        self._turn = threading.Lock()  # one job at a time
        self._state = threading.RLock()  # the kept process, the job running, the cancelled ids
        self._kept: _Kept | None = None
        self._running: str | None = None
        self._cancel_running = threading.Event()
        self._cancel_waiting: set[str] = set()
        self._idle_timer: threading.Timer | None = None

    @property
    def alive(self) -> bool:
        """A kept process is running (it holds what its handlers loaded)."""
        with self._state:
            return self._kept is not None and self._kept.process.is_alive()

    def run(self, request: JobRequest, on_progress: ProgressObserver | None = None) -> JobResult:
        asked = time.monotonic()
        with self._turn:
            with self._state:
                self._stop_idle_timer()
                if request.job_id in self._cancel_waiting:
                    self._cancel_waiting.discard(request.job_id)
                    return self._stopped(request.job_id, "cancelled", "cancelled", "Job was cancelled.", asked)
                self._running = request.job_id
                self._cancel_running.clear()
                kept = self._process()
            try:
                result = self._wait(kept, request, on_progress)
            finally:
                with self._state:
                    self._running = None
                    self._start_idle_timer()
        return JobResult(
            job_id=result.job_id,
            status=result.status,
            value=result.value,
            error=result.error,
            elapsed_seconds=time.monotonic() - asked,
        )

    def _process(self) -> _Kept:
        if self._kept is not None and self._kept.process.is_alive():
            return self._kept
        if self._kept is not None:
            self._kept.end(politely=False)
        kept = _Kept(self._context)
        with light_main():
            kept.process.start()
        self._kept = kept
        return kept

    def _wait(self, kept: _Kept, request: JobRequest, on_progress: ProgressObserver | None) -> JobResult:
        kept.requests.put(request.to_dict())
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if self._cancel_running.is_set():
                self._end(kept)
                return self._stopped(request.job_id, "cancelled", "cancelled", "Job was cancelled.", started)
            if request.timeout_seconds is not None and elapsed >= request.timeout_seconds:
                self._end(kept)
                return self._stopped(
                    request.job_id, "timed_out", "timeout",
                    f"Job exceeded timeout of {request.timeout_seconds:g} seconds.", started,
                )
            try:
                message = kept.messages.get(timeout=self.poll_interval_seconds)
            except queue.Empty:
                if not kept.process.is_alive():
                    result = self._drain_or_crash(kept.process, kept.messages, request.job_id, elapsed, on_progress)
                    self._end(kept)
                    return result
                continue
            except (OSError, ValueError, EOFError):  # the queue went with the process (shut down meanwhile)
                self._end(kept)
                return self._stopped(request.job_id, "cancelled", "cancelled", "Job was cancelled.", started)
            result = self._consume_message(message, on_progress)
            if result is not None and result.job_id == request.job_id:
                return result

    def _end(self, kept: _Kept, *, politely: bool = False) -> None:
        with self._state:
            if self._kept is kept:
                self._kept = None
        kept.end(politely=politely)

    @staticmethod
    def _stopped(job_id: str, status, code: str, message: str, since: float) -> JobResult:
        return JobResult(
            job_id=job_id,
            status=status,
            error=JobError(code=code, message=message),
            elapsed_seconds=time.monotonic() - since,
        )

    def cancel(self, job_id: str) -> bool:
        """Cancel a job: the running one ends the kept process; one still waiting its turn never starts."""
        with self._state:
            if self._running == job_id:
                self._cancel_running.set()
                return True
            if self._turn.locked():
                self._cancel_waiting.add(job_id)
                return True
            return False

    def shutdown(self) -> None:
        """Cancel the running job and end the kept process (quitting)."""
        with self._state:
            self._stop_idle_timer()
            self._cancel_waiting.clear()
            if self._running is not None:
                self._cancel_running.set()  # its run ends the process within one poll
                return
            kept, self._kept = self._kept, None
        if kept is not None:
            kept.end(politely=True)

    def _start_idle_timer(self) -> None:
        if self.idle_seconds <= 0 or self._kept is None:
            return
        timer = threading.Timer(self.idle_seconds, self._end_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _stop_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _end_idle(self) -> None:
        if not self._turn.acquire(blocking=False):
            return  # a job is running or waiting: it restarts the timer when done
        try:
            with self._state:
                kept, self._kept = self._kept, None
                self._idle_timer = None
            if kept is not None:
                kept.end(politely=True)
        finally:
            self._turn.release()


__all__ = ["IDLE_SECONDS", "KeptProcessJobRunner"]
