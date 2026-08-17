"""JobRunner 的 multiprocessing adapter。"""

from __future__ import annotations

import importlib
import multiprocessing
import queue
import threading
import time
import traceback
from typing import Any

from ...app.jobs import JobError, JobProgress, JobRequest, JobResult, ProgressObserver
from ...app.jobs.models import ensure_serializable


def _resolve_handler(handler_path: str):
    module_name, function_name = handler_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    if not callable(handler):
        raise TypeError(f"Job handler is not callable: {handler_path}")
    return handler


def _worker_main(request_data, messages, cancelled) -> None:
    request = JobRequest.from_dict(request_data)

    def report(
        completed: float,
        total: float,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        progress = JobProgress(
            job_id=request.job_id,
            completed=float(completed),
            total=float(total),
            message=str(message),
            details=details or {},
        )
        payload = {"kind": "progress", "value": progress.to_dict()}
        ensure_serializable(payload, "worker progress message")
        messages.put(payload)

    try:
        handler = _resolve_handler(request.handler)
        value = handler(request.payload, report, cancelled.is_set)
        ensure_serializable(value, "worker result")
        payload = {
            "kind": "result",
            "value": JobResult(
                job_id=request.job_id,
                status="succeeded",
                value=value,
            ).to_dict(),
        }
        ensure_serializable(payload, "worker result message")
        messages.put(payload)
    except BaseException as exc:
        error = JobError(
            code="handler_error",
            message=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        messages.put(
            {
                "kind": "result",
                "value": JobResult(
                    job_id=request.job_id,
                    status="failed",
                    error=error,
                ).to_dict(),
            }
        )


class LocalProcessJobRunner:
    """每个 job 使用独立 spawned process，隔离 native runtime crash。"""

    def __init__(self, poll_interval_seconds: float = 0.02):
        self.poll_interval_seconds = max(0.005, float(poll_interval_seconds))
        self._context = multiprocessing.get_context("spawn")
        self._active: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.RLock()

    def run(
        self,
        request: JobRequest,
        on_progress: ProgressObserver | None = None,
    ) -> JobResult:
        messages = self._context.Queue()
        cancelled = self._context.Event()
        process = self._context.Process(
            target=_worker_main,
            args=(request.to_dict(), messages, cancelled),
            name=f"gimap-job-{request.job_id[:8]}",
        )
        with self._lock:
            if request.job_id in self._active:
                raise ValueError(f"Job is already running: {request.job_id}")
            self._active[request.job_id] = (process, cancelled)

        started = time.monotonic()
        result: JobResult | None = None
        process.start()
        try:
            while result is None:
                elapsed = time.monotonic() - started
                if cancelled.is_set():
                    result = self._stop_result(
                        process,
                        request.job_id,
                        "cancelled",
                        "cancelled",
                        "Job was cancelled.",
                        elapsed,
                    )
                    break
                if request.timeout_seconds is not None and elapsed >= request.timeout_seconds:
                    cancelled.set()
                    result = self._stop_result(
                        process,
                        request.job_id,
                        "timed_out",
                        "timeout",
                        f"Job exceeded timeout of {request.timeout_seconds:g} seconds.",
                        elapsed,
                    )
                    break
                try:
                    message = messages.get(timeout=self.poll_interval_seconds)
                except queue.Empty:
                    if not process.is_alive():
                        result = self._drain_or_crash(
                            process,
                            messages,
                            request.job_id,
                            elapsed,
                            on_progress,
                        )
                    continue
                result = self._consume_message(message, on_progress)

            return JobResult(
                job_id=result.job_id,
                status=result.status,
                value=result.value,
                error=result.error,
                elapsed_seconds=time.monotonic() - started,
            )
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            with self._lock:
                self._active.pop(request.job_id, None)
            messages.close()
            messages.join_thread()

    def _consume_message(
        self,
        message: dict[str, Any],
        on_progress: ProgressObserver | None,
    ) -> JobResult | None:
        ensure_serializable(message, "worker message")
        kind = message.get("kind")
        value = message.get("value")
        if kind == "progress":
            progress = JobProgress.from_dict(value)
            if on_progress is not None:
                try:
                    on_progress(progress)
                except Exception:
                    pass
            return None
        if kind == "result":
            return JobResult.from_dict(value)
        raise ValueError(f"Unknown worker message kind: {kind!r}")

    def _drain_or_crash(
        self,
        process,
        messages,
        job_id: str,
        elapsed: float,
        on_progress: ProgressObserver | None,
    ) -> JobResult:
        first = True
        while True:
            try:
                message = (
                    messages.get(timeout=0.1)
                    if first
                    else messages.get_nowait()
                )
            except queue.Empty:
                break
            first = False
            consumed = self._consume_message(message, on_progress)
            if consumed is not None:
                return consumed
        return JobResult(
            job_id=job_id,
            status="failed",
            error=JobError(
                code="worker_crash",
                message=f"Worker exited unexpectedly with code {process.exitcode}.",
            ),
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _stop_result(
        process,
        job_id: str,
        status,
        code: str,
        message: str,
        elapsed: float,
    ) -> JobResult:
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
        return JobResult(
            job_id=job_id,
            status=status,
            error=JobError(code=code, message=message),
            elapsed_seconds=elapsed,
        )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            active = self._active.get(job_id)
            if active is None:
                return False
            _process, cancelled = active
            cancelled.set()
            return True

    def shutdown(self) -> None:
        with self._lock:
            active = list(self._active.items())
        for job_id, (_process, cancelled) in active:
            cancelled.set()
            self.cancel(job_id)
