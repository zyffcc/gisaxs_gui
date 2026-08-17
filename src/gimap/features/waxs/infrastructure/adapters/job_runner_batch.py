"""Shared JobRunner backed WAXS batch adapter。"""

from pathlib import Path

from src.gimap.app.jobs import JobRequest

from ...application import WaxsBatchProgress, WaxsBatchResult
from ..batch_serialization import request_to_payload, result_from_payload


class JobRunnerWaxsBatchAdapter:
    def __init__(self, runner):
        self.runner = runner
        self._active_job_id = None
        self._control_file = None
        self._cancel_requested = False
        self._pause_requested = False

    def run(self, request, *, on_progress=None):
        job = JobRequest(
            handler="src.gimap.features.waxs.infrastructure.workers:process_waxs_batch_job",
            payload=request_to_payload(request),
            timeout_seconds=request.timeout_seconds,
        )
        control_file = Path(".gimap_cache/waxs_jobs") / f"{job.job_id}.control"
        control_file.parent.mkdir(parents=True, exist_ok=True)
        control_file.write_text(
            "paused" if self._pause_requested else "running",
            encoding="utf-8",
        )
        payload = dict(job.payload)
        payload["_control_file"] = str(control_file.resolve())
        job = JobRequest(
            handler=job.handler,
            payload=payload,
            timeout_seconds=job.timeout_seconds,
            job_id=job.job_id,
        )
        self._active_job_id = job.job_id
        self._control_file = control_file

        def progress(value):
            if on_progress:
                on_progress(
                    WaxsBatchProgress(
                        int(value.completed),
                        int(value.total),
                        str(value.details.get("name", value.message)),
                        str(value.details.get("status", "running")),
                    )
                )

        try:
            if self._cancel_requested:
                return WaxsBatchResult((), cancelled=True)
            result = self.runner.run(job, on_progress=progress)
        finally:
            self._active_job_id = None
            self._control_file = None
            self._cancel_requested = False
            self._pause_requested = False
            control_file.unlink(missing_ok=True)
        if not result.succeeded:
            raise RuntimeError(result.error.message if result.error else result.status)
        return result_from_payload(result.value)

    def cancel(self) -> bool:
        self._cancel_requested = True
        if self._active_job_id:
            return bool(self.runner.cancel(self._active_job_id))
        return True

    def set_paused(self, paused: bool) -> bool:
        self._pause_requested = bool(paused)
        if self._control_file is None:
            return True
        self._control_file.write_text(
            "paused" if paused else "running", encoding="utf-8"
        )
        return True
