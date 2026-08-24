"""XRR extraction runner implemented by the shared process JobRunner."""

from __future__ import annotations

from src.gimap.app.jobs import JobRequest

from ...application import XrrExtractionProgress, XrrExtractionResult
from ..serialization import (
    decode_preview,
    point_from_payload,
    request_to_payload,
    result_from_payload,
)


class JobRunnerXrrExtractionAdapter:
    def __init__(self, runner):
        self._runner = runner
        self._active_job_id: str | None = None

    def run(self, request, *, on_progress=None) -> XrrExtractionResult:
        job = JobRequest(
            handler="src.gimap.features.xrr.infrastructure.workers:process_xrr_series_job",
            payload=request_to_payload(request),
        )
        self._active_job_id = job.job_id

        def progress(value) -> None:
            if on_progress is None:
                return
            preview, full_shape = decode_preview(value.details["preview"])
            on_progress(
                XrrExtractionProgress(
                    completed=int(value.completed),
                    total=int(value.total),
                    point=point_from_payload(value.details["point"]),
                    preview=preview,
                    preview_shape=full_shape,
                )
            )

        try:
            result = self._runner.run(job, on_progress=progress)
        finally:
            self._active_job_id = None
        if result.status == "cancelled":
            return XrrExtractionResult(())
        if not result.succeeded:
            raise RuntimeError(result.error.message if result.error else result.status)
        return result_from_payload(result.value)

    def cancel(self) -> bool:
        if self._active_job_id is None:
            return False
        return bool(self._runner.cancel(self._active_job_id))


__all__ = ["JobRunnerXrrExtractionAdapter"]
