"""JobRunner application port。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .models import JobProgress, JobRequest, JobResult


ProgressObserver = Callable[[JobProgress], None]


class JobRunner(Protocol):
    def run(
        self,
        request: JobRequest,
        on_progress: ProgressObserver | None = None,
    ) -> JobResult: ...

    def cancel(self, job_id: str) -> bool: ...

    def shutdown(self) -> None: ...
