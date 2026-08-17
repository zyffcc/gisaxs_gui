"""AI candidate generation、verification、ranking 和 refinement use cases。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.gimap.app.jobs import JobProgress, JobRunner

from .ai_models import (
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateJobError,
)
from .ports import CandidateRepository, Predictor
from ..domain import (
    CandidateParameterMapping,
    candidate_parameter_mapping,
    verify_and_rank_candidates,
)


class GenerateCandidates:
    def __init__(self, predictor: Predictor, job_runner: JobRunner):
        self._predictor = predictor
        self._job_runner = job_runner
        self._active_job_id: str | None = None
        self._lock = threading.RLock()

    def execute(
        self,
        request: CandidateGenerationRequest,
        on_progress: Callable[[JobProgress], None] | None = None,
    ) -> CandidateGenerationResult:
        job_request = self._predictor.create_job_request(request)
        with self._lock:
            if self._active_job_id is not None:
                raise CandidateJobError("already_running", "An AI fitting job is already running")
            self._active_job_id = job_request.job_id
        try:
            result = self._job_runner.run(job_request, on_progress=on_progress)
        finally:
            with self._lock:
                self._active_job_id = None
        if not result.succeeded:
            error = result.error
            raise CandidateJobError(
                error.code if error is not None else result.status,
                error.message if error is not None else f"AI fitting job {result.status}",
            )
        return self._predictor.decode_result(request, result.value)

    def cancel(self) -> bool:
        with self._lock:
            job_id = self._active_job_id
        return False if job_id is None else self._job_runner.cancel(job_id)


class RefineCandidates:
    """Full profile semantic entry; the same predictor preserves the unified pipeline."""

    def __init__(self, generation: GenerateCandidates):
        self._generation = generation

    def execute(self, request, on_progress=None) -> CandidateGenerationResult:
        if int(request.profile.get("refinement_count", 0)) <= 0:
            raise ValueError("Candidate refinement requires refinement_count > 0")
        return self._generation.execute(request, on_progress=on_progress)

    def cancel(self) -> bool:
        return self._generation.cancel()


class ReviewCandidates:
    def execute(
        self,
        rows: Sequence[Mapping[str, Any]],
        constraint_options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return verify_and_rank_candidates(rows, constraint_options)


class MapCandidateParameters:
    def execute(self, row: Mapping[str, Any]) -> CandidateParameterMapping:
        return candidate_parameter_mapping(row)


class LoadCandidateResults:
    def __init__(self, repository: CandidateRepository):
        self._repository = repository

    def execute(self, output_dir):
        return self._repository.load(output_dir)
