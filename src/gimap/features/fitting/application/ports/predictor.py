"""AI fitting neural pipeline 的 Predictor port。"""

from __future__ import annotations

from typing import Any, Protocol

from src.gimap.app.jobs import JobRequest

from ..ai_models import CandidateGenerationRequest, CandidateGenerationResult


class Predictor(Protocol):
    def create_job_request(self, request: CandidateGenerationRequest) -> JobRequest: ...

    def decode_result(
        self,
        request: CandidateGenerationRequest,
        value: Any,
    ) -> CandidateGenerationResult: ...
