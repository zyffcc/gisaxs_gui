"""The Predictor/JobRunner adapter of 1D Predict: one curve through the 1D Predict job (``workflow_v5``) with the
options of the request (Predict parameters by default)."""

from __future__ import annotations

from pathlib import Path


from src.gimap.app.jobs import JobRequest

from ...application.ai_models import CandidateGenerationRequest, CandidateGenerationResult


class AiPipelinePredictor:
    def create_job_request(self, request: CandidateGenerationRequest) -> JobRequest:
        from .workflow_v5 import validate_options
        options = validate_options(request.constraints.get("workflow_v5", {}))
        output = Path(request.output_dir).resolve()
        return JobRequest(
            handler="src.gimap.features.fitting.infrastructure.adapters.workflow_v5:run_workflow_job",
            payload=dict(output_dir=str(output), q=request.q.tolist(), intensity=request.intensity.tolist(),
                         sigma=request.sigma.tolist(), options=options,
                         observation_metadata=request.constraints.get("observation_metadata", {}),
                         sigma_estimated=bool(request.constraints.get("sigma_estimated", False))),
            timeout_seconds=3600,
        )

    def decode_result(self, request, value) -> CandidateGenerationResult:
        if not isinstance(value, dict):
            raise ValueError("AI fitting worker returned an invalid result")
        summary = value.get("summary") or {}
        rows = value.get("candidates") or []
        return CandidateGenerationResult(
            output_dir=Path(value.get("output_dir") or request.output_dir),
            profile_name=str(summary.get("profile", request.profile.get("name", ""))),
            runtime_seconds=float(summary.get("runtime_seconds", 0.0)),
            configured_candidates=int(
                summary.get("configured_candidates", request.profile.get("candidate_count", 0))
            ),
            candidates=tuple(dict(row) for row in rows),
            best_log_rmse=(
                None if summary.get("best_log_rmse") is None else float(summary["best_log_rmse"])
            ),
            exit_code=int(summary.get("exit_code", 0)),
        )
