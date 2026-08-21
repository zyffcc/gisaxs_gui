"""Framework-neutral AI fitting command group."""

from __future__ import annotations

from dataclasses import replace

from ..application import CandidateGenerationRequest, CandidateJobError


class FittingAiViewModel:
    def __init__(
        self,
        owner,
        *,
        generate_candidates,
        refine_candidates,
        review_candidates,
        map_candidate_parameters,
        load_candidate_results,
    ) -> None:
        self._owner = owner
        self._generate = generate_candidates
        self._refine = refine_candidates
        self._review = review_candidates
        self._map = map_candidate_parameters
        self._load = load_candidate_results

    def run_ai_candidates(
        self,
        request: CandidateGenerationRequest,
        *,
        refine: bool = False,
        on_progress=None,
    ):
        self._set_state(
            ai_fit_status="running",
            ai_progress=0.0,
            ai_progress_message="Starting AI fitting...",
            ai_error_code=None,
            error_message=None,
        )

        def progress_update(progress):
            self._set_state(
                ai_progress=progress.fraction,
                ai_progress_message=progress.message,
                status_message=progress.message or "AI fitting running",
            )
            if on_progress is not None:
                on_progress(progress)

        use_case = self._refine if refine else self._generate
        try:
            result = use_case.execute(request, on_progress=progress_update)
        except CandidateJobError as exc:
            self._set_state(
                ai_fit_status="cancelled" if exc.code == "cancelled" else "error",
                ai_error_code=exc.code,
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self._set_state(
            ai_fit_status="ready",
            ai_progress=1.0,
            ai_progress_message="AI fitting completed",
            ai_fit_result=result,
            ai_error_code=None,
            error_message=None,
            status_message="AI fitting completed",
        )
        return result

    def cancel_ai_candidates(self) -> bool:
        return self._generate.cancel()

    def review_candidates(self, rows, constraint_options=None):
        return self._review.execute(rows, constraint_options)

    def map_candidate_parameters(self, row):
        return self._map.execute(row)

    def load_candidate_results(self, output_dir):
        try:
            rows = self._load.execute(output_dir)
        except (OSError, TypeError, ValueError) as exc:
            self._set_state(error_message=str(exc), status_message=str(exc))
            return None
        self._set_state(
            error_message=None,
            status_message=f"Loaded {len(rows)} AI candidates",
        )
        return rows

    def _set_state(self, **changes) -> None:
        self._owner.state = replace(self._owner.state, **changes)


__all__ = ["FittingAiViewModel"]
