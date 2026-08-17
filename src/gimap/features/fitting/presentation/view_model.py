"""Fitting 的 framework-neutral ViewModel。"""

from __future__ import annotations

from dataclasses import replace

from src.gimap.app import AppContext

from ..application import (
    CandidateGenerationRequest,
    CandidateJobError,
    ExportFitResult,
    ExportFitResultRequest,
    LoadCurve,
    LoadCurveRequest,
    LoadScatteringFile,
    LoadScatteringFileRequest,
    LoadCandidateResults,
    RunManualFit,
    GenerateCandidates,
    MapCandidateParameters,
    RefineCandidates,
    ReviewCandidates,
    InSituWorkflowCoordinator,
)
from ..application.models import ScatteringFileData
from ..domain import ManualFitRequest
from .state import FittingState


class FittingViewModel:
    def __init__(
        self,
        *,
        context: AppContext,
        load_scattering_file: LoadScatteringFile,
        load_curve: LoadCurve,
        export_fit_result: ExportFitResult,
        run_manual_fit: RunManualFit,
        generate_candidates: GenerateCandidates,
        refine_candidates: RefineCandidates,
        review_candidates: ReviewCandidates,
        map_candidate_parameters: MapCandidateParameters,
        load_candidate_results: LoadCandidateResults,
        insitu_workflow: InSituWorkflowCoordinator,
    ):
        self.context = context
        self._load_scattering_file = load_scattering_file
        self._load_curve = load_curve
        self._export_fit_result = export_fit_result
        self._run_manual_fit = run_manual_fit
        self._generate_candidates = generate_candidates
        self._refine_candidates = refine_candidates
        self._review_candidates = review_candidates
        self._map_candidate_parameters = map_candidate_parameters
        self._load_candidate_results = load_candidate_results
        self._insitu_workflow = insitu_workflow
        self.state = FittingState(insitu_workflow=insitu_workflow.state)

    def load_scattering(self, request: LoadScatteringFileRequest):
        self.state = replace(
            self.state,
            image_status="loading",
            error_message=None,
            status_message=f"Loading {request.path.name}...",
        )
        outcome = self._load_scattering_file.execute(request)
        if outcome.error is not None:
            self.state = replace(
                self.state,
                image_status="error",
                error_message=outcome.error.message,
                status_message=outcome.error.message,
            )
            return outcome
        self.accept_loaded_image(outcome.value)
        return outcome

    def begin_image_load(self, file_name: str) -> None:
        self.state = replace(
            self.state,
            image_status="loading",
            error_message=None,
            status_message=f"Loading {file_name}...",
        )

    def accept_loaded_image(self, image: ScatteringFileData) -> None:
        self.state = replace(
            self.state,
            image_status="ready",
            current_image=image,
            error_message=None,
            status_message=f"Loaded {image.source_path.name}",
        )

    def fail_image_load(self, message: str) -> None:
        self.state = replace(
            self.state,
            image_status="error",
            error_message=str(message),
            status_message=str(message),
        )

    def load_curve(self, request: LoadCurveRequest):
        self.state = replace(
            self.state,
            curve_status="loading",
            error_message=None,
            status_message=f"Loading {request.path.name}...",
        )
        outcome = self._load_curve.execute(request)
        if outcome.error is not None:
            self.state = replace(
                self.state,
                curve_status="error",
                error_message=outcome.error.message,
                status_message=outcome.error.message,
            )
            return outcome
        self.state = replace(
            self.state,
            curve_status="ready",
            current_curve=outcome.value,
            error_message=None,
            status_message=f"Loaded {request.path.name}",
        )
        return outcome

    def export_fit_result(self, request: ExportFitResultRequest):
        outcome = self._export_fit_result.execute(request)
        if outcome.error is not None:
            self.state = replace(
                self.state,
                error_message=outcome.error.message,
                status_message=outcome.error.message,
            )
        else:
            self.state = replace(
                self.state,
                error_message=None,
                status_message=f"Exported {outcome.value.path.name}",
            )
        return outcome

    def run_manual_fit(self, request: ManualFitRequest):
        self.state = replace(
            self.state,
            manual_fit_status="running",
            error_message=None,
            status_message="Running manual fitting...",
        )
        try:
            result = self._run_manual_fit.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state,
                manual_fit_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            manual_fit_status="ready",
            manual_fit_result=result,
            error_message=None,
            status_message="Manual fitting completed",
        )
        return result

    def run_ai_candidates(self, request: CandidateGenerationRequest, *, refine=False, on_progress=None):
        self.state = replace(
            self.state,
            ai_fit_status="running",
            ai_progress=0.0,
            ai_progress_message="Starting AI fitting...",
            ai_error_code=None,
            error_message=None,
        )

        def progress_update(progress):
            self.state = replace(
                self.state,
                ai_progress=progress.fraction,
                ai_progress_message=progress.message,
                status_message=progress.message or "AI fitting running",
            )
            if on_progress is not None:
                on_progress(progress)

        use_case = self._refine_candidates if refine else self._generate_candidates
        try:
            result = use_case.execute(request, on_progress=progress_update)
        except CandidateJobError as exc:
            status = "cancelled" if exc.code == "cancelled" else "error"
            self.state = replace(
                self.state,
                ai_fit_status=status,
                ai_error_code=exc.code,
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
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
        return self._generate_candidates.cancel()

    def review_candidates(self, rows, constraint_options=None):
        return self._review_candidates.execute(rows, constraint_options)

    def map_candidate_parameters(self, row):
        return self._map_candidate_parameters.execute(row)

    def load_candidate_results(self, output_dir):
        try:
            rows = self._load_candidate_results.execute(output_dir)
        except (OSError, TypeError, ValueError) as exc:
            self.state = replace(
                self.state,
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            error_message=None,
            status_message=f"Loaded {len(rows)} AI candidates",
        )
        return rows

    def start_insitu_workflow(self, paths, *, continue_on_error=True) -> None:
        self._insitu_workflow.start(paths, continue_on_error=continue_on_error)
        self._sync_insitu_state("In-situ workflow started")

    def enqueue_insitu_files(self, paths) -> None:
        self._insitu_workflow.enqueue(paths)
        self._sync_insitu_state("In-situ files queued")

    def begin_next_insitu_file(self, batch_size=1):
        record = self._insitu_workflow.begin_next(batch_size)
        self._sync_insitu_state("Processing in-situ file")
        return record

    def complete_insitu_file(self, values=None):
        record = self._insitu_workflow.complete_current(values)
        self._sync_insitu_state("In-situ file completed")
        return record

    def fail_insitu_file(self, error_message, values=None):
        record = self._insitu_workflow.fail_current(error_message, values)
        self._sync_insitu_state("In-situ file failed")
        return record

    def pause_insitu_workflow(self) -> None:
        self._insitu_workflow.pause()
        self._sync_insitu_state("In-situ workflow paused")

    def resume_insitu_workflow(self) -> None:
        self._insitu_workflow.resume()
        self._sync_insitu_state("In-situ workflow resumed")

    def cancel_insitu_workflow(self) -> None:
        self._insitu_workflow.cancel()
        self._sync_insitu_state("In-situ workflow cancelled")

    def snapshot_insitu_workflow(self):
        return self._insitu_workflow.snapshot()

    def restore_insitu_workflow(self, snapshot) -> None:
        self._insitu_workflow.restore(snapshot)
        self._sync_insitu_state("In-situ workflow restored")

    def _sync_insitu_state(self, message: str) -> None:
        self.state = replace(
            self.state,
            insitu_workflow=self._insitu_workflow.state,
            status_message=message,
        )
