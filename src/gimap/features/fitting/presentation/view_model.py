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
    LoadDetectorSettings,
    LoadCurveRequest,
    LoadScatteringFile,
    LoadScatteringFileRequest,
    LoadCandidateResults,
    RunManualFit,
    GenerateCandidates,
    MapCandidateParameters,
    RefineCandidates,
    ReviewCandidates,
    SaveDetectorSettings,
    InSituWorkflowCoordinator,
)
from ..application.models import ScatteringFileData
from ..domain import ManualFitRequest
from .insitu_view_model import FittingInSituViewModel
from .state import FittingState
from .storage_view_model import FittingStorageViewModel


class FittingViewModel:
    def __init__(
        self,
        *,
        context: AppContext,
        load_scattering_file: LoadScatteringFile,
        inspect_scattering_sequence,
        load_curve: LoadCurve,
        export_fit_result: ExportFitResult,
        run_manual_fit: RunManualFit,
        generate_candidates: GenerateCandidates,
        refine_candidates: RefineCandidates,
        review_candidates: ReviewCandidates,
        map_candidate_parameters: MapCandidateParameters,
        load_candidate_results: LoadCandidateResults,
        insitu_workflow: InSituWorkflowCoordinator,
        load_detector_settings: LoadDetectorSettings,
        save_detector_settings: SaveDetectorSettings,
        scattering_loader_factory=None,
        remote_file_cache=None,
        insitu_records=None,
        parameter_files=None,
        ai_artifacts=None,
        save_fitting_log=None,
        check_dependency=None,
        scientific=None, model_parameters=None, ai_catalog=None,
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
        self._load_detector_settings = load_detector_settings
        self._save_detector_settings = save_detector_settings
        self.state = FittingState(insitu_workflow=insitu_workflow.state)
        self.storage = FittingStorageViewModel(
            load_scattering_file=load_scattering_file,
            inspect_scattering_sequence=inspect_scattering_sequence,
            scattering_loader_factory=scattering_loader_factory,
            remote_file_cache=remote_file_cache,
            insitu_records=insitu_records,
            parameter_files=parameter_files,
            ai_artifacts=ai_artifacts,
            save_fitting_log=save_fitting_log,
            check_dependency=check_dependency,
            model_parameters=model_parameters,
            ai_catalog=ai_catalog,
        )
        self.insitu = FittingInSituViewModel(
            insitu_workflow, self._sync_insitu_state
        )
        self.science = scientific

    def __getattr__(self, name):
        """Keep migrated legacy callers working while they adopt command groups."""

        for group_name in ("storage", "insitu", "science"):
            group = self.__dict__.get(group_name)
            if group is not None and hasattr(group, name):
                return getattr(group, name)
        raise AttributeError(name)

    def get_setting(self, section: str, key: str, default=None):
        return self.context.settings.get(section, key, default)

    def set_setting(self, section: str, key: str, value) -> None:
        self.context.settings.set(section, key, value)

    def save_settings(self) -> None:
        self.context.settings.save()

    def load_detector_settings(self):
        return self._load_detector_settings.execute()

    def save_detector_settings(self, settings) -> None:
        self._save_detector_settings.execute(settings)

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

    def _sync_insitu_state(self, workflow_state, message: str) -> None:
        self.state = replace(
            self.state, insitu_workflow=workflow_state, status_message=message
        )
