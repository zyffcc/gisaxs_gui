"""Fitting 的 framework-neutral ViewModel。"""
from __future__ import annotations

from dataclasses import replace
from src.gimap.app import AppContext

from ..application import (
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
    CreateInSituRecipe,
    ReviseInSituRecipe,
)
from ..application.models import ScatteringFileData
from ..application import ManualFitRequest
from .insitu_view_model import FittingInSituViewModel
from .ai_view_model import FittingAiViewModel
from .state import CurveViewState, DetectorDisplayState, FittingState
from .workflow_state import (
    begin_workflow_step,
    complete_workflow_step,
    fail_workflow_step,
)
from .storage_view_model import FittingStorageViewModel
from .workflow_view_model import FittingWorkflowViewModelMixin


class FittingViewModel(FittingWorkflowViewModelMixin):
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
        create_insitu_recipe: CreateInSituRecipe,
        revise_insitu_recipe: ReviseInSituRecipe,
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
            insitu_workflow,
            create_insitu_recipe,
            revise_insitu_recipe,
            self._sync_insitu_state,
            self._sync_insitu_recipe,
        )
        self.ai = FittingAiViewModel(
            self,
            generate_candidates=generate_candidates,
            refine_candidates=refine_candidates,
            review_candidates=review_candidates,
            map_candidate_parameters=map_candidate_parameters,
            load_candidate_results=load_candidate_results,
        )
        self.science = scientific

    def __getattr__(self, name):
        """Keep migrated legacy callers working while they adopt command groups."""

        for group_name in ("storage", "insitu", "ai", "science"):
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

    def update_curve_view(self, state: CurveViewState) -> None:
        self.state = replace(self.state, curve_view=state)

    def update_detector_display(self, state: DetectorDisplayState) -> None:
        self.state = replace(self.state, detector_display=state)

    def load_detector_settings(self):
        return self._load_detector_settings.execute()

    def save_detector_settings(self, settings) -> None:
        self._save_detector_settings.execute(settings)

    def load_scattering(self, request: LoadScatteringFileRequest):
        workflow = begin_workflow_step(
            self.state.workflow, "import", f"Loading {request.path.name}"
        )
        self.state = replace(
            self.state,
            image_status="loading",
            error_message=None,
            status_message=f"Loading {request.path.name}...",
            workflow=workflow,
        )
        outcome = self._load_scattering_file.execute(request)
        if outcome.error is not None:
            self.state = replace(
                self.state,
                image_status="error",
                error_message=outcome.error.message,
                status_message=outcome.error.message,
                workflow=fail_workflow_step(
                    self.state.workflow, "import", outcome.error.message
                ),
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
            workflow=begin_workflow_step(
                self.state.workflow, "import", f"Loading {file_name}"
            ),
        )

    def accept_loaded_image(self, image: ScatteringFileData) -> None:
        cut_status = "stale" if self.state.cut_status == "ready" else self.state.cut_status
        self.state = replace(
            self.state,
            image_status="ready",
            current_image=image,
            error_message=None,
            status_message=f"Loaded {image.source_path.name}",
            workflow=complete_workflow_step(
                self.state.workflow,
                "import",
                f"Loaded {image.source_path.name}",
                preserve_completed=("setup",),
            ),
            cut_status=cut_status,
        )

    def fail_image_load(self, message: str) -> None:
        self.state = replace(
            self.state,
            image_status="error",
            error_message=str(message),
            status_message=str(message),
            workflow=fail_workflow_step(self.state.workflow, "import", str(message)),
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
            status_message="Calculating the current model...",
            workflow=begin_workflow_step(
                self.state.workflow, "fit", "Calculating the current model"
            ),
        )
        try:
            result = self._run_manual_fit.execute(request)
        except Exception as exc:
            self.state = replace(
                self.state,
                manual_fit_status="error",
                error_message=str(exc),
                status_message=str(exc),
                workflow=fail_workflow_step(self.state.workflow, "fit", str(exc)),
            )
            return None
        self.state = replace(
            self.state,
            manual_fit_status="ready",
            manual_fit_result=result,
            error_message=None,
            status_message="Current model plotted",
            workflow=complete_workflow_step(
                self.state.workflow, "fit", "Current model plotted"
            ),
        )
        return result

    def _sync_insitu_state(self, workflow_state, message: str) -> None:
        self.state = replace(
            self.state, insitu_workflow=workflow_state, status_message=message
        )

    def _sync_insitu_recipe(self, recipe, scope: str, message: str) -> None:
        self.state = replace(
            self.state,
            insitu_recipe=recipe,
            insitu_recipe_scope=str(scope),
            status_message=message,
        )
