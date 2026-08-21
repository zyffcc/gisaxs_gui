"""Fitting feature 的 composition root。"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app import AppContext

from .application import (
    ExportFitResult,
    GenerateCandidates,
    LoadCurve,
    LoadDetectorSettings,
    LoadCandidateResults,
    LoadScatteringFile,
    InspectScatteringSequence,
    ManageRemoteFileCache,
    ManageInSituRecords,
    ManageFittingParameterFiles,
    ManageAiFittingArtifacts,
    SaveFittingLog,
    CheckFittingDependency,
    FittingAiCalculations,
    FittingCurveCalculations,
    FittingCutCalculations,
    FittingImageCalculations,
    ManualRefinementCalculations,
    ComputeInSituCut,
    FittingModelCalculations,
    FittingQSpaceCalculations,
    MapCandidateParameters,
    ManageFittingModelParameters,
    AiFittingCatalog,
    RefineCandidates,
    ReviewCandidates,
    RunManualFit,
    SaveDetectorSettings,
    InSituWorkflowCoordinator,
    CreateInSituRecipe,
    ReviseInSituRecipe,
)
from .infrastructure.adapters import (
    MixedScatteringModelAdapter,
    AiPipelinePredictor,
    LocalCurveRepository,
    JsonCandidateRepository,
    LocalFitResultRepository,
    LocalScatteringFileRepository,
    LocalRemoteFileCacheAdapter,
    LocalInSituRecordRepository,
    LocalFittingParameterFileRepository,
    LocalAiFittingArtifactRepository,
    LocalFittingLogRepository,
    ImportlibFittingDependencyAvailabilityAdapter,
    QSpaceGeometryAdapter,
    FittingModelParametersAdapter,
    AiFittingCatalogAdapter,
)
from .presentation import FittingScientificViewModel, FittingViewModel


def _create_scattering_loader(*, prepare_path=None, progress=None):
    return LoadScatteringFile(
        LocalScatteringFileRepository(
            prepare_path=prepare_path,
            progress=progress,
        )
    )


def create_fitting_view_model(context: AppContext) -> FittingViewModel:
    if context.jobs is None:
        raise ValueError("FittingViewModel requires AppContext.jobs")
    candidate_generation = GenerateCandidates(AiPipelinePredictor(), context.jobs)
    fitting_model = MixedScatteringModelAdapter()
    return FittingViewModel(
        context=context,
        load_scattering_file=LoadScatteringFile(LocalScatteringFileRepository()),
        inspect_scattering_sequence=InspectScatteringSequence(
            LocalScatteringFileRepository()
        ),
        load_curve=LoadCurve(LocalCurveRepository()),
        export_fit_result=ExportFitResult(LocalFitResultRepository()),
        run_manual_fit=RunManualFit(fitting_model),
        generate_candidates=candidate_generation,
        refine_candidates=RefineCandidates(candidate_generation),
        review_candidates=ReviewCandidates(),
        map_candidate_parameters=MapCandidateParameters(),
        load_candidate_results=LoadCandidateResults(JsonCandidateRepository()),
        insitu_workflow=InSituWorkflowCoordinator(),
        create_insitu_recipe=CreateInSituRecipe(),
        revise_insitu_recipe=ReviseInSituRecipe(),
        load_detector_settings=LoadDetectorSettings(context.settings),
        save_detector_settings=SaveDetectorSettings(context.settings),
        scattering_loader_factory=_create_scattering_loader,
        remote_file_cache=ManageRemoteFileCache(
            LocalRemoteFileCacheAdapter(Path(__file__).resolve().parents[4])
        ),
        insitu_records=ManageInSituRecords(LocalInSituRecordRepository()),
        parameter_files=ManageFittingParameterFiles(
            LocalFittingParameterFileRepository()
        ),
        ai_artifacts=ManageAiFittingArtifacts(
            LocalAiFittingArtifactRepository()
        ),
        save_fitting_log=SaveFittingLog(LocalFittingLogRepository()),
        check_dependency=CheckFittingDependency(
            ImportlibFittingDependencyAvailabilityAdapter()
        ),
        model_parameters=ManageFittingModelParameters(
            FittingModelParametersAdapter()
        ),
        ai_catalog=AiFittingCatalog(AiFittingCatalogAdapter()),
        scientific=FittingScientificViewModel(
            image=FittingImageCalculations(),
            cut=FittingCutCalculations(),
            curve=FittingCurveCalculations(),
            ai=FittingAiCalculations(),
            refinement=ManualRefinementCalculations(),
            insitu_cut=ComputeInSituCut(),
            model=FittingModelCalculations(fitting_model),
            q_space=FittingQSpaceCalculations(QSpaceGeometryAdapter()),
        ),
    )
