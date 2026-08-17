"""Fitting feature 的 composition root。"""

from __future__ import annotations

from src.gimap.app import AppContext

from .application import (
    ExportFitResult,
    GenerateCandidates,
    LoadCurve,
    LoadCandidateResults,
    LoadScatteringFile,
    MapCandidateParameters,
    RefineCandidates,
    ReviewCandidates,
    RunManualFit,
    InSituWorkflowCoordinator,
)
from .infrastructure.adapters import (
    LegacyMixedModelAdapter,
    AiPipelinePredictor,
    LocalCurveRepository,
    JsonCandidateRepository,
    LocalFitResultRepository,
    LocalScatteringFileRepository,
)
from .presentation import FittingViewModel


def create_fitting_view_model(context: AppContext) -> FittingViewModel:
    if context.jobs is None:
        raise ValueError("FittingViewModel requires AppContext.jobs")
    candidate_generation = GenerateCandidates(AiPipelinePredictor(), context.jobs)
    return FittingViewModel(
        context=context,
        load_scattering_file=LoadScatteringFile(LocalScatteringFileRepository()),
        load_curve=LoadCurve(LocalCurveRepository()),
        export_fit_result=ExportFitResult(LocalFitResultRepository()),
        run_manual_fit=RunManualFit(LegacyMixedModelAdapter()),
        generate_candidates=candidate_generation,
        refine_candidates=RefineCandidates(candidate_generation),
        review_candidates=ReviewCandidates(),
        map_candidate_parameters=MapCandidateParameters(),
        load_candidate_results=LoadCandidateResults(JsonCandidateRepository()),
        insitu_workflow=InSituWorkflowCoordinator(),
    )
