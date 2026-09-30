"""Composition root of the Fitting feature."""

from __future__ import annotations

from src.gimap.app import AppContext

from .application import (
    ExportCurveFigure,
    ExportFitResult,
    GenerateCandidates,
    LoadCurve,
    LoadCandidateResults,
    DiscoverInSituFrames,
    ManageInSituRecords,
    ManageFittingParameterFiles,
    ManageAiFittingArtifacts,
    SaveFittingLog,
    CheckFittingDependency,
    FittingAiCalculations,
    FittingCurveCalculations,
    ManualRefinementCalculations,
    FittingModelCalculations,
    MapCandidateParameters,
    ManageFittingModelParameters,
    AiFittingCatalog,
    RefineCandidates,
    ReviewCandidates,
    RunManualFit,
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
    LocalInSituFrameRepository,
    LocalInSituRecordRepository,
    MatplotlibCurveFigureWriter,
    LocalFittingParameterFileRepository,
    LocalAiFittingArtifactRepository,
    LocalFittingLogRepository,
    ImportlibFittingDependencyAvailabilityAdapter,
    FittingModelParametersAdapter,
    AiFittingCatalogAdapter,
)
from .presentation import FittingScientificViewModel, FittingViewModel


def create_fitting_view_model(context: AppContext) -> FittingViewModel:
    if context.jobs is None:
        raise ValueError("FittingViewModel requires AppContext.jobs")
    candidate_generation = GenerateCandidates(AiPipelinePredictor(), context.jobs)
    fitting_model = MixedScatteringModelAdapter()
    return FittingViewModel(
        context=context,
        discover_insitu_frames=DiscoverInSituFrames(LocalInSituFrameRepository()),
        load_curve=LoadCurve(LocalCurveRepository()),
        export_fit_result=ExportFitResult(LocalFitResultRepository()),
        export_curve_figure=ExportCurveFigure(MatplotlibCurveFigureWriter()),
        run_manual_fit=RunManualFit(fitting_model),
        generate_candidates=candidate_generation,
        refine_candidates=RefineCandidates(candidate_generation),
        review_candidates=ReviewCandidates(),
        map_candidate_parameters=MapCandidateParameters(),
        load_candidate_results=LoadCandidateResults(JsonCandidateRepository()),
        insitu_workflow=InSituWorkflowCoordinator(),
        create_insitu_recipe=CreateInSituRecipe(),
        revise_insitu_recipe=ReviseInSituRecipe(),
        insitu_records=ManageInSituRecords(LocalInSituRecordRepository()),
        parameter_files=ManageFittingParameterFiles(LocalFittingParameterFileRepository()),
        ai_artifacts=ManageAiFittingArtifacts(LocalAiFittingArtifactRepository()),
        save_fitting_log=SaveFittingLog(LocalFittingLogRepository()),
        check_dependency=CheckFittingDependency(ImportlibFittingDependencyAvailabilityAdapter()),
        model_parameters=ManageFittingModelParameters(FittingModelParametersAdapter()),
        ai_catalog=AiFittingCatalog(AiFittingCatalogAdapter()),
        scientific=FittingScientificViewModel(
            curve=FittingCurveCalculations(),
            ai=FittingAiCalculations(),
            refinement=ManualRefinementCalculations(),
            model=FittingModelCalculations(fitting_model),
        ),
    )


def create_quick_fit():
    """The torch-free numerical physical fit (``Quick physical fit``) as a plain function.

    ``quick_fit(q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None)``
    fits |q| > 0 of one curve with sphere / vertical-cylinder / random-cylinder families (or the
    given composition, 1 = sphere, 2 = random cylinder, 3 = vertical cylinder), each with size
    dispersity and an interparticle distance (also started from ``distance_nm`` when given, e.g.
    2π/q of a correlation peak), and returns the distinct solutions, best first. Units
    of the result: q, σ_Res in nm⁻¹; R, h, D in nm (``unit_contract`` of each row).
    """
    import numpy as np

    from .application.workflow_v5 import default_options
    from .infrastructure.adapters.experimental_fit import fit_candidates
    from .infrastructure.adapters.workflow_v5 import prepare_sides

    def quick_fit(q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None):
        options = {
            **default_options(), "method": "experimental", "q_unit": "A^-1", "side": "positive",
            "components": list(components), "max_solutions": 5, "distance_hint_nm": distance_nm,
        }
        q = np.abs(np.asarray(q_inv_angstrom, dtype=float))
        items = prepare_sides(q, np.asarray(intensity, float), None if sigma is None else np.asarray(sigma, float), options)
        return fit_candidates(items[0], options, report or (lambda *_args: None), cancelled or (lambda: False))

    return quick_fit
