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

    ``quick_fit(q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None,
    evaluated=None, halves=None)``
    fits one curve on |q| with sphere / vertical-cylinder / random-cylinder families (or the
    given composition, 1 = sphere, 2 = random cylinder, 3 = vertical cylinder), each with size
    dispersity and an interparticle distance (also started from ``distance_nm`` when given, e.g.
    2π/q of a correlation peak), and returns the distinct solutions, best first. Units
    of the result: q, σ_Res in nm⁻¹; R, h, D in nm (``unit_contract`` of each row).
    ``report`` is told every start, ``evaluated`` every evaluation of the model (``fit_candidates``).

    The curve is prepared first (``shared/signed_cut``), so a cut with both halves and of any length can
    be given. ``halves`` says what to fit of a cut through the beam: ``mean`` / ``positive`` / ``negative``
    (``both`` and ``both_abs`` mean the mean); every finite point is an observation, a chosen half without
    any raises ``MissingHalf`` and under ``mean`` a cut with one half is that half. ``None``: a signed q
    is fitted on the mean of its halves, a one-sided curve as it comes (without a point at q = 0). More than
    ``MAX_FIT_POINTS`` points are merged. Every row says what was fitted: ``prepared`` (the note, in
    English; empty when the curve was taken as it came) and ``preparation`` (``halves``: mean / positive /
    negative, empty for a curve taken as it came; ``points`` fitted; ``averaged`` and ``one_half``: points
    with two measurements and with one, before merging; ``merged``: neighbouring points per point of the
    fit).

    A curve that cannot be fitted is a ``ValueError`` that names what was to be fitted: fewer than
    ``QUICK_FIT_MIN_POINTS`` points (``TOO_FEW_FOR_QUICK_FIT``, by the half), a σ that is not finite and
    positive (``shared/signed_cut.BAD_SIGMA``, also when points are merged).
    """
    import numpy as np

    from src.gimap.shared.signed_cut import PreparedCut, fit_halves, merge_points, prepare_cut

    from .application.single_fit import QUICK_FIT_MIN_POINTS, TOO_FEW_FOR_QUICK_FIT
    from .application.workflow_v5 import default_options
    from .infrastructure.adapters.experimental_fit import fit_candidates
    from .infrastructure.adapters.workflow_v5 import prepare_sides

    def prepared(q, intensity, sigma, halves) -> PreparedCut:
        finite = q[np.isfinite(q)]
        if halves is not None or (np.any(finite < 0) and np.any(finite > 0)):
            return prepare_cut(q, intensity, sigma, None, fit_halves(halves), every_point=True)
        order = np.argsort(np.abs(q), kind="stable")
        order = order[q[order] != 0]  # a point at q = 0 is on neither side of the beam: the fit does not take it
        x, y, s, group = merge_points(np.abs(q)[order], intensity[order], None if sigma is None else sigma[order])
        note = f"{group} neighbouring points merged ({x.size} points for the fit)" if group > 1 else ""
        return PreparedCut(x, y, s, note, "", 0, int(order.size), group)

    def quick_fit(q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None,
                  evaluated=None, halves=None):
        options = {
            **default_options(), "method": "experimental", "q_unit": "A^-1", "side": "positive",
            "components": list(components), "max_solutions": 5, "distance_hint_nm": distance_nm,
        }
        cut = prepared(np.asarray(q_inv_angstrom, dtype=float).reshape(-1), np.asarray(intensity, float).reshape(-1),
                       None if sigma is None else np.asarray(sigma, float).reshape(-1), halves)
        if cut.points < QUICK_FIT_MIN_POINTS:
            raise ValueError(TOO_FEW_FOR_QUICK_FIT[cut.halves_used].format(points=cut.points, minimum=QUICK_FIT_MIN_POINTS))
        try:
            items = prepare_sides(cut.q, cut.intensity, cut.sigma, options)
        except ValueError as exc:  # "positive: …" is its name for the curve on |q| it was handed, not a half of the cut
            reason = str(exc)
            raise ValueError(reason[len("positive: "):] if reason.startswith("positive: ") else reason) from None
        rows = fit_candidates(items[0], options, report or (lambda *_args: None), cancelled or (lambda: False),
                              evaluated)
        made_of = {"halves": cut.halves_used, "points": int(len(items[0]["q"])), "averaged": cut.averaged,
                   "one_half": cut.single, "merged": cut.merged}
        for row in rows:
            row.update(prepared=cut.note, preparation=dict(made_of))
        return rows

    return quick_fit
