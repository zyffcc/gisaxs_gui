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


CURVE_FIT_SEED = 0
"""The seed of every random draw of ``create_curve_fit``'s Predict parameters (Physics and Methods 14.8)."""


def create_curve_fit():
    """Predict parameters as a plain function, for the analyses that fit one cut on their own (Run Automatic
    Analysis, Batch Export's particle model, Process with AI and the command line).

    ``curve_fit(q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None,
    evaluated=None, halves=None, cut_qz=None)`` finds the particles of one cut on |q|: the given composition
    (1 = sphere, 2 = random cylinder, 3 = vertical cylinder, each with a paracrystal) or, with none given, the
    network's choice among one or two populations of spheres and upright cylinders, with or without a paracrystal
    (random cylinders cost far more time: give them as the composition). It returns the distinct solutions, most probable first, as the rows of
    the 1D Predict window (``posterior_job.curve_rows``: values, curves, ``probability``, ``at_bounds``, ``warnings``).
    Every row also names what made it: ``network`` (``{"name", "sha256"}``: the proposal network and the SHA-256 of
    its weights file, as the 1D Predict record writes them) and ``seed`` (``CURVE_FIT_SEED``).
    Units of the result: q, σ_Res in nm⁻¹; R, h, D in nm. ``distance_nm`` and ``evaluated`` are accepted for the
    callers of the former quick fit and not used (the network needs no start); ``report`` hears the steps.
    When the angles of the Yoneda cut are known, ``cut_qz`` is its vacuum q_z = k(sin α_i + sin α_f) in nm⁻¹.
    Predict fits the Born term there (level 2) and reports its reflected-beam cut flags. With no cut q_z it
    fits the effective interface at q_z = 0 (level 1).

    The curve is prepared first (``shared/signed_cut``), so a cut with both halves and of any length can be given.
    ``halves`` says what to fit of a cut through the beam: ``mean`` / ``positive`` / ``negative`` (``both`` and
    ``both_abs`` mean the mean); every finite point is an observation, a chosen half without any raises
    ``MissingHalf`` and under ``mean`` a cut with one half is that half. ``None``: a signed q is fitted on the mean of
    its halves, a one-sided curve as it comes (without a point at q = 0). More than ``MAX_FIT_POINTS`` points are
    merged. Every row says what was fitted: ``prepared`` (the note, in English; empty when the curve was taken as it
    came), ``preparation`` (``halves``, ``points``, ``averaged``, ``one_half``, ``merged``) and ``curve_notes``
    (``application.curve_intake``: a σ estimated for a curve without one — then the points with I ≤ 0 are left out —
    and the negative and zero intensities kept with their σ; each ``{"kind", "text", counts}``).

    A curve that cannot be fitted is a ``ValueError`` that names what was to be fitted: fewer than
    ``CURVE_FIT_MIN_POINTS`` points (``TOO_FEW_FOR_CURVE_FIT``, by the half), a σ that is not finite and positive
    (``shared/signed_cut.BAD_SIGMA``, also when points are merged), the network missing.
    """
    import numpy as np

    from src.gimap.shared.signed_cut import PreparedCut, fit_halves, merge_points, prepare_cut

    from .application.single_fit import CURVE_FIT_MIN_POINTS, TOO_FEW_FOR_CURVE_FIT
    from .application.fit_reproduction import prediction_record
    from .application.workflow_v5 import default_options
    from .domain.posterior import encoding, network, refine
    from .infrastructure.adapters.posterior_job import batch_compositions, curve_rows, encode
    from .infrastructure.reproduce_fit import predict_options, prediction_plan
    from .infrastructure.adapters.workflow_v5 import prepare_sides, side_notes

    def prepared(q, intensity, sigma, halves) -> PreparedCut:
        finite = q[np.isfinite(q)]
        if halves is not None or (np.any(finite < 0) and np.any(finite > 0)):
            return prepare_cut(q, intensity, sigma, None, fit_halves(halves), every_point=True)
        order = np.argsort(np.abs(q), kind="stable")
        order = order[q[order] != 0]  # a point at q = 0 is on neither side of the beam: the fit does not take it
        x, y, s, group = merge_points(np.abs(q)[order], intensity[order], None if sigma is None else sigma[order])
        note = f"{group} neighbouring points merged ({x.size} points for the fit)" if group > 1 else ""
        return PreparedCut(x, y, s, note, "", 0, int(order.size), group)

    def curve_fit(q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None,
                  evaluated=None, halves=None, cut_qz=None):
        options = {**default_options(), "q_unit": "A^-1", "side": "positive", "components": list(components)}
        cut = prepared(np.asarray(q_inv_angstrom, dtype=float).reshape(-1), np.asarray(intensity, float).reshape(-1),
                       None if sigma is None else np.asarray(sigma, float).reshape(-1), halves)
        if cut.points < CURVE_FIT_MIN_POINTS:
            raise ValueError(TOO_FEW_FOR_CURVE_FIT[cut.halves_used].format(points=cut.points,
                                                                           minimum=CURVE_FIT_MIN_POINTS))
        try:
            items = prepare_sides(cut.q, cut.intensity, cut.sigma, options)
        except ValueError as exc:  # "positive: …" is its name for the curve on |q| it was handed, not a half of the cut
            reason = str(exc)
            raise ValueError(reason[len("positive: "):] if reason.startswith("positive: ") else reason) from None
        item = items[0]
        (report or (lambda *_args: None))(0, 1, "Predict parameters: the network's proposals, then least squares")
        chosen = (batch_compositions(list(components)) if components else
                  {"allowed": encoding.select(("sphere", "vertical_cylinder"), 2)})
        proposals = network.load()
        settings = predict_options(dict(seed=CURVE_FIT_SEED, qz=cut_qz,
                                        reflected_beam=True if cut_qz else None, **chosen))
        solutions = refine.predict(proposals, item["q"], item["observed"], item["sigma"],
                                   cancelled=cancelled, **settings)
        if not solutions:
            raise ValueError("Predict parameters found no solution for this curve")
        q = np.asarray(item["q"], float)
        rows = curve_rows(item, solutions[:5], np.linspace(q.min(), q.max(), options["render_points"]),
                          chosen_by="user_composition" if components else "network_choice")
        made_of = {"halves": cut.halves_used, "points": int(len(item["q"])), "averaged": cut.averaged,
                   "one_half": cut.single, "merged": cut.merged}
        # As the 1D Predict record names it: the network and the SHA-256 of its weights (checked when it is loaded).
        used = {"name": network.path().stem, "sha256": (getattr(proposals, "meta", None) or {}).get("sha256")}
        if used["sha256"]:
            plan = prediction_plan(dict(q=q.tolist(), intensity=item["observed"].tolist(), sigma=item["sigma"].tolist()),
                                   settings, network.path(), proposals.meta,
                                   display_q=np.linspace(q.min(), q.max(), options["render_points"]), limit=5)
            rows[0]["reproduction"] = prediction_record(plan, [encode(s) for s in solutions[:5]])
        for row in rows:
            row.update(prepared=cut.note, preparation=dict(made_of), network=dict(used), seed=CURVE_FIT_SEED,
                       curve_notes=side_notes(item, {}))  # an estimated σ, negative and zero intensities
        return rows

    return curve_fit
