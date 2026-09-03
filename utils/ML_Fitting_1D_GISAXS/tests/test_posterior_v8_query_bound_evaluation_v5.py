from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import (
    V5_AMPLITUDE_QUERY_SCHEMA,
    V5_AMPLITUDE_QUERY_VERSION,
    V5AmplitudeQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    AxisRangeDesign,
    V5_BOUNDS_QUERY_SCHEMA,
    V5_BOUNDS_QUERY_VERSION,
    V5BoundsQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import ResolutionBounds
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    gui_component_to_latent,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    EvaluationThresholds,
    LinearSolutionSnapshot,
    ObservedCurve,
    ReferenceMode,
    normalized_latent_parameter_distance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import ResolutionShape
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import (
    MODEL_V5_NAME,
    MODEL_V5_NUMERIC_POLICY_CONTRACT,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_bound_evaluation_v5 import (
    V5_QUERY_BOUND_EVALUATION_PAYLOAD,
    V5_QUERY_BOUND_EVALUATION_SCHEMA,
    V5_QUERY_BOUND_EVALUATION_SHA256,
    V5_QUERY_BOUND_EVALUATION_VERSION,
    V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
    evaluate_v5_query_bound_candidates,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_inference_contract_v5 import (
    V5UniversalCandidateProvenance,
    V5UniversalInferenceBudget,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import universal_inference_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    V5_UNIVERSAL_QUERY_SCHEMA,
    V5_UNIVERSAL_QUERY_VERSION,
    V5TopologyQuery,
    build_v5_universal_candidate_context,
)


def _axis_designs(
    bounds: tuple[GuiComponentBounds, ...],
    resolution_bounds: ResolutionBounds | None = None,
) -> tuple[AxisRangeDesign, ...]:
    keys = []
    for slot, value in enumerate(bounds):
        axes = ["R", "sigma_R"]
        if value.D is not None:
            axes.extend(("D", "sigma_D"))
        keys.extend(f"component[{slot}].{axis}" for axis in axes)
    if resolution_bounds is not None:
        keys.extend(("resolution.sigma_res", "resolution.nu_res"))
    return tuple(AxisRangeDesign(key, "wide", "interior") for key in keys)


def _context(
    bounds: tuple[GuiComponentBounds, ...],
    *,
    intensities: tuple[ClosedInterval, ...],
    background: ClosedInterval = ClosedInterval(0.0, 0.0),
    k: ClosedInterval = ClosedInterval(1.0, 1.0),
    resolution_presence_policy: str = "absent",
    resolution_bounds: ResolutionBounds | None = None,
    int_res: ClosedInterval | None = None,
):
    geometry = V5BoundsQuery.create(
        query_seed=710,
        generation_attempt=0,
        component_bounds=bounds,
        resolution_presence_policy=resolution_presence_policy,
        resolution_bounds=resolution_bounds,
        axis_designs=_axis_designs(bounds, resolution_bounds),
    )
    amplitude = V5AmplitudeQuery.create(
        background=background,
        k=k,
        component_intensities=intensities,
        resolution_presence_policy=resolution_presence_policy,
        int_res=int_res,
    )
    q = np.geomspace(0.008, 1.2, 72)
    intensity = np.ones(q.shape, dtype=np.float64)
    sigma = 0.02 * intensity
    context = build_v5_universal_candidate_context(
        preprocess_curve(q, intensity, sigma),
        V5UncertaintyProvenance("simulated_sigma"),
        (V5TopologyQuery(geometry, amplitude),),
    )
    curve = ObservedCurve(
        curve_id="query-bound-evaluator",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
        sigma_log=np.full(q.shape, 0.02),
    )
    return context, curve


def _sphere_bounds(
    low: float = 5.0,
    high: float = 30.0,
    *,
    optional_d: bool = False,
) -> GuiComponentBounds:
    return GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(low, high),
        sigma_R=ClosedInterval(0.5, 3.5),
        D=(ClosedInterval(25.0, 50.0) if optional_d else None),
        sigma_D=(ClosedInterval(2.5, 5.0) if optional_d else None),
        allow_D_absent=optional_d,
    )


def _candidate(
    context,
    curve,
    *,
    candidate_id: str,
    rank: int,
    radii: tuple[float, ...],
    amplitudes: tuple[float, ...],
    k: float,
    error: float,
    pattern_id: int = 0,
    resolution: ResolutionShape | None = None,
    resolution_amplitude: float = 0.0,
) -> tuple[CandidateInput, V5UniversalCandidateProvenance]:
    components = tuple(
        gui_component_to_latent(
            GuiComponentParameters(
                SPHERE,
                R=radius,
                sigma_R=0.1 * radius,
                D=(30.0 if pattern_id & (1 << slot) else None),
                sigma_D=(3.0 if pattern_id & (1 << slot) else None),
            )
        )
        for slot, radius in enumerate(radii)
    )
    topology_id = topology_id_for((SPHERE,) * len(components))
    branch = context.branch_for(f"topology-{topology_id:02d}:wire-{pattern_id:02d}")
    candidate = CandidateInput(
        candidate_id=candidate_id,
        proposal_rank=rank,
        topology_id=topology_id,
        components=components,
        resolution=resolution,
        linear_solution=LinearSolutionSnapshot(
            background=0.0,
            particle_amplitudes=amplitudes,
            resolution_amplitude=resolution_amplitude,
            k=k,
        ),
        exact_intensity=np.exp(error) * curve.intensity,
        bounds_pass=True,
        physics_pass=True,
    )
    provenance = V5UniversalCandidateProvenance(
        candidate_id=candidate_id,
        proposal_rank=rank,
        attempt_rank=rank,
        stage="neural_primary",
        source="neural",
        source_id=f"seed-{rank}",
        global_branch_key=branch.global_key.wire_key,
        topology_id=topology_id,
        pattern_id=pattern_id,
        model_global_index=branch.global_index,
        search_yield_logit=0.0,
        mixture_log_weight=0.0,
    )
    return candidate, provenance


def _thresholds(
    *,
    parameter_distance: float = 0.08,
    reference_distance: float = 0.10,
) -> EvaluationThresholds:
    return EvaluationThresholds(
        raw_exact_log_rmse_max=1.0,
        standardized_exact_log_rmse_max=100.0,
        parameter_mode_distance_max=parameter_distance,
        raw_curve_equivalence_log_rmse_max=1.0,
        reference_mode_distance_max=reference_distance,
    )


def _evaluate(context, curve, rows, *, thresholds=None, references=()):
    candidates, provenance = zip(*rows)
    return evaluate_v5_query_bound_candidates(
        context,
        curve,
        candidates,
        candidate_provenance=provenance,
        thresholds=thresholds or _thresholds(),
        best_of_n=(len(candidates),),
        reference_modes=references,
    )


def test_evaluator_identity_binds_latest_query_model_and_numeric_contract() -> None:
    assert V5_QUERY_BOUND_EVALUATION_SCHEMA.endswith("/v2")
    assert V5_QUERY_BOUND_EVALUATION_VERSION.endswith("_v2")
    assert V5_QUERY_BOUND_EVALUATION_PAYLOAD["query_contract"] == {
        "universal_schema": V5_UNIVERSAL_QUERY_SCHEMA,
        "universal_version": V5_UNIVERSAL_QUERY_VERSION,
        "geometry_schema": V5_BOUNDS_QUERY_SCHEMA,
        "geometry_version": V5_BOUNDS_QUERY_VERSION,
        "amplitude_schema": V5_AMPLITUDE_QUERY_SCHEMA,
        "amplitude_version": V5_AMPLITUDE_QUERY_VERSION,
    }
    assert V5_QUERY_BOUND_EVALUATION_PAYLOAD["model_contract"] == {
        "schema": MODEL_V5_SCHEMA,
        "version": MODEL_V5_VERSION,
        "name": MODEL_V5_NAME,
        "numeric_policy_contract": {
            name: dict(value)
            for name, value in MODEL_V5_NUMERIC_POLICY_CONTRACT.items()
        },
    }


def test_heterogeneous_same_shape_slots_are_not_legacy_permuted() -> None:
    context, curve = _context(
        (_sphere_bounds(5.0, 30.0), _sphere_bounds(10.0, 35.0)),
        intensities=(ClosedInterval(0.1, 1.0), ClosedInterval(0.1, 1.0)),
    )
    first = _candidate(
        context,
        curve,
        candidate_id="first",
        rank=1,
        radii=(15.0, 20.0),
        amplitudes=(0.2, 0.8),
        k=1.0,
        error=0.001,
    )
    swapped = _candidate(
        context,
        curve,
        candidate_id="swapped",
        rank=2,
        radii=(20.0, 15.0),
        amplitudes=(0.8, 0.2),
        k=1.0,
        error=0.002,
    )

    assert normalized_latent_parameter_distance(first[0], swapped[0]) == pytest.approx(0.0)
    report = _evaluate(
        context, curve, (first, swapped), thresholds=_thresholds(parameter_distance=0.01)
    )
    assert len(report.parameter_modes) == 2


def test_complete_query_exchangeable_slots_are_one_parameter_mode() -> None:
    bounds = _sphere_bounds()
    context, curve = _context(
        (bounds, bounds),
        intensities=(ClosedInterval(0.1, 1.0), ClosedInterval(0.1, 1.0)),
    )
    first = _candidate(
        context,
        curve,
        candidate_id="first",
        rank=1,
        radii=(10.0, 20.0),
        amplitudes=(0.2, 0.8),
        k=1.0,
        error=0.001,
    )
    swapped = _candidate(
        context,
        curve,
        candidate_id="swapped",
        rank=2,
        radii=(20.0, 10.0),
        amplitudes=(0.8, 0.2),
        k=1.0,
        error=0.002,
    )

    report = _evaluate(context, curve, (first, swapped))
    assert len(report.parameter_modes) == 1
    assert report.parameter_modes[0].member_candidate_ids == ("first", "swapped")


def test_shared_k_gauge_merges_but_a_scale_decade_does_not() -> None:
    bounds = (_sphere_bounds(),)
    gauge_context, gauge_curve = _context(
        bounds,
        intensities=(ClosedInterval(1.0, 2.0),),
        k=ClosedInterval(1.0, 2.0),
    )
    gauge_rows = (
        _candidate(
            gauge_context,
            gauge_curve,
            candidate_id="k-one",
            rank=1,
            radii=(10.0,),
            amplitudes=(2.0,),
            k=1.0,
            error=0.001,
        ),
        _candidate(
            gauge_context,
            gauge_curve,
            candidate_id="k-two",
            rank=2,
            radii=(10.0,),
            amplitudes=(2.0,),
            k=2.0,
            error=0.002,
        ),
    )
    assert len(_evaluate(gauge_context, gauge_curve, gauge_rows).parameter_modes) == 1

    scale_context, scale_curve = _context(
        bounds,
        intensities=(ClosedInterval(0.1, 10.0),),
    )
    scale_rows = (
        _candidate(
            scale_context,
            scale_curve,
            candidate_id="weak",
            rank=1,
            radii=(10.0,),
            amplitudes=(0.1,),
            k=1.0,
            error=0.001,
        ),
        _candidate(
            scale_context,
            scale_curve,
            candidate_id="strong",
            rank=2,
            radii=(10.0,),
            amplitudes=(10.0,),
            k=1.0,
            error=0.002,
        ),
    )
    scale_report = _evaluate(scale_context, scale_curve, scale_rows)
    assert len(scale_report.parameter_modes) == 2
    assert scale_report.audit_schema == V5_QUERY_BOUND_EVALUATION_SCHEMA
    assert scale_report.parameter_normalization_version == V5_QUERY_PARAMETER_DISTANCE_VERSION


def test_cross_contextual_D_branches_have_infinite_grouping_distance() -> None:
    context, curve = _context(
        (_sphere_bounds(optional_d=True),),
        intensities=(ClosedInterval(1.0, 1.0),),
    )
    absent = _candidate(
        context,
        curve,
        candidate_id="d-absent",
        rank=1,
        radii=(10.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.001,
        pattern_id=0,
    )
    present = _candidate(
        context,
        curve,
        candidate_id="d-present",
        rank=2,
        radii=(10.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.002,
        pattern_id=1,
    )
    reference = ReferenceMode(
        reference_id="D-present-reference",
        topology_id=present[0].topology_id,
        components=present[0].components,
        resolution=None,
        linear_solution=present[0].linear_solution,
    )
    report = _evaluate(context, curve, (absent, present), references=(reference,))
    assert len(report.parameter_modes) == 2
    assert report.reference_matches[0].matched_candidate_id == "d-present"


def test_reference_resolution_pattern_maps_to_the_present_contextual_branch() -> None:
    resolution_bounds = ResolutionBounds(
        sigma_res=ClosedInterval(0.005, 0.02),
        nu_res=ClosedInterval(2.0, 10.0),
    )
    context, curve = _context(
        (_sphere_bounds(),),
        intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="optional",
        resolution_bounds=resolution_bounds,
        int_res=ClosedInterval(0.1, 1.0),
    )
    absent = _candidate(
        context,
        curve,
        candidate_id="resolution-absent",
        rank=1,
        radii=(10.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.001,
    )
    resolution = ResolutionShape(sigma_res=0.01, nu_res=5.0)
    present = _candidate(
        context,
        curve,
        candidate_id="resolution-present",
        rank=2,
        radii=(10.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.002,
        pattern_id=16,
        resolution=resolution,
        resolution_amplitude=0.5,
    )
    reference = ReferenceMode(
        reference_id="resolution-present-reference",
        topology_id=present[0].topology_id,
        components=present[0].components,
        resolution=resolution,
        linear_solution=present[0].linear_solution,
    )
    report = _evaluate(context, curve, (absent, present), references=(reference,))

    assert len(report.parameter_modes) == 2
    assert report.reference_matches[0].matched_candidate_id == "resolution-present"


def test_reference_matching_uses_actual_emitted_representative_only() -> None:
    context, curve = _context(
        (_sphere_bounds(),),
        intensities=(ClosedInterval(1.0, 1.0),),
    )
    representative = _candidate(
        context,
        curve,
        candidate_id="actual-representative",
        rank=1,
        radii=(12.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.001,
    )
    hidden_closer = _candidate(
        context,
        curve,
        candidate_id="hidden-closer-member",
        rank=2,
        radii=(10.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.01,
    )
    reference = ReferenceMode(
        reference_id="reference",
        topology_id=topology_id_for((SPHERE,)),
        components=hidden_closer[0].components,
        resolution=None,
        linear_solution=hidden_closer[0].linear_solution,
    )
    report = _evaluate(
        context,
        curve,
        (representative, hidden_closer),
        thresholds=_thresholds(parameter_distance=0.12, reference_distance=0.02),
        references=(reference,),
    )

    assert len(report.parameter_modes) == 1
    assert report.parameter_modes[0].representative_candidate_id == "actual-representative"
    assert not report.reference_matches[0].recalled
    assert report.reference_matches[0].nearest_candidate_id == "actual-representative"
    assert report.reference_matching_version == V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION


def test_reference_and_candidate_branch_binding_fail_closed() -> None:
    context, curve = _context(
        (_sphere_bounds(),),
        intensities=(ClosedInterval(1.0, 1.0),),
    )
    row = _candidate(
        context,
        curve,
        candidate_id="candidate",
        rank=1,
        radii=(10.0,),
        amplitudes=(1.0,),
        k=1.0,
        error=0.001,
    )
    bad_provenance = replace(row[1], model_global_index=row[1].model_global_index + 1)
    with pytest.raises(ValueError, match="provenance disagrees"):
        _evaluate(context, curve, ((row[0], bad_provenance),))

    d_reference = ReferenceMode(
        reference_id="D-outside-selected-pattern",
        topology_id=row[0].topology_id,
        components=(
            gui_component_to_latent(
                GuiComponentParameters(SPHERE, R=10.0, sigma_R=1.0, D=30.0, sigma_D=3.0)
            ),
        ),
        resolution=None,
        linear_solution=row[0].linear_solution,
    )
    with pytest.raises(ValueError, match="does not map to exactly one"):
        _evaluate(context, curve, (row,), references=(d_reference,))

    changed_curve = ObservedCurve(
        curve_id=curve.curve_id,
        source_kind=curve.source_kind,
        q=curve.q,
        intensity=2.0 * curve.intensity,
        sigma_log=curve.sigma_log,
    )
    with pytest.raises(ValueError, match="not built from this ObservedCurve"):
        _evaluate(context, changed_curve, (row,))


def test_noncanonical_exchangeable_D_reference_fails_closed() -> None:
    bounds = _sphere_bounds(optional_d=True)
    context, curve = _context(
        (bounds, bounds),
        intensities=(ClosedInterval(1.0, 1.0), ClosedInterval(1.0, 1.0)),
    )
    candidate = _candidate(
        context,
        curve,
        candidate_id="canonical-absent",
        rank=1,
        radii=(10.0, 20.0),
        amplitudes=(1.0, 1.0),
        k=1.0,
        error=0.001,
    )
    noncanonical = ReferenceMode(
        reference_id="noncanonical-wire-01",
        topology_id=candidate[0].topology_id,
        components=(
            gui_component_to_latent(
                GuiComponentParameters(SPHERE, R=10.0, sigma_R=1.0, D=30.0, sigma_D=3.0)
            ),
            gui_component_to_latent(GuiComponentParameters(SPHERE, R=20.0, sigma_R=2.0)),
        ),
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=0.0,
            particle_amplitudes=(1.0, 1.0),
            resolution_amplitude=0.0,
            k=1.0,
        ),
    )

    assert f"topology-{candidate[0].topology_id:02d}:wire-01" not in context.global_branch_keys
    with pytest.raises(ValueError, match="does not map to exactly one"):
        _evaluate(context, curve, (candidate,), references=(noncanonical,))


def test_universal_early_stop_counts_scale_distinct_query_bound_modes(monkeypatch) -> None:
    context, curve = _context(
        (_sphere_bounds(),),
        intensities=(ClosedInterval(0.1, 10.0),),
    )

    class TwoSeedModel:
        def __call__(self, inputs, *, training):
            count = np.asarray(inputs["branch_topology_id"]).reshape(-1).size
            locations = np.zeros((count, 12, 26), dtype=np.float32)
            locations[:, 1, :] = 1.0
            logits = np.full((count, 12), -10.0, dtype=np.float32)
            logits[:, :2] = (1.0, 0.0)
            return {
                "proposal_search_yield_logit": np.zeros((count, 1), dtype=np.float32),
                "mixture_logits": logits,
                "mixture_loc": locations,
                "mixture_logscale": np.full_like(locations, -8.0),
            }

    emitted = []

    def scripted_refinement(batch, observed, seeds, **kwargs):
        amplitude = (0.1, 10.0)[len(emitted)]
        candidate, _ = _candidate(
            context,
            curve,
            candidate_id="candidate_00001",
            rank=1,
            radii=(10.0,),
            amplitudes=(amplitude,),
            k=1.0,
            error=0.001,
        )
        emitted.append(candidate)
        seed = seeds[0]
        attempt = SimpleNamespace(
            status="refined",
            message="scripted exact candidate",
            search_yield_logit=seed.search_yield_logit,
            mixture_log_weight=seed.mixture_log_weight,
        )
        return SimpleNamespace(
            attempts=(attempt,),
            candidates=(candidate,),
            ledger=SimpleNamespace(calls_used=1),
        )

    monkeypatch.setattr(universal_inference_v5, "run_v5_exact_refinement", scripted_refinement)
    result = universal_inference_v5.run_v5_universal_one_click_inference(
        context,
        curve,
        model=TwoSeedModel(),
        thresholds=_thresholds(),
        target_parameter_mode_count=2,
        budget=V5UniversalInferenceBudget(
                mixture_components_per_branch=4,
            stochastic_draws_per_mixture=0,
            include_mixture_medians=True,
            neural_primary_attempt_limit=2,
            fallback_attempt_limit=0,
            sobol_seeds_per_branch=0,
            per_candidate_forward_evaluation_limit=4,
            forward_evaluation_limit=8,
        ),
        best_of_n=(1, 2),
        seed=711,
    )

    assert result.status == result.termination_reason == "compatible_target_reached"
    assert result.compatible_parameter_mode_count == 2
    assert len(result.attempts) == 2
    assert result.query_bound_evaluator_sha256 == V5_QUERY_BOUND_EVALUATION_SHA256
    assert result.to_audit_dict()["query_bound_evaluator"]["sha256"] == (
        V5_QUERY_BOUND_EVALUATION_SHA256
    )
    tampered = replace(
        result.candidate_provenance[0],
        model_global_index=result.candidate_provenance[0].model_global_index + 1,
    )
    with pytest.raises(ValueError, match="provenance disagrees with its refined attempt"):
        replace(
            result,
            candidate_provenance=(tampered, *result.candidate_provenance[1:]),
        )
