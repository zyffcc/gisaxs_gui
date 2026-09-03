from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    AxisRangeDesign,
    V5BoundsQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import (
    INACTIVE_UNIT_VALUE,
    ResolutionBounds,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_batch_v5 import (
    build_v5_candidate_context_batch,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import candidate_refinement_v5 as exact_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_refinement_v5 import (
    run_v5_exact_refinement,
    v5_external_local_refinement_seed,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_proposals_v5 import V5LocalProposal
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    latent_component_to_gui,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    EvaluationThresholds,
    ObservedCurve,
    evaluate_candidates,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import profiled_refinement as profiled_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import build_design_matrix
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)


def _designs(component_bounds, resolution_bounds, regime):
    values = []
    for slot, bounds in enumerate(component_bounds):
        axes = ["R", "sigma_R"]
        if bounds.shape == CYLINDER:
            axes.extend(("h", "sigma_h"))
        if bounds.D is not None:
            axes.extend(("D", "sigma_D"))
        values.extend(
            AxisRangeDesign(f"component[{slot}].{axis}", regime, "interior") for axis in axes
        )
    if resolution_bounds is not None:
        values.extend(
            (
                AxisRangeDesign("resolution.sigma_res", regime, "interior"),
                AxisRangeDesign("resolution.nu_res", regime, "interior"),
            )
        )
    return tuple(values)


def _query(component_bounds, *, resolution_policy="absent", resolution_bounds=None, regime="fixed"):
    return V5BoundsQuery.create(
        query_seed=91,
        generation_attempt=0,
        component_bounds=tuple(component_bounds),
        resolution_presence_policy=resolution_policy,
        resolution_bounds=resolution_bounds,
        axis_designs=_designs(component_bounds, resolution_bounds, regime),
    )


def _proposal(batch, *, local=None, mixture_index=0, draw_index=0):
    condition = batch.branch_conditions[0]
    codec = batch.query.codec_for(condition.pattern_id)
    if local is None:
        local = np.full(26, INACTIVE_UNIT_VALUE)
    components, resolution = codec.decode(local)
    encoded = codec.encode(components, resolution)
    source = "mixture_median" if draw_index == 0 else "stochastic_draw"
    return V5LocalProposal(
        query_sha256=batch.query_sha256,
        topology_id=condition.topology_id,
        pattern_id=condition.pattern_id,
        branch_batch_index=0,
        branch_rank=1,
        mixture_index=mixture_index,
        mixture_rank=mixture_index + 1,
        draw_index=draw_index,
        source=source,
        search_yield_logit=1.25,
        mixture_log_weight=-0.2 - mixture_index,
        local_unit=encoded.unit_cube,
        latent_components=components,
        resolution=resolution,
    )


def _batch_and_curve(query, amplitude_query, q, intensity, pattern_id):
    sigma = 0.02 * intensity
    preprocessed = preprocess_curve(q, intensity, sigma)
    batch = build_v5_candidate_context_batch(
        preprocessed,
        V5UncertaintyProvenance("simulated_sigma"),
        query,
        amplitude_query,
        pattern_ids=(pattern_id,),
    )
    curve = ObservedCurve(
        curve_id="v5-exact-test",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
        sigma_log=np.full(q.shape, 0.02),
    )
    return batch, curve


def test_fixed_local_ranges_and_coupled_amplitudes_survive_to_final_absent_resolution():
    bounds = (
        GuiComponentBounds("sphere", ClosedInterval(10.0, 10.0), ClosedInterval(1.0, 1.0)),
        GuiComponentBounds(
            "vertical_cylinder",
            ClosedInterval(24.0, 24.0),
            ClosedInterval(0.18, 0.18),
        ),
    )
    query = _query(bounds)
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.03, 0.05),
        k=ClosedInterval(2.0, 2.0),
        component_intensities=(ClosedInterval(0.25, 0.35), ClosedInterval(0.65, 0.75)),
        resolution_presence_policy="absent",
        int_res=None,
    )
    codec = query.codec_for(0)
    latent, resolution = codec.decode(np.full(26, 0.5))
    gui = tuple(latent_component_to_gui(item) for item in latent)
    q = np.geomspace(0.008, 1.3, 90)
    # The unconstrained optimum has weights (0.8, 0.2); the exact V5 path
    # must retain the coupled GUI weight polytope and move to its boundary.
    intensity = build_design_matrix(q, gui) @ np.asarray((0.04, 1.6, 0.4))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, 0)

    result = run_v5_exact_refinement(
        batch,
        curve,
        (_proposal(batch),),
        per_candidate_forward_evaluation_limit=8,
        forward_evaluation_limit=8,
    )

    assert result.status == "exact_candidates_ready_for_verification"
    assert result.ledger.calls_used == 1
    assert dict(result.ledger.calls_by_phase)["seed_profile_verification"] == 1
    candidate = result.candidates[0]
    assert candidate.resolution is None
    assert candidate.linear_solution.resolution_amplitude == 0.0
    assert candidate.linear_solution.k == pytest.approx(2.0, abs=2e-8)
    assert candidate.linear_solution.component_weights[0] <= 0.35 + 2e-7
    audit = result.attempts[0].prerequisite_audit
    assert audit is not None and audit.all_prerequisites_satisfied
    assert audit.initial_local_unit == audit.final_local_unit == (0.5,) * 26
    assert audit.amplitude_constraint_identity_preserved
    assert audit.final_amplitude_audit.coefficient_order == ("BG", "a_1", "a_2")
    assert audit.final_amplitude_audit.all_constraints_satisfied
    assert result.attempts[0].search_yield_logit == pytest.approx(1.25)
    assert candidate.proposal_score_raw is None

    second = replace(
        _proposal(batch),
        mixture_index=1,
        mixture_rank=2,
        source="stochastic_draw",
        draw_index=1,
    )
    limited = run_v5_exact_refinement(
        batch,
        curve,
        (_proposal(batch), second),
        per_candidate_forward_evaluation_limit=8,
        forward_evaluation_limit=1,
    )
    assert limited.status == "exact_candidates_partial_budget_exhausted"
    assert [item.status for item in limited.attempts] == [
        "refined",
        "total_forward_budget_exhausted_before_seed",
    ]
    assert limited.ledger.calls_used == 1
    assert limited.ledger.calls_remaining == 0
    assert limited.ledger.total_budget_exhausted_before_seed_attempts == 1
    assert limited.ledger.per_candidate_budget_exhausted_attempts == 0
    assert not limited.all_input_seeds_processed


def test_fixed_resolution_present_branch_profiles_int_res_inside_same_polytope():
    bounds = (GuiComponentBounds("sphere", ClosedInterval(12.0, 12.0), ClosedInterval(1.2, 1.2)),)
    resolution_bounds = ResolutionBounds(ClosedInterval(0.018, 0.018), ClosedInterval(5.0, 5.0))
    query = _query(
        bounds,
        resolution_policy="required",
        resolution_bounds=resolution_bounds,
    )
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.1, 0.1),
        k=ClosedInterval(2.0, 2.0),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="required",
        int_res=ClosedInterval(0.3, 0.3),
    )
    pattern_id = query.feasible_wire_pattern_ids[0]
    codec = query.codec_for(pattern_id)
    latent, resolution = codec.decode(np.full(26, 0.5))
    gui = tuple(latent_component_to_gui(item) for item in latent)
    q = np.geomspace(0.006, 1.2, 90)
    intensity = build_design_matrix(q, gui, resolution) @ np.asarray((0.1, 2.0, 0.6))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, pattern_id)

    result = run_v5_exact_refinement(batch, curve, (_proposal(batch),))

    candidate = result.evaluation_candidates[0]
    assert candidate.resolution is not None
    assert candidate.linear_solution.resolution_amplitude == pytest.approx(0.6, abs=2e-7)
    assert candidate.linear_solution.int_res == pytest.approx(0.3, abs=2e-7)
    audit = result.attempts[0].prerequisite_audit
    assert audit is not None
    assert audit.final_amplitude_audit.coefficient_order == ("BG", "a_1", "a_res")
    assert {item.name for item in audit.final_amplitude_audit.range_checks} == {
        "BG",
        "k",
        "Int_1",
        "int_Res",
    }
    verified = evaluate_candidates(
        curve,
        result.evaluation_candidates,
        thresholds=EvaluationThresholds(
            raw_exact_log_rmse_max=1.0e-6,
            standardized_exact_log_rmse_max=1.0e-4,
            parameter_mode_distance_max=0.08,
            raw_curve_equivalence_log_rmse_max=0.02,
            reference_mode_distance_max=0.10,
        ),
        best_of_n=(1,),
    )
    assert verified.accepted_count == 1
    assert len(verified.parameter_modes) == 1


def test_narrow_geometry_uses_the_exact_constraint_object_at_every_refinement(monkeypatch):
    bounds = (GuiComponentBounds("sphere", ClosedInterval(11.8, 12.2), ClosedInterval(0.9, 1.1)),)
    query = _query(bounds, regime="narrow")
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.02, 0.02),
        k=ClosedInterval(2.0, 2.0),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    codec = query.codec_for(0)
    truth_local = np.full(26, 0.5)
    truth_local[0:2] = (0.65, 0.35)
    truth, _ = codec.decode(truth_local)
    gui = tuple(latent_component_to_gui(item) for item in truth)
    q = np.geomspace(0.008, 1.3, 100)
    intensity = build_design_matrix(q, gui) @ np.asarray((0.02, 2.0))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, 0)
    proposal_local = np.full(26, 0.5)
    proposal_local[0:2] = (0.15, 0.85)
    proposal = _proposal(batch, local=proposal_local)
    real_refine = exact_module.refine_profiled_branch
    real_profile = profiled_module.profile_linear_amplitudes
    refiner_constraints = []
    per_evaluation_constraints = []

    def checked_refine(*args, amplitude_constraint=None, **kwargs):
        refiner_constraints.append(amplitude_constraint)
        return real_refine(*args, amplitude_constraint=amplitude_constraint, **kwargs)

    def checked_profile(*args, amplitude_constraint=None, **kwargs):
        per_evaluation_constraints.append(amplitude_constraint)
        return real_profile(*args, amplitude_constraint=amplitude_constraint, **kwargs)

    monkeypatch.setattr(exact_module, "refine_profiled_branch", checked_refine)
    monkeypatch.setattr(profiled_module, "profile_linear_amplitudes", checked_profile)
    result = run_v5_exact_refinement(
        batch,
        curve,
        (proposal,),
        per_candidate_forward_evaluation_limit=80,
        forward_evaluation_limit=80,
    )

    assert refiner_constraints == [batch.amplitude_constraints[0]]
    assert len(per_evaluation_constraints) > 3
    assert all(value is batch.amplitude_constraints[0] for value in per_evaluation_constraints)
    assert result.ledger.calls_used <= 80
    assert dict(result.ledger.calls_by_phase)["optimizer_residual"] > 0
    audit = result.attempts[0].prerequisite_audit
    assert audit is not None and audit.geometry_bounds_satisfied
    final = np.asarray(audit.final_local_unit)
    assert np.all(final >= 0.0) and np.all(final <= 1.0)
    assert audit.final_amplitude_audit.all_constraints_satisfied


def test_k4_budget_uses_only_varying_axes_and_allocates_materially_more_nfev(monkeypatch):
    bounds = (
        GuiComponentBounds("sphere", ClosedInterval(9.0, 11.0), ClosedInterval(1.0, 1.0)),
        GuiComponentBounds("sphere", ClosedInterval(14.0, 14.0), ClosedInterval(1.4, 1.4)),
        GuiComponentBounds("sphere", ClosedInterval(19.0, 19.0), ClosedInterval(1.9, 1.9)),
        GuiComponentBounds("sphere", ClosedInterval(24.0, 24.0), ClosedInterval(2.4, 2.4)),
    )
    query = _query(bounds, regime="narrow")
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.01, 0.01),
        k=ClosedInterval(1.0, 1.0),
        component_intensities=(
            ClosedInterval(0.25, 0.25),
            ClosedInterval(0.25, 0.25),
            ClosedInterval(0.25, 0.25),
            ClosedInterval(0.25, 0.25),
        ),
        resolution_presence_policy="absent",
        int_res=None,
    )
    codec = query.codec_for(0)
    latent, _ = codec.decode(np.full(26, 0.5))
    gui = tuple(latent_component_to_gui(item) for item in latent)
    q = np.geomspace(0.008, 1.3, 90)
    intensity = build_design_matrix(q, gui) @ np.asarray((0.01, 0.25, 0.25, 0.25, 0.25))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, 0)
    real_refine = exact_module.refine_profiled_branch
    allocated: list[int] = []

    def capture_allocation(*args, max_nfev, **kwargs):
        allocated.append(max_nfev)
        return real_refine(*args, max_nfev=max_nfev, **kwargs)

    monkeypatch.setattr(exact_module, "refine_profiled_branch", capture_allocation)
    allowance = 43
    result = run_v5_exact_refinement(
        batch,
        curve,
        (_proposal(batch),),
        per_candidate_forward_evaluation_limit=allowance,
        forward_evaluation_limit=allowance,
    )

    assert len(codec.active_indices) == 8
    assert len(codec.varying_indices) == 1
    expected = (allowance - 3) // (len(codec.varying_indices) + 1)
    obsolete_active_based = (allowance - 3) // (len(codec.active_indices) + 1)
    assert allocated == [expected]
    assert expected == 20
    assert expected >= 5 * obsolete_active_based
    assert result.attempts[0].status == "refined"
    assert result.ledger.calls_used <= allowance


def test_one_refinement_failure_is_audited_and_does_not_abort_next_seed(monkeypatch):
    bounds = (GuiComponentBounds("sphere", ClosedInterval(11.8, 12.2), ClosedInterval(0.9, 1.1)),)
    query = _query(bounds, regime="narrow")
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.01, 0.03),
        k=ClosedInterval(1.5, 2.5),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    codec = query.codec_for(0)
    latent, _ = codec.decode(np.full(26, 0.5))
    gui = tuple(latent_component_to_gui(item) for item in latent)
    q = np.geomspace(0.008, 1.3, 80)
    intensity = build_design_matrix(q, gui) @ np.asarray((0.02, 2.0))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, 0)
    first = _proposal(batch)
    second = replace(
        first,
        mixture_index=1,
        mixture_rank=2,
        source="stochastic_draw",
        draw_index=1,
    )
    real_refine = exact_module.refine_profiled_branch
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic candidate-local failure")
        return real_refine(*args, **kwargs)

    monkeypatch.setattr(exact_module, "refine_profiled_branch", fail_once)
    result = run_v5_exact_refinement(
        batch,
        curve,
        (first, second),
        per_candidate_forward_evaluation_limit=40,
        forward_evaluation_limit=80,
    )

    assert [item.status for item in result.attempts] == ["refinement_failed", "refined"]
    assert result.attempts[0].exact_forward_calls == 0
    assert result.candidates[0].proposal_rank == 1
    assert result.all_input_seeds_processed
    assert result.ledger.refinement_failures == 1
    assert result.ledger.refinement_successes == 1
    assert result.ledger.calls_used == result.attempts[1].exact_forward_calls


def test_per_candidate_budget_exhaustion_is_not_a_numerical_or_total_budget_failure(
    monkeypatch,
):
    bounds = (GuiComponentBounds("sphere", ClosedInterval(11.8, 12.2), ClosedInterval(0.9, 1.1)),)
    query = _query(bounds, regime="narrow")
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.01, 0.03),
        k=ClosedInterval(1.5, 2.5),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    codec = query.codec_for(0)
    latent, _ = codec.decode(np.full(26, 0.5))
    q = np.geomspace(0.008, 1.3, 80)
    intensity = build_design_matrix(
        q, tuple(latent_component_to_gui(item) for item in latent)
    ) @ np.asarray((0.02, 2.0))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, 0)

    def exhaust_candidate_budget(*args, exact_forward_call_hook=None, **kwargs):
        assert exact_forward_call_hook is not None
        for _ in range(41):
            exact_forward_call_hook("optimizer_residual")
        raise AssertionError("hard budget hook must stop the fake refiner")

    monkeypatch.setattr(exact_module, "refine_profiled_branch", exhaust_candidate_budget)
    result = run_v5_exact_refinement(
        batch,
        curve,
        (_proposal(batch),),
        per_candidate_forward_evaluation_limit=40,
        forward_evaluation_limit=80,
    )

    assert result.attempts[0].status == "per_candidate_forward_budget_exhausted"
    assert result.attempts[0].exact_forward_calls == 40
    assert result.ledger.per_candidate_budget_exhausted_attempts == 1
    assert result.ledger.total_budget_exhausted_before_seed_attempts == 0
    assert result.ledger.refinement_failures == 0
    assert result.all_input_seeds_processed
    assert result.ledger.budget_unit.endswith(
        "amplitude_profile_and_authoritative_gui_forward_verification"
    )


def test_sobol_local_seed_uses_the_same_branch_and_amplitude_acceptance_path():
    bounds = (GuiComponentBounds("sphere", ClosedInterval(10.0, 10.0), ClosedInterval(1.0, 1.0)),)
    query = _query(bounds)
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.02, 0.02),
        k=ClosedInterval(2.0, 2.0),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    codec = query.codec_for(0)
    latent, _ = codec.decode(np.full(26, 0.5))
    q = np.geomspace(0.008, 1.3, 80)
    intensity = build_design_matrix(
        q, tuple(latent_component_to_gui(item) for item in latent)
    ) @ np.asarray((0.02, 2.0))
    batch, curve = _batch_and_curve(query, amplitude, q, intensity, 0)
    sobol = v5_external_local_refinement_seed(
        batch,
        source="sobol",
        source_id="sobol-0001",
        pattern_id=0,
        local_unit=np.full(26, 0.5),
    )

    result = run_v5_exact_refinement(batch, curve, (sobol,))

    assert result.attempts[0].source == "sobol"
    assert result.attempts[0].status == "refined"
    assert result.attempts[0].prerequisite_audit is not None
    assert result.attempts[0].prerequisite_audit.amplitude_constraint_identity_preserved
    assert result.status != "no_solution"
    assert result.status != "posterior"
    assert result.scientific_scope.endswith("no_solution_or_posterior_claim")
