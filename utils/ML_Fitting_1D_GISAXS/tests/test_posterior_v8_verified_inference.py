from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.contract import (
    CYLINDER,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    full_component_bounds,
    latent_component_to_gui,
    topology_id_for,
)
from PosteriorV8.branch_catalog import branch_pattern_id
from PosteriorV8.branch_codec import ResolutionBounds
from PosteriorV8.evaluation import (
    EVALUATION_AUDIT_SCHEMA,
    EvaluationThresholds,
    LinearSolutionSnapshot,
    ObservedCurve,
)
from PosteriorV8.inference_proposals import ContinuousProposalOutput, DiscreteProposalOutput
from PosteriorV8.one_click_inference import InferenceBudget, RefinementOutcome
from PosteriorV8.production_bridge import (
    ProductionBranchFactory,
    ProductionExactRefiner,
    ResolutionSearchPolicy,
    UserSearchSpace,
)
from PosteriorV8.profiled_forward import build_design_matrix, component_unit_basis
from PosteriorV8.reference_bank import CompetingBranch
from PosteriorV8.rescue_inference import RESCUE_INFERENCE_VERSION, RescuePolicy
from PosteriorV8.verified_inference import (
    VERIFIED_ONE_CLICK_SCHEMA,
    run_verified_one_click_inference,
)


class _OutsideUserBoundsModel:
    def __init__(self, topology_id, pattern_id=0):
        self.topology_id = topology_id
        self.pattern_id = pattern_id

    def predict_discrete(self, curve_inputs):
        topology = np.full(34, -20.0)
        topology[self.topology_id] = 20.0
        patterns = np.full((34, 32), -20.0)
        patterns[self.topology_id, self.pattern_id] = 20.0
        return DiscreteProposalOutput(
            topology_logits=topology,
            branch_pattern_logits=patterns,
        )

    def predict_continuous(self, curve_inputs, condition):
        return ContinuousProposalOutput(
            mixture_logits=np.zeros(1),
            mixture_loc=np.full((1, 26), -20.0),
            mixture_logscale=np.full((1, 26), -20.0),
        )


def _setup():
    bounds = GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(8.0, 12.0),
        sigma_R=ClosedInterval(0.8, 1.4),
    )
    factory = ProductionBranchFactory(UserSearchSpace.for_components((bounds,)))
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    assert context is not None
    components, _ = context.user_bounds_codec.decode(np.full(26, 0.5))
    q = np.geomspace(1e-3, 1.0, 48)
    intensity = 0.1 + 1.7 * component_unit_basis(
        q,
        latent_component_to_gui(components[0]),
    )
    curve = ObservedCurve(
        curve_id="verified-k1",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    curve_inputs = {
        "x": np.ones((32, 3), dtype=np.float32),
        "point_mask": np.ones(32, dtype=bool),
        "global_features": np.zeros(5, dtype=np.float32),
    }
    return factory, branch, components, curve, curve_inputs


def _budget():
    return InferenceBudget(
        topology_beam_size=1,
        branch_beam_size=1,
        mixture_components_per_branch=1,
        samples_per_mixture=1,
        per_candidate_forward_evaluation_limit=1,
        forward_evaluation_limit=16,
    )


def _policy(*, attempts):
    return RescuePolicy(
        target_candidate_count=1,
        fallback_attempt_limit=attempts,
        sobol_seeds_per_branch=8,
    )


def _thresholds(*, parameter_distance=0.0):
    return EvaluationThresholds(
        raw_exact_log_rmse_max=0.01,
        standardized_exact_log_rmse_max=1.0,
        parameter_mode_distance_max=parameter_distance,
        raw_curve_equivalence_log_rmse_max=0.01,
        reference_mode_distance_max=0.03,
    )


def _replace_success(outcome, **changes):
    assert outcome.status == "success"
    return replace(outcome, value=replace(outcome.value, **changes))


def test_incompatible_raw_candidates_continue_without_compatible_success():
    factory, branch, _, curve, inputs = _setup()

    class IncompatibleRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            return _replace_success(
                outcome,
                exact_intensity=self.curve.intensity * np.e,
            )

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=3),
        refiner=IncompatibleRefiner(curve=curve),
        seed=11,
    )

    assert result.status == "no_compatible_mode_found_within_budget"
    assert len(result.raw_candidates) == 3
    assert result.audit.fallback_attempts_used == 3
    assert result.evaluation_report.accepted_count == 0
    assert result.verified_representative_candidate_ids == ()
    assert result.raw_generation.status == "raw_target_reached"


def test_repeated_parameter_modes_do_not_count_toward_compatible_target():
    factory, branch, fixed_components, curve, inputs = _setup()
    fixed_linear = LinearSolutionSnapshot(
        background=0.1,
        particle_amplitudes=(1.7,),
        resolution_amplitude=0.0,
    )

    class SameModeRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            return _replace_success(
                outcome,
                components=fixed_components,
                linear_solution=fixed_linear,
                exact_intensity=self.curve.intensity,
            )

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(parameter_distance=0.0),
        target_parameter_mode_count=2,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=4),
        refiner=SameModeRefiner(curve=curve),
        seed=13,
    )

    assert result.status == "compatible_partial"
    assert len(result.raw_candidates) == 4
    assert result.verified_parameter_mode_count == 1
    assert len(result.evaluation_report.parameter_modes) == 1
    assert result.audit.fallback_attempts_used == 4


def test_search_stops_after_n_distinct_exact_compatible_parameter_modes():
    factory, branch, fixed_components, curve, inputs = _setup()
    calls = 0

    class CompatibleRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            nonlocal calls
            outcome = super().refine_physical(*args, **kwargs)
            calls += 1
            component = replace(
                fixed_components[0],
                log_R=fixed_components[0].log_R + calls * 1.0e-4,
            )
            exact = 0.1 + 1.7 * component_unit_basis(
                self.curve.q, latent_component_to_gui(component)
            )
            return _replace_success(
                outcome,
                components=(component,),
                linear_solution=LinearSolutionSnapshot(
                    background=0.1,
                    particle_amplitudes=(1.7,),
                    resolution_amplitude=0.0,
                ),
                exact_intensity=exact,
            )

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(parameter_distance=0.0),
        target_parameter_mode_count=2,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=8),
        refiner=CompatibleRefiner(curve=curve),
        seed=17,
    )

    assert result.status == "compatible_target_reached"
    assert len(result.raw_candidates) == 2
    assert result.verified_parameter_mode_count == 2
    assert result.audit.fallback_attempts_used == 2
    assert result.raw_refinement_forward_evaluations == 2
    assert result.observability_exact_forward_evaluations == 2
    assert result.total_forward_evaluations == 4
    assert result.total_forward_evaluations <= result.total_forward_evaluation_limit
    assert result.audit_schema == VERIFIED_ONE_CLICK_SCHEMA
    assert result.raw_generation_schema == RESCUE_INFERENCE_VERSION
    assert result.evaluation_audit_schema == EVALUATION_AUDIT_SCHEMA
    assert result.evaluation_report.audit_schema == EVALUATION_AUDIT_SCHEMA


def test_zero_raw_candidates_is_explicitly_not_found_within_budget_and_never_no_solution():
    factory, branch, _, curve, inputs = _setup()

    class AlwaysFails(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            return RefinementOutcome(
                status="failed",
                forward_evaluations=1,
                message="intentional test failure",
            )

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=3),
        refiner=AlwaysFails(curve=curve),
        seed=19,
    )

    assert result.status == "no_candidate_found_within_budget"
    assert result.raw_candidates == ()
    assert result.evaluation_report is None
    assert result.raw_generation.status == "no_candidate_found_within_budget"
    with pytest.raises(ValueError, match="invalid raw"):
        replace(result.raw_generation, status="no_solution")


def test_compatible_but_redundant_particle_still_satisfies_primary_search():
    factory, branch, components, _, inputs = _setup()
    q = np.geomspace(1e-3, 1.0, 48)
    exact = 1.0 + 1.0e-9 * component_unit_basis(q, latent_component_to_gui(components[0]))
    curve = ObservedCurve(
        curve_id="verified-redundant-k1",
        source_kind="synthetic",
        q=q,
        intensity=exact,
    )

    class RedundantRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            return _replace_success(
                outcome,
                components=components,
                linear_solution=LinearSolutionSnapshot(
                    background=1.0,
                    particle_amplitudes=(1.0e-9,),
                    resolution_amplitude=0.0,
                ),
                exact_intensity=exact,
            )

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=3),
        refiner=RedundantRefiner(curve=curve),
        seed=23,
    )

    assert result.status == "compatible_target_reached"
    assert len(result.raw_candidates) == 1
    assert result.audit.fallback_attempts_used == 1
    assert all(item.status == "confirmed_redundant" for item in result.observability_assessments)
    assert result.compatible_parameter_mode_count == 1
    assert result.effective_parameter_mode_count == 0


def test_compatible_candidate_with_no_observability_budget_is_unknown():
    factory, branch, components, curve, inputs = _setup()
    linear = LinearSolutionSnapshot(
        background=0.1,
        particle_amplitudes=(1.7,),
        resolution_amplitude=0.0,
    )

    class CompatibleRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            return _replace_success(
                outcome,
                components=components,
                linear_solution=linear,
                exact_intensity=self.curve.intensity,
            )

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=2),
        refiner=CompatibleRefiner(curve=curve),
        observability_forward_evaluation_limit=0,
        seed=29,
    )

    assert result.status == "compatible_target_reached"
    assert len(result.raw_candidates) == 1
    assert result.observability_exact_forward_evaluations == 0
    assert all(item.budget_exhausted for item in result.observability_assessments)
    assert result.compatible_parameter_mode_count == 1
    assert result.effective_parameter_mode_count == 0
    assert result.total_forward_evaluations == result.raw_refinement_forward_evaluations


def test_k2_d_and_resolution_unknown_candidate_reaches_primary_compatible_target():
    component_bounds = (
        full_component_bounds(SPHERE, d_policy="required"),
        full_component_bounds(CYLINDER, d_policy="required"),
    )
    factory = ProductionBranchFactory(
        UserSearchSpace.for_components(
            component_bounds,
            resolution=ResolutionSearchPolicy(
                presence="required",
                bounds=ResolutionBounds(
                    RESOLUTION_SIGMA_DOMAIN,
                    RESOLUTION_NU_DOMAIN,
                ),
            ),
        )
    )
    topology_id = topology_id_for((SPHERE, CYLINDER))
    pattern_id = branch_pattern_id((True, True, False, False), True)
    branch = CompetingBranch(topology_id=topology_id, pattern_id=pattern_id)
    context = factory.context_for(branch)
    assert context is not None
    components, resolution = context.user_bounds_codec.decode(np.full(26, 0.5))
    assert resolution is not None
    q = np.geomspace(0.006, 1.2, 64)
    intensity = build_design_matrix(
        q,
        tuple(latent_component_to_gui(item) for item in components),
        resolution,
    ) @ np.asarray((0.1, 1.0, 0.8, 0.2))
    curve = ObservedCurve(
        curve_id="complex-compatible",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    inputs = {
        "x": np.ones((32, 3), dtype=np.float32),
        "point_mask": np.ones(32, dtype=bool),
        "global_features": np.zeros(5, dtype=np.float32),
    }

    class CompatibleComplexRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            return _replace_success(outcome, exact_intensity=self.curve.intensity)

    result = run_verified_one_click_inference(
        inputs,
        curve,
        model=_OutsideUserBoundsModel(topology_id, pattern_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(),
        rescue_policy=_policy(attempts=3),
        refiner=CompatibleComplexRefiner(curve=curve),
        observability_forward_evaluation_limit=0,
        seed=31,
    )

    assert result.status == "compatible_target_reached"
    assert len(result.raw_candidates) == 1
    assert result.compatible_parameter_mode_count == 1
    assert result.effective_parameter_mode_count == 0
    assert result.observability_assessments[0].status == "provisional_or_unknown"
    assert len(result.raw_candidates[0].components) == 2
    assert all(item.log_D is not None for item in result.raw_candidates[0].components)
    assert result.raw_candidates[0].resolution is not None
