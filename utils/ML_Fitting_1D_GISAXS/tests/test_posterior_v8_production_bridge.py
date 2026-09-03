from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import PosteriorV8.production_bridge as production_bridge
from PosteriorV8.branch_codec import INACTIVE_UNIT_VALUE, ResolutionBounds
from PosteriorV8.contract import (
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    full_component_bounds,
    latent_component_to_gui,
    topology_id_for,
)
from PosteriorV8.evaluation import CandidateInput, ObservedCurve
from PosteriorV8.inference_proposals import (
    BoundedProposalSample,
    JointBranchScore,
)
from PosteriorV8.production_bridge import (
    ProductionBranchFactory,
    ProductionExactRefiner,
    ResolutionSearchPolicy,
    TopologyUserBounds,
    UserSearchSpace,
    physical_seed_from_external,
    physical_seed_from_neural,
    physically_duplicate,
)
from PosteriorV8.profiled_forward import component_unit_basis
from PosteriorV8.reference_bank import CompetingBranch


def _sphere_bounds(*, d_policy="absent", fixed=False):
    radius = ClosedInterval(9.0, 9.0) if fixed else ClosedInterval(8.0, 12.0)
    sigma = ClosedInterval(0.9, 0.9) if fixed else ClosedInterval(0.8, 1.4)
    values = {"shape": SPHERE, "R": radius, "sigma_R": sigma}
    if d_policy != "absent":
        values.update(
            D=ClosedInterval(25.0, 45.0),
            sigma_D=ClosedInterval(2.5, 5.0),
            allow_D_absent=d_policy == "optional",
        )
    return GuiComponentBounds(**values)


def _score(branch):
    return JointBranchScore(
        branch=branch,
        topology_rank=1,
        pattern_rank_within_topology=1,
        joint_rank=1,
        topology_log_score=-0.1,
        conditional_pattern_log_score=-0.2,
        joint_log_score=-0.3,
    )


def _proposal(context, global_unit, *, mixture=0, sample=1):
    condition = context
    active = np.asarray(condition.active_dimension_mask)
    local = np.asarray(global_unit, dtype=np.float64).copy()
    local[~active] = INACTIVE_UNIT_VALUE
    return BoundedProposalSample(
        condition=condition,
        mixture_index=mixture,
        mixture_rank=1,
        sample_index=sample,
        conditioned_mixture_log_weight=-0.4,
        local_box_unit=tuple(local),
        global_unit=tuple(global_unit),
    )


def _factory(bounds, *, resolution=None):
    return ProductionBranchFactory(
        UserSearchSpace.for_components(
            bounds,
            resolution=resolution,
        )
    )


def test_k1_to_k4_optional_policies_construct_every_feasible_hard_branch():
    resolution = ResolutionSearchPolicy(
        presence="optional",
        bounds=ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN),
    )
    topologies = tuple(
        TopologyUserBounds(
            component_bounds=tuple(
                full_component_bounds(SPHERE, d_policy="optional") for _ in range(k)
            )
        )
        for k in range(1, 5)
    )
    factory = ProductionBranchFactory(UserSearchSpace(topologies=topologies, resolution=resolution))
    contexts = factory.feasible_contexts()

    for k in range(1, 5):
        topology_id = topology_id_for((SPHERE,) * k)
        selected = [item for item in contexts if item.branch.topology_id == topology_id]
        assert len(selected) == 2 ** (k + 1)
        assert all(item.full_domain_codec.topology_id == topology_id for item in selected)


def test_d_resolution_policy_and_hard_core_infeasibility_filter_branches():
    impossible_d = GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(20.0, 30.0),
        sigma_R=ClosedInterval(2.0, 3.0),
        D=ClosedInterval(3.0, 30.0),
        sigma_D=ClosedInterval(0.3, 3.0),
        allow_D_absent=True,
    )
    required_resolution = ResolutionSearchPolicy(
        presence="required",
        bounds=ResolutionBounds(ClosedInterval(0.01, 0.05), ClosedInterval(2.0, 7.0)),
    )
    factory = _factory((impossible_d,), resolution=required_resolution)
    topology_id = topology_id_for((SPHERE,))
    absent = CompetingBranch(topology_id=topology_id, pattern_id=16)
    present = CompetingBranch(topology_id=topology_id, pattern_id=17)

    assert factory.context_for(absent) is not None
    assert factory.context_for(present) is None
    assert factory.context_for(CompetingBranch(topology_id=topology_id, pattern_id=0)) is None


def test_neural_coordinate_crosses_full_decode_then_user_encode_not_local_decode():
    factory = _factory((_sphere_bounds(),))
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    runtime = factory.context_for(branch)
    assert runtime is not None
    user_midpoint = np.full(26, INACTIVE_UNIT_VALUE)
    components, resolution = runtime.user_bounds_codec.decode(user_midpoint)
    global_coordinates = runtime.full_domain_codec.encode(components, resolution)
    condition = factory.condition_for(_score(branch))
    assert condition is not None
    active = np.asarray(condition.active_dimension_mask)
    assert np.all(np.asarray(condition.branch_low)[active] == 0.0)
    assert np.all(np.asarray(condition.branch_high)[active] == 1.0)
    proposal = _proposal(condition, global_coordinates.unit_cube)

    recovered = physical_seed_from_neural(proposal)
    assert recovered.components == components
    np.testing.assert_allclose(
        recovered.user_local_coordinates.unit_cube,
        user_midpoint,
        rtol=0.0,
        atol=2e-14,
    )
    directly_decoded, _ = runtime.user_bounds_codec.decode(proposal.global_unit)
    assert directly_decoded != components

    outside = list(global_coordinates.unit_cube)
    outside[0] = 0.0
    with pytest.raises(ValueError, match="outside"):
        physical_seed_from_neural(_proposal(condition, outside, sample=2))


def test_physical_dedup_handles_fixed_ranges_and_retrieval_sobol_sources():
    factory = _factory((_sphere_bounds(fixed=True),))
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    assert context is not None
    components, resolution = context.user_bounds_codec.decode(np.full(26, 0.5))
    retrieval = physical_seed_from_external(
        source="retrieval",
        source_id="bank-1",
        context=context,
        components=components,
        resolution=resolution,
    )
    sobol = physical_seed_from_external(
        source="sobol",
        source_id="sobol-1",
        context=context,
        components=components,
        resolution=resolution,
    )
    assert physically_duplicate(sobol, (retrieval,), tolerance=0.0)


def test_exact_refiner_returns_candidate_and_numerical_failure_isolated(monkeypatch):
    factory = _factory((_sphere_bounds(),))
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    assert context is not None
    components, resolution = context.user_bounds_codec.decode(np.full(26, 0.5))
    gui = latent_component_to_gui(components[0])
    q = np.geomspace(1e-3, 1.5, 72)
    intensity = 0.08 + 2.5 * component_unit_basis(q, gui)
    curve = ObservedCurve(
        curve_id="clean-k1",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    seed = physical_seed_from_external(
        source="sobol",
        source_id="sobol-mid",
        context=context,
        components=components,
        resolution=resolution,
    )
    refiner = ProductionExactRefiner(curve=curve)
    outcome = refiner.refine_physical(
        seed,
        candidate_id="candidate_00001",
        proposal_rank=1,
        max_forward_evaluations=32,
    )
    assert outcome.status == "success"
    assert 0 < outcome.forward_evaluations <= 32
    assert isinstance(outcome.value, CandidateInput)
    np.testing.assert_allclose(outcome.value.exact_intensity, intensity, rtol=1e-7)
    assert outcome.value.bounds_pass and outcome.value.physics_pass

    def explode(*args, **kwargs):
        raise FloatingPointError("synthetic numeric failure")

    monkeypatch.setattr(production_bridge, "refine_profiled_branch", explode)
    failed = refiner.refine_physical(
        seed,
        candidate_id="candidate_00002",
        proposal_rank=2,
        max_forward_evaluations=32,
    )
    assert failed.status == "failed"
    assert failed.value is None
    assert failed.forward_evaluations == 32
    assert "synthetic numeric failure" in failed.message


def test_tiny_per_candidate_budget_returns_exact_seed_profile_without_overrun():
    factory = _factory((_sphere_bounds(),))
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    components, resolution = context.user_bounds_codec.decode(np.full(26, 0.5))
    gui = latent_component_to_gui(components[0])
    q = np.geomspace(1e-3, 1.0, 48)
    intensity = 0.1 + component_unit_basis(q, gui)
    curve = ObservedCurve(curve_id="tiny-budget", source_kind="synthetic", q=q, intensity=intensity)
    physical = physical_seed_from_external(
        source="retrieval",
        source_id="tiny",
        context=context,
        components=components,
        resolution=resolution,
    )
    outcome = ProductionExactRefiner(curve=curve).refine_physical(
        physical,
        candidate_id="candidate_00001",
        proposal_rank=1,
        max_forward_evaluations=1,
    )
    assert outcome.status == "success"
    assert outcome.forward_evaluations == 1
    assert isinstance(outcome.value, CandidateInput)
