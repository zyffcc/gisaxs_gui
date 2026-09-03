from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import PosteriorV8.rescue_inference as rescue_inference
from PosteriorV8.branch_codec import ResolutionBounds
from PosteriorV8.contract import (
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SPHERE,
    TOPOLOGIES,
    ClosedInterval,
    GuiComponentBounds,
    full_component_bounds,
    latent_component_to_gui,
    topology_id_for,
)
from PosteriorV8.evaluation import ObservedCurve
from PosteriorV8.inference_proposals import (
    ContinuousProposalOutput,
    DiscreteProposalOutput,
)
from PosteriorV8.one_click_inference import InferenceBudget, RefinementOutcome
from PosteriorV8.production_bridge import (
    ProductionBranchFactory,
    ProductionExactRefiner,
    ResolutionSearchPolicy,
    TopologyUserBounds,
    UserSearchSpace,
)
from PosteriorV8.profiled_forward import component_unit_basis
from PosteriorV8.reference_bank import CompetingBranch
from PosteriorV8.rescue_inference import (
    RescuePolicy,
    RetrievalSeed,
    run_production_inference,
)


def _bounds(*, fixed=False):
    return GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(9.0, 9.0) if fixed else ClosedInterval(8.0, 12.0),
        sigma_R=(ClosedInterval(0.9, 0.9) if fixed else ClosedInterval(0.8, 1.4)),
    )


def _setup(*, fixed=False):
    factory = ProductionBranchFactory(UserSearchSpace.for_components((_bounds(fixed=fixed),)))
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    assert context is not None
    components, resolution = context.user_bounds_codec.decode(np.full(26, 0.5))
    q = np.geomspace(1e-3, 1.0, 48)
    gui = latent_component_to_gui(components[0])
    intensity = 0.1 + 1.7 * component_unit_basis(q, gui)
    curve = ObservedCurve(curve_id="rescue-k1", source_kind="synthetic", q=q, intensity=intensity)
    inputs = {
        "x": np.ones((32, 3), dtype=np.float32),
        "point_mask": np.ones(32, dtype=bool),
        "global_features": np.zeros(5, dtype=np.float32),
    }
    return factory, branch, components, resolution, curve, inputs


def _all_topology_factory():
    topologies = tuple(
        TopologyUserBounds(
            component_bounds=tuple(
                full_component_bounds(shape, d_policy="optional") for shape in topology
            )
        )
        for topology in TOPOLOGIES
    )
    return ProductionBranchFactory(
        UserSearchSpace(
            topologies=topologies,
            resolution=ResolutionSearchPolicy(
                presence="optional",
                bounds=ResolutionBounds(
                    RESOLUTION_SIGMA_DOMAIN,
                    RESOLUTION_NU_DOMAIN,
                ),
            ),
        )
    )


class _OutOfRangeModel:
    def __init__(self, topology_id):
        self.topology_id = topology_id
        self.conditions = []

    def predict_discrete(self, curve_inputs):
        topology = np.full(34, -20.0)
        topology[self.topology_id] = 20.0
        patterns = np.full((34, 32), -20.0)
        patterns[self.topology_id, 0] = 20.0
        return DiscreteProposalOutput(
            topology_logits=topology,
            branch_pattern_logits=patterns,
        )

    def predict_continuous(self, curve_inputs, condition):
        self.conditions.append(condition)
        return ContinuousProposalOutput(
            mixture_logits=np.zeros(1),
            mixture_loc=np.full((1, 26), -20.0),
            mixture_logscale=np.full((1, 26), -20.0),
        )


def _budget(*, samples=2, total=12):
    return InferenceBudget(
        topology_beam_size=1,
        branch_beam_size=1,
        mixture_components_per_branch=1,
        samples_per_mixture=samples,
        per_candidate_forward_evaluation_limit=1,
        forward_evaluation_limit=total,
    )


def _source(result, name):
    return next(item for item in result.audit.sources if item.source == name)


def test_all_neural_out_of_bounds_is_rescued_by_user_codec_sobol():
    factory, branch, _, _, curve, inputs = _setup()
    model = _OutOfRangeModel(branch.topology_id)
    result = run_production_inference(
        inputs,
        curve,
        model=model,
        branch_factory=factory,
        inference_budget=_budget(),
        rescue_policy=RescuePolicy(
            target_candidate_count=2,
            fallback_attempt_limit=8,
            sobol_seeds_per_branch=4,
        ),
        seed=71,
    )

    assert result.status == "raw_target_reached"
    assert len(result.candidates) == 2
    assert [item.proposal_rank for item in result.candidates] == [1, 2]
    assert all(item.bounds_pass and item.physics_pass for item in result.candidates)
    assert _source(result, "neural").validation_failures == 2
    assert _source(result, "neural").candidates_returned == 0
    assert _source(result, "retrieval").available == 0
    assert _source(result, "sobol").candidates_returned == 2
    assert result.audit.forward_evaluations_used == 2
    active = np.asarray(model.conditions[0].active_dimension_mask)
    assert np.all(np.asarray(model.conditions[0].branch_low)[active] == 0.0)
    assert np.all(np.asarray(model.conditions[0].branch_high)[active] == 1.0)


def test_retrieval_precedes_sobol_and_physical_duplicates_are_audited():
    factory, branch, components, resolution, curve, inputs = _setup()
    retrieval = (
        RetrievalSeed(
            retrieval_id="same-1",
            branch=branch,
            components=components,
            resolution=resolution,
        ),
        RetrievalSeed(
            retrieval_id="same-2",
            branch=branch,
            components=components,
            resolution=resolution,
        ),
    )
    result = run_production_inference(
        inputs,
        curve,
        model=_OutOfRangeModel(branch.topology_id),
        branch_factory=factory,
        inference_budget=_budget(samples=1),
        rescue_policy=RescuePolicy(
            target_candidate_count=2,
            fallback_attempt_limit=8,
            sobol_seeds_per_branch=3,
        ),
        retrieval_seeds=retrieval,
        seed=19,
    )

    returned = [item for item in result.audit.attempts if item.status == "refined"]
    assert [item.source for item in returned] == ["retrieval", "sobol"]
    assert _source(result, "retrieval").duplicates == 1
    assert _source(result, "retrieval").candidates_returned == 1
    assert _source(result, "sobol").candidates_returned == 1
    assert result.audit.fallback_attempts_used == 3


def test_fixed_user_ranges_deduplicate_sobol_physical_repeats():
    factory, branch, _, _, curve, inputs = _setup(fixed=True)
    result = run_production_inference(
        inputs,
        curve,
        model=_OutOfRangeModel(branch.topology_id),
        branch_factory=factory,
        inference_budget=_budget(samples=1),
        rescue_policy=RescuePolicy(
            target_candidate_count=2,
            fallback_attempt_limit=4,
            sobol_seeds_per_branch=4,
            physical_dedup_tolerance=0.0,
        ),
        seed=23,
    )

    assert result.status == "raw_partial"
    assert len(result.candidates) == 1
    assert _source(result, "sobol").candidates_returned == 1
    assert _source(result, "sobol").duplicates == 3


def test_one_refinement_failure_does_not_abort_following_candidates():
    factory, branch, _, _, curve, inputs = _setup()

    class FailOnce(ProductionExactRefiner):
        calls = 0

        def refine_physical(self, *args, **kwargs):
            type(self).calls += 1
            if type(self).calls == 1:
                return RefinementOutcome(
                    status="failed",
                    forward_evaluations=1,
                    message="intentional isolated failure",
                )
            return super().refine_physical(*args, **kwargs)

    refiner = FailOnce(curve=curve)
    result = run_production_inference(
        inputs,
        curve,
        model=_OutOfRangeModel(branch.topology_id),
        branch_factory=factory,
        inference_budget=_budget(samples=1),
        rescue_policy=RescuePolicy(
            target_candidate_count=1,
            fallback_attempt_limit=4,
            sobol_seeds_per_branch=3,
        ),
        refiner=refiner,
        seed=29,
    )

    assert result.status == "raw_target_reached"
    assert len(result.candidates) == 1
    sobol_attempts = [item for item in result.audit.attempts if item.source == "sobol"]
    assert [item.status for item in sobol_attempts] == [
        "refinement_failed",
        "refined",
    ]
    assert result.audit.forward_evaluations_used == 2


def test_runs_are_deterministic_in_candidate_parameters_and_source_audit():
    factory, branch, _, _, curve, inputs = _setup()

    def run():
        return run_production_inference(
            inputs,
            curve,
            model=_OutOfRangeModel(branch.topology_id),
            branch_factory=factory,
            inference_budget=_budget(samples=1),
            rescue_policy=RescuePolicy(
                target_candidate_count=2,
                fallback_attempt_limit=4,
                sobol_seeds_per_branch=3,
            ),
            seed=37,
        )

    first, second = run(), run()
    assert first.audit == second.audit
    assert [item.components for item in first.candidates] == [
        item.components for item in second.candidates
    ]
    for left, right in zip(first.candidates, second.candidates):
        np.testing.assert_array_equal(left.exact_intensity, right.exact_intensity)


def test_all34_sobol_materialization_and_context_calls_obey_fallback_limit(monkeypatch):
    _, branch, _, _, curve, inputs = _setup()
    factory = _all_topology_factory()
    context_calls = []
    indexed_calls = []
    original_context = factory.context_for
    original_indexed = rescue_inference.generate_profiled_branch_seed_at_index

    def counted_context(candidate_branch):
        context_calls.append(candidate_branch.key)
        return original_context(candidate_branch)

    def counted_indexed(*args, **kwargs):
        indexed_calls.append(kwargs["sequence_index"])
        return original_indexed(*args, **kwargs)

    monkeypatch.setattr(factory, "context_for", counted_context)
    monkeypatch.setattr(
        rescue_inference,
        "generate_profiled_branch_seed_at_index",
        counted_indexed,
    )
    result = run_production_inference(
        inputs,
        curve,
        model=_OutOfRangeModel(branch.topology_id),
        branch_factory=factory,
        inference_budget=_budget(samples=1, total=12),
        rescue_policy=RescuePolicy(
            target_candidate_count=10,
            fallback_attempt_limit=3,
            sobol_seeds_per_branch=8,
        ),
        seed=101,
    )

    sobol = _source(result, "sobol")
    assert sobol.potential_available == 700 * 8
    assert sobol.materialized == sobol.considered == 3
    assert len(indexed_calls) == 3
    assert indexed_calls == [0, 0, 0]
    # One neural context plus at most one new context per fallback attempt.
    assert len(context_calls) == 1 + result.audit.fallback_attempt_limit
    sobol_attempts = [item for item in result.audit.attempts if item.source == "sobol"]
    assert len({item.branch_key for item in sobol_attempts}) == 3


def test_sobol_is_not_materialized_past_global_forward_budget():
    _, branch, _, _, curve, inputs = _setup()
    result = run_production_inference(
        inputs,
        curve,
        model=_OutOfRangeModel(branch.topology_id),
        branch_factory=_all_topology_factory(),
        inference_budget=_budget(samples=1, total=2),
        rescue_policy=RescuePolicy(
            target_candidate_count=10,
            fallback_attempt_limit=8,
            sobol_seeds_per_branch=8,
        ),
        seed=103,
    )

    sobol = _source(result, "sobol")
    assert result.audit.forward_evaluations_used == 2
    assert sobol.potential_available == 700 * 8
    assert sobol.materialized == sobol.considered == 1
    assert result.audit.fallback_attempts_used == 1
