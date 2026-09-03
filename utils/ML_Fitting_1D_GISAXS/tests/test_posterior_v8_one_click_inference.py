from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.branch_catalog import branch_pattern_is_valid
from PosteriorV8.contract import CYLINDER, SPHERE, topology_id_for
from PosteriorV8.inference_proposals import (
    GLOBAL_TARGET_AFFINE_SEMANTICS,
    BranchCondition,
    CallableProposalModel,
    ContinuousProposalOutput,
    DiscreteProposalOutput,
    JointBranchScore,
    build_model_inputs,
    expected_active_mask,
    full_range_condition,
    rank_joint_branches,
    sample_bounded_global_mixture,
)
from PosteriorV8.one_click_inference import (
    InferenceBudget,
    RefinementOutcome,
    run_one_click_inference,
)
from PosteriorV8.reference_bank import CompetingBranch


def _discrete(*, topology_id=0, pattern_id=0):
    topology = np.full(34, -7.0)
    topology[topology_id] = 7.0
    patterns = np.zeros((34, 32))
    patterns[topology_id, pattern_id] = 6.0
    return DiscreteProposalOutput(topology_logits=topology, branch_pattern_logits=patterns)


def _score(*, topology_id=0, pattern_id=0):
    return next(
        item
        for item in rank_joint_branches(
            _discrete(topology_id=topology_id, pattern_id=pattern_id),
            topology_limit=1,
        )
        if item.branch.pattern_id == pattern_id
    )


def _condition(score, *, first=(0.15, 0.85), second=(0.3, 0.3)):
    active = np.asarray(expected_active_mask(score.branch), dtype=bool)
    low = np.full(26, 0.5)
    high = low.copy()
    low[active], high[active] = 0.0, 1.0
    indices = np.flatnonzero(active)
    low[indices[0]], high[indices[0]] = first
    low[indices[1]], high[indices[1]] = second
    return BranchCondition(
        scored_branch=score,
        branch_low=tuple(low),
        branch_high=tuple(high),
        active_dimension_mask=tuple(active),
    )


def _curve_inputs(points=12):
    return {
        "x": np.arange(points * 3, dtype=np.float32).reshape(points, 3),
        "point_mask": np.asarray([True] * (points - 2) + [False, False]),
        "global_features": np.linspace(-1.0, 1.0, 5, dtype=np.float32),
    }


def test_joint_beam_uses_conditional_valid_pattern_scores_and_stable_ties():
    topology = np.zeros(34)
    topology[0] = topology[5] = 3.0
    patterns = np.zeros((34, 32))
    patterns[0, 2] = 1.0e6  # invalid K=1 slot bit must never enter the beam
    patterns[0, 17] = 4.0
    ranked = rank_joint_branches(
        DiscreteProposalOutput(topology_logits=topology, branch_pattern_logits=patterns),
        topology_limit=2,
    )

    assert {item.branch.topology_id for item in ranked} == {0, 5}
    assert all(
        branch_pattern_is_valid(item.branch.topology_id, item.branch.pattern_id) for item in ranked
    )
    assert not any(item.branch.topology_id == 0 and item.branch.pattern_id == 2 for item in ranked)
    assert ranked[0].branch == CompetingBranch(topology_id=0, pattern_id=17)
    assert [item.joint_rank for item in ranked] == list(range(1, len(ranked) + 1))
    assert ranked == rank_joint_branches(
        DiscreteProposalOutput(topology_logits=topology, branch_pattern_logits=patterns),
        topology_limit=2,
    )

    tied = rank_joint_branches(
        DiscreteProposalOutput(
            topology_logits=np.zeros(34), branch_pattern_logits=np.zeros((34, 32))
        ),
        topology_limit=1,
    )
    assert [(item.branch.topology_id, item.branch.pattern_id) for item in tied] == [
        (0, 0),
        (0, 1),
        (0, 16),
        (0, 17),
    ]


def test_condition_mask_is_derived_from_topology_d_and_resolution_bits():
    topology_id = topology_id_for((SPHERE, CYLINDER))
    pattern_id = 1 | 2 | 16
    branch = CompetingBranch(topology_id=topology_id, pattern_id=pattern_id)
    score = JointBranchScore(
        branch=branch,
        topology_rank=1,
        pattern_rank_within_topology=1,
        joint_rank=1,
        topology_log_score=-0.1,
        conditional_pattern_log_score=-0.2,
        joint_log_score=-0.3,
    )
    active = expected_active_mask(branch)
    assert np.flatnonzero(active).tolist() == [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 24, 25]
    condition = full_range_condition(score)
    assert condition.branch == branch
    with pytest.raises(ValueError, match="does not match the hard branch"):
        BranchCondition(
            scored_branch=score,
            branch_low=condition.branch_low,
            branch_high=condition.branch_high,
            active_dimension_mask=(True,) * 26,
        )


def test_global_target_sampling_is_deterministic_bounded_affine_and_fixed_safe():
    condition = _condition(_score(), first=(0.2, 0.2000001), second=(0.3, 0.3))
    output = ContinuousProposalOutput(
        mixture_logits=np.asarray([0.0, 0.0]),
        mixture_loc=np.asarray([[5.0] * 26, [0.0] * 26]),
        mixture_logscale=np.full((2, 26), -1.0),
    )
    first = sample_bounded_global_mixture(
        output,
        condition,
        mixture_limit=2,
        samples_per_mixture=20,
        seed=123,
    )
    second = sample_bounded_global_mixture(
        output,
        condition,
        mixture_limit=2,
        samples_per_mixture=20,
        seed=123,
    )

    assert first == second
    assert first[0].mixture_index == 1  # box mass reweights the global mixture
    low = np.asarray(condition.branch_low)
    high = np.asarray(condition.branch_high)
    active = np.asarray(condition.active_dimension_mask)
    for item in first:
        local = np.asarray(item.local_box_unit)
        global_unit = np.asarray(item.global_unit)
        expected = low + (high - low) * local
        expected[~active] = 0.5
        np.testing.assert_array_equal(global_unit, expected)
        assert item.coordinate_semantics == GLOBAL_TARGET_AFFINE_SEMANTICS
        assert np.all(global_unit >= low) and np.all(global_unit <= high)
        assert global_unit[1] == 0.3
        assert local[1] == 0.5
        assert np.all(global_unit[~active] == 0.5)
        assert low[0] < global_unit[0] < high[0]


def test_model_input_adapter_supplies_one_consistent_hard_branch():
    condition = full_range_condition(_score(pattern_id=17))
    captured = []

    def predictor(inputs):
        captured.append(inputs)
        return {
            "topology_logits": np.zeros((1, 34)),
            "branch_pattern_logits": np.zeros((1, 34, 32)),
            "mixture_logits": np.zeros((1, 3)),
            "mixture_loc": np.zeros((1, 3, 26)),
            "mixture_logscale": np.zeros((1, 3, 26)),
        }

    model = CallableProposalModel(predictor, probe_condition=condition)
    discrete = model.predict_discrete(_curve_inputs())
    continuous = model.predict_continuous(_curve_inputs(), condition)
    assert discrete.topology_logits.shape == (34,)
    assert continuous.mixture_loc.shape == (3, 26)
    assert len(captured) == 2
    assert set(captured[0]) == {
        "x",
        "point_mask",
        "global_features",
        "branch_topology",
        "branch_d_present",
        "branch_resolution_present",
        "branch_low",
        "branch_high",
        "active_dimension_mask",
    }
    assert captured[0]["branch_topology"].shape == (1, 34)
    assert captured[0]["branch_d_present"].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert captured[0]["branch_resolution_present"].tolist() == [[1.0]]

    malformed = _curve_inputs()
    malformed["point_mask"] = np.ones(12, dtype=np.int32)
    with pytest.raises(ValueError, match="boolean vector"):
        build_model_inputs(malformed, condition)


class _FakeModel:
    def __init__(self):
        self.conditions = []

    def predict_discrete(self, curve_inputs):
        return _discrete()

    def predict_continuous(self, curve_inputs, condition):
        self.conditions.append(condition)
        return ContinuousProposalOutput(
            mixture_logits=np.asarray([1.0, 0.0, -1.0]),
            mixture_loc=np.zeros((3, 26)),
            mixture_logscale=np.full((3, 26), -1.0),
        )


class _BudgetedRefiner:
    def __init__(self):
        self.remaining = []

    def refine(self, proposal, *, max_forward_evaluations):
        self.remaining.append(max_forward_evaluations)
        used = min(2, max_forward_evaluations)
        if len(self.remaining) == 2:
            return RefinementOutcome(
                status="failed", forward_evaluations=used, message="local solve failed"
            )
        return RefinementOutcome(
            status="success", forward_evaluations=used, value=proposal.global_unit
        )


def test_orchestration_replenishes_unavailable_branch_and_accounts_every_budget():
    model = _FakeModel()
    refiner = _BudgetedRefiner()
    first_key = rank_joint_branches(_discrete(), topology_limit=1)[0].branch.key

    def provider(scored):
        if scored.branch.key == first_key:
            return None
        return full_range_condition(scored)

    budget = InferenceBudget(
        topology_beam_size=1,
        branch_beam_size=2,
        mixture_components_per_branch=2,
        samples_per_mixture=2,
        forward_evaluation_limit=5,
    )
    result = run_one_click_inference(
        _curve_inputs(),
        model=model,
        condition_for_branch=provider,
        budget=budget,
        seed=99,
        refiner=refiner,
    )

    assert len(result.selected_branches) == 2
    assert result.unavailable_branch_keys == (first_key,)
    assert len(result.candidates) == 8
    assert [item.proposal_rank for item in result.candidates] == list(range(1, 9))
    assert [item.status for item in result.candidates[:3]] == [
        "refined",
        "refinement_failed",
        "refined",
    ]
    assert all(item.status == "forward_budget_exhausted" for item in result.candidates[3:])
    assert refiner.remaining == [5, 3, 1]
    assert len(result.successful_refinement_values) == 2
    assert result.accounting.topology_classes_scored == 34
    assert result.accounting.valid_joint_branches_scored == 4
    assert result.accounting.branch_conditions_attempted == 3
    assert result.accounting.branch_conditions_unavailable == 1
    assert result.accounting.branches_sampled == 2
    assert result.accounting.mixture_components_returned == 6
    assert result.accounting.mixture_components_sampled == 4
    assert result.accounting.sample_budget == 8
    assert result.accounting.samples_generated == 8
    assert result.accounting.refinement_attempts == 3
    assert result.accounting.refinement_successes == 2
    assert result.accounting.refinement_failures == 1
    assert result.accounting.per_candidate_forward_evaluation_limit == 128
    assert result.accounting.forward_evaluations_used == 5
    assert result.accounting.forward_budget_exhausted_candidates == 5


def test_proposal_only_orchestration_is_reproducible():
    budget = InferenceBudget(
        topology_beam_size=1,
        branch_beam_size=1,
        mixture_components_per_branch=1,
        samples_per_mixture=3,
        forward_evaluation_limit=0,
    )

    def run():
        return run_one_click_inference(
            _curve_inputs(),
            model=_FakeModel(),
            condition_for_branch=full_range_condition,
            budget=budget,
            seed=11,
        )

    first, second = run(), run()
    assert [item.proposal.global_unit for item in first.candidates] == [
        item.proposal.global_unit for item in second.candidates
    ]
    assert all(item.status == "proposal_only" for item in first.candidates)
    assert first.accounting.forward_evaluations_used == 0

    no_solution = run_one_click_inference(
        _curve_inputs(),
        model=_FakeModel(),
        condition_for_branch=lambda scored: None,
        budget=budget,
        seed=11,
    )
    assert no_solution.status == "no_feasible_branch"
    assert no_solution.candidates == ()


def test_invalid_outputs_and_refiner_budget_overrun_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        ContinuousProposalOutput(
            mixture_logits=np.asarray([np.nan]),
            mixture_loc=np.zeros((1, 26)),
            mixture_logscale=np.zeros((1, 26)),
        )
    with pytest.raises(ValueError, match="0 <= low"):
        bad = _condition(_score())
        BranchCondition(
            scored_branch=bad.scored_branch,
            branch_low=(-0.1,) + bad.branch_low[1:],
            branch_high=bad.branch_high,
            active_dimension_mask=bad.active_dimension_mask,
        )

    class Overrun:
        def refine(self, proposal, *, max_forward_evaluations):
            return RefinementOutcome(
                status="success",
                forward_evaluations=max_forward_evaluations + 1,
                value="invalid",
            )

    with pytest.raises(RuntimeError, match="exceeded"):
        run_one_click_inference(
            _curve_inputs(),
            model=_FakeModel(),
            condition_for_branch=full_range_condition,
            budget=InferenceBudget(
                topology_beam_size=1,
                branch_beam_size=1,
                mixture_components_per_branch=1,
                samples_per_mixture=1,
                per_candidate_forward_evaluation_limit=1,
                forward_evaluation_limit=2,
            ),
            refiner=Overrun(),
        )


def test_per_candidate_cap_prevents_first_proposal_from_spending_global_budget():
    class GreedyRefiner:
        def __init__(self):
            self.allowances = []

        def refine(self, proposal, *, max_forward_evaluations):
            self.allowances.append(max_forward_evaluations)
            return RefinementOutcome(
                status="success",
                forward_evaluations=max_forward_evaluations,
                value=proposal.global_unit,
            )

    refiner = GreedyRefiner()
    result = run_one_click_inference(
        _curve_inputs(),
        model=_FakeModel(),
        condition_for_branch=full_range_condition,
        budget=InferenceBudget(
            topology_beam_size=1,
            branch_beam_size=1,
            mixture_components_per_branch=1,
            samples_per_mixture=3,
            per_candidate_forward_evaluation_limit=2,
            forward_evaluation_limit=5,
        ),
        refiner=refiner,
    )

    assert refiner.allowances == [2, 2, 1]
    assert [item.status for item in result.candidates] == ["refined"] * 3
    assert result.accounting.forward_evaluations_used == 5
