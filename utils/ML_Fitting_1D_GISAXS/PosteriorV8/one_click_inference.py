"""Budgeted orchestration for Posterior V8 neural multi-candidate proposals.

This module deliberately stops at an injectable refinement port.  The port is
the seam for the authoritative full-domain decode, user-bound validation,
``refine_profiled_branch`` and optional amplitude polish.  Successful exact
candidates can then be passed unchanged to ``evaluation.evaluate_candidates``
for compatibility ranking and complete-linkage deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Literal, Mapping, Protocol

import numpy as np

from .inference_proposals import (
    INFERENCE_PROPOSAL_VERSION,
    BoundedProposalSample,
    BranchCondition,
    JointBranchScore,
    ProposalModelPort,
    rank_joint_branches,
    sample_bounded_global_mixture,
)


ONE_CLICK_INFERENCE_VERSION = "posterior_v8_budgeted_one_click_inference_v1"
RefinementStatus = Literal["success", "failed"]
CandidateStatus = Literal[
    "proposal_only", "refined", "refinement_failed", "forward_budget_exhausted"
]


def _integer(value: int, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        qualifier = f">= {minimum}"
        if maximum is not None:
            qualifier += f" and <= {maximum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True, kw_only=True)
class InferenceBudget:
    """Explicit deterministic budgets for every proposal stage."""

    topology_beam_size: int = 8
    branch_beam_size: int = 32
    mixture_components_per_branch: int = 4
    samples_per_mixture: int = 2
    per_candidate_forward_evaluation_limit: int = 128
    forward_evaluation_limit: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topology_beam_size",
            _integer(self.topology_beam_size, "topology_beam_size", minimum=1, maximum=34),
        )
        for name in (
            "branch_beam_size",
            "mixture_components_per_branch",
            "samples_per_mixture",
            "per_candidate_forward_evaluation_limit",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        object.__setattr__(
            self,
            "forward_evaluation_limit",
            _integer(self.forward_evaluation_limit, "forward_evaluation_limit"),
        )

    @property
    def maximum_sample_count(self) -> int:
        return self.branch_beam_size * self.mixture_components_per_branch * self.samples_per_mixture


@dataclass(frozen=True, kw_only=True)
class RefinementOutcome:
    """Result returned by an exact-refinement adapter for one proposal.

    The production success payload should be an ``evaluation.CandidateInput``
    built from GUI-consistent exact intensity.  Its opaque proposal score is
    ``BoundedProposalSample.raw_model_log_score``; scientific ranking remains
    the responsibility of ``evaluation.evaluate_candidates``.
    """

    status: RefinementStatus
    forward_evaluations: int
    value: object | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed"}:
            raise ValueError("refinement status must be success or failed")
        object.__setattr__(
            self,
            "forward_evaluations",
            _integer(self.forward_evaluations, "forward_evaluations"),
        )
        if self.status == "success" and self.value is None:
            raise ValueError("successful refinement must return a value")
        if self.status == "failed" and self.value is not None:
            raise ValueError("failed refinement must not return a value")
        if self.message is not None and (
            not isinstance(self.message, str) or not self.message.strip()
        ):
            raise ValueError("refinement message must be non-empty or None")


class ExactRefinerPort(Protocol):
    """Adapter for full decode -> exact refine -> optional amplitude polish.

    The adapter must account for every authoritative forward evaluation and
    must never use more than ``max_forward_evaluations``.
    """

    def refine(
        self,
        proposal: BoundedProposalSample,
        *,
        max_forward_evaluations: int,
    ) -> RefinementOutcome: ...


@dataclass(frozen=True, kw_only=True)
class InferenceCandidate:
    proposal_rank: int
    proposal: BoundedProposalSample
    status: CandidateStatus
    refinement: RefinementOutcome | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_rank", _integer(self.proposal_rank, "proposal_rank", minimum=1)
        )
        if not isinstance(self.proposal, BoundedProposalSample):
            raise TypeError("proposal must be a BoundedProposalSample")
        allowed = {
            "proposal_only",
            "refined",
            "refinement_failed",
            "forward_budget_exhausted",
        }
        if self.status not in allowed:
            raise ValueError("unknown candidate status")
        expected = {
            "proposal_only": None,
            "forward_budget_exhausted": None,
            "refined": "success",
            "refinement_failed": "failed",
        }[self.status]
        if expected is None and self.refinement is not None:
            raise ValueError(f"{self.status} candidate must not have a refinement")
        if expected is not None and (self.refinement is None or self.refinement.status != expected):
            raise ValueError(f"{self.status} candidate has inconsistent refinement")


@dataclass(frozen=True, kw_only=True)
class InferenceAccounting:
    topology_classes_scored: int
    topology_beam_limit: int
    topology_beam_used: int
    valid_joint_branches_scored: int
    branch_beam_limit: int
    branch_conditions_attempted: int
    branch_conditions_unavailable: int
    branches_sampled: int
    mixture_limit_per_branch: int
    mixture_components_returned: int
    mixture_components_sampled: int
    sample_budget: int
    samples_generated: int
    refinement_attempts: int
    refinement_successes: int
    refinement_failures: int
    per_candidate_forward_evaluation_limit: int
    forward_evaluation_limit: int
    forward_evaluations_used: int
    forward_budget_exhausted_candidates: int


@dataclass(frozen=True, kw_only=True)
class OneClickInferenceResult:
    selected_branches: tuple[JointBranchScore, ...]
    candidates: tuple[InferenceCandidate, ...]
    unavailable_branch_keys: tuple[str, ...]
    accounting: InferenceAccounting
    inference_version: str = ONE_CLICK_INFERENCE_VERSION
    proposal_version: str = INFERENCE_PROPOSAL_VERSION

    def __post_init__(self) -> None:
        if self.inference_version != ONE_CLICK_INFERENCE_VERSION:
            raise ValueError("unsupported one-click inference version")
        if self.proposal_version != INFERENCE_PROPOSAL_VERSION:
            raise ValueError("unsupported inference proposal version")
        ranks = tuple(item.proposal_rank for item in self.candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("candidate proposal ranks must be contiguous from one")

    @property
    def successful_refinement_values(self) -> tuple[object, ...]:
        """Payloads ready for exact compatibility ranking and deduplication."""

        return tuple(
            item.refinement.value
            for item in self.candidates
            if item.status == "refined" and item.refinement is not None
        )

    @property
    def status(self) -> Literal["candidates_generated", "no_feasible_branch"]:
        return "candidates_generated" if self.candidates else "no_feasible_branch"


BranchConditionProvider = Callable[[JointBranchScore], BranchCondition | None]


def run_one_click_inference(
    curve_inputs: Mapping[str, object],
    *,
    model: ProposalModelPort,
    condition_for_branch: BranchConditionProvider,
    budget: InferenceBudget = InferenceBudget(),
    seed: int = 0,
    refiner: ExactRefinerPort | None = None,
) -> OneClickInferenceResult:
    """Generate bounded, diverse proposals and optionally refine under one budget.

    ``condition_for_branch`` returns ``None`` for a branch made infeasible by
    user D/Resolution policy or physical bounds.  Other provider/model errors
    fail closed instead of being silently converted into unavailable branches.
    """

    if not isinstance(curve_inputs, Mapping):
        raise TypeError("curve_inputs must be a mapping")
    if not callable(getattr(model, "predict_discrete", None)) or not callable(
        getattr(model, "predict_continuous", None)
    ):
        raise TypeError("model must implement the ProposalModelPort")
    if not callable(condition_for_branch):
        raise TypeError("condition_for_branch must be callable")
    if not isinstance(budget, InferenceBudget):
        raise TypeError("budget must be an InferenceBudget")
    seed = _integer(seed, "seed")
    if refiner is not None and not callable(getattr(refiner, "refine", None)):
        raise TypeError("refiner must implement ExactRefinerPort")

    discrete = model.predict_discrete(curve_inputs)
    ranked = rank_joint_branches(discrete, topology_limit=budget.topology_beam_size)
    selected_conditions = []
    unavailable = []
    conditions_attempted = 0
    for scored_branch in ranked:
        if len(selected_conditions) >= budget.branch_beam_size:
            break
        conditions_attempted += 1
        condition = condition_for_branch(scored_branch)
        if condition is None:
            unavailable.append(scored_branch.branch.key)
            continue
        if not isinstance(condition, BranchCondition):
            raise TypeError("condition_for_branch must return BranchCondition or None")
        if condition.scored_branch != scored_branch:
            raise ValueError("condition provider returned a different scored branch")
        selected_conditions.append(condition)

    proposals = []
    mixture_returned = 0
    mixture_sampled = 0
    for condition in selected_conditions:
        continuous = model.predict_continuous(curve_inputs, condition)
        mixture_returned += continuous.mixture_logits.size
        sampled = sample_bounded_global_mixture(
            continuous,
            condition,
            mixture_limit=budget.mixture_components_per_branch,
            samples_per_mixture=budget.samples_per_mixture,
            seed=seed,
        )
        proposals.extend(sampled)
        mixture_sampled += len({item.mixture_index for item in sampled})

    candidates = []
    forward_used = 0
    attempts = successes = failures = exhausted = 0
    for proposal_rank, proposal in enumerate(proposals, 1):
        if refiner is None:
            candidates.append(
                InferenceCandidate(
                    proposal_rank=proposal_rank,
                    proposal=proposal,
                    status="proposal_only",
                )
            )
            continue
        total_remaining = budget.forward_evaluation_limit - forward_used
        if total_remaining <= 0:
            exhausted += 1
            candidates.append(
                InferenceCandidate(
                    proposal_rank=proposal_rank,
                    proposal=proposal,
                    status="forward_budget_exhausted",
                )
            )
            continue
        candidate_allowance = min(
            total_remaining,
            budget.per_candidate_forward_evaluation_limit,
        )
        outcome = refiner.refine(
            proposal,
            max_forward_evaluations=candidate_allowance,
        )
        if not isinstance(outcome, RefinementOutcome):
            raise TypeError("refiner must return RefinementOutcome")
        if outcome.forward_evaluations > candidate_allowance:
            raise RuntimeError("refiner exceeded its forward-evaluation allowance")
        forward_used += outcome.forward_evaluations
        attempts += 1
        if outcome.status == "success":
            successes += 1
            status: CandidateStatus = "refined"
        else:
            failures += 1
            status = "refinement_failed"
        candidates.append(
            InferenceCandidate(
                proposal_rank=proposal_rank,
                proposal=proposal,
                status=status,
                refinement=outcome,
            )
        )

    topology_used = len({item.branch.topology_id for item in ranked})
    accounting = InferenceAccounting(
        topology_classes_scored=int(discrete.topology_logits.size),
        topology_beam_limit=budget.topology_beam_size,
        topology_beam_used=topology_used,
        valid_joint_branches_scored=len(ranked),
        branch_beam_limit=budget.branch_beam_size,
        branch_conditions_attempted=conditions_attempted,
        branch_conditions_unavailable=len(unavailable),
        branches_sampled=len(selected_conditions),
        mixture_limit_per_branch=budget.mixture_components_per_branch,
        mixture_components_returned=mixture_returned,
        mixture_components_sampled=mixture_sampled,
        sample_budget=budget.maximum_sample_count,
        samples_generated=len(proposals),
        refinement_attempts=attempts,
        refinement_successes=successes,
        refinement_failures=failures,
        per_candidate_forward_evaluation_limit=(
            budget.per_candidate_forward_evaluation_limit
        ),
        forward_evaluation_limit=budget.forward_evaluation_limit,
        forward_evaluations_used=forward_used,
        forward_budget_exhausted_candidates=exhausted,
    )
    return OneClickInferenceResult(
        selected_branches=tuple(item.scored_branch for item in selected_conditions),
        candidates=tuple(candidates),
        unavailable_branch_keys=tuple(unavailable),
        accounting=accounting,
    )


__all__ = [
    "ONE_CLICK_INFERENCE_VERSION",
    "BranchConditionProvider",
    "ExactRefinerPort",
    "InferenceAccounting",
    "InferenceBudget",
    "InferenceCandidate",
    "OneClickInferenceResult",
    "RefinementOutcome",
    "run_one_click_inference",
]
