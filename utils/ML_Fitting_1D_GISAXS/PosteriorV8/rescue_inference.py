"""Neural-first production inference with retrieval and Sobol rescue seeds."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .contract import LatentComponentParameters
from .evaluation import CandidateInput, ObservedCurve
from .inference_proposals import BoundedProposalSample, ProposalModelPort
from .one_click_inference import (
    InferenceBudget,
    OneClickInferenceResult,
    run_one_click_inference,
)
from .production_bridge import (
    PhysicalRefinementSeed,
    ProductionBranchFactory,
    ProductionExactRefiner,
    physically_duplicate,
    physical_seed_from_external,
    physical_seed_from_neural,
)
from .profiled_forward import ResolutionShape
from .proposal_sampling import generate_profiled_branch_seed_at_index
from .reference_bank import CompetingBranch, enumerate_legacy_competing_branches


RESCUE_INFERENCE_VERSION = "posterior_v8_raw_candidate_rescue_v3"
AttemptSource = Literal["neural", "retrieval", "sobol"]
RawSearchStatus = Literal[
    "raw_target_reached",
    "raw_partial",
    "no_candidate_found_within_budget",
]
AttemptStatus = Literal[
    "validation_failed",
    "duplicate",
    "refined",
    "refinement_failed",
    "forward_budget_exhausted",
    "generation_failed",
]


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True, kw_only=True)
class RescuePolicy:
    target_candidate_count: int = 16
    fallback_attempt_limit: int = 128
    sobol_seeds_per_branch: int = 8
    physical_dedup_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "target_candidate_count",
            "fallback_attempt_limit",
            "sobol_seeds_per_branch",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        tolerance = float(self.physical_dedup_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("physical_dedup_tolerance must be finite and non-negative")
        object.__setattr__(self, "physical_dedup_tolerance", tolerance)


@dataclass(frozen=True, kw_only=True)
class RetrievalSeed:
    retrieval_id: str
    branch: CompetingBranch
    components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None

    def __post_init__(self) -> None:
        if not isinstance(self.retrieval_id, str) or not self.retrieval_id.strip():
            raise ValueError("retrieval_id must be non-empty")
        if not isinstance(self.branch, CompetingBranch):
            raise TypeError("branch must be CompetingBranch")
        components = tuple(self.components)
        if not all(isinstance(item, LatentComponentParameters) for item in components):
            raise TypeError("components must contain LatentComponentParameters")
        if tuple(item.shape for item in components) != self.branch.topology:
            raise ValueError("retrieval components do not match the branch topology")
        if tuple(item.log_D is not None for item in components) != self.branch.d_present:
            raise ValueError("retrieval D presence does not match the branch")
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be ResolutionShape or None")
        if (self.resolution is not None) != self.branch.resolution_present:
            raise ValueError("retrieval Resolution presence does not match the branch")
        object.__setattr__(self, "retrieval_id", self.retrieval_id.strip())
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, kw_only=True)
class CandidateAttempt:
    attempt_rank: int
    source: AttemptSource
    source_id: str
    branch_key: str
    status: AttemptStatus
    forward_evaluations: int
    candidate_id: str | None = None
    message: str | None = None


@dataclass(frozen=True, kw_only=True)
class SourceAccounting:
    """Source counts; Sobol potential is before lazy hard-core codec validation."""

    source: AttemptSource
    potential_available: int
    materialized: int
    considered: int
    validation_failures: int
    duplicates: int
    refinement_attempts: int
    refinement_failures: int
    candidates_returned: int
    forward_evaluations: int

    @property
    def available(self) -> int:
        """Compatibility alias for the former theoretical availability field."""

        return self.potential_available


@dataclass(frozen=True, kw_only=True)
class RescueAudit:
    target_candidate_count: int
    returned_candidate_count: int
    fallback_attempt_limit: int
    fallback_attempts_used: int
    per_candidate_forward_evaluation_limit: int
    forward_evaluation_limit: int
    forward_evaluations_used: int
    physical_dedup_tolerance: float
    sources: tuple[SourceAccounting, ...]
    attempts: tuple[CandidateAttempt, ...]


@dataclass(frozen=True, kw_only=True)
class RawCandidateGenerationResult:
    """Exact raw candidates; this layer makes no compatibility/solution claim."""

    status: RawSearchStatus
    candidates: tuple[CandidateInput, ...]
    neural_generation: OneClickInferenceResult
    audit: RescueAudit
    version: str = RESCUE_INFERENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != RESCUE_INFERENCE_VERSION:
            raise ValueError("unsupported rescue inference version")
        if self.status not in {
            "raw_target_reached",
            "raw_partial",
            "no_candidate_found_within_budget",
        }:
            raise ValueError("invalid raw candidate-generation status")
        ranks = tuple(item.proposal_rank for item in self.candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("returned CandidateInput ranks must be contiguous")


def _neural_source_id(proposal: BoundedProposalSample) -> str:
    branch = proposal.condition.branch
    return f"{branch.key}:mixture_{proposal.mixture_index:02d}:sample_{proposal.sample_index:03d}"


def _source_accounting(
    source: AttemptSource,
    *,
    potential_available: int,
    materialized: int,
    attempts: Sequence[CandidateAttempt],
) -> SourceAccounting:
    selected = tuple(item for item in attempts if item.source == source)
    return SourceAccounting(
        source=source,
        potential_available=potential_available,
        materialized=materialized,
        considered=sum(item.status != "generation_failed" for item in selected),
        validation_failures=sum(item.status == "validation_failed" for item in selected),
        duplicates=sum(item.status == "duplicate" for item in selected),
        refinement_attempts=sum(
            item.status in {"refined", "refinement_failed"} for item in selected
        ),
        refinement_failures=sum(item.status == "refinement_failed" for item in selected),
        candidates_returned=sum(item.status == "refined" for item in selected),
        forward_evaluations=sum(item.forward_evaluations for item in selected),
    )


def _potential_sobol_branches(
    factory: ProductionBranchFactory,
    selected_branches: Sequence[CompetingBranch],
) -> tuple[CompetingBranch, ...]:
    topology_bounds = {
        topology.topology_id: topology.component_bounds
        for topology in factory.search_space.topologies
    }
    # The V2 global proposal artifact retains the legacy 700 wire branches.
    # V3/V4 reference-bank and bounds-local paths use the canonical 418 set.
    catalog = enumerate_legacy_competing_branches(topology_ids=tuple(topology_bounds))
    allowed = []
    for branch in catalog:
        if not factory.search_space.resolution.permits(branch.resolution_present):
            continue
        bounds = topology_bounds[branch.topology_id]
        if any(
            (item.D is None and present)
            or (item.D is not None and not item.allow_D_absent and not present)
            for item, present in zip(bounds, branch.d_present)
        ):
            continue
        allowed.append(branch)
    by_key = {branch.key: branch for branch in allowed}
    priority = []
    seen = set()
    for selected in selected_branches:
        branch = by_key.get(selected.key)
        if branch is not None and branch.key not in seen:
            priority.append(branch)
            seen.add(branch.key)
    return tuple(priority + [branch for branch in allowed if branch.key not in seen])


def _iter_sobol_physical_seeds(
    factory: ProductionBranchFactory,
    branches: Sequence[CompetingBranch],
    *,
    seed: int,
    count_per_branch: int,
):
    contexts = {}
    # Sequence index is outermost so every feasible branch gets one seed before
    # any branch gets a second. Contexts and seeds are created only on demand.
    for sequence_index in range(count_per_branch):
        for branch in branches:
            if branch not in contexts:
                contexts[branch] = factory.context_for(branch)
            context = contexts[branch]
            if context is None:
                continue
            item = generate_profiled_branch_seed_at_index(
                context.user_bounds_codec,
                seed=seed,
                sequence_index=sequence_index,
            )
            yield physical_seed_from_external(
                source="sobol",
                source_id=f"{context.branch.key}:sobol_{item.sequence_index:04d}",
                context=context,
                components=item.seed_components,
                resolution=item.resolution_seed,
            )


CandidateStopPredicate = Callable[[tuple[CandidateInput, ...], int], bool]


def _run_candidate_generation(
    curve_inputs: Mapping[str, object],
    curve: ObservedCurve,
    *,
    model: ProposalModelPort,
    branch_factory: ProductionBranchFactory,
    inference_budget: InferenceBudget = InferenceBudget(),
    rescue_policy: RescuePolicy = RescuePolicy(),
    retrieval_seeds: Sequence[RetrievalSeed] = (),
    seed: int = 0,
    refiner: ProductionExactRefiner | None = None,
    stop_when: CandidateStopPredicate,
    additional_forward_evaluations_used: Callable[[], int] | None = None,
) -> RawCandidateGenerationResult:
    """Run neural proposals, then retrieval and local Sobol until target/budget."""

    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be ObservedCurve")
    if not isinstance(branch_factory, ProductionBranchFactory):
        raise TypeError("branch_factory must be ProductionBranchFactory")
    if not isinstance(inference_budget, InferenceBudget):
        raise TypeError("inference_budget must be InferenceBudget")
    if not isinstance(rescue_policy, RescuePolicy):
        raise TypeError("rescue_policy must be RescuePolicy")
    if not callable(stop_when):
        raise TypeError("stop_when must be callable")
    seed = _integer(seed, "seed")
    retrieval = tuple(retrieval_seeds)
    if not all(isinstance(item, RetrievalSeed) for item in retrieval):
        raise TypeError("retrieval_seeds must contain RetrievalSeed values")
    exact_refiner = ProductionExactRefiner(curve=curve) if refiner is None else refiner
    if not isinstance(exact_refiner, ProductionExactRefiner):
        raise TypeError("refiner must be ProductionExactRefiner")
    if exact_refiner.curve is not curve:
        raise ValueError("refiner must reference the same ObservedCurve object")

    neural = run_one_click_inference(
        curve_inputs,
        model=model,
        condition_for_branch=branch_factory.condition_for,
        budget=inference_budget,
        seed=seed,
        refiner=None,
    )
    attempts: list[CandidateAttempt] = []
    candidates: list[CandidateInput] = []
    physical_seen: list[PhysicalRefinementSeed] = []
    forward_used = 0
    fallback_attempts = 0

    def extra_forward_used() -> int:
        if additional_forward_evaluations_used is None:
            return 0
        value = additional_forward_evaluations_used()
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("additional forward usage must be an integer")
        value = int(value)
        if value < 0:
            raise ValueError("additional forward usage must be non-negative")
        return value

    def total_forward_used() -> int:
        return forward_used + extra_forward_used()

    def stop_requested() -> bool:
        result = stop_when(tuple(candidates), forward_used)
        if type(result) is not bool:
            raise TypeError("stop_when must return an explicit boolean")
        return result

    def record(
        source: AttemptSource,
        source_id: str,
        branch_key: str,
        status: AttemptStatus,
        *,
        forward: int = 0,
        candidate_id: str | None = None,
        message: str | None = None,
    ) -> None:
        attempts.append(
            CandidateAttempt(
                attempt_rank=len(attempts) + 1,
                source=source,
                source_id=source_id,
                branch_key=branch_key,
                status=status,
                forward_evaluations=forward,
                candidate_id=candidate_id,
                message=message,
            )
        )

    def process(physical: PhysicalRefinementSeed) -> bool:
        nonlocal forward_used
        if physically_duplicate(
            physical,
            physical_seen,
            tolerance=rescue_policy.physical_dedup_tolerance,
        ):
            record(
                physical.source,
                physical.source_id,
                physical.branch.key,
                "duplicate",
            )
            return False
        physical_seen.append(physical)
        remaining = inference_budget.forward_evaluation_limit - total_forward_used()
        if remaining <= 0:
            record(
                physical.source,
                physical.source_id,
                physical.branch.key,
                "forward_budget_exhausted",
            )
            return True
        allowance = min(remaining, inference_budget.per_candidate_forward_evaluation_limit)
        candidate_id = f"candidate_{len(candidates) + 1:05d}"
        outcome = exact_refiner.refine_physical(
            physical,
            candidate_id=candidate_id,
            proposal_rank=len(candidates) + 1,
            max_forward_evaluations=allowance,
        )
        forward_used += outcome.forward_evaluations
        if outcome.status == "failed":
            record(
                physical.source,
                physical.source_id,
                physical.branch.key,
                "refinement_failed",
                forward=outcome.forward_evaluations,
                message=outcome.message,
            )
            return False
        if not isinstance(outcome.value, CandidateInput):
            raise TypeError("production refiner success must return CandidateInput")
        candidates.append(outcome.value)
        record(
            physical.source,
            physical.source_id,
            physical.branch.key,
            "refined",
            forward=outcome.forward_evaluations,
            candidate_id=candidate_id,
            message=outcome.message,
        )
        return False

    for generated in neural.candidates:
        if stop_requested():
            break
        proposal = generated.proposal
        source_id = _neural_source_id(proposal)
        try:
            physical = physical_seed_from_neural(proposal)
        except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            record(
                "neural",
                source_id,
                proposal.condition.branch.key,
                "validation_failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            continue
        if process(physical):
            break

    if not stop_requested() and total_forward_used() < inference_budget.forward_evaluation_limit:
        for item in retrieval:
            if stop_requested() or fallback_attempts >= rescue_policy.fallback_attempt_limit:
                break
            fallback_attempts += 1
            context = branch_factory.context_for(item.branch)
            if context is None:
                record(
                    "retrieval",
                    item.retrieval_id,
                    item.branch.key,
                    "validation_failed",
                    message="retrieval branch is outside the user search space",
                )
                continue
            try:
                physical = physical_seed_from_external(
                    source="retrieval",
                    source_id=item.retrieval_id,
                    context=context,
                    components=item.components,
                    resolution=item.resolution,
                )
            except (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
                record(
                    "retrieval",
                    item.retrieval_id,
                    item.branch.key,
                    "validation_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
                continue
            if process(physical):
                break

    sobol_branches = _potential_sobol_branches(
        branch_factory,
        tuple(item.branch for item in neural.selected_branches),
    )
    sobol_potential = len(sobol_branches) * rescue_policy.sobol_seeds_per_branch
    sobol_materialized = 0
    if (
        not stop_requested()
        and total_forward_used() < inference_budget.forward_evaluation_limit
        and fallback_attempts < rescue_policy.fallback_attempt_limit
    ):
        sobol = iter(
            _iter_sobol_physical_seeds(
                branch_factory,
                sobol_branches,
                seed=seed,
                count_per_branch=rescue_policy.sobol_seeds_per_branch,
            )
        )
        while (
            not stop_requested()
            and fallback_attempts < rescue_policy.fallback_attempt_limit
            and total_forward_used() < inference_budget.forward_evaluation_limit
        ):
            try:
                physical = next(sobol)
            except StopIteration:
                break
            except (
                FloatingPointError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                record(
                    "sobol",
                    "sobol_generation",
                    "all_potential_branches",
                    "generation_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
                break
            sobol_materialized += 1
            fallback_attempts += 1
            if process(physical):
                break

    if len(candidates) >= rescue_policy.target_candidate_count:
        status: RawSearchStatus = "raw_target_reached"
    elif candidates:
        status = "raw_partial"
    else:
        status = "no_candidate_found_within_budget"
    source_accounting = tuple(
        _source_accounting(
            source,
            potential_available={
                "neural": len(neural.candidates),
                "retrieval": len(retrieval),
                "sobol": sobol_potential,
            }[source],
            materialized={
                "neural": len(neural.candidates),
                "retrieval": len(retrieval),
                "sobol": sobol_materialized,
            }[source],
            attempts=attempts,
        )
        for source in ("neural", "retrieval", "sobol")
    )
    return RawCandidateGenerationResult(
        status=status,
        candidates=tuple(candidates),
        neural_generation=neural,
        audit=RescueAudit(
            target_candidate_count=rescue_policy.target_candidate_count,
            returned_candidate_count=len(candidates),
            fallback_attempt_limit=rescue_policy.fallback_attempt_limit,
            fallback_attempts_used=fallback_attempts,
            per_candidate_forward_evaluation_limit=(
                inference_budget.per_candidate_forward_evaluation_limit
            ),
            forward_evaluation_limit=inference_budget.forward_evaluation_limit,
            forward_evaluations_used=forward_used,
            physical_dedup_tolerance=rescue_policy.physical_dedup_tolerance,
            sources=source_accounting,
            attempts=tuple(attempts),
        ),
    )


# Compatibility name for callers that imported the pre-v3 result class. Old
# status strings intentionally remain invalid under the bumped schema.
ProductionInferenceResult = RawCandidateGenerationResult


def run_production_inference(
    curve_inputs: Mapping[str, object],
    curve: ObservedCurve,
    *,
    model: ProposalModelPort,
    branch_factory: ProductionBranchFactory,
    inference_budget: InferenceBudget = InferenceBudget(),
    rescue_policy: RescuePolicy = RescuePolicy(),
    retrieval_seeds: Sequence[RetrievalSeed] = (),
    seed: int = 0,
    refiner: ProductionExactRefiner | None = None,
) -> RawCandidateGenerationResult:
    """Generate exact raw candidates without claiming compatibility or solutions."""

    return _run_candidate_generation(
        curve_inputs,
        curve,
        model=model,
        branch_factory=branch_factory,
        inference_budget=inference_budget,
        rescue_policy=rescue_policy,
        retrieval_seeds=retrieval_seeds,
        seed=seed,
        refiner=refiner,
        stop_when=lambda values, _raw_forward_used: (
            len(values) >= rescue_policy.target_candidate_count
        ),
    )


__all__ = [
    "RESCUE_INFERENCE_VERSION",
    "CandidateAttempt",
    "ProductionInferenceResult",
    "RawCandidateGenerationResult",
    "RawSearchStatus",
    "RescueAudit",
    "RescuePolicy",
    "RetrievalSeed",
    "SourceAccounting",
    "run_production_inference",
]
