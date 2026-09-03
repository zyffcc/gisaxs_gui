"""Production search for bounds-first V3 local-coordinate proposals.

Neural seeds, retrieval seeds, and Sobol rescue seeds meet only after they
have become validated physical seeds.  Compatibility and mode deduplication
remain exclusively owned by :func:`evaluation.evaluate_candidates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .bounds_local_inference import (
    BOUNDS_LOCAL_INFERENCE_VERSION,
    LOCAL_SAMPLE_COORDINATE_SEMANTICS,
    BoundsLocalBranchCondition,
    BoundsLocalProposalModelPort,
    LocalProposalSample,
    rank_canonical_joint_branches,
    sample_local_logistic_normal_mixture,
    validate_bounds_local_model_port,
)
from .bounds_model_contract import (
    BOUNDS_PROPOSAL_MODEL_VERSION,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
)
from .canonical_branch_catalog import canonical_branch_pattern_is_valid
from .canonical_component_slots import canonicalize_component_slots
from .evaluation import (
    CandidateInput,
    ObservedCurve,
)
from .inference_proposals import JointBranchScore
from .one_click_inference import InferenceBudget
from .production_bridge import (
    PhysicalRefinementSeed,
    ProductionBranchFactory,
    ProductionExactRefiner,
    physical_seed_from_external,
    physically_duplicate,
)
from .rescue_inference import (
    CandidateAttempt,
    RescueAudit,
    RescuePolicy,
    RetrievalSeed,
    SourceAccounting,
    _iter_sobol_physical_seeds,
    _potential_sobol_branches,
)


BOUNDS_LOCAL_RAW_SCHEMA = "gisaxs.posterior_v8.bounds_local_raw_search/v1"
RawSearchStatus = Literal[
    "raw_target_reached",
    "raw_partial",
    "no_candidate_found_within_budget",
]
_CANDIDATE_ERRORS = (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError)


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True, kw_only=True)
class LocalNeuralAccounting:
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
    samples_per_mixture: int
    samples_generated: int


@dataclass(frozen=True, kw_only=True)
class LocalNeuralGenerationResult:
    selected_branches: tuple[JointBranchScore, ...]
    proposals: tuple[LocalProposalSample, ...]
    unavailable_branch_keys: tuple[str, ...]
    accounting: LocalNeuralAccounting
    inference_version: str = BOUNDS_LOCAL_INFERENCE_VERSION
    model_version: str = BOUNDS_PROPOSAL_MODEL_VERSION
    coordinate_semantics: str = LOCAL_SAMPLE_COORDINATE_SEMANTICS
    branch_catalog_version: str = MODEL_BRANCH_CATALOG_VERSION
    component_slots_version: str = MODEL_COMPONENT_SLOTS_VERSION

    def __post_init__(self) -> None:
        if self.inference_version != BOUNDS_LOCAL_INFERENCE_VERSION:
            raise ValueError("unsupported bounds-local inference version")
        if self.model_version != BOUNDS_PROPOSAL_MODEL_VERSION:
            raise ValueError("bounds-local result has an incompatible model version")
        if self.coordinate_semantics != LOCAL_SAMPLE_COORDINATE_SEMANTICS:
            raise ValueError("bounds-local result has incompatible coordinates")
        if self.branch_catalog_version != MODEL_BRANCH_CATALOG_VERSION:
            raise ValueError("bounds-local result has an incompatible branch catalog")
        if self.component_slots_version != MODEL_COMPONENT_SLOTS_VERSION:
            raise ValueError("bounds-local result has incompatible component-slot semantics")


def generate_local_neural_proposals(
    curve_inputs: Mapping[str, object],
    *,
    model: BoundsLocalProposalModelPort,
    branch_factory: ProductionBranchFactory,
    budget: InferenceBudget,
    seed: int,
) -> LocalNeuralGenerationResult:
    """Run the curve-only discrete beam and bounds-conditioned local heads."""

    if not isinstance(curve_inputs, Mapping):
        raise TypeError("curve_inputs must be a mapping")
    validate_bounds_local_model_port(model)
    if not isinstance(branch_factory, ProductionBranchFactory):
        raise TypeError("branch_factory must be ProductionBranchFactory")
    if not isinstance(budget, InferenceBudget):
        raise TypeError("budget must be InferenceBudget")
    seed = _integer(seed, "seed")
    discrete = model.predict_discrete(curve_inputs)
    ranked = rank_canonical_joint_branches(
        discrete,
        topology_limit=budget.topology_beam_size,
    )
    selected: list[tuple[JointBranchScore, BoundsLocalBranchCondition]] = []
    unavailable: list[str] = []
    attempted = 0
    for scored in ranked:
        if len(selected) >= budget.branch_beam_size:
            break
        attempted += 1
        context = branch_factory.context_for(scored.branch)
        if context is None:
            unavailable.append(scored.branch.key)
            continue
        selected.append((scored, BoundsLocalBranchCondition.build(scored, context)))

    proposals: list[LocalProposalSample] = []
    returned_mixtures = 0
    sampled_mixtures = 0
    for _, condition in selected:
        continuous = model.predict_continuous(curve_inputs, condition)
        returned_mixtures += continuous.mixture_logits.size
        samples = sample_local_logistic_normal_mixture(
            continuous,
            condition,
            mixture_limit=budget.mixture_components_per_branch,
            samples_per_mixture=budget.samples_per_mixture,
            seed=seed,
        )
        proposals.extend(samples)
        sampled_mixtures += len({item.mixture_index for item in samples})
    return LocalNeuralGenerationResult(
        selected_branches=tuple(item[0] for item in selected),
        proposals=tuple(proposals),
        unavailable_branch_keys=tuple(unavailable),
        accounting=LocalNeuralAccounting(
            topology_classes_scored=34,
            topology_beam_limit=budget.topology_beam_size,
            topology_beam_used=len({item[0].branch.topology_id for item in selected}),
            valid_joint_branches_scored=len(ranked),
            branch_beam_limit=budget.branch_beam_size,
            branch_conditions_attempted=attempted,
            branch_conditions_unavailable=len(unavailable),
            branches_sampled=len(selected),
            mixture_limit_per_branch=budget.mixture_components_per_branch,
            mixture_components_returned=returned_mixtures,
            mixture_components_sampled=sampled_mixtures,
            samples_per_mixture=budget.samples_per_mixture,
            samples_generated=len(proposals),
        ),
    )


def physical_seed_from_local(proposal: LocalProposalSample) -> PhysicalRefinementSeed:
    """Decode directly with the user codec and enforce an exact local round trip."""

    if not isinstance(proposal, LocalProposalSample):
        raise TypeError("proposal must be LocalProposalSample")
    context = proposal.condition.context
    components, resolution = context.user_bounds_codec.decode(proposal.local_unit)
    original_local = context.user_bounds_codec.encode(components, resolution)
    if not np.allclose(
        original_local.unit_cube,
        proposal.local_unit,
        rtol=0.0,
        atol=5e-12,
    ):
        raise RuntimeError("user-bounds local proposal codec did not round-trip")
    canonical = canonicalize_component_slots(
        context.user_bounds_codec,
        components,
        resolution,
    )
    components = canonical.components
    local = canonical.coordinates
    # The global coordinate is audit-only here.  It is never interpreted as a
    # model sample and never fed back through the V1 proposal path.
    global_coordinates = context.full_domain_codec.encode(components, resolution)
    return PhysicalRefinementSeed(
        source="neural",
        source_id=(
            f"{context.branch.key}:local_mixture_{proposal.mixture_index:02d}:"
            f"sample_{proposal.sample_index:03d}"
        ),
        context=context,
        components=components,
        resolution=resolution,
        global_coordinates=global_coordinates,
        user_local_coordinates=local,
        proposal_score_raw=proposal.raw_model_log_score,
    )


def _canonical_external_seed(
    physical: PhysicalRefinementSeed,
) -> PhysicalRefinementSeed:
    canonical = canonicalize_component_slots(
        physical.context.user_bounds_codec,
        physical.components,
        physical.resolution,
    )
    components = canonical.components
    return PhysicalRefinementSeed(
        source=physical.source,
        source_id=physical.source_id,
        context=physical.context,
        components=components,
        resolution=physical.resolution,
        global_coordinates=physical.context.full_domain_codec.encode(
            components,
            physical.resolution,
        ),
        user_local_coordinates=canonical.coordinates,
        proposal_score_raw=physical.proposal_score_raw,
    )


@dataclass(frozen=True, kw_only=True)
class BoundsLocalRawResult:
    status: RawSearchStatus
    candidates: tuple[CandidateInput, ...]
    neural_generation: LocalNeuralGenerationResult
    audit: RescueAudit
    schema: str = BOUNDS_LOCAL_RAW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BOUNDS_LOCAL_RAW_SCHEMA:
            raise ValueError("unsupported bounds-local raw schema")
        if self.status not in {
            "raw_target_reached",
            "raw_partial",
            "no_candidate_found_within_budget",
        }:
            raise ValueError("invalid bounds-local raw search status")
        if not isinstance(self.neural_generation, LocalNeuralGenerationResult):
            raise TypeError("neural_generation must be LocalNeuralGenerationResult")
        if not isinstance(self.audit, RescueAudit):
            raise TypeError("audit must be RescueAudit")
        ranks = tuple(item.proposal_rank for item in self.candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("candidate proposal ranks must be contiguous from one")
        if self.audit.returned_candidate_count != len(self.candidates):
            raise ValueError("raw audit candidate count does not match candidates")
        if len(self.candidates) >= self.audit.target_candidate_count:
            expected = "raw_target_reached"
        elif self.candidates:
            expected = "raw_partial"
        else:
            expected = "no_candidate_found_within_budget"
        if self.status != expected:
            raise ValueError(f"raw status must be {expected!r}")


def _source_accounting(
    source: Literal["neural", "retrieval", "sobol"],
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


CandidateStopPredicate = Callable[[tuple[CandidateInput, ...], int], bool]


def _run_bounds_local_search(
    curve_inputs: Mapping[str, object],
    curve: ObservedCurve,
    *,
    model: BoundsLocalProposalModelPort,
    branch_factory: ProductionBranchFactory,
    inference_budget: InferenceBudget,
    rescue_policy: RescuePolicy,
    retrieval_seeds: Sequence[RetrievalSeed],
    seed: int,
    refiner: ProductionExactRefiner | None,
    stop_when: CandidateStopPredicate,
    additional_forward_evaluations_used: Callable[[], int] | None = None,
) -> BoundsLocalRawResult:
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
    neural = generate_local_neural_proposals(
        curve_inputs,
        model=model,
        branch_factory=branch_factory,
        budget=inference_budget,
        seed=seed,
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
        result = int(value)
        if result < 0:
            raise ValueError("additional forward usage must be non-negative")
        return result

    def total_forward_used() -> int:
        return forward_used + extra_forward_used()

    def stop_requested() -> bool:
        result = stop_when(tuple(candidates), forward_used)
        if type(result) is not bool:
            raise TypeError("stop_when must return an explicit boolean")
        return result

    def record(physical, status, *, forward=0, candidate_id=None, message=None):
        attempts.append(
            CandidateAttempt(
                attempt_rank=len(attempts) + 1,
                source=physical.source,
                source_id=physical.source_id,
                branch_key=physical.branch.key,
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
            record(physical, "duplicate")
            return False
        physical_seen.append(physical)
        remaining = inference_budget.forward_evaluation_limit - total_forward_used()
        if remaining <= 0:
            record(physical, "forward_budget_exhausted")
            return True
        allowance = min(remaining, inference_budget.per_candidate_forward_evaluation_limit)
        candidate_id = f"candidate_{len(candidates) + 1:05d}"
        outcome = exact_refiner.refine_physical(
            physical,
            candidate_id=candidate_id,
            proposal_rank=len(candidates) + 1,
            max_forward_evaluations=allowance,
        )
        if outcome.forward_evaluations > allowance:
            raise RuntimeError("exact refiner exceeded its per-candidate forward budget")
        forward_used += outcome.forward_evaluations
        if outcome.status == "failed":
            record(
                physical,
                "refinement_failed",
                forward=outcome.forward_evaluations,
                message=outcome.message,
            )
            return False
        if not isinstance(outcome.value, CandidateInput):
            raise TypeError("production refiner success must return CandidateInput")
        candidates.append(outcome.value)
        record(
            physical,
            "refined",
            forward=outcome.forward_evaluations,
            candidate_id=candidate_id,
            message=outcome.message,
        )
        return False

    for proposal in neural.proposals:
        if stop_requested():
            break
        try:
            physical = physical_seed_from_local(proposal)
        except _CANDIDATE_ERRORS as exc:
            source_id = (
                f"{proposal.condition.branch.key}:local_mixture_"
                f"{proposal.mixture_index:02d}:sample_{proposal.sample_index:03d}"
            )
            attempts.append(
                CandidateAttempt(
                    attempt_rank=len(attempts) + 1,
                    source="neural",
                    source_id=source_id,
                    branch_key=proposal.condition.branch.key,
                    status="validation_failed",
                    forward_evaluations=0,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if process(physical):
            break

    if not stop_requested() and total_forward_used() < inference_budget.forward_evaluation_limit:
        for item in retrieval:
            if stop_requested() or fallback_attempts >= rescue_policy.fallback_attempt_limit:
                break
            fallback_attempts += 1
            if not canonical_branch_pattern_is_valid(
                item.branch.topology_id,
                item.branch.pattern_id,
            ):
                attempts.append(
                    CandidateAttempt(
                        attempt_rank=len(attempts) + 1,
                        source="retrieval",
                        source_id=item.retrieval_id,
                        branch_key=item.branch.key,
                        status="validation_failed",
                        forward_evaluations=0,
                        message="retrieval branch is noncanonical for V3/V4",
                    )
                )
                continue
            context = branch_factory.context_for(item.branch)
            if context is None:
                attempts.append(
                    CandidateAttempt(
                        attempt_rank=len(attempts) + 1,
                        source="retrieval",
                        source_id=item.retrieval_id,
                        branch_key=item.branch.key,
                        status="validation_failed",
                        forward_evaluations=0,
                        message="retrieval branch is outside the user search space",
                    )
                )
                continue
            try:
                canonical = canonicalize_component_slots(
                    context.user_bounds_codec,
                    item.components,
                    item.resolution,
                )
                physical = physical_seed_from_external(
                    source="retrieval",
                    source_id=item.retrieval_id,
                    context=context,
                    components=canonical.components,
                    resolution=item.resolution,
                )
            except _CANDIDATE_ERRORS as exc:
                attempts.append(
                    CandidateAttempt(
                        attempt_rank=len(attempts) + 1,
                        source="retrieval",
                        source_id=item.retrieval_id,
                        branch_key=item.branch.key,
                        status="validation_failed",
                        forward_evaluations=0,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if process(physical):
                break

    branches = tuple(
        item
        for item in _potential_sobol_branches(
            branch_factory,
            tuple(item.branch for item in neural.selected_branches),
        )
        if canonical_branch_pattern_is_valid(item.topology_id, item.pattern_id)
    )
    sobol_potential = len(branches) * rescue_policy.sobol_seeds_per_branch
    sobol_materialized = 0
    if (
        not stop_requested()
        and total_forward_used() < inference_budget.forward_evaluation_limit
        and fallback_attempts < rescue_policy.fallback_attempt_limit
    ):
        sobol = iter(
            _iter_sobol_physical_seeds(
                branch_factory,
                branches,
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
                physical = _canonical_external_seed(next(sobol))
            except StopIteration:
                break
            except _CANDIDATE_ERRORS as exc:
                attempts.append(
                    CandidateAttempt(
                        attempt_rank=len(attempts) + 1,
                        source="sobol",
                        source_id="sobol_generation",
                        branch_key="all_potential_branches",
                        status="generation_failed",
                        forward_evaluations=0,
                        message=f"{type(exc).__name__}: {exc}",
                    )
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
    source_accounts = tuple(
        _source_accounting(
            source,
            potential_available={
                "neural": len(neural.proposals),
                "retrieval": len(retrieval),
                "sobol": sobol_potential,
            }[source],
            materialized={
                "neural": len(neural.proposals),
                "retrieval": len(retrieval),
                "sobol": sobol_materialized,
            }[source],
            attempts=attempts,
        )
        for source in ("neural", "retrieval", "sobol")
    )
    return BoundsLocalRawResult(
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
            sources=source_accounts,
            attempts=tuple(attempts),
        ),
    )


def run_bounds_local_raw_inference(
    curve_inputs: Mapping[str, object],
    curve: ObservedCurve,
    *,
    model: BoundsLocalProposalModelPort,
    branch_factory: ProductionBranchFactory,
    inference_budget: InferenceBudget = InferenceBudget(),
    rescue_policy: RescuePolicy = RescuePolicy(),
    retrieval_seeds: Sequence[RetrievalSeed] = (),
    seed: int = 0,
    refiner: ProductionExactRefiner | None = None,
) -> BoundsLocalRawResult:
    """Return raw exact candidates without making a compatibility claim."""

    return _run_bounds_local_search(
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
    "BOUNDS_LOCAL_RAW_SCHEMA",
    "BoundsLocalRawResult",
    "LocalNeuralAccounting",
    "LocalNeuralGenerationResult",
    "RawSearchStatus",
    "generate_local_neural_proposals",
    "physical_seed_from_local",
    "run_bounds_local_raw_inference",
]
