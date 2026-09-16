"""Universal V5.2 neural proposals with exact verification and rescue.

The model is called once over the complete selected contextual catalog.  Its
scores determine only a frozen search order.  Measurement compatibility,
complete-linkage parameter-cluster representatives, and cross-topology curve
equivalence remain owned by the query-bound evaluator after exact GUI-forward
refinement.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .candidate_proposals_v5 import (
    V5BatchedProposalOutput,
    V5LocalProposal,
    sample_v5_local_proposals,
)
from .candidate_refinement_v5 import (
    V5LocalRefinementSeed,
    run_v5_exact_refinement,
    v5_external_local_refinement_seed,
    v5_refinement_seed_from_proposal,
)
from .evaluation import (
    CandidateInput,
    EvaluationReport,
    EvaluationThresholds,
    ObservedCurve,
    ReferenceMode,
)
from .proposal_sampling import generate_profiled_branch_seed_at_index
from .proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
)
from .query_bound_evaluation_v5 import evaluate_v5_query_bound_candidates
from .universal_inference_contract_v5 import (
    V5AttemptStage,
    V5TerminationReason,
    V5UniversalAttemptAudit,
    V5UniversalCandidateProvenance,
    V5UniversalInferenceBudget,
    V5UniversalInferenceResult,
    V5UniversalSourceAccounting,
    V5UniversalStatus,
)
from .universal_query_v5 import (
    V5UniversalCandidateContext,
    validate_v5_universal_curve_alignment,
)


_SEED_ERRORS = (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError)


class V5UniversalProposalModelPort(Protocol):
    """Keras-compatible callable used exactly once for a universal query."""

    def __call__(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        training: bool,
    ) -> Mapping[str, object]: ...


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _evaluation_cutoffs(configured: Sequence[int], count: int) -> tuple[int, ...]:
    if isinstance(configured, (str, bytes)):
        raise TypeError("best_of_n must be a sequence of positive integers")
    values = tuple(configured)
    if not values or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError("best_of_n must contain positive integers")
    integers = tuple(int(value) for value in values)
    if any(value < 1 for value in integers) or len(set(integers)) != len(integers):
        raise ValueError("best_of_n must contain unique positive integers")
    return tuple(sorted({value for value in integers if value <= count} | {count}))


def _validate_retrieval_context(
    context: V5UniversalCandidateContext,
    seed: V5LocalRefinementSeed,
) -> None:
    matches = tuple(
        value
        for value in context.branches
        if (value.global_key.topology_id, value.global_key.pattern_id)
        == (seed.topology_id, seed.pattern_id)
    )
    if len(matches) != 1:
        raise ValueError("retrieval seed branch is outside the universal query")
    branch = matches[0]
    batch = context.batches[branch.topology_batch_index]
    if seed.query_sha256 != batch.query_sha256:
        raise ValueError("stale retrieval seed belongs to a different complete user query")
    if seed.branch_batch_index != branch.branch_batch_index:
        raise ValueError("stale retrieval seed has a different contextual branch row")
    condition = batch.branch_conditions[seed.branch_batch_index]
    if (condition.topology_id, condition.pattern_id) != (seed.topology_id, seed.pattern_id):
        raise ValueError("retrieval seed identity disagrees with its contextual branch")
    codec = batch.query.codec_for(seed.pattern_id)
    encoded = codec.encode(seed.latent_components, seed.resolution)
    if not np.allclose(encoded.unit_cube, seed.local_unit, rtol=0.0, atol=5.0e-12):
        raise ValueError("retrieval seed does not round-trip through the complete user query")


def _global_neural_proposals(
    context: V5UniversalCandidateContext,
    parsed: V5BatchedProposalOutput,
    *,
    budget: V5UniversalInferenceBudget,
    seed: int,
) -> tuple[tuple[V5LocalProposal, ...], tuple[int, ...]]:
    ranked_indices = tuple(
        sorted(
            range(context.branch_count),
            key=lambda index: (
                -float(parsed.search_yield_logit[index]),
                context.branches[index].global_key,
            ),
        )
    )
    rank_by_key = {
        context.branches[index].global_key: rank for rank, index in enumerate(ranked_indices, 1)
    }
    proposals: list[V5LocalProposal] = []
    for batch, batch_slice in zip(context.batches, context.batch_slices):
        local = V5BatchedProposalOutput(
            search_yield_logit=parsed.search_yield_logit[batch_slice],
            mixture_logits=parsed.mixture_logits[batch_slice],
            mixture_loc=parsed.mixture_loc[batch_slice],
            mixture_logscale=parsed.mixture_logscale[batch_slice],
        )
        sampled = sample_v5_local_proposals(
            batch,
            local,
            mixture_limit=budget.mixture_components_per_branch,
            stochastic_draws_per_mixture=budget.stochastic_draws_per_mixture,
            include_mixture_medians=budget.include_mixture_medians,
            seed=seed,
        )
        proposals.extend(
            replace(
                value,
                branch_rank=rank_by_key[
                    context.branches[batch_slice.start + value.branch_batch_index].global_key
                ],
            )
            for value in sampled
        )
    return _branch_round_robin_proposals(proposals), ranked_indices


def _branch_round_robin_proposals(
    proposals: Sequence[V5LocalProposal],
) -> tuple[V5LocalProposal, ...]:
    """Interleave branches before spending a second neural seed on any branch.

    Search-yield logits rank *branches*, while mixture weights rank conditional
    seeds inside one branch.  Comparing those two score scales lexicographically
    let a high-ranked branch consume the entire small-N proposal budget.  The
    paper/product schedule therefore visits each ranked branch once per round;
    within a branch it tries all mixture medians before stochastic draws.
    """

    grouped: dict[tuple[int, int], list[V5LocalProposal]] = {}
    for proposal in proposals:
        grouped.setdefault((proposal.topology_id, proposal.pattern_id), []).append(proposal)
    if not grouped:
        return ()
    ordered_groups = sorted(
        grouped.values(),
        key=lambda values: (
            values[0].branch_rank,
            values[0].topology_id,
            values[0].pattern_id,
        ),
    )
    for values in ordered_groups:
        branch_ranks = {value.branch_rank for value in values}
        if len(branch_ranks) != 1:
            raise ValueError("one contextual branch has inconsistent global ranks")
        values.sort(
            key=lambda value: (
                value.draw_index,
                value.mixture_rank,
                value.mixture_index,
            )
        )
    scheduled = []
    for round_index in range(max(len(values) for values in ordered_groups)):
        scheduled.extend(
            values[round_index] for values in ordered_groups if round_index < len(values)
        )
    return tuple(scheduled)


def _neural_seed(context: V5UniversalCandidateContext, proposal: V5LocalProposal):
    seed = v5_refinement_seed_from_proposal(proposal)
    branch = context.branches[
        context.batch_slices[
            next(
                index
                for index, batch in enumerate(context.batches)
                if batch.query.topology_id == proposal.topology_id
            )
        ].start
        + proposal.branch_batch_index
    ]
    return replace(seed, source_id=f"{branch.global_key.wire_key}:{seed.source_id}")


def _sobol_seeds(
    context: V5UniversalCandidateContext,
    ranked_indices: Sequence[int],
    *,
    generator_seed: int,
    count_per_branch: int,
):
    for sequence_index in range(count_per_branch):
        for global_index in ranked_indices:
            branch = context.branches[global_index]
            batch = context.batches[branch.topology_batch_index]
            generated = generate_profiled_branch_seed_at_index(
                batch.query.codec_for(branch.global_key.pattern_id),
                seed=generator_seed,
                sequence_index=sequence_index,
            )
            yield v5_external_local_refinement_seed(
                batch,
                source="sobol",
                source_id=(f"{branch.global_key.wire_key}:sobol_{sequence_index:04d}"),
                pattern_id=branch.global_key.pattern_id,
                local_unit=generated.unit_cube,
            )


class V5VerifiedReportObservationError(Exception):
    """A report audit sink failed; search must not silently omit evidence."""


class V5EvaluatedCandidateObservationError(Exception):
    """A lossless candidate audit failed; do not continue with incomplete evidence."""


def run_v5_universal_one_click_inference(
    context: V5UniversalCandidateContext,
    curve: ObservedCurve,
    *,
    model: V5UniversalProposalModelPort,
    thresholds: EvaluationThresholds,
    target_parameter_mode_count: int,
    budget: V5UniversalInferenceBudget = V5UniversalInferenceBudget(),
    retrieval_seeds: Sequence[V5LocalRefinementSeed] = (),
    best_of_n: Sequence[int] = (1, 4, 8, 16),
    reference_modes: Sequence[ReferenceMode] = (),
    seed: int = 0,
    exact_forward_call_observer: Callable[[int, str], None] | None = None,
    verified_report_observer: Callable[[int, EvaluationReport], None] | None = None,
    evaluated_candidate_observer: Callable[
        [int, CandidateInput, V5UniversalCandidateProvenance, EvaluationReport], None
    ] | None = None,
) -> V5UniversalInferenceResult:
    """Return verified modes; optional call observations use global budget indices."""

    if exact_forward_call_observer is not None and not callable(exact_forward_call_observer):
        raise TypeError("exact_forward_call_observer must be callable or None")
    if verified_report_observer is not None and not callable(verified_report_observer):
        raise TypeError("verified_report_observer must be callable or None")
    if evaluated_candidate_observer is not None and not callable(evaluated_candidate_observer):
        raise TypeError("evaluated_candidate_observer must be callable or None")
    if not isinstance(context, V5UniversalCandidateContext):
        raise TypeError("context must be a V5UniversalCandidateContext")
    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be an ObservedCurve")
    if not callable(model):
        raise TypeError("model must be a callable V5 proposal model")
    if not isinstance(thresholds, EvaluationThresholds):
        raise TypeError("thresholds must be EvaluationThresholds")
    if not isinstance(budget, V5UniversalInferenceBudget):
        raise TypeError("budget must be V5UniversalInferenceBudget")
    target = _nonnegative_integer(target_parameter_mode_count, "target_parameter_mode_count")
    if target < 1:
        raise ValueError("target_parameter_mode_count must be positive")
    seed_value = _nonnegative_integer(seed, "seed")
    _evaluation_cutoffs(best_of_n, 1)
    references = tuple(reference_modes)
    if not all(isinstance(value, ReferenceMode) for value in references):
        raise TypeError("reference_modes must contain ReferenceMode values")
    if len({value.reference_id for value in references}) != len(references):
        raise ValueError("reference_modes must use unique reference_id values")
    retrieval = tuple(retrieval_seeds)
    if not all(
        isinstance(value, V5LocalRefinementSeed) and value.source == "retrieval"
        for value in retrieval
    ):
        raise TypeError("retrieval_seeds must contain retrieval V5LocalRefinementSeed values")
    for value in retrieval:
        _validate_retrieval_context(context, value)
    validate_v5_universal_curve_alignment(context, curve)

    outputs = model(context.for_model(), training=False)
    parsed = V5BatchedProposalOutput.from_mapping(outputs, branch_count=context.branch_count)
    if parsed.mixture_count != V5_PROPOSAL_EXECUTION_POLICY.mixture_component_count:
        raise ValueError(
            "model mixture count does not match the frozen proposal execution policy"
        )
    proposals, ranked_indices = _global_neural_proposals(
        context, parsed, budget=budget, seed=seed_value
    )
    neural = tuple(_neural_seed(context, value) for value in proposals)
    branches = {
        (value.global_key.topology_id, value.global_key.pattern_id): value
        for value in context.branches
    }

    attempts: list[V5UniversalAttemptAudit] = []
    candidates: list[CandidateInput] = []
    provenance: list[V5UniversalCandidateProvenance] = []
    report: EvaluationReport | None = None
    forward_used = 0
    evaluation_calls = 0
    sobol_materialized = 0

    def process(seed_value_: V5LocalRefinementSeed, stage: V5AttemptStage):
        nonlocal forward_used, report, evaluation_calls
        remaining = budget.forward_evaluation_limit - forward_used
        if remaining <= 0:
            return "budget"
        branch = branches[(seed_value_.topology_id, seed_value_.pattern_id)]
        batch = context.batches[branch.topology_batch_index]
        observation_arguments = {}
        if exact_forward_call_observer is not None:
            offset = forward_used

            def observe(local_index: int, phase: str) -> None:
                exact_forward_call_observer(offset + local_index, phase)

            observation_arguments["exact_forward_call_observer"] = observe
        exact = run_v5_exact_refinement(
            batch,
            curve,
            (seed_value_,),
            per_candidate_forward_evaluation_limit=(budget.per_candidate_forward_evaluation_limit),
            forward_evaluation_limit=remaining,
            **observation_arguments,
        )
        if len(exact.attempts) != 1:
            raise RuntimeError("single-seed exact refinement returned the wrong attempt count")
        item = exact.attempts[0]
        before = forward_used
        forward_used += exact.ledger.calls_used
        candidate_id = None
        if item.status == "refined":
            if len(exact.candidates) != 1:
                raise RuntimeError("refined exact attempt did not return one CandidateInput")
            candidate_id = f"candidate_{len(candidates) + 1:05d}"
            candidate = replace(
                exact.candidates[0],
                candidate_id=candidate_id,
                proposal_rank=len(candidates) + 1,
            )
            candidates.append(candidate)
            provenance.append(
                V5UniversalCandidateProvenance(
                    candidate_id=candidate_id,
                    proposal_rank=candidate.proposal_rank,
                    attempt_rank=len(attempts) + 1,
                    stage=stage,
                    source=seed_value_.source,
                    source_id=seed_value_.source_id,
                    global_branch_key=branch.global_key.wire_key,
                    topology_id=seed_value_.topology_id,
                    pattern_id=seed_value_.pattern_id,
                    model_global_index=branch.global_index,
                    search_yield_logit=seed_value_.search_yield_logit,
                    mixture_log_weight=seed_value_.mixture_log_weight,
                )
            )
            report = evaluate_v5_query_bound_candidates(
                context,
                curve,
                candidates,
                candidate_provenance=provenance,
                thresholds=thresholds,
                best_of_n=_evaluation_cutoffs(best_of_n, len(candidates)),
                reference_modes=references,
            )
            evaluation_calls += 1
            # Capture the actual lossless input and branch provenance at birth,
            # before a later report can replace its user-visible representative.
            if evaluated_candidate_observer is not None:
                try:
                    evaluated_candidate_observer(forward_used, candidate, provenance[-1], report)
                except Exception as exc:
                    raise V5EvaluatedCandidateObservationError("candidate observer failed") from exc
            # Observe this immutable report when it actually becomes available,
            # before later candidates can change clustering or representative ranks.
            if verified_report_observer is not None:
                try:
                    verified_report_observer(forward_used, report)
                except Exception as exc:
                    raise V5VerifiedReportObservationError("verified-report observer failed") from exc
        attempts.append(
            V5UniversalAttemptAudit(
                attempt_rank=len(attempts) + 1,
                stage=stage,
                source=seed_value_.source,
                source_id=seed_value_.source_id,
                global_branch_key=branch.global_key.wire_key,
                model_global_index=branch.global_index,
                topology_id=seed_value_.topology_id,
                pattern_id=seed_value_.pattern_id,
                exact_status=item.status,
                exact_forward_calls=exact.ledger.calls_used,
                cumulative_calls_before=before,
                cumulative_calls_after=forward_used,
                candidate_id=candidate_id,
                message=item.message,
                search_yield_logit=item.search_yield_logit,
                mixture_log_weight=item.mixture_log_weight,
            )
        )
        if report is not None and len(report.parameter_modes) >= target:
            return "target"
        return "budget" if forward_used >= budget.forward_evaluation_limit else None

    reason: V5TerminationReason = "seed_schedule_exhausted"
    primary_count = min(budget.neural_primary_attempt_limit, len(neural))
    for value in neural[:primary_count]:
        stopped = process(value, "neural_primary")
        if stopped:
            reason = (
                "compatible_target_reached"
                if stopped == "target"
                else "exact_forward_budget_exhausted"
            )
            break
    else:
        stopped = None

    fallback_used = 0
    if stopped is None and forward_used >= budget.forward_evaluation_limit:
        stopped = "budget"
        reason = "exact_forward_budget_exhausted"
    if stopped is None:
        for value in retrieval:
            if fallback_used >= budget.fallback_attempt_limit:
                break
            fallback_used += 1
            stopped = process(value, "retrieval_rescue")
            if stopped:
                reason = (
                    "compatible_target_reached"
                    if stopped == "target"
                    else "exact_forward_budget_exhausted"
                )
                break

    if stopped is None and fallback_used < budget.fallback_attempt_limit:
        generated_sobol = _sobol_seeds(
            context,
            ranked_indices,
            generator_seed=seed_value,
            count_per_branch=budget.sobol_seeds_per_branch,
        )
        while fallback_used < budget.fallback_attempt_limit:
            try:
                value = next(generated_sobol)
            except StopIteration:
                break
            except _SEED_ERRORS as exc:
                raise RuntimeError("V5 Sobol rescue seed generation failed") from exc
            sobol_materialized += 1
            fallback_used += 1
            stopped = process(value, "sobol_rescue")
            if stopped:
                reason = (
                    "compatible_target_reached"
                    if stopped == "target"
                    else "exact_forward_budget_exhausted"
                )
                break

    if stopped is None:
        for value in neural[primary_count:]:
            stopped = process(value, "neural_spillover")
            if stopped:
                reason = (
                    "compatible_target_reached"
                    if stopped == "target"
                    else "exact_forward_budget_exhausted"
                )
                break

    compatible_ids = (
        ()
        if report is None
        else tuple(value.representative_candidate_id for value in report.parameter_modes)
    )
    if len(compatible_ids) >= target:
        status: V5UniversalStatus = "compatible_target_reached"
    elif compatible_ids:
        status = "compatible_partial"
    elif candidates:
        status = "no_compatible_mode_found_within_budget"
    else:
        status = "no_candidate_found_within_budget"
    source_rows = tuple(
        V5UniversalSourceAccounting(
            source=source,
            potential_seed_count={
                "neural": len(neural),
                "retrieval": len(retrieval),
                "sobol": context.branch_count * budget.sobol_seeds_per_branch,
            }[source],
            materialized_seed_count={
                "neural": len(neural),
                "retrieval": len(retrieval),
                "sobol": sobol_materialized,
            }[source],
            attempts=sum(value.source == source for value in attempts),
            candidates_returned=sum(value.source == source for value in provenance),
            exact_forward_calls=sum(
                value.exact_forward_calls for value in attempts if value.source == source
            ),
        )
        for source in ("neural", "retrieval", "sobol")
    )
    return V5UniversalInferenceResult(
        status=status,
        termination_reason=reason,
        target_parameter_mode_count=target,
        compatible_parameter_mode_count=len(compatible_ids),
        compatible_representative_candidate_ids=compatible_ids,
        context_audit_sha256=context.audit_sha256,
        globally_ranked_branch_keys=tuple(
            context.branches[index].global_key.wire_key for index in ranked_indices
        ),
        candidates=tuple(candidates),
        candidate_provenance=tuple(provenance),
        attempts=tuple(attempts),
        source_accounting=source_rows,
        evaluation_report=report,
        model_call_count=1,
        evaluation_call_count=evaluation_calls,
        forward_evaluation_limit=budget.forward_evaluation_limit,
        forward_evaluations_used=forward_used,
        forward_evaluations_remaining=budget.forward_evaluation_limit - forward_used,
        all_scheduled_seeds_processed=(reason == "seed_schedule_exhausted"),
        proposal_execution_policy_sha256=V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    )


__all__ = [
    "V5EvaluatedCandidateObservationError",
    "V5VerifiedReportObservationError",
    "V5UniversalProposalModelPort",
    "run_v5_universal_one_click_inference",
]
