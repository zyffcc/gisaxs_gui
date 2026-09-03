"""Reference-bound evaluation over exact-forward-call budget prefixes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import fsum, isfinite
from numbers import Integral, Real
import re
from typing import Callable, Sequence

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from .grouped_artifact_v5 import canonical_json
from .paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
    OUTPUT_CAPS,
    PRIMARY_OUTPUT_CAP,
    normalized_log2_budget_auc,
    reference_recall,
)
from .paper_representative_payload_v5 import (
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
    V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
    V5PaperParameterRepresentativePayload,
)


V5_PAPER_BUDGET_EVALUATOR_SCHEMA = "gisaxs.posterior_v8.paper_budget_evaluator/v6"
V5_PAPER_BUDGET_EVALUATOR_VERSION = (
    "query_context_bound_typed_actual_emitted_representative_replay_budget_evaluation_v6"
)
V5_EQUIVALENCE_MATCHING_VERSION = (
    "maximum_cardinality_then_minimum_total_normalized_distance_hungarian/v1"
)
V5_EQUIVALENCE_MATCHING_ENGINE = f"scipy.optimize.linear_sum_assignment@{scipy.__version__}"
V5_PAIRED_BOOTSTRAP_VERSION = (
    "fixed_realized_qualified_cohort_paired_parent_percentile_bootstrap/v2"
)
V5_PAIRED_BOOTSTRAP_RNG = "numpy.random.Generator(PCG64)"
V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE = (
    "secondary_fixed_realized_qualified_cohort_descriptive_only_not_population_inference"
)
V5_EXACT_COMPATIBLE = "exact_compatible"
V5_EXACT_INCOMPATIBLE = "exact_incompatible"
V5_EXACT_UNVERIFIED = "unverified"
V5_EXACT_COMPATIBILITY_STATUSES = (V5_EXACT_COMPATIBLE, V5_EXACT_INCOMPATIBLE, V5_EXACT_UNVERIFIED)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _increasing(values: Sequence[int], name: str, *, minimum_size: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(
        _positive_integer(value, f"{name}[{index}]") for index, value in enumerate(values)
    )
    if len(result) < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FrozenReferenceRepresentative:
    representative_id: str
    payload: V5PaperParameterRepresentativePayload

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "representative_id", _text(self.representative_id, "representative_id")
        )
        if not isinstance(self.payload, V5PaperParameterRepresentativePayload):
            raise TypeError("reference payload must be a typed parameter representative")
        if (
            self.payload.role != V5_REFERENCE_REPRESENTATIVE_ROLE
            or self.payload.representative_id != self.representative_id
        ):
            raise ValueError("reference payload role/ID disagrees with its envelope")


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FrozenReferenceSet:
    query_id: str
    pairing_unit_id: str
    reference_set_id: str
    reference_set_sha256: str
    comparison_protocol_id: str
    comparison_protocol_sha256: str
    representatives: tuple[V5FrozenReferenceRepresentative, ...]

    def __post_init__(self) -> None:
        for name in ("query_id", "pairing_unit_id", "reference_set_id", "comparison_protocol_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("reference_set_sha256", "comparison_protocol_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        values = tuple(self.representatives)
        if not values or not all(
            isinstance(value, V5FrozenReferenceRepresentative) for value in values
        ):
            raise ValueError("representatives must contain frozen reference representatives")
        ids = [value.representative_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("reference representative IDs must be unique")
        query_contexts = {value.payload.query_context_sha256 for value in values}
        if len(query_contexts) != 1:
            raise ValueError("all reference representatives must bind one query context")
        object.__setattr__(
            self,
            "representatives",
            tuple(sorted(values, key=lambda value: value.representative_id)),
        )

    @property
    def representative_payload_set_sha256(self) -> str:
        return sha256(
            canonical_json(
                [
                    {
                        "representative_id": value.representative_id,
                        "payload_sha256": value.payload.sha256,
                    }
                    for value in self.representatives
                ]
            ).encode("utf-8")
        ).hexdigest()

    @property
    def query_context_sha256(self) -> str:
        return self.representatives[0].payload.query_context_sha256


@dataclass(frozen=True, eq=False, kw_only=True)
class V5ExactForwardCall:
    exact_call_index: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "exact_call_index", _positive_integer(self.exact_call_index, "exact_call_index")
        )
        object.__setattr__(
            self, "elapsed_seconds", _finite_nonnegative(self.elapsed_seconds, "elapsed_seconds")
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class V5CandidateEmission:
    available_after_call: int
    output_rank: int
    candidate_id: str
    compatibility_status: str
    elapsed_seconds: float
    payload: V5PaperParameterRepresentativePayload

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "available_after_call",
            _positive_integer(self.available_after_call, "available_after_call"),
        )
        object.__setattr__(self, "output_rank", _positive_integer(self.output_rank, "output_rank"))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        status = _text(self.compatibility_status, "compatibility_status")
        if status not in V5_EXACT_COMPATIBILITY_STATUSES:
            raise ValueError("compatibility_status is unsupported")
        object.__setattr__(self, "compatibility_status", status)
        object.__setattr__(
            self, "elapsed_seconds", _finite_nonnegative(self.elapsed_seconds, "elapsed_seconds")
        )
        if not isinstance(self.payload, V5PaperParameterRepresentativePayload):
            raise TypeError("candidate payload must be a typed parameter representative")
        if (
            self.payload.role != V5_EMITTED_REPRESENTATIVE_ROLE
            or self.payload.representative_id != self.candidate_id
        ):
            raise ValueError("candidate payload role/ID disagrees with its emission")
        if status == V5_EXACT_COMPATIBLE and (
            not self.payload.parameter.bounds_pass or not self.payload.parameter.physics_pass
        ):
            raise ValueError("exact-compatible emission must pass bounds and physics gates")


@dataclass(frozen=True, eq=False, kw_only=True)
class V5MethodExactCallTrace:
    query_id: str
    pairing_unit_id: str
    method_id: str
    method_protocol_id: str
    method_protocol_sha256: str
    trace_id: str
    trace_artifact_sha256: str
    reference_set_id: str
    reference_set_sha256: str
    comparison_protocol_id: str
    comparison_protocol_sha256: str
    exact_forward_call_budget: int
    exact_forward_calls: tuple[V5ExactForwardCall, ...]
    candidate_emissions: tuple[V5CandidateEmission, ...]

    def __post_init__(self) -> None:
        text_fields = (
            "query_id",
            "pairing_unit_id",
            "method_id",
            "method_protocol_id",
            "trace_id",
            "reference_set_id",
            "comparison_protocol_id",
        )
        for name in text_fields:
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "method_protocol_sha256",
            "trace_artifact_sha256",
            "reference_set_sha256",
            "comparison_protocol_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        budget = _positive_integer(self.exact_forward_call_budget, "exact_forward_call_budget")
        calls = tuple(self.exact_forward_calls)
        if not all(isinstance(value, V5ExactForwardCall) for value in calls):
            raise TypeError("exact_forward_calls must contain V5ExactForwardCall values")
        indices = tuple(value.exact_call_index for value in calls)
        if indices != tuple(range(1, budget + 1)):
            raise ValueError(
                "exact call indices must be complete, ordered, and contiguous from one"
            )
        call_times = tuple(value.elapsed_seconds for value in calls)
        if any(left > right for left, right in zip(call_times, call_times[1:])):
            raise ValueError("exact-call elapsed_seconds must be non-decreasing")

        emissions = tuple(self.candidate_emissions)
        if not all(isinstance(value, V5CandidateEmission) for value in emissions):
            raise TypeError("candidate_emissions must contain V5CandidateEmission values")
        if any(value.available_after_call > budget for value in emissions):
            raise ValueError("candidate emission leaks beyond the trace exact-call budget")
        ranks = tuple(value.output_rank for value in emissions)
        if set(ranks) != set(range(1, len(emissions) + 1)):
            raise ValueError("candidate output ranks must be unique and contiguous from one")
        ids = tuple(value.candidate_id for value in emissions)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique within a trace")
        for value in emissions:
            earliest = calls[value.available_after_call - 1].elapsed_seconds
            latest = (
                calls[value.available_after_call].elapsed_seconds
                if value.available_after_call < budget
                else None
            )
            if value.elapsed_seconds < earliest or (
                latest is not None and value.elapsed_seconds > latest
            ):
                raise ValueError(
                    "candidate elapsed_seconds must fall between its availability call "
                    "and the next exact call"
                )
        object.__setattr__(self, "exact_forward_call_budget", budget)
        object.__setattr__(self, "exact_forward_calls", calls)
        object.__setattr__(
            self,
            "candidate_emissions",
            tuple(sorted(emissions, key=lambda value: value.output_rank)),
        )

    @property
    def ledger_sha256(self) -> str:
        payload = {
            "exact_forward_calls": [asdict(value) for value in self.exact_forward_calls],
            "candidate_emissions": [
                {
                    "available_after_call": value.available_after_call,
                    "output_rank": value.output_rank,
                    "candidate_id": value.candidate_id,
                    "compatibility_status": value.compatibility_status,
                    "elapsed_seconds": value.elapsed_seconds,
                    "emitted_representative_payload_sha256": value.payload.sha256,
                }
                for value in self.candidate_emissions
            ],
        }
        return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5PaperBudgetEvaluationConfig:
    comparison_protocol_id: str
    comparison_protocol_sha256: str
    equivalence_matcher_id: str
    equivalence_matcher_sha256: str
    equivalence_threshold_id: str
    equivalence_threshold_sha256: str
    maximum_normalized_distance: float
    output_caps: tuple[int, ...] = OUTPUT_CAPS
    exact_forward_budgets: tuple[int, ...] = EXACT_FORWARD_BUDGETS

    def __post_init__(self) -> None:
        for name in (
            "comparison_protocol_id",
            "equivalence_matcher_id",
            "equivalence_threshold_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "comparison_protocol_sha256",
            "equivalence_matcher_sha256",
            "equivalence_threshold_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "maximum_normalized_distance",
            _finite_nonnegative(self.maximum_normalized_distance, "maximum_normalized_distance"),
        )
        object.__setattr__(
            self, "output_caps", _increasing(self.output_caps, "output_caps", minimum_size=1)
        )
        object.__setattr__(
            self,
            "exact_forward_budgets",
            _increasing(self.exact_forward_budgets, "exact_forward_budgets", minimum_size=2),
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
            "version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
            **asdict(self),
            "budget_prefix_rule": "candidate_available_after_call_less_than_or_equal_to_budget",
            "exact_call_accounting_rule": "complete_contiguous_ledger_from_one_to_trace_budget",
            "output_cap_rule": (
                "filter_exact_compatible_emissions_available_within_budget_then_select_"
                "the_first_N_by_frozen_global_output_rank;incompatible_or_unverified_"
                "attempts_consume_exact_calls_but_not_the_returned_solution_cap"
            ),
            "reference_matching_payload_rule": (
                "match_the_exact_payload_of_each_user_visible_emitted_cluster_"
                "representative_never_any_hidden_cluster_member"
            ),
            "representative_payload_schema": V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
            "representative_payload_version": V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
            "representative_query_context_rule": (
                "all_reference_and_emitted_payloads_bind_one_identical_query_context_sha256"
            ),
            "matching_rule": V5_EQUIVALENCE_MATCHING_VERSION,
            "matching_engine": V5_EQUIVALENCE_MATCHING_ENGINE,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


EquivalenceDistanceMatcher = Callable[
    [V5PaperParameterRepresentativePayload, V5PaperParameterRepresentativePayload],
    float | None,
]


@dataclass(frozen=True, kw_only=True)
class V5PaperBudgetMethodResult:
    method_id: str
    method_protocol_id: str
    method_protocol_sha256: str
    trace_id: str
    trace_artifact_sha256: str
    trace_ledger_sha256: str
    hit_count_matrix: tuple[tuple[int, ...], ...]
    recall_matrix: tuple[tuple[float, ...], ...]
    recall_ceiling_by_output_cap: tuple[tuple[int, float], ...]
    selected_candidate_ids_matrix: tuple[tuple[tuple[str, ...], ...], ...]
    matched_pairs_matrix: tuple[tuple[tuple[tuple[str, str, float], ...], ...], ...]
    matched_distance_sum_matrix: tuple[tuple[float, ...], ...]
    auc_by_output_cap: tuple[tuple[int, float], ...]
    first_compatible_exact_call: int | None
    time_to_first_compatible_seconds: float | None
    exact_forward_calls_evaluated: int
    candidate_emissions_evaluated: int
    compatibility_status_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, kw_only=True)
class V5PairedPaperBudgetQueryRecord:
    query_id: str
    query_context_sha256: str
    pairing_unit_id: str
    reference_set_id: str
    reference_set_sha256: str
    reference_representative_ids: tuple[str, ...]
    reference_representative_payload_set_sha256: str
    evaluator_config_sha256: str
    comparison_protocol_id: str
    comparison_protocol_sha256: str
    equivalence_matcher_id: str
    equivalence_matcher_sha256: str
    equivalence_threshold_id: str
    equivalence_threshold_sha256: str
    maximum_normalized_distance: float
    output_caps: tuple[int, ...]
    exact_forward_budgets: tuple[int, ...]
    method_results: tuple[V5PaperBudgetMethodResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_context_sha256",
            _digest(self.query_context_sha256, "query_context_sha256"),
        )
        object.__setattr__(
            self,
            "reference_representative_payload_set_sha256",
            _digest(
                self.reference_representative_payload_set_sha256,
                "reference_representative_payload_set_sha256",
            ),
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
            "version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def _minimum_distance_maximum_cardinality_pairs(
    references: Sequence[V5FrozenReferenceRepresentative],
    candidates: Sequence[V5CandidateEmission],
    distances: dict[tuple[str, str], float | None],
    *,
    maximum_normalized_distance: float,
) -> tuple[tuple[str, str, float], ...]:
    """Maximize matched modes, then minimize their total caller-defined distance."""

    canonical_references = tuple(sorted(references, key=lambda value: value.representative_id))
    canonical_candidates = tuple(sorted(candidates, key=lambda value: value.candidate_id))
    if not canonical_references or not canonical_candidates:
        return ()
    matrix = np.full(
        (len(canonical_references), len(canonical_candidates)), np.inf, dtype=np.float64
    )
    for reference_index, reference in enumerate(canonical_references):
        for candidate_index, candidate in enumerate(canonical_candidates):
            distance = distances[(reference.representative_id, candidate.candidate_id)]
            if distance is not None and distance <= maximum_normalized_distance:
                matrix[reference_index, candidate_index] = distance

    # Scale valid costs to [0, 1]. An invalid edge then costs more than every
    # valid edge combined, so Hungarian minimization has an exact lexicographic
    # objective: maximum valid cardinality, then minimum total distance.
    assignment_size = min(matrix.shape)
    if maximum_normalized_distance > 0.0:
        valid_costs = matrix / maximum_normalized_distance
    else:
        valid_costs = matrix
    invalid_cost = float(assignment_size + 1)
    costs = np.where(np.isfinite(matrix), valid_costs, invalid_cost)
    reference_indices, candidate_indices = linear_sum_assignment(costs)
    pairs = []
    for reference_index, candidate_index in zip(
        reference_indices.tolist(), candidate_indices.tolist()
    ):
        distance = float(matrix[reference_index, candidate_index])
        if isfinite(distance):
            pairs.append(
                (
                    canonical_references[reference_index].representative_id,
                    canonical_candidates[candidate_index].candidate_id,
                    distance,
                )
            )
    return tuple(sorted(pairs, key=lambda value: (value[0], value[1])))


def _evaluate_method(
    references: V5FrozenReferenceSet,
    trace: V5MethodExactCallTrace,
    config: V5PaperBudgetEvaluationConfig,
    matcher: EquivalenceDistanceMatcher,
) -> V5PaperBudgetMethodResult:
    largest_budget = config.exact_forward_budgets[-1]
    evaluation_emissions = tuple(
        value for value in trace.candidate_emissions if value.available_after_call <= largest_budget
    )
    compatible = tuple(
        value for value in evaluation_emissions if value.compatibility_status == V5_EXACT_COMPATIBLE
    )
    distances: dict[tuple[str, str], float | None] = {}
    for reference in references.representatives:
        for candidate in sorted(compatible, key=lambda value: value.candidate_id):
            verdict = matcher(reference.payload, candidate.payload)
            if verdict is None:
                distance = None
            else:
                distance = _finite_nonnegative(verdict, "equivalence matcher distance")
            distances[(reference.representative_id, candidate.candidate_id)] = distance
    hits, recalls, selections, pair_rows, distance_rows = [], [], [], [], []
    for output_cap in config.output_caps:
        hit_row, recall_row, selection_row, pairs_row, distance_row = [], [], [], [], []
        for budget in config.exact_forward_budgets:
            available = [value for value in compatible if value.available_after_call <= budget]
            selected = tuple(sorted(available, key=lambda value: value.output_rank)[:output_cap])
            pairs = _minimum_distance_maximum_cardinality_pairs(
                references.representatives,
                selected,
                distances,
                maximum_normalized_distance=config.maximum_normalized_distance,
            )
            recall = reference_recall(
                len(pairs), len(references.representatives), output_cap=output_cap
            )
            hit_row.append(len(pairs))
            recall_row.append(recall.recall)
            selection_row.append(tuple(value.candidate_id for value in selected))
            pairs_row.append(pairs)
            distance_row.append(float(fsum(value[2] for value in pairs)))
        hits.append(tuple(hit_row))
        recalls.append(tuple(recall_row))
        selections.append(tuple(selection_row))
        pair_rows.append(tuple(pairs_row))
        distance_rows.append(tuple(distance_row))
    first = min(
        compatible,
        key=lambda value: (value.available_after_call, value.elapsed_seconds, value.output_rank),
        default=None,
    )
    aucs = tuple(
        (
            cap,
            normalized_log2_budget_auc(
                dict(zip(config.exact_forward_budgets, row)), budgets=config.exact_forward_budgets
            ),
        )
        for cap, row in zip(config.output_caps, recalls)
    )
    return V5PaperBudgetMethodResult(
        method_id=trace.method_id,
        method_protocol_id=trace.method_protocol_id,
        method_protocol_sha256=trace.method_protocol_sha256,
        trace_id=trace.trace_id,
        trace_artifact_sha256=trace.trace_artifact_sha256,
        trace_ledger_sha256=trace.ledger_sha256,
        hit_count_matrix=tuple(hits),
        recall_matrix=tuple(recalls),
        recall_ceiling_by_output_cap=tuple(
            (cap, reference_recall(0, len(references.representatives), output_cap=cap).ceiling)
            for cap in config.output_caps
        ),
        selected_candidate_ids_matrix=tuple(selections),
        matched_pairs_matrix=tuple(pair_rows),
        matched_distance_sum_matrix=tuple(distance_rows),
        auc_by_output_cap=aucs,
        first_compatible_exact_call=None if first is None else first.available_after_call,
        time_to_first_compatible_seconds=None if first is None else first.elapsed_seconds,
        exact_forward_calls_evaluated=largest_budget,
        candidate_emissions_evaluated=len(evaluation_emissions),
        compatibility_status_counts=tuple(
            (status, sum(value.compatibility_status == status for value in evaluation_emissions))
            for status in V5_EXACT_COMPATIBILITY_STATUSES
        ),
    )


def evaluate_v5_paired_paper_budget_query(
    references: V5FrozenReferenceSet,
    traces: Sequence[V5MethodExactCallTrace],
    *,
    config: V5PaperBudgetEvaluationConfig,
    equivalence_distance_matcher: EquivalenceDistanceMatcher,
) -> V5PairedPaperBudgetQueryRecord:
    """Evaluate every method against one frozen reference set and call budget."""

    if not isinstance(references, V5FrozenReferenceSet):
        raise TypeError("references must be a V5FrozenReferenceSet")
    if not isinstance(config, V5PaperBudgetEvaluationConfig):
        raise TypeError("config must be a V5PaperBudgetEvaluationConfig")
    if not callable(equivalence_distance_matcher):
        raise TypeError("equivalence_distance_matcher must be callable")
    values = tuple(traces)
    if not values or not all(isinstance(value, V5MethodExactCallTrace) for value in values):
        raise ValueError("traces must contain at least one method trace")
    method_ids = [value.method_id for value in values]
    if len(method_ids) != len(set(method_ids)):
        raise ValueError("each query requires one unique trace per method")
    expected = (
        references.query_id,
        references.pairing_unit_id,
        references.reference_set_id,
        references.reference_set_sha256,
        config.comparison_protocol_id,
        config.comparison_protocol_sha256,
    )
    if (references.comparison_protocol_id, references.comparison_protocol_sha256) != expected[4:6]:
        raise ValueError("reference set escaped the evaluator comparison protocol")
    trace_budgets = {trace.exact_forward_call_budget for trace in values}
    if len(trace_budgets) != 1:
        raise ValueError("paired method traces must use one exact-forward call budget")
    if next(iter(trace_budgets)) < config.exact_forward_budgets[-1]:
        raise ValueError("method trace does not cover the largest evaluation budget")
    for trace in values:
        observed = (
            trace.query_id,
            trace.pairing_unit_id,
            trace.reference_set_id,
            trace.reference_set_sha256,
            trace.comparison_protocol_id,
            trace.comparison_protocol_sha256,
        )
        if observed != expected:
            raise ValueError("method trace has a reference, query, protocol, or budget mismatch")
        if any(
            emission.payload.query_context_sha256 != references.query_context_sha256
            for emission in trace.candidate_emissions
        ):
            raise ValueError("candidate emission escaped the frozen query context")
    results = tuple(
        _evaluate_method(references, trace, config, equivalence_distance_matcher)
        for trace in sorted(values, key=lambda value: value.method_id)
    )
    return V5PairedPaperBudgetQueryRecord(
        query_id=references.query_id,
        query_context_sha256=references.query_context_sha256,
        pairing_unit_id=references.pairing_unit_id,
        reference_set_id=references.reference_set_id,
        reference_set_sha256=references.reference_set_sha256,
        reference_representative_ids=tuple(
            value.representative_id for value in references.representatives
        ),
        reference_representative_payload_set_sha256=(references.representative_payload_set_sha256),
        evaluator_config_sha256=config.sha256,
        comparison_protocol_id=config.comparison_protocol_id,
        comparison_protocol_sha256=config.comparison_protocol_sha256,
        equivalence_matcher_id=config.equivalence_matcher_id,
        equivalence_matcher_sha256=config.equivalence_matcher_sha256,
        equivalence_threshold_id=config.equivalence_threshold_id,
        equivalence_threshold_sha256=config.equivalence_threshold_sha256,
        maximum_normalized_distance=config.maximum_normalized_distance,
        output_caps=config.output_caps,
        exact_forward_budgets=config.exact_forward_budgets,
        method_results=results,
    )


@dataclass(frozen=True, kw_only=True)
class V5PairedBootstrapAucInput:
    evaluator_config_sha256: str
    output_cap: int
    method_ids: tuple[str, ...]
    pairing_unit_ids: tuple[str, ...]
    query_ids_by_pairing_unit: tuple[tuple[str, ...], ...]
    auc_matrix: tuple[tuple[float, ...], ...]

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


def build_v5_paired_bootstrap_auc_input(
    records: Sequence[V5PairedPaperBudgetQueryRecord],
    *,
    output_cap: int = PRIMARY_OUTPUT_CAP,
) -> V5PairedBootstrapAucInput:
    """Build a secondary fixed-cohort descriptive resampling input.

    This helper does not turn one scrambled Sobol point set into iid draws and
    must not be used for the paper's population interval.  Formal inference is
    owned by the independent-scramble RQMC evaluator.
    """

    values = tuple(records)
    if not values or not all(isinstance(value, V5PairedPaperBudgetQueryRecord) for value in values):
        raise ValueError("records must contain paired query records")
    cap = _positive_integer(output_cap, "output_cap")
    if len({value.query_id for value in values}) != len(values):
        raise ValueError("paired query IDs must be unique")
    record_contracts = {
        (
            _digest(value.evaluator_config_sha256, "evaluator_config_sha256"),
            value.comparison_protocol_id,
            value.comparison_protocol_sha256,
            value.equivalence_matcher_id,
            value.equivalence_matcher_sha256,
            value.equivalence_threshold_id,
            value.equivalence_threshold_sha256,
            value.maximum_normalized_distance,
            value.output_caps,
            value.exact_forward_budgets,
        )
        for value in values
    }
    method_contracts = {
        tuple(
            (result.method_id, result.method_protocol_id, result.method_protocol_sha256)
            for result in value.method_results
        )
        for value in values
    }
    if len(record_contracts) != 1 or len(method_contracts) != 1:
        raise ValueError("paired records must share one evaluator and method contract")
    method_contract = next(iter(method_contracts))
    method_ids = tuple(value[0] for value in method_contract)
    if not method_ids or len(method_ids) != len(set(method_ids)):
        raise ValueError("paired records require unique method IDs")
    grouped: dict[str, list[V5PairedPaperBudgetQueryRecord]] = {}
    for value in values:
        grouped.setdefault(value.pairing_unit_id, []).append(value)
    unit_ids, query_rows, auc_rows = [], [], []
    for unit_id in sorted(grouped):
        rows = sorted(grouped[unit_id], key=lambda value: value.query_id)
        unit_ids.append(unit_id)
        query_rows.append(tuple(value.query_id for value in rows))
        method_values = []
        for method_index in range(len(method_ids)):
            selected = []
            for row in rows:
                auc_by_cap = dict(row.method_results[method_index].auc_by_output_cap)
                if len(auc_by_cap) != len(row.method_results[method_index].auc_by_output_cap):
                    raise ValueError("method result contains duplicate output-cap AUCs")
                if cap not in auc_by_cap:
                    raise ValueError("requested bootstrap output cap is absent")
                value = float(auc_by_cap[cap])
                if not isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("method AUC must be finite and in [0, 1]")
                selected.append(value)
            method_values.append(float(fsum(selected) / len(selected)))
        auc_rows.append(tuple(method_values))
    return V5PairedBootstrapAucInput(
        evaluator_config_sha256=next(iter(record_contracts))[0],
        output_cap=cap,
        method_ids=method_ids,
        pairing_unit_ids=tuple(unit_ids),
        query_ids_by_pairing_unit=tuple(query_rows),
        auc_matrix=tuple(auc_rows),
    )


@dataclass(frozen=True, kw_only=True)
class V5PairedBootstrapAucSummary:
    bootstrap_input_sha256: str
    baseline_method_id: str
    comparison_method_id: str
    pairing_unit_count: int
    baseline_mean_auc: float
    comparison_mean_auc: float
    observed_comparison_minus_baseline: float
    bootstrap_mean_difference: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    bootstrap_replicates: int
    rng_seed: int
    rng_algorithm: str
    numpy_version: str
    bootstrap_version: str = V5_PAIRED_BOOTSTRAP_VERSION
    claim_scope: str = V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE


def summarize_v5_paired_bootstrap_auc(
    values: V5PairedBootstrapAucInput,
    *,
    baseline_method_id: str,
    comparison_method_id: str,
    bootstrap_replicates: int = 10_000,
    rng_seed: int,
    confidence_level: float = 0.95,
) -> V5PairedBootstrapAucSummary:
    """Return a descriptive interval for one fixed realized qualified cohort."""

    if not isinstance(values, V5PairedBootstrapAucInput):
        raise TypeError("values must be V5PairedBootstrapAucInput")
    if not values.method_ids or not values.pairing_unit_ids:
        raise ValueError("paired bootstrap input must contain methods and pairing units")
    baseline = _text(baseline_method_id, "baseline_method_id")
    comparison = _text(comparison_method_id, "comparison_method_id")
    if (
        baseline == comparison
        or baseline not in values.method_ids
        or comparison not in values.method_ids
    ):
        raise ValueError("bootstrap methods must be distinct members of the paired input")
    replicates = _positive_integer(bootstrap_replicates, "bootstrap_replicates")
    if isinstance(rng_seed, (bool, np.bool_)) or not isinstance(rng_seed, Integral):
        raise TypeError("rng_seed must be an integer")
    seed = int(rng_seed)
    if not 0 <= seed < 2**64:
        raise ValueError("rng_seed must be in [0, 2**64)")
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a real number")
    level = float(confidence_level)
    if not isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    matrix = np.asarray(values.auc_matrix, dtype=np.float64)
    if matrix.shape != (len(values.pairing_unit_ids), len(values.method_ids)) or not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError("bootstrap AUC matrix shape/content is invalid")
    baseline_values = matrix[:, values.method_ids.index(baseline)]
    comparison_values = matrix[:, values.method_ids.index(comparison)]
    differences = comparison_values - baseline_values
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(replicates, dtype=np.float64)
    unit_count = differences.size
    chunk_size = max(1, min(replicates, 1_000_000 // unit_count))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, unit_count, size=(stop - start, unit_count))
        draws[start:stop] = np.mean(differences[indices], axis=1)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, (tail, 1.0 - tail), method="linear")
    return V5PairedBootstrapAucSummary(
        bootstrap_input_sha256=values.sha256,
        baseline_method_id=baseline,
        comparison_method_id=comparison,
        pairing_unit_count=unit_count,
        baseline_mean_auc=float(np.mean(baseline_values)),
        comparison_mean_auc=float(np.mean(comparison_values)),
        observed_comparison_minus_baseline=float(np.mean(differences)),
        bootstrap_mean_difference=float(np.mean(draws)),
        confidence_level=level,
        confidence_interval=(float(lower), float(upper)),
        bootstrap_replicates=replicates,
        rng_seed=seed,
        rng_algorithm=V5_PAIRED_BOOTSTRAP_RNG,
        numpy_version=np.__version__,
    )


__all__ = [
    "EquivalenceDistanceMatcher",
    "V5CandidateEmission",
    "V5ExactForwardCall",
    "V5FrozenReferenceRepresentative",
    "V5FrozenReferenceSet",
    "V5MethodExactCallTrace",
    "V5PairedBootstrapAucInput",
    "V5PairedBootstrapAucSummary",
    "V5PairedPaperBudgetQueryRecord",
    "V5PaperParameterRepresentativePayload",
    "V5PaperBudgetEvaluationConfig",
    "V5PaperBudgetMethodResult",
    "V5_EXACT_COMPATIBILITY_STATUSES",
    "V5_EXACT_COMPATIBLE",
    "V5_EXACT_INCOMPATIBLE",
    "V5_EXACT_UNVERIFIED",
    "V5_EQUIVALENCE_MATCHING_VERSION",
    "V5_EQUIVALENCE_MATCHING_ENGINE",
    "V5_PAIRED_BOOTSTRAP_RNG",
    "V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE",
    "V5_PAIRED_BOOTSTRAP_VERSION",
    "V5_PAPER_BUDGET_EVALUATOR_SCHEMA",
    "V5_PAPER_BUDGET_EVALUATOR_VERSION",
    "V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA",
    "V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION",
    "V5_REFERENCE_REPRESENTATIVE_ROLE",
    "V5_EMITTED_REPRESENTATIVE_ROLE",
    "build_v5_paired_bootstrap_auc_input",
    "evaluate_v5_paired_paper_budget_query",
    "summarize_v5_paired_bootstrap_auc",
]
