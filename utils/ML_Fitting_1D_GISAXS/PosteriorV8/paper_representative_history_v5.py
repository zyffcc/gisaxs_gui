"""Budget-local, replaceable user-visible representatives for live paper traces.

This is deliberately not a V5MethodExactCallTrace subclass: legacy append-only
writers must not silently discard replacement events from this newer contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from numbers import Integral, Real
from collections.abc import Mapping

from .grouped_artifact_v5 import canonical_json
from .paper_budget_evaluator_v5 import (
    V5CandidateEmission,
    V5MethodExactCallTrace,
    V5_EXACT_COMPATIBLE,
    V5FrozenReferenceSet,
    V5PaperBudgetEvaluationConfig,
    V5PaperBudgetMethodResult,
    EquivalenceDistanceMatcher,
    _evaluate_method,
)


V5_REPRESENTATIVE_HISTORY_SCHEMA = "gisaxs.posterior_v8.paper_representative_history/v1"


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class V5RepresentativeSnapshot:
    """Ordered compatible representatives actually visible at this checkpoint."""

    available_after_call: int
    elapsed_seconds: float
    representative_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_after_call",
                           _positive(self.available_after_call, "available_after_call"))
        elapsed = self.elapsed_seconds
        if isinstance(elapsed, bool) or not isinstance(elapsed, Real):
            raise TypeError("elapsed_seconds must be real")
        if not 0 <= elapsed < float("inf"):
            raise ValueError("elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "elapsed_seconds", float(elapsed))
        if isinstance(self.representative_ids, (str, bytes)):
            raise TypeError("representative_ids must be a sequence of IDs, not text")
        ids = tuple(self.representative_ids)
        if any(not isinstance(value, str) or not value or value.strip() != value for value in ids):
            raise ValueError("representative IDs must be non-empty stripped strings")
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot representative IDs must be unique")
        object.__setattr__(self, "representative_ids", ids)


@dataclass(frozen=True, kw_only=True)
class V5RepresentativeHistory:
    """Validate live snapshots against a complete physical emission/call ledger.

The emission inventory provides immutable parameter payloads; its global ranks
do not determine snapshot order. Producers must capture the evaluator's actual
order without access to reference matches. Empty snapshots explicitly withdraw
all representatives; an absent earlier snapshot means nothing was visible yet.
"""

    trace: V5MethodExactCallTrace
    snapshots: tuple[V5RepresentativeSnapshot, ...]

    def __post_init__(self) -> None:
        if type(self.trace) is not V5MethodExactCallTrace:
            raise TypeError("history requires an exact V5MethodExactCallTrace")
        snapshots = tuple(self.snapshots)
        if not snapshots or any(type(row) is not V5RepresentativeSnapshot for row in snapshots):
            raise ValueError("history requires explicit typed snapshots")
        inventory = {row.candidate_id: row for row in self.trace.candidate_emissions}
        previous_call, previous_time = 0, 0.0
        calls = self.trace.exact_forward_calls
        for row in snapshots:
            index = row.available_after_call
            if not previous_call < index <= self.trace.exact_forward_call_budget:
                raise ValueError("snapshot call indices must be strictly increasing within budget")
            if row.elapsed_seconds < max(previous_time, calls[index - 1].elapsed_seconds):
                raise ValueError("snapshot precedes its exact call or previous snapshot")
            if index < len(calls) and row.elapsed_seconds > calls[index].elapsed_seconds:
                raise ValueError("snapshot occurred after the next exact call")
            for candidate_id in row.representative_ids:
                emission = inventory.get(candidate_id)
                if emission is None:
                    raise ValueError("snapshot references an unknown emission")
                if emission.compatibility_status != V5_EXACT_COMPATIBLE:
                    raise ValueError("snapshot includes an unverified or incompatible emission")
                if emission.available_after_call > index or emission.elapsed_seconds > row.elapsed_seconds:
                    raise ValueError("snapshot includes a future emission")
            previous_call, previous_time = index, row.elapsed_seconds
        object.__setattr__(self, "snapshots", snapshots)

    def representatives_at(self, budget: int, output_cap: int) -> tuple[V5CandidateEmission, ...]:
        budget = _positive(budget, "budget")
        cap = _positive(output_cap, "output_cap")
        if budget > self.trace.exact_forward_call_budget:
            raise ValueError("requested budget exceeds the recorded trace")
        available = [row for row in self.snapshots if row.available_after_call <= budget]
        if not available:
            return ()
        inventory = {row.candidate_id: row for row in self.trace.candidate_emissions}
        return tuple(inventory[key] for key in available[-1].representative_ids[:cap])

    def to_payload(self) -> dict[str, object]:
        """Bind snapshots to the physical trace without duplicating its payloads."""
        payload = {
            "schema": V5_REPRESENTATIVE_HISTORY_SCHEMA,
            "trace_ledger_sha256": self.trace.ledger_sha256,
            "trace_artifact_sha256": self.trace.trace_artifact_sha256,
            "query_id": self.trace.query_id,
            "pairing_unit_id": self.trace.pairing_unit_id,
            "method_id": self.trace.method_id,
            "method_protocol_sha256": self.trace.method_protocol_sha256,
            "reference_set_sha256": self.trace.reference_set_sha256,
            "comparison_protocol_sha256": self.trace.comparison_protocol_sha256,
            "snapshots": [{**asdict(row), "representative_ids": list(row.representative_ids)}
                          for row in self.snapshots],
        }
        return {**payload, "history_sha256": sha256(canonical_json(payload).encode("utf-8")).hexdigest()}

    @property
    def sha256(self) -> str:
        return self.to_payload()["history_sha256"]

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], *, trace: V5MethodExactCallTrace,
    ) -> V5RepresentativeHistory:
        """Replay a JSON record against an independently verified physical trace."""
        if not isinstance(payload, Mapping):
            raise TypeError("history payload must be a mapping")
        rows = payload.get("snapshots")
        if not isinstance(rows, list):
            raise TypeError("history snapshots must be a JSON array")
        snapshots = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "available_after_call", "elapsed_seconds", "representative_ids",
            }:
                raise ValueError("snapshot fields differ from the schema")
            if not isinstance(row["representative_ids"], list):
                raise TypeError("representative_ids must be a JSON array")
            snapshots.append(V5RepresentativeSnapshot(**row))
        history = cls(trace=trace, snapshots=tuple(snapshots))
        # Comparing the complete canonical record verifies both its self hash and
        # every external trace identity, including fields an attacker rehashed.
        if canonical_json(dict(payload)) != canonical_json(history.to_payload()):
            raise ValueError("history schema, hash, or physical trace binding differs")
        return history


@dataclass(frozen=True, kw_only=True)
class V5SnapshotBudgetEvaluation:
    """Distinct result identity; never masquerades as an append-only result."""

    history_sha256: str
    reference_set_sha256: str
    evaluator_config_sha256: str
    method_result: V5PaperBudgetMethodResult

    @property
    def sha256(self) -> str:
        return sha256(canonical_json({
            "schema": "gisaxs.posterior_v8.snapshot_budget_evaluation/v1",
            **asdict(self),
        }).encode("utf-8")).hexdigest()


def evaluate_v5_representative_history(
    references: V5FrozenReferenceSet,
    history: V5RepresentativeHistory,
    *,
    config: V5PaperBudgetEvaluationConfig,
    equivalence_distance_matcher: EquivalenceDistanceMatcher,
) -> V5SnapshotBudgetEvaluation:
    """Use budget-local visible modes with the unchanged authoritative matcher."""

    if type(history) is not V5RepresentativeHistory:
        raise TypeError("history must be V5RepresentativeHistory")
    if not isinstance(references, V5FrozenReferenceSet):
        raise TypeError("references must be V5FrozenReferenceSet")
    if not isinstance(config, V5PaperBudgetEvaluationConfig):
        raise TypeError("config must be V5PaperBudgetEvaluationConfig")
    if not callable(equivalence_distance_matcher):
        raise TypeError("equivalence_distance_matcher must be callable")
    trace = history.trace
    if (trace.query_id, trace.pairing_unit_id, trace.reference_set_id,
        trace.reference_set_sha256, trace.comparison_protocol_id,
        trace.comparison_protocol_sha256) != (
        references.query_id, references.pairing_unit_id, references.reference_set_id,
        references.reference_set_sha256, config.comparison_protocol_id,
        config.comparison_protocol_sha256,
    ):
        raise ValueError("history escaped reference/query/comparison binding")
    if (references.comparison_protocol_id, references.comparison_protocol_sha256) != (
        config.comparison_protocol_id, config.comparison_protocol_sha256,
    ):
        raise ValueError("reference escaped comparison binding")
    if trace.exact_forward_call_budget < config.exact_forward_budgets[-1]:
        raise ValueError("history does not cover the evaluation budget")
    if any(row.payload.query_context_sha256 != references.query_context_sha256
           for row in trace.candidate_emissions):
        raise ValueError("history emission escaped query context")
    result = _evaluate_method(
        references, trace, config, equivalence_distance_matcher,
        representative_selector=history.representatives_at,
    )
    first = next((row for row in history.snapshots if row.representative_ids
                  and row.available_after_call <= config.exact_forward_budgets[-1]), None)
    result = replace(
        result,
        representative_selection_policy="budget_snapshot",
        representative_history_sha256=history.sha256,
        first_compatible_exact_call=None if first is None else first.available_after_call,
        time_to_first_compatible_seconds=None if first is None else first.elapsed_seconds,
    )
    return V5SnapshotBudgetEvaluation(
        history_sha256=history.sha256,
        reference_set_sha256=references.reference_set_sha256,
        evaluator_config_sha256=config.sha256,
        method_result=result,
    )
