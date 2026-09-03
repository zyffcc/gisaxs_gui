"""Immutable audit contract for V5.1 exact candidate refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from typing import Literal

import numpy as np

from .branch_codec import BRANCH_CODEC_VERSION, UNIT_CUBE_DIMENSIONS
from .evaluation import CandidateInput
from .gui_amplitude_constraints import (
    GUI_AMPLITUDE_CONSTRAINT_VERSION,
    GuiAmplitudeConstraintAudit,
)


V5_EXACT_REFINEMENT_SCHEMA = "gisaxs.posterior_v8.v5_exact_candidate_refinement/v2"
V5_EXACT_REFINEMENT_VERSION = (
    "posterior_v8_v5_user_local_varying_only_exact_refinement_same_amplitude_polytope_v2"
)
V5_EXACT_REFINEMENT_SCOPE = (
    "exact_forward_and_user_bounds_physics_amplitude_prerequisites_only_"
    "no_measurement_compatibility_no_solution_or_posterior_claim"
)
V5_PROPOSAL_SCORE_HANDOFF = (
    "lexicographic_search_yield_and_mixture_scores_retained_in_attempt_audit_"
    "not_collapsed_to_probability"
)
V5_EXACT_FORWARD_BUDGET_UNIT = (
    "one_geometry_objective_evaluation_including_amplitude_profile_"
    "and_authoritative_gui_forward_verification"
)

V5SeedSource = Literal["neural", "retrieval", "sobol"]
V5AttemptStatus = Literal[
    "refined",
    "validation_failed",
    "refinement_failed",
    "per_candidate_forward_budget_exhausted",
    "total_forward_budget_exhausted_before_seed",
]
V5ExactBatchStatus = Literal[
    "exact_candidates_ready_for_verification",
    "exact_candidates_partial_budget_exhausted",
    "no_candidate_found_within_budget",
]
EXACT_FORWARD_PHASES = (
    "seed_profile_verification",
    "initial_profile_verification",
    "optimizer_residual",
    "optimizer_terminal_verification",
)


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def exact_forward_phase_counts(phases) -> tuple[tuple[str, int], ...]:
    values = tuple(phases)
    if any(value not in EXACT_FORWARD_PHASES for value in values):
        raise ValueError("exact-forward ledger contains an unknown call phase")
    return tuple((phase, values.count(phase)) for phase in EXACT_FORWARD_PHASES)


def _unit_vector(value, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (UNIT_CUBE_DIMENSIONS,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {UNIT_CUBE_DIMENSIONS}-vector")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, kw_only=True)
class V5CandidatePrerequisiteAudit:
    query_sha256: str
    geometry_query_sha256: str
    amplitude_query_sha256: str
    pattern_id: int
    initial_local_unit: tuple[float, ...]
    final_local_unit: tuple[float, ...]
    user_local_codec_version: str
    amplitude_constraint_version: str
    amplitude_constraint_sha256: str
    amplitude_constraint_identity_preserved: bool
    initial_amplitude_audit: GuiAmplitudeConstraintAudit
    final_amplitude_audit: GuiAmplitudeConstraintAudit
    geometry_bounds_satisfied: bool
    physics_satisfied: bool
    physics_violations: tuple[str, ...]
    resolution_presence_satisfied: bool
    exact_forward_consistency_satisfied: bool
    all_prerequisites_satisfied: bool

    def __post_init__(self) -> None:
        if self.user_local_codec_version != BRANCH_CODEC_VERSION:
            raise ValueError("unexpected V5 branch codec version")
        if self.amplitude_constraint_version != GUI_AMPLITUDE_CONSTRAINT_VERSION:
            raise ValueError("unexpected GUI amplitude constraint version")
        for name in ("query_sha256", "geometry_query_sha256", "amplitude_query_sha256"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if (
            not isinstance(self.amplitude_constraint_sha256, str)
            or len(self.amplitude_constraint_sha256) != 64
        ):
            raise ValueError("amplitude_constraint_sha256 must be a SHA256 hex digest")
        _integer(self.pattern_id, "pattern_id", minimum=0)
        object.__setattr__(
            self, "initial_local_unit", _unit_vector(self.initial_local_unit, "initial_local_unit")
        )
        object.__setattr__(
            self, "final_local_unit", _unit_vector(self.final_local_unit, "final_local_unit")
        )
        if not isinstance(
            self.initial_amplitude_audit, GuiAmplitudeConstraintAudit
        ) or not isinstance(self.final_amplitude_audit, GuiAmplitudeConstraintAudit):
            raise TypeError("candidate requires initial and final amplitude audits")
        flags = (
            self.amplitude_constraint_identity_preserved,
            self.geometry_bounds_satisfied,
            self.physics_satisfied,
            self.resolution_presence_satisfied,
            self.exact_forward_consistency_satisfied,
            self.all_prerequisites_satisfied,
        )
        if not all(type(value) is bool for value in flags):
            raise TypeError("candidate prerequisite flags must be explicit booleans")
        required = (
            self.amplitude_constraint_identity_preserved,
            self.initial_amplitude_audit.all_constraints_satisfied,
            self.final_amplitude_audit.all_constraints_satisfied,
            self.geometry_bounds_satisfied,
            self.physics_satisfied,
            self.resolution_presence_satisfied,
            self.exact_forward_consistency_satisfied,
        )
        violations = tuple(str(value) for value in self.physics_violations)
        if self.physics_satisfied != (not violations):
            raise ValueError("physics audit and violation list disagree")
        if self.all_prerequisites_satisfied != all(required):
            raise ValueError("all_prerequisites_satisfied disagrees with its audits")
        object.__setattr__(self, "physics_violations", violations)


@dataclass(frozen=True, kw_only=True)
class V5ExactRefinementAttempt:
    attempt_rank: int
    source: V5SeedSource
    source_id: str
    topology_id: int
    pattern_id: int
    branch_batch_index: int
    status: V5AttemptStatus
    candidate_id: str | None
    candidate_rank: int | None
    forward_call_limit: int
    exact_forward_calls: int
    exact_forward_calls_by_phase: tuple[tuple[str, int], ...]
    cumulative_calls_before: int
    cumulative_calls_after: int
    message: str
    prerequisite_audit: V5CandidatePrerequisiteAudit | None = None
    search_yield_logit: float | None = None
    mixture_log_weight: float | None = None

    def __post_init__(self) -> None:
        if self.source not in {"neural", "retrieval", "sobol"}:
            raise ValueError("attempt source must be neural, retrieval, or sobol")
        if self.status not in {
            "refined",
            "validation_failed",
            "refinement_failed",
            "per_candidate_forward_budget_exhausted",
            "total_forward_budget_exhausted_before_seed",
        }:
            raise ValueError("unknown V5 exact-refinement attempt status")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("attempt source_id must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("attempt message must be non-empty")
        _integer(self.attempt_rank, "attempt_rank", minimum=1)
        _integer(self.topology_id, "topology_id", minimum=0)
        _integer(self.pattern_id, "pattern_id", minimum=0)
        _integer(self.branch_batch_index, "branch_batch_index", minimum=0)
        limit = _integer(self.forward_call_limit, "forward_call_limit", minimum=0)
        used = _integer(self.exact_forward_calls, "exact_forward_calls", minimum=0)
        before = _integer(self.cumulative_calls_before, "cumulative_calls_before", minimum=0)
        after = _integer(self.cumulative_calls_after, "cumulative_calls_after", minimum=0)
        counts = tuple(self.exact_forward_calls_by_phase)
        if tuple(phase for phase, _ in counts) != EXACT_FORWARD_PHASES:
            raise ValueError("attempt exact-forward phases use the wrong order")
        counted = sum(
            _integer(count, f"exact_forward_calls_by_phase[{phase}]", minimum=0)
            for phase, count in counts
        )
        if used > limit or after != before + used or counted != used:
            raise ValueError("attempt exact-forward ledger is inconsistent")
        if self.status == "refined":
            if self.candidate_id is None or self.candidate_rank is None:
                raise ValueError("refined attempt requires candidate identity")
            _integer(self.candidate_rank, "candidate_rank", minimum=1)
            if (
                self.prerequisite_audit is None
                or not self.prerequisite_audit.all_prerequisites_satisfied
            ):
                raise ValueError("refined attempt requires a passing prerequisite audit")
        elif self.candidate_id is not None or self.candidate_rank is not None:
            raise ValueError("failed attempt must not claim a candidate")
        for name in ("search_yield_logit", "mixture_log_weight"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"attempt {name} must be finite or None")


@dataclass(frozen=True, kw_only=True)
class V5ExactForwardLedger:
    budget_unit: str
    configured_total_limit: int
    configured_per_candidate_limit: int
    calls_used: int
    calls_remaining: int
    calls_by_phase: tuple[tuple[str, int], ...]
    input_seed_count: int
    attempts_recorded: int
    refinement_successes: int
    validation_failures: int
    refinement_failures: int
    per_candidate_budget_exhausted_attempts: int
    total_budget_exhausted_before_seed_attempts: int

    def __post_init__(self) -> None:
        if self.budget_unit != V5_EXACT_FORWARD_BUDGET_UNIT:
            raise ValueError("unexpected exact-forward budget unit")
        total = _integer(self.configured_total_limit, "configured_total_limit", minimum=0)
        _integer(
            self.configured_per_candidate_limit,
            "configured_per_candidate_limit",
            minimum=1,
        )
        used = _integer(self.calls_used, "calls_used", minimum=0)
        remaining = _integer(self.calls_remaining, "calls_remaining", minimum=0)
        if used + remaining != total:
            raise ValueError("exact-forward used/remaining ledger does not match its limit")
        counts = tuple(self.calls_by_phase)
        if tuple(phase for phase, _ in counts) != EXACT_FORWARD_PHASES:
            raise ValueError("batch exact-forward phases use the wrong order")
        if (
            sum(_integer(count, f"calls_by_phase[{phase}]", minimum=0) for phase, count in counts)
            != used
        ):
            raise ValueError("batch exact-forward phase counts do not match calls_used")
        outcomes = (
            self.refinement_successes,
            self.validation_failures,
            self.refinement_failures,
            self.per_candidate_budget_exhausted_attempts,
            self.total_budget_exhausted_before_seed_attempts,
        )
        input_count = _integer(self.input_seed_count, "input_seed_count", minimum=1)
        attempts = _integer(self.attempts_recorded, "attempts_recorded", minimum=1)
        if (
            sum(_integer(value, "attempt outcome count", minimum=0) for value in outcomes)
            != attempts
        ):
            raise ValueError("attempt outcome counts do not match attempts_recorded")
        if attempts > input_count:
            raise ValueError("attempts_recorded cannot exceed input_seed_count")


@dataclass(frozen=True, kw_only=True)
class V5ExactRefinementBatchResult:
    status: V5ExactBatchStatus
    query_sha256: str
    candidates: tuple[CandidateInput, ...]
    attempts: tuple[V5ExactRefinementAttempt, ...]
    ledger: V5ExactForwardLedger
    all_input_seeds_processed: bool
    schema: str = V5_EXACT_REFINEMENT_SCHEMA
    version: str = V5_EXACT_REFINEMENT_VERSION
    scientific_scope: str = V5_EXACT_REFINEMENT_SCOPE
    proposal_score_handoff: str = V5_PROPOSAL_SCORE_HANDOFF

    def __post_init__(self) -> None:
        if self.schema != V5_EXACT_REFINEMENT_SCHEMA or self.version != V5_EXACT_REFINEMENT_VERSION:
            raise ValueError("unsupported V5 exact-refinement result contract")
        if self.scientific_scope != V5_EXACT_REFINEMENT_SCOPE:
            raise ValueError("unexpected V5 exact-refinement scientific scope")
        if self.proposal_score_handoff != V5_PROPOSAL_SCORE_HANDOFF:
            raise ValueError("unexpected V5 proposal-score handoff")
        if type(self.all_input_seeds_processed) is not bool:
            raise TypeError("all_input_seeds_processed must be a bool")
        candidates = tuple(self.candidates)
        attempts = tuple(self.attempts)
        if not all(isinstance(item, CandidateInput) for item in candidates):
            raise TypeError("candidates must contain CandidateInput values")
        if not all(isinstance(item, V5ExactRefinementAttempt) for item in attempts):
            raise TypeError("attempts must contain V5ExactRefinementAttempt values")
        if not isinstance(self.ledger, V5ExactForwardLedger):
            raise TypeError("ledger must be a V5ExactForwardLedger")
        ranks = tuple(item.proposal_rank for item in candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be contiguous from one")
        if tuple(item.attempt_rank for item in attempts) != tuple(range(1, len(attempts) + 1)):
            raise ValueError("attempt ranks must be contiguous from one")
        previous = 0
        for item in attempts:
            if item.cumulative_calls_before != previous:
                raise ValueError("attempt cumulative exact-forward ledgers are not contiguous")
            previous = item.cumulative_calls_after
        success_ids = tuple(item.candidate_id for item in attempts if item.status == "refined")
        if success_ids != tuple(item.candidate_id for item in candidates):
            raise ValueError("successful attempt IDs do not match candidate outputs")
        if self.ledger.calls_used != sum(item.exact_forward_calls for item in attempts):
            raise ValueError("batch and attempt exact-forward ledgers disagree")
        if self.ledger.attempts_recorded != len(attempts):
            raise ValueError("ledger attempt count is inconsistent")
        expected_phases = tuple(
            (phase, sum(dict(item.exact_forward_calls_by_phase)[phase] for item in attempts))
            for phase in EXACT_FORWARD_PHASES
        )
        if self.ledger.calls_by_phase != expected_phases:
            raise ValueError("batch and attempt exact-forward phase ledgers disagree")
        expected_outcomes = (
            sum(item.status == "refined" for item in attempts),
            sum(item.status == "validation_failed" for item in attempts),
            sum(item.status == "refinement_failed" for item in attempts),
            sum(item.status == "per_candidate_forward_budget_exhausted" for item in attempts),
            sum(item.status == "total_forward_budget_exhausted_before_seed" for item in attempts),
        )
        if expected_outcomes != (
            self.ledger.refinement_successes,
            self.ledger.validation_failures,
            self.ledger.refinement_failures,
            self.ledger.per_candidate_budget_exhausted_attempts,
            self.ledger.total_budget_exhausted_before_seed_attempts,
        ):
            raise ValueError("batch and attempt outcome ledgers disagree")
        if self.all_input_seeds_processed != (
            self.ledger.attempts_recorded == self.ledger.input_seed_count
            and self.ledger.total_budget_exhausted_before_seed_attempts == 0
        ):
            raise ValueError("all_input_seeds_processed disagrees with the attempt ledger")
        expected_status: V5ExactBatchStatus
        if candidates:
            expected_status = (
                "exact_candidates_ready_for_verification"
                if self.all_input_seeds_processed
                else "exact_candidates_partial_budget_exhausted"
            )
        else:
            expected_status = "no_candidate_found_within_budget"
        if self.status != expected_status:
            raise ValueError(f"V5 exact-refinement status must be {expected_status!r}")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "attempts", attempts)

    @property
    def evaluation_candidates(self) -> tuple[CandidateInput, ...]:
        """Candidates ready for ``evaluation.evaluate_candidates``."""

        return self.candidates

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "scientific_scope": self.scientific_scope,
            "proposal_score_handoff": self.proposal_score_handoff,
            "status": self.status,
            "query_sha256": self.query_sha256,
            "all_input_seeds_processed": self.all_input_seeds_processed,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "proposal_rank": item.proposal_rank,
                    "topology_id": item.topology_id,
                    "resolution_present": item.resolution is not None,
                }
                for item in self.candidates
            ],
            "attempts": [asdict(item) for item in self.attempts],
            "exact_forward_ledger": asdict(self.ledger),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_audit_dict(), sort_keys=True, allow_nan=False, ensure_ascii=False, indent=indent
        )


__all__ = [
    "EXACT_FORWARD_PHASES",
    "V5_EXACT_REFINEMENT_SCHEMA",
    "V5_EXACT_REFINEMENT_SCOPE",
    "V5_EXACT_REFINEMENT_VERSION",
    "V5_EXACT_FORWARD_BUDGET_UNIT",
    "V5CandidatePrerequisiteAudit",
    "V5ExactForwardLedger",
    "V5ExactRefinementAttempt",
    "V5ExactRefinementBatchResult",
    "exact_forward_phase_counts",
]
