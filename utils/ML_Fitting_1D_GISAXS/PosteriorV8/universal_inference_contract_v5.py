"""Auditable result contract for V5.2 universal one-click inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from .candidate_refinement_contract_v5 import V5AttemptStatus, V5SeedSource
from .evaluation import CandidateInput, EvaluationReport
from .model_v5_contract import MODEL_V5_NAME, MODEL_V5_SCHEMA, MODEL_V5_VERSION
from .proposal_execution_policy_v5 import (
    V5_PROPOSAL_BRANCH_RANKING,
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    validate_v5_proposal_execution_policy_sha256,
)
from .query_bound_evaluation_v5 import (
    V5_QUERY_BOUND_EVALUATION_SCHEMA,
    V5_QUERY_BOUND_EVALUATION_SHA256,
    V5_QUERY_BOUND_EVALUATION_VERSION,
    V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
)
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
)


V5_UNIVERSAL_INFERENCE_SCHEMA = "gisaxs.posterior_v8.universal_one_click_inference/v3"
V5_UNIVERSAL_INFERENCE_VERSION = (
    "posterior_v8_v5_2_r2_query_bound_policy_bound_branch_round_robin_"
    "complete_linkage_mdn_verified_rescue_v6"
)
V5_UNIVERSAL_INFERENCE_BUDGET_SCHEMA = (
    "gisaxs.posterior_v8.universal_one_click_inference_budget/v2"
)
V5_UNIVERSAL_INFERENCE_BUDGET_VERSION = (
    "proposal_execution_policy_bound_exact_forward_limits_v2"
)
V5_UNIVERSAL_BRANCH_RANKING = V5_PROPOSAL_BRANCH_RANKING
V5_UNIVERSAL_SEED_SCHEDULE = (
    "ranked_branch_round_robin_neural_primary_then_caller_ordered_retrieval_"
    "then_ranked_branch_round_robin_sobol_then_remaining_ranked_branch_round_"
    "robin_neural_v2"
)
V5_UNIVERSAL_MODE_STOP = (
    "stop_only_on_exact_compatible_query_bound_complete_linkage_diameter_cluster_"
    "representatives_or_exact_"
    "forward_budget_or_seed_schedule_never_raw_candidate_count_or_observability_v1"
)
V5_UNIVERSAL_RESULT_CLAIM = (
    "finite_budget_forward_model_verified_candidates_not_posterior_not_all_"
    "mathematical_solutions_and_zero_found_is_not_no_solution_v1"
)

V5UniversalStatus = Literal[
    "compatible_target_reached",
    "compatible_partial",
    "no_compatible_mode_found_within_budget",
    "no_candidate_found_within_budget",
]
V5TerminationReason = Literal[
    "compatible_target_reached",
    "exact_forward_budget_exhausted",
    "seed_schedule_exhausted",
]
V5AttemptStage = Literal[
    "neural_primary",
    "retrieval_rescue",
    "sobol_rescue",
    "neural_spillover",
]


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True, kw_only=True)
class V5UniversalInferenceBudget:
    """Frozen proposal and exact-forward limits for one universal query."""

    mixture_components_per_branch: int = 4
    stochastic_draws_per_mixture: int = 1
    include_mixture_medians: bool = True
    neural_primary_attempt_limit: int = 16
    fallback_attempt_limit: int = 64
    sobol_seeds_per_branch: int = 2
    per_candidate_forward_evaluation_limit: int = 128
    forward_evaluation_limit: int = 4096
    proposal_execution_policy_sha256: str = V5_PROPOSAL_EXECUTION_POLICY_SHA256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mixture_components_per_branch",
            _integer(
                self.mixture_components_per_branch,
                "mixture_components_per_branch",
                minimum=1,
            ),
        )
        for name in (
            "stochastic_draws_per_mixture",
            "neural_primary_attempt_limit",
            "fallback_attempt_limit",
            "sobol_seeds_per_branch",
            "forward_evaluation_limit",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        object.__setattr__(
            self,
            "per_candidate_forward_evaluation_limit",
            _integer(
                self.per_candidate_forward_evaluation_limit,
                "per_candidate_forward_evaluation_limit",
                minimum=1,
            ),
        )
        if type(self.include_mixture_medians) is not bool:
            raise TypeError("include_mixture_medians must be a bool")
        if (
            self.include_mixture_medians
            is not V5_PROPOSAL_EXECUTION_POLICY.include_mixture_medians
        ):
            raise ValueError(
                "include_mixture_medians must match the frozen proposal execution policy"
            )
        if (
            self.mixture_components_per_branch
            != V5_PROPOSAL_EXECUTION_POLICY.per_branch_top_l
        ):
            raise ValueError(
                "mixture_components_per_branch must match the frozen proposal "
                "execution policy"
            )
        object.__setattr__(
            self,
            "proposal_execution_policy_sha256",
            validate_v5_proposal_execution_policy_sha256(
                self.proposal_execution_policy_sha256
            ),
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "schema": V5_UNIVERSAL_INFERENCE_BUDGET_SCHEMA,
            "version": V5_UNIVERSAL_INFERENCE_BUDGET_VERSION,
            "proposal_execution_policy": V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
        }


@dataclass(frozen=True, kw_only=True)
class V5UniversalAttemptAudit:
    attempt_rank: int
    stage: V5AttemptStage
    source: V5SeedSource
    source_id: str
    global_branch_key: str
    model_global_index: int
    topology_id: int
    pattern_id: int
    exact_status: V5AttemptStatus
    exact_forward_calls: int
    cumulative_calls_before: int
    cumulative_calls_after: int
    candidate_id: str | None
    message: str
    search_yield_logit: float | None = None
    mixture_log_weight: float | None = None

    def __post_init__(self) -> None:
        for name, minimum in (
            ("attempt_rank", 1),
            ("model_global_index", 0),
            ("topology_id", 0),
            ("pattern_id", 0),
            ("exact_forward_calls", 0),
            ("cumulative_calls_before", 0),
            ("cumulative_calls_after", 0),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        if self.stage not in {
            "neural_primary",
            "retrieval_rescue",
            "sobol_rescue",
            "neural_spillover",
        }:
            raise ValueError("unknown universal inference attempt stage")
        expected_source = {
            "neural_primary": "neural",
            "retrieval_rescue": "retrieval",
            "sobol_rescue": "sobol",
            "neural_spillover": "neural",
        }[self.stage]
        if self.source != expected_source:
            raise ValueError("attempt stage and seed source disagree")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("attempt message must be non-empty")
        expected_key = f"topology-{self.topology_id:02d}:wire-{self.pattern_id:02d}"
        if self.global_branch_key != expected_key:
            raise ValueError("attempt global branch key disagrees with topology/pattern")
        if self.cumulative_calls_after != (self.cumulative_calls_before + self.exact_forward_calls):
            raise ValueError("attempt cumulative exact-forward ledger is inconsistent")
        if (self.candidate_id is not None) != (self.exact_status == "refined"):
            raise ValueError("only a refined attempt may claim a candidate")
        for name in ("search_yield_logit", "mixture_log_weight"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")


@dataclass(frozen=True, kw_only=True)
class V5UniversalCandidateProvenance:
    candidate_id: str
    proposal_rank: int
    attempt_rank: int
    stage: V5AttemptStage
    source: V5SeedSource
    source_id: str
    global_branch_key: str
    topology_id: int
    pattern_id: int
    model_global_index: int
    search_yield_logit: float | None
    mixture_log_weight: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate provenance candidate_id must be non-empty")
        for name, minimum in (
            ("proposal_rank", 1),
            ("attempt_rank", 1),
            ("topology_id", 0),
            ("pattern_id", 0),
            ("model_global_index", 0),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        if self.stage not in {
            "neural_primary",
            "retrieval_rescue",
            "sobol_rescue",
            "neural_spillover",
        }:
            raise ValueError("unknown candidate provenance stage")
        expected_source = {
            "neural_primary": "neural",
            "retrieval_rescue": "retrieval",
            "sobol_rescue": "sobol",
            "neural_spillover": "neural",
        }[self.stage]
        if self.source != expected_source:
            raise ValueError("candidate provenance stage and source disagree")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("candidate provenance source_id must be non-empty")
        expected_key = f"topology-{self.topology_id:02d}:wire-{self.pattern_id:02d}"
        if self.global_branch_key != expected_key:
            raise ValueError("candidate provenance branch key disagrees with topology/pattern")
        for name in ("search_yield_logit", "mixture_log_weight"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"candidate provenance {name} must be finite or None")


@dataclass(frozen=True, kw_only=True)
class V5UniversalSourceAccounting:
    source: V5SeedSource
    potential_seed_count: int
    materialized_seed_count: int
    attempts: int
    candidates_returned: int
    exact_forward_calls: int

    def __post_init__(self) -> None:
        if self.source not in {"neural", "retrieval", "sobol"}:
            raise ValueError("unknown source accounting row")
        for name in (
            "potential_seed_count",
            "materialized_seed_count",
            "attempts",
            "candidates_returned",
            "exact_forward_calls",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        if self.materialized_seed_count > self.potential_seed_count:
            raise ValueError("materialized seeds cannot exceed potential seeds")
        if self.attempts > self.materialized_seed_count:
            raise ValueError("attempted seeds cannot exceed materialized seeds")
        if self.candidates_returned > self.attempts:
            raise ValueError("returned candidates cannot exceed attempts")


@dataclass(frozen=True, kw_only=True)
class V5UniversalInferenceResult:
    status: V5UniversalStatus
    termination_reason: V5TerminationReason
    target_parameter_mode_count: int
    compatible_parameter_mode_count: int
    compatible_representative_candidate_ids: tuple[str, ...]
    context_audit_sha256: str
    globally_ranked_branch_keys: tuple[str, ...]
    candidates: tuple[CandidateInput, ...]
    candidate_provenance: tuple[V5UniversalCandidateProvenance, ...]
    attempts: tuple[V5UniversalAttemptAudit, ...]
    source_accounting: tuple[V5UniversalSourceAccounting, ...]
    evaluation_report: EvaluationReport | None
    model_call_count: int
    evaluation_call_count: int
    forward_evaluation_limit: int
    forward_evaluations_used: int
    forward_evaluations_remaining: int
    all_scheduled_seeds_processed: bool
    schema: str = V5_UNIVERSAL_INFERENCE_SCHEMA
    version: str = V5_UNIVERSAL_INFERENCE_VERSION
    branch_ranking: str = V5_UNIVERSAL_BRANCH_RANKING
    seed_schedule: str = V5_UNIVERSAL_SEED_SCHEDULE
    mode_stop_semantics: str = V5_UNIVERSAL_MODE_STOP
    scientific_claim: str = V5_UNIVERSAL_RESULT_CLAIM
    model_schema: str = MODEL_V5_SCHEMA
    model_version: str = MODEL_V5_VERSION
    model_name: str = MODEL_V5_NAME
    query_bound_evaluator_schema: str = V5_QUERY_BOUND_EVALUATION_SCHEMA
    query_bound_evaluator_version: str = V5_QUERY_BOUND_EVALUATION_VERSION
    query_bound_evaluator_sha256: str = V5_QUERY_BOUND_EVALUATION_SHA256
    proposal_execution_policy_sha256: str = V5_PROPOSAL_EXECUTION_POLICY_SHA256

    def __post_init__(self) -> None:
        if self.schema != V5_UNIVERSAL_INFERENCE_SCHEMA:
            raise ValueError("unsupported universal inference schema")
        if self.version != V5_UNIVERSAL_INFERENCE_VERSION:
            raise ValueError("unsupported universal inference version")
        if self.branch_ranking != V5_UNIVERSAL_BRANCH_RANKING:
            raise ValueError("unexpected universal branch-ranking semantics")
        if self.seed_schedule != V5_UNIVERSAL_SEED_SCHEDULE:
            raise ValueError("unexpected universal seed schedule")
        if self.mode_stop_semantics != V5_UNIVERSAL_MODE_STOP:
            raise ValueError("unexpected universal stopping semantics")
        if self.scientific_claim != V5_UNIVERSAL_RESULT_CLAIM:
            raise ValueError("unexpected universal scientific claim")
        if (
            self.model_schema != MODEL_V5_SCHEMA
            or self.model_version != MODEL_V5_VERSION
            or self.model_name != MODEL_V5_NAME
        ):
            raise ValueError("unexpected V5 model contract identity")
        if self.query_bound_evaluator_schema != V5_QUERY_BOUND_EVALUATION_SCHEMA:
            raise ValueError("unexpected V5 query-bound evaluator schema")
        if self.query_bound_evaluator_version != V5_QUERY_BOUND_EVALUATION_VERSION:
            raise ValueError("unexpected V5 query-bound evaluator version")
        if self.query_bound_evaluator_sha256 != V5_QUERY_BOUND_EVALUATION_SHA256:
            raise ValueError("unexpected V5 query-bound evaluator digest")
        object.__setattr__(
            self,
            "proposal_execution_policy_sha256",
            validate_v5_proposal_execution_policy_sha256(
                self.proposal_execution_policy_sha256
            ),
        )
        target = _integer(
            self.target_parameter_mode_count, "target_parameter_mode_count", minimum=1
        )
        count = _integer(
            self.compatible_parameter_mode_count,
            "compatible_parameter_mode_count",
            minimum=0,
        )
        candidates = tuple(self.candidates)
        ids = tuple(value.candidate_id for value in candidates)
        if ids != tuple(f"candidate_{index:05d}" for index in range(1, len(ids) + 1)):
            raise ValueError("universal candidate IDs must be stable and contiguous")
        if tuple(value.proposal_rank for value in candidates) != tuple(
            range(1, len(candidates) + 1)
        ):
            raise ValueError("universal candidate ranks must be contiguous")
        provenance = tuple(self.candidate_provenance)
        if not all(isinstance(value, V5UniversalCandidateProvenance) for value in provenance):
            raise TypeError("candidate_provenance must contain V5 provenance rows")
        if tuple(value.candidate_id for value in provenance) != ids:
            raise ValueError("candidate provenance does not match candidate order")
        attempts = tuple(self.attempts)
        if tuple(value.attempt_rank for value in attempts) != tuple(range(1, len(attempts) + 1)):
            raise ValueError("universal attempt ranks must be contiguous")
        for candidate, row in zip(candidates, provenance):
            if row.attempt_rank > len(attempts):
                raise ValueError("candidate provenance points outside the attempt ledger")
            attempt = attempts[row.attempt_rank - 1]
            provenance_identity = (
                row.candidate_id,
                row.proposal_rank,
                row.stage,
                row.source,
                row.source_id,
                row.global_branch_key,
                row.topology_id,
                row.pattern_id,
                row.model_global_index,
                row.search_yield_logit,
                row.mixture_log_weight,
            )
            attempt_identity = (
                attempt.candidate_id,
                candidate.proposal_rank,
                attempt.stage,
                attempt.source,
                attempt.source_id,
                attempt.global_branch_key,
                attempt.topology_id,
                attempt.pattern_id,
                attempt.model_global_index,
                attempt.search_yield_logit,
                attempt.mixture_log_weight,
            )
            if provenance_identity != attempt_identity:
                raise ValueError("candidate provenance disagrees with its refined attempt")
        used = _integer(self.forward_evaluations_used, "forward_evaluations_used", minimum=0)
        limit = _integer(self.forward_evaluation_limit, "forward_evaluation_limit", minimum=0)
        remaining = _integer(
            self.forward_evaluations_remaining,
            "forward_evaluations_remaining",
            minimum=0,
        )
        if used + remaining != limit or used != sum(
            value.exact_forward_calls for value in attempts
        ):
            raise ValueError("universal exact-forward ledger is inconsistent")
        if _integer(self.model_call_count, "model_call_count", minimum=0) != 1:
            raise ValueError("universal inference requires exactly one batched model call")
        if _integer(self.evaluation_call_count, "evaluation_call_count", minimum=0) != len(
            candidates
        ):
            raise ValueError("every successful exact candidate must be evaluated once")
        if type(self.all_scheduled_seeds_processed) is not bool:
            raise TypeError("all_scheduled_seeds_processed must be a bool")
        report = self.evaluation_report
        if candidates:
            if report is None or tuple(value.candidate_id for value in report.candidates) != ids:
                raise ValueError("final evaluation does not cover every exact candidate")
            if (
                report.audit_schema != V5_QUERY_BOUND_EVALUATION_SCHEMA
                or report.parameter_normalization_version
                != V5_QUERY_PARAMETER_DISTANCE_VERSION
                or report.parameter_distance_scope != V5_QUERY_PARAMETER_DISTANCE_SCOPE
                or report.reference_matching_version
                != V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION
            ):
                raise ValueError("final evaluation is not the frozen V5 query-bound evaluator")
            for candidate, assessment in zip(candidates, report.candidates):
                candidate_identity = (
                    candidate.proposal_rank,
                    candidate.topology_id,
                    candidate.components,
                    candidate.resolution,
                    candidate.linear_solution,
                    candidate.proposal_score_raw,
                    candidate.bounds_pass,
                    candidate.physics_pass,
                )
                assessment_identity = (
                    assessment.proposal_rank,
                    assessment.topology_id,
                    assessment.components,
                    assessment.resolution,
                    assessment.linear_solution,
                    assessment.proposal_score_raw,
                    assessment.bounds_pass,
                    assessment.physics_pass,
                )
                if candidate_identity != assessment_identity:
                    raise ValueError("final evaluation candidate content disagrees with outputs")
            expected_ids = tuple(
                value.representative_candidate_id for value in report.parameter_modes
            )
            if count != len(report.parameter_modes):
                raise ValueError("compatible mode count disagrees with final evaluation")
        else:
            if report is not None or count:
                raise ValueError("an empty exact result cannot have an evaluation or modes")
            expected_ids = ()
        if tuple(self.compatible_representative_candidate_ids) != expected_ids:
            raise ValueError("compatible representative IDs disagree with final evaluation")
        expected_status: V5UniversalStatus
        if count >= target:
            expected_status = "compatible_target_reached"
        elif count:
            expected_status = "compatible_partial"
        elif candidates:
            expected_status = "no_compatible_mode_found_within_budget"
        else:
            expected_status = "no_candidate_found_within_budget"
        if self.status != expected_status:
            raise ValueError(f"universal inference status must be {expected_status!r}")
        if (self.termination_reason == "compatible_target_reached") != (count >= target):
            raise ValueError("target termination disagrees with compatible mode count")
        accounting = tuple(self.source_accounting)
        if tuple(value.source for value in accounting) != ("neural", "retrieval", "sobol"):
            raise ValueError("source accounting must use stable neural/retrieval/Sobol order")
        if sum(value.exact_forward_calls for value in accounting) != used:
            raise ValueError("source and universal exact-forward ledgers disagree")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "candidate_provenance", provenance)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "source_accounting", accounting)

    @property
    def compatible_representatives(self) -> tuple[CandidateInput, ...]:
        """Exact-compatible parameter-cluster representatives for the GUI."""

        selected = set(self.compatible_representative_candidate_ids)
        return tuple(value for value in self.candidates if value.candidate_id in selected)

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "scientific_claim": self.scientific_claim,
            "branch_ranking": self.branch_ranking,
            "seed_schedule": self.seed_schedule,
            "mode_stop_semantics": self.mode_stop_semantics,
            "model_contract": {
                "schema": self.model_schema,
                "version": self.model_version,
                "name": self.model_name,
            },
            "query_bound_evaluator": {
                "schema": self.query_bound_evaluator_schema,
                "version": self.query_bound_evaluator_version,
                "sha256": self.query_bound_evaluator_sha256,
            },
            "proposal_execution_policy": V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
            "proposal_execution_policy_sha256": (
                self.proposal_execution_policy_sha256
            ),
            "target_parameter_mode_count": self.target_parameter_mode_count,
            "compatible_parameter_mode_count": self.compatible_parameter_mode_count,
            "compatible_representative_candidate_ids": list(
                self.compatible_representative_candidate_ids
            ),
            "context_audit_sha256": self.context_audit_sha256,
            "globally_ranked_branch_keys": list(self.globally_ranked_branch_keys),
            "candidate_ids": [value.candidate_id for value in self.candidates],
            "candidate_provenance": [asdict(value) for value in self.candidate_provenance],
            "attempts": [asdict(value) for value in self.attempts],
            "source_accounting": [asdict(value) for value in self.source_accounting],
            "evaluation": (
                None if self.evaluation_report is None else self.evaluation_report.to_audit_dict()
            ),
            "model_call_count": self.model_call_count,
            "evaluation_call_count": self.evaluation_call_count,
            "exact_forward_budget": {
                "limit": self.forward_evaluation_limit,
                "used": self.forward_evaluations_used,
                "remaining": self.forward_evaluations_remaining,
            },
            "all_scheduled_seeds_processed": self.all_scheduled_seeds_processed,
        }


__all__ = [
    "V5_UNIVERSAL_BRANCH_RANKING",
    "V5_UNIVERSAL_INFERENCE_SCHEMA",
    "V5_UNIVERSAL_INFERENCE_VERSION",
    "V5_UNIVERSAL_INFERENCE_BUDGET_SCHEMA",
    "V5_UNIVERSAL_INFERENCE_BUDGET_VERSION",
    "V5_UNIVERSAL_MODE_STOP",
    "V5_UNIVERSAL_RESULT_CLAIM",
    "V5_UNIVERSAL_SEED_SCHEDULE",
    "V5AttemptStage",
    "V5TerminationReason",
    "V5UniversalAttemptAudit",
    "V5UniversalCandidateProvenance",
    "V5UniversalInferenceBudget",
    "V5UniversalInferenceResult",
    "V5UniversalSourceAccounting",
    "V5UniversalStatus",
]
