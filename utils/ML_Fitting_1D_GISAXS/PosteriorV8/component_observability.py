"""Public observability contract for Posterior V8.

Delete/reprofile diagnostics live in ``component_observability_diagnostics``;
fail-closed decisions and fair reduced-search budgeting live in
``observability_assessment``.  This stable facade keeps the original public
imports while making the scientific responsibilities independently testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Literal, Protocol, Sequence

import numpy as np

from .evaluation import CandidateInput, ObservedCurve
from .profiled_forward import ProfiledForwardResult


COMPONENT_OBSERVABILITY_VERSION = "posterior_v8_delete_reprofile_scale_invariant_observability_v2"
OBSERVABILITY_POLICY_VERSION = "posterior_v8_nested_model_observability_policy_v2"
PROFILE_WEIGHTING_SEMANTICS = (
    "weighted_linear_profile_with_sigma_I_proxy_equal_to_observed_times_sigma_log;"
    " clean_curves_use_observed_as_the_relative_linear_scale"
)
DELETION_EVALUATED = "evaluated"
DELETION_K0_EVALUATED = "evaluated_k0_background_resolution_nested_null"
DELETION_ZERO_REMAINING_PARTICLE_AMPLITUDE = "not_evaluated_zero_remaining_particle_amplitude"
DELETION_FORWARD_BUDGET_EXHAUSTED = "not_evaluated_forward_budget_exhausted"
DELETION_PROFILE_FAILED = "not_evaluated_profile_failed"

ObservabilityDecision = Literal[
    "needed",
    "unneeded",
    "provisional_needed",
    "unknown",
    "not_present",
]
ObservabilityStatus = Literal[
    "confirmed_effective",
    "confirmed_redundant",
    "provisional_or_unknown",
]


@dataclass(frozen=True)
class ExactLogScores:
    """Exact GUI-forward scores against one observed curve."""

    raw_log_rmse: float
    standardized_log_rmse: float | None


@dataclass(frozen=True)
class ContributionDiagnostic:
    """Dimensionless size of one additive curve contribution."""

    label: str
    amplitude: float
    relative_to_observed_rms: float
    relative_to_model_rms: float
    measurement_standardized_rms: float | None


@dataclass(frozen=True)
class ComponentDeletionDiagnostic:
    """Observability evidence for one particle in original component order."""

    component_index: int
    shape: str
    contribution: ContributionDiagnostic
    deletion_status: str
    remaining_component_indices: tuple[int, ...]
    reduced_profile: ProfiledForwardResult | K0ProfiledForwardResult | None
    reduced_scores: ExactLogScores | None
    raw_log_score_increment: float | None
    standardized_log_score_increment: float | None


@dataclass(frozen=True)
class K0ProfiledForwardResult:
    """Non-negative BG +/- fixed-Resolution nested null for diagnostics."""

    resolution_present: bool
    background: float
    resolution_amplitude: float
    fitted_intensity: np.ndarray
    solver_status: int
    solver_message: str
    solver_optimality: float


@dataclass(frozen=True)
class ResolutionDeletionDiagnostic:
    """Fixed-particle-geometry deletion/reprofile evidence for Resolution."""

    deletion_status: str
    reduced_profile: ProfiledForwardResult | None
    reduced_scores: ExactLogScores | None
    raw_log_score_increment: float | None
    standardized_log_score_increment: float | None


@dataclass(frozen=True)
class ComponentObservabilityReport:
    """Scale-invariant deletion and contribution diagnostics for one profile."""

    version: str
    profile_weighting_semantics: str
    full_scores: ExactLogScores
    full_exact_intensity: np.ndarray
    background_contribution: ContributionDiagnostic
    resolution_contribution: ContributionDiagnostic | None
    components: tuple[ComponentDeletionDiagnostic, ...]
    resolution_deletion: ResolutionDeletionDiagnostic | None
    exact_forward_calls: int
    exact_forward_call_limit: int | None
    reused_authoritative_full_exact: bool


@dataclass(frozen=True, kw_only=True)
class ObservabilityPolicy:
    """Versioned interpretation and finite reduced-search budget policy.

    Compatible reduced fits prove a term unnecessary.  A finite search that
    finds none is provisional.  Only a Resolution-absent K1 -> non-negative-BG
    K0 null is exhaustive under this contract.
    """

    exact_forward_call_limit: int = 64
    reduced_search_starts: int = 3
    reduced_search_per_start_forward_limit: int = 16
    solver_tolerance: float = 1.0e-12
    max_iterations: int | None = None
    version: str = OBSERVABILITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVABILITY_POLICY_VERSION:
            raise ValueError("unsupported observability policy version")
        if isinstance(self.exact_forward_call_limit, bool) or not isinstance(
            self.exact_forward_call_limit, Integral
        ):
            raise TypeError("exact_forward_call_limit must be an integer")
        if int(self.exact_forward_call_limit) < 0:
            raise ValueError("exact_forward_call_limit must be non-negative")
        for name in (
            "reduced_search_starts",
            "reduced_search_per_start_forward_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        tolerance = float(self.solver_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("solver_tolerance must be finite and positive")
        if isinstance(self.max_iterations, bool) or (
            self.max_iterations is not None
            and (not isinstance(self.max_iterations, Integral) or int(self.max_iterations) < 1)
        ):
            raise ValueError("max_iterations must be a positive integer or None")
        object.__setattr__(self, "exact_forward_call_limit", int(self.exact_forward_call_limit))
        object.__setattr__(self, "solver_tolerance", tolerance)
        if self.max_iterations is not None:
            object.__setattr__(self, "max_iterations", int(self.max_iterations))


@dataclass(frozen=True, kw_only=True)
class FeatureObservabilityAssessment:
    label: str
    decision: ObservabilityDecision
    deletion_status: str
    reduced_primary_score: float | None
    primary_compatibility_threshold: float
    evidence_scope: str
    reduced_search_status: str | None = None
    reduced_search_best_primary_score: float | None = None
    reduced_search_forward_call_limit: int = 0
    reduced_search_forward_calls: int = 0
    reduced_search_attempts: int = 0
    reduced_search_evidence: ReducedModelSearchEvidence | None = None


@dataclass(frozen=True, kw_only=True)
class ReducedModelSearchAttempt:
    start_index: int
    source: str
    status: str
    forward_calls: int
    primary_score: float | None
    detail: str | None = None


@dataclass(frozen=True, kw_only=True)
class ReducedModelSearchEvidence:
    """Counted finite-budget evidence from an independently refined null."""

    label: str
    status: Literal[
        "compatible_reduced_model_found",
        "completed_no_compatible_reduced_model",
        "budget_exhausted",
        "search_unavailable",
    ]
    best_primary_score: float | None
    primary_compatibility_threshold: float
    forward_calls: int
    attempted_starts: int
    required_starts: int
    search_version: str
    attempts: tuple[ReducedModelSearchAttempt, ...]
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("reduced-model evidence label must be non-empty")
        if self.status not in {
            "compatible_reduced_model_found",
            "completed_no_compatible_reduced_model",
            "budget_exhausted",
            "search_unavailable",
        }:
            raise ValueError("invalid reduced-model search status")
        for name in ("forward_calls", "attempted_starts", "required_starts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.required_starts < 1:
            raise ValueError("required_starts must be positive")
        attempts = tuple(self.attempts)
        if len(attempts) != self.attempted_starts or not all(
            isinstance(item, ReducedModelSearchAttempt) for item in attempts
        ):
            raise ValueError("reduced-model attempt audit is inconsistent")
        if sum(item.forward_calls for item in attempts) != self.forward_calls:
            raise ValueError("reduced-model forward-call audit is inconsistent")
        object.__setattr__(self, "attempts", attempts)
        if self.best_primary_score is not None and (
            not np.isfinite(float(self.best_primary_score)) or float(self.best_primary_score) < 0.0
        ):
            raise ValueError("best_primary_score must be finite and non-negative")
        threshold = float(self.primary_compatibility_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("primary_compatibility_threshold is invalid")
        if not isinstance(self.search_version, str) or not self.search_version:
            raise ValueError("search_version must be non-empty")


class ReducedModelSearchPort(Protocol):
    def search(
        self,
        curve: ObservedCurve,
        candidate: CandidateInput,
        *,
        label: str,
        primary_metric_name: str,
        primary_compatibility_threshold: float,
        max_forward_evaluations: int,
        required_starts: int,
        per_start_forward_evaluation_limit: int,
    ) -> ReducedModelSearchEvidence: ...


@dataclass(frozen=True, kw_only=True)
class CandidateObservabilityAssessment:
    """Fail-closed effective-model assessment for one exact candidate."""

    candidate_id: str
    status: ObservabilityStatus
    primary_metric_name: str
    primary_compatibility_threshold: float
    full_primary_score: float
    full_curve_compatible: bool
    particles: tuple[FeatureObservabilityAssessment, ...]
    resolution: FeatureObservabilityAssessment
    d_terms: tuple[FeatureObservabilityAssessment, ...]
    exact_forward_calls: int
    exact_forward_call_limit: int
    budget_exhausted: bool
    diagnostic_error: str | None
    policy_version: str
    diagnostic_version: str

    @property
    def confirmed_effective(self) -> bool:
        return self.status == "confirmed_effective"

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def diagnose_component_observability(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    profile: ProfiledForwardResult,
    *,
    sigma_log: Sequence[float] | np.ndarray | None = None,
    solver_tolerance: float = 1.0e-12,
    max_iterations: int | None = None,
    authoritative_full_exact: Sequence[float] | np.ndarray | None = None,
    exact_forward_call_limit: int | None = None,
) -> ComponentObservabilityReport:
    from .component_observability_diagnostics import (
        diagnose_component_observability as implementation,
    )

    return implementation(
        q,
        intensity,
        profile,
        sigma_log=sigma_log,
        solver_tolerance=solver_tolerance,
        max_iterations=max_iterations,
        authoritative_full_exact=authoritative_full_exact,
        exact_forward_call_limit=exact_forward_call_limit,
    )


def assess_candidate_observability(
    curve: ObservedCurve,
    candidate: CandidateInput,
    *,
    primary_metric_name: str,
    primary_compatibility_threshold: float,
    policy: ObservabilityPolicy = ObservabilityPolicy(),
    exact_forward_call_limit: int | None = None,
    reduced_model_searcher: ReducedModelSearchPort | None = None,
) -> CandidateObservabilityAssessment:
    from .observability_assessment import (
        assess_candidate_observability as implementation,
    )

    return implementation(
        curve,
        candidate,
        primary_metric_name=primary_metric_name,
        primary_compatibility_threshold=primary_compatibility_threshold,
        policy=policy,
        exact_forward_call_limit=exact_forward_call_limit,
        reduced_model_searcher=reduced_model_searcher,
    )


__all__ = [
    "COMPONENT_OBSERVABILITY_VERSION",
    "DELETION_EVALUATED",
    "DELETION_FORWARD_BUDGET_EXHAUSTED",
    "DELETION_K0_EVALUATED",
    "DELETION_PROFILE_FAILED",
    "DELETION_ZERO_REMAINING_PARTICLE_AMPLITUDE",
    "OBSERVABILITY_POLICY_VERSION",
    "PROFILE_WEIGHTING_SEMANTICS",
    "CandidateObservabilityAssessment",
    "ComponentDeletionDiagnostic",
    "ComponentObservabilityReport",
    "ContributionDiagnostic",
    "ExactLogScores",
    "FeatureObservabilityAssessment",
    "K0ProfiledForwardResult",
    "ObservabilityDecision",
    "ObservabilityPolicy",
    "ObservabilityStatus",
    "ResolutionDeletionDiagnostic",
    "ReducedModelSearchAttempt",
    "ReducedModelSearchEvidence",
    "ReducedModelSearchPort",
    "assess_candidate_observability",
    "diagnose_component_observability",
]
