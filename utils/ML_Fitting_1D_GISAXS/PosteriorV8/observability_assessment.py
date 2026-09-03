"""Fail-closed observability decisions and fair reduced-search budgeting."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

import numpy as np

from .component_observability import (
    COMPONENT_OBSERVABILITY_VERSION,
    DELETION_FORWARD_BUDGET_EXHAUSTED,
    DELETION_PROFILE_FAILED,
    CandidateObservabilityAssessment,
    ExactLogScores,
    FeatureObservabilityAssessment,
    ObservabilityDecision,
    ObservabilityPolicy,
    ObservabilityStatus,
    ReducedModelSearchEvidence,
    ReducedModelSearchPort,
)
from .component_observability_diagnostics import (
    diagnose_component_observability,
    exact_log_scores,
)
from .contract import latent_component_to_gui
from .evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    CandidateInput,
    ObservedCurve,
)
from .profiled_forward import ProfiledForwardResult


def _profile_from_candidate(
    curve: ObservedCurve, candidate: CandidateInput
) -> ProfiledForwardResult:
    components = tuple(latent_component_to_gui(value) for value in candidate.components)
    linear = candidate.linear_solution
    exact = np.asarray(candidate.exact_intensity, dtype=np.float64)
    residual = exact - curve.intensity
    sigma = curve.intensity if curve.sigma_log is None else curve.intensity * curve.sigma_log
    return ProfiledForwardResult(
        components=components,
        resolution=candidate.resolution,
        background=linear.background,
        particle_amplitudes=linear.particle_amplitudes,
        resolution_amplitude=linear.resolution_amplitude,
        k=linear.k,
        component_weights=linear.component_weights,
        int_res=linear.int_res,
        fitted_intensity=exact,
        residual=residual,
        weighted_residual=residual / sigma,
        weighted_rss=float(np.sum(np.square(residual / sigma))),
        solver_status=0,
        solver_message="authoritative CandidateInput snapshot",
        solver_optimality=0.0,
    )


def _primary_score(scores: ExactLogScores, metric_name: str) -> float:
    if metric_name == RAW_LOG_RMSE_METRIC:
        return scores.raw_log_rmse
    if metric_name == STANDARDIZED_LOG_RMSE_METRIC:
        if scores.standardized_log_rmse is None:
            raise ValueError("standardized observability requires sigma_log")
        return scores.standardized_log_rmse
    raise ValueError("unsupported primary observability metric")


def _feature_assessment(
    *,
    label: str,
    deletion_status: str,
    reduced_scores: ExactLogScores | None,
    metric_name: str,
    threshold: float,
    exhaustive_null: bool,
) -> FeatureObservabilityAssessment:
    reduced_score = None if reduced_scores is None else _primary_score(reduced_scores, metric_name)
    if reduced_score is None:
        decision: ObservabilityDecision = "unknown"
        scope = "deletion_not_scored"
    elif reduced_score <= threshold:
        decision = "unneeded"
        scope = "compatible_delete_and_reprofile_nested_model"
    elif exhaustive_null:
        decision = "needed"
        scope = "exhaustive_k0_nested_null"
    else:
        decision = "provisional_needed"
        scope = "fixed_geometry_reduced_model_only"
    return FeatureObservabilityAssessment(
        label=label,
        decision=decision,
        deletion_status=deletion_status,
        reduced_primary_score=reduced_score,
        primary_compatibility_threshold=threshold,
        evidence_scope=scope,
    )


def _with_reduced_search_evidence(
    assessment: FeatureObservabilityAssessment,
    evidence: ReducedModelSearchEvidence,
    *,
    allocated_forward_calls: int,
) -> FeatureObservabilityAssessment:
    if evidence.label != assessment.label:
        raise ValueError("reduced-model evidence label does not match feature")
    if evidence.status == "compatible_reduced_model_found":
        decision: ObservabilityDecision = "unneeded"
        scope = "compatible_independently_refined_reduced_model"
    elif evidence.status == "completed_no_compatible_reduced_model":
        decision = "provisional_needed"
        scope = "finite_reduced_search_failed_without_global_certificate"
    else:
        decision = "unknown"
        scope = "reduced_search_incomplete_or_unavailable"
    return replace(
        assessment,
        decision=decision,
        evidence_scope=scope,
        reduced_search_status=evidence.status,
        reduced_search_best_primary_score=evidence.best_primary_score,
        reduced_search_forward_call_limit=allocated_forward_calls,
        reduced_search_forward_calls=evidence.forward_calls,
        reduced_search_attempts=evidence.attempted_starts,
        reduced_search_evidence=evidence,
    )


def _fair_search_allocations(
    features: tuple[FeatureObservabilityAssessment, ...],
    total_forward_calls: int,
) -> dict[str, int]:
    """Reserve equal feature budgets before running any sequential search."""

    if not features:
        return {}
    quotient, remainder = divmod(total_forward_calls, len(features))
    return {item.label: quotient + int(index < remainder) for index, item in enumerate(features)}


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
    """Assess whether every declared term is demonstrably needed.

    A compatible reduced fit proves ``unneeded``.  Only the
    Resolution-absent K1 -> BG-only K0 null can prove ``needed`` here.  All
    finite nonlinear search failures remain provisional, and every exact
    forward call is charged to a preallocated per-feature budget.
    """

    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be ObservedCurve")
    if not isinstance(candidate, CandidateInput):
        raise TypeError("candidate must be CandidateInput")
    if not isinstance(policy, ObservabilityPolicy):
        raise TypeError("policy must be ObservabilityPolicy")
    threshold = float(primary_compatibility_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("primary_compatibility_threshold must be finite and non-negative")
    limit = policy.exact_forward_call_limit
    if exact_forward_call_limit is not None:
        if isinstance(exact_forward_call_limit, bool) or not isinstance(
            exact_forward_call_limit, Integral
        ):
            raise TypeError("exact_forward_call_limit must be an integer or None")
        limit = min(limit, int(exact_forward_call_limit))
    if limit < 0:
        raise ValueError("exact_forward_call_limit must be non-negative")
    if reduced_model_searcher is not None and not hasattr(reduced_model_searcher, "search"):
        raise TypeError("reduced_model_searcher must implement search")
    if primary_metric_name == STANDARDIZED_LOG_RMSE_METRIC and curve.sigma_log is None:
        raise ValueError("standardized observability requires sigma_log")

    try:
        report = diagnose_component_observability(
            curve.q,
            curve.intensity,
            _profile_from_candidate(curve, candidate),
            sigma_log=curve.sigma_log,
            solver_tolerance=policy.solver_tolerance,
            max_iterations=policy.max_iterations,
            authoritative_full_exact=candidate.exact_intensity,
            exact_forward_call_limit=limit,
        )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        full_scores = exact_log_scores(candidate.exact_intensity, curve.intensity, curve.sigma_log)
        full_score = _primary_score(full_scores, primary_metric_name)
        particles = tuple(
            FeatureObservabilityAssessment(
                label=f"particle_{index}",
                decision="unknown",
                deletion_status=DELETION_PROFILE_FAILED,
                reduced_primary_score=None,
                primary_compatibility_threshold=threshold,
                evidence_scope="diagnostic_failed_closed",
            )
            for index in range(len(candidate.components))
        )
        resolution = FeatureObservabilityAssessment(
            label="resolution",
            decision=("unknown" if candidate.resolution is not None else "not_present"),
            deletion_status=(
                DELETION_PROFILE_FAILED if candidate.resolution is not None else "not_present"
            ),
            reduced_primary_score=None,
            primary_compatibility_threshold=threshold,
            evidence_scope=(
                "diagnostic_failed_closed" if candidate.resolution is not None else "not_present"
            ),
        )
        d_terms = tuple(
            FeatureObservabilityAssessment(
                label=f"d_{index}",
                decision=("unknown" if component.log_D is not None else "not_present"),
                deletion_status=("not_evaluated" if component.log_D is not None else "not_present"),
                reduced_primary_score=None,
                primary_compatibility_threshold=threshold,
                evidence_scope=(
                    "diagnostic_failed_closed" if component.log_D is not None else "not_present"
                ),
            )
            for index, component in enumerate(candidate.components)
        )
        return CandidateObservabilityAssessment(
            candidate_id=candidate.candidate_id,
            status="provisional_or_unknown",
            primary_metric_name=primary_metric_name,
            primary_compatibility_threshold=threshold,
            full_primary_score=full_score,
            full_curve_compatible=full_score <= threshold,
            particles=particles,
            resolution=resolution,
            d_terms=d_terms,
            exact_forward_calls=0,
            exact_forward_call_limit=limit,
            budget_exhausted=False,
            diagnostic_error=f"{type(exc).__name__}: {exc}",
            policy_version=policy.version,
            diagnostic_version=COMPONENT_OBSERVABILITY_VERSION,
        )

    full_score = _primary_score(report.full_scores, primary_metric_name)
    particles = [
        _feature_assessment(
            label=f"particle_{item.component_index}",
            deletion_status=item.deletion_status,
            reduced_scores=item.reduced_scores,
            metric_name=primary_metric_name,
            threshold=threshold,
            exhaustive_null=(len(candidate.components) == 1 and candidate.resolution is None),
        )
        for item in report.components
    ]
    if report.resolution_deletion is None:
        resolution = FeatureObservabilityAssessment(
            label="resolution",
            decision="not_present",
            deletion_status="not_present",
            reduced_primary_score=None,
            primary_compatibility_threshold=threshold,
            evidence_scope="not_present",
        )
    else:
        resolution = _feature_assessment(
            label="resolution",
            deletion_status=report.resolution_deletion.deletion_status,
            reduced_scores=report.resolution_deletion.reduced_scores,
            metric_name=primary_metric_name,
            threshold=threshold,
            exhaustive_null=False,
        )
    d_terms = []
    for index, component in enumerate(candidate.components):
        label = f"d_{index}"
        if component.log_D is None:
            decision: ObservabilityDecision = "not_present"
            scope = "not_present"
        else:
            decision = "unknown"
            scope = "nonlinear_D_toggle_search_not_run"
        d_terms.append(
            FeatureObservabilityAssessment(
                label=label,
                decision=decision,
                deletion_status=("not_present" if component.log_D is None else "not_evaluated"),
                reduced_primary_score=None,
                primary_compatibility_threshold=threshold,
                evidence_scope=scope,
            )
        )

    total_calls = report.exact_forward_calls
    unresolved = tuple(
        item
        for item in (
            *particles,
            *((resolution,) if candidate.resolution is not None else ()),
            *(item for item in d_terms if item.decision != "not_present"),
        )
        if item.decision not in {"needed", "unneeded", "not_present"}
    )
    search_allocations = (
        {}
        if reduced_model_searcher is None
        else _fair_search_allocations(unresolved, max(0, limit - total_calls))
    )

    def search_if_needed(
        feature: FeatureObservabilityAssessment,
    ) -> FeatureObservabilityAssessment:
        nonlocal total_calls
        if feature.decision in {"needed", "unneeded", "not_present"}:
            return feature
        if reduced_model_searcher is None:
            return feature
        allowance = search_allocations[feature.label]
        evidence = reduced_model_searcher.search(
            curve,
            candidate,
            label=feature.label,
            primary_metric_name=primary_metric_name,
            primary_compatibility_threshold=threshold,
            max_forward_evaluations=allowance,
            required_starts=policy.reduced_search_starts,
            per_start_forward_evaluation_limit=(policy.reduced_search_per_start_forward_limit),
        )
        if not isinstance(evidence, ReducedModelSearchEvidence):
            raise TypeError("reduced-model searcher returned invalid evidence")
        if evidence.forward_calls > allowance:
            raise RuntimeError("reduced-model searcher exceeded exact-forward allowance")
        total_calls += evidence.forward_calls
        return _with_reduced_search_evidence(
            feature,
            evidence,
            allocated_forward_calls=allowance,
        )

    particles = [search_if_needed(item) for item in particles]
    if candidate.resolution is not None:
        resolution = search_if_needed(resolution)
    d_terms = [search_if_needed(item) for item in d_terms]
    particles_tuple = tuple(particles)
    d_terms_tuple = tuple(d_terms)
    declared = (
        particles_tuple
        + ((resolution,) if candidate.resolution is not None else ())
        + tuple(item for item in d_terms_tuple if item.decision != "not_present")
    )
    decisions = tuple(item.decision for item in declared)
    if any(value == "unneeded" for value in decisions):
        status: ObservabilityStatus = "confirmed_redundant"
    elif full_score <= threshold and decisions and all(value == "needed" for value in decisions):
        status = "confirmed_effective"
    else:
        status = "provisional_or_unknown"
    budget_exhausted = (
        any(item.deletion_status == DELETION_FORWARD_BUDGET_EXHAUSTED for item in report.components)
        or (
            report.resolution_deletion is not None
            and report.resolution_deletion.deletion_status == DELETION_FORWARD_BUDGET_EXHAUSTED
        )
        or any(
            item.reduced_search_status == "budget_exhausted"
            for item in (*particles_tuple, resolution, *d_terms_tuple)
        )
    )
    return CandidateObservabilityAssessment(
        candidate_id=candidate.candidate_id,
        status=status,
        primary_metric_name=primary_metric_name,
        primary_compatibility_threshold=threshold,
        full_primary_score=full_score,
        full_curve_compatible=full_score <= threshold,
        particles=particles_tuple,
        resolution=resolution,
        d_terms=d_terms_tuple,
        exact_forward_calls=total_calls,
        exact_forward_call_limit=limit,
        budget_exhausted=budget_exhausted,
        diagnostic_error=None,
        policy_version=policy.version,
        diagnostic_version=report.version,
    )


__all__ = ["assess_candidate_observability"]
