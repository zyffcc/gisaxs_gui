"""Exact delete-and-reprofile diagnostics for Posterior V8 observability."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear

from .component_observability import (
    COMPONENT_OBSERVABILITY_VERSION,
    DELETION_EVALUATED,
    DELETION_FORWARD_BUDGET_EXHAUSTED,
    DELETION_K0_EVALUATED,
    DELETION_PROFILE_FAILED,
    DELETION_ZERO_REMAINING_PARTICLE_AMPLITUDE,
    PROFILE_WEIGHTING_SEMANTICS,
    ComponentDeletionDiagnostic,
    ComponentObservabilityReport,
    ContributionDiagnostic,
    ExactLogScores,
    K0ProfiledForwardResult,
    ResolutionDeletionDiagnostic,
)
from .evaluation import natural_log_rmse
from .profiled_forward import (
    ProfiledForwardResult,
    component_unit_basis,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
    resolution_unit_basis,
)


def _curve_inputs(q, intensity, sigma_log):
    q_array = np.asarray(q, dtype=np.float64)
    observed = np.asarray(intensity, dtype=np.float64)
    if q_array.ndim != 1 or q_array.size == 0:
        raise ValueError("q must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(q_array)) or np.any(q_array <= 0.0):
        raise ValueError("q must contain finite, strictly positive values")
    if observed.ndim != 1 or observed.shape != q_array.shape:
        raise ValueError("intensity must have the same one-dimensional shape as q")
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("intensity must contain finite, strictly positive values")
    if sigma_log is None:
        log_sigma = None
        profile_sigma = observed
    else:
        log_sigma = np.asarray(sigma_log, dtype=np.float64)
        if log_sigma.ndim != 1 or log_sigma.shape != observed.shape:
            raise ValueError("sigma_log must have the same one-dimensional shape as q")
        if not np.all(np.isfinite(log_sigma)) or np.any(log_sigma <= 0.0):
            raise ValueError("sigma_log must contain finite, strictly positive values")
        profile_sigma = observed * log_sigma
        if not np.all(np.isfinite(profile_sigma)) or np.any(profile_sigma <= 0.0):
            raise ValueError("observed * sigma_log must be finite and strictly positive")
    return q_array, observed, log_sigma, profile_sigma


def _validate_profile(profile: ProfiledForwardResult) -> None:
    if not isinstance(profile, ProfiledForwardResult):
        raise TypeError("profile must be a ProfiledForwardResult")
    count = len(profile.components)
    amplitudes = np.asarray(profile.particle_amplitudes, dtype=np.float64)
    if count < 1 or amplitudes.shape != (count,):
        raise ValueError("profile must contain one amplitude per particle component")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("profile particle amplitudes must be finite and non-negative")
    total = float(np.sum(amplitudes))
    if total <= 0.0 or not np.isfinite(profile.k) or profile.k <= 0.0:
        raise ValueError("profile must have positive particle amplitude and GUI k")
    expected_weights = amplitudes / profile.k
    weights = np.asarray(profile.component_weights, dtype=np.float64)
    if weights.shape != (count,) or not np.allclose(
        weights,
        expected_weights,
        rtol=1e-10,
        atol=1e-13,
    ):
        raise ValueError("profile component weights are inconsistent with its amplitudes")
    scalars = (profile.background, profile.resolution_amplitude, profile.int_res)
    if not np.all(np.isfinite(scalars)) or profile.background < 0.0:
        raise ValueError("profile global amplitudes must be finite and non-negative")
    if profile.resolution is None:
        if profile.resolution_amplitude != 0.0 or profile.int_res != 0.0:
            raise ValueError("resolution-absent profile must have zero resolution amplitude")
    elif profile.resolution_amplitude < 0.0 or not np.isclose(
        profile.int_res,
        profile.resolution_amplitude / profile.k,
        rtol=1e-10,
        atol=1e-13,
    ):
        raise ValueError("profile resolution gauge is inconsistent")
    audit = profile.amplitude_constraint_audit
    if audit is not None and (
        not audit.all_constraints_satisfied
        or not audit.auxiliary_k_is_witness
        or not np.isclose(audit.auxiliary_k, profile.k, rtol=1e-10, atol=1e-13)
    ):
        raise ValueError("profile GUI constraint audit does not match its explicit k")


def exact_log_scores(exact, observed, sigma_log) -> ExactLogScores:
    raw = natural_log_rmse(exact, observed)
    standardized = (
        None if sigma_log is None else natural_log_rmse(exact, observed, sigma_log=sigma_log)
    )
    return ExactLogScores(raw, standardized)


def _rms_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = float(np.sqrt(np.mean(np.square(numerator / denominator))))
    if not np.isfinite(value):
        raise ValueError("component contribution normalisation produced a non-finite value")
    return value


def _contribution(
    label: str,
    amplitude: float,
    curve: np.ndarray,
    observed: np.ndarray,
    full_exact: np.ndarray,
    sigma_log: np.ndarray | None,
) -> ContributionDiagnostic:
    standardized = None
    if sigma_log is not None:
        standardized = _rms_ratio(curve, observed * sigma_log)
    return ContributionDiagnostic(
        label=label,
        amplitude=float(amplitude),
        relative_to_observed_rms=_rms_ratio(curve, observed),
        relative_to_model_rms=_rms_ratio(curve, full_exact),
        measurement_standardized_rms=standardized,
    )


def _increment(reduced: ExactLogScores, full: ExactLogScores):
    standardized = None
    if reduced.standardized_log_rmse is not None:
        assert full.standardized_log_rmse is not None
        standardized = reduced.standardized_log_rmse - full.standardized_log_rmse
    return reduced.raw_log_rmse - full.raw_log_rmse, standardized


def _profile_k0(
    q: np.ndarray,
    observed: np.ndarray,
    profile_sigma: np.ndarray,
    resolution,
    *,
    solver_tolerance: float,
    max_iterations: int | None,
) -> K0ProfiledForwardResult:
    columns = [np.ones_like(q)]
    if resolution is not None:
        columns.append(resolution_unit_basis(q, resolution))
    design = np.column_stack(columns)
    weighted_design = design / profile_sigma[:, np.newaxis]
    weighted_observed = observed / profile_sigma
    column_scale = np.linalg.norm(weighted_design, axis=0)
    if not np.all(np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise ValueError("K0 nested-null design has a zero or non-finite column")
    solved = lsq_linear(
        weighted_design / column_scale[np.newaxis, :],
        weighted_observed,
        bounds=(0.0, np.inf),
        method="trf",
        tol=solver_tolerance,
        lsmr_tol="auto",
        max_iter=max_iterations,
    )
    if not bool(solved.success):
        raise RuntimeError(f"K0 non-negative profiling failed: {solved.message}")
    coefficients = np.maximum(np.asarray(solved.x, dtype=np.float64) / column_scale, 0.0)
    fitted = np.asarray(design @ coefficients, dtype=np.float64)
    if not np.all(np.isfinite(fitted)) or np.any(fitted <= 0.0):
        raise ValueError("K0 nested-null exact intensity is not strictly positive")
    fitted.setflags(write=False)
    return K0ProfiledForwardResult(
        resolution_present=resolution is not None,
        background=float(coefficients[0]),
        resolution_amplitude=(float(coefficients[1]) if resolution is not None else 0.0),
        fitted_intensity=fitted,
        solver_status=int(solved.status),
        solver_message=str(solved.message),
        solver_optimality=float(solved.optimality),
    )


def _unscored_component_deletion(
    removed_index: int,
    shape: str,
    contribution: ContributionDiagnostic,
    status: str,
    remaining_indices: tuple[int, ...],
) -> ComponentDeletionDiagnostic:
    return ComponentDeletionDiagnostic(
        removed_index,
        shape,
        contribution,
        status,
        remaining_indices,
        None,
        None,
        None,
        None,
    )


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
    """Delete each particle/Resolution and re-profile non-negative amplitudes.

    K1 deletion uses a diagnostic K0 null containing BG and, when declared,
    the candidate's fixed Resolution shape.  It does not add K0 to the output
    topology catalog, and the fixed-shape Resolution case is not exhaustive.
    """

    q_array, observed, log_sigma, profile_sigma = _curve_inputs(q, intensity, sigma_log)
    _validate_profile(profile)
    if isinstance(max_iterations, bool) or (
        max_iterations is not None
        and (not isinstance(max_iterations, Integral) or int(max_iterations) < 1)
    ):
        raise ValueError("max_iterations must be a positive integer or None")
    tolerance = float(solver_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("solver_tolerance must be finite and strictly positive")
    if exact_forward_call_limit is not None:
        if isinstance(exact_forward_call_limit, bool) or not isinstance(
            exact_forward_call_limit, Integral
        ):
            raise TypeError("exact_forward_call_limit must be an integer or None")
        exact_forward_call_limit = int(exact_forward_call_limit)
        if exact_forward_call_limit < 0:
            raise ValueError("exact_forward_call_limit must be non-negative")
    reused_full = authoritative_full_exact is not None
    calls = 0
    if authoritative_full_exact is None:
        if exact_forward_call_limit == 0:
            raise ValueError("one exact-forward call is required without a supplied full curve")
        full_exact = np.asarray(evaluate_profiled_forward(q_array, profile), dtype=np.float64)
        calls += 1
    else:
        full_exact = np.asarray(authoritative_full_exact, dtype=np.float64)
        if full_exact.ndim != 1 or full_exact.shape != q_array.shape:
            raise ValueError("authoritative_full_exact must have the same shape as q")
        if not np.all(np.isfinite(full_exact)):
            raise ValueError("authoritative_full_exact must contain only finite values")
    if np.any(full_exact <= 0.0):
        raise ValueError("exact profile intensity must be strictly positive")
    particle_curves = tuple(
        float(amplitude) * component_unit_basis(q_array, component)
        for amplitude, component in zip(profile.particle_amplitudes, profile.components)
    )
    background_curve = np.full_like(q_array, float(profile.background))
    resolution_curve = None
    if profile.resolution is not None:
        resolution_curve = float(profile.resolution_amplitude) * resolution_unit_basis(
            q_array, profile.resolution
        )
    decomposed = background_curve + np.sum(np.stack(particle_curves), axis=0)
    if resolution_curve is not None:
        decomposed = decomposed + resolution_curve
    if not np.allclose(decomposed, full_exact, rtol=2e-10, atol=1e-12):
        raise RuntimeError("profile contributions disagree with the authoritative GUI forward")

    full_scores = exact_log_scores(full_exact, observed, log_sigma)
    component_reports = []
    count = len(profile.components)
    for removed_index, (component, amplitude, curve) in enumerate(
        zip(profile.components, profile.particle_amplitudes, particle_curves)
    ):
        remaining_indices = tuple(index for index in range(count) if index != removed_index)
        contribution = _contribution(
            f"particle_{removed_index}",
            amplitude,
            curve,
            observed,
            full_exact,
            log_sigma,
        )
        if exact_forward_call_limit is not None and calls >= exact_forward_call_limit:
            component_reports.append(
                _unscored_component_deletion(
                    removed_index,
                    component.shape,
                    contribution,
                    DELETION_FORWARD_BUDGET_EXHAUSTED,
                    remaining_indices,
                )
            )
            continue
        if not remaining_indices:
            try:
                reduced_profile = _profile_k0(
                    q_array,
                    observed,
                    profile_sigma,
                    profile.resolution,
                    solver_tolerance=tolerance,
                    max_iterations=(None if max_iterations is None else int(max_iterations)),
                )
            except (FloatingPointError, RuntimeError, ValueError):
                component_reports.append(
                    _unscored_component_deletion(
                        removed_index,
                        component.shape,
                        contribution,
                        DELETION_PROFILE_FAILED,
                        remaining_indices,
                    )
                )
                continue
            calls += 1
            reduced_scores = exact_log_scores(reduced_profile.fitted_intensity, observed, log_sigma)
            raw_increment, standardized_increment = _increment(reduced_scores, full_scores)
            component_reports.append(
                ComponentDeletionDiagnostic(
                    removed_index,
                    component.shape,
                    contribution,
                    DELETION_K0_EVALUATED,
                    remaining_indices,
                    reduced_profile,
                    reduced_scores,
                    raw_increment,
                    standardized_increment,
                )
            )
            continue

        remaining = tuple(profile.components[index] for index in remaining_indices)
        try:
            reduced_profile = profile_linear_amplitudes(
                q_array,
                observed,
                remaining,
                resolution=profile.resolution,
                sigma=profile_sigma,
                solver_tolerance=tolerance,
                max_iterations=(None if max_iterations is None else int(max_iterations)),
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            status = (
                DELETION_ZERO_REMAINING_PARTICLE_AMPLITUDE
                if "particle amplitude sum is zero" in str(exc)
                else DELETION_PROFILE_FAILED
            )
            component_reports.append(
                _unscored_component_deletion(
                    removed_index,
                    component.shape,
                    contribution,
                    status,
                    remaining_indices,
                )
            )
            continue
        reduced_exact = evaluate_profiled_forward(q_array, reduced_profile)
        calls += 1
        reduced_scores = exact_log_scores(reduced_exact, observed, log_sigma)
        raw_increment, standardized_increment = _increment(reduced_scores, full_scores)
        component_reports.append(
            ComponentDeletionDiagnostic(
                removed_index,
                component.shape,
                contribution,
                DELETION_EVALUATED,
                remaining_indices,
                reduced_profile,
                reduced_scores,
                raw_increment,
                standardized_increment,
            )
        )

    resolution_deletion = None
    if profile.resolution is not None:
        if exact_forward_call_limit is not None and calls >= exact_forward_call_limit:
            resolution_deletion = ResolutionDeletionDiagnostic(
                DELETION_FORWARD_BUDGET_EXHAUSTED, None, None, None, None
            )
        else:
            try:
                reduced_profile = profile_linear_amplitudes(
                    q_array,
                    observed,
                    profile.components,
                    resolution=None,
                    sigma=profile_sigma,
                    solver_tolerance=tolerance,
                    max_iterations=(None if max_iterations is None else int(max_iterations)),
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                status = (
                    DELETION_ZERO_REMAINING_PARTICLE_AMPLITUDE
                    if "particle amplitude sum is zero" in str(exc)
                    else DELETION_PROFILE_FAILED
                )
                resolution_deletion = ResolutionDeletionDiagnostic(status, None, None, None, None)
            else:
                reduced_exact = evaluate_profiled_forward(q_array, reduced_profile)
                calls += 1
                reduced_scores = exact_log_scores(reduced_exact, observed, log_sigma)
                raw_increment, standardized_increment = _increment(reduced_scores, full_scores)
                resolution_deletion = ResolutionDeletionDiagnostic(
                    DELETION_EVALUATED,
                    reduced_profile,
                    reduced_scores,
                    raw_increment,
                    standardized_increment,
                )

    exact_copy = np.array(full_exact, copy=True)
    exact_copy.setflags(write=False)
    return ComponentObservabilityReport(
        version=COMPONENT_OBSERVABILITY_VERSION,
        profile_weighting_semantics=PROFILE_WEIGHTING_SEMANTICS,
        full_scores=full_scores,
        full_exact_intensity=exact_copy,
        background_contribution=_contribution(
            "background",
            profile.background,
            background_curve,
            observed,
            full_exact,
            log_sigma,
        ),
        resolution_contribution=(
            None
            if resolution_curve is None
            else _contribution(
                "resolution",
                profile.resolution_amplitude,
                resolution_curve,
                observed,
                full_exact,
                log_sigma,
            )
        ),
        components=tuple(component_reports),
        resolution_deletion=resolution_deletion,
        exact_forward_calls=calls,
        exact_forward_call_limit=exact_forward_call_limit,
        reused_authoritative_full_exact=reused_full,
    )


__all__ = ["diagnose_component_observability", "exact_log_scores"]
