"""Exact log-residual polish of profiled linear GISAXS amplitudes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize

from .evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    natural_log_rmse,
)
from .gui_amplitude_constraints import (
    CoefficientPolytope,
    GuiAmplitudeConstraint,
    GuiAmplitudeConstraintAudit,
)
from .profiled_forward import (
    ProfiledForwardResult,
    build_design_matrix,
    evaluate_profiled_forward,
)


AMPLITUDE_POLISH_VERSION = "posterior_v8_exact_log_amplitude_auxiliary_kappa_polish_v5"
SUPPORTED_METRICS = (RAW_LOG_RMSE_METRIC, STANDARDIZED_LOG_RMSE_METRIC)


@dataclass(frozen=True)
class AmplitudeBounds:
    """Coefficient bounds in ``[BG, a_1, ..., optional a_res]`` order."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            lower = tuple(float(value) for value in self.lower)
            upper = tuple(float(value) for value in self.upper)
        except (TypeError, ValueError) as exc:
            raise ValueError("amplitude bounds must be numeric sequences") from exc
        if not lower or len(lower) != len(upper):
            raise ValueError("amplitude lower/upper bounds must have the same non-zero length")
        if not np.all(np.isfinite(lower)):
            raise ValueError("amplitude lower bounds must be finite")
        upper_array = np.asarray(upper, dtype=np.float64)
        if np.any(np.isnan(upper_array)) or np.any(np.isneginf(upper_array)):
            raise ValueError("amplitude upper bounds must be finite or positive infinity")
        if lower[0] < 0.0:
            raise ValueError("background lower bound must be non-negative")
        if any(value < 0.0 for value in lower[1:]):
            raise ValueError("particle/resolution amplitude lower bounds must be non-negative")
        if any(high < low for low, high in zip(lower, upper)):
            raise ValueError("amplitude upper bounds must not be below lower bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, coefficients: Sequence[float], *, atol: float = 1e-12) -> bool:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.lower),) or not np.all(np.isfinite(values)):
            return False
        return bool(
            np.all(values >= np.asarray(self.lower) - atol)
            and np.all(values <= np.asarray(self.upper) + atol)
        )


@dataclass(frozen=True)
class AmplitudePolishResult:
    """Best fixed-geometry amplitude state and its constraint audit.

    When ``coefficient_polytope`` is present, ``coefficient_bounds`` is only
    the numerical outer box; the polytope and GUI audit are authoritative.
    """

    metric_name: str
    initial_coefficients: tuple[float, ...]
    final_coefficients: tuple[float, ...]
    coefficient_bounds: AmplitudeBounds
    coefficient_polytope: CoefficientPolytope | None
    gui_amplitude_constraint: GuiAmplitudeConstraint | None
    initial_constraint_audit: GuiAmplitudeConstraintAudit | None
    final_constraint_audit: GuiAmplitudeConstraintAudit | None
    initial_metric: float
    final_metric: float
    initial_raw_log_rmse: float
    final_raw_log_rmse: float
    initial_standardized_log_rmse: float | None
    final_standardized_log_rmse: float | None
    initial_profile: ProfiledForwardResult
    final_profile: ProfiledForwardResult
    exact_intensity: np.ndarray
    success: bool
    status: int
    message: str
    nfev: int
    njev: int | None
    residual_calls: int
    returned_source: str
    best_residual_call: int
    bounds_satisfied: bool


def _curve_inputs(q, intensity, sigma_log):
    q_array = np.asarray(q, dtype=np.float64)
    observed = np.asarray(intensity, dtype=np.float64)
    if q_array.ndim != 1 or q_array.size == 0:
        raise ValueError("q must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(q_array)) or np.any(q_array <= 0.0):
        raise ValueError("q must contain finite, strictly positive values")
    if np.any(np.diff(q_array) <= 0.0):
        raise ValueError("q must be strictly increasing")
    if observed.ndim != 1 or observed.shape != q_array.shape:
        raise ValueError("intensity must have the same one-dimensional shape as q")
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("intensity must contain finite, strictly positive values")
    sigma = None
    if sigma_log is not None:
        sigma = np.asarray(sigma_log, dtype=np.float64)
        if sigma.ndim != 1 or sigma.shape != observed.shape:
            raise ValueError("sigma_log must have the same one-dimensional shape as q")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma_log must contain finite, strictly positive values")
    return q_array, observed, sigma


def _initial_coefficients(profile: ProfiledForwardResult) -> np.ndarray:
    if not isinstance(profile, ProfiledForwardResult):
        raise TypeError("initial_profile must be a ProfiledForwardResult")
    coefficients = [profile.background, *profile.particle_amplitudes]
    if profile.resolution is not None:
        coefficients.append(profile.resolution_amplitude)
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (len(profile.components) + 1 + int(profile.resolution is not None),):
        raise ValueError("initial profile has the wrong number of particle amplitudes")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("initial amplitudes must be finite and non-negative")
    if np.sum(values[1 : len(profile.components) + 1]) <= 0.0:
        raise ValueError("initial profile must contain positive total particle amplitude")
    if profile.resolution is None and profile.resolution_amplitude != 0.0:
        raise ValueError("absent resolution requires zero resolution amplitude")
    return values


def _default_bounds(count: int) -> AmplitudeBounds:
    # BG=0 is part of the authoritative GUI contract.  A synthetic numerical
    # floor would incorrectly reject a valid small positive NNLS background;
    # positivity of the total curve is enforced directly by the log residual.
    return AmplitudeBounds(
        lower=(0.0,) * count,
        upper=(np.inf,) * count,
    )


def _feasible_initial(coefficients: np.ndarray, bounds: AmplitudeBounds) -> np.ndarray:
    if coefficients.size != len(bounds.lower):
        raise ValueError("amplitude bounds length does not match the fixed forward basis")
    lower = np.asarray(bounds.lower, dtype=np.float64)
    upper = np.asarray(bounds.upper, dtype=np.float64)
    result = coefficients.copy()
    below = result < lower
    # An exact NNLS zero has no logarithm and is intentionally lifted to the
    # requested lower bound.  Other out-of-range starting values fail closed.
    if np.any(below & (result != 0.0)) or np.any(result > upper):
        raise ValueError("initial amplitudes are outside the requested bounds")
    result[below] = lower[below]
    if not bounds.contains(result):
        raise RuntimeError("failed to construct a feasible amplitude-polish start")
    return result


def _make_profile(
    template: ProfiledForwardResult,
    coefficients: np.ndarray,
    design: np.ndarray,
    observed: np.ndarray,
    objective_residual: np.ndarray,
    *,
    status: int,
    message: str,
    optimality: float,
    constraint_audit: GuiAmplitudeConstraintAudit | None,
    gui_k: float | None = None,
) -> ProfiledForwardResult:
    count = len(template.components)
    particles = np.maximum(coefficients[1 : count + 1], 0.0)
    particle_total = float(np.sum(particles))
    if not np.isfinite(particle_total) or particle_total <= 0.0:
        raise ValueError("polished particle amplitude sum is zero; GUI gauge is undefined")
    k_value = particle_total if gui_k is None else float(gui_k)
    if not np.isfinite(k_value) or k_value <= 0.0:
        raise ValueError("polished GUI k must be finite and strictly positive")
    resolution_amplitude = (
        float(max(coefficients[-1], 0.0)) if template.resolution is not None else 0.0
    )
    fitted = np.asarray(design @ coefficients, dtype=np.float64)
    if not np.all(np.isfinite(fitted)) or np.any(fitted <= 0.0):
        raise ValueError("amplitude polish produced non-positive or non-finite intensity")
    residual = fitted - observed
    weighted = np.asarray(objective_residual, dtype=np.float64)
    if weighted.shape != observed.shape or not np.all(np.isfinite(weighted)):
        raise ValueError("amplitude-polish objective residual is invalid")
    return ProfiledForwardResult(
        components=template.components,
        resolution=template.resolution,
        background=float(coefficients[0]),
        particle_amplitudes=tuple(float(value) for value in particles),
        resolution_amplitude=resolution_amplitude,
        k=k_value,
        component_weights=tuple(float(value) for value in particles / k_value),
        int_res=resolution_amplitude / k_value,
        fitted_intensity=fitted,
        residual=residual,
        # In an amplitude-polish result these fields record the raw or
        # sigma-standardized log residual that was actually optimized.
        weighted_residual=weighted,
        weighted_rss=float(np.dot(weighted, weighted)),
        solver_status=int(status),
        solver_message=str(message),
        solver_optimality=float(optimality),
        amplitude_constraint_audit=constraint_audit,
    )


def _exact_metrics(q, observed, sigma_log, profile):
    exact = np.asarray(evaluate_profiled_forward(q, profile), dtype=np.float64)
    if not np.allclose(exact, profile.fitted_intensity, rtol=2e-10, atol=1e-12):
        raise RuntimeError("basis and GUI-consistent authoritative forward disagree")
    raw = natural_log_rmse(exact, observed)
    standardized = (
        None if sigma_log is None else natural_log_rmse(exact, observed, sigma_log=sigma_log)
    )
    if not np.isfinite(raw) or (standardized is not None and not np.isfinite(standardized)):
        raise ValueError("exact log metric is non-finite")
    return exact, raw, standardized


def polish_profiled_amplitudes(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    initial_profile: ProfiledForwardResult,
    *,
    sigma_log: Sequence[float] | np.ndarray | None = None,
    metric_name: str = RAW_LOG_RMSE_METRIC,
    bounds: AmplitudeBounds | None = None,
    amplitude_constraint: GuiAmplitudeConstraint | None = None,
    max_nfev: int = 100,
    ftol: float = 1e-11,
    xtol: float = 1e-11,
    gtol: float = 1e-11,
) -> AmplitudePolishResult:
    """Jointly polish fixed-basis amplitudes against the exact log objective.

    ``amplitude_constraint`` preserves the coupled GUI ranges as one exact
    coefficient polytope.  It is deliberately mutually exclusive with the
    legacy independent-axis ``bounds`` argument.
    """

    q_array, observed, sigma = _curve_inputs(q, intensity, sigma_log)
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"metric_name must be one of {SUPPORTED_METRICS!r}")
    if metric_name == STANDARDIZED_LOG_RMSE_METRIC and sigma is None:
        raise ValueError("standardized log metric requires sigma_log")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    tolerances = tuple(float(value) for value in (ftol, xtol, gtol))
    if not all(np.isfinite(value) and value > np.finfo(float).eps for value in tolerances):
        raise ValueError("ftol, xtol and gtol must be finite and greater than machine epsilon")

    coefficients = _initial_coefficients(initial_profile)
    design = build_design_matrix(q_array, initial_profile.components, initial_profile.resolution)
    if bounds is not None and amplitude_constraint is not None:
        raise ValueError(
            "bounds and amplitude_constraint are mutually exclusive; coupled GUI "
            "constraints must not be approximated by an axis box"
        )
    coefficient_polytope = None
    if amplitude_constraint is not None:
        if not isinstance(amplitude_constraint, GuiAmplitudeConstraint):
            raise TypeError("amplitude_constraint must be a GuiAmplitudeConstraint or None")
        amplitude_constraint.validate_branch(
            len(initial_profile.components), initial_profile.resolution is not None
        )
        coefficient_polytope = amplitude_constraint.coefficient_polytope()
        selected_bounds = AmplitudeBounds(
            coefficient_polytope.axis_lower,
            coefficient_polytope.axis_upper,
        )
    else:
        selected_bounds = _default_bounds(coefficients.size) if bounds is None else bounds
    if not isinstance(selected_bounds, AmplitudeBounds):
        raise TypeError("bounds must be an AmplitudeBounds or None")
    initial_values = _feasible_initial(coefficients, selected_bounds)
    if coefficient_polytope is not None and not coefficient_polytope.contains(initial_values):
        raise ValueError("initial amplitudes are outside the requested GUI amplitude polytope")
    lower = np.asarray(selected_bounds.lower, dtype=np.float64)
    upper = np.asarray(selected_bounds.upper, dtype=np.float64)
    free = upper > lower
    fixed = lower.copy()
    fixed[free] = initial_values[free]
    divisor = sigma if metric_name == STANDARDIZED_LOG_RMSE_METRIC else 1.0

    def coefficients_from_free(values):
        result = fixed.copy()
        result[free] = values
        return result

    def log_residual(full_coefficients):
        if np.sum(full_coefficients[1 : len(initial_profile.components) + 1]) <= 0.0:
            raise ValueError("particle amplitude sum must stay positive")
        fitted = design @ full_coefficients
        if not np.all(np.isfinite(fitted)) or np.any(fitted <= 0.0):
            raise ValueError("non-positive intensity during amplitude polish")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            current = (np.log(fitted) - np.log(observed)) / divisor
        if not np.all(np.isfinite(current)):
            raise ValueError("non-finite log residual during amplitude polish")
        return current

    initial_residual = log_residual(initial_values)
    best_coefficients = initial_values.copy()
    best_objective = float(np.dot(initial_residual, initial_residual))
    best_residual_call = 0
    residual_calls = 0

    if np.any(free) and coefficient_polytope is None:
        x0 = initial_values[free]
        lower_free, upper_free = lower[free], upper[free]
        basis_peak = np.maximum(np.max(np.abs(design[:, free]), axis=0), 1e-300)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scale_hint = float(np.median(observed)) / basis_peak
        scale_hint = np.where(np.isfinite(scale_hint), scale_hint, 1.0)
        scale = np.maximum(x0, scale_hint)
        scale = np.clip(
            scale,
            np.finfo(np.float64).tiny,
            np.sqrt(np.finfo(np.float64).max),
        )

        def residual(values):
            nonlocal best_coefficients, best_objective, best_residual_call, residual_calls
            residual_calls += 1
            try:
                candidate = coefficients_from_free(values)
                current = log_residual(candidate)
            except (FloatingPointError, ValueError):
                return np.full_like(observed, 1e12)
            objective = float(np.dot(current, current))
            if objective < best_objective:
                best_objective = objective
                best_coefficients = candidate.copy()
                best_residual_call = residual_calls
            return current

        solved = least_squares(
            residual,
            x0,
            bounds=(lower_free, upper_free),
            method="trf",
            x_scale=scale,
            ftol=tolerances[0],
            xtol=tolerances[1],
            gtol=tolerances[2],
            max_nfev=int(max_nfev),
        )
        solved_coefficients = coefficients_from_free(solved.x)
        solved_residual = log_residual(solved_coefficients)
        solved_objective = float(np.dot(solved_residual, solved_residual))
        if solved_objective < best_objective:
            best_coefficients = solved_coefficients
            best_objective = solved_objective
            best_residual_call = residual_calls + 1
        success, status, message = bool(solved.success), int(solved.status), str(solved.message)
        nfev = int(solved.nfev)
        njev = None if solved.njev is None else int(solved.njev)
        optimality = float(solved.optimality)
        terminal_is_returned = bool(
            np.allclose(
                best_coefficients,
                solved_coefficients,
                rtol=1e-13,
                atol=np.finfo(np.float64).tiny,
            )
        )
        if best_residual_call == 0:
            returned_source = "initial"
        elif terminal_is_returned:
            returned_source = "optimizer_terminal"
        else:
            returned_source = "optimizer_intermediate"
    elif np.any(free):
        assert coefficient_polytope is not None
        initial_fitted = design @ initial_values
        initial_jacobian = design / (initial_fitted * divisor)[:, np.newaxis]
        coefficient_scale = np.linalg.norm(initial_jacobian, axis=0)
        if not np.all(np.isfinite(coefficient_scale)) or np.any(coefficient_scale <= 0.0):
            raise ValueError("cannot scale the constrained amplitude-polish variables")
        scaled_lower, scaled_upper, scaled_matrix, scaled_rhs = coefficient_polytope.scaled_system(
            coefficient_scale
        )
        scaled_initial = initial_values * coefficient_scale
        budget_exhausted = False

        class _EvaluationBudgetExhausted(RuntimeError):
            pass

        def objective_and_gradient(values):
            nonlocal best_coefficients, best_objective, best_residual_call, residual_calls
            if residual_calls >= int(max_nfev):
                raise _EvaluationBudgetExhausted
            residual_calls += 1
            candidate = np.asarray(values, dtype=np.float64) / coefficient_scale
            try:
                current = log_residual(candidate)
            except (FloatingPointError, ValueError):
                return 0.5e24, np.zeros_like(values)
            objective = float(np.dot(current, current))
            if coefficient_polytope.contains(candidate) and objective < best_objective:
                best_objective = objective
                best_coefficients = candidate.copy()
                best_residual_call = residual_calls
            fitted = design @ candidate
            gradient_coefficients = design.T @ (current / (fitted * divisor))
            gradient_scaled = gradient_coefficients / coefficient_scale
            return 0.5 * objective, gradient_scaled

        try:
            constraints = (
                ()
                if scaled_matrix.shape[0] == 0
                else (LinearConstraint(scaled_matrix, -np.inf, scaled_rhs),)
            )
            solved = minimize(
                objective_and_gradient,
                scaled_initial,
                jac=True,
                method="SLSQP",
                bounds=Bounds(scaled_lower, scaled_upper),
                constraints=constraints,
                options={"ftol": tolerances[0], "maxiter": int(max_nfev)},
            )
        except _EvaluationBudgetExhausted:
            solved = None
            budget_exhausted = True

        if solved is None:
            success = False
            status = 0
            message = "maximum exact-log amplitude evaluations reached"
            nfev = residual_calls
            njev = None
            best_fitted = design @ best_coefficients
            best_current = log_residual(best_coefficients)
            best_gradient = design.T @ (best_current / (best_fitted * divisor))
            optimality = float(np.linalg.norm(best_gradient / coefficient_scale, ord=np.inf))
            terminal_is_returned = False
        else:
            solved_coefficients = np.asarray(solved.x, dtype=np.float64) / coefficient_scale
            terminal_is_feasible = coefficient_polytope.contains(solved_coefficients)
            if terminal_is_feasible:
                solved_residual = log_residual(solved_coefficients)
                solved_objective = float(np.dot(solved_residual, solved_residual))
                if solved_objective < best_objective:
                    best_coefficients = solved_coefficients
                    best_objective = solved_objective
                    best_residual_call = residual_calls
            success = bool(solved.success and terminal_is_feasible)
            status = int(solved.status)
            message = str(solved.message)
            if not terminal_is_feasible:
                message += "; terminal point failed GUI-polytope validation"
            nfev = int(solved.nfev)
            njev = None if solved.njev is None else int(solved.njev)
            optimality = float(np.linalg.norm(np.asarray(solved.jac), ord=np.inf))
            terminal_is_returned = bool(
                terminal_is_feasible
                and np.allclose(
                    best_coefficients,
                    solved_coefficients,
                    rtol=1e-13,
                    atol=np.finfo(np.float64).tiny,
                )
            )
        if budget_exhausted:
            success = False
        if best_residual_call == 0:
            returned_source = "initial"
        elif terminal_is_returned:
            returned_source = "optimizer_terminal"
        else:
            returned_source = "optimizer_intermediate"
    else:
        success, status, message = True, 0, "all amplitude coefficients are fixed"
        nfev, njev, optimality = 0, 0, 0.0
        returned_source = "all_coefficients_fixed"

    initial_k = (
        float(np.sum(initial_values[1 : len(initial_profile.components) + 1]))
        if coefficient_polytope is None
        else coefficient_polytope.select_k(initial_values, preferred=initial_profile.k)
    )
    initial_constraint_audit = (
        None
        if amplitude_constraint is None
        else amplitude_constraint.assess(initial_values, k=initial_k)
    )
    initial_built = _make_profile(
        initial_profile,
        initial_values,
        design,
        observed,
        initial_residual,
        status=0,
        message="feasible amplitude-polish start",
        optimality=0.0,
        constraint_audit=initial_constraint_audit,
        gui_k=initial_k,
    )
    initial_exact, initial_raw, initial_standardized = _exact_metrics(
        q_array, observed, sigma, initial_built
    )
    final_objective_residual = log_residual(best_coefficients)
    final_k = (
        float(np.sum(best_coefficients[1 : len(initial_profile.components) + 1]))
        if coefficient_polytope is None
        else coefficient_polytope.select_k(best_coefficients, preferred=initial_k)
    )
    final_constraint_audit = (
        None
        if amplitude_constraint is None
        else amplitude_constraint.assess(best_coefficients, k=final_k)
    )
    final_built = _make_profile(
        initial_profile,
        best_coefficients,
        design,
        observed,
        final_objective_residual,
        status=status,
        message=message,
        optimality=optimality,
        constraint_audit=final_constraint_audit,
        gui_k=final_k,
    )
    final_exact, final_raw, final_standardized = _exact_metrics(
        q_array, observed, sigma, final_built
    )
    initial_metric = initial_raw if metric_name == RAW_LOG_RMSE_METRIC else initial_standardized
    final_metric = final_raw if metric_name == RAW_LOG_RMSE_METRIC else final_standardized
    assert initial_metric is not None and final_metric is not None
    if final_metric > initial_metric:
        best_coefficients = initial_values
        final_built = initial_built
        final_exact, final_raw, final_standardized = (
            initial_exact,
            initial_raw,
            initial_standardized,
        )
        final_metric = initial_metric
        returned_source = "initial"
        best_residual_call = 0
        final_constraint_audit = initial_constraint_audit

    exact = np.array(final_exact, dtype=np.float64, copy=True)
    exact.setflags(write=False)
    bounds_satisfied = selected_bounds.contains(best_coefficients) and (
        coefficient_polytope is None or coefficient_polytope.contains(best_coefficients)
    )
    if not bounds_satisfied:
        raise RuntimeError("amplitude polish escaped its coefficient constraints")
    if final_constraint_audit is not None and not final_constraint_audit.all_constraints_satisfied:
        raise RuntimeError("amplitude polish violates the requested GUI amplitude ranges")
    return AmplitudePolishResult(
        metric_name=metric_name,
        initial_coefficients=tuple(float(value) for value in initial_values),
        final_coefficients=tuple(float(value) for value in best_coefficients),
        coefficient_bounds=selected_bounds,
        coefficient_polytope=coefficient_polytope,
        gui_amplitude_constraint=amplitude_constraint,
        initial_constraint_audit=initial_constraint_audit,
        final_constraint_audit=final_constraint_audit,
        initial_metric=float(initial_metric),
        final_metric=float(final_metric),
        initial_raw_log_rmse=float(initial_raw),
        final_raw_log_rmse=float(final_raw),
        initial_standardized_log_rmse=(
            None if initial_standardized is None else float(initial_standardized)
        ),
        final_standardized_log_rmse=(
            None if final_standardized is None else float(final_standardized)
        ),
        initial_profile=initial_built,
        final_profile=final_built,
        exact_intensity=exact,
        success=success,
        status=status,
        message=message,
        nfev=nfev,
        njev=njev,
        residual_calls=residual_calls,
        returned_source=returned_source,
        best_residual_call=best_residual_call,
        bounds_satisfied=True,
    )


__all__ = [
    "AMPLITUDE_POLISH_VERSION",
    "SUPPORTED_METRICS",
    "AmplitudeBounds",
    "AmplitudePolishResult",
    "polish_profiled_amplitudes",
]
