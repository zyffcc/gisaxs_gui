"""Budgeted gold-standard joint geometry/amplitude optimization.

This small-sample oracle is intentionally separate from the production
refiner.  It optimizes the authoritative branch-codec unit coordinates and
all additive amplitudes in one vector, and every objective evaluation is made
through the exact GUI mixed forward.  Legacy independent amplitude bounds use
a zero-preserving ``log1p(a / scale)`` coordinate.  GUI ranges instead use
their direct BG/k/independent-Int/int_Res product box, so no exact objective
call can leave the requested parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .amplitude_polish import AmplitudeBounds
from .branch_codec import ProfiledBranchCodec, ResolutionBounds
from .contract import MAX_COMPONENTS, GuiComponentBounds, LatentComponentParameters
from .evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
)
from .gui_amplitude_constraints import (
    CoefficientPolytope,
    GuiAmplitudeConstraint,
    GuiAmplitudeConstraintAudit,
)
from .joint_gui_amplitude_coordinates import FeasibleGuiAmplitudeCoordinates
from .profiled_forward import (
    ProfiledForwardResult,
    ResolutionShape,
    build_design_matrix,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)


JOINT_EXACT_OPTIMIZATION_VERSION = "posterior_v8_joint_exact_auxiliary_kappa_gold_standard_v3"
SUPPORTED_METRICS = (RAW_LOG_RMSE_METRIC, STANDARDIZED_LOG_RMSE_METRIC)
DEFAULT_AMPLITUDE_HEADROOM = 1.0e8
INVALID_RESIDUAL = 1.0e12


@dataclass(frozen=True)
class JointExactOptimizationResult:
    """Best valid exact-forward state seen inside one hard evaluation budget."""

    version: str
    metric_name: str
    initial_active_coordinates: tuple[float, ...]
    final_active_coordinates: tuple[float, ...]
    initial_coefficients: tuple[float, ...]
    final_coefficients: tuple[float, ...]
    requested_amplitude_bounds: AmplitudeBounds | None
    effective_amplitude_bounds: AmplitudeBounds
    gui_amplitude_constraint: GuiAmplitudeConstraint | None
    coefficient_polytope: CoefficientPolytope | None
    initial_constraint_audit: GuiAmplitudeConstraintAudit | None
    final_constraint_audit: GuiAmplitudeConstraintAudit | None
    amplitude_scales: tuple[float, ...]
    initial_latent_components: tuple[LatentComponentParameters, ...]
    final_latent_components: tuple[LatentComponentParameters, ...]
    initial_resolution: ResolutionShape | None
    final_resolution: ResolutionShape | None
    initial_profile: ProfiledForwardResult
    final_profile: ProfiledForwardResult
    initial_metric: float
    final_metric: float
    initial_raw_log_rmse: float
    final_raw_log_rmse: float
    initial_standardized_log_rmse: float | None
    final_standardized_log_rmse: float | None
    exact_intensity: np.ndarray
    success: bool
    status: int
    message: str
    optimizer_nfev: int | None
    optimizer_njev: int | None
    objective_requests: int
    exact_forward_calls: int
    cache_hits: int
    invalid_exact_evaluations: int
    zero_particle_evaluations: int
    max_exact_evaluations: int
    budget_exhausted: bool
    returned_source: str
    best_exact_call: int
    bounds_satisfied: bool


@dataclass(frozen=True)
class _Evaluation:
    vector: np.ndarray
    active_coordinates: tuple[float, ...]
    coefficients: tuple[float, ...]
    latent_components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    profile: ProfiledForwardResult
    exact: np.ndarray
    selected_residual: np.ndarray
    raw_log_rmse: float
    standardized_log_rmse: float | None
    objective: float
    exact_call: int
    branch_valid: bool


class _BudgetExhausted(RuntimeError):
    pass


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


def _branch_codec(component_bounds, seed_components, resolution_bounds, resolution_seed):
    bounds = tuple(component_bounds)
    seeds = tuple(seed_components)
    if not 1 <= len(bounds) <= MAX_COMPONENTS:
        raise ValueError(f"gold-standard optimization supports K=1..{MAX_COMPONENTS}")
    if len(bounds) != len(seeds):
        raise ValueError("component bounds and latent seeds must have the same length")
    if not all(isinstance(item, GuiComponentBounds) for item in bounds):
        raise TypeError("component_bounds must contain GuiComponentBounds values")
    if not all(isinstance(item, LatentComponentParameters) for item in seeds):
        raise TypeError("seed_components must contain LatentComponentParameters values")
    if (resolution_bounds is None) != (resolution_seed is None):
        raise ValueError("resolution bounds and seed must both be supplied or both be absent")
    codec = ProfiledBranchCodec.build(
        tuple(item.shape for item in bounds),
        bounds,
        tuple(item.log_D is not None for item in seeds),
        resolution_bounds=resolution_bounds,
    )
    codec.encode(seeds, resolution_seed)
    return codec, seeds


def _coefficients(profile: ProfiledForwardResult) -> np.ndarray:
    values = [profile.background, *profile.particle_amplitudes]
    if profile.resolution is not None:
        values.append(profile.resolution_amplitude)
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("profiled amplitude seed must be finite and non-negative")
    return result


def _amplitude_scales(design: np.ndarray, observed: np.ndarray) -> np.ndarray:
    basis_rms = np.sqrt(np.mean(np.square(design), axis=0))
    typical = float(np.median(observed))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scales = typical / basis_rms
    safe_high = np.sqrt(np.finfo(np.float64).max)
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
    return np.clip(scales, np.finfo(np.float64).tiny, safe_high)


def _effective_bounds(
    requested: AmplitudeBounds | None,
    count: int,
    scales: np.ndarray,
    design: np.ndarray,
    observed: np.ndarray,
) -> AmplitudeBounds:
    selected = (
        AmplitudeBounds((0.0,) * count, (np.inf,) * count) if requested is None else requested
    )
    if not isinstance(selected, AmplitudeBounds):
        raise TypeError("amplitude_bounds must be an AmplitudeBounds or None")
    if len(selected.lower) != count:
        raise ValueError("amplitude bounds length does not match the fixed branch basis")
    lower = np.asarray(selected.lower, dtype=np.float64)
    upper = np.asarray(selected.upper, dtype=np.float64)
    basis_peak = np.max(np.abs(design), axis=0)
    observed_peak = float(np.max(observed))
    intensity_cap = min(
        observed_peak * DEFAULT_AMPLITUDE_HEADROOM,
        np.sqrt(np.finfo(np.float64).max),
    )
    numerical_cap = intensity_cap / np.maximum(basis_peak, np.finfo(float).tiny)
    upper = np.where(np.isfinite(upper), upper, np.maximum(numerical_cap, scales))
    if not np.all(np.isfinite(upper)) or np.any(upper < lower):
        raise ValueError("could not construct finite effective amplitude bounds")
    return AmplitudeBounds(tuple(lower), tuple(upper))


def _project_seed(
    coefficients: np.ndarray,
    bounds: AmplitudeBounds,
    scales: np.ndarray,
    particle_count: int,
) -> np.ndarray:
    lower = np.asarray(bounds.lower, dtype=np.float64)
    upper = np.asarray(bounds.upper, dtype=np.float64)
    result = np.clip(coefficients, lower, upper)
    particle_slice = slice(1, particle_count + 1)
    if not np.any(upper[particle_slice] > 0.0):
        raise ValueError("at least one particle amplitude upper bound must be positive")
    if float(np.sum(result[particle_slice])) <= 0.0:
        candidates = np.flatnonzero(upper[particle_slice] > 0.0) + 1
        anchor = int(candidates[np.argmax(scales[candidates])])
        positive = min(float(upper[anchor]), max(float(scales[anchor]), 1.0e-12))
        if positive <= 0.0:
            positive = float(np.nextafter(0.0, 1.0))
        result[anchor] = positive
    if not bounds.contains(result):
        raise RuntimeError("failed to construct a feasible joint-amplitude seed")
    return result


def _profile_shell(
    components,
    resolution,
    coefficients,
    observed,
    constraint_audit: GuiAmplitudeConstraintAudit | None = None,
    gui_k: float | None = None,
):
    count = len(components)
    particles = np.maximum(np.asarray(coefficients[1 : count + 1]), 0.0)
    particle_total = float(np.sum(particles))
    resolution_amplitude = float(coefficients[-1]) if resolution is not None else 0.0
    branch_valid = particle_total > 0.0
    if branch_valid:
        k_value = particle_total if gui_k is None else float(gui_k)
        if not np.isfinite(k_value) or k_value <= 0.0:
            raise ValueError("joint exact GUI k must be finite and strictly positive")
        weights = particles / k_value
        int_res = resolution_amplitude / k_value
    else:
        # Exact limiting zero-particle curve: k=1 and zero GUI weights retain
        # an independently representable resolution amplitude.  This state is
        # evaluated but never eligible as a valid V8 branch result.
        k_value = 1.0
        weights = np.zeros(count, dtype=np.float64)
        int_res = resolution_amplitude
    zeros = np.zeros_like(observed)
    shell = ProfiledForwardResult(
        components=tuple(components),
        resolution=resolution,
        background=float(coefficients[0]),
        particle_amplitudes=tuple(float(value) for value in particles),
        resolution_amplitude=resolution_amplitude,
        k=k_value,
        component_weights=tuple(float(value) for value in weights),
        int_res=float(int_res),
        fitted_intensity=zeros,
        residual=zeros,
        weighted_residual=zeros,
        weighted_rss=0.0,
        solver_status=0,
        solver_message="joint exact objective state",
        solver_optimality=0.0,
        amplitude_constraint_audit=constraint_audit,
    )
    return shell, branch_valid


def optimize_joint_exact_branch(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    component_bounds: Sequence[GuiComponentBounds],
    seed_components: Sequence[LatentComponentParameters],
    *,
    resolution_bounds: ResolutionBounds | None = None,
    resolution_seed: ResolutionShape | None = None,
    sigma_log: Sequence[float] | np.ndarray | None = None,
    metric_name: str = RAW_LOG_RMSE_METRIC,
    amplitude_bounds: AmplitudeBounds | None = None,
    gui_amplitude_constraint: GuiAmplitudeConstraint | None = None,
    amplitude_seed: Sequence[float] | np.ndarray | None = None,
    max_exact_evaluations: int = 500,
    ftol: float = 1.0e-11,
    xtol: float = 1.0e-11,
    gtol: float = 1.0e-11,
) -> JointExactOptimizationResult:
    """Jointly refine one K=1..4 branch under a hard exact-call budget.

    ``gui_amplitude_constraint`` is mutually exclusive with legacy
    independent ``amplitude_bounds``.  Its coupled ranges are optimized via a
    feasible parameterization rather than an enclosing coefficient box.
    """

    q_array, observed, sigma = _curve_inputs(q, intensity, sigma_log)
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"metric_name must be one of {SUPPORTED_METRICS!r}")
    if metric_name == STANDARDIZED_LOG_RMSE_METRIC and sigma is None:
        raise ValueError("standardized log metric requires sigma_log")
    if (
        isinstance(max_exact_evaluations, bool)
        or not isinstance(max_exact_evaluations, Integral)
        or max_exact_evaluations < 1
    ):
        raise ValueError("max_exact_evaluations must be a positive integer")
    tolerances = tuple(float(value) for value in (ftol, xtol, gtol))
    if not all(np.isfinite(value) and value > np.finfo(float).eps for value in tolerances):
        raise ValueError("ftol, xtol and gtol must be finite and above machine epsilon")

    codec, seeds = _branch_codec(
        component_bounds, seed_components, resolution_bounds, resolution_seed
    )
    if amplitude_bounds is not None and gui_amplitude_constraint is not None:
        raise ValueError(
            "amplitude_bounds and gui_amplitude_constraint are mutually exclusive; "
            "coupled GUI constraints must not be approximated by an axis box"
        )
    amplitude_coordinates = None
    coefficient_polytope = None
    if gui_amplitude_constraint is not None:
        if not isinstance(gui_amplitude_constraint, GuiAmplitudeConstraint):
            raise TypeError("gui_amplitude_constraint must be a GuiAmplitudeConstraint or None")
        gui_amplitude_constraint.validate_branch(len(seeds), resolution_seed is not None)
        amplitude_coordinates = FeasibleGuiAmplitudeCoordinates.build(gui_amplitude_constraint)
        coefficient_polytope = amplitude_coordinates.polytope
    geometry_seed = codec.encode_active(seeds, resolution_seed)
    _, seed_gui, decoded_resolution = codec.decode_active(geometry_seed)
    profile_sigma = observed * (sigma if metric_name == STANDARDIZED_LOG_RMSE_METRIC else 1.0)
    profiled_seed = profile_linear_amplitudes(
        q_array,
        observed,
        seed_gui,
        resolution=decoded_resolution,
        sigma=profile_sigma,
        amplitude_constraint=gui_amplitude_constraint,
    )
    seed_coefficients = _coefficients(profiled_seed)
    design = build_design_matrix(q_array, seed_gui, decoded_resolution)
    scales = _amplitude_scales(design, observed)
    effective_bounds = (
        AmplitudeBounds(
            coefficient_polytope.axis_lower,
            coefficient_polytope.axis_upper,
        )
        if coefficient_polytope is not None
        else _effective_bounds(amplitude_bounds, seed_coefficients.size, scales, design, observed)
    )
    if amplitude_seed is not None:
        seed_coefficients = np.asarray(amplitude_seed, dtype=np.float64)
        if seed_coefficients.shape != (design.shape[1],):
            raise ValueError("amplitude_seed length does not match the fixed branch basis")
        if not np.all(np.isfinite(seed_coefficients)) or np.any(seed_coefficients < 0.0):
            raise ValueError("amplitude_seed must be finite and non-negative")
    if amplitude_coordinates is None:
        seed_coefficients = _project_seed(
            seed_coefficients, effective_bounds, scales, len(seed_gui)
        )
        lower = np.asarray(effective_bounds.lower, dtype=np.float64)
        upper = np.asarray(effective_bounds.upper, dtype=np.float64)
        free = upper > lower
        fixed = lower.copy()
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            amplitude_encoded = np.log1p(seed_coefficients[free] / scales[free])
            amplitude_lower = np.log1p(lower[free] / scales[free])
            amplitude_upper = np.log1p(upper[free] / scales[free])
    else:
        # User-provided seeds fail closed instead of being projected onto a
        # different combination of coupled GUI values.
        full_amplitude_seed = amplitude_coordinates.encode(
            seed_coefficients,
            k=profiled_seed.k,
        )
        free = amplitude_coordinates.varying_mask
        fixed = full_amplitude_seed.copy()
        amplitude_encoded = full_amplitude_seed[free]
        amplitude_lower = amplitude_coordinates.lower[free]
        amplitude_upper = amplitude_coordinates.upper[free]
    geometry_size = geometry_seed.size
    vector_seed = np.concatenate((geometry_seed, amplitude_encoded))
    vector_lower = np.concatenate((np.zeros(geometry_size), amplitude_lower))
    vector_upper = np.concatenate((np.ones(geometry_size), amplitude_upper))

    exact_calls = 0
    objective_requests = 0
    cache_hits = 0
    invalid_exact = 0
    zero_particle = 0
    last_vector = None
    last_evaluation = None
    best = None

    def decode_coefficients(values):
        if amplitude_coordinates is not None:
            assert fixed is not None and free is not None
            full = fixed.copy()
            full[free] = values[geometry_size:]
            return amplitude_coordinates.decode_with_k(full)
        assert fixed is not None and free is not None
        result = fixed.copy()
        with np.errstate(over="raise", invalid="raise"):
            result[free] = scales[free] * np.expm1(values[geometry_size:])
        result = np.maximum(result, 0.0)
        return result, float(np.sum(result[1 : len(seed_gui) + 1]))

    def evaluate(values):
        nonlocal exact_calls, invalid_exact, zero_particle
        if exact_calls >= int(max_exact_evaluations):
            raise _BudgetExhausted
        active = np.asarray(values[:geometry_size], dtype=np.float64)
        latent, gui, resolution = codec.decode_active(active)
        coefficients, gui_k = decode_coefficients(values)
        constraint_audit = (
            None
            if gui_amplitude_constraint is None
            else gui_amplitude_constraint.assess(coefficients, k=gui_k)
        )
        if constraint_audit is not None and not constraint_audit.all_constraints_satisfied:
            raise RuntimeError("joint feasible parameterization escaped the GUI amplitude polytope")
        shell, branch_valid = _profile_shell(
            gui,
            resolution,
            coefficients,
            observed,
            constraint_audit,
            gui_k,
        )
        exact_calls += 1
        if not branch_valid:
            zero_particle += 1
        try:
            exact = np.asarray(evaluate_profiled_forward(q_array, shell), dtype=np.float64)
        except (FloatingPointError, RuntimeError, ValueError):
            invalid_exact += 1
            return None
        if not np.all(np.isfinite(exact)) or np.any(exact <= 0.0):
            invalid_exact += 1
            return None
        raw_residual = np.log(exact) - np.log(observed)
        selected = raw_residual if metric_name == RAW_LOG_RMSE_METRIC else raw_residual / sigma
        if not np.all(np.isfinite(selected)):
            invalid_exact += 1
            return None
        raw_rmse = float(np.sqrt(np.mean(np.square(raw_residual))))
        standardized = (
            None if sigma is None else float(np.sqrt(np.mean(np.square(raw_residual / sigma))))
        )
        built = replace(
            shell,
            fitted_intensity=exact,
            residual=exact - observed,
            weighted_residual=selected,
            weighted_rss=float(np.dot(selected, selected)),
        )
        return _Evaluation(
            np.asarray(values, dtype=np.float64).copy(),
            tuple(float(value) for value in active),
            tuple(float(value) for value in coefficients),
            latent,
            resolution,
            built,
            exact,
            selected,
            raw_rmse,
            standardized,
            float(np.dot(selected, selected)),
            exact_calls,
            branch_valid,
        )

    initial = evaluate(vector_seed)
    if initial is None or not initial.branch_valid:
        raise ValueError("the feasible joint seed does not produce a valid positive curve")
    best = initial
    last_vector, last_evaluation = vector_seed.copy(), initial

    def residual(values):
        nonlocal objective_requests, cache_hits, last_vector, last_evaluation, best
        objective_requests += 1
        array = np.asarray(values, dtype=np.float64)
        if last_vector is not None and np.array_equal(array, last_vector):
            cache_hits += 1
            evaluated = last_evaluation
        else:
            evaluated = evaluate(array)
            last_vector = array.copy()
            last_evaluation = evaluated
        if evaluated is None:
            return np.full_like(observed, INVALID_RESIDUAL)
        if evaluated.branch_valid and evaluated.objective < best.objective:
            best = evaluated
        return evaluated.selected_residual

    solved = None
    budget_exhausted = False
    try:
        solved = least_squares(
            residual,
            vector_seed,
            bounds=(vector_lower, vector_upper),
            method="trf",
            x_scale=1.0,
            ftol=tolerances[0],
            xtol=tolerances[1],
            gtol=tolerances[2],
            max_nfev=int(max_exact_evaluations),
        )
        success, status, message = bool(solved.success), int(solved.status), str(solved.message)
        optimizer_nfev = int(solved.nfev)
        optimizer_njev = None if solved.njev is None else int(solved.njev)
        optimality = float(solved.optimality)
    except _BudgetExhausted:
        budget_exhausted = True
        success, status = False, 0
        message = f"strict exact-forward budget exhausted at {max_exact_evaluations} calls"
        optimizer_nfev = optimizer_njev = None
        optimality = float("nan")

    assert best is not None
    final_profile = replace(
        best.profile,
        solver_status=status,
        solver_message=message,
        solver_optimality=optimality,
    )
    exact_copy = np.array(best.exact, dtype=np.float64, copy=True)
    exact_copy.setflags(write=False)
    final_metric = (
        best.raw_log_rmse if metric_name == RAW_LOG_RMSE_METRIC else best.standardized_log_rmse
    )
    initial_metric = (
        initial.raw_log_rmse
        if metric_name == RAW_LOG_RMSE_METRIC
        else initial.standardized_log_rmse
    )
    assert initial_metric is not None and final_metric is not None
    terminal_returned = solved is not None and np.allclose(
        best.vector, solved.x, rtol=1e-12, atol=1e-14
    )
    returned_source = (
        "initial"
        if best.exact_call == initial.exact_call
        else "optimizer_terminal"
        if terminal_returned
        else "optimizer_intermediate"
    )
    bounds_satisfied = effective_bounds.contains(best.coefficients) and all(
        bound.contains(value) for bound, value in zip(codec.latent_bounds, best.latent_components)
    )
    if codec.resolution_bounds is not None:
        bounds_satisfied = bounds_satisfied and codec.resolution_bounds.contains(best.resolution)
    final_constraint_audit = (
        None
        if gui_amplitude_constraint is None
        else gui_amplitude_constraint.assess(best.coefficients, k=best.profile.k)
    )
    initial_constraint_audit = (
        None
        if gui_amplitude_constraint is None
        else gui_amplitude_constraint.assess(initial.coefficients, k=initial.profile.k)
    )
    if final_constraint_audit is not None:
        bounds_satisfied = (
            bounds_satisfied
            and initial_constraint_audit is not None
            and initial_constraint_audit.all_constraints_satisfied
            and final_constraint_audit.all_constraints_satisfied
        )
    if not bounds_satisfied or sum(best.coefficients[1 : len(seed_gui) + 1]) <= 0.0:
        raise RuntimeError("joint exact optimizer returned an invalid branch state")
    if exact_calls > int(max_exact_evaluations):  # pragma: no cover
        raise RuntimeError("joint exact optimizer exceeded its exact-forward budget")

    return JointExactOptimizationResult(
        version=JOINT_EXACT_OPTIMIZATION_VERSION,
        metric_name=metric_name,
        initial_active_coordinates=initial.active_coordinates,
        final_active_coordinates=best.active_coordinates,
        initial_coefficients=initial.coefficients,
        final_coefficients=best.coefficients,
        requested_amplitude_bounds=amplitude_bounds,
        effective_amplitude_bounds=effective_bounds,
        gui_amplitude_constraint=gui_amplitude_constraint,
        coefficient_polytope=coefficient_polytope,
        initial_constraint_audit=initial_constraint_audit,
        final_constraint_audit=final_constraint_audit,
        amplitude_scales=tuple(float(value) for value in scales),
        initial_latent_components=initial.latent_components,
        final_latent_components=best.latent_components,
        initial_resolution=initial.resolution,
        final_resolution=best.resolution,
        initial_profile=initial.profile,
        final_profile=final_profile,
        initial_metric=float(initial_metric),
        final_metric=float(final_metric),
        initial_raw_log_rmse=initial.raw_log_rmse,
        final_raw_log_rmse=best.raw_log_rmse,
        initial_standardized_log_rmse=initial.standardized_log_rmse,
        final_standardized_log_rmse=best.standardized_log_rmse,
        exact_intensity=exact_copy,
        success=success,
        status=status,
        message=message,
        optimizer_nfev=optimizer_nfev,
        optimizer_njev=optimizer_njev,
        objective_requests=objective_requests,
        exact_forward_calls=exact_calls,
        cache_hits=cache_hits,
        invalid_exact_evaluations=invalid_exact,
        zero_particle_evaluations=zero_particle,
        max_exact_evaluations=int(max_exact_evaluations),
        budget_exhausted=budget_exhausted,
        returned_source=returned_source,
        best_exact_call=best.exact_call,
        bounds_satisfied=True,
    )


__all__ = [
    "DEFAULT_AMPLITUDE_HEADROOM",
    "JOINT_EXACT_OPTIMIZATION_VERSION",
    "SUPPORTED_METRICS",
    "JointExactOptimizationResult",
    "optimize_joint_exact_branch",
]
