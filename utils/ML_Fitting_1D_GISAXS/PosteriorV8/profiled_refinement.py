"""Bounded nonlinear refinement with exact linear-amplitude profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from .branch_codec import (
    HARD_CORE_SPACING_MARGIN,
    ProfiledBranchCodec,
    ResolutionBounds,
)
from .contract import (
    MAX_COMPONENTS,
    MIN_COMPONENTS,
    GuiComponentBounds,
    LatentComponentParameters,
)
from .gui_amplitude_constraints import (
    GuiAmplitudeConstraint,
    GuiAmplitudeConstraintAudit,
)
from .profiled_forward import (
    ProfiledForwardResult,
    ResolutionShape,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)
from .sobol_numeric_canonicalization_v5 import V5_FAST_NUMERIC_POLICY_VERSION


@dataclass(frozen=True)
class ProfiledRefinementResult:
    initial_profile: ProfiledForwardResult
    final_profile: ProfiledForwardResult
    initial_latent_components: tuple[LatentComponentParameters, ...]
    final_latent_components: tuple[LatentComponentParameters, ...]
    initial_resolution: ResolutionShape | None
    final_resolution: ResolutionShape | None
    initial_log_rmse: float
    final_log_rmse: float
    initial_weighted_log_rmse: float
    final_weighted_log_rmse: float
    exact_forward_intensity: np.ndarray
    success: bool
    status: int
    message: str
    nfev: int
    njev: int | None
    residual_calls: int
    bounds_satisfied: bool
    gui_amplitude_constraint: GuiAmplitudeConstraint | None
    initial_constraint_audit: GuiAmplitudeConstraintAudit | None
    final_constraint_audit: GuiAmplitudeConstraintAudit | None
    exact_forward_calls: int
    exact_forward_call_phases: tuple[str, ...]


def _curve_inputs(q, intensity, sigma_log):
    q_array = np.asarray(q, dtype=np.float64)
    observed = np.asarray(intensity, dtype=np.float64)
    if q_array.ndim != 1 or q_array.size == 0:
        raise ValueError("q must be a non-empty one-dimensional array")
    if observed.ndim != 1 or observed.shape != q_array.shape:
        raise ValueError("intensity must have the same one-dimensional shape as q")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q contains non-finite values")
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("intensity must contain finite, strictly positive values")
    if sigma_log is None:
        log_sigma = np.ones_like(observed)
    else:
        log_sigma = np.asarray(sigma_log, dtype=np.float64)
        if log_sigma.ndim != 1 or log_sigma.shape != observed.shape:
            raise ValueError("sigma_log must have the same one-dimensional shape as q")
        if not np.all(np.isfinite(log_sigma)) or np.any(log_sigma <= 0.0):
            raise ValueError("sigma_log must contain finite, strictly positive values")
    return q_array, observed, log_sigma


def _log_metrics(fitted, observed, sigma_log):
    residual = np.log(np.maximum(fitted, 1e-300)) - np.log(observed)
    return (
        residual,
        float(np.sqrt(np.mean(np.square(residual)))),
        float(np.sqrt(np.mean(np.square(residual / sigma_log)))),
    )


def _branch_codec(
    component_bounds: Sequence[GuiComponentBounds],
    seed_components: Sequence[LatentComponentParameters],
    resolution_bounds: ResolutionBounds | None,
    resolution_seed: ResolutionShape | None,
    numeric_policy_version: str,
) -> tuple[
    ProfiledBranchCodec,
    tuple[LatentComponentParameters, ...],
]:
    bounds = tuple(component_bounds)
    seeds = tuple(seed_components)
    if not MIN_COMPONENTS <= len(bounds) <= MAX_COMPONENTS:
        raise ValueError("component bounds must contain between one and four components")
    if len(bounds) != len(seeds):
        raise ValueError("component bounds and latent seeds must have the same length")
    if not all(isinstance(item, GuiComponentBounds) for item in bounds):
        raise TypeError("component_bounds must contain GuiComponentBounds values")
    if not all(isinstance(item, LatentComponentParameters) for item in seeds):
        raise TypeError("seed_components must contain LatentComponentParameters values")
    if (resolution_bounds is None) != (resolution_seed is None):
        raise ValueError(
            "resolution bounds and seed must either both be supplied or both be absent"
        )
    topology = tuple(item.shape for item in bounds)
    d_present = tuple(item.log_D is not None for item in seeds)
    codec = ProfiledBranchCodec.build(
        topology,
        bounds,
        d_present,
        resolution_bounds=resolution_bounds,
        numeric_policy_version=numeric_policy_version,
    )
    # Encoding is also the fail-closed seed and hard-core validation step.
    codec.encode(seeds, resolution_seed)
    return codec, seeds


def refine_profiled_branch(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    component_bounds: Sequence[GuiComponentBounds],
    seed_components: Sequence[LatentComponentParameters],
    *,
    resolution_bounds: ResolutionBounds | None = None,
    resolution_seed: ResolutionShape | None = None,
    sigma_log: Sequence[float] | np.ndarray | None = None,
    amplitude_constraint: GuiAmplitudeConstraint | None = None,
    max_nfev: int = 120,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
    exact_forward_call_hook: Callable[[str], None] | None = None,
    numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
) -> ProfiledRefinementResult:
    """Refine one fixed topology/D/resolution branch from one latent seed.

    ``amplitude_constraint`` is optional for legacy callers.  When supplied,
    every geometry evaluation profiles amplitudes inside that exact coupled
    GUI coefficient polytope.  ``exact_forward_call_hook`` is an accounting
    seam: it is invoked before each complete geometry-objective evaluation,
    whose budget unit includes amplitude profiling and the authoritative
    GUI-forward verification.  The hook must not perform scientific work.
    """

    q_array, observed, log_sigma = _curve_inputs(q, intensity, sigma_log)
    if isinstance(max_nfev, bool) or int(max_nfev) != max_nfev or int(max_nfev) < 1:
        raise ValueError("max_nfev must be a positive integer")
    tolerances = [float(ftol), float(xtol), float(gtol)]
    if not all(np.isfinite(value) and value > np.finfo(float).eps for value in tolerances):
        raise ValueError("ftol, xtol and gtol must be finite and greater than machine epsilon")

    codec, seeds = _branch_codec(
        component_bounds,
        seed_components,
        resolution_bounds,
        resolution_seed,
        numeric_policy_version,
    )
    if amplitude_constraint is not None:
        if not isinstance(amplitude_constraint, GuiAmplitudeConstraint):
            raise TypeError("amplitude_constraint must be a GuiAmplitudeConstraint or None")
        amplitude_constraint.validate_branch(len(seeds), resolution_seed is not None)
    if exact_forward_call_hook is not None and not callable(exact_forward_call_hook):
        raise TypeError("exact_forward_call_hook must be callable or None")
    template_coordinates = codec.encode(seeds, resolution_seed)
    x0 = codec.encode_varying(seeds, resolution_seed)
    profile_sigma = observed * log_sigma
    exact_forward_call_phases: list[str] = []

    def evaluate(values, phase: str):
        latent, gui, resolution = codec.decode_varying(values, template_coordinates)
        if exact_forward_call_hook is not None:
            exact_forward_call_hook(phase)
        exact_forward_call_phases.append(phase)
        profile = profile_linear_amplitudes(
            q_array,
            observed,
            gui,
            resolution=resolution,
            sigma=profile_sigma,
            amplitude_constraint=amplitude_constraint,
        )
        exact = evaluate_profiled_forward(q_array, profile)
        if not np.allclose(exact, profile.fitted_intensity, rtol=2e-10, atol=1e-12):
            raise RuntimeError("profiled and authoritative exact forward curves disagree")
        log_residual, log_rmse, weighted_log_rmse = _log_metrics(exact, observed, log_sigma)
        return latent, resolution, profile, exact, log_residual, log_rmse, weighted_log_rmse

    initial = evaluate(x0, "initial_profile_verification")
    best_x = x0.copy()
    best_objective = float(np.mean(np.square(initial[4] / log_sigma)))
    residual_calls = 0

    if x0.size == 0:
        return ProfiledRefinementResult(
            initial_profile=initial[2],
            final_profile=initial[2],
            initial_latent_components=initial[0],
            final_latent_components=initial[0],
            initial_resolution=resolution_seed,
            final_resolution=initial[1],
            initial_log_rmse=initial[5],
            final_log_rmse=initial[5],
            initial_weighted_log_rmse=initial[6],
            final_weighted_log_rmse=initial[6],
            exact_forward_intensity=initial[3],
            success=True,
            status=0,
            message="no user-query varying geometry dimensions; exact profile only",
            nfev=0,
            njev=0,
            residual_calls=0,
            bounds_satisfied=True,
            gui_amplitude_constraint=amplitude_constraint,
            initial_constraint_audit=initial[2].amplitude_constraint_audit,
            final_constraint_audit=initial[2].amplitude_constraint_audit,
            exact_forward_calls=len(exact_forward_call_phases),
            exact_forward_call_phases=tuple(exact_forward_call_phases),
        )

    def residual(values):
        nonlocal best_x, best_objective, residual_calls
        residual_calls += 1
        try:
            evaluated = evaluate(values, "optimizer_residual")
            weighted = evaluated[4] / log_sigma
        except (FloatingPointError, RuntimeError, ValueError):
            return np.full_like(observed, 1e6)
        objective = float(np.mean(np.square(weighted)))
        if objective < best_objective:
            best_objective = objective
            best_x = np.asarray(values, dtype=np.float64).copy()
        return weighted

    solved = least_squares(
        residual,
        x0,
        bounds=(np.zeros_like(x0), np.ones_like(x0)),
        method="trf",
        ftol=tolerances[0],
        xtol=tolerances[1],
        gtol=tolerances[2],
        max_nfev=int(max_nfev),
        x_scale=1.0,
    )
    solved_evaluation = evaluate(solved.x, "optimizer_terminal_verification")
    solved_objective = float(np.mean(np.square(solved_evaluation[4] / log_sigma)))
    final = evaluate(
        best_x if best_objective <= solved_objective else solved.x,
        "optimizer_terminal_verification",
    )

    bounds_satisfied = all(
        bounds.contains(latent, codec.numeric_policy_version)
        for bounds, latent in zip(codec.latent_bounds, final[0])
    ) and (codec.resolution_bounds is None or codec.resolution_bounds.contains(final[1]))
    if not bounds_satisfied:
        raise RuntimeError("refinement produced an out-of-bounds result")

    return ProfiledRefinementResult(
        initial_profile=initial[2],
        final_profile=final[2],
        initial_latent_components=initial[0],
        final_latent_components=final[0],
        initial_resolution=resolution_seed,
        final_resolution=final[1],
        initial_log_rmse=initial[5],
        final_log_rmse=final[5],
        initial_weighted_log_rmse=initial[6],
        final_weighted_log_rmse=final[6],
        exact_forward_intensity=final[3],
        success=bool(solved.success),
        status=int(solved.status),
        message=str(solved.message),
        nfev=int(solved.nfev),
        njev=None if solved.njev is None else int(solved.njev),
        residual_calls=int(residual_calls),
        bounds_satisfied=True,
        gui_amplitude_constraint=amplitude_constraint,
        initial_constraint_audit=initial[2].amplitude_constraint_audit,
        final_constraint_audit=final[2].amplitude_constraint_audit,
        exact_forward_calls=len(exact_forward_call_phases),
        exact_forward_call_phases=tuple(exact_forward_call_phases),
    )


__all__ = [
    "HARD_CORE_SPACING_MARGIN",
    "ProfiledRefinementResult",
    "ResolutionBounds",
    "refine_profiled_branch",
]
