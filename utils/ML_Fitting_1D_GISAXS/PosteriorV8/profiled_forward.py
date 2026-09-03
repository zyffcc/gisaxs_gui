"""Exact GISAXS forward bases with profiled non-negative amplitudes.

The nonlinear proposal model should predict only shape parameters.  For fixed
shape parameters, the authoritative fitting equation is linear in background,
particle amplitudes and the optional resolution amplitude::

    I(q) = BG + sum(a_i * F_i(q)) + a_res * R(q)

Solving those coefficients removes the otherwise degenerate ``k * Int_i`` and
``k * int_Res`` directions from neural prediction and nonlinear refinement.
When GUI ranges are supplied, the projected coefficient solve retains the
existence of one shared legal ``k`` and reconstructs that witness afterwards;
the independent GUI ``Int_i`` values are never forced onto a simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

from src.gimap.features.fitting.domain.scattering_model import (
    _resolution_peak,
    cylinder_form_factor_pd,
    make_mixed_model,
    sphere_form_factor_pd,
    structure_factor_1d,
    vertical_cylinder_form_factor_pd,
)

from .contract import GuiComponentParameters
from .gui_amplitude_constraints import (
    CoefficientPolytope,
    GuiAmplitudeConstraint,
    GuiAmplitudeConstraintAudit,
)


# Keep the numerical role explicit at call sites while using the versioned
# contract as the single owner of component parameter semantics.
NonlinearComponent = GuiComponentParameters

PROFILED_AMPLITUDE_SOLVER_VERSION = (
    "posterior_v8_scaled_linear_profile_auxiliary_kappa_projection_v3"
)
POLYTOPE_FEASIBLE_FALLBACK_STATUS = -101


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class ResolutionShape:
    """Nonlinear shape of the optional additive resolution component."""

    sigma_res: float
    nu_res: float

    def __post_init__(self) -> None:
        sigma_res = _finite_float(self.sigma_res, "sigma_res")
        nu_res = _finite_float(self.nu_res, "nu_res")
        if sigma_res <= 0.0 or nu_res <= 0.0:
            raise ValueError("sigma_res and nu_res must be strictly positive")
        object.__setattr__(self, "sigma_res", sigma_res)
        object.__setattr__(self, "nu_res", nu_res)


@dataclass(frozen=True)
class ProfiledForwardResult:
    """Profiled coefficients and one exact GUI parameterization.

    ``component_weights`` is retained as a compatibility field name, but its
    values are the GUI's independent ``Int_i`` parameters.  They need not sum
    to one.  Gauge-independent visibility should use the effective-amplitude
    properties below.
    """

    components: tuple[NonlinearComponent, ...]
    resolution: ResolutionShape | None
    background: float
    particle_amplitudes: tuple[float, ...]
    resolution_amplitude: float
    k: float
    component_weights: tuple[float, ...]
    int_res: float
    fitted_intensity: np.ndarray
    residual: np.ndarray
    weighted_residual: np.ndarray
    weighted_rss: float
    solver_status: int
    solver_message: str
    solver_optimality: float
    amplitude_constraint_audit: GuiAmplitudeConstraintAudit | None = None

    @property
    def resolution_present(self) -> bool:
        return self.resolution is not None

    @property
    def effective_particle_total(self) -> float:
        return float(sum(self.particle_amplitudes))

    @property
    def effective_component_fractions(self) -> tuple[float, ...]:
        total = self.effective_particle_total
        if total <= 0.0:
            return tuple(0.0 for _ in self.particle_amplitudes)
        return tuple(float(value / total) for value in self.particle_amplitudes)

    @property
    def effective_resolution_ratio(self) -> float:
        total = self.effective_particle_total
        return 0.0 if total <= 0.0 else float(self.resolution_amplitude / total)

    def gui_global_parameters(self) -> dict[str, float]:
        """Return the global tail expected by ``make_mixed_model``."""
        return {
            "BG": self.background,
            "sigma_Res": 0.0 if self.resolution is None else self.resolution.sigma_res,
            "nu_Res": 0.0 if self.resolution is None else self.resolution.nu_res,
            "int_Res": self.int_res,
            "k": self.k,
        }


def _as_q(values: Sequence[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(q)):
        raise ValueError("q contains non-finite values")
    return q


def _as_observed(values: Sequence[float] | np.ndarray, q: np.ndarray) -> np.ndarray:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1 or observed.shape != q.shape:
        raise ValueError("intensity must be one-dimensional and have the same shape as q")
    if not np.all(np.isfinite(observed)):
        raise ValueError("intensity contains non-finite values")
    if np.any(observed <= 0.0):
        raise ValueError("intensity must be strictly positive")
    return observed


def _as_sigma(
    values: Sequence[float] | np.ndarray | None,
    q: np.ndarray,
) -> np.ndarray:
    if values is None:
        return np.ones_like(q)
    sigma = np.asarray(values, dtype=np.float64)
    if sigma.ndim != 1 or sigma.shape != q.shape:
        raise ValueError("sigma must be one-dimensional and have the same shape as q")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma must contain only finite, strictly positive values")
    return sigma


def _validate_components(
    components: Sequence[NonlinearComponent],
) -> tuple[NonlinearComponent, ...]:
    result = tuple(components)
    if not result:
        raise ValueError("at least one nonlinear component is required")
    if not all(isinstance(item, NonlinearComponent) for item in result):
        raise TypeError("components must contain only NonlinearComponent values")
    return result


def component_unit_basis(
    q: Sequence[float] | np.ndarray,
    component: NonlinearComponent,
) -> np.ndarray:
    """Evaluate one unit-amplitude particle basis with the exact domain model."""
    q_array = _as_q(q)
    if not isinstance(component, NonlinearComponent):
        raise TypeError("component must be a NonlinearComponent")

    if component.shape == "sphere":
        form = sphere_form_factor_pd(q_array, component.R, component.sigma_R)
    elif component.shape == "cylinder":
        assert component.h is not None and component.sigma_h is not None
        form = cylinder_form_factor_pd(
            q_array,
            component.R,
            component.sigma_R,
            component.h,
            component.sigma_h,
        )
    else:
        form = vertical_cylinder_form_factor_pd(
            q_array,
            component.R,
            component.sigma_R,
        )

    if component.D is None:
        basis = form
    else:
        assert component.sigma_D is not None
        basis = form * structure_factor_1d(q_array, component.D, component.sigma_D)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != q_array.shape or not np.all(np.isfinite(basis)):
        raise ValueError("component forward basis is non-finite or has an invalid shape")
    return basis


def resolution_unit_basis(
    q: Sequence[float] | np.ndarray,
    resolution: ResolutionShape,
) -> np.ndarray:
    """Evaluate the exact unit-amplitude additive resolution basis."""
    q_array = _as_q(q)
    if not isinstance(resolution, ResolutionShape):
        raise TypeError("resolution must be a ResolutionShape")
    basis = np.asarray(
        _resolution_peak(q_array, resolution.sigma_res, resolution.nu_res),
        dtype=np.float64,
    )
    if basis.shape != q_array.shape or not np.all(np.isfinite(basis)):
        raise ValueError("resolution forward basis is non-finite or has an invalid shape")
    return basis


def build_design_matrix(
    q: Sequence[float] | np.ndarray,
    components: Sequence[NonlinearComponent],
    resolution: ResolutionShape | None = None,
) -> np.ndarray:
    """Build columns ``[1, F_1, ..., F_K, optional R]``."""
    q_array = _as_q(q)
    component_tuple = _validate_components(components)
    columns = [np.ones_like(q_array)]
    columns.extend(component_unit_basis(q_array, item) for item in component_tuple)
    if resolution is not None:
        if not isinstance(resolution, ResolutionShape):
            raise TypeError("resolution must be None or a ResolutionShape")
        columns.append(resolution_unit_basis(q_array, resolution))
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("forward design matrix contains non-finite values")
    return matrix


def _solve_coefficient_polytope(
    scaled_design: np.ndarray,
    weighted_observed: np.ndarray,
    column_scale: np.ndarray,
    polytope: CoefficientPolytope,
    *,
    tolerance: float,
    max_iterations: int | None,
) -> tuple[np.ndarray, int, str, float]:
    lower, upper, matrix, rhs = polytope.scaled_system(column_scale)
    initial = np.asarray(polytope.feasible_coefficients) * column_scale

    def objective(values):
        residual = scaled_design @ values - weighted_observed
        return 0.5 * float(np.dot(residual, residual))

    def gradient(values):
        residual = scaled_design @ values - weighted_observed
        return scaled_design.T @ residual

    constraints = (
        ()
        if matrix.shape[0] == 0
        else (LinearConstraint(matrix, -np.inf, rhs),)
    )
    solved = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "ftol": tolerance,
            "maxiter": 1000 if max_iterations is None else int(max_iterations),
        },
    )
    terminal = np.asarray(solved.x, dtype=np.float64)
    terminal_coefficients = terminal / column_scale
    terminal_feasible = bool(
        np.all(np.isfinite(terminal)) and polytope.contains(terminal_coefficients)
    )
    if bool(solved.success) and terminal_feasible:
        optimality = float(np.linalg.norm(gradient(terminal), ord=np.inf))
        return (
            terminal_coefficients,
            int(getattr(solved, "status", 0)),
            str(solved.message),
            optimality,
        )

    # SLSQP can report a line-search or constraint-compatibility failure for
    # this convex quadratic even though the contract supplies an exact feasible
    # witness.  A failed optimizer is not evidence that the branch is invalid.
    # Retain the best *verified feasible* iterate deterministically, while
    # marking it explicitly as a fallback rather than an optimal profile.
    feasible = [("contract_witness", initial)]
    if terminal_feasible:
        feasible.append(("slsqp_terminal", terminal))
    finite = [
        (objective(values), source, values)
        for source, values in feasible
        if np.isfinite(objective(values))
    ]
    if not finite:  # pragma: no cover - validated finite inputs provide the witness
        raise RuntimeError("polytope amplitude solver has no finite feasible iterate")
    _, source, selected = min(finite, key=lambda value: (value[0], value[1]))
    coefficients = np.asarray(selected, dtype=np.float64) / column_scale
    if not polytope.contains(coefficients):  # pragma: no cover - defensive replay guard
        raise RuntimeError("polytope feasible fallback escaped its constraints")
    optimality = float(np.linalg.norm(gradient(selected), ord=np.inf))
    terminal_status = int(getattr(solved, "status", -1))
    message = (
        "feasible fallback after SLSQP failure; not an optimality certificate; "
        f"source={source}; status={terminal_status}; message={str(solved.message)}"
    )
    return coefficients, POLYTOPE_FEASIBLE_FALLBACK_STATUS, message, optimality


def profile_linear_amplitudes(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    components: Sequence[NonlinearComponent],
    *,
    resolution: ResolutionShape | None = None,
    sigma: Sequence[float] | np.ndarray | None = None,
    amplitude_constraint: GuiAmplitudeConstraint | None = None,
    solver_tolerance: float = 1e-12,
    max_iterations: int | None = None,
) -> ProfiledForwardResult:
    """Fit non-negative ``[BG, a_i..., optional a_res]`` by weighted LSQ.

    ``sigma`` defines pointwise standard deviations, so the minimized residual
    is ``(I_fit - I_observed) / sigma``.  The dense design columns are scaled
    before solving to avoid the very different absolute normalization of the
    vertical-cylinder form factor.  When supplied, ``amplitude_constraint``
    keeps the canonical GUI ranges as their full coupled coefficient
    polytope; it is not reduced to independent coefficient bounds.
    """
    q_array = _as_q(q)
    observed = _as_observed(intensity, q_array)
    sigma_array = _as_sigma(sigma, q_array)
    component_tuple = _validate_components(components)
    if amplitude_constraint is not None:
        if not isinstance(amplitude_constraint, GuiAmplitudeConstraint):
            raise TypeError("amplitude_constraint must be a GuiAmplitudeConstraint or None")
        amplitude_constraint.validate_branch(len(component_tuple), resolution is not None)
    tolerance = _finite_float(solver_tolerance, "solver_tolerance")
    if tolerance <= 0.0:
        raise ValueError("solver_tolerance must be strictly positive")
    if max_iterations is not None and int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least one or None")

    design = build_design_matrix(q_array, component_tuple, resolution)
    weighted_design = design / sigma_array[:, np.newaxis]
    weighted_observed = observed / sigma_array

    column_scale = np.linalg.norm(weighted_design, axis=0)
    if not np.all(np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise ValueError("forward design matrix contains a zero or non-finite column")
    scaled_design = weighted_design / column_scale[np.newaxis, :]

    if amplitude_constraint is None:
        solved = lsq_linear(
            scaled_design,
            weighted_observed,
            bounds=(0.0, np.inf),
            method="trf",
            tol=tolerance,
            lsmr_tol="auto",
            max_iter=None if max_iterations is None else int(max_iterations),
        )
        if not bool(solved.success):
            raise RuntimeError(f"non-negative amplitude profiling failed: {solved.message}")
        coefficients = np.asarray(solved.x, dtype=np.float64) / column_scale
        coefficients = np.maximum(coefficients, 0.0)
        solver_status = int(solved.status)
        solver_message = str(solved.message)
        solver_optimality = float(solved.optimality)
    else:
        coefficients, solver_status, solver_message, solver_optimality = (
            _solve_coefficient_polytope(
                scaled_design,
                weighted_observed,
                column_scale,
                amplitude_constraint.coefficient_polytope(),
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        )
    background = float(coefficients[0])
    count = len(component_tuple)
    particle_amplitudes = coefficients[1 : count + 1]
    particle_total = float(np.sum(particle_amplitudes))
    if not np.isfinite(particle_total) or particle_total <= 0.0:
        raise ValueError(
            "profiled particle amplitude sum is zero; the GUI k/Int gauge is undefined"
        )
    if amplitude_constraint is None:
        k_value = particle_total
    else:
        polytope = amplitude_constraint.coefficient_polytope()
        k_value = polytope.select_k(
            coefficients,
            preferred=polytope.feasible_auxiliary_k,
        )
    component_weights = particle_amplitudes / k_value

    resolution_amplitude = float(coefficients[-1]) if resolution is not None else 0.0
    int_res = resolution_amplitude / k_value if resolution is not None else 0.0
    fitted = design @ coefficients
    residual = fitted - observed
    weighted_residual = residual / sigma_array
    constraint_audit = (
        None
        if amplitude_constraint is None
        else amplitude_constraint.assess(coefficients, k=k_value)
    )
    if constraint_audit is not None and not constraint_audit.all_constraints_satisfied:
        raise RuntimeError("profiled amplitudes violate the requested GUI constraints")

    return ProfiledForwardResult(
        components=component_tuple,
        resolution=resolution,
        background=background,
        particle_amplitudes=tuple(float(value) for value in particle_amplitudes),
        resolution_amplitude=resolution_amplitude,
        k=k_value,
        component_weights=tuple(float(value) for value in component_weights),
        int_res=float(int_res),
        fitted_intensity=np.asarray(fitted, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        weighted_residual=np.asarray(weighted_residual, dtype=np.float64),
        weighted_rss=float(np.dot(weighted_residual, weighted_residual)),
        solver_status=solver_status,
        solver_message=solver_message,
        solver_optimality=solver_optimality,
        amplitude_constraint_audit=constraint_audit,
    )


def to_gui_mixed_model_parameters(
    result: ProfiledForwardResult,
) -> tuple[list[str], list[float]]:
    """Convert a profiled solution to ``make_mixed_model`` parameter order."""
    if not isinstance(result, ProfiledForwardResult):
        raise TypeError("result must be a ProfiledForwardResult")
    return _gui_mixed_model_parameters_from_coefficients(
        result.components,
        resolution=result.resolution,
        background=result.background,
        particle_amplitudes=result.particle_amplitudes,
        resolution_amplitude=result.resolution_amplitude,
        gui_k=result.k,
    )


def _gui_mixed_model_parameters_from_coefficients(
    components: Sequence[NonlinearComponent],
    *,
    resolution: ResolutionShape | None,
    background: float,
    particle_amplitudes: Sequence[float],
    resolution_amplitude: float,
    gui_k: float | None = None,
) -> tuple[list[str], list[float]]:
    """Map one geometry/linear snapshot into the lossless current GUI gauge."""

    component_tuple = _validate_components(components)
    amplitudes = tuple(
        _finite_float(value, f"particle_amplitudes[{index}]")
        for index, value in enumerate(particle_amplitudes)
    )
    if len(amplitudes) != len(component_tuple):
        raise ValueError("particle_amplitudes length must match components")
    if any(value < 0.0 for value in amplitudes):
        raise ValueError("particle_amplitudes must be non-negative")
    particle_total = float(sum(amplitudes))
    if particle_total <= 0.0:
        raise ValueError("particle_amplitudes must have a strictly positive total")
    k_value = particle_total if gui_k is None else _finite_float(gui_k, "gui_k")
    if k_value <= 0.0:
        raise ValueError("gui_k must be strictly positive")
    background_value = _finite_float(background, "background")
    resolution_amplitude_value = _finite_float(
        resolution_amplitude, "resolution_amplitude"
    )
    if background_value < 0.0 or resolution_amplitude_value < 0.0:
        raise ValueError("linear amplitudes must be non-negative")
    if resolution is None:
        if resolution_amplitude_value != 0.0:
            raise ValueError("absent resolution requires zero resolution_amplitude")
    elif not isinstance(resolution, ResolutionShape):
        raise TypeError("resolution must be None or a ResolutionShape")

    spec: list[str] = []
    params: list[float] = []
    for component, amplitude in zip(component_tuple, amplitudes):
        weight = amplitude / k_value
        spec.append(component.shape)
        if component.shape == "cylinder":
            assert component.h is not None and component.sigma_h is not None
            params.extend(
                [
                    weight,
                    component.R,
                    component.sigma_R,
                    component.h,
                    component.sigma_h,
                    0.0 if component.D is None else component.D,
                    0.0 if component.sigma_D is None else component.sigma_D,
                ]
            )
        else:
            params.extend(
                [
                    weight,
                    component.R,
                    component.sigma_R,
                    0.0 if component.D is None else component.D,
                    0.0 if component.sigma_D is None else component.sigma_D,
                ]
            )
    params.extend(
        [
            background_value,
            0.0 if resolution is None else resolution.sigma_res,
            0.0 if resolution is None else resolution.nu_res,
            0.0 if resolution is None else resolution_amplitude_value / k_value,
            k_value,
        ]
    )
    return spec, params


def evaluate_gui_forward_snapshot(
    q: Sequence[float] | np.ndarray,
    components: Sequence[NonlinearComponent],
    *,
    resolution: ResolutionShape | None,
    background: float,
    particle_amplitudes: Sequence[float],
    resolution_amplitude: float,
    gui_k: float | None = None,
) -> np.ndarray:
    """Replay a compact nonlinear/linear snapshot through the GUI model.

    This is the authoritative reconstruction path for persisted candidate
    evidence: it uses the same lossless ``k``/component-weight gauge mapping as
    :func:`evaluate_profiled_forward`, without re-solving any coefficient.
    """

    q_array = _as_q(q)
    spec, params = _gui_mixed_model_parameters_from_coefficients(
        components,
        resolution=resolution,
        background=background,
        particle_amplitudes=particle_amplitudes,
        resolution_amplitude=resolution_amplitude,
        gui_k=gui_k,
    )
    evaluated = np.asarray(make_mixed_model(spec)(q_array, *params), dtype=np.float64)
    if evaluated.shape != q_array.shape or not np.all(np.isfinite(evaluated)):
        raise ValueError("GUI forward snapshot evaluation returned invalid values")
    return evaluated


def evaluate_profiled_forward(
    q: Sequence[float] | np.ndarray,
    result: ProfiledForwardResult,
) -> np.ndarray:
    """Re-evaluate a result through the authoritative GUI mixed model."""
    if not isinstance(result, ProfiledForwardResult):
        raise TypeError("result must be a ProfiledForwardResult")
    return evaluate_gui_forward_snapshot(
        q,
        result.components,
        resolution=result.resolution,
        background=result.background,
        particle_amplitudes=result.particle_amplitudes,
        resolution_amplitude=result.resolution_amplitude,
        gui_k=result.k,
    )


__all__ = [
    "POLYTOPE_FEASIBLE_FALLBACK_STATUS",
    "PROFILED_AMPLITUDE_SOLVER_VERSION",
    "NonlinearComponent",
    "ProfiledForwardResult",
    "ResolutionShape",
    "build_design_matrix",
    "component_unit_basis",
    "evaluate_gui_forward_snapshot",
    "evaluate_profiled_forward",
    "profile_linear_amplitudes",
    "resolution_unit_basis",
    "to_gui_mixed_model_parameters",
]
