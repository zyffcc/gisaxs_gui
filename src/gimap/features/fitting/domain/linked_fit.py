"""Links and inequality constraints inside a fit (``fit_engine._Problem``; Physics and Methods, section 9.15).

Kept apart from the fitting engine: a model without links or inequalities never reaches these functions, so it is
fitted exactly as before. ``problem`` is the engine's ``_Problem`` (its free values, ranges, links and cache).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear

from . import fit_links
from .fit_model import CURVE_SCALES, LINEAR, FitModel, particle_forms, term_shape


def inequalities(problem, model: FitModel) -> list:
    """The inequalities that act: the person's (``FitModel.inequalities``) and, with ``contact``, those of the
    families and interference functions of ``model_library`` (``contact_inequalities``), each on a fitted
    non-linear value: ``(index, relation, expression)`` in order. One on a fixed value or a scale does not act."""
    extra = contact_inequalities(model) if model.contact else []
    listed = list(extra) + list(model.inequalities or ())
    if not listed:
        return []
    where = {fit_links.parameter_name(path): index for index, (path, _p) in enumerate(problem.nonlinear)}
    out = []
    for inequality in listed:
        if inequality.name in where:
            out.append((where[inequality.name], inequality.relation, fit_links.Expression(inequality.expression)))
    names = fit_links.names(model)
    for _index, _relation, expression in out:
        unknown = sorted(expression.names - set(names))
        if unknown:
            raise ValueError(f"A constraint uses {', '.join(unknown)}, which the model does not have")
    return out


def linear_ties(problem) -> list:
    """Linked scales that read fitted scales (their columns join those scales'); a size linked to a fitted scale
    is refused, and so is a link that is not linear in the scales it reads."""
    if not problem.links.ties:
        return []
    fitted = {fit_links.parameter_name(path) for path in problem.linear}
    out = []
    for name, path, _expression in problem.links.ties:
        if not problem.links.depends_on(name) & fitted:
            continue
        if path[1] not in LINEAR:
            raise ValueError(f"{name} cannot follow a fitted scale: link sizes to sizes and scales to scales")
        out.append(path)
    if out:
        check_linear(problem, out)
    return out


def check_linear(problem, tied) -> None:
    model = problem.model
    zero = model.with_values({path: 0.0 for path in problem.linear})
    base = problem.links.apply(zero)
    rng = np.random.default_rng(0)
    first, second = rng.uniform(0.5, 2.0, len(problem.linear)), rng.uniform(0.5, 2.0, len(problem.linear))
    parts = [problem.links.apply(zero.with_values(dict(zip(problem.linear, x)))) for x in (first, second, first + second)]
    for path in tied:
        values = [part.get(path).value - base.get(path).value for part in parts]
        if not math.isclose(values[0] + values[1], values[2], rel_tol=1e-6, abs_tol=1e-12 * max(1.0, abs(values[2]))):
            raise ValueError(f"The link of {fit_links.parameter_name(path)} must be linear in the scales it reads "
                             "(e.g. 0.5*Int_1)")


def limits(problem, index: int, relation: str, bound: float) -> tuple[float, float]:
    """The interval of value ``index`` under ``relation`` ``bound`` (its range when the bound is outside it)."""
    low, high = problem.lower[index], problem.upper[index]
    if not math.isfinite(bound):
        return low, high
    if relation == ">=":
        return min(max(low, bound), high - (high - low) * 1e-9), high
    return low, max(min(high, bound), low + (high - low) * 1e-9)


def current_values(problem, values) -> dict:
    current = fit_links.values(problem.model)
    current.update({fit_links.parameter_name(path): float(value) for (path, _p), value in zip(problem.nonlinear, values)})
    return current


def bound(problem, values, t=None) -> np.ndarray:
    """Each inequality in turn on ``values`` (in place): from ``t`` (decode) or into unit values (encode)."""
    current, units = current_values(problem, values), None if t is not None else np.zeros(len(values))
    intervals = {}
    for index, relation, expression in problem.bounded:
        try:
            edge = expression(current)
        except ValueError:
            edge = math.nan
        low, high = limits(problem, index, relation, edge)
        if problem.model.predict_level is not None:
            # Predict has both a lower and an upper bound on D/R and h/R.
            # Intersect them before converting this coordinate in either direction.
            old_low, old_high = intervals.get(index, (low, high))
            low, high = max(low, old_low), min(high, old_high)
            intervals[index] = low, high
        log = bool(problem.logarithmic[index])
        if t is not None:
            values[index] = problem._to_value(t[index], low, high, log)
        else:
            units[index] = problem._to_unit(values[index], low, high, log)
        current[fit_links.parameter_name(problem.nonlinear[index][0])] = float(values[index])
    return units


def scales_linked(problem, model: FitModel) -> FitModel:
    """``_scales`` with linked scales that read the fitted ones: each linked scale is c₀ + Σ cⱼ xⱼ of the fitted
    scales xⱼ (``check_linear``), so its term joins their columns and its c₀ part the fixed curve."""
    q, k = problem.data.q, float(model.globals["k"].value)
    forms = particle_forms(model, q, problem.data.dq, problem.cache, scale_columns=True)

    def basis(path):
        owner, key = path
        if key == "background":
            return np.ones_like(q)
        if key in CURVE_SCALES:
            return k * term_shape(model, key, q)
        return k * forms[owner]

    tied = list(problem.linear_ties)
    scales = [path for path, _p in model.parameters() if path[1] in LINEAR]
    fixed = np.zeros_like(q)
    for path in scales:
        if path not in problem.linear and path not in tied and model.get(path).value:
            fixed = fixed + model.get(path).value * basis(path)
    zero = model.with_values({path: 0.0 for path in problem.linear})
    base = problem.links.apply(zero)
    columns = [basis(path) for path in problem.linear]
    for path in tied:
        fixed = fixed + base.get(path).value * basis(path)
    for column, path in enumerate(problem.linear):
        unit = problem.links.apply(zero.with_values({path: 1.0}))
        for other in tied:
            factor = unit.get(other).value - base.get(other).value
            if factor:
                columns[column] = columns[column] + factor * basis(other)
    matrix = np.column_stack(columns) * problem.weights[:, None]
    target = (problem.data.intensity - fixed) * problem.weights
    low = np.array([model.get(path).lower for path in problem.linear])
    high = np.array([model.get(path).upper for path in problem.linear])
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(target))):
        return problem.links.apply(model.with_values({path: np.clip(model.get(path).value, lo, hi)
                                                   for path, lo, hi in zip(problem.linear, low, high)}))
    norms = np.maximum(np.linalg.norm(matrix, axis=0), np.finfo(float).tiny)
    solved = lsq_linear(matrix / norms, target, bounds=(low * norms, high * norms),
                        method="bvls", tol=1e-10, max_iter=100 * len(problem.linear))
    return problem.links.apply(model.with_values(dict(zip(problem.linear, np.clip(solved.x / norms, low, high)))))


def contact_inequalities(model: FitModel) -> list:
    """The contact of the families and interference functions of ``model_library`` as inequalities: the distance
    of a paracrystal at least the outer diameter 2(R + t) of a core–shell sphere, the hard-sphere radius at least
    the particle's outer radius."""
    out = []
    for index, component in enumerate(model.components):
        if not component.structure:
            continue
        number = index + 1
        outer = f"R_{number}" + (f" + t_shell_{number}" if component.family == "core_shell_sphere" else "")
        if component.interference == "hard_sphere":
            out.append(fit_links.Inequality(f"R_hs_{number}", ">=", outer))
        elif component.family == "core_shell_sphere":
            out.append(fit_links.Inequality(f"D_{number}", ">=", f"2*({outer})"))
    return out


def propagated(model: FitModel, links, paths, values, scale, covariance) -> dict:
    """Errors of the linked values: gᵀ C g, g the derivatives of each link by the free values."""
    out = {}
    for _name, tied, _expression in links.ties:
        gradient = np.zeros(len(paths))
        for column, (path, value) in enumerate(zip(paths, values)):
            step = 1e-4 * scale[column]
            try:
                upper = links.apply(model.with_values({path: value + step})).get(tied).value
                lower = links.apply(model.with_values({path: value - step})).get(tied).value
            except ValueError:
                gradient[column] = math.nan
                continue
            gradient[column] = (upper - lower) / (2.0 * step)
        used = gradient != 0
        if not np.any(used):
            out[tied] = 0.0
            continue
        part = covariance[np.ix_(used, used)]
        variance = float(gradient[used] @ part @ gradient[used]) if np.all(np.isfinite(part)) else math.nan
        out[tied] = math.sqrt(variance) if math.isfinite(variance) and variance >= 0 else math.nan
    return out


__all__ = ["bound", "check_linear", "contact_inequalities", "current_values", "inequalities", "limits", "linear_ties",
           "propagated", "scales_linked"]
