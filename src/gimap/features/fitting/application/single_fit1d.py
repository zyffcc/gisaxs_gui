"""A fit1d job for the single-curve page, independent of V5/TensorFlow assets.

Two ways to say what is fitted (``fit1d_composition``): ``model`` — the families of Model, with its fixed values
and the ranges a user changed passed to fit1d as constraints (``domain.fit1d_bounds``) — or an ``auto:`` mode of the
kit, which fits every composition of the given shapes and ranks them (the Model is not used then: the populations
differ from one composition to the next). The job resolves parameters that stop at an edge of their range by
itself (``domain.fit1d_edges``, ``ROUNDS`` rounds at most) and says what it did.
"""

from __future__ import annotations

import math

import numpy as np

from src.gimap.app.jobs import JobRequest

from ..domain.fit1d_bounds import MODEL_KEYS, TO_KIT, Constraint, model_constraints, population_order, range_notes
from ..domain.fit1d_edges import ROUNDS
from ..domain.fit_engine import FitResult, log_rmse
from ..domain.fit_model import FitModel
from .single_fit import model_from_solution
from .workflow_v5 import validate_options

SHAPES = {"sphere": 1, "vertical_cylinder": 3}
COMPOSITIONS = (
    ("model", "The particles in Model"),
    ("auto:vc:3", "Try 1–3 vertical cylinders"),
    ("auto:sphere,vc:3", "Try spheres and vertical cylinders (1–3)"),
)
"""What fit1d fits: Model's families (with its constraints), or every composition of these shapes, ranked."""
AUTO_SHAPES = {"auto:vc:3": [3], "auto:sphere,vc:3": [1, 3]}
AUTO_IGNORES_MODEL = ("Trying compositions: the Model's fixed values and ranges are not passed (the populations "
                      "differ between compositions). Use a candidate, then predict again with its particles to "
                      "use them.")
NO_PARTICLES = "Model has no particles: add Sphere / Vertical cylinder particles, or choose “Try …” compositions."


def is_auto(settings: dict) -> bool:
    return str(settings.get("fit1d_composition") or "model").startswith("auto")


def fit1d_constraints(model: FitModel, settings: dict) -> list[Constraint]:
    if is_auto(settings):
        return []
    return model_constraints(model, structure_factor=settings.get("fit1d_mode", "gisaxs") == "gisaxs")


def prefit_notes(data, model: FitModel, settings: dict) -> list:
    """What to know before a fit (``Note``s): ranges narrower than the curve can measure or beyond it."""
    if data is None or data.q.size < 3:
        return []
    q = np.abs(np.asarray(data.q, float))
    q = q[q > 0]
    if q.size < 3:
        return []
    try:
        constraints = fit1d_constraints(model, settings)
    except ValueError:
        return []  # said when the fit is asked for
    return range_notes(constraints, float(q.min()), float(q.max()))


def fit1d_request(data, model, settings: dict, *, resolve_edges: bool = True, previous=None) -> JobRequest:
    """The families in Model are conditions (or an ``auto:`` composition); constraints from Model's fixed values
    and changed ranges."""
    composition = str(settings.get("fit1d_composition") or "model")
    if composition == "model":
        if not 1 <= len(model.components) <= 4 or any(c.family not in SHAPES for c in model.components):
            raise ValueError("fit1d needs 1–4 particles in Model: Sphere or Vertical cylinder")
        components = [SHAPES[c.family] for c in model.components]
    elif composition in AUTO_SHAPES:
        components = AUTO_SHAPES[composition]
    else:
        raise ValueError(f"Unknown fit1d composition {composition!r}")
    options = validate_options(dict(method="fit1d", components=components,
                                    **{**settings, "fit1d_composition": composition}))
    constraints = fit1d_constraints(model, settings)
    order = population_order([c.family for c in model.components]) if composition == "model" else []
    return JobRequest(
        handler="src.gimap.features.fitting.infrastructure.adapters.single_fit1d:run_single_fit1d",
        payload=dict(q=data.q.tolist(), intensity=data.intensity.tolist(),
                     sigma=None if data.sigma is None else data.sigma.tolist(), options=options,
                     composition=composition, constraints=[item.to_dict() for item in constraints],
                     population_order=order, rounds=ROUNDS if resolve_edges else 0,
                     previous=list(previous or ())),
        timeout_seconds=None,
    )


def model_order(model: FitModel) -> list[int]:
    """The Model's component of pop1, pop2, … (how fit1d numbers the populations)."""
    return population_order([c.family for c in model.components])


ERRORS = {"R_nm_err": "R", "sigma_R_rel_err": "sigma_R", "D_nm_err": "D", "w_rel_err": "sigma_D"}
"""The kit's 1σ errors → the Model's parameter (σD/D = w/D / sqrt(2π), its error likewise)."""


def fit1d_result(value: dict, base: FitModel, data) -> FitResult:
    """A fit1d job's result as a result of Fitting's model (a frame of a series): the kit's values in the Model's
    parameters (``model_from_solution``, components in the Model's order), the kit's errors, χ²/dof of the kit's own
    curve. The kit's result itself stays with it (``FrameFit.native``)."""
    native = value["native"]
    model, _deviation = model_from_solution(in_model_order(value["mapping_row"], base), base)
    errors = {}
    for population in native.get("populations") or ():
        for source, key in ERRORS.items():
            path = model_path(model, f"{population.get('name')}.{source[:-4]}")
            error = population.get(source)
            if path is not None and isinstance(error, (int, float)) and math.isfinite(error) and error > 0:
                errors[path] = float(error) / (TO_KIT.get(key, 1.0))
    summary = native["summary"]
    points = int(summary.get("n_points") or data.q.size)
    dof = max(int(summary.get("dof") or points), 1)
    chi2 = summary.get("chi2")
    chi2 = float(chi2) if isinstance(chi2, (int, float)) else float(summary.get("chi2_per_point") or 0) * points
    free = sum(1 for _path, parameter in model.parameters() if parameter.free)
    return FitResult(model=model, method="fit1d", weighting="sigma", points=points, free=free, chi2_reduced=chi2 / dof,
                     log_rmse=float(value["mapping_row"].get("best_log_rmse") or log_rmse(model, data)),
                     evaluations=0, seconds=float(summary.get("seconds") or 0.0), converged=True, stopped=False,
                     message="fit1d", errors=errors)


def model_path(model: FitModel, name: str):
    """The Model's (component, key) of a fit1d parameter (``"pop2.R_nm"``); None for a global term or another
    composition than Model's."""
    target, _, key = str(name).partition(".")
    order = model_order(model) if all(c.family in SHAPES for c in model.components) else []
    if not target.startswith("pop") or not target[3:].isdigit() or key not in MODEL_KEYS:
        return None
    number = int(target[3:])
    if not 1 <= number <= len(order) or MODEL_KEYS[key] not in model.components[order[number - 1]].params:
        return None
    return order[number - 1], MODEL_KEYS[key]


def in_model_order(row: dict, model: FitModel) -> dict:
    """A kit row (populations in the kit's order) with its components in Model's order, when it has Model's
    families, so that fixed values and ranges stay with their particle."""
    families = [c.family for c in model.components]
    components = list(row.get("components") or ())
    kinds = [str(c.get("type")) for c in components]
    if sorted(kinds) != sorted(families) or len(kinds) != len(families):
        return row
    order = population_order(families)
    placed = [None] * len(components)
    for position, index in enumerate(order):
        placed[index] = components[position]
    return {**row, "components": placed}


def native_export(result: dict) -> tuple[list[str], np.ndarray]:
    """Measured points, effective errors and the original v2 curve/terms, without re-evaluation."""
    curves = result["curves"]
    q = np.asarray(curves["q_nm_inv"], float)
    names = ["q_nm^-1", "q_A^-1", "I", "sigma", "model", "residual"]
    columns = [q, q / 10.0, curves["intensity"], curves["sigma"], curves["model"], curves["residual"]]
    for name, values in curves.get("components", {}).items():
        names.append(name.replace(" ", "_").replace(":", ""))
        columns.append(values)
    return names, np.column_stack(columns)


__all__ = ["AUTO_IGNORES_MODEL", "COMPOSITIONS", "NO_PARTICLES", "fit1d_constraints", "fit1d_result",
           "fit1d_request", "in_model_order", "is_auto", "model_order", "model_path", "native_export", "prefit_notes"]
