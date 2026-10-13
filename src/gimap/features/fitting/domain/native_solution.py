"""A solution of the quick physical fit (``native_v5``) in the parameters of the manual model.

The quick fit and the manual model share their formulas — sphere, random cylinder (the manual
“Cylinder”), vertical cylinder, the paracrystal S(q; D, σD) and the resolution peak
``Int_res / (1 + (q/σ_Res)^ν_Res)`` — but not their conventions:

* spreads: the quick fit gives σR/R, σh/h and σD/D; the manual Sphere and Cylinder take σR, σh
  in nm, every σD in nm, and the manual Vertical Cylinder σR/R;
* amplitudes: the quick fit multiplies normalised form factors; the manual Vertical Cylinder
  weights each radius by R⁴ (``(R·J1(qR)/q)²·10⁻⁶``), so its Int is not the quick fit's amplitude.

``native_solution_mapping`` converts the sizes and spreads, then finds the Int of every component
by non-negative least squares on the solution's own fitted curve (relative residuals, q in nm⁻¹) —
with its background and Int_res, or with those fitted too when that reproduces the curve better
(solutions whose background terms are normalised) — and says how closely the manual model
reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import nnls

from .candidates import CandidateComponentParameters, CandidateParameterMapping
from .scattering_model import _resolution_component, make_mixed_model

SHAPES = {"sphere": "sphere", "random_cylinder": "cylinder", "cylinder": "cylinder",
          "vertical_cylinder": "vertical_cylinder"}
DISPLAY = {"sphere": "Sphere", "cylinder": "Cylinder", "vertical_cylinder": "Vertical Cylinder"}


@dataclass(frozen=True)
class NativeSolutionMapping:
    mapping: CandidateParameterMapping
    max_deviation: float
    """Largest |manual model / solution − 1| on the solution's q points (NaN without them)."""


def _number(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _shape_parameters(shape: str, params: Mapping[str, Any]) -> dict[str, float]:
    """Sizes and spreads in the manual store's units; no D (1D Predict without interference): no S(q)."""
    radius = float(params["R"])
    distance, disorder = _number(params.get("D")), _number(params.get("sigma_D"))
    if distance <= 0 or disorder <= 0:
        distance = disorder = 0.0
    values = {"radius": radius, "diameter": distance, "sigma_diameter": distance * disorder}
    spread = _number(params.get("sigma_R"))
    values["sigma_radius"] = spread if shape == "vertical_cylinder" else radius * spread  # Vertical Cylinder: σR/R
    if shape == "cylinder":
        height = _number(params.get("h"), 10.0) or 10.0
        values.update(height=height, sigma_height=height * _number(params.get("sigma_h"), 0.2))
    return values


def _flat(shape: str, intensity: float, values: Mapping[str, float]) -> list[float]:
    flat = [intensity, values["radius"], values["sigma_radius"]]
    if shape == "cylinder":
        flat += [values["height"], values["sigma_height"]]
    return flat + [values["diameter"], values["sigma_diameter"]]


def _curve(shapes, intensities, shape_values, globals_, q) -> np.ndarray:
    parameters = []
    for shape, intensity, values in zip(shapes, intensities, shape_values):
        parameters += _flat(shape, intensity, values)
    parameters += [globals_["background"], globals_["sigma_res"], globals_["nu_res"], globals_["int_res"], 1.0]
    return np.asarray(make_mixed_model(list(shapes))(q, *parameters), dtype=float)


def native_solution_mapping(row: Mapping[str, Any]) -> NativeSolutionMapping:
    """``row``: a ``native_v5`` candidate (components with ``type``, ``params``, ``amplitude``; ``global_params``)."""
    components = list(row.get("components") or ())
    if not components:
        raise ValueError("The solution has no components")
    shapes, shape_values, amplitudes = [], [], []
    for component in components:
        kind = str(component.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if kind not in SHAPES:
            raise ValueError(f"Fitting has no manual model for {component.get('type')!r}")
        shape = SHAPES[kind]
        shapes.append(shape)
        shape_values.append(_shape_parameters(shape, component.get("params") or component))
        amplitudes.append(_number(component.get("amplitude", component.get("weight")), 1.0))
    source = row.get("global_params") or {}
    width = _number(source.get("sigma_Res", source.get("sigma_res")))
    exponent = _number(source.get("nu_Res", source.get("nu_res")))
    peak = width > 0 and exponent > 0  # 1D Predict may give none (its terms are then normalised)
    globals_ = {
        "background": _number(source.get("background")),
        "sigma_res": width if peak else 0.0,
        "nu_res": exponent if peak else 0.0,
        "int_res": _number(source.get("resolution_amplitude", source.get("int_res"))) if peak else 0.0,
    }
    q = np.abs(np.asarray(row.get("native_q") or (), dtype=float))
    target = np.asarray(row.get("native_fit") or (), dtype=float)
    intensities, deviation = amplitudes, float("nan")
    if q.size and q.size == target.size:
        keep = np.isfinite(q) & np.isfinite(target) & (target > 0)
        q, target = q[keep], target[keep]
        empty = {**globals_, "background": 0.0, "int_res": 0.0}
        forms = [_curve([shape], [1.0], [values], empty, q) for shape, values in zip(shapes, shape_values)]
        resolution = _resolution_component(q, globals_["sigma_res"], globals_["nu_res"], 1.0) if peak else None
        trials = []
        # (1) the solution's background and Int_res, only the Int fitted; (2) every linear coefficient
        # fitted, for solutions whose background terms are in other (normalised) units. The closer wins.
        rest = target - globals_["background"] - (0.0 if resolution is None else globals_["int_res"] * resolution)
        own = nnls(np.column_stack(forms) / target[:, None], rest / target)[0]
        trials.append((list(own), dict(globals_)))
        columns = [*forms, np.ones_like(q)] + ([] if resolution is None else [resolution])
        every = nnls(np.column_stack(columns) / target[:, None], np.ones_like(target))[0]
        solved = {"background": float(every[len(shapes)])}
        if resolution is not None:
            solved["int_res"] = float(every[len(shapes) + 1])
        trials.append((list(every[: len(shapes)]), {**globals_, **solved}))
        scored = []
        for trial_intensities, trial_globals in trials:
            model = _curve(shapes, trial_intensities, shape_values, trial_globals, q)
            scored.append((float(np.max(np.abs(model / target - 1.0))), trial_intensities, trial_globals))
        deviation, intensities, globals_ = min(scored, key=lambda item: item[0])
    mapped = tuple(
        CandidateComponentParameters(shape=DISPLAY[shape], weight=float(component.get("weight", 1.0) or 0.0),
                                     parameters={"intensity": float(intensity), **values})
        for shape, intensity, values, component in zip(shapes, intensities, shape_values, components)
    )
    global_parameters = {**globals_, "k_value": 1.0}
    return NativeSolutionMapping(CandidateParameterMapping(mapped, global_parameters), deviation)


__all__ = ["NativeSolutionMapping", "native_solution_mapping"]
