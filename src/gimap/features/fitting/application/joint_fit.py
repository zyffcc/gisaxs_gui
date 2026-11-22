"""Fit several curves together (Fitting ▸ Fit Several Curves Together…; Physics and Methods 9.18): the curves, their
kind and geometry, the models made for each from the page's model, the fit and its record.

A curve is a horizontal cut (I(|q_y|) at one exit angle: its αi, αf and λ; both halves of a signed cut on |q|), a
vertical cut (I(q_z) at one in-plane q: its αi, λ, q_∥ and, when Analyze refracted its q_z, the film's critical angle)
or another curve (no angles: the effective interface). Analyze ▸ Send Both Cuts to Fitting writes the vertical cut
with its geometry in the ``# observation:`` line (``cut`` = ``vertical``, ``alpha_i_deg``, ``wavelength_nm``,
``q_parallel_nm``, ``critical_angle_deg``, ``horizontal_alpha_f_deg``), which ``hints`` reads.

Every curve's model is the page's model with that curve's geometry: a horizontal cut with both angles and a known
substrate is in the four-term DWBA on it, without the angles at q_z = 0, otherwise at the page model's level; a
vertical cut needs the substrate (``vertical_cut``). Its beam tail is off (Tail A fixed at 0) and has no q resolution;
a tail held at 0 has its width and exponent fixed in every curve. Upright cylinders get their height h and σh/h as
parameters in every curve when a vertical cut is among them (the horizontal cut does not see it; the height is
shared).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from src.gimap.shared.version import SOFTWARE

from ..domain import fit_links
from ..domain.fit_engine import FitData
from ..domain.fit_resolution import MISSING_DQ
from ..domain.fit_model import FitModel, evaluate, model_to_dict
from ..domain.joint_fit import SHARED, JointCurve, JointResult, curve_errors, default_shared, fit_curves
from ..domain.substrates import CUSTOM, MATERIALS, Substrate
from ..domain.vertical_cut import VerticalCut

KINDS = (("horizontal", "Horizontal cut"), ("vertical", "Vertical cut"), ("other", "Other curve (q_z = 0)"))
NEEDS_SUBSTRATE = "{name}: a vertical cut needs the substrate (choose a material or type δ and β)"
NEEDS_ANGLES = "{name}: a vertical cut needs αi and λ"
NO_EXIT_ANGLE = "{name}: no point of the vertical cut lies above the sample horizon"
FEW_POINTS = "{name}: fewer than 3 points to fit"
ON_SPECULAR = ("{name}: a vertical cut at q∥ = {q} nm⁻¹ lies on or near the specular rod, which the particles' model "
               "leaves out: take the vertical cut beside it in Analyze (Cuts step), e.g. at the structure peak.")
CROSSES_ROD = ("The vertical band of {name} crosses the specular rod (qy = 0). The particle model excludes this "
               "scattering. Move the band beside the rod in Analyze and extract the cut again.")
SPECULAR_Q = 0.02
"""A vertical cut whose q∥ is below this (nm⁻¹) is warned of (``warnings``)."""


@dataclass(frozen=True)
class CurveSpec:
    """One curve of the joint fit: points (|q| or q_z in nm⁻¹), its kind and geometry."""
    name: str
    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None
    kind: str = "horizontal"
    alpha_i: float = 0.0
    alpha_f: float = 0.0
    wavelength: float = 0.1
    q_parallel: float = 0.0
    critical_angle: float = 0.0
    path: str = ""
    dq: Optional[np.ndarray] = None
    band_crosses_specular: Optional[bool] = None
    source: Optional[dict] = None

    def with_geometry(self, **changes) -> "CurveSpec":
        return replace(self, **changes)


def hints(observation: Optional[Mapping]) -> dict:
    """The kind and geometry a curve file states (``# observation:`` of Analyze's vertical cut)."""
    observation = dict(observation or {})
    if observation.get("cut") != "vertical":
        return {"kind": "horizontal"} if observation.get("source") == "native_detector_columns" else {}
    out = {"kind": "vertical"}
    if isinstance(observation.get("band_crosses_specular"), bool):
        out["band_crosses_specular"] = observation["band_crosses_specular"]
    for key, name in (("alpha_i_deg", "alpha_i"), ("wavelength_nm", "wavelength"), ("q_parallel_nm", "q_parallel"),
                      ("critical_angle_deg", "critical_angle"), ("horizontal_alpha_f_deg", "horizontal_alpha_f")):
        value = observation.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            out[name] = float(value)
    return out


def warnings(specs: Sequence["CurveSpec"]) -> list[tuple[str, dict]]:
    """(template, values) of what a person should know of the curves before fitting them together."""
    messages = []
    for spec in specs:
        if spec.kind != "vertical":
            continue
        if spec.band_crosses_specular:
            messages.append((CROSSES_ROD, {"name": spec.name}))
        elif abs(spec.q_parallel) < SPECULAR_Q:
            messages.append((ON_SPECULAR, {"name": spec.name, "q": f"{spec.q_parallel:.3g}"}))
    return messages


def substrate_of(material: str, delta: float = 0.0, beta: float = 0.0) -> Optional[Substrate]:
    """The substrate chosen: a material of the table, ``custom`` with δ and β, or ``""`` (none)."""
    if not material:
        return None
    if material == CUSTOM:
        return Substrate(CUSTOM, float(delta), float(beta))
    if material not in MATERIALS:
        raise ValueError(f"Unknown substrate {material!r}")
    return Substrate(material)


def _without_tail(model: FitModel) -> FitModel:
    model = model.with_parameter(("globals", "res_amplitude"), value=0.0, free=False, tie=None)
    return model.with_parameter(("globals", "q_resolution"), value=0.0, free=False)


def _tail_shape_fixed(model: FitModel) -> FitModel:
    """A tail held at 0 has no shape to fit."""
    tail = model.globals["res_amplitude"]
    if tail.free or tail.tie or tail.value:
        return model
    for key in ("res_width", "res_exponent"):
        model = model.with_parameter(("globals", key), free=False)
    return model


def with_heights(model: FitModel) -> FitModel:
    """Upright cylinders with their height as parameters (values kept where they were already)."""
    from ..domain.fit_model import Parameter

    components = []
    for component in model.components:
        if component.family == "vertical_cylinder" and not component.heights:
            params = dict(component.params)
            for key in ("h", "sigma_h"):
                params.setdefault(key, Parameter.default(key))
            component = replace(component, params=params, heights=True)
        components.append(component)
    return replace(model, components=tuple(components))


def curve_model(spec: CurveSpec, model: FitModel, substrate: Optional[Substrate], heights: bool) -> FitModel:
    """The page's model with this curve's geometry (module docstring)."""
    model = replace(model, cut=None)
    if heights:
        model = with_heights(model)
    if spec.kind == "vertical":
        if substrate is None:
            raise ValueError(NEEDS_SUBSTRATE.format(name=spec.name))
        if not (spec.alpha_i > 0 and spec.wavelength > 0):
            raise ValueError(NEEDS_ANGLES.format(name=spec.name))
        delta, beta = substrate.optical(spec.wavelength)
        cut = VerticalCut(float(spec.q_parallel), float(spec.wavelength), float(spec.alpha_i), delta, beta,
                          float(spec.critical_angle))
        model = _without_tail(replace(model, cut=cut, substrate=substrate, resolution_mode="constant"))
    else:
        angles = (spec.alpha_i, spec.alpha_f) if spec.kind == "horizontal" else (0.0, 0.0)
        model = model.with_values({("globals", "alpha_i"): angles[0], ("globals", "alpha_f"): angles[1],
                                   ("globals", "wavelength"): spec.wavelength})
        if substrate is not None:
            model = replace(model, substrate=substrate)
    return _tail_shape_fixed(model)


def curve_data(spec: CurveSpec, model: FitModel) -> FitData:
    """The points of a curve: |q| > 0; for a vertical cut only those with an exit angle above the horizon."""
    q = np.abs(np.asarray(spec.q, float))
    intensity = np.asarray(spec.intensity, float)
    sigma = None if spec.sigma is None else np.asarray(spec.sigma, float)
    if model.cut is not None:
        keep = np.isfinite(model.cut.exit_angles_deg(q))
        if not keep.any():
            raise ValueError(NO_EXIT_ANGLE.format(name=spec.name))
        q, intensity = q[keep], intensity[keep]
        sigma = None if sigma is None else sigma[keep]
    if model.resolution_mode == "file" and spec.dq is None:
        raise ValueError(MISSING_DQ)
    data = FitData.prepare(q, intensity, sigma, dq=spec.dq if model.resolution_mode == "file" else None,
                           weighting=model.weighting)
    if data.q.size < 3:
        raise ValueError(FEW_POINTS.format(name=spec.name))
    return data


def joint_curves(specs: Sequence[CurveSpec], model: FitModel, substrate: Optional[Substrate]) -> list[JointCurve]:
    if not specs:
        raise ValueError("Add at least one curve")
    heights = any(spec.kind == "vertical" for spec in specs)
    curves = []
    for spec in specs:
        made = curve_model(spec, model, substrate, heights)
        curves.append(JointCurve(spec.name, curve_data(spec, made), made))
    return curves


def shareable(model: FitModel) -> list[tuple]:
    """The paths a person may share or not: the free non-linear values (scales are always each curve's own)."""
    from ..domain.fit_model import LINEAR

    return [path for path, p in model.parameters() if p.free and not p.tie and path[1] not in LINEAR]


def run_joint_fit(specs: Sequence[CurveSpec], model: FitModel, substrate: Optional[Substrate], shared, *,
                  method: str = "local", progress=None, stop=None) -> JointResult:
    curves = joint_curves(specs, model, substrate)
    heights = {(owner, key) for owner, c in enumerate(curves[0].model.components) if c.heights
               for key in ("h", "sigma_h")}
    return fit_curves(curves, frozenset(shared) | heights, method=method, progress=progress, stop=stop)


def plot_points(spec: CurveSpec, model: FitModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, I, model) of a curve's fitted points (|q_y| or q_z in nm⁻¹)."""
    data = curve_data(spec, model)
    return data.q, data.intensity, evaluate(model, data.q, dq=data.dq)


def values_of(result: JointResult, path) -> list[tuple[float, float]]:
    """(value, 1σ) of one parameter in every curve (the same for a shared one)."""
    out = []
    for number, model in enumerate(result.models):
        error = result.errors.get((SHARED, path), result.errors.get((number, path), math.nan))
        out.append((float(model.get(path).value), float(error)))
    return out


def key_name(key) -> str:
    owner, path = key
    name = fit_links.parameter_name(path)
    return name if owner == SHARED else f"{name} [{owner + 1}]"


def joint_record(result: JointResult, specs: Sequence[CurveSpec], substrate: Optional[Substrate]) -> dict:
    """The record of a joint fit (Save Record…): curves, models, shared values, χ², errors, covariance."""
    def number(value):
        return None if value is None or not math.isfinite(float(value)) else float(value)

    curves = []
    for index, (spec, model) in enumerate(zip(specs, result.models)):
        curves.append({
            "name": spec.name, "path": spec.path, "kind": spec.kind, "points": result.points[index],
            "weighting": result.weightings[index], "chi2_per_point": number(result.curve_chi2[index]),
            "geometry": {"alpha_i_deg": spec.alpha_i, "alpha_f_deg": spec.alpha_f, "wavelength_nm": spec.wavelength,
                         "q_parallel_nm": spec.q_parallel if spec.kind == "vertical" else None,
                         "band_crosses_specular": spec.band_crosses_specular if spec.kind == "vertical" else None,
                         "critical_angle_deg": spec.critical_angle if spec.kind == "vertical" else None},
            "model": model_to_dict(model),
            "errors": {fit_links.parameter_name(path): number(error)
                       for path, error in curve_errors(result, index).items()},
        })
    from .fit_reproduction import joint_recipe

    return {
        "software": SOFTWARE, "kind": "gimap_joint_fit_v1", "q_unit": "nm^-1",
        **({"reproduction": joint_recipe(result, specs)} if result.start_curves else {}),
        "objective": "sum over the curves of sum r^2, r = (I - m)/sigma (ln(I/m) without sigma)",
        "method": result.method, "converged": result.converged, "stopped": result.stopped, "message": result.message,
        "chi2_reduced": number(result.chi2_reduced), "free_values": result.free, "evaluations": result.evaluations,
        "seconds": result.seconds, "substrate": None if substrate is None else substrate.to_dict(),
        "shared": sorted(fit_links.parameter_name(path) for path in result.shared),
        "at_bounds": [key_name(key) for key in result.at_bounds],
        "covariance_keys": [key_name(key) for key in result.covariance_keys],
        "covariance": [[number(v) for v in row] for row in result.covariance],
        "curves": curves,
    }


def model_for_page(page_model: FitModel, result: JointResult, number: int = 0) -> FitModel:
    """The page's model with the fitted values of curve ``number`` (shared ones included), its scales and, for a
    horizontal cut, its angles and substrate; a height the horizontal cut does not see is kept fixed at the joint
    value."""
    fitted = result.models[number]
    model = page_model
    if fitted.cut is None:
        model = replace(model, substrate=fitted.substrate).with_values(
            {("globals", key): fitted.globals[key].value for key in ("alpha_i", "alpha_f", "wavelength")})
    if any(c.heights for c in fitted.components):
        model = with_heights(model)
    values = {path: p.value for path, p in fitted.parameters() if path[0] != "globals" or path[1] in (
        "background", "res_amplitude", "res_width", "res_exponent")}
    model = model.with_values({path: value for path, value in values.items()
                               if path[0] == "globals" or path[0] < len(model.components)})
    for owner, component in enumerate(model.components):
        if component.heights:
            for key in ("h", "sigma_h"):
                model = model.with_parameter((owner, key), free=False)
    return model


__all__ = ["KINDS", "CurveSpec", "SHARED", "curve_data", "curve_model", "default_shared", "hints", "joint_curves",
           "joint_record", "key_name", "model_for_page", "plot_points", "run_joint_fit", "shareable", "substrate_of",
           "values_of", "warnings", "with_heights"]
