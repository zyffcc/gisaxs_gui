"""The model of one curve in Fitting: its parameters (value, free or fixed, range) and its intensity.

    I(q) = BG + k · [ Σ_i Int_i · P_i(q) · S_i(q)  +  A / (1 + (|q| / w)^ν) ]

* P_i: a particle family — sphere, random cylinder, vertical cylinder — averaged over a Gaussian
  size distribution; its spreads are **relative** here (σR/R, σh/h) whatever the family;
* S_i: the paracrystal (1D) interference of distance D and relative disorder σD/D, when the
  component has one (``structure``), else 1;
* BG a constant background, A/(1+(|q|/w)^ν) the resolution peak at q = 0, k an overall factor.

Spheres of a horizontal GISAXS cut: with αi = αf = 0 (the default) the sphere is evaluated at q_z = 0, the
effective interface approximation, which needs nothing about the substrate. With the angles and λ of the cut
the sphere amplitude is the sum of the direct and the exit-reflected path of the DWBA,
    A = f(√(q² + q₊²) R) e^{-i q₊ R} + r f(√(q² + q₋²) R) e^{-i q₋ R},  q₊ = k_i + k_f,  q₋ = k_i − k_f,
with r = |r| e^{i arg r} a free effective reflection coefficient (|r| = 0: Born at the vacuum q_z).
Cylinders keep their form: a vertical cylinder's paths share the in-plane factor (exact), and a random
cylinder is orientation-averaged at |q|.

q in nm⁻¹, sizes in nm. The formulas are those of ``scattering_model`` (the manual model) with
the same sampling, so a model here and the same model in the manual parameter store give the
same curve; ``to_manual`` / ``from_manual`` convert the spreads (nm for spheres and random
cylinders, σR/R for vertical cylinders, σD in nm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

import numpy as np

from .candidates import CandidateComponentParameters, CandidateParameterMapping
from .coherent_terms import cylinder_coherent_pd, sphere_coherent_pd, vertical_cylinder_coherent_pd
from .scattering_model import (
    _resolution_peak,
    cylinder_axial_average,
    cylinder_orientation_average,
    cylinder_radial_average,
    sphere_form_factor_pd,
    sphere_two_path_pd,
    structure_factor_1d,
    vertical_cylinder_form_factor_pd,
)

CACHE_SIZE = 96

FAMILIES = {
    "sphere": ("Sphere", ("R", "sigma_R")),
    "cylinder": ("Random cylinder", ("R", "sigma_R", "h", "sigma_h")),
    "vertical_cylinder": ("Vertical cylinder", ("R", "sigma_R")),
}
MANUAL_NAMES = {"sphere": "Sphere", "cylinder": "Cylinder", "vertical_cylinder": "Vertical Cylinder"}
STRUCTURE = ("D", "sigma_D")
BASE_GLOBALS = ("background", "res_amplitude", "res_width", "res_exponent", "k")
GRAZING = ("alpha_i", "alpha_f", "wavelength", "refl_amp", "refl_phase")
"""Geometry of the cut and the effective reflection coefficient r of the two-path sphere (fixed by default)."""
GLOBALS = BASE_GLOBALS + GRAZING
LINEAR = frozenset({"Int", "background", "res_amplitude"})
"""Scales: with the others fixed, the intensity is linear in them (solved exactly while fitting)."""
INF = math.inf


@dataclass(frozen=True)
class ParameterInfo:
    label: str
    unit: str
    value: float
    lower: float
    upper: float
    free: bool = True
    tip: str = ""


INFO = {
    "Int": ParameterInfo("Scale", "", 1.0, 0.0, INF, tip=(
        "Intensity of this component (sphere and random cylinder: the form factor is 1 at q = 0; "
        "vertical cylinder: 10⁻⁶·⟨R⁴⟩/4 at q = 0, R in nm)"
    )),
    "R": ParameterInfo("R", "nm", 5.0, 0.3, 500.0, tip="Mean radius"),
    "sigma_R": ParameterInfo("σR/R", "", 0.2, 0.01, 1.0, tip="Relative spread of the radius (Gaussian)"),
    "h": ParameterInfo("h", "nm", 10.0, 0.5, 1000.0, tip="Mean height (length) of the cylinders"),
    "sigma_h": ParameterInfo("σh/h", "", 0.2, 0.01, 1.0, tip="Relative spread of the height (Gaussian)"),
    "D": ParameterInfo("D", "nm", 30.0, 1.0, 3000.0, tip="Mean distance between neighbours (paracrystal)"),
    "sigma_D": ParameterInfo("σD/D", "", 0.3, 0.02, 1.0, tip="Relative disorder of the distance"),
    "background": ParameterInfo("Background", "", 0.0, 0.0, INF, tip="Constant added to the whole curve"),
    "res_amplitude": ParameterInfo("Peak A", "", 0.0, 0.0, INF,
                                   tip="Height of the resolution peak A / (1 + (|q|/w)^ν) at q = 0"),
    "res_width": ParameterInfo("Peak w", "nm⁻¹", 0.02, 0.001, 0.1, tip="Width w of the resolution peak"),
    "res_exponent": ParameterInfo("Peak ν", "", 3.0, 1.0, 20.0, tip="Exponent ν of the resolution peak (its tail)"),
    "k": ParameterInfo("Factor k", "", 1.0, 1e-6, 1e6, free=False,
                       tip="Overall factor of everything but the background (usually fixed at 1)"),
    "alpha_i": ParameterInfo("αi", "°", 0.0, 0.0, 5.0, free=False,
                             tip="Incidence angle of the cut; with αf = 0 too, spheres at q_z = 0 (effective interface)"),
    "alpha_f": ParameterInfo("αf", "°", 0.0, 0.0, 5.0, free=False,
                             tip="Exit angle of the cut (the detector row of the horizontal cut)"),
    "wavelength": ParameterInfo("λ", "nm", 0.1, 0.005, 1.0, free=False, tip="Wavelength"),
    "refl_amp": ParameterInfo("|r|", "", 0.0, 0.0, 1.0, free=False,
                              tip="Size of the effective reflection coefficient of the exit-reflected path (spheres)"),
    "refl_phase": ParameterInfo("arg r", "rad", 0.0, -math.pi, math.pi, free=False,
                                tip="Phase of the effective reflection coefficient (spheres)"),
}


@dataclass(frozen=True)
class Parameter:
    value: float
    free: bool = True
    lower: float = 0.0
    upper: float = INF

    @classmethod
    def default(cls, key: str, value: Optional[float] = None) -> "Parameter":
        info = INFO[key]
        return cls(info.value if value is None else float(value), info.free, info.lower, info.upper)


@dataclass(frozen=True)
class Component:
    family: str
    params: Mapping[str, Parameter]
    structure: bool = True
    coupling: str = "lma"
    """How the paracrystal enters (``COUPLINGS``): ``lma``, P(q)·S(q) (local monodisperse approximation), or
    ``da``, P(q) + |<F>|²(S(q) − 1) (decoupling approximation: the sizes are uncorrelated with the positions)."""

    def keys(self) -> tuple[str, ...]:
        return ("Int", *FAMILIES[self.family][1], *(STRUCTURE if self.structure else ()))

    def value(self, key: str) -> float:
        return float(self.params[key].value)


@dataclass(frozen=True)
class FitModel:
    components: tuple[Component, ...] = ()
    globals: Mapping[str, Parameter] = field(default_factory=lambda: {
        key: Parameter.default(key) for key in GLOBALS
    })

    # -- addressing: ("globals", key) or (component index, key) ----------------------

    def parameters(self) -> list[tuple[tuple, Parameter]]:
        """Every parameter in use, components first."""
        items = [((index, key), component.params[key])
                 for index, component in enumerate(self.components) for key in component.keys()]
        return items + [(("globals", key), self.globals[key]) for key in GLOBALS]

    def get(self, path: tuple) -> Parameter:
        owner, key = path
        return self.globals[key] if owner == "globals" else self.components[owner].params[key]

    def with_parameter(self, path: tuple, **changes) -> "FitModel":
        owner, key = path
        if owner == "globals":
            return replace(self, globals={**self.globals, key: replace(self.globals[key], **changes)})
        components = list(self.components)
        component = components[owner]
        components[owner] = replace(component, params={**component.params, key: replace(component.params[key], **changes)})
        return replace(self, components=tuple(components))

    def with_values(self, values: Mapping[tuple, float]) -> "FitModel":
        model = self
        for path, value in values.items():
            model = model.with_parameter(path, value=float(value))
        return model


def new_component(family: str, *, radius: Optional[float] = None, distance: Optional[float] = None,
                  structure: bool = True) -> Component:
    if family not in FAMILIES:
        raise ValueError(f"Unknown particle family {family!r}")
    params = {key: Parameter.default(key) for key in ("Int", *FAMILIES[family][1], *STRUCTURE)}
    if radius is not None:
        params["R"] = replace(params["R"], value=float(radius))
    if distance is not None:
        params["D"] = replace(params["D"], value=float(distance))
    elif radius is not None:
        params["D"] = replace(params["D"], value=max(3.0 * float(radius), params["D"].lower))
    return Component(family, params, structure)


def _cached(cache: Optional[dict], key: tuple, compute):
    """``compute()`` once per ``key`` while fitting one set of points (``cache`` belongs to that fit)."""
    if cache is None:
        return compute()
    if key not in cache:
        if len(cache) >= CACHE_SIZE:
            cache.clear()
        cache[key] = compute()
    return cache[key]


def grazing(model: FitModel) -> Optional[tuple]:
    """(q₊, q₋, k_f, |r|, arg r) of the two-path sphere, or None for the effective interface (q_z = 0)."""
    g = {key: float(model.globals[key].value) for key in GRAZING}
    if g["alpha_i"] == 0 and g["alpha_f"] == 0 and g["refl_amp"] == 0:
        return None
    k = 2 * math.pi / g["wavelength"]
    k_i, k_f = k * math.sin(math.radians(g["alpha_i"])), k * math.sin(math.radians(g["alpha_f"]))
    return (k_i + k_f, k_i - k_f, k_f, g["refl_amp"], g["refl_phase"])


def qz_plus_r(model: FitModel) -> list[tuple[int, float]]:
    """q₊·R of every sphere (component index, value), q₊ = k_i + k_f; empty without the angles of the cut."""
    paths = grazing(model)
    if paths is None or paths[0] <= 0:
        return []
    return [(index, paths[0] * c.value("R")) for index, c in enumerate(model.components) if c.family == "sphere"]


NUMBER_POWER = {"sphere": 6, "cylinder": 4, "vertical_cylinder": 0}
"""What turns the radius weights of a family into number weights: the sphere and the random cylinder are normalised
to P(0) = 1, so their Gaussian is one of scattered intensity (number × V², V² ∝ R⁶ and R⁴ at a given height); the
vertical cylinder keeps R⁴ in its form factor (9.10), so its Gaussian already counts particles."""
SIZE_NODES = {"sphere": (25, 4.0), "cylinder": (13, 4.0), "vertical_cylinder": (26, 3.0)}


TAIL_SHARE = 0.1
"""The number distribution is reported only when it peaks inside the grid and its smallest node holds less than this
share of its largest."""


def number_size(component: Component) -> Optional[tuple[float, float]]:
    """(mean, relative standard deviation) of the radius by number, on the nodes of the component's size average,
    for comparison with sizes counted in microscopy or with fit1d (whose R is number-weighted).

    ``None`` when it is not defined: an intensity-weighted Gaussian of spheres is G(R)/R⁶ by number, which rises
    towards small R once σ_R/R exceeds about 1/√24 ≈ 0.2 (1/√16 = 0.25 for random cylinders); its mean would then be
    set by where the size grid is cut, not by the data."""
    from .scattering_model import _gaussian_grid

    radius, spread = component.value("R"), component.value("sigma_R")
    nodes, sigmas = SIZE_NODES[component.family]
    x, w = _gaussian_grid(radius, radius * spread, nsig=sigmas, n=nodes, clip_min=0.0)
    x = np.asarray(x, float)
    w = np.asarray(w, float) / np.maximum(x, 1e-12) ** NUMBER_POWER[component.family]
    if NUMBER_POWER[component.family] and x.size > 1 and (
            not np.isfinite(w).all() or int(np.argmax(w)) == 0 or w[0] >= TAIL_SHARE * w.max()):
        return None
    w = w / w.sum()
    mean = float(np.sum(w * x))
    return mean, float(np.sqrt(np.sum(w * (x - mean) ** 2)) / mean) if mean > 0 else 0.0


def sigma_nn_over_d(component: Component) -> Optional[float]:
    """σ_nn/D, the relative standard deviation of the neighbour distance (Hosemann; fit1d's w/D), from Fitting's
    σ_D: its paracrystal damps with exp(−π q² (σ_D D)²), Hosemann's with exp(−q² σ_nn² / 2)."""
    if not component.structure:
        return None
    return math.sqrt(2.0 * math.pi) * component.value("sigma_D")


COUPLINGS = ("lma", "da")


def _coherent_form(component: Component, q: np.ndarray, paths: Optional[tuple]) -> np.ndarray:
    radius, spread = component.value("R"), component.value("sigma_R")
    if component.family == "sphere":
        return sphere_coherent_pd(q, radius, radius * spread, n_samples=25, nsig=4.0, paths=paths)
    if component.family == "cylinder":
        height, height_spread = component.value("h"), component.value("sigma_h")
        return cylinder_coherent_pd(q, radius, radius * spread, height, height * height_spread, n_R=13, n_h=13,
                                    nsig=4.0, n_orient=24)
    return vertical_cylinder_coherent_pd(q, radius, spread, n_samples=25, nsig=3.0)


def component_form(component: Component, q: np.ndarray, cache: Optional[dict] = None,
                   paths: Optional[tuple] = None) -> np.ndarray:
    """P(q)·S(q) of one component with scale 1 (q in nm⁻¹). ``cache``: reuse the parts that did not
    change during one fit (a random cylinder's radius and height averages are separate). ``paths``:
    :func:`grazing` of the model (used by spheres only)."""
    q = np.asarray(q, dtype=float)
    radius, spread = component.value("R"), component.value("sigma_R")
    if component.family == "sphere" and paths is not None:
        form = _cached(cache, ("sphere2", radius, spread, *paths),
                       lambda: sphere_two_path_pd(q, radius, radius * spread, *paths, n_samples=25, nsig=4.0))
    elif component.family == "sphere":
        form = _cached(cache, ("sphere", radius, spread),
                       lambda: sphere_form_factor_pd(q, radius, radius * spread, n_samples=25, nsig=4.0))
    elif component.family == "cylinder":
        height, height_spread = component.value("h"), component.value("sigma_h")
        radial = _cached(cache, ("radial", radius, spread),
                         lambda: cylinder_radial_average(q, radius, radius * spread, n_R=13, nsig=4.0, n_orient=24))
        axial = _cached(cache, ("axial", height, height_spread),
                        lambda: cylinder_axial_average(q, height, height * height_spread, n_h=13, nsig=4.0, n_orient=24))
        form = _cached(cache, ("cylinder", radius, spread, height, height_spread),
                       lambda: cylinder_orientation_average(radial, axial, 24))
    else:
        form = _cached(cache, ("vertical", radius, spread),
                       lambda: vertical_cylinder_form_factor_pd(q, radius, spread, n_samples=25, nsig=3.0))
    if component.structure:
        distance, disorder = component.value("D"), component.value("sigma_D")
        if distance > 0 and disorder > 0:
            structure = _cached(cache, ("S", distance, disorder),
                                lambda: structure_factor_1d(q, distance, distance * disorder))
            if component.coupling == "da":
                key = ("coherent", component.family, radius, spread, *(
                    (component.value("h"), component.value("sigma_h")) if component.family == "cylinder" else ()),
                       *((paths or ()) if component.family == "sphere" else ()))
                coherent = _cached(cache, key, lambda: _coherent_form(component, q, paths
                                                                      if component.family == "sphere" else None))
                form = form + coherent * (structure - 1.0)
            else:
                form = form * structure
    return np.asarray(form, dtype=float)


def resolution_shape(model: FitModel, q: np.ndarray) -> np.ndarray:
    return np.asarray(_resolution_peak(q, model.globals["res_width"].value, model.globals["res_exponent"].value), float)


def evaluate(model: FitModel, q: np.ndarray, *, parts: bool = False, cache: Optional[dict] = None):
    """The model on ``q`` (nm⁻¹); with ``parts``, also each term: ``(total, {name: curve})``."""
    q = np.abs(np.asarray(q, dtype=float))
    g = {key: float(parameter.value) for key, parameter in model.globals.items()}
    terms, paths = {}, grazing(model)
    for index, component in enumerate(model.components):
        terms[f"{index + 1}: {FAMILIES[component.family][0]}"] = (
            g["k"] * component.value("Int") * component_form(component, q, cache, paths))
    if g["res_amplitude"] != 0:
        terms["resolution peak"] = g["k"] * g["res_amplitude"] * resolution_shape(model, q)
    total = g["background"] + (np.sum(list(terms.values()), axis=0) if terms else np.zeros_like(q))
    if parts:
        return total, {**terms, "constant background": np.full_like(q, g["background"])}
    return total


# -- the manual parameter store (spreads in nm) and quick-fit solutions ------------------


def to_manual(model: FitModel) -> CandidateParameterMapping:
    components = []
    for component in model.components:
        radius = component.value("R")
        values = {"intensity": component.value("Int"), "radius": radius}
        values["sigma_radius"] = component.value("sigma_R") * (1.0 if component.family == "vertical_cylinder" else radius)
        if component.family == "cylinder":
            values.update(height=component.value("h"), sigma_height=component.value("h") * component.value("sigma_h"))
        distance = component.value("D") if component.structure else 0.0
        values.update(diameter=distance, sigma_diameter=distance * component.value("sigma_D") if component.structure else 0.0)
        components.append(CandidateComponentParameters(MANUAL_NAMES[component.family], 1.0, values))
    g = model.globals
    return CandidateParameterMapping(tuple(components), {
        "background": g["background"].value, "sigma_res": g["res_width"].value, "nu_res": g["res_exponent"].value,
        "int_res": g["res_amplitude"].value, "k_value": g["k"].value,
    })


def from_manual(mapping: CandidateParameterMapping, base: Optional[FitModel] = None) -> FitModel:
    """A model from manual-store values (sizes and spreads converted); free/fixed and ranges from ``base``."""
    families = {name: key for key, name in MANUAL_NAMES.items()}
    components = []
    for index, source in enumerate(mapping.components):
        family = families.get(source.shape)
        if family is None:
            raise ValueError(f"Fitting has no family {source.shape!r}")
        values = source.parameters
        radius = float(values.get("radius", INFO["R"].value))
        distance = float(values.get("diameter", 0.0) or 0.0)
        structure = distance > 0 and float(values.get("sigma_diameter", 0.0) or 0.0) > 0
        component = new_component(family, radius=radius, distance=distance if structure else None, structure=structure)
        spread = float(values.get("sigma_radius", 0.0) or 0.0)
        relative = {"Int": float(values.get("intensity", 1.0)), "R": radius,
                    "sigma_R": spread if family == "vertical_cylinder" else spread / radius if radius else 0.0}
        if family == "cylinder":
            height = float(values.get("height", INFO["h"].value))
            relative.update(h=height, sigma_h=float(values.get("sigma_height", 0.0) or 0.0) / height if height else 0.0)
        if structure:
            relative.update(D=distance, sigma_D=float(values["sigma_diameter"]) / distance)
        previous = base.components[index] if base is not None and index < len(base.components) else None
        params = dict(component.params)
        for key, value in relative.items():
            template = previous.params.get(key) if previous is not None and previous.family == family else None
            params[key] = replace(template or params[key], value=float(value))
            params[key] = _widened(params[key])
        components.append(replace(component, params=params))
    globals_ = dict((base or FitModel()).globals)
    for key, source in (("background", "background"), ("res_width", "sigma_res"), ("res_exponent", "nu_res"),
                        ("res_amplitude", "int_res"), ("k", "k_value")):
        value = mapping.global_parameters.get(source)
        if value is None or (key in ("res_width", "res_exponent") and value <= 0):
            continue  # no resolution peak in the source: keep the peak's shape, its A is 0
        globals_[key] = _widened(replace(globals_[key], value=float(value)))
    return FitModel(tuple(components), globals_)


def _widened(parameter: Parameter) -> Parameter:
    """A value outside its range widens the range (a loaded value is never clipped)."""
    lower, upper = parameter.lower, parameter.upper
    if parameter.value < lower:
        lower = parameter.value if parameter.value <= 0 else parameter.value / 2
    if parameter.value > upper:
        upper = parameter.value * 2 if parameter.value > 0 else parameter.value
    return replace(parameter, lower=lower, upper=upper)


# -- JSON -----------------------------------------------------------------------------


def model_to_dict(model: FitModel) -> dict:
    def parameter(p: Parameter) -> dict:
        return {"value": p.value, "free": p.free, "lower": p.lower if math.isfinite(p.lower) else None,
                "upper": p.upper if math.isfinite(p.upper) else None}

    return {
        "schema": "gimap_fit_model_v1", "q_unit": "nm^-1", "spreads": "relative",
        "components": [{"family": c.family, "structure": c.structure, "coupling": c.coupling,
                        "params": {key: parameter(c.params[key]) for key in c.keys()}} for c in model.components],
        "globals": {key: parameter(model.globals[key]) for key in GLOBALS},
    }


def model_from_dict(data: Mapping) -> FitModel:
    if data.get("schema") != "gimap_fit_model_v1":
        raise ValueError("Not a GIMaP fit model (schema gimap_fit_model_v1)")

    def parameter(key: str, raw: Mapping) -> Parameter:
        info = INFO[key]
        lower, upper = raw.get("lower"), raw.get("upper")
        return Parameter(float(raw.get("value", info.value)), bool(raw.get("free", info.free)),
                         info.lower if lower is None else float(lower), INF if upper is None else float(upper))

    components = []
    for raw in data.get("components") or ():
        component = new_component(str(raw["family"]), structure=bool(raw.get("structure", True)))
        coupling = str(raw.get("coupling") or "lma")
        component = replace(component, coupling=coupling if coupling in COUPLINGS else "lma")
        params = dict(component.params)
        for key, value in (raw.get("params") or {}).items():
            if key in params:
                params[key] = parameter(key, value)
        components.append(replace(component, params=params))
    globals_ = {key: Parameter.default(key) for key in GLOBALS}
    for key, value in (data.get("globals") or {}).items():
        if key in globals_:
            globals_[key] = parameter(key, value)
    return FitModel(tuple(components), globals_)


__all__ = [
    "BASE_GLOBALS", "COUPLINGS", "FAMILIES", "GLOBALS", "GRAZING", "INFO", "LINEAR", "Component", "FitModel", "Parameter",
    "ParameterInfo", "component_form", "evaluate", "grazing", "number_size", "qz_plus_r", "sigma_nn_over_d", "from_manual", "model_from_dict", "model_to_dict", "new_component",
    "resolution_shape", "to_manual",
]
