"""The model of one curve in Fitting: its parameters (value, free or fixed, range) and its intensity.

    I(q) = BG + k · [ Σ_i Int_i · T_i(q)  +  A / (1 + (|q| / w)^ν)  +  A_d (1 + (|q| / w_d)²)^−(1+H) ]

The intensity is GIMaP's one forward model of a cut, ``cut_model`` (Physics and Methods, chapter 9):

* T_i: a particle family — sphere, random cylinder, vertical cylinder — averaged over its size distribution, each
  normalised to T(0) = 1, so Int_i is the forward intensity of component i whatever its family. The sizes follow the
  component's ``law`` (Gaussian by number by default). New log-normal populations use medians and log widths;
  the Schulz law uses means and relative widths. Saved models retain their parameter conventions;
* with ``structure``, the interference function of ``interference``: the 1D paracrystal of mean distance D and
  σnn/D, the relative standard deviation of the neighbour distance (Hosemann; the default), or, for manual fits only
  (``model_library``), the 2D hexagonal paracrystal (D, σnn/D) or Percus–Yevick hard spheres (R_HS, φ); in the LMA
  (P·S) or the DA (P + |<F>|²(S − 1));
* the families: sphere, random cylinder, vertical cylinder (the proposal network's, ``NETWORK_FAMILIES``) and, for
  manual fits, the core–shell sphere (``model_library.core_shell``);
* links between parameters (``Parameter.tie``) and inequalities (``FitModel.inequalities``): ``fit_links``;
* BG a constant background, A/(1+(|q|/w)^ν) the beam tail at q = 0 (key ``resolution peak``; Predict parameters
  takes ν = 2, the Lorentzian wings of the resolution function), A_d (1 + (|q|/w_d)²)^−(1+H) the diffuse scattering
  of a rough interface with the K-correlation function, w_d = 1/ξ (key ``diffuse scattering``; off in a new model:
  A_d = 0, fixed), k an overall factor; with σ_q > 0 the particle terms are convolved with a Gaussian q resolution.

Spheres of a horizontal GISAXS cut: with αi = αf = 0 (the default) the sphere is evaluated at q_z = 0, the
effective interface approximation. With the angles and λ of the cut it is the sum of the direct and the
exit-reflected path, A = f(√(q² + q₊²) R) e^{-i q₊ R} + r f(√(q² + q₋²) R) e^{-i q₋ R}, r = |r| e^{i arg r} free;
with a known ``substrate`` (``substrates``) as well, the four-term DWBA with Fresnel reflection at a bare
interface or Parratt reflection from its layers.
A model with a ``cut`` is a vertical cut of a fit of several curves (``vertical_cut``, ``joint_fit``).

q in nm⁻¹, sizes in nm. Model files of earlier versions (``gimap_fit_model_v1``: σD/D = (σnn/D)/√(2π), intensity-weighted
spheres and random cylinders, a vertical cylinder scaled by 10⁻⁶⟨R⁴⟩/4) are converted when read
(``cut_model.conventions``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional

import numpy as np

from .candidates import CandidateComponentParameters, CandidateParameterMapping
from .cut_model import Geometry, size_averages
from .cut_model import conventions, resolution, sizes
from .cut_model import manual_sizes
from .cut_model.geometry import from_angles
from .cut_model.model import diffuse, tail
from . import fit_resolution
from .cut_model.shapes import RADIUS_POWER
from .cut_model.structure import paracrystal
from . import model_library
from .substrates import Substrate
from .vertical_cut import VerticalCut
from .vertical_cut import component_term as vertical_term

CACHE_SIZE = 96

FAMILIES = {
    "sphere": ("Sphere", ("R", "sigma_R")),
    "cylinder": ("Random cylinder", ("R", "sigma_R", "h", "sigma_h")),
    "vertical_cylinder": ("Vertical cylinder", ("R", "sigma_R")),
    "core_shell_sphere": ("Core–shell sphere", ("R", "sigma_R", "t_shell", "core_contrast")),
}
NETWORK_FAMILIES = ("sphere", "cylinder", "vertical_cylinder")
"""The families of Predict parameters; the others are for Refine and Search (``model_library``)."""
MANUAL_NAMES = {"sphere": "Sphere", "cylinder": "Cylinder", "vertical_cylinder": "Vertical Cylinder"}
STRUCTURE = ("D", "sigma_nn")
HEIGHTS = ("h", "sigma_h")
"""The height of upright cylinders, a parameter only where a vertical cut sees it (``Component.heights``)."""
INTERFERENCES = {
    "paracrystal": ("1D paracrystal", STRUCTURE),
    "hex2d": ("2D hexagonal paracrystal", STRUCTURE),
    "hard_sphere": ("Hard spheres (Percus–Yevick)", ("R_hs", "phi")),
}
"""The interference functions of a component with ``structure`` (key: label, parameters); the first is the default
and the only one of Predict parameters."""
DEFAULT_INTERFERENCE = "paracrystal"
DIFFUSE = ("diffuse_amplitude", "diffuse_width", "diffuse_hurst")
"""The diffuse scattering of a rough interface: A_d, w_d = 1/ξ and the Hurst exponent H."""
BASE_GLOBALS = ("background", "res_amplitude", "res_width", "res_exponent") + DIFFUSE + ("k",)
RESOLUTION = ("q_resolution",)
"""The Gaussian q resolution (1σ) of the particle terms; 0, the default, is none."""
GRAZING = ("alpha_i", "alpha_f", "wavelength", "refl_amp", "refl_phase")
"""Geometry of the cut and the effective reflection coefficient r of the two-path sphere (fixed by default)."""
GLOBALS = BASE_GLOBALS + RESOLUTION + GRAZING
LINEAR = frozenset({"Int", "background", "res_amplitude", "diffuse_amplitude"})
"""Scales: with the others fixed, the intensity is linear in them (solved exactly while fitting)."""
INF = math.inf
SIZE_LAWS = manual_sizes.LABELS
"""The size laws of a component (``cut_model.sizes``) and their labels; the first is the default."""
DEFAULT_LAW = "gaussian"
LEGACY_LAWS = {"sphere": "intensity_gauss", "cylinder": "intensity_gauss", "vertical_cylinder": "number_gauss"}
"""The sizes of earlier Fitting, per family (what a ``gimap_fit_model_v1`` file means)."""
SCHEMA, LEGACY_SCHEMA = "gimap_fit_model_v2", "gimap_fit_model_v1"


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
    "Int": ParameterInfo("Scale", "", 1.0, 0.0, INF, tip="Intensity of this component at q = 0 (its form factor is 1 there)"),
    "R": ParameterInfo("R", "nm", 5.0, 0.3, 500.0, tip="Mean radius (of the size distribution chosen in Sizes)"),
    "sigma_R": ParameterInfo("σR/R", "", 0.2, 0.01, 1.0, tip="Relative spread of the radius (of the size distribution chosen in Sizes)"),
    "h": ParameterInfo("h", "nm", 10.0, 0.5, 1000.0, tip="Mean height (length) of the cylinders"),
    "sigma_h": ParameterInfo("σh/h", "", 0.2, 0.01, 1.0, tip="Relative spread of the height"),
    "D": ParameterInfo("D", "nm", 30.0, 1.0, 3000.0, tip="Mean distance between neighbours (paracrystal)"),
    "sigma_nn": ParameterInfo("σnn/D", "", 0.25, 0.01, 1.0,
                              tip="Relative standard deviation of the distance between neighbours (Hosemann paracrystal)"),
    "t_shell": ParameterInfo("t", "nm", 2.0, 0.0, 200.0, tip="Thickness of the shell: the outer radius is R + t"),
    "core_contrast": ParameterInfo("ρc/ρs", "", 2.0, 0.0, 100.0,
                                   tip="Scattering contrast of the core relative to that of the shell (0: a hollow "
                                       "shell, 1: a homogeneous sphere of radius R + t)"),
    "R_hs": ParameterInfo("R_HS", "nm", 6.0, 0.3, 1000.0,
                          tip="Hard-sphere radius: half the closest distance between the centres (Percus–Yevick)"),
    "phi": ParameterInfo("φ", "", 0.2, 0.0, 0.55, tip="Volume fraction of the hard spheres (Percus–Yevick)"),
    "background": ParameterInfo("Background", "", 0.0, 0.0, INF, tip="Constant added to the whole curve"),
    "res_amplitude": ParameterInfo("Tail A", "", 0.0, 0.0, INF,
                                   tip="Height at q = 0 of the beam tail A / (1 + (|q|/w)^ν): the empirical tail of "
                                       "the direct and reflected beam"),
    "res_width": ParameterInfo("Tail w", "nm⁻¹", 0.02, 0.001, 0.1, tip="Width w of the beam tail"),
    "res_exponent": ParameterInfo("Tail ν", "", 3.0, 1.0, 20.0, tip="Exponent ν of the beam tail (how fast it falls)"),
    "diffuse_amplitude": ParameterInfo("Diffuse A", "", 0.0, 0.0, INF, free=False,
                                       tip="Height at q = 0 of the diffuse scattering of a rough interface "
                                           "A_d (1 + (|q|/w)²)^−(1+H); 0: no diffuse term"),
    "diffuse_width": ParameterInfo("Diffuse w", "nm⁻¹", 0.02, 1e-4, 0.1, free=False,
                                   tip="Width w = 1/ξ of the diffuse term, ξ the correlation length of the roughness"),
    "diffuse_hurst": ParameterInfo("Diffuse H", "", 0.5, 0.0, 1.0, free=False,
                                   tip="Hurst exponent H of the roughness: the diffuse term falls as q^−2(1+H) "
                                       "(H = 1: also the Debye–Bueche form of large inhomogeneities)"),
    "k": ParameterInfo("Factor k", "", 1.0, 1e-6, 1e6, free=False,
                       tip="Overall factor of everything but the background (usually fixed at 1)"),
    "q_resolution": ParameterInfo("σ_q", "nm⁻¹", 0.0, 0.0, 0.1, free=False,
                                  tip="Gaussian q resolution (1σ) applied to the particle terms; 0: none"),
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
    tie: Optional[str] = None
    """A link: the value is this expression of other parameters (``fit_links``), not fitted."""

    @classmethod
    def default(cls, key: str, value: Optional[float] = None) -> "Parameter":
        info = INFO[key]
        return cls(info.value if value is None else float(value), info.free, info.lower, info.upper)


def parameter_info(component, key: str) -> ParameterInfo:
    info = INFO[key]
    if component is not None and component.law == "lognormal_median" and key in manual_sizes.LOGNORMAL_PARAMETERS:
        label, tip = manual_sizes.LOGNORMAL_PARAMETERS[key]
        return replace(info, label=label, tip=tip)
    return info


@dataclass(frozen=True)
class Component:
    family: str
    params: Mapping[str, Parameter]
    structure: bool = True
    coupling: str = "lma"
    """How the paracrystal enters (``COUPLINGS``): ``lma``, P(q)·S(q) (local monodisperse approximation), or
    ``da``, P(q) + |<F>|²(S(q) − 1) (decoupling approximation: the sizes are uncorrelated with the positions)."""
    law: str = DEFAULT_LAW
    """The size distribution (``SIZE_LAWS``)."""
    interference: str = DEFAULT_INTERFERENCE
    """The interference function with ``structure`` (``INTERFERENCES``)."""
    heights: bool = False
    """Upright cylinders with their height h and σh/h as parameters (a vertical cut sees them, ``vertical_cut``)."""

    def keys(self) -> tuple[str, ...]:
        shape = FAMILIES[self.family][1] + (HEIGHTS if self.heights and self.family == "vertical_cylinder" else ())
        return ("Int", *shape, *(INTERFERENCES[self.interference][1] if self.structure else ()))

    def value(self, key: str) -> float:
        return float(self.params[key].value)


@dataclass(frozen=True)
class FitModel:
    components: tuple[Component, ...] = ()
    globals: Mapping[str, Parameter] = field(default_factory=lambda: {
        key: Parameter.default(key) for key in GLOBALS
    })
    contact: bool = True
    """Particles do not overlap: D ≥ 2R (2(R + t) for core–shell spheres) of a paracrystal, R_HS ≥ R (+ t) of hard
    spheres, while both are fitted (``fit_engine``)."""
    inequalities: tuple = ()
    """``fit_links.Inequality`` constraints of the fit (``D_1 >= 2*R_1 + 3``)."""
    substrate: Optional[Substrate] = None
    """A known substrate: with αi and αf, spheres in the four-term DWBA on it (``grazing``)."""
    cut: Optional[VerticalCut] = None
    """A vertical cut (fits of several curves): the curve is I(q_z) at the cut's in-plane q."""
    resolution_mode: str = "constant"
    weighting: str = "file"
    predict_level: Optional[int] = None
    """Forward model of a Predict solution, including its resolution quadrature and the level-2 Born transfer."""

    def __post_init__(self):
        if self.resolution_mode not in ("constant", "file"):
            raise ValueError(f"Unknown resolution mode {self.resolution_mode!r}")
        if self.weighting not in ("file", "relative", "poisson", "none"):
            raise ValueError(f"Unknown fit weighting {self.weighting!r}")

    # -- addressing: ("globals", key) or (component index, key) ----------------------

    def parameters(self) -> list[tuple[tuple, Parameter]]:
        """Every parameter in use, components first."""
        items = [((index, key), component.params[key])
                 for index, component in enumerate(self.components) for key in component.keys()]
        return items + [(("globals", key), self.globals[key]) for key in GLOBALS
                        if key != "q_resolution" or self.resolution_mode == "constant"]

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
                  structure: bool = True, law: str = DEFAULT_LAW, interference: str = DEFAULT_INTERFERENCE,
                  heights: bool = False) -> Component:
    if family not in FAMILIES:
        raise ValueError(f"Unknown particle family {family!r}")
    if law not in SIZE_LAWS:
        raise ValueError(f"Unknown size law {law!r}")
    params = {key: Parameter.default(key) for key in ("Int", *FAMILIES[family][1], *STRUCTURE)}
    if heights and family == "vertical_cylinder":
        params.update({key: Parameter.default(key) for key in HEIGHTS})
    if radius is not None:
        params["R"] = replace(params["R"], value=float(radius))
    if distance is not None:
        params["D"] = replace(params["D"], value=float(distance))
    elif radius is not None:
        params["D"] = replace(params["D"], value=max(3.0 * float(radius), params["D"].lower))
    component = Component(family, params, structure, law=law, heights=heights and family == "vertical_cylinder")
    return with_interference(component, interference) if interference != DEFAULT_INTERFERENCE else component


def with_interference(component: Component, interference: str) -> Component:
    """``component`` with another interference function; its parameters keep their values, new ones start from the
    defaults (the hard-sphere radius from the particle's outer radius)."""
    if interference not in INTERFERENCES:
        raise ValueError(f"Unknown interference function {interference!r}")
    params = dict(component.params)
    for key in INTERFERENCES[interference][1]:
        if key not in params:
            params[key] = Parameter.default(key)
            if key == "R_hs":
                outer = component.value("R") + (component.value("t_shell") if "t_shell" in params else 0.0)
                params[key] = replace(params[key], value=max(1.1 * outer, params[key].lower))
    return replace(component, params=params, interference=interference)


def _cached(cache: Optional[dict], key: tuple, compute):
    """``compute()`` once per ``key`` while fitting one set of points (``cache`` belongs to that fit)."""
    if cache is None:
        return compute()
    if key not in cache:
        if len(cache) >= CACHE_SIZE:
            cache.clear()
        cache[key] = compute()
    return cache[key]


def grazing(model: FitModel):
    """The geometry of the cut's spheres: None for the effective interface (q_z = 0), the Born term or the two paths
    (``Geometry``); with a known substrate and both angles, the four-term DWBA on it. A vertical cut: its
    ``VerticalCut``."""
    if model.cut is not None:
        return model.cut
    g = {key: float(model.globals[key].value) for key in GRAZING}
    if model.substrate is not None and g["alpha_i"] > 0 and g["alpha_f"] > 0:
        return model.substrate.geometry(g["wavelength"], g["alpha_i"], g["alpha_f"])
    if model.predict_level == 2 and g["refl_amp"] == 0 and (g["alpha_i"] or g["alpha_f"]):
        return Geometry("born", g["wavelength"], g["alpha_i"], g["alpha_f"])
    geometry = from_angles(g["alpha_i"], g["alpha_f"], g["wavelength"], g["refl_amp"], g["refl_phase"])
    return None if geometry.mode == "eia" else geometry


def qz_plus_r(model: FitModel) -> list[tuple[int, float]]:
    """q₊·R of every sphere (component index, value), q₊ = k_i + k_f; empty without the angles of the cut."""
    geometry = grazing(model)
    if not isinstance(geometry, Geometry) or geometry.qz_born <= 0:
        return []
    return [(index, geometry.qz_born * c.value("R")) for index, c in enumerate(model.components) if c.family == "sphere"]


def _radius_power(family: str) -> int:
    return RADIUS_POWER.get(family, model_library.SHELL_POWER)


def number_size(component: Component) -> Optional[tuple[float, float]]:
    """(mean, relative standard deviation) of the radius by number, for comparison with sizes counted in
    microscopy. ``None`` when not defined: a Gaussian by intensity is G(R)/R⁶ by number, which rises towards small R
    once σR/R exceeds about 1/√24 ≈ 0.2 (1/√16 = 0.25 for cylinders); its mean would be set by the grid, not the data
    (``cut_model.sizes.size_report``)."""
    return sizes.size_report(component.law, component.value("R"), component.value("sigma_R"),
                             _radius_power(component.family))["number"]


def intensity_size(component: Component) -> tuple[float, float]:
    """(mean, relative standard deviation) of the radius weighted by V² (the scattered intensity at q = 0)."""
    return sizes.size_report(component.law, component.value("R"), component.value("sigma_R"),
                             _radius_power(component.family))["intensity"]


def sigma_nn_over_d(component: Component) -> Optional[float]:
    """σnn/D, the relative standard deviation of the neighbour distance (Hosemann; fit1d's w/D)."""
    return component.value("sigma_nn") if component.structure and "sigma_nn" in component.keys() else None


COUPLINGS = ("lma", "da")


def component_form(component: Component, q: np.ndarray, cache: Optional[dict] = None,
                   geometry: Optional[Geometry] = None) -> np.ndarray:
    """T(q) of one component with scale 1 (q in nm⁻¹). ``cache``: reuse the size average while only the spacing
    changes during one fit, and the reverse. ``geometry``: :func:`grazing` of the model."""
    q = np.asarray(q, dtype=float)
    if isinstance(geometry, VerticalCut):
        values = {key: component.value(key) for key in component.keys()}
        return vertical_term(component.family, component.law, component.interference, component.coupling,
                             component.structure, values, q, geometry, lambda key, compute: _cached(cache, key, compute))
    if component.heights or component.family not in NETWORK_FAMILIES or component.interference != DEFAULT_INTERFERENCE:
        values = {key: component.value(key) for key in component.keys()}
        return model_library.component_term(component.family, component.law, component.interference,
                                             component.coupling, component.structure, values, q,
                                             lambda key, compute: _cached(cache, key, compute), geometry or Geometry())
    geometry = geometry or Geometry()
    family, law = component.family, component.law
    radius, spread = component.value("R"), component.value("sigma_R")
    height, height_spread = ((component.value("h"), component.value("sigma_h")) if family == "cylinder" else (10.0, 0.0))
    distance, disorder = (component.value("D"), component.value("sigma_nn")) if component.structure else (0.0, 0.0)
    structured = distance > 0 and disorder > 0
    coherent = structured and component.coupling == "da"
    key = ("sizes", family, law, radius, spread, height, height_spread, geometry if family != "vertical_cylinder" else None,
           coherent)
    form, average = _cached(cache, key, lambda: size_averages(
        family, q, radius=radius, radius_spread=spread, law=law, height=height, height_spread=height_spread,
        coherent=coherent, geometry=geometry))
    if not structured:
        return np.asarray(form[0], dtype=float)
    structure = _cached(cache, ("S", distance, disorder), lambda: paracrystal(q, distance, disorder))
    out = form + average * (structure - 1.0) if coherent else form * structure
    return np.asarray(out[0], dtype=float)


def resolution_shape(model: FitModel, q: np.ndarray) -> np.ndarray:
    return np.asarray(tail(q, model.globals["res_width"].value, model.globals["res_exponent"].value), float)


def diffuse_shape(model: FitModel, q: np.ndarray) -> np.ndarray:
    return np.asarray(diffuse(q, model.globals["diffuse_width"].value, model.globals["diffuse_hurst"].value), float)


def term_shape(model: FitModel, key: str, q: np.ndarray) -> np.ndarray:
    """The unit curve of a scale of the whole curve: the beam tail (``res_amplitude``) or the diffuse term."""
    return resolution_shape(model, q) if key == "res_amplitude" else diffuse_shape(model, q)


CURVE_SCALES = ("res_amplitude", "diffuse_amplitude")
"""Scales of a term of the whole curve with a shape of its own (``term_shape``)."""


def measured_form(model, component, q, dq, cache, geometry):
    """A scale column on the measured points, including their individual resolution when selected."""
    if model.resolution_mode != "file" or model.cut is not None:
        return component_form(component, q, cache, geometry)
    key = ("file resolution", q.tobytes(), None if dq is None else np.asarray(dq).tobytes(),
           component.family, component.law, component.interference, component.coupling, component.structure,
           tuple(component.value(k) for k in component.keys() if k != "Int"), geometry)
    return _cached(cache, key, lambda: fit_resolution.pointwise_form(component, q, dq, geometry, component_form))


def particle_forms(model, q, dq, cache=None, *, scale_columns=False):
    """Particle terms for evaluation or variable projection, with the selected q resolution.

    Predict and earlier models used unsmeared columns for their scalar-resolution scale solve. Preserve that
    calculation for their original size laws and weighting. Manual laws and new weights profile the smeared forms.
    """
    geometry = grazing(model)
    sigma = model.globals["q_resolution"].value
    if model.resolution_mode == "constant" and sigma > 0 and q.size and model.cut is None:
        if model.predict_level is not None:
            if sigma > resolution.predict_sharp_limit(q):
                return fit_resolution.constant_forms(model, q, sigma, cache, geometry, component_form,
                                                     grid=resolution.predict_grid(q, sigma))
            return [component_form(c, q, cache, geometry) for c in model.components]
        legacy = scale_columns and model.weighting == "file" and all(c.law not in manual_sizes.LAWS for c in model.components)
        if not legacy:
            return fit_resolution.constant_forms(model, q, sigma, cache, geometry, component_form)
    return [measured_form(model, c, q, dq, cache, geometry) for c in model.components]


def evaluate(model: FitModel, q: np.ndarray, *, parts: bool = False, cache: Optional[dict] = None, dq=None):
    """The model on ``q`` (nm⁻¹); with ``parts``, also each term: ``(total, {name: curve})``."""
    q = np.abs(np.asarray(q, dtype=float))
    g = {key: float(parameter.value) for key, parameter in model.globals.items()}
    terms = {}
    forms = particle_forms(model, q, dq, cache)
    for index, (component, form) in enumerate(zip(model.components, forms)):
        terms[f"{index + 1}: {FAMILIES[component.family][0]}"] = g["k"] * component.value("Int") * form
    if g["res_amplitude"] != 0:
        terms["resolution peak"] = g["k"] * g["res_amplitude"] * resolution_shape(model, q)
    if g["diffuse_amplitude"] != 0:
        terms["diffuse scattering"] = g["k"] * g["diffuse_amplitude"] * diffuse_shape(model, q)
    total = g["background"] + (np.sum(list(terms.values()), axis=0) if terms else np.zeros_like(q))
    if parts:
        return total, {**terms, "constant background": np.full_like(q, g["background"])}
    return total


# -- the manual parameter store (Fitting's conventions in earlier versions) and quick-fit solutions --------------


def _legacy_scale(family: str, radius: float, spread: float) -> float:
    """Manual-store Int × this = Int here: the vertical cylinder's form factor there is 10⁻⁶⟨R⁴⟩/4 at q = 0."""
    return conventions.legacy_vertical_cylinder_scale(radius, spread) if family == "vertical_cylinder" else 1.0


def to_manual(model: FitModel) -> CandidateParameterMapping:
    """The manual store's values (σ_D in nm = D·(σnn/D)/√(2π); spreads in nm, σR/R for vertical cylinders). Its sizes
    are those of ``LEGACY_LAWS``: a component with another size law is written with its R and σR/R as they are."""
    components = []
    for component in model.components:
        if component.family not in MANUAL_NAMES or component.interference != DEFAULT_INTERFERENCE:
            raise ValueError(f"{FAMILIES[component.family][0]} has no equivalent in the parameters of earlier versions")
        radius, spread = component.value("R"), component.value("sigma_R")
        values = {"intensity": component.value("Int") / _legacy_scale(component.family, radius, spread),
                  "radius": radius,
                  "sigma_radius": spread * (1.0 if component.family == "vertical_cylinder" else radius)}
        if component.family == "cylinder":
            values.update(height=component.value("h"), sigma_height=component.value("h") * component.value("sigma_h"))
        distance = component.value("D") if component.structure else 0.0
        sigma_d = float(conventions.sigma_d_from_sigma_nn(component.value("sigma_nn"))) if component.structure else 0.0
        values.update(diameter=distance, sigma_diameter=distance * sigma_d)
        components.append(CandidateComponentParameters(MANUAL_NAMES[component.family], 1.0, values))
    g = model.globals
    return CandidateParameterMapping(tuple(components), {
        "background": g["background"].value, "sigma_res": g["res_width"].value, "nu_res": g["res_exponent"].value,
        "int_res": g["res_amplitude"].value, "k_value": g["k"].value,
    })


def from_manual(mapping: CandidateParameterMapping, base: Optional[FitModel] = None) -> FitModel:
    """A model from manual-store values (Fitting's conventions in earlier versions, ``LEGACY_LAWS``), sizes and spreads
    converted; free/fixed and ranges from ``base``."""
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
        component = new_component(family, radius=radius, distance=distance if structure else None, structure=structure,
                                   law=LEGACY_LAWS[family])
        spread = float(values.get("sigma_radius", 0.0) or 0.0)
        relative = spread if family == "vertical_cylinder" else spread / radius if radius else 0.0
        converted = {"Int": float(values.get("intensity", 1.0)) * _legacy_scale(family, radius, relative), "R": radius,
                     "sigma_R": relative}
        if family == "cylinder":
            height = float(values.get("height", INFO["h"].value))
            converted.update(h=height, sigma_h=float(values.get("sigma_height", 0.0) or 0.0) / height if height else 0.0)
        if structure:
            converted.update(D=distance, sigma_nn=float(conventions.sigma_nn_from_sigma_d(
                float(values["sigma_diameter"]) / distance)))
        previous = base.components[index] if base is not None and index < len(base.components) else None
        params = dict(component.params)
        for key, value in converted.items():
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


def model_to_dict(model: FitModel) -> dict:
    from . import fit_model_json
    return fit_model_json.model_to_dict(model)


def model_from_dict(data: Mapping) -> FitModel:
    from . import fit_model_json
    return fit_model_json.model_from_dict(data)


__all__ = [
    "BASE_GLOBALS", "COUPLINGS", "CURVE_SCALES", "DEFAULT_INTERFERENCE", "DEFAULT_LAW", "DIFFUSE", "FAMILIES",
    "INTERFERENCES", "NETWORK_FAMILIES",
    "RESOLUTION", "GLOBALS", "GRAZING", "INFO", "LEGACY_LAWS",
    "LINEAR", "SIZE_LAWS", "Component", "FitModel", "Parameter", "ParameterInfo", "component_form", "evaluate",
    "grazing", "intensity_size", "number_size", "qz_plus_r", "sigma_nn_over_d", "from_manual", "model_from_dict",
    "model_to_dict", "new_component", "parameter_info", "diffuse_shape", "resolution_shape", "term_shape", "to_manual",
    "with_interference",
]
