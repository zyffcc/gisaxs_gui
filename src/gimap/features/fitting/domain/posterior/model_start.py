"""Build the bounded Fitting model at a posterior proposal, with the cut's geometry and scales."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import fit_engine
from .. import fit_model as fm
from ..cut_model.geometry import Geometry
from . import encoding as E
from .infer import CORE_RANGE, Cut
from .substrate_level import ordered_model

KEYS = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn")
GLOBAL_KEYS = {"tail_width": "res_width", "diffuse_width": "diffuse_width", "diffuse_hurst": "diffuse_hurst",
               "q_resolution": "q_resolution"}
"""The Fitting keys of the cut's values of the network, in their order (``encoding.TAIL_DIM`` …)."""


def _cut_geometry(data, grazing):
    """Keep the measured angles and wavelength when they describe the vertical transfers used in the likelihood."""
    geometry = data.geometry()
    if data.level in (2, 3) and grazing:
        optics = [grazing.get(key) for key in ("wavelength", "alpha_i", "alpha_f")]
        if any(value is None for value in optics):
            return geometry
        measured = Geometry(geometry.mode, *map(float, optics))
        if math.isclose(measured.qz_born, data.qz, rel_tol=1e-10) and (
                not data.reflected or math.isclose(measured.k_i - measured.k_f, data.qz_minus, abs_tol=1e-12)):
            return measured
    return geometry


def _clip(parameter: fm.Parameter, value: float) -> fm.Parameter:
    lower, upper = parameter.lower, parameter.upper
    if value < lower:
        lower = value
    if value > upper:
        upper = value
    return fm.Parameter(float(value), parameter.free, lower, upper)


def _bounded(parameter, lower, upper):
    return fm.Parameter(float(np.clip(parameter.value, lower, upper)), parameter.free, float(lower), float(upper))


def _box(params, low, high, data, cylinder_ahead):
    """Absolute physical bounds of the encoded prior. The radius ratios are imposed jointly in ordered_model."""
    radius_min = low[0] / data.q_max
    radius_max = min(high[0], E.CYLINDER_RADIUS) / data.q_max if cylinder_ahead else high[0] / data.q_max
    bounds = {"R": (radius_min, radius_max), "sigma_R": (low[1], high[1]), "sigma_h": (low[3], high[3]),
              "sigma_nn": (low[5], high[5]), "D": (2 * radius_min * low[4], 2 * radius_max * high[4]),
              "h": (2 * radius_min * low[2], min(2 * radius_max * high[2], E.CYLINDER_HEIGHT / data.q_max))}
    return {key: _bounded(p, *bounds[key]) if key in bounds else p for key, p in params.items()}


def fit_model(values, composition: int, data: Cut, law: str = "number_gauss", coupling: str = "lma",
              grazing: Optional[dict] = None) -> fm.FitModel:
    """Fitting's model of physical values (DIMS,) of a composition, scales solved for the cut, its shape values held
    to the prior's box (``_box``). ``grazing``: the values of ``fit_model.GRAZING`` to keep (the cut's angles and λ of
    the person's model); else angles at λ = 0.1 nm that give the cut's q₊ and q₋ (or the effective interface). On a
    cut with two paths (``Cut.reflected``) the effective reflection coefficient is the values' r, free (|r| ≤ 1, its
    phase within ±π of the start) for compositions containing spheres."""
    components = []
    types = E.composition_types(composition)
    low, high = E.bounds([data.ratio])
    low, high = [np.where(E.LOGARITHMIC, np.exp(row[0]), row[0]) for row in (low, high)]
    for slot, (family, structure) in enumerate(types):
        j = slot * E.PER_SLOT
        height = data.upright_height(slot) if family == "vertical_cylinder" else None
        size_law, method = height.settings(law, coupling) if height else (law, coupling)
        component = fm.new_component(family, radius=values[j], distance=values[j + 4] if structure else None,
                                     structure=structure, law=size_law, heights=height is not None)
        params = dict(component.params)
        for offset, key in enumerate(KEYS):
            if key in params and np.isfinite(values[j + offset]):
                params[key] = _clip(params[key], values[j + offset])
        params = _box(params, low, high, data, any(f == "cylinder" for f, _ in types[slot:]))
        if height is not None:
            for key, value in (("h", height.height), ("sigma_h", height.spread)):
                p = params[key]
                params[key] = fm.Parameter(value, False, min(p.lower, value), max(p.upper, value))
        components.append(fm.Component(family, params, structure, method, size_law, heights=height is not None))
    model = ordered_model(fm.FitModel(tuple(components), substrate=data.substrate, predict_level=data.level),
                          low, high)
    model = model.with_parameter(("globals", "res_amplitude"), value=1.0, free=True)
    model = model.with_parameter(("globals", "res_exponent"), value=2.0, free=False)  # the Lorentzian wings
    model = model.with_parameter(("globals", "diffuse_amplitude"), value=1.0, free=True)
    model = model.with_parameter(("globals", "core_amplitude"), value=0.0, free=True,  # the core of the beam
                                 upper=CORE_RANGE * float(np.max(np.abs(data.intensity))))
    for index, key in enumerate(GLOBAL_KEYS.values(), E.GLOBAL.start):
        scale = 1.0 if key == "diffuse_hurst" else data.q_max
        lower, upper = low[index] * scale, high[index] * scale
        model = model.with_parameter(("globals", key), value=float(np.clip(values[index], lower, upper)),
                                     lower=lower, upper=upper, free=True)
    geometry = _cut_geometry(data, grazing)
    for key, value in (("alpha_i", geometry.alpha_i_deg), ("alpha_f", geometry.alpha_f_deg),
                       ("wavelength", geometry.wavelength_nm)):
        model = model.with_parameter(("globals", key), value=value)
    if data.reflected and any(family == "sphere" for family, _ in types):
        r = complex(*(float(x) if np.isfinite(x) else 0.0 for x in values[E.REFLECTION]))
        size, phase = min(abs(r), 1.0), math.atan2(r.imag, r.real)
        model = model.with_parameter(("globals", "refl_amp"), value=size, free=True, lower=0.0, upper=1.0)
        model = model.with_parameter(("globals", "refl_phase"), value=phase, free=True, lower=phase - math.pi,
                                     upper=phase + math.pi)
    return fit_engine.fit_scales(model, fit_engine.FitData.prepare(data.q, data.intensity, data.sigma))


REFLECTION_KEYS = ("refl_amp", "refl_phase")
