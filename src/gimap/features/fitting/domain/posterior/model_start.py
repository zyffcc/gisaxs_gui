"""Build the bounded Fitting model at a posterior proposal, with the cut's geometry and scales."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import fit_engine
from .. import fit_model as fm
from . import encoding as E
from .infer import TAIL_WIDTH_MAX, Cut
from .substrate_level import ordered_model

KEYS = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn")
GLOBAL_KEYS = {"tail_width": "res_width", "tail_exponent": "res_exponent", "q_resolution": "q_resolution"}


def _clip(parameter: fm.Parameter, value: float) -> fm.Parameter:
    lower, upper = parameter.lower, parameter.upper
    if value < lower:
        lower = value
    if value > upper:
        upper = value
    return fm.Parameter(float(value), parameter.free, lower, upper)


SPREAD_BOX = (0.02, 0.5)
"""The prior's range of σR/R, σh/h and σnn/D (``encoding.bounds``)."""
GROWTH = 1.5
"""Least squares may take a population's radius this factor beyond the start; D/2R and h/2R keep the prior's ranges
(1–20, 0.1–10) of radii that far."""


def _box(params: dict, family: str, data: Cut) -> dict:
    """The parameters of one population held to the box of the network's prior: a refinement does not leave the
    model the posterior is of (on BornAgain's GALAXI cut least squares otherwise reached σR/R = 1 and D = 490 R)."""
    out = dict(params)
    q_max, q_min = data.q_max, data.q_max / data.ratio
    radius = float(out["R"].value)
    tops = {"R": min(5.0 / q_min, E.CYLINDER_RADIUS / q_max) if family == "cylinder" else 5.0 / q_min}
    boxes = {"R": (0.5 / q_max, tops["R"]),
             "sigma_R": SPREAD_BOX, "sigma_h": SPREAD_BOX, "sigma_nn": SPREAD_BOX,
             "D": (2.0 * radius / GROWTH, 40.0 * radius * GROWTH),
             "h": (0.2 * radius / GROWTH, min(20.0 * radius * GROWTH, E.CYLINDER_HEIGHT / q_max))}
    for key, (low, high) in boxes.items():
        if key not in out:
            continue
        parameter = out[key]
        low, high = max(parameter.lower, low), min(parameter.upper, high)
        value = min(max(float(parameter.value), low), high)
        out[key] = fm.Parameter(value, parameter.free, min(low, value), max(high, value))
    return out


def fit_model(values, composition: int, data: Cut, law: str = "number_gauss", coupling: str = "lma",
              grazing: Optional[dict] = None) -> fm.FitModel:
    """Fitting's model of physical values (DIMS,) of a composition, scales solved for the cut, its shape values held
    to the prior's box (``_box``). ``grazing``: the values of ``fit_model.GRAZING`` to keep (the cut's angles and λ of
    the person's model); else angles at λ = 0.1 nm that give the cut's q₊ and q₋ (or the effective interface). On a
    cut with two paths (``Cut.reflected``) the effective reflection coefficient is the values' r, free (|r| ≤ 1, its
    phase within ±π of the start); else the person's r is kept."""
    components = []
    for slot, (family, structure) in enumerate(E.composition_types(composition)):
        j = slot * E.PER_SLOT
        height = data.upright_height(slot) if family == "vertical_cylinder" else None
        size_law, method = height.settings(law, coupling) if height else (law, coupling)
        component = fm.new_component(family, radius=values[j], distance=values[j + 4] if structure else None,
                                     structure=structure, law=size_law, heights=height is not None)
        params = dict(component.params)
        for offset, key in enumerate(KEYS):
            if key in params and np.isfinite(values[j + offset]):
                params[key] = _clip(params[key], values[j + offset])
        params = _box(params, family, data)
        if height is not None:
            for key, value in (("h", height.height), ("sigma_h", height.spread)):
                p = params[key]
                params[key] = fm.Parameter(value, False, min(p.lower, value), max(p.upper, value))
        components.append(fm.Component(family, params, structure, method, size_law, heights=height is not None))
    model = fm.FitModel(tuple(components), substrate=data.substrate)
    if data.upright_heights:
        model = ordered_model(model)
    g = E.GLOBAL.start
    model = model.with_parameter(("globals", "res_amplitude"), value=1.0, free=True)
    for offset, (name, key) in enumerate(GLOBAL_KEYS.items()):
        value = values[g + offset]
        if key == "res_width":  # the tail stays a tail (``infer.TAIL_WIDTH_MAX``)
            top = TAIL_WIDTH_MAX * data.q_max / data.ratio
            value = min(value, top)
            model = model.with_parameter(("globals", key), upper=top, lower=min(model.globals[key].lower, value))
        elif key == "res_exponent":  # Lorentzian wings (``encoding.TAIL_EXPONENT``)
            value = min(max(value, E.TAIL_EXPONENT[0]), E.TAIL_EXPONENT[1])
            model = model.with_parameter(("globals", key), lower=E.TAIL_EXPONENT[0], upper=E.TAIL_EXPONENT[1])
        parameter = _clip(model.globals[key], value)
        model = model.with_parameter(("globals", key), value=parameter.value, lower=parameter.lower,
                                     upper=parameter.upper, free=True)
    if data.substrate is not None:
        geometry = data.geometry()
        grazing = dict(alpha_i=geometry.alpha_i_deg, alpha_f=geometry.alpha_f_deg, wavelength=geometry.wavelength_nm)
    if grazing:
        for key, value in grazing.items():
            if not (data.reflected and key in REFLECTION_KEYS):
                model = model.with_parameter(("globals", key), value=float(value))
    elif data.qz > 0:
        alpha_i, alpha_f = data.angles()
        for key, value in (("alpha_i", alpha_i), ("alpha_f", alpha_f), ("wavelength", 0.1)):
            model = model.with_parameter(("globals", key), value=value)
    if data.reflected:
        r = complex(*(float(x) if np.isfinite(x) else 0.0 for x in values[g + 4:g + 6]))
        size, phase = min(abs(r), 1.0), math.atan2(r.imag, r.real)
        model = model.with_parameter(("globals", "refl_amp"), value=size, free=True, lower=0.0, upper=1.0)
        model = model.with_parameter(("globals", "refl_phase"), value=phase, free=True, lower=phase - math.pi,
                                     upper=phase + math.pi)
    return fit_engine.fit_scales(model, fit_engine.FitData.prepare(data.q, data.intensity, data.sigma))


REFLECTION_KEYS = ("refl_amp", "refl_phase")
