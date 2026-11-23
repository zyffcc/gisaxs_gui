"""Persistence of the manual fit model, including earlier models' units and size laws."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping

from . import fit_links
from .cut_model import conventions, manual_sizes
from .fit_model import (
    COUPLINGS, DEFAULT_INTERFERENCE, DEFAULT_LAW, GLOBALS, INF, INFO, INTERFERENCES, LEGACY_LAWS,
    LEGACY_SCHEMA, SCHEMA, SIZE_LAWS, Component, FitModel, Parameter, _legacy_scale, new_component, with_interference,
)
from .substrates import Substrate
from .vertical_cut import VerticalCut

MANUAL_SCHEMA = "gimap_fit_model_v3"


def model_to_dict(model: FitModel) -> dict:
    """The model as JSON. Links (``tie``), another interference function, contact off and inequality constraints
    are written only when used, so a model without them reads as in earlier versions."""
    def parameter(p: Parameter) -> dict:
        out = {"value": p.value, "free": p.free, "lower": p.lower if math.isfinite(p.lower) else None,
               "upper": p.upper if math.isfinite(p.upper) else None}
        if p.tie:
            out["tie"] = p.tie
        return out

    def component(c: Component) -> dict:
        out = {"family": c.family, "structure": c.structure, "coupling": c.coupling, "sizes": c.law,
               "params": {key: parameter(c.params[key]) for key in c.keys()}}
        if c.law == "lognormal_median":
            out["size_parameters"] = "median and standard deviation of ln(size)"
        if c.interference != DEFAULT_INTERFERENCE:
            out["interference"] = c.interference
        if c.heights:
            out["heights"] = True
        return out

    manual = (model.resolution_mode == "file" or model.weighting != "file" or
              any(c.law in manual_sizes.LAWS for c in model.components))
    out = {
        "schema": MANUAL_SCHEMA if manual else SCHEMA,
        "q_unit": "nm^-1",
        "spreads": ("relative except lognormal_median: SD(ln size)"
                    if any(c.law == "lognormal_median" for c in model.components) else "relative"),
        "spacing_disorder": "sigma_nn/D (Hosemann)",
        "components": [component(c) for c in model.components],
        "globals": {key: parameter(model.globals[key]) for key in GLOBALS},
    }
    if not model.contact:
        out["contact"] = False
    if manual:
        out["resolution_mode"] = model.resolution_mode
        out["weighting"] = model.weighting
    if model.inequalities:
        out["constraints"] = [inequality.to_dict() for inequality in model.inequalities]
    if model.substrate is not None:
        out["substrate"] = model.substrate.to_dict()
    if model.predict_level is not None:
        out["predict_level"] = model.predict_level
    if model.cut is not None:
        out["vertical_cut"] = vars(model.cut).copy()
    return out


def _legacy_params(family: str, raw: Mapping) -> dict:
    """A component's parameters of a ``gimap_fit_model_v1`` file in this model's conventions: σD/D → σnn/D
    (× √(2π)); a vertical cylinder's Int × 10⁻⁶⟨R⁴⟩/4."""
    params = {key: dict(value) for key, value in raw.items() if key != "sigma_D"}
    if "sigma_D" in raw:
        params["sigma_nn"] = {name: (None if raw["sigma_D"].get(name) is None else
                                     float(conventions.sigma_nn_from_sigma_d(raw["sigma_D"][name])))
                              if name in ("value", "lower", "upper") else value
                              for name, value in raw["sigma_D"].items()}
    if family == "vertical_cylinder" and "Int" in params:
        radius = float((raw.get("R") or {}).get("value", INFO["R"].value))
        spread = float((raw.get("sigma_R") or {}).get("value", INFO["sigma_R"].value))
        scale = _legacy_scale(family, radius, spread)
        params["Int"] = {name: (value * scale if name in ("value", "lower", "upper") and value is not None else value)
                         for name, value in params["Int"].items()}
    return params


def model_from_dict(data: Mapping) -> FitModel:
    schema = data.get("schema")
    if schema not in (MANUAL_SCHEMA, SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Unrecognized GIMaP fit model format.")

    def parameter(key: str, raw: Mapping) -> Parameter:
        info = INFO[key]
        lower, upper = raw.get("lower"), raw.get("upper")
        tie = raw.get("tie")
        return Parameter(float(raw.get("value", info.value)), bool(raw.get("free", info.free)) and not tie,
                         info.lower if lower is None else float(lower), INF if upper is None else float(upper),
                         str(tie) if tie else None)

    components = []
    for raw in data.get("components") or ():
        family = str(raw["family"])
        legacy = schema == LEGACY_SCHEMA
        law = LEGACY_LAWS.get(family, DEFAULT_LAW) if legacy else str(raw.get("sizes") or "number_gauss")
        component = new_component(family, structure=bool(raw.get("structure", True)),
                                  law=law if schema == MANUAL_SCHEMA or law in SIZE_LAWS else DEFAULT_LAW,
                                  heights=bool(raw.get("heights")))
        coupling = str(raw.get("coupling") or "lma")
        component = replace(component, coupling=coupling if coupling in COUPLINGS else "lma")
        interference = str(raw.get("interference") or DEFAULT_INTERFERENCE)
        if interference not in INTERFERENCES:
            raise ValueError(f"Unknown interference function {interference!r}")
        component = with_interference(component, interference)
        params = dict(component.params)
        source = _legacy_params(family, raw.get("params") or {}) if legacy else (raw.get("params") or {})
        for key, value in source.items():
            if key in params:
                params[key] = parameter(key, value)
        components.append(replace(component, params=params))
    globals_ = {key: Parameter.default(key) for key in GLOBALS}
    for key, value in (data.get("globals") or {}).items():
        if key in globals_:
            globals_[key] = parameter(key, value)
    constraints = tuple(fit_links.Inequality(str(item["parameter"]), str(item["relation"]), str(item["expression"]))
                        for item in data.get("constraints") or ())
    model = FitModel(tuple(components), globals_, contact=bool(data.get("contact", True)), inequalities=constraints,
                     substrate=Substrate.from_dict(data.get("substrate")),
                     cut=VerticalCut(**data["vertical_cut"]) if data.get("vertical_cut") else None,
                     resolution_mode=data.get("resolution_mode", "constant"), weighting=data.get("weighting", "file"),
                     predict_level=data.get("predict_level"))
    return fit_links.apply_links(fit_links.with_inequalities(model, constraints)) if constraints or any(
        p.tie for _path, p in model.parameters()) else model
