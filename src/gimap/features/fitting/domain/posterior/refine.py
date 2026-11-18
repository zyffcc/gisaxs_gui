"""From the proposal network's posterior to Fitting's models: each mode refined by least squares, and one result
record for every method (``PosteriorSolution``).

``predict`` is the entry point: the posterior of the compositions (``infer``), their modes, and per mode a Fitting
``FitModel`` started at the mode's heaviest sample and refined by ``fit_engine`` (bounded least squares, the scales
solved exactly). A solution carries the refined values with their errors (Jacobian), the posterior median and the
68 % and 95 % intervals of the importance-sampled posterior, χ²/N, its probability (the composition's posterior
times the mode's share of it) and flags:

* ``at_bound:NAME``     the refined value sits at a bound of its range (its error means little);
* ``unconstrained:NAME`` the 68 % interval covers more than ``UNCONSTRAINED`` of the prior's range (the curve says
  little about it);
* ``correlated:A~B``    the refined values are correlated beyond ±0.95;
* ``low_ess``           fewer than ``ESS_MIN`` effective samples: the intervals rest on few samples.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import fit_engine
from .. import fit_model as fm
from . import encoding as E
from .infer import PHYSICAL_NAMES, Cut, infer

UNCONSTRAINED = 0.7
ESS_MIN = 50
KEYS = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn")
GLOBAL_KEYS = {"tail_width": "res_width", "tail_exponent": "res_exponent", "q_resolution": "q_resolution"}


@dataclass(frozen=True)
class Interval:
    best: float
    median: float
    low68: float
    high68: float
    low95: float
    high95: float
    error: Optional[float] = None


@dataclass
class PosteriorSolution:
    composition: int
    name: str
    probability: float
    model: fm.FitModel
    chi2_reduced: float
    log_rmse: float
    values: dict
    """PHYSICAL_NAMES of the composition → Interval (nm, nm⁻¹; spreads relative)."""
    flags: tuple
    method: str
    log_evidence: float
    ess: float
    samples: int
    seconds: float
    extras: dict = field(default_factory=dict)


def _clip(parameter: fm.Parameter, value: float) -> fm.Parameter:
    lower, upper = parameter.lower, parameter.upper
    if value < lower:
        lower = value
    if value > upper:
        upper = value
    return fm.Parameter(float(value), parameter.free, lower, upper)


def fit_model(values, composition: int, data: Cut, law: str = "number_gauss", coupling: str = "lma",
              grazing: Optional[dict] = None) -> fm.FitModel:
    """Fitting's model of physical values (DIMS,) of a composition, scales solved for the cut. ``grazing``: the
    values of ``fit_model.GRAZING`` to keep (the cut's angles, λ and reflection of the person's model); else the
    Born term at the cut's q_z (or the effective interface)."""
    components = []
    for slot, (family, structure) in enumerate(E.composition_types(composition)):
        j = slot * E.PER_SLOT
        component = fm.new_component(family, radius=values[j], distance=values[j + 4] if structure else None,
                                     structure=structure, law=law)
        params = dict(component.params)
        for offset, key in enumerate(KEYS):
            if key in params and np.isfinite(values[j + offset]):
                params[key] = _clip(params[key], values[j + offset])
        components.append(fm.Component(family, params, structure, coupling, law))
    model = fm.FitModel(tuple(components))
    g = E.GLOBAL.start
    model = model.with_parameter(("globals", "res_amplitude"), value=1.0, free=True)
    for offset, (name, key) in enumerate(GLOBAL_KEYS.items()):
        parameter = _clip(model.globals[key], values[g + offset])
        model = model.with_parameter(("globals", key), value=parameter.value, lower=parameter.lower,
                                     upper=parameter.upper, free=True)
    if grazing:
        for key, value in grazing.items():
            model = model.with_parameter(("globals", key), value=float(value))
    elif data.qz > 0:
        angle = math.degrees(math.asin(data.qz * 0.1 / (4 * math.pi)))
        for key, value in (("alpha_i", angle), ("alpha_f", angle), ("wavelength", 0.1)):
            model = model.with_parameter(("globals", key), value=value)
    return fit_engine.fit_scales(model, fit_engine.FitData.prepare(data.q, data.intensity, data.sigma))


def _names(composition: int) -> list[tuple[int, str]]:
    active = E.active(composition)[0]
    return [(d, PHYSICAL_NAMES[d]) for d in range(E.DIMS) if active[d] and E.NAMES[d] != "model_error"]


def _path(composition: int, dim: int):
    if dim >= E.GLOBAL.start:
        return ("globals", GLOBAL_KEYS[E.NAMES[dim]])
    return (dim // E.PER_SLOT, KEYS[dim % E.PER_SLOT])


METHODS = ("lm", "global", "posterior")
"""``lm``: each mode refined by least squares from its heaviest sample (the default); ``global``: one search per
composition (differential evolution, its first population the modes and the heaviest samples), then least squares;
``posterior``: no least squares, the heaviest sample of each mode and the importance-sampled intervals."""
SEEDS = 16


def predict(network, q, intensity, sigma=None, *, qz: float = 0.0, compositions: Optional[Sequence[int]] = None,
            allowed=None, shortlist: int = 4, samples: int = 2000, method: str = "lm", seed: int = 0,
            law: str = "number_gauss", coupling: str = "lma", min_probability: float = 0.01,
            grazing: Optional[dict] = None) -> list[PosteriorSolution]:
    """Solutions of a cut (q in nm⁻¹, qz the vacuum q_z of the Born term in nm⁻¹, 0: the effective interface), most
    probable first: every mode of every composition holding at least ``min_probability`` of the posterior
    (``global``: the best of each composition)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r} (one of {', '.join(METHODS)})")
    tick = time.perf_counter()
    data = Cut.prepare(q, intensity, sigma, qz)
    fit_data = fit_engine.FitData.prepare(data.q, data.intensity, data.sigma)
    results = infer(network, data, compositions, allowed=allowed, shortlist=shortlist, samples=samples, seed=seed,
                    law=law, coupling=coupling)
    lo, hi = E.bounds([data.ratio])
    out = []
    for result in results:
        modes = result.modes
        if method == "global" and modes:
            heaviest = np.argsort(-result.weights)[:SEEDS]
            seeds = tuple(fit_model(result.samples[i], result.index, data, law, coupling, grazing) for i in heaviest)
            modes = [modes[0]]
        for mode in modes:
            probability = result.probability * (1.0 if method == "global" else mode.weight)
            if probability < min_probability:
                continue
            model = fit_model(mode.best, result.index, data, law, coupling, grazing)
            refined = None
            if method == "lm":
                refined = fit_engine.fit(model, fit_data, method="local")
            elif method == "global":
                refined = fit_engine.fit(model, fit_data, method="global", seeds=seeds, seed=seed)
            if refined is not None:
                model = refined.model
            values, flags = {}, []
            for dim, name in _names(result.index):
                path = _path(result.index, dim)
                best = float(model.get(path).value)
                error = refined.errors.get(path) if refined is not None else None
                values[name] = Interval(best, float(mode.median[dim]), float(mode.low68[dim]), float(mode.high68[dim]),
                                        float(mode.low95[dim]), float(mode.high95[dim]),
                                        None if error is None else float(error))
                span = (np.log(mode.high68[dim] / mode.low68[dim]) if E.LOGARITHMIC[dim]
                        else mode.high68[dim] - mode.low68[dim])
                width = hi[0, dim] - lo[0, dim]
                if np.isfinite(width) and width > 0 and span / width > UNCONSTRAINED:
                    flags.append(f"unconstrained:{name}")
            if refined is not None:
                flags += [f"at_bound:{_label(p)}" for p in refined.at_bounds]
                flags += [f"correlated:{_label(a)}~{_label(b)}" for a, b, _rho in refined.correlated]
            if result.ess < ESS_MIN:
                flags.append("low_ess")
            chi2 = refined.chi2_reduced if refined is not None else mode.chi2_reduced
            rmse = refined.log_rmse if refined is not None else fit_engine.log_rmse(model, fit_data)
            out.append(PosteriorSolution(result.index, result.name, probability, model, float(chi2), float(rmse), values,
                                tuple(flags), method, result.log_evidence,
                                result.ess, int(result.samples.shape[0]), 0.0,
                                {"head_log_prob": result.head_log_prob, "mode_weight": mode.weight,
                                 "dropped_points": data.dropped}))
    seconds = time.perf_counter() - tick
    for solution in out:
        solution.seconds = seconds
    return sorted(out, key=lambda s: -s.probability)


def _label(path) -> str:
    owner, key = path
    return key if owner == "globals" else f"{key}_{owner + 1}"


__all__ = ["ESS_MIN", "METHODS", "Interval", "PosteriorSolution", "UNCONSTRAINED", "fit_model", "predict"]
