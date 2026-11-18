"""From the proposal network's posterior to Fitting's models: each mode refined by least squares, and one result
record for every method (``PosteriorSolution``).

``predict`` is the entry point: the posterior of the compositions (``infer``), their modes, and per mode a Fitting
``FitModel`` started at the mode's heaviest sample and refined by ``fit_engine`` (bounded least squares, the scales
solved exactly); when no mode of a composition refines below ``CHI2_RETRY`` (the network's proposals miss the curve),
one search of the ranges started from the heaviest samples is added. A solution carries the refined values with their errors (Jacobian), the posterior median and the
68 % and 95 % intervals of the importance-sampled posterior, χ²/N, its probability (the composition's posterior
times the mode's share of it) and flags:

* ``at_bound:NAME``     the refined value sits at a bound of its range (its error means little);
* ``unconstrained:NAME`` the 68 % interval covers more than ``UNCONSTRAINED`` of the prior's range (the curve says
  little about it);
* ``correlated:A~B``    the refined values are correlated beyond ±0.95;
* ``bic``               the sampling of a solution did not converge (fewer than ``ESS_FAILED`` effective samples):
  the solutions and compositions are compared by BIC (χ² + k ln N), not by their evidence;
* ``low_ess``           fewer than ``ESS_MIN`` effective samples: the intervals are taken with tempered weights
  (``laplace.Local.tempered``), wider, and are rough.
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
from .infer import PHYSICAL_NAMES, TAIL_WIDTH_MAX, Cut, infer
from .laplace import local_posterior

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
``posterior``: no least squares, the heaviest sample of each mode and the network's importance-sampled intervals.
After ``lm`` and ``global`` the posterior of every refined solution is sampled around it (``laplace``): its intervals
and evidence; the compositions' probabilities follow from the evidences of their solutions."""
SEEDS = 16
REFINED = 3
CHI2_RETRY = 2.0
"""``lm``: when no mode of a composition refines below this χ²/N, one search of the ranges from the samples too."""
ESS_FAILED = 10
"""Below this many effective samples at any solution, the solutions are compared by BIC (``bic`` flag)."""
"""Modes of a composition refined by least squares, the heaviest first."""


def model_values(model: fm.FitModel, composition: int) -> Optional[np.ndarray]:
    """Physical values (DIMS,) of a model of ``composition`` (its components in the order of their radii); None when
    the refined radii no longer keep the composition's order of types."""
    ordered = sorted(model.components, key=lambda c: c.value("R"))
    if tuple((c.family, bool(c.structure)) for c in ordered) != E.composition_types(composition):
        return None
    values = np.full(E.DIMS, np.nan)
    for slot, component in enumerate(ordered):
        for offset, key in enumerate(KEYS):
            if key in component.keys():
                values[slot * E.PER_SLOT + offset] = component.value(key)
    for offset, key in enumerate(GLOBAL_KEYS.values()):
        values[E.GLOBAL.start + offset] = model.globals[key].value
    return values


def _same(a, b, types) -> bool:
    from .infer import _distance
    return bool(_distance(a[None, :], b, types)[0] <= 1.0)


def _solution(result, entry, data, fit_data, lo, hi, method, probability, extra) -> PosteriorSolution:
    mode, model, refined, local = entry
    quantiles = (local.quantiles([0.5, 0.16, 0.84, 0.025, 0.975], ESS_MIN) if local is not None
                 else np.stack([mode.median, mode.low68, mode.high68, mode.low95, mode.high95]))
    values, flags = {}, []
    for dim, name in _names(result.index):
        path = _path(result.index, dim)
        error = refined.errors.get(path) if refined is not None else None
        values[name] = Interval(float(model.get(path).value), *(float(x) for x in quantiles[:, dim]),
                                None if error is None else float(error))
        low, high = quantiles[1, dim], quantiles[2, dim]
        span = np.log(high / low) if E.LOGARITHMIC[dim] else high - low
        width = hi[0, dim] - lo[0, dim]
        if np.isfinite(width) and width > 0 and span / width > UNCONSTRAINED:
            flags.append(f"unconstrained:{name}")
    if refined is not None:
        flags += [f"at_bound:{_label(p)}" for p in refined.at_bounds]
        flags += [f"correlated:{_label(a)}~{_label(b)}" for a, b, _rho in refined.correlated]
    ess = local.ess if local is not None else result.ess
    if ess < ESS_MIN:
        flags.append("low_ess")
    chi2 = refined.chi2_reduced if refined is not None else mode.chi2_reduced
    rmse = refined.log_rmse if refined is not None else fit_engine.log_rmse(model, fit_data)
    evidence = local.log_evidence if local is not None else result.log_evidence
    count = local.samples.shape[0] if local is not None else result.samples.shape[0]
    return PosteriorSolution(result.index, result.name, probability, model, float(chi2), float(rmse), values,
                             tuple(flags), method, evidence, ess, int(count), 0.0, extra)


def predict(network, q, intensity, sigma=None, *, qz: float = 0.0, compositions: Optional[Sequence[int]] = None,
            allowed=None, shortlist: int = 4, samples: int = 2000, method: str = "lm", seed: int = 0,
            law: str = "number_gauss", coupling: str = "lma", min_probability: float = 0.01,
            grazing: Optional[dict] = None, local_samples: int = 600,
            refine_modes: int = REFINED) -> list[PosteriorSolution]:
    """Solutions of a cut (q in nm⁻¹, qz the vacuum q_z of the Born term in nm⁻¹, 0: the effective interface), most
    probable first: every distinct refined solution of every composition holding at least ``min_probability`` of
    the posterior (``global``: the best of each composition)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r} (one of {', '.join(METHODS)})")
    tick = time.perf_counter()
    rng = np.random.default_rng(seed + 7919)
    data = Cut.prepare(q, intensity, sigma, qz)
    fit_data = fit_engine.FitData.prepare(data.q, data.intensity, data.sigma)
    results = infer(network, data, compositions, allowed=allowed, shortlist=shortlist, samples=samples, seed=seed,
                    law=law, coupling=coupling)
    lo, hi = E.bounds([data.ratio])
    tau_dim = E.GLOBAL.start + 3
    per_composition = []
    for result in results:
        modes = result.modes[:refine_modes]
        seeds = ()
        if method == "global" and modes:
            heaviest = np.argsort(-result.weights)[:SEEDS]
            seeds = tuple(fit_model(result.samples[i], result.index, data, law, coupling, grazing) for i in heaviest)
            modes = modes[:1]
        entries = []

        def add(mode, refine_method, seeds=()):
            model = fit_model(mode.best, result.index, data, law, coupling, grazing)
            refined, local = None, None
            if refine_method == "lm":
                refined = fit_engine.fit(model, fit_data, method="local")
            elif refine_method == "global":
                refined = fit_engine.fit(model, fit_data, method="global", seeds=seeds, seed=seed)
            if refined is not None:
                model = refined.model
                centre = model_values(model, result.index)
                if centre is not None:
                    tau = float(mode.median[tau_dim]) if np.isfinite(mode.median[tau_dim]) else 0.02
                    local = local_posterior(data, result.index, centre, tau, rng, local_samples, law, coupling)
                    if any(e[3] is not None and _same(e[3].centre, local.centre, E.composition_types(result.index))
                           for e in entries):
                        return  # the same solution again
            entries.append((mode, model, refined, local))

        for mode in modes:
            add(mode, method, seeds)
        refined_chi2 = [e[2].chi2_reduced for e in entries if e[2] is not None]
        if method == "lm" and modes and refined_chi2 and min(refined_chi2) > CHI2_RETRY:
            # None of the modes' refinements describes the curve: one search of the ranges started from the samples
            heaviest = np.argsort(-result.weights)[:SEEDS]
            add(modes[0], "global",
                tuple(fit_model(result.samples[i], result.index, data, law, coupling, grazing) for i in heaviest))
        per_composition.append([result, entries, None, None])
    if not per_composition:
        return []
    # The solutions are compared by their evidence (importance sampling); when the sampling of one of them did not
    # converge (fewer than ESS_FAILED effective samples, e.g. a curve the model does not describe, whose posterior
    # with the model error lies away from the least-squares optimum), all of them are compared by BIC, as fit1d does.
    by_bic = method != "posterior" and any(e[3] is None or e[3].ess < ESS_FAILED
                                           for c in per_composition for e in c[1])
    for c in per_composition:
        result, entries = c[0], c[1]
        if by_bic and entries and all(e[2] is not None for e in entries):
            evidences = np.array([-0.5 * _bic(e[2]) for e in entries])
        elif entries and all(e[3] is not None for e in entries):
            evidences = np.array([e[3].log_evidence for e in entries])
        else:
            evidences = None
        if evidences is not None:
            top = evidences.max()
            c[2] = float(top + np.log(np.sum(np.exp(evidences - top))))
            c[3] = np.exp(evidences - c[2])
        else:
            c[2] = result.log_evidence
            total = sum(e[0].weight for e in entries) or 1.0
            c[3] = np.array([e[0].weight / total for e in entries])
    log_z = np.array([c[2] for c in per_composition])
    probabilities = np.exp(log_z - log_z.max())
    probabilities /= probabilities.sum()
    out = []
    for (result, entries, composition_log_z, shares), p in zip(per_composition, probabilities):
        result.probability = float(p)
        for entry, share in zip(entries, shares):
            probability = float(p * share)
            if probability < min_probability:
                continue
            solution = _solution(result, entry, data, fit_data, lo, hi, method, probability,
                                 {"head_log_prob": result.head_log_prob, "mode_weight": float(share),
                                  "composition_log_evidence": composition_log_z, "network_ess": result.ess,
                                  "dropped_points": data.dropped, "compared_by": "bic" if by_bic else "evidence"})
            if by_bic:
                solution.flags = solution.flags + ("bic",)
            out.append(solution)
    seconds = time.perf_counter() - tick
    for solution in out:
        solution.seconds = seconds
    return sorted(out, key=lambda s: -s.probability)


def _bic(refined: fit_engine.FitResult) -> float:
    """χ² + k ln N of a least-squares solution (k its free values, the scales included)."""
    chi2 = refined.chi2_reduced * max(1, refined.points - refined.free)
    return float(chi2 + refined.free * math.log(max(refined.points, 2)))


def _label(path) -> str:
    owner, key = path
    return key if owner == "globals" else f"{key}_{owner + 1}"


__all__ = ["CHI2_RETRY", "ESS_FAILED", "ESS_MIN", "METHODS", "Interval", "PosteriorSolution", "UNCONSTRAINED", "fit_model", "model_values",
           "predict"]
