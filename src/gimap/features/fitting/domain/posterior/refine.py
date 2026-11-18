"""From the proposal network's posterior to Fitting's models: the solutions of a cut and their result record
(``PosteriorSolution``).

``predict`` is the one entry point (its docstring lists the steps): the posterior of the compositions (``infer``),
their modes refined by ``fit_engine`` (bounded least squares, the scales solved exactly), a search of the ranges where
the network's proposals miss the curve, and the posterior of the best solutions sampled around them (``laplace``). A solution carries the refined values with their errors (Jacobian), the posterior median and the
68 % and 95 % intervals of the importance-sampled posterior, χ²/N, its probability (the composition's posterior
times the mode's share of it) and flags:

* ``at_bound:NAME``     the refined value sits at a bound of its range (its error means little);
* ``unconstrained:NAME`` the 68 % interval covers more than ``UNCONSTRAINED`` of the prior's range (the curve says
  little about it);
* ``correlated:A~B``    the refined values are correlated beyond ±0.95;
* ``bic``               the sampling of a solution did not converge (fewer than ``ESS_FAILED`` effective samples):
  the solutions and compositions are compared by BIC (χ² + k ln N), not by their evidence, and the intervals are
  the least-squares errors (scaled by χ²/N);
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


SAMPLES = 1000
"""Samples of the network per composition."""
SHORTLIST = 3
"""Compositions tried when the network chooses (the head's most probable among those allowed)."""
LOCAL_SAMPLES = 400
"""Samples per round of the local posterior of a solution."""
SEEDS = 16
"""The heaviest samples that start a search of the ranges."""
REFINED = 3
"""Modes of a composition refined by least squares, the heaviest first."""
CHI2_RETRY = 2.0
"""When no solution of one of the ``RETRY_COMPOSITIONS`` best compositions refines below this χ²/N (the network's
proposals miss the curve), one search of the ranges from its heaviest samples is added."""
RETRY_COMPOSITIONS = 2
DELTA_BIC = 20.0
"""Solutions more than this above the best BIC are dropped before their posterior is sampled (their probability
would be below e⁻¹⁰ of the best's)."""
LOCAL_MAX = 4
"""At most this many solutions get their posterior sampled (the best by BIC)."""
ESS_FAILED = 10
"""Below this many effective samples at any solution, the solutions are compared by BIC (``bic`` flag)."""


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


def _solution(result, entry, data, fit_data, lo, hi, method, probability, extra,
              from_errors: bool = False) -> PosteriorSolution:
    """``from_errors``: the intervals from the least-squares errors (the refined value ± 1 and 1.96 errors), for
    solutions compared by BIC, whose posterior sampling did not converge."""
    mode, model, refined, local = entry
    quantiles = (local.quantiles([0.5, 0.16, 0.84, 0.025, 0.975], ESS_MIN) if local is not None
                 else np.stack([mode.median, mode.low68, mode.high68, mode.low95, mode.high95]))
    values, flags = {}, []
    for dim, name in _names(result.index):
        path = _path(result.index, dim)
        error = refined.errors.get(path) if refined is not None else None
        if from_errors and error is not None and np.isfinite(error):
            best = float(model.get(path).value)
            quantiles[:, dim] = [best, best - error, best + error, best - 1.96 * error, best + 1.96 * error]
            if E.LOGARITHMIC[dim]:  # positive: the interval on a log scale, its width as the error's
                quantiles[1:, dim] = best * np.exp(np.array([-1, 1, -1.96, 1.96]) * min(error / best, 5.0))
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
            allowed=None, shortlist: int = SHORTLIST, samples: int = SAMPLES, seed: int = 0,
            law: str = "number_gauss", coupling: str = "lma", min_probability: float = 0.01,
            grazing: Optional[dict] = None, local_samples: int = LOCAL_SAMPLES,
            refine_modes: int = REFINED) -> list[PosteriorSolution]:
    """Solutions of a cut (q in nm⁻¹, qz the vacuum q_z of the Born term in nm⁻¹, 0: the effective interface), most
    probable first: every distinct refined solution of every composition holding at least ``min_probability`` of
    the posterior.

    1. The network's importance-weighted samples of each composition (given, or the head's ``shortlist`` among
       ``allowed``) and their modes (``infer``).
    2. Each mode refined by least squares from its heaviest sample; one search of the ranges for the
       ``RETRY_COMPOSITIONS`` best compositions whose solutions all stay above χ²/N = ``CHI2_RETRY``.
    3. The posterior of the solutions within ``DELTA_BIC`` of the best (at most ``LOCAL_MAX``) sampled around each
       (``laplace``): intervals and evidence.
    4. Their probabilities from the evidences, or from BIC when a sampling did not converge (``ESS_FAILED``)."""
    tick = time.perf_counter()
    rng = np.random.default_rng(seed + 7919)
    data = Cut.prepare(q, intensity, sigma, qz)
    fit_data = fit_engine.FitData.prepare(data.q, data.intensity, data.sigma)
    results = infer(network, data, compositions, allowed=allowed, shortlist=shortlist, samples=samples, seed=seed,
                    law=law, coupling=coupling)
    lo, hi = E.bounds([data.ratio])
    tau_dim = E.GLOBAL.start + 3
    found = {r.index: [] for r in results}  # composition -> [(mode, refined, centre)]

    def add(result, mode, refined) -> None:
        centre = model_values(refined.model, result.index)
        if centre is None:
            return  # the refined radii no longer keep the composition's order of types
        types = E.composition_types(result.index)
        if not any(_same(c, centre, types) for _m, _r, c in found[result.index]):
            found[result.index].append((mode, refined, centre))

    for result in results:
        for mode in result.modes[:refine_modes]:
            add(result, mode, fit_engine.fit(fit_model(mode.best, result.index, data, law, coupling, grazing),
                                             fit_data, method="local"))
    by_score = sorted((r for r in results if found[r.index]), key=lambda r: min(_bic(e[1]) for e in found[r.index]))
    for result in by_score[:RETRY_COMPOSITIONS]:
        if result.modes and min(e[1].chi2_reduced for e in found[result.index]) > CHI2_RETRY:
            heaviest = np.argsort(-result.weights)[:SEEDS]
            seeds = tuple(fit_model(result.samples[i], result.index, data, law, coupling, grazing) for i in heaviest)
            start = fit_model(result.modes[0].best, result.index, data, law, coupling, grazing)
            add(result, result.modes[0], fit_engine.fit(start, fit_data, method="global", seeds=seeds, seed=seed))
    ranked = sorted(((_bic(e[1]), result, e) for result in results for e in found[result.index]),
                    key=lambda t: t[0])
    if not ranked:
        return []
    kept = [(result, e) for score, result, e in ranked[:LOCAL_MAX] if score <= ranked[0][0] + DELTA_BIC]
    entries = []  # (result, mode, model, refined, local)
    for result, (mode, refined, centre) in kept:
        tau = float(mode.median[tau_dim]) if np.isfinite(mode.median[tau_dim]) else 0.02
        local = local_posterior(data, result.index, centre, tau, rng, local_samples, law, coupling)
        entries.append((result, mode, refined.model, refined, local))
    # The solutions are compared by their evidence (importance sampling); when the sampling of one of them did not
    # converge (e.g. a curve the model does not describe, whose posterior with the model error lies away from the
    # least-squares optimum), all of them are compared by BIC, as fit1d does.
    by_bic = any(e[4].ess < ESS_FAILED for e in entries)
    log_z = np.array([-0.5 * _bic(e[3]) if by_bic else e[4].log_evidence for e in entries])
    weights = np.exp(log_z - log_z.max())
    weights /= weights.sum()
    composition_weight = {}
    for (result, *_rest), w in zip(entries, weights):
        composition_weight[result.index] = composition_weight.get(result.index, 0.0) + float(w)
    for result in results:
        result.probability = composition_weight.get(result.index, 0.0)
    out = []
    for (result, mode, model, refined, local), w, z in zip(entries, weights, log_z):
        if w < min_probability:
            continue
        share = float(w / composition_weight[result.index])
        solution = _solution(result, (mode, model, refined, local), data, fit_data, lo, hi, "lm", float(w),
                             {"head_log_prob": result.head_log_prob, "mode_weight": share,
                              "composition_log_evidence": float(z), "network_ess": result.ess,
                              "dropped_points": data.dropped, "compared_by": "bic" if by_bic else "evidence"},
                             from_errors=by_bic)
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


__all__ = ["CHI2_RETRY", "DELTA_BIC", "ESS_FAILED", "ESS_MIN", "Interval", "LOCAL_MAX", "LOCAL_SAMPLES",
           "PosteriorSolution", "SAMPLES", "SHORTLIST", "UNCONSTRAINED", "fit_model", "model_values", "predict"]
