"""From the proposal network's posterior to Fitting's models: the solutions of a cut and their result record
(``PosteriorSolution``).

``predict`` is the one entry point (its docstring lists the steps): the posterior of the compositions (``infer``),
their modes refined by ``fit_engine`` (bounded least squares, the scales solved exactly), a search of the ranges where
the network's proposals miss the curve, and the posterior of the best solutions sampled around them (``laplace``). A solution carries the refined values with
their errors (Jacobian), the posterior median and the 68 % and 95 % intervals of the importance-sampled posterior,
χ²/N, its probability (the composition's posterior times the mode's share of it) and flags. An interval always holds
the refined value: the posterior has the model error and the prior's ranges, least squares the counts' σ, so on a
curve the model does not describe within σ, or with a value at a bound, the refined value can lie beside the
central interval, which is then extended to it. Flags:

* ``at_bound:NAME``     the refined value sits at a bound of its range (its error means little);
* ``unconstrained:NAME`` the 68 % interval covers more than ``UNCONSTRAINED`` of the prior's range (the curve says
  little about it);
* ``correlated:A~B``    the refined values are correlated beyond ±0.95;
* ``bic``               the sampling of a solution did not converge (fewer than ``ESS_FAILED`` effective samples):
  the solutions and compositions are compared by BIC (χ² + k ln N, σ with the model error), not by their
  evidence, and the intervals are the least-squares errors (σ with the model error; scaled by χ²/N when above 1);
* ``model_error``       the values are those of a least-squares fit with the model error in σ: the fit with the
  counts' σ alone lay outside the posterior's 95 % interval (``_within_posterior``);
* ``low_ess``           fewer than ``ESS_MIN`` effective samples: the intervals are taken with tempered weights
  (``laplace.Local.tempered``), wider, and are rough;
* ``large:R_n``         q_min·R of population n at least ``LARGE``: its Guinier region lies at or below the curve's
  lowest q, and the curve shows only its decay, which roughness or a rise at low q produce as well;
* ``small:R_n``         q_max·R of population n below ``SMALL``: smaller than the curve resolves, it acts as a smooth
  background (as fit1d named such a population);
* ``qzR:R_n``           (``interface_flags``, added by the caller that knows the cut's angles) a sphere or random
  cylinder fitted at q_z = 0, the effective interface, whose radius times the cut's q₊ is above ``QZR_LIMIT``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .. import fit_engine
from .. import fit_model as fm
from . import encoding as E
from .infer import PHYSICAL_NAMES, Cut
from . import reporting
from .model_start import GLOBAL_KEYS, KEYS, REFLECTION_KEYS, fit_model
from .prediction import (CHI2_RETRY, DELTA_BIC, ESS_FAILED, EXPLORED, LARGE, LOCAL_MAX, LOCAL_ROUNDS, LOCAL_SAMPLES,
                         MODEL_ERROR_FLOOR, MOVED_SOLUTIONS, MOVES, REFINED, RETRY_COMPOSITIONS, SAMPLES, SEEDS,
                         SHORTLIST, SLOW_ROUNDS, SMALL, predict)

UNCONSTRAINED = 0.7
ESS_MIN = 50


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


NUISANCE = ("model_error", "reflection_re", "reflection_im")
"""Values of the posterior that are not reported as the solution's: the model error and the reflection coefficient
(in ``extras``)."""


def _names(composition: int) -> list[tuple[int, str]]:
    active = E.active(composition)[0]
    return [(d, PHYSICAL_NAMES[d]) for d in range(E.DIMS) if active[d] and E.NAMES[d] not in NUISANCE]


def reflection(model: fm.FitModel) -> complex:
    """The effective reflection coefficient r of a model's spheres (0 without the exit-reflected path)."""
    geometry = fm.grazing(model)
    if geometry is None or geometry.mode != "two_path":
        return 0j
    return geometry.refl_amp * complex(math.cos(geometry.refl_phase), math.sin(geometry.refl_phase))


def _path(composition: int, dim: int):
    if dim >= E.GLOBAL.start:
        return ("globals", GLOBAL_KEYS[E.NAMES[dim]])
    return (dim // E.PER_SLOT, KEYS[dim % E.PER_SLOT])


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
    r = reflection(model)
    values[E.GLOBAL.start + 4:E.GLOBAL.start + 6] = (r.real, r.imag)
    return values


def _same(a, b, types) -> bool:
    from .infer import _distance
    return bool(_distance(a[None, :], b, types)[0] <= 1.0)


def window_flags(model: fm.FitModel, q) -> list[str]:
    """``large:R_n`` / ``small:R_n`` for the populations (in the order of their radii) whose size the measured q
    range does not show: q_min·R ≥ ``LARGE`` or q_max·R < ``SMALL``."""
    q = np.asarray(q, float)
    q_min, q_max = float(np.min(q)), float(np.max(q))
    radii = sorted(c.value("R") for c in model.components)
    out = []
    for n, radius in enumerate(radii, 1):
        if q_min * radius >= LARGE:
            out.append(f"large:R_{n}")
        elif q_max * radius < SMALL:
            out.append(f"small:R_{n}")
    return out


QZR_LIMIT = 1.7
"""q₊R of a sphere above which the form factor at q_z = 0 (the effective interface) puts its radius more than about
5 % off (Physics and Methods 9.10; on the four-term DWBA benchmark of 9.13 the median error is 15–29 % above it)."""
AT_QZ_ZERO = ("sphere", "cylinder")
"""The families a model without the cut's angles evaluates at q_z = 0 (an upright cylinder's cut is exact)."""


def interface_flags(model: fm.FitModel, qz: Optional[float]) -> list[str]:
    """``qzR:R_n`` for the spheres and random cylinders (populations numbered in the order of their radii, as
    ``window_flags``) with ``qz``·R above ``QZR_LIMIT``, when ``model`` was fitted without the cut's angles (q_z = 0,
    the effective interface). ``qz``: the cut's vacuum q₊ = k(sin α_i + sin α_f) in nm⁻¹; None or 0 when the angles
    are not known, and then no flag. A model with the angles (the Born term or the two paths) gets none. A flag only:
    no value of the solution changes."""
    if qz is None or not math.isfinite(float(qz)) or float(qz) <= 0:
        return []
    geometry = fm.grazing(model)
    if geometry is not None and geometry.qz_born > 0:
        return []
    ordered = sorted(model.components, key=lambda c: c.value("R"))
    return [f"qzR:R_{n}" for n, component in enumerate(ordered, 1)
            if component.family in AT_QZ_ZERO and float(qz) * component.value("R") > QZR_LIMIT]


def _solution(result, entry, data, fit_data, lo, hi, method, probability, extra, lsq=None,
              from_errors: bool = False) -> PosteriorSolution:
    """``from_errors``: the intervals from the least-squares errors (the refined value ± 1 and 1.96 errors, within
    the value's range), for solutions compared by BIC, whose posterior sampling did not converge. Every interval is
    extended to the refined value when that lies beside it (the module's docstring)."""
    mode, model, refined, local = entry
    # ``lsq``: (errors, χ²/N) with the σ the posterior uses (the model error added); least squares scales its errors
    # by χ²/N, and below 1 they keep their size.
    lsq_errors, lsq_chi2 = lsq if lsq is not None else ((refined.errors, refined.chi2_reduced) if refined else ({}, 1))
    widen = 1.0 / math.sqrt(min(1.0, lsq_chi2)) if lsq_chi2 > 0 else 1.0
    quantiles = (local.quantiles([0.5, 0.16, 0.84, 0.025, 0.975], ESS_MIN) if local is not None
                 else np.stack([mode.median, mode.low68, mode.high68, mode.low95, mode.high95]))
    extra["posterior_summary"] = {
        name: dict(zip(("median", "low68", "high68", "low95", "high95"), map(float, quantiles[:, dim])))
        for dim, name in _names(result.index)
    } if local is not None else {}
    if local is not None and data.reflected and any(c.family == "sphere" for c in model.components):
        extra["posterior_summary"].update(reporting.reflection_quantiles(local, model.globals["refl_phase"].value, ESS_MIN))
    extra["importance_starts"] = reporting.check_starts(local, ESS_MIN)
    extra["diagnostics"]["posterior_kind"] = (
        None if local is None else "tempered" if local.ess < ESS_MIN else "evidence")
    extra["diagnostics"]["interval_kind"] = "bic" if from_errors else extra["diagnostics"]["posterior_kind"]
    values, flags = {}, []
    for dim, name in _names(result.index):
        path = _path(result.index, dim)
        error = lsq_errors.get(path) if refined is not None else None
        error = None if error is None else error * widen
        parameter = model.get(path)
        best = float(parameter.value)
        if from_errors and error is not None and np.isfinite(error):
            quantiles[:, dim] = [best, best - error, best + error, best - 1.96 * error, best + 1.96 * error]
            if E.LOGARITHMIC[dim]:  # positive: the interval on a log scale, its width as the error's
                quantiles[1:, dim] = best * np.exp(np.array([-1, 1, -1.96, 1.96]) * min(error / best, 5.0))
            quantiles[:, dim] = np.clip(quantiles[:, dim], parameter.lower, parameter.upper)
        quantiles[[1, 3], dim] = np.minimum(quantiles[[1, 3], dim], best)
        quantiles[[2, 4], dim] = np.maximum(quantiles[[2, 4], dim], best)
        values[name] = Interval(best, *(float(x) for x in quantiles[:, dim]), None if error is None else float(error))
        low, high = quantiles[1, dim], quantiles[2, dim]
        span = np.log(high / low) if E.LOGARITHMIC[dim] else high - low
        width = hi[0, dim] - lo[0, dim]
        if np.isfinite(width) and width > 0 and span / width > UNCONSTRAINED:
            flags.append(f"unconstrained:{name}")
    if refined is not None:
        if refined.message == MODEL_ERROR_NOTE:
            flags.append("model_error")
        flags += [f"at_bound:{_label(p)}" for p in refined.at_bounds]
        flags += [f"correlated:{_label(a)}~{_label(b)}" for a, b, _rho in refined.correlated]
    flags += window_flags(model, data.q)
    ess = local.ess if local is not None else result.ess
    if ess < ESS_MIN:
        flags.append("low_ess")
    chi2 = mode.chi2_reduced
    if refined is not None:  # with the curve's own σ (``fit_data``)
        residual = fit_engine.residuals(model, fit_data)
        chi2 = float(residual @ residual) / max(1, refined.points - refined.free)
    rmse = refined.log_rmse if refined is not None else fit_engine.log_rmse(model, fit_data)
    evidence = local.log_evidence if local is not None else result.log_evidence
    count = local.samples.shape[0] if local is not None else result.samples.shape[0]
    return PosteriorSolution(result.index, result.name, probability, model, float(chi2), float(rmse), values,
                             tuple(flags), method, evidence, ess, int(count), 0.0, extra)


def _outside(composition: int, model: fm.FitModel, limits) -> int:
    """How many of the populations' values of the solution lie outside ``limits`` (2, DIMS)."""
    values = model_values(model, composition)
    if values is None:
        return 0
    return sum(not limits[0, dim] <= values[dim] <= limits[1, dim] for dim, _name in _names(composition)
               if dim < E.GLOBAL.start)  # the populations: a tail or a resolution at its bound is no disagreement


def _within_posterior(composition: int, refined, again, local, data: Cut, weighed, law, coupling, grazing):
    """The least-squares solution the posterior of ``local`` is of: ``refined`` (the counts' σ), ``again`` (the same
    fit continued with the model error in σ) or a fit with that σ from the posterior's medians.

    On a curve the model describes within σ they agree, and ``refined`` is the solution. Where a value of
    ``refined`` lies outside the posterior's central 95 % interval, the counts' optimum is set by the points the
    model describes worst, or is a local one beside the posterior's mass: the candidate with the fewest values
    outside is the solution (flag ``model_error`` when it is not ``refined``)."""
    limits = local.quantiles([0.025, 0.975], ESS_MIN)
    best, fewest = refined, _outside(composition, refined.model, limits)
    if not fewest:
        return refined
    start = fit_model(local.quantiles([0.5], ESS_MIN)[0], composition, data, law, coupling, grazing)
    for candidate in (again, fit_engine.fit(start, weighed, method="local")):
        if model_values(candidate.model, composition) is None:
            continue
        outside = _outside(composition, candidate.model, limits)
        if outside < fewest:
            best, fewest = replace(candidate, message=MODEL_ERROR_NOTE), outside
    return best


MODEL_ERROR_NOTE = "model_error"


def _reflection_extra(model: fm.FitModel) -> dict:
    r = reflection(model)
    return {"amplitude": float(abs(r)), "phase": float(math.atan2(r.imag, r.real))}


def _label(path) -> str:
    owner, key = path
    return key if owner == "globals" else f"{key}_{owner + 1}"


__all__ = ["CHI2_RETRY", "DELTA_BIC", "ESS_FAILED", "ESS_MIN", "EXPLORED", "LARGE", "LOCAL_MAX", "LOCAL_SAMPLES",
           "MODEL_ERROR_FLOOR", "MOVES", "QZR_LIMIT", "SAMPLES", "SHORTLIST", "SMALL", "UNCONSTRAINED", "Interval",
           "NUISANCE", "PosteriorSolution", "fit_model", "interface_flags", "model_values", "predict", "reflection",
           "window_flags"]
