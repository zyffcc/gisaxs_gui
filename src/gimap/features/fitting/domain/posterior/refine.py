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
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .. import fit_engine
from .. import fit_model as fm
from . import encoding as E
from .infer import PHYSICAL_NAMES, TAIL_WIDTH_MAX, Cut, infer
from .laplace import local_posterior
from . import reporting
from .moves import starts as moved_starts

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
        component = fm.new_component(family, radius=values[j], distance=values[j + 4] if structure else None,
                                     structure=structure, law=law)
        params = dict(component.params)
        for offset, key in enumerate(KEYS):
            if key in params and np.isfinite(values[j + offset]):
                params[key] = _clip(params[key], values[j + offset])
        components.append(fm.Component(family, _box(params, family, data), structure, coupling, law))
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


SAMPLES = 1000
"""Samples of the network per composition."""
SHORTLIST = 4
"""Compositions tried when the network chooses: the head's most probable among those allowed, and its most probable
of every number of populations (``infer.PER_COUNT``)."""
LOCAL_SAMPLES = 400
"""Samples per round of the local posterior of a solution."""
SEEDS = 16
"""The heaviest samples that start a search of the ranges."""
REFINED = 3
"""Modes of a composition refined by least squares, the heaviest first."""
MOVES = 2
"""Moved starts (``moves``) refined from the best solutions of each of the ``RETRY_COMPOSITIONS`` best compositions of
two or three populations without a random cylinder: the best of them by χ² with the scales solved."""
MOVED_SOLUTIONS = 2
"""The best solutions of a composition (by BIC) the moves start from."""
EXPLORED = 3
"""Modes of the tempered weights (``infer.Composition.explore``) refined as well, when their start is another sample."""
CHI2_RETRY = 2.0
"""When no solution of one of the ``RETRY_COMPOSITIONS`` best compositions refines below this χ²/N (the network's
proposals miss the curve), one search of the ranges from its heaviest samples is added."""
RETRY_COMPOSITIONS = 2
DELTA_BIC = 20.0
"""Solutions more than this above the best BIC are dropped before their posterior is sampled (their probability
would be below e⁻¹⁰ of the best's)."""
LOCAL_MAX = 4
"""At most this many solutions get their posterior sampled (the best by BIC)."""
LOCAL_ROUNDS, SLOW_ROUNDS = 8, 3
"""At most this many rounds of the local posterior (``laplace.local_posterior``); of a composition with a random
cylinder, whose evaluations are slow, ``SLOW_ROUNDS``."""
ESS_FAILED = 10
LARGE, SMALL = 0.7, 1.0
"""q_min·R and q_max·R limits of the ``large`` and ``small`` flags (``window_flags``)."""
"""Below this many effective samples at any solution, the solutions are compared by BIC (``bic`` flag)."""
MODEL_ERROR_FLOOR = 0.05
"""A relative model error per point, added to the σ of the counts (σ² + (f·I)²) in the posterior, the solutions'
probabilities and their errors: Fitting's model describes a measured cut only to about this fraction, so the
intervals are not narrower than that allows. The least-squares fit and χ²/N keep the curve's own σ."""


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


def predict(network, q, intensity, sigma=None, *, qz: float = 0.0, qz_minus: Optional[float] = None,
            compositions: Optional[Sequence[int]] = None,
            allowed=None, shortlist: int = SHORTLIST, samples: int = SAMPLES, seed: int = 0,
            law: str = "number_gauss", coupling: str = "lma", min_probability: float = 0.01,
            grazing: Optional[dict] = None, local_samples: int = LOCAL_SAMPLES,
            refine_modes: int = REFINED, explored: int = EXPLORED, model_error_floor: float = MODEL_ERROR_FLOOR,
            cancelled=None, progress=None, partial=None, reflected_beam=None, known_heights=None, measured_qz=None) -> list[PosteriorSolution]:
    """Solutions of a cut (q in nm⁻¹, qz the vacuum q₊ = k_i + k_f of the Born term in nm⁻¹, 0: the effective
    interface; ``qz_minus`` k_i − k_f when both angles are known: spheres get the exit-reflected path with a free
    effective reflection coefficient, ``Cut.reflected``), most probable first: every distinct refined solution of every composition holding at least ``min_probability`` of
    the posterior.

    1. The network's importance-weighted samples of each composition (given, or the head's ``shortlist`` among
       ``allowed``) and their modes (``infer``).
    2. Each mode (``refine_modes`` of them, and ``explored`` modes of the tempered weights) refined by least squares
       from its heaviest sample; for the ``RETRY_COMPOSITIONS`` best compositions of two or three populations,
       ``MOVES`` starts at their degeneracies (``moves``); one search of the ranges for those whose solutions all stay
       above χ²/N = ``CHI2_RETRY``. Steps 1–2 use the curve's σ.
    3. The posterior of the solutions within ``DELTA_BIC`` of the best (at most ``LOCAL_MAX``) sampled around each
       (``laplace``): intervals and evidence, with the model error ``model_error_floor`` (``MODEL_ERROR_FLOOR``)
       added to σ, as in BIC and the least-squares errors.
    4. Their probabilities from the evidences, or from BIC when a sampling did not converge (``ESS_FAILED``).

    ``cancelled``: a function asked between the steps; when it returns True, ``RuntimeError("Fitting cancelled")``.
    ``progress``: given ``(stage, done, total)`` at the same points — ``proposals`` (compositions sampled by the
    network), ``refinement`` (compositions whose modes were refined), ``starts`` (compositions given moved starts),
    ``search`` (a search of the ranges), ``sampling`` (solutions whose posterior was sampled). ``partial``: given,
    after each sampling but the last, the solutions of those sampled so far, their probabilities over them alone
    (step 4 on what is done). Neither changes what is computed nor the solutions returned."""
    tick = time.perf_counter()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise RuntimeError("Fitting cancelled")

    def tell(stage: str, done: int, total: int) -> None:
        if progress is not None:
            progress(stage, done, total)

    rng = np.random.default_rng(seed + 7919)
    # The network's samples, their modes and the least-squares fits use the counts' σ (``counts``, ``fit_data``); the
    # posterior of each solution, the probabilities and the errors the model error added to σ (``data``, ``weighed``).
    counts = Cut.prepare(q, intensity, sigma, qz, qz_minus)
    sampling = reporting.density_record(counts.q, counts.intensity)
    fit_data = fit_engine.FitData.prepare(counts.q, counts.intensity, counts.sigma)
    data = counts
    if model_error_floor > 0:
        data = replace(counts, sigma=np.hypot(counts.sigma, model_error_floor * np.abs(counts.intensity)))
    weighed = fit_engine.FitData.prepare(data.q, data.intensity, data.sigma)

    def _bic(refined) -> float:
        """χ² (with the model error) + k ln N of a least-squares solution (k its free values, the scales included)."""
        residual = fit_engine.residuals(refined.model, weighed)
        return float(residual @ residual + refined.free * math.log(max(refined.points, 2)))

    results = infer(network, counts, compositions, allowed=allowed, shortlist=shortlist, samples=samples, seed=seed,
                    law=law, coupling=coupling,
                    progress=None if progress is None else (lambda done, total: tell("proposals", done, total)))
    lo, hi = E.bounds([data.ratio])
    tau_dim = E.GLOBAL.start + 3
    found = {r.index: [] for r in results}  # composition -> [(mode, refined, centre)]

    def add(result, mode, refined) -> None:
        centre = model_values(refined.model, result.index)
        if centre is None:  # the refined radii no longer keep the composition's order of types: no local posterior
            found[result.index].append((mode, refined, None))
            return
        types = E.composition_types(result.index)
        if not any(c is not None and _same(c, centre, types) for _m, _r, c in found[result.index]):
            found[result.index].append((mode, refined, centre))

    def slow(index) -> bool:
        """A random cylinder (15–800 ms per evaluation, 20–30 s per least-squares fit): only its heaviest modes are
        refined, without the tempered modes and the moves (their benchmarks have no random cylinders)."""
        return any(family == "cylinder" for family, _s in E.composition_types(index))

    for number, result in enumerate(results):
        check()
        tell("refinement", number, len(results))
        started = []
        for mode in result.modes[:refine_modes] + ([] if slow(result.index) else result.explore[:explored]):
            if any(np.array_equal(mode.best, s, equal_nan=True) for s in started):
                continue
            started.append(mode.best)
            add(result, mode, fit_engine.fit(fit_model(mode.best, result.index, counts, law, coupling, grazing),
                                             fit_data, method="local"))
    tell("refinement", len(results), len(results))
    by_score = sorted((r for r in results if found[r.index]), key=lambda r: min(_bic(e[1]) for e in found[r.index]))
    for number, result in enumerate(by_score[:RETRY_COMPOSITIONS]):
        placed = [e for e in found[result.index] if e[2] is not None]
        if not MOVES or not placed or slow(result.index):
            continue
        check()
        tell("starts", number, len(by_score[:RETRY_COMPOSITIONS]))
        for mode, _refined, centre in sorted(placed, key=lambda e: _bic(e[1]))[:MOVED_SOLUTIONS]:
            tried = []
            for values in moved_starts(centre, result.index):
                start = fit_model(values, result.index, counts, law, coupling, grazing)
                residual = fit_engine.residuals(start, fit_data)
                tried.append((float(residual @ residual), len(tried), start))
            for _chi2, _i, start in sorted(tried)[:MOVES]:
                add(result, mode, fit_engine.fit(start, fit_data, method="local"))
    by_score = sorted((r for r in results if found[r.index]), key=lambda r: min(_bic(e[1]) for e in found[r.index]))
    for result in by_score[:RETRY_COMPOSITIONS]:
        if result.modes and min(e[1].chi2_reduced for e in found[result.index]) > CHI2_RETRY:
            heaviest = np.argsort(-result.weights)[:SEEDS]
            seeds = tuple(fit_model(result.samples[i], result.index, counts, law, coupling, grazing) for i in heaviest)
            start = fit_model(result.modes[0].best, result.index, counts, law, coupling, grazing)
            tell("search", 0, 1)
            add(result, result.modes[0], fit_engine.fit(start, fit_data, method="global", seeds=seeds, seed=seed))
    ranked = sorted(((_bic(e[1]), result, e) for result in results for e in found[result.index]),
                    key=lambda t: t[0])
    if not ranked:
        return []
    kept = [(result, e) for score, result, e in ranked[:LOCAL_MAX] if score <= ranked[0][0] + DELTA_BIC]
    entries = []  # (result, mode, model, refined, local)
    lsq_of = {}  # an entry's place -> the least-squares errors and chi2/N of its model (with the model error)

    def finish(entries: list, final: bool = True) -> list[PosteriorSolution]:
        """Step 4 on ``entries``: their probabilities, and every solution above ``min_probability`` (``final``:
        also the compositions' probabilities, ``result.probability``)."""
        # The solutions are compared by their evidence (importance sampling); when the sampling of one of them did
        # not converge (e.g. a curve the model does not describe, whose posterior with the model error lies away
        # from the least-squares optimum), all of them are compared by BIC, as fit1d does.
        by_bic = any(e[4] is None or e[4].ess < ESS_FAILED for e in entries)
        log_z = np.array([-0.5 * _bic(e[3]) if by_bic else e[4].log_evidence for e in entries])
        weights = np.exp(log_z - log_z.max())
        weights /= weights.sum()
        composition_weight = {}
        for (result, *_rest), w in zip(entries, weights):
            composition_weight[result.index] = composition_weight.get(result.index, 0.0) + float(w)
        if final:
            for result in results:
                result.probability = composition_weight.get(result.index, 0.0)
        diagnostics = reporting.comparison_diagnostics(entries, by_bic, composition_weight)
        level = 3 if counts.reflected else 2 if counts.qz > 0 else 1
        warning_qz = counts.qz if level > 1 else measured_qz
        out = []
        for place, ((result, mode, model, refined, local), w, z) in enumerate(zip(entries, weights, log_z)):
            if w < min_probability:
                continue
            share = float(w / composition_weight[result.index])
            if place not in lsq_of:  # the same numbers in every call: computed once
                lsq_errors, _pairs = fit_engine.errors(model, weighed)
                lsq_chi2 = float(np.sum(fit_engine.residuals(model, weighed) ** 2)) / max(1, refined.points - refined.free)
                lsq_of[place] = (lsq_errors, lsq_chi2)
            solution = _solution(result, (mode, model, refined, local), data, fit_data, lo, hi, "lm", float(w),
                                 {"head_log_prob": result.head_log_prob, "mode_weight": share,
                                  "diagnostics": {**diagnostics[place], **sampling,
                                                  "cut_qz": counts.qz, "cut_qz_minus": counts.qz_minus,
                                                  "model_error_floor": model_error_floor, "substrate_level": level,
                                                  "reflected_beam": reflected_beam, "measured_qz": warning_qz,
                                                  "supplied_heights_nm": known_heights or {}},
                                  "reflection": _reflection_extra(model) if counts.reflected else None,
                                  "composition_log_evidence": float(z), "network_ess": result.ess,
                                  "dropped_points": data.dropped, "compared_by": "bic" if by_bic else "evidence"},
                                 lsq=lsq_of[place], from_errors=by_bic)
            if level == 1 and measured_qz:
                solution.flags += tuple(interface_flags(model, measured_qz))
            solution.flags += tuple(reporting.grazing_flags(model, warning_qz, level, reflected_beam, known_heights))
            if by_bic:
                solution.flags = solution.flags + ("bic",)
            out.append(solution)
        seconds = time.perf_counter() - tick
        for solution in out:
            solution.seconds = seconds
        return sorted(out, key=lambda s: -s.probability)

    for number, (result, (mode, refined, centre)) in enumerate(kept):
        check()
        tell("sampling", number, len(kept))
        local = None
        if centre is not None:
            tau = float(mode.median[tau_dim]) if np.isfinite(mode.median[tau_dim]) else 0.02
            # The posterior has the model error in σ: it is sampled around the least-squares optimum with that σ,
            # which on a bright curve the model describes only in part lies away from the optimum of the counts' σ.
            again = None if slow(result.index) else fit_engine.fit(refined.model, weighed, method="local")
            around = None if again is None else model_values(again.model, result.index)
            local = local_posterior(data, result.index, centre if around is None else around, tau, rng,
                                    local_samples, law, coupling,
                                    rounds=SLOW_ROUNDS if slow(result.index) else LOCAL_ROUNDS)
            if around is not None and local.ess >= ESS_FAILED:
                refined = _within_posterior(result.index, refined, again, local, data, weighed, law, coupling,
                                            grazing)
        entries.append((result, mode, refined.model, refined, local))
        if partial is not None and len(entries) < len(kept):
            partial(finish(entries, final=False))
    tell("sampling", len(kept), len(kept))
    return finish(entries)


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
