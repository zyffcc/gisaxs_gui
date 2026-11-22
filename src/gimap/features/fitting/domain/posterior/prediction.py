"""Predict's proposal, refinement and importance-sampling stages, shared by all four substrate levels."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .. import fit_engine
from . import encoding as E
from . import reporting
from .infer import Cut, infer
from .laplace import local_posterior
from .model_start import fit_model
from .moves import starts as moved_starts

if TYPE_CHECKING:
    from .refine import PosteriorSolution


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


def predict(network, q, intensity, sigma=None, *, qz: float = 0.0, qz_minus: Optional[float] = None,
            compositions: Optional[Sequence[int]] = None,
            allowed=None, shortlist: int = SHORTLIST, samples: int = SAMPLES, seed: int = 0,
            law: str = "number_gauss", coupling: str = "lma", min_probability: float = 0.01,
            grazing: Optional[dict] = None, local_samples: int = LOCAL_SAMPLES,
            refine_modes: int = REFINED, explored: int = EXPLORED, model_error_floor: float = MODEL_ERROR_FLOOR,
            cancelled=None, progress=None, partial=None, reflected_beam=None, known_heights=None, measured_qz=None,
            substrate=None, upright_heights=None) -> list[PosteriorSolution]:
    """Solutions of a cut (q in nm⁻¹, qz the vacuum q₊ = k_i + k_f of the Born term in nm⁻¹, 0: the effective
    interface; ``qz_minus`` k_i − k_f when both angles are known: spheres get the exit-reflected path with a free
    effective reflection coefficient, ``Cut.reflected``), most probable first: every distinct refined solution of every composition holding at least ``min_probability`` of
    the posterior. With a known substrate and the wavelength and both angles in grazing, all stages evaluate the
    four-term DWBA. The proposal network still conditions on the two measured vertical wave-vector combinations.

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
    from . import refine

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
    counts = Cut.prepare(q, intensity, sigma, qz, qz_minus, substrate=substrate, grazing=grazing,
                         upright_heights=upright_heights)
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
        centre = refine.model_values(refined.model, result.index)
        if centre is None:  # the refined radii no longer keep the composition's order of types: no local posterior
            found[result.index].append((mode, refined, None))
            return
        types = E.composition_types(result.index)
        if not any(c is not None and refine._same(c, centre, types) for _m, _r, c in found[result.index]):
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
    for number, result in enumerate(by_score[:refine.RETRY_COMPOSITIONS]):
        placed = [e for e in found[result.index] if e[2] is not None]
        if not refine.MOVES or not placed or slow(result.index):
            continue
        check()
        tell("starts", number, len(by_score[:refine.RETRY_COMPOSITIONS]))
        for mode, _refined, centre in sorted(placed, key=lambda e: _bic(e[1]))[:refine.MOVED_SOLUTIONS]:
            tried = []
            for values in moved_starts(centre, result.index):
                start = fit_model(values, result.index, counts, law, coupling, grazing)
                residual = fit_engine.residuals(start, fit_data)
                tried.append((float(residual @ residual), len(tried), start))
            for _chi2, _i, start in sorted(tried)[:refine.MOVES]:
                add(result, mode, fit_engine.fit(start, fit_data, method="local"))
    by_score = sorted((r for r in results if found[r.index]), key=lambda r: min(_bic(e[1]) for e in found[r.index]))
    for result in by_score[:refine.RETRY_COMPOSITIONS]:
        if result.modes and min(e[1].chi2_reduced for e in found[result.index]) > refine.CHI2_RETRY:
            heaviest = np.argsort(-result.weights)[:refine.SEEDS]
            seeds = tuple(fit_model(result.samples[i], result.index, counts, law, coupling, grazing) for i in heaviest)
            start = fit_model(result.modes[0].best, result.index, counts, law, coupling, grazing)
            tell("search", 0, 1)
            add(result, result.modes[0], fit_engine.fit(start, fit_data, method="global", seeds=seeds, seed=seed))
    ranked = sorted(((_bic(e[1]), result, e) for result in results for e in found[result.index]),
                    key=lambda t: t[0])
    if not ranked:
        return []
    kept = [(result, e) for score, result, e in ranked[:refine.LOCAL_MAX] if score <= ranked[0][0] + refine.DELTA_BIC]
    entries = []  # (result, mode, model, refined, local)
    lsq_of = {}  # an entry's place -> the least-squares errors and chi2/N of its model (with the model error)

    def finish(entries: list, final: bool = True) -> list[PosteriorSolution]:
        """Step 4 on ``entries``: their probabilities, and every solution above ``min_probability`` (``final``:
        also the compositions' probabilities, ``result.probability``)."""
        # The solutions are compared by their evidence (importance sampling); when the sampling of one of them did
        # not converge (e.g. a curve the model does not describe, whose posterior with the model error lies away
        # from the least-squares optimum), all of them are compared by BIC, as fit1d does.
        by_bic = any(e[4] is None or e[4].ess < refine.ESS_FAILED for e in entries)
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
        level = counts.level
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
            solution = refine._solution(result, (mode, model, refined, local), data, fit_data, lo, hi, "lm", float(w),
                                 {"head_log_prob": result.head_log_prob, "mode_weight": share,
                                  "diagnostics": {**diagnostics[place], **sampling, **counts.substrate_record(),
                                                  "cut_qz": counts.qz, "cut_qz_minus": counts.qz_minus,
                                                  "model_error_floor": model_error_floor, "substrate_level": level,
                                                  "reflected_beam": reflected_beam, "measured_qz": warning_qz,
                                                  "supplied_heights_nm": known_heights or {}},
                                  "reflection": refine._reflection_extra(model) if counts.reflected else None,
                                  "composition_log_evidence": float(z), "network_ess": result.ess,
                                  "dropped_points": data.dropped, "compared_by": "bic" if by_bic else "evidence"},
                                 lsq=lsq_of[place], from_errors=by_bic)
            if level == 1 and measured_qz:
                solution.flags += tuple(refine.interface_flags(model, measured_qz))
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
            around = None if again is None else refine.model_values(again.model, result.index)
            local = local_posterior(data, result.index, centre if around is None else around, tau, rng,
                                    local_samples, law, coupling,
                                    rounds=refine.SLOW_ROUNDS if slow(result.index) else refine.LOCAL_ROUNDS)
            if around is not None and local.ess >= refine.ESS_FAILED:
                refined = refine._within_posterior(result.index, refined, again, local, data, weighed, law, coupling,
                                            grazing)
        entries.append((result, mode, refined.model, refined, local))
        if partial is not None and len(entries) < len(kept):
            partial(finish(entries, final=False))
    tell("sampling", len(kept), len(kept))
    return finish(entries)
