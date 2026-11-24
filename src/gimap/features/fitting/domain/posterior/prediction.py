"""Predict's proposal, refinement and importance-sampling stages, shared by all four substrate levels."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .. import fit_engine
from . import encoding as E
from . import parallel, reporting
from .infer import Composition, Cut, Mode, infer
from .laplace import Local, local_posterior, pool_composition
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
"""Use BIC for all compositions if any composition has fewer effective pooled draws than this."""
LARGE, SMALL = 0.7, 1.0
"""q_min·R and q_max·R limits of the ``large`` and ``small`` flags (``window_flags``)."""
MODEL_ERROR_FLOOR = 0.05
"""A relative model error per point, added to the σ of the counts (σ² + (f·I)²) in the posterior, the solutions'
probabilities and their errors: Fitting's model describes a measured cut only to about this fraction, so the
intervals are not narrower than that allows. The least-squares fit and χ²/N keep the curve's own σ."""


@dataclass
class Candidate:
    result: Composition
    mode: Mode
    counts_fit: fit_engine.FitResult
    floor_fit: fit_engine.FitResult
    centre: Optional[np.ndarray]
    bic: float
    local: Optional[Local] = None


def _normalized(scores):
    weights = np.exp(scores - np.max(scores))
    return weights / weights.sum()


def _comparison(groups, pools, bics, by_bic):
    """Compare one score per composition, then partition its weight among its starts."""
    indices = list(groups)
    best = {c: np.min(bics[groups[c]]) for c in indices}
    scores = [-best[c] / 2 if by_bic else pools[c].log_evidence for c in indices]
    probabilities = dict(zip(indices, _normalized(np.asarray(scores))))
    shares = np.empty(len(bics))
    for c, places in groups.items():
        shares[places] = _normalized(-0.5 * bics[places]) if by_bic else pools[c].shares
    return probabilities, shares, best


@parallel.one_blas_thread
def predict(network, q, intensity, sigma=None, *, qz: float = 0.0, qz_minus: Optional[float] = None,
            compositions: Optional[Sequence[int]] = None,
            allowed=None, shortlist: int = SHORTLIST, samples: int = SAMPLES, seed: int = 0,
            law: str = "number_gauss", coupling: str = "lma", min_probability: float = 0.01,
            grazing: Optional[dict] = None, local_samples: int = LOCAL_SAMPLES,
            refine_modes: int = REFINED, explored: int = EXPLORED, model_error_floor: float = MODEL_ERROR_FLOOR,
            cancelled=None, progress=None, partial=None, reflected_beam=None, known_heights=None, measured_qz=None,
            substrate=None, upright_heights=None, threads: int = 1) -> list[PosteriorSolution]:
    """Solutions of a cut (q in nm⁻¹, qz the vacuum q₊ = k_i + k_f of the Born term in nm⁻¹, 0: the effective
    interface; ``qz_minus`` k_i − k_f when both angles are known: spheres get the exit-reflected path with a free
    effective reflection coefficient, ``Cut.reflected``), most probable first: every distinct refined solution of every composition holding at least ``min_probability`` of
    the posterior. With a known substrate and the wavelength and both angles in grazing, all stages use four paths
    for spheres and the finite Born transfer for random cylinders.

    1. The network's importance-weighted samples of each composition (given, or the head's ``shortlist`` among
       ``allowed``) and their modes (``infer``).
    2. Each mode (``refine_modes`` of them, and ``explored`` modes of the tempered weights) refined by least squares
       from its heaviest sample; for the ``RETRY_COMPOSITIONS`` best compositions of two or three populations,
       ``MOVES`` starts at their degeneracies (``moves``); one search of the ranges for those whose solutions all stay
       above χ²/N = ``CHI2_RETRY``. Starts are fitted with the counting errors and refitted with the floor.
    3. Sample around the floor-weighted optima within ``DELTA_BIC`` of the best (at most ``LOCAL_MAX``).
       Pool all proposal rounds within each composition for its evidence and each start's share.
    4. Normalize the evidences across compositions. If any composition fails sampling (``ESS_FAILED``),
       normalize exp(−min BIC/2) across compositions and split each weight by relative BIC.

    ``cancelled``: a function asked between the steps; when it returns True, ``RuntimeError("Fitting cancelled")``.
    ``progress``: given ``(stage, done, total)`` at the same points — ``proposals`` (compositions sampled by the
    network), ``refinement`` (compositions whose modes were refined), ``starts`` (compositions given moved starts),
    ``search`` (a search of the ranges), ``sampling`` (solutions whose posterior was sampled). ``partial``: given,
    after each sampling but the last, the solutions of those sampled so far, their probabilities over them alone
    (step 4 on what is done). Neither changes what is computed nor the solutions returned, nor does ``threads``, the
    threads of the batch likelihoods of steps 1 and 3 and of the least-squares fits of step 2 (``parallel``)."""
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

    def bic(floor_fit) -> float:
        residual = fit_engine.residuals(floor_fit.model, weighed)
        return float(residual @ residual + floor_fit.free * math.log(floor_fit.points))

    results = infer(network, counts, compositions, allowed=allowed, shortlist=shortlist, samples=samples, seed=seed,
                    law=law, coupling=coupling, threads=threads,
                    progress=None if progress is None else (lambda done, total: tell("proposals", done, total)))
    lo, hi = E.bounds([data.ratio])
    tau_dim = E.ERROR_DIM
    found = {r.index: [] for r in results}

    def measured(result, mode, refined) -> Candidate:
        floor_fit = fit_engine.fit(refined.model, weighed, method="local") if model_error_floor > 0 else refined
        centre = refine.model_values(floor_fit.model, result.index)
        return Candidate(result, mode, refined, floor_fit, centre, bic(floor_fit))

    def fitted(job) -> Candidate:
        """A start ``(result, mode, model)`` refined by least squares, then with the model-error floor."""
        result, mode, start = job
        return measured(result, mode, fit_engine.fit(start, fit_data, method="local"))

    def add(result, mode, refined) -> None:
        place(measured(result, mode, refined))

    def place_all(jobs, numbers, stage: str, total: int) -> None:
        """The starts ``jobs`` ((composition's number, (result, mode, start)), in order) refined on ``threads``
        threads and placed in that order, the compositions ``numbers`` reported as ``stage``."""
        refined = parallel.ordered(fitted, [job for _number, job in jobs], threads)
        try:
            for number in numbers:
                check()
                tell(stage, number, total)
                for _ in range(sum(n == number for n, _job in jobs)):
                    place(next(refined))
        finally:
            refined.close()

    def place(candidate) -> None:
        """Keep a candidate, or the better of it and an earlier one at the same values (so the order matters)."""
        result, centre = candidate.result, candidate.centre
        types = E.composition_types(result.index)
        modes = found[result.index]
        for i, previous in enumerate(modes):
            if centre is not None and previous.centre is not None and refine._same(previous.centre, centre, types):
                if candidate.bic < previous.bic:
                    modes[i] = candidate
                return
        modes.append(candidate)

    def slow(index) -> bool:
        """A random cylinder (15–800 ms per evaluation, 20–30 s per least-squares fit): only its heaviest modes are
        refined, without the tempered modes and the moves (their benchmarks have no random cylinders)."""
        return any(family == "cylinder" for family, _s in E.composition_types(index))

    jobs = []  # (composition's number, (result, mode, start)), in the order they are placed
    for number, result in enumerate(results):
        started = []
        for mode in result.modes[:refine_modes] + ([] if slow(result.index) else result.explore[:explored]):
            if any(np.array_equal(mode.best, s, equal_nan=True) for s in started):
                continue
            started.append(mode.best)
            jobs.append((number, (result, mode, fit_model(mode.best, result.index, counts, law, coupling, grazing))))
    place_all(jobs, range(len(results)), "refinement", len(results))
    tell("refinement", len(results), len(results))
    def ranked_compositions():
        return sorted((r for r in results if found[r.index]), key=lambda r: min(e.bic for e in found[r.index]))

    by_score = ranked_compositions()
    moved = []  # (composition's number, (result, mode, start)): the solutions placed so far are the moves' centres
    for number, result in enumerate(by_score[:refine.RETRY_COMPOSITIONS]):
        placed = [e for e in found[result.index] if e.centre is not None]
        if not refine.MOVES or not placed or slow(result.index):
            continue
        for candidate in sorted(placed, key=lambda e: e.bic)[:refine.MOVED_SOLUTIONS]:
            tried = []
            for values in moved_starts(candidate.centre, result.index):
                start = fit_model(values, result.index, counts, law, coupling, grazing)
                residual = fit_engine.residuals(start, fit_data)
                tried.append((float(residual @ residual), len(tried), start))
            moved += [(number, (result, candidate.mode, start)) for _chi2, _i, start in sorted(tried)[:refine.MOVES]]
    place_all(moved, sorted({n for n, _job in moved}), "starts", len(by_score[:refine.RETRY_COMPOSITIONS]))
    by_score = ranked_compositions()
    for result in by_score[:refine.RETRY_COMPOSITIONS]:
        if result.modes and min(e.counts_fit.chi2_reduced for e in found[result.index]) > refine.CHI2_RETRY:
            heaviest = np.argsort(-result.weights)[:refine.SEEDS]
            seeds = tuple(fit_model(result.samples[i], result.index, counts, law, coupling, grazing) for i in heaviest)
            start = fit_model(result.modes[0].best, result.index, counts, law, coupling, grazing)
            tell("search", 0, 1)
            add(result, result.modes[0], fit_engine.fit(start, fit_data, method="global", seeds=seeds, seed=seed))
    ranked = sorted((e for modes in found.values() for e in modes), key=lambda e: e.bic)
    if not ranked:
        return []
    kept = [e for e in ranked[:refine.LOCAL_MAX] if e.bic <= ranked[0].bic + refine.DELTA_BIC]

    def finish(entries, final=False):
        groups = {}
        for place, entry in enumerate(entries):
            groups.setdefault(entry.result.index, []).append(place)
        pools = {}
        locals_ = [None] * len(entries)
        for index, places in groups.items():
            sampled = [i for i in places if entries[i].local is not None]
            if sampled:
                pool = pool_composition([entries[i].local for i in sampled])
                pools[index] = pool
                for i, local in zip(sampled, pool.locals):
                    locals_[i] = local
        by_bic = any(local is None for local in locals_) or any(p.ess < refine.ESS_FAILED for p in pools.values())
        fits = []
        bics = np.array([e.bic for e in entries])
        for i, (entry, local) in enumerate(zip(entries, locals_)):
            displayed = entry.counts_fit
            if local is not None and local.ess >= refine.ESS_FAILED:
                displayed, best_floor = refine._within_posterior(entry.result.index, displayed, entry.floor_fit,
                                                                 local, data, weighed, law, coupling, grazing)
                bics[i] = bic(best_floor)
            fits.append(displayed)

        probabilities, shares, best_bic = _comparison(groups, pools, bics, by_bic)
        if final:
            for result in results:
                result.probability = probabilities.get(result.index, 0.0)
        diagnostics = reporting.comparison_diagnostics(
            [e.result for e in entries], locals_, pools, by_bic, probabilities)
        level = counts.level
        warning_qz = counts.qz if level > 1 else measured_qz
        out = []
        for i, (entry, refined, local) in enumerate(zip(entries, fits, locals_)):
            result, model = entry.result, refined.model
            share = float(shares[i])
            weight = float(probabilities[result.index] * share)
            if weight < min_probability:
                continue
            lsq_errors, _pairs = fit_engine.errors(model, weighed)
            residuals = fit_engine.residuals(model, weighed)
            lsq_chi2 = float(residuals @ residuals) / max(1, refined.points - refined.free)
            diagnostics[i].update(sampling, **counts.substrate_record(), cut_qz=counts.qz, cut_qz_minus=counts.qz_minus,
                                  model_error_floor=model_error_floor, substrate_level=level, reflected_beam=reflected_beam,
                                  measured_qz=warning_qz, supplied_heights_nm=known_heights or {},
                                  mode_bic=float(bics[i]), composition_bic=float(best_bic[result.index]))
            extra = dict(head_log_prob=result.head_log_prob, mode_weight=share, diagnostics=diagnostics[i],
                         reflection=refine._reflection_extra(model) if counts.reflected else None,
                         network_ess=result.ess, dropped_points=data.dropped, compared_by="bic" if by_bic else "evidence")
            if result.index in pools:
                extra["composition_log_evidence"] = pools[result.index].log_evidence
            solution = refine._solution(result, (entry.mode, model, refined, local), data, fit_data, lo, hi, "lm",
                                        weight, extra, lsq=(lsq_errors, lsq_chi2), from_errors=by_bic)
            if level == 1 and measured_qz:
                solution.flags += tuple(refine.interface_flags(model, measured_qz))
            solution.flags += tuple(reporting.grazing_flags(model, warning_qz, level, reflected_beam, known_heights))
            if by_bic:
                solution.flags += ("bic",)
            out.append(solution)
        seconds = time.perf_counter() - tick
        for solution in out:
            solution.seconds = seconds
        return sorted(out, key=lambda s: -s.probability)

    for number, entry in enumerate(kept):
        check()
        tell("sampling", number, len(kept))
        if entry.centre is not None:
            tau = entry.mode.median[tau_dim]
            entry.local = local_posterior(data, entry.result.index, entry.centre, float(tau) if np.isfinite(tau) else 0.02,
                                          rng, local_samples, law, coupling,
                                          rounds=refine.SLOW_ROUNDS if slow(entry.result.index) else refine.LOCAL_ROUNDS,
                                          threads=threads)
        if partial is not None and number + 1 < len(kept):
            partial(finish(kept[:number + 1]))
    tell("sampling", len(kept), len(kept))
    return finish(kept, final=True)
