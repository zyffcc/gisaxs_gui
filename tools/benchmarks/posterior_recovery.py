"""How well the proposal network and its refinement recover the parameters of curves they have never seen.

The same held-out curves as fit1d_recovery.py (its draw() and curve(), unchanged: the independent truth model
yoneda_truth.py, seed0 20261116, two levels of physics, four compositions), and with --seed0 another set from the same
draw. For every curve, GIMaP's posterior path (src/gimap/features/fitting/domain/posterior) runs three times:
  network   the network's samples of the true composition, their median (no likelihood): what fit1d calls 'network'
  fit       the true composition: importance-sampled posterior, its modes refined by least squares; the most
            probable solution is scored
  choose    compositions among spheres and upright cylinders with a paracrystal, one or two of them (fit1d's
            'auto:sphere,vc:2'): is the most probable one the true composition?
Reported as in fit1d_recovery.py (median and 90th percentile of the error, bias, |z| ≤ 1 with the least-squares
error), and in addition the coverage of the posterior's 68 % and 95 % intervals, how often the most probable solution
of the true composition is in the truth's mode and how often that mode is listed at all (``truth_mode``), the
effective sample size and the time per curve on one core. --model-error-floor sets Predict's relative model error per
point (default: refine.MODEL_ERROR_FLOOR; 0 as before 0.2.0's calibration).
With --set extended the cases are heldout_cases.py's (three populations and the DA as well; the coupling of a case is
given to the fit, and the choice is among one to three populations).
With --truth own the same cases are drawn in Predict parameters' own model instead (cut_model as its likelihood
evaluates it: the cut's q_z, the case's coupling, no beam tail, no resolution), both levels alike: the intervals'
coverage when the model is exactly right.
usage: python tools/benchmarks/posterior_recovery.py --network PATH/NAME [--n 200] [--workers 8] [--seed0 20261116]
           [--samples N] [--local-samples N] [--shortlist N] [--refined 3] [--set fit1d|extended]
           [--truth independent|own] [--model-error-floor F] [--explored N] [--moves N] [--widths 1,2]
           [--out NAME]
  (defaults: Predict parameters' own, refine.SAMPLES, LOCAL_SAMPLES, SHORTLIST, MODEL_ERROR_FLOOR, EXPLORED, MOVES,
  infer.PROPOSAL_WIDTHS)
Writes the summary to tools/benchmarks/NAME_summary.json (default posterior_recovery) and every curve to results/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
GIMAP = HERE.parents[1]
for p in (str(HERE), str(GIMAP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import recovery_cases as fr  # noqa: E402
import heldout_cases as hc  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import infer, refine  # noqa: E402
from src.gimap.features.fitting.domain.posterior.flow import Network  # noqa: E402

KEYS = ("R", "sR", "D", "s_nn")
MODE_TOLERANCE = 0.2
OURS = {"R": "R", "sR": "sigma_R", "D": "D", "s_nn": "sigma_nn"}
FAMILY = {"sphere": "sphere", "vc": "vertical_cylinder"}
NETWORK = None
OPTIONS = {}


def _composition(case) -> int:
    """The true composition, its populations ordered by R."""
    pops = sorted(case["populations"], key=lambda p: p["R"])
    return E.composition_index([(FAMILY[p["shape"]], True) for p in pops])


def _allowed(populations: int = 2) -> list[int]:
    return E.select(("sphere", "vertical_cylinder"), populations, structure=True)


def _families(index: int) -> list[str]:
    return sorted(f for f, _s in E.composition_types(index))


def _populations(values: dict, k: int, field: str, error_field: str | None = None) -> list[dict]:
    pops = []
    for slot in range(k):
        row = {}
        for key in KEYS:
            interval = values.get(f"{OURS[key]}_{slot + 1}")
            if interval is None:
                continue
            row[key] = getattr(interval, field)
            if error_field:
                row[key + "_err"] = getattr(interval, error_field)
            row[key + "_68"] = (interval.low68, interval.high68)
            row[key + "_95"] = (interval.low95, interval.high95)
        pops.append(row)
    return pops


def _init(network_path: str, options: dict):
    global NETWORK
    NETWORK = Network.load(network_path)
    OPTIONS.update(options)
    refine.MOVES = options.get("moves", refine.MOVES)
    refine.MOVED_SOLUTIONS = options.get("moved_solutions", refine.MOVED_SOLUTIONS)
    infer.PER_COUNT = options.get("per_count", infer.PER_COUNT)
    for setting in options.get("constants", []):  # module.NAME=value, e.g. refine.LOCAL_MAX=6
        target, value = setting.split("=", 1)
        module, name = target.split(".", 1)
        owner = {"refine": refine, "infer": infer}[module]
        old = getattr(owner, name)
        setattr(owner, name, tuple(float(v) for v in value.split(",")) if isinstance(old, tuple) else type(old)(value))
    infer.PROPOSAL_WIDTHS = tuple(options.get("widths", infer.PROPOSAL_WIDTHS))


def own_curve(case) -> tuple[np.ndarray, np.ndarray]:
    """The case in Predict parameters' own model: each population's column of its likelihood (the peak at 1), the
    later ones at their share of the first one's peak, the background, Poisson counts as in recovery_cases.curve."""
    pops = sorted(case["populations"], key=lambda p: p["R"])
    data = infer.Cut.prepare(fr.Q, np.ones_like(fr.Q), np.ones_like(fr.Q), float(fr.GEOM.qz_vacuum))
    values = np.zeros(E.DIMS)
    for slot, p in enumerate(pops):
        j = slot * E.PER_SLOT
        values[j:j + E.PER_SLOT] = (p["R"], p["sR"], 2.0 * p["R"], 0.0, p["D"], p["s_nn"])
    g = E.GLOBAL.start
    values[g:g + 4] = (data.q_max / data.ratio, 2.0, 1e-6 * data.q_max, 0.003)
    columns = infer.Likelihood(data, _composition(case), coupling=case.get("coupling", "lma")).columns(values)[0]
    slot = {id(p): i for i, p in enumerate(pops)}
    total = np.zeros_like(fr.Q)
    keep = np.isin(fr.Q, data.q)
    total[keep] = sum((1.0 if k == 0 else p["share"]) * columns[slot[id(p)]]
                      for k, p in enumerate(case["populations"]))
    total[~keep] = total[keep][0]
    total += case["background"] * float(total.max())
    scale = case["peak_counts"] / float(total[fr.Q > 0.1].max())
    counts = np.random.default_rng(case["seed"]).poisson(total * scale).astype(float)
    return counts, np.sqrt(np.maximum(counts, 1.0))


def run(index: int) -> dict:
    extended = OPTIONS.get("set") == "extended"
    case = (hc.draw if extended else fr.draw)(index, OPTIONS["seed0"])
    I, s = own_curve(case) if OPTIONS.get("truth") == "own" else (hc.curve if extended else fr.curve)(case)
    coupling = case.get("coupling", "lma")
    out = dict(case=case)
    qz = float(fr.GEOM.qz_vacuum)
    truth = _composition(case)
    k = len(case["populations"])
    # network: samples of the true composition, their median
    tick = time.perf_counter()
    data = infer.Cut.prepare(fr.Q, I, s, qz)
    binned = E.bin_curve(data.q, data.intensity, data.sigma, data.q_max, data.ratio)
    h = NETWORK.summary(E.features(*(a[None] for a in binned)), E.conditions([data.ratio], [qz / data.q_max]))
    z, _ = NETWORK.sample(NETWORK.context(h, truth), truth, OPTIONS["samples"], np.random.default_rng(index))
    values = infer.physical(E.from_z(z, np.full(z.shape[0], data.ratio), truth), data.q_max)
    median = np.median(values, axis=0)
    pops = []
    for slot in range(k):
        j = slot * E.PER_SLOT
        pops.append(dict(R=median[j], sR=median[j + 1], D=median[j + 4], s_nn=median[j + 5]))
    out["network"] = dict(ok=True, populations=pops, seconds=time.perf_counter() - tick)
    for name, kwargs in (("fit", dict(compositions=[truth])),
                         ("choose", dict(allowed=_allowed(3 if extended else 2), shortlist=OPTIONS["shortlist"]))):
        tick = time.perf_counter()
        try:
            solutions = refine.predict(NETWORK, fr.Q, I, s, qz=qz, samples=OPTIONS["samples"], seed=index,
                                       coupling=coupling, local_samples=OPTIONS["local_samples"],
                                       refine_modes=OPTIONS["refined"],
                                       explored=OPTIONS.get("explored", refine.EXPLORED),
                                       model_error_floor=OPTIONS.get("floor", refine.MODEL_ERROR_FLOOR), **kwargs)
            best = solutions[0]
            k_best = len(E.COMPOSITIONS[best.composition])
            out[name] = dict(ok=True, composition=best.name, composition_index=best.composition,
                             families=_families(best.composition), chi2=best.chi2_reduced, flags=list(best.flags),
                             ess=best.ess, probability=best.probability,
                             populations=_populations(best.values, k_best, "best", "error"),
                             posterior=_populations(best.values, k_best, "median"),
                             in_set=[_families(x.composition) for x in solutions if x.probability >= 0.05],
                             alternatives=[dict(probability=x.probability, families=_families(x.composition),
                                                populations=_populations(x.values, len(E.COMPOSITIONS[x.composition]),
                                                                         "best", "error"))
                                           for x in solutions[1:] if x.probability >= 0.05])
        except Exception as exc:  # noqa: BLE001 - one failing curve must not stop the study
            out[name] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        out[name]["seconds"] = time.perf_counter() - tick
    out["truth_families"] = sorted(FAMILY[p["shape"]] for p in case["populations"])
    return out


def _errors(results, stage, level, field="populations"):
    errs = {k: [] for k in KEYS}
    zs = {k: [] for k in KEYS}
    in68 = {k: [] for k in KEYS}
    in95 = {k: [] for k in KEYS}
    for r in results:
        if level != "all" and r["case"]["level"] != level:
            continue
        got = r.get(stage) or {}
        if not got.get("ok") or len(got[field]) != len(r["case"]["populations"]):
            continue
        truth = sorted(r["case"]["populations"], key=lambda p: p["R"])
        for t, g in zip(truth, got[field]):
            for k in KEYS:
                if g.get(k) is None or not np.isfinite(g[k]):
                    continue
                errs[k].append((g[k] - t[k]) / t[k] if k in ("R", "D") else g[k] - t[k])
                e = g.get(k + "_err")
                if e and e > 0 and math.isfinite(e):
                    zs[k].append((g[k] - t[k]) / e)
                if k + "_68" in g:
                    in68[k].append(g[k + "_68"][0] <= t[k] <= g[k + "_68"][1])
                    in95[k].append(g[k + "_95"][0] <= t[k] <= g[k + "_95"][1])
    row = {}
    for k in KEYS:
        a = np.abs(np.asarray(errs[k]))
        row[k] = dict(n=int(a.size), median_abs=float(np.median(a)) if a.size else None,
                      p90_abs=float(np.percentile(a, 90)) if a.size else None,
                      bias=float(np.median(errs[k])) if a.size else None,
                      coverage_1sigma=float(np.mean(np.abs(zs[k]) <= 1)) if zs[k] else None, n_z=len(zs[k]),
                      posterior_coverage68=float(np.mean(in68[k])) if in68[k] else None,
                      posterior_coverage95=float(np.mean(in95[k])) if in95[k] else None)
    return row


def _listed(results, level):
    """How often the true composition's most probable solution is in the truth's mode (every R and D within
    ``MODE_TOLERANCE``), and how often that mode is among the solutions listed (it, or another with a probability of
    at least 0.05 and as many populations)."""
    first, listed = [], []
    for r in results:
        got = r.get("fit") or {}
        if (level != "all" and r["case"]["level"] != level) or not got.get("ok"):
            continue
        truth = sorted(r["case"]["populations"], key=lambda p: p["R"])

        def in_mode(pops):
            return len(pops) == len(truth) and all(
                abs(g[k] - t[k]) <= MODE_TOLERANCE * t[k] for t, g in zip(truth, pops) for k in ("R", "D"))

        first.append(in_mode(got["populations"]))
        listed.append(first[-1] or any(in_mode(a["populations"]) for a in got.get("alternatives", [])))
    return dict(n=len(first), first=float(np.mean(first)) if first else None,
                listed=float(np.mean(listed)) if listed else None)


def summary(results):
    table = {}
    for stage in ("network", "fit"):
        for level in ("model", "measured", "all"):
            table[f"{stage}/{level}"] = _errors(results, stage, level)
    for level in ("model", "measured", "all"):
        table[f"posterior_median/{level}"] = _errors(results, "fit", level, field="posterior")
    choose = {}
    for level in ("model", "measured", "all"):
        rows = [r for r in results if (level == "all" or r["case"]["level"] == level) and r.get("choose", {}).get("ok")]
        exact = [r["choose"]["families"] == r["truth_families"] for r in rows]
        in_set = [r["truth_families"] in r["choose"]["in_set"] for r in rows]
        choose[level] = dict(n=len(rows), exact=float(np.mean(exact)) if rows else None,
                             true_in_equivalent_set=float(np.mean(in_set)) if rows else None)
    seconds = {stage: float(np.median([r[stage]["seconds"] for r in results if stage in r]))
               for stage in ("network", "fit", "choose")}
    ess = [r["fit"]["ess"] for r in results if r.get("fit", {}).get("ok")]
    failed = {stage: int(sum(not r.get(stage, {}).get("ok") for r in results)) for stage in ("fit", "choose")}
    mode = {level: _listed(results, level) for level in ("model", "measured", "all")}
    return dict(parameters=table, composition=choose, truth_mode=mode, median_seconds=seconds, n=len(results),
                fit_ess=dict(median=float(np.median(ess)) if ess else None,
                             below_50=float(np.mean(np.asarray(ess) < 50)) if ess else None),
                failed=failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=20261116)
    ap.add_argument("--samples", type=int, default=refine.SAMPLES)
    ap.add_argument("--out", default="posterior_recovery")
    ap.add_argument("--set", default="fit1d", choices=("fit1d", "extended"))
    ap.add_argument("--local-samples", type=int, default=refine.LOCAL_SAMPLES)
    ap.add_argument("--shortlist", type=int, default=refine.SHORTLIST)
    ap.add_argument("--refined", type=int, default=3)
    ap.add_argument("--truth", default="independent", choices=("independent", "own"))
    ap.add_argument("--model-error-floor", type=float, default=refine.MODEL_ERROR_FLOOR)
    ap.add_argument("--explored", type=int, default=refine.EXPLORED)
    ap.add_argument("--moves", type=int, default=refine.MOVES)
    ap.add_argument("--moved-solutions", type=int, default=refine.MOVED_SOLUTIONS)
    ap.add_argument("--per-count", type=int, default=infer.PER_COUNT)
    ap.add_argument("--constant", action="append", default=[], help="module.NAME=value (refine or infer), repeatable")
    ap.add_argument("--widths", default=",".join(str(w) for w in infer.PROPOSAL_WIDTHS))
    a = ap.parse_args()
    t0 = time.perf_counter()
    results = []
    options = dict(seed0=a.seed0, samples=a.samples, set=a.set, local_samples=a.local_samples, shortlist=a.shortlist,
                   refined=a.refined, truth=a.truth, floor=a.model_error_floor,
                   explored=a.explored, moves=a.moves, widths=[float(w) for w in a.widths.split(",")],
                   moved_solutions=a.moved_solutions, per_count=a.per_count, constants=a.constant)
    with Pool(min(a.workers, a.n), initializer=_init, initargs=(a.network, options)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, range(a.n)), 1):
            results.append(r)
            c = r["case"]
            print(f"[{i}/{a.n}] {c['composition']:13s} {c['level']:8s} "
                  + " ".join(f"{k}:{'ok' if r[k].get('ok') else 'ERR'} {r[k]['seconds']:.1f}s"
                             for k in ("network", "fit", "choose"))
                  + f" chose {r['choose'].get('composition')}", flush=True)
    s = summary(results)
    s["network"] = Path(a.network).name
    s["network_sha256"] = Network.load(a.network).meta["sha256"]
    s["seed0"] = a.seed0
    s["set"] = a.set
    s["truth"] = a.truth
    s["settings"] = options
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / f"{a.out}_summary.json").write_text(json.dumps(s, indent=1))
    (HERE / "results" / f"{a.out}.json").write_text(json.dumps(dict(summary=s, results=results,
                                                                     seconds=time.perf_counter() - t0),
                                                                indent=1, default=str))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
