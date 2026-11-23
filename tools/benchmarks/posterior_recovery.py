"""How well the proposal network and its refinement recover the parameters of curves they have never seen.

By default the independent truth model draws the 300 cuts of the named fresh set. The legacy-first,
legacy-extended and legacy-third sets replay the archived draws, Poisson seeds and beam-tail convention.
By default each curve goes through the network, fit and choose stages. --stages selects any of these or auto:
  network   the network's samples of the true composition, their median (no likelihood): what fit1d calls 'network'
  fit       the true composition: importance-sampled posterior, its modes refined by least squares; the most
            probable solution is scored
  choose    compositions among spheres and upright cylinders with a paracrystal, one or two of them (fit1d's
            'auto:sphere,vc:2'): is the most probable one the true composition?
  auto      the automatic analysis's choice among one or two populations of spheres and upright cylinders, with
            or without a paracrystal. Level 2 evaluates the Born term at the cut's vacuum q_z (the default). Level 4
            takes the angles, wavelength and known substrate. Level 1 fits at q_z = 0, with the cut's q_z for flags.
For choose and auto the head's candidates are stored, and whether they include the true radius-ordered composition.
The winner has the greatest total weight over all retained modes of its composition, including paracrystal status.
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
With --scattering dwba the independent truth adds the waves the surface reflects (the four DWBA terms of
yoneda_truth.py), on bare Si or with --substrate ps100 on 100 nm polystyrene on Si (recovery_cases.SUBSTRATES); the
same particle draws and Poisson seeds for a given set. The expected counts and noise realizations depend on the truth.
--cut says what Predict is given of the cut: its q_z = k_i + k_f ('born', the default: the Born term, as the 1D Predict
window's Cut qz), both angles ('two_path': k_i − k_f too, spheres with the exit-reflected path and a free effective
reflection coefficient, as with the angles in Model), both angles and the known substrate ('dwba': all four
paths), or nothing ('eia': the effective interface).
usage: python tools/benchmarks/posterior_recovery.py --network PATH/NAME [--n 200] [--workers 8] [--seed0 20261116]
           [--samples N] [--local-samples N] [--shortlist N] [--refined 3] [--set SET]
           [--truth independent|own] [--model-error-floor F] [--explored N] [--moves N] [--widths 1,2]
           [--scattering born_qz|dwba] [--substrate si|ps100] [--cut born|two_path|dwba|eia] [--out NAME]
           [--stages network,fit,choose,auto] [--auto] [--auto-level 1|2|4]
  (defaults: Predict parameters' own, refine.SAMPLES, LOCAL_SAMPLES, SHORTLIST, MODEL_ERROR_FLOOR, EXPLORED, MOVES,
  infer.PROPOSAL_WIDTHS)
Writes NAME_summary.json and results/NAME.json in --output-dir (default tools/benchmarks).
"""
from __future__ import annotations

import argparse
import hashlib
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
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
GIMAP = HERE.parents[1]
for p in (str(HERE), str(GIMAP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import recovery_cases as fr  # noqa: E402
import heldout_cases as hc  # noqa: E402
from case_streams import SETS  # noqa: E402
from composition_scores import LIST_THRESHOLD, METHOD, archive_weights, summarize_compositions  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import infer, refine  # noqa: E402
from src.gimap.features.fitting.domain.posterior.flow import Network  # noqa: E402

KEYS = ("R", "sR", "D", "s_nn")
STAGES = ("network", "fit", "choose", "auto")
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
    global NETWORK, OPTIONS
    NETWORK = Network.load(network_path)
    OPTIONS = dict(options)
    if "stream_scheme" not in options and options.get("set", "fit1d") in ("fit1d", "extended"):
        # Settings stored before named streams identify the legacy generators.
        OPTIONS["set"] = "legacy-extended" if options.get("set") == "extended" else "legacy-first"
    # Older benchmark records have only the --auto switch and used level 1.
    OPTIONS.setdefault("stages", STAGES if options.get("auto") else STAGES[:-1])
    OPTIONS.setdefault("auto_level", 1 if options.get("auto") and "stages" not in options else 2)
    fr.set_truth(options.get("scattering", "born_qz"), options.get("substrate", "si"))
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
    qz, qz_minus = _cut()
    data = infer.Cut.prepare(fr.Q, np.ones_like(fr.Q), np.ones_like(fr.Q), qz, qz_minus, **_known_substrate())
    values = np.zeros(E.DIMS)
    for slot, p in enumerate(pops):
        j = slot * E.PER_SLOT
        values[j:j + E.PER_SLOT] = (p["R"], p["sR"], 2.0 * p["R"], 0.0, p["D"], p["s_nn"])
    g = E.GLOBAL.start
    values[g:g + 4] = (data.q_max / data.ratio, 2.0, 1e-6 * data.q_max, 0.003)
    if data.reflected:  # the substrate's own reflection at the exit angle
        r = fr.exit_reflection()
        values[g + 4:g + 6] = (r.real, r.imag)
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


def _cut() -> tuple[float, float | None]:
    """(q_z, k_i − k_f or None) Predict is given (``--cut``)."""
    kind = OPTIONS.get("cut", "born")
    if kind == "eia":
        return 0.0, None
    return float(fr.GEOM.qz_vacuum), (fr.qz_minus() if kind == "two_path" else None)


def _known_substrate(kind=None) -> dict:
    """Inputs a user could supply: the independent simulator's optical constants, wavelength and measured angles."""
    if (kind or OPTIONS.get("cut")) != "dwba":
        return {}
    geom = fr.truth_geometry()
    delta, beta = geom.index()
    base = dict(material="custom", delta=float(delta), beta=float(beta))
    if geom.film_thickness:
        base["layers"] = [dict(thickness_nm=geom.film_thickness, delta=geom.film_delta, beta=geom.film_beta)]
    return dict(substrate=base, grazing=dict(wavelength=geom.wavelength_nm, alpha_i=geom.alpha_i_deg,
                                             alpha_f=float(geom.alpha_f)))


def run(index: int, *, fit_only: bool = False) -> dict:
    """Evaluate one cut. The release regression needs only the fit with the given composition."""
    set_name = OPTIONS["set"]
    extended = set_name in ("extended", "legacy-extended")
    case = (hc.draw if extended else fr.draw)(index, OPTIONS["seed0"], set_name=set_name)
    I, s = own_curve(case) if OPTIONS.get("truth") == "own" else fr.curve(case)
    coupling = case.get("coupling", "lma")
    out = dict(case=case)
    qz, qz_minus = _cut()
    truth = _composition(case)
    out["truth_composition_index"] = truth
    k = len(case["populations"])
    selected = ("fit",) if fit_only else OPTIONS["stages"]
    if "network" in selected:
        # network: samples of the true composition, their median
        tick = time.perf_counter()
        data = infer.Cut.prepare(fr.Q, I, s, qz, qz_minus, **_known_substrate())
        binned = E.bin_curve(data.q, data.intensity, data.sigma, data.q_max, data.ratio)
        h = NETWORK.summary(E.features(*(a[None] for a in binned)), infer.conditions(data))
        z, _ = NETWORK.sample(NETWORK.context(h, truth), truth, OPTIONS["samples"], np.random.default_rng(index))
        values = infer.physical(E.from_z(z, np.full(z.shape[0], data.ratio), truth), data.q_max)
        median = np.median(values, axis=0)
        pops = []
        for slot in range(k):
            j = slot * E.PER_SLOT
            pops.append(dict(R=median[j], sR=median[j + 1], D=median[j + 4], s_nn=median[j + 5]))
        out["network"] = dict(ok=True, populations=pops, seconds=time.perf_counter() - tick)
    stages = []
    if "fit" in selected:
        stages.append(("fit", dict(compositions=[truth], **_known_substrate()), qz, qz_minus))
    if "choose" in selected:
        stages.append(("choose", dict(allowed=_allowed(3 if extended else 2), shortlist=OPTIONS["shortlist"], **_known_substrate()),
                       qz, qz_minus))
    if "auto" in selected:
        cut_qz = float(fr.GEOM.qz_vacuum)
        auto_options = dict(allowed=E.select(("sphere", "vertical_cylinder"), 2), reflected_beam=True)
        auto_qz = cut_qz
        if OPTIONS["auto_level"] == 1:
            auto_qz = 0.0
            auto_options["measured_qz"] = cut_qz
        elif OPTIONS["auto_level"] == 4:
            auto_options.update(_known_substrate("dwba"))
        stages.append(("auto", auto_options, auto_qz, None))
    truth_families = sorted(FAMILY[p["shape"]] for p in case["populations"])
    for name, kwargs, qz_stage, qz_minus_stage in stages:
        tick = time.perf_counter()
        candidates = None
        if "allowed" in kwargs:  # the head's candidates, as infer.infer chooses them
            cut_options = {key: kwargs[key] for key in ("substrate", "grazing") if key in kwargs}
            data_stage = infer.Cut.prepare(fr.Q, I, s, qz_stage, qz_minus_stage, **cut_options)
            binned_stage = E.bin_curve(data_stage.q, data_stage.intensity, data_stage.sigma, data_stage.q_max,
                                       data_stage.ratio)
            h_stage = NETWORK.summary(E.features(*(a[None] for a in binned_stage)), infer.conditions(data_stage))
            candidates = [int(c) for c, _p in infer.candidates(NETWORK, h_stage, kwargs["allowed"],
                                                               kwargs.get("shortlist", refine.SHORTLIST))]
        try:
            solutions = refine.predict(NETWORK, fr.Q, I, s, qz=qz_stage, qz_minus=qz_minus_stage,
                                       samples=OPTIONS["samples"],
                                       # Every positive mode, including subnormals. Predict computes a mode's
                                       # share within its composition only for modes passing this threshold.
                                       seed=index, min_probability=np.nextafter(0.0, 1.0),
                                       coupling=coupling, local_samples=OPTIONS["local_samples"],
                                       refine_modes=OPTIONS["refined"],
                                       explored=OPTIONS.get("explored", refine.EXPLORED),
                                       model_error_floor=OPTIONS.get("floor", refine.MODEL_ERROR_FLOOR), **kwargs)
            best = solutions[0]
            k_best = len(E.COMPOSITIONS[best.composition])
            out[name] = dict(ok=True, composition=best.name, composition_index=best.composition,
                             families=_families(best.composition), chi2=best.chi2_reduced, flags=list(best.flags),
                             ess=best.ess, probability=best.probability,
                             reflection=[float(abs(refine.reflection(best.model))),
                                         float(np.angle(refine.reflection(best.model)))],
                             populations=_populations(best.values, k_best, "best", "error"),
                             posterior=_populations(best.values, k_best, "median"),
                             **archive_weights(solutions, candidates if candidates is not None else [truth]),
                             in_set=[_families(x.composition) for x in solutions if x.probability >= LIST_THRESHOLD],
                             alternatives=[dict(probability=x.probability, families=_families(x.composition),
                                                populations=_populations(x.values, len(E.COMPOSITIONS[x.composition]),
                                                                         "best", "error"))
                                           for x in solutions[1:] if x.probability >= LIST_THRESHOLD])
            if kwargs.get("substrate"):
                # Level 4 fixes both coefficients. The free exit-reflection parameter is absent.
                del out[name]["reflection"]
                out[name]["substrate_level"] = best.extras["diagnostics"]["substrate_level"]
                out[name]["substrate"] = kwargs["substrate"]
                out[name]["grazing"] = kwargs["grazing"]
        except Exception as exc:  # noqa: BLE001 - one failing curve must not stop the study
            out[name] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        out[name]["seconds"] = time.perf_counter() - tick
        if candidates is not None:
            out[name]["candidates"] = candidates
            out[name]["truth_in_candidates"] = truth in candidates
    out["truth_families"] = truth_families
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
    stages = [stage for stage in STAGES if any(stage in r for r in results)]
    levels = ("model", "measured", "all")
    table = {}
    for stage in ("network", "fit", "auto"):
        if stage in stages:
            for level in levels:
                table[f"{stage}/{level}"] = _errors(results, stage, level)
    if "fit" in stages:
        for level in levels:
            table[f"posterior_median/{level}"] = _errors(results, "fit", level, field="posterior")
    choose, auto = {}, {}
    for stage, target in (("choose", choose), ("auto", auto)):
        for level in levels:
            rows = [r for r in results if (level == "all" or r["case"]["level"] == level) and stage in r]
            if rows:
                target[level] = summarize_compositions(rows, stage)
    seconds = {stage: float(np.median([r[stage]["seconds"] for r in results if stage in r]))
               for stage in stages}
    ess = [r["fit"]["ess"] for r in results if r.get("fit", {}).get("ok")]
    failed = {stage: sum(not r[stage]["ok"] for r in results if stage in r)
              for stage in stages if stage != "network"}
    mode = {level: _listed(results, level) for level in levels} if "fit" in stages else {}
    return dict(parameters=table, composition_method=METHOD, composition=choose, composition_auto=auto or None,
                truth_mode=mode, median_seconds=seconds, n=len(results),
                fit_ess=dict(median=float(np.median(ess)) if ess else None,
                             below_50=float(np.mean(np.asarray(ess) < 50)) if ess else None),
                failed=failed)


def _code_state() -> dict:
    """The commit of the run, the git tree of the fitting code (the same tree at two commits is the same code) and
    whether that code had uncommitted changes."""
    import subprocess

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=HERE.parent.parent, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    folder = "src/gimap/features/fitting/domain"
    return {"commit": git("rev-parse", "--short", "HEAD"), "fitting_tree": git("rev-parse", f"HEAD:{folder}"),
            "fitting_code_modified": bool(git("status", "--porcelain", "--", folder))}


def _stages(value):
    names = [name.strip() for name in value.split(",")]
    if any(name not in STAGES for name in names):
        raise argparse.ArgumentTypeError("choose comma-separated stages from " + ", ".join(STAGES))
    return [stage for stage in STAGES if stage in names]


def _source_digest():
    source = hashlib.sha256()
    for path in sorted([*(GIMAP / "src/gimap/features/fitting/domain").rglob("*.py"),
                        *(HERE / name for name in ("posterior_recovery.py", "yoneda_truth.py", "recovery_cases.py",
                                                  "heldout_cases.py", "case_streams.py", "composition_scores.py"))]):
        source.update(path.relative_to(GIMAP).as_posix().encode())
        source.update(path.read_bytes())
    return source.hexdigest()


def _checkpoint(folder, options, network_sha256, source_sha256):
    """Keep finished draws across long runs. Reuse only identical inputs, truth, weights and fitting code."""
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(settings=options, network_sha256=network_sha256, source_sha256=source_sha256,
                python=sys.version, numpy=np.__version__, scipy=scipy.__version__)
    manifest = folder / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != meta:
            raise ValueError(f"The checkpoint inputs changed. Use a new --checkpoint folder: {folder}")
    else:
        manifest.write_text(json.dumps(meta, indent=1))
    return {int(p.stem): json.loads(p.read_text()) for p in folder.glob("[0-9]*.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--n", type=int, help="number of cuts (default: full named set)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, help="override the named set's root seed")
    ap.add_argument("--samples", type=int, default=refine.SAMPLES)
    ap.add_argument("--out", help="output basename (default: posterior_SET_rerun)")
    ap.add_argument("--output-dir", type=Path, default=HERE)
    ap.add_argument("--checkpoint", type=Path, help="save each completed draw here and resume identical runs")
    ap.add_argument("--set", default="fresh", choices=tuple(SETS))
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
    ap.add_argument("--scattering", default="born_qz", choices=("born_qz", "dwba"))
    ap.add_argument("--substrate", default="si", choices=tuple(fr.SUBSTRATES))
    ap.add_argument("--stages", type=_stages, default="network,fit,choose",
                    help="comma-separated list of network,fit,choose,auto (default: network,fit,choose)")
    ap.add_argument("--auto", action="store_true", help="add auto to the selected stages")
    ap.add_argument("--auto-level", type=int, choices=(1, 2, 4), default=2,
                    help="auto fit level: 1 for q_z = 0, 2 for vacuum q_z (default), 4 for the known substrate")
    ap.add_argument("--cut", default="born", choices=("born", "two_path", "dwba", "eia"),
                    help="what Predict is given: vacuum q_z, both angles, both angles and substrate, or nothing")
    a = ap.parse_args()
    default_n, default_seed = SETS[a.set]
    a.n = default_n if a.n is None else a.n
    a.seed0 = default_seed if a.seed0 is None else a.seed0
    if a.n < 1 or a.workers < 1 or a.seed0 < 0:
        ap.error("--n and --workers must be positive and --seed0 nonnegative")
    a.out = a.out or f"posterior_{a.set}_rerun"
    if a.auto and "auto" not in a.stages:
        a.stages.append("auto")
    t0 = time.perf_counter()
    results = []
    options = dict(seed0=a.seed0, samples=a.samples, set=a.set, local_samples=a.local_samples, shortlist=a.shortlist,
                   stream_scheme="legacy-offset" if a.set.startswith("legacy-") else "seedsequence-v1",
                   refined=a.refined, truth=a.truth, floor=a.model_error_floor,
                   explored=a.explored, moves=a.moves, widths=[float(w) for w in a.widths.split(",")],
                   moved_solutions=a.moved_solutions, per_count=a.per_count, constants=a.constant,
                   scattering=a.scattering, substrate=a.substrate, cut=a.cut, stages=a.stages,
                   auto="auto" in a.stages, auto_level=a.auto_level)
    network_sha256 = Network.load(a.network).meta["sha256"]
    source_sha256 = _source_digest()
    prior = _checkpoint(a.checkpoint, options, network_sha256, source_sha256) if a.checkpoint else {}
    results.extend(prior[i] for i in range(a.n) if i in prior)
    remaining = [i for i in range(a.n) if i not in prior]
    with Pool(min(a.workers, a.n), initializer=_init, initargs=(a.network, options)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, remaining), len(results) + 1):
            results.append(r)
            c = r["case"]
            if a.checkpoint and all(r[stage]["ok"] for stage in a.stages):
                path = a.checkpoint / f"{c['index']:04d}.json"
                partial = path.with_suffix(".tmp")
                partial.write_text(json.dumps(r, default=str))
                partial.replace(path)
            print(f"[{i}/{a.n}] {c['composition']:13s} {c['level']:8s} "
                  + " ".join(f"{k}:{'ok' if r[k].get('ok') else 'ERR'} {r[k]['seconds']:.1f}s"
                             for k in a.stages)
                  + "".join(f" {k} leading solution: {r[k]['composition']}" for k in ("choose", "auto")
                            if "composition" in r.get(k, {})), flush=True)
    results.sort(key=lambda r: r["case"]["index"])
    s = summary(results)
    s["network"] = Path(a.network).name
    s["network_sha256"] = network_sha256
    s["code"] = dict(_code_state(), source_sha256=source_sha256)
    s["seed0"] = a.seed0
    s["set"] = a.set
    s["truth"] = a.truth
    s["settings"] = options
    (a.output_dir / "results").mkdir(parents=True, exist_ok=True)
    (a.output_dir / f"{a.out}_summary.json").write_text(json.dumps(s, indent=1))
    (a.output_dir / "results" / f"{a.out}.json").write_text(json.dumps(dict(summary=s, results=results,
                                                                     seconds=time.perf_counter() - t0),
                                                                indent=1, default=str))
    print(json.dumps(s, indent=1))
    return int(any(s["failed"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
