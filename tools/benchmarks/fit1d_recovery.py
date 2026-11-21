"""M3: how well fit1d recovers the parameters of curves it has never seen.

Held-out curves come from the independent truth model (yoneda_truth.py, no code shared with fit1d), drawn at random
inside the range fit1d is meant for, with a fresh seed. Two levels of physics:
  model     Born at the vacuum q_z of the Yoneda row, LMA, Hosemann paracrystal, number-weighted Gaussian sizes,
            Poisson counts and a flat background: the physics of fit1d itself (an in-model test of network and fit)
  measured  + Gaussian resolution (sigma 0.005 nm^-1), a beam tail at q = 0 and resolvable roughness at low q:
            what a measured cut adds and fit1d describes with its low-q term
Compositions: one sphere, one upright cylinder, sphere + sphere, sphere + cylinder (two size scales).
For every curve fit1d runs three times with GIMaP's default network ('general'):
  network   policy 'preview' with the true composition: network proposals and NNLS amplitudes only
  fit       policy 'auto' with the true composition: the full pipeline (proposals, LM, search on signals)
  choose    composition 'auto:sphere,vc:2': does the BIC rule pick the true composition?
Reported per parameter (R, sigma_R/R, D, sigma_nn/D; populations matched by R): median and 90th percentile of the
relative error, bias, the share of |z| <= 1 with the reported 1-sigma error (coverage), and the time per curve.
With --set extended (and --seed0) the cases are heldout_cases.py's: three populations and the DA as well (fit1d's
structure factor is the LMA's whatever the case's coupling; 'choose' is then among up to three populations).
usage: python tools/benchmarks/fit1d_recovery.py [--n 200] [--workers 10] [--quick] [--set fit1d|extended] [--seed0 N]
The summary of the run quoted in Physics and Methods, section 9.12, is history/fit1d_recovery_summary.json; every curve is in results/ (not committed).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GIMAP = HERE.parents[1]                                          # the repository
KIT = Path(os.environ.get("GISAXS1D_KIT_HOME") or GIMAP / "utils" / "gisaxs1d_gui_kit")  # removed from GIMaP in 0.2.0
KIT_APP = KIT / "app"
MODEL_DIR = KIT / "models" / "general"
for p in (str(HERE), str(KIT_APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yoneda_truth as yt  # noqa: E402

from recovery_cases import COMPOSITIONS, GEOM, Q, curve, draw  # noqa: E402,F401 - the cases, also from here

KEYS = ("R", "sR", "D", "s_nn")
FIT_KEYS = dict(R=("R_nm", "R_nm_err"), sR=("sigma_R_rel", "sigma_R_rel_err"), D=("D_nm", "D_nm_err"),
                s_nn=("w_rel", "w_rel_err"))


def _populations(result):
    best = result["results"][0]
    pops = []
    for p in best["populations"]:
        row = dict(shape="sphere" if p.get("shape", p.get("type")) in ("sphere", 1) else "vc")
        for key, (value, error) in FIT_KEYS.items():
            row[key], row[key + "_err"] = p.get(value), p.get(error)
        pops.append(row)
    return best, sorted(pops, key=lambda r: r["R"] or 0.0)


SET = {"name": "fit1d", "seed0": 20261116}


def _set(name, seed0):
    SET.update(name=name, seed0=seed0)


def run(index):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # By name, as GIMaP loads the kit: an optional folder beside the program, not a package of requirements.txt.
    fit1d = importlib.import_module("research.fit1d.api").fit1d
    if SET["name"] == "extended":
        import heldout_cases
        case = heldout_cases.draw(index, SET["seed0"])
        I, s = heldout_cases.curve(case)
    else:
        case = draw(index, SET["seed0"])
        I, s = curve(case)
    auto = "auto:sphere,vc:3" if SET["name"] == "extended" else "auto:sphere,vc:2"
    out = dict(case=case)
    common = dict(mode="gisaxs", qz=float(GEOM.qz_vacuum), workers=1, seed=0, model_dir=str(MODEL_DIR))
    for name, kw in (("network", dict(composition=case["composition"], policy="preview")),
                     ("fit", dict(composition=case["composition"], policy="auto")),
                     ("choose", dict(composition=auto, policy="auto"))):
        t0 = time.perf_counter()
        try:
            res = fit1d(Q, I, s, **kw, **common)
            best, pops = _populations(res)
            flags = best.get("flags") or {}
            out[name] = dict(ok=True, populations=pops, composition=best.get("composition_text"),
                             chi2=best.get("chi2_per_point"), raised=list(flags.get("raised") or []),
                             indistinguishable=bool(best.get("indistinguishable")),
                             in_set=[r.get("composition_text") for r in res["results"] if r.get("in_candidate_set")])
        except Exception as exc:  # noqa: BLE001 - one failing fit must not stop the study
            out[name] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        out[name]["seconds"] = time.perf_counter() - t0
    return out


def summary(results):
    """Errors per parameter and stage, composition choice, coverage, time."""
    table = {}
    for stage in ("network", "fit"):
        for level in ("model", "measured", "all"):
            errs = {k: [] for k in KEYS}
            zs = {k: [] for k in KEYS}
            for r in results:
                if level != "all" and r["case"]["level"] != level:
                    continue
                got = r.get(stage) or {}
                if not got.get("ok") or len(got["populations"]) != len(r["case"]["populations"]):
                    continue
                truth = sorted(r["case"]["populations"], key=lambda p: p["R"])
                for t, g in zip(truth, got["populations"]):
                    for k in KEYS:
                        if g.get(k) is None:
                            continue
                        rel = (g[k] - t[k]) / t[k] if k in ("R", "D") else g[k] - t[k]
                        errs[k].append(rel)
                        e = g.get(k + "_err")
                        if e and e > 0 and math.isfinite(e):
                            zs[k].append((g[k] - t[k]) / e)
            row = {}
            for k in KEYS:
                a = np.abs(np.asarray(errs[k]))
                row[k] = dict(n=int(a.size), median_abs=float(np.median(a)) if a.size else None,
                              p90_abs=float(np.percentile(a, 90)) if a.size else None,
                              bias=float(np.median(errs[k])) if a.size else None,
                              coverage_1sigma=float(np.mean(np.abs(zs[k]) <= 1)) if zs[k] else None, n_z=len(zs[k]))
            table[f"{stage}/{level}"] = row
    choose = {}
    for level in ("model", "measured", "all"):
        rows = [r for r in results if (level == "all" or r["case"]["level"] == level) and r.get("choose", {}).get("ok")]
        if any("coupling" in r["case"] for r in rows):  # the extended set: the same families in any order
            key = lambda text: sorted(str(text or "").split(","))  # noqa: E731
            exact = [key(r["choose"]["composition"]) == key(r["case"]["composition"]) for r in rows]
            in_set = [key(r["case"]["composition"]) in [key(t) for t in r["choose"].get("in_set") or []] for r in rows]
        else:
            exact = [r["choose"]["composition"] == r["case"]["composition"] for r in rows]
            in_set = [r["case"]["composition"] in (r["choose"].get("in_set") or []) for r in rows]
        choose[level] = dict(n=len(rows), exact=float(np.mean(exact)) if rows else None,
                             true_in_equivalent_set=float(np.mean(in_set)) if rows else None)
    seconds = {stage: float(np.median([r[stage]["seconds"] for r in results if stage in r]))
               for stage in ("network", "fit", "choose")}
    return dict(parameters=table, composition=choose, median_seconds=seconds, n=len(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--set", default="fit1d", choices=("fit1d", "extended"))
    ap.add_argument("--seed0", type=int, default=20261116)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = 8 if a.quick else a.n
    t0 = time.perf_counter()
    results = []
    with Pool(min(a.workers, n), initializer=_set, initargs=(a.set, a.seed0)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, range(n)), 1):
            results.append(r)
            c = r["case"]
            print(f"[{i}/{n}] {c['composition']:13s} {c['level']:8s} "
                  + " ".join(f"{k}:{'ok' if r[k].get('ok') else 'ERR'} {r[k]['seconds']:.1f}s" for k in ("network", "fit", "choose"))
                  + f" chose {r['choose'].get('composition')}", flush=True)
    s = summary(results)
    stem = a.out or "fit1d_recovery"
    name = f"{stem}_quick.json" if a.quick else f"{stem}.json"
    (HERE / "results").mkdir(exist_ok=True)
    if not a.quick:
        (HERE / f"{stem}_summary.json").write_text(json.dumps(s, indent=1))
    (HERE / "results" / name).write_text(json.dumps(dict(summary=s, results=results, q=[float(Q[0]), float(Q[-1]), len(Q)],
                                                         geometry=dict(wavelength_nm=GEOM.wavelength_nm,
                                                                       alpha_i_deg=GEOM.alpha_i_deg,
                                                                       qz_vacuum=float(GEOM.qz_vacuum)),
                                                         seconds=time.perf_counter() - t0), indent=1, default=str))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
