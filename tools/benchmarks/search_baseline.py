"""The baseline without a network: Fitting's *Search the ranges, then refine* on the held-out curves.

The same 200 curves as posterior_recovery.py (recovery_cases.py; the independent forward model yoneda_truth.py), the
same model (cut_model, sizes by number, LMA, the Born term at the cut's q_z, the beam tail free) and the true
composition given, but no proposal: every population starts at Fitting's defaults and is searched over Fitting's
default ranges (differential evolution, fit_engine.fit(method="global")), then refined by least squares. --budget is
the number of model evaluations of the search (Fitting's own default: 4000).
usage: python tools/benchmarks/search_baseline.py [--n 200] [--workers 8] [--budget 4000] [--out search_baseline]
Writes tools/benchmarks/NAME_summary.json and every curve to results/NAME.json.
"""
from __future__ import annotations

import argparse
import json
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

import posterior_recovery as pr  # noqa: E402
import recovery_cases as fr  # noqa: E402
from src.gimap.features.fitting.domain import fit_engine, fit_model as fm  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import infer  # noqa: E402

BUDGET = 4000
SEED0 = 20261116


def start_model(composition: int, data) -> fm.FitModel:
    """Fitting's default model of the composition at the cut's q_z: default values and ranges, the tail free."""
    components = tuple(fm.new_component(family, structure=structure, law="number_gauss")
                       for family, structure in E.composition_types(composition))
    model = fm.FitModel(components)
    model = model.with_parameter(("globals", "res_amplitude"), value=1.0, free=True)
    for key in ("res_width", "res_exponent"):
        model = model.with_parameter(("globals", key), free=True)
    if data.qz > 0:
        alpha_i, alpha_f = data.angles()
        for key, value in (("alpha_i", alpha_i), ("alpha_f", alpha_f), ("wavelength", 0.1)):
            model = model.with_parameter(("globals", key), value=value)
    return model


def run(job) -> dict:
    index, budget = job
    case = fr.draw(index, SEED0)
    intensity, sigma = fr.curve(case)
    composition = pr._composition(case)
    data = infer.Cut.prepare(fr.Q, intensity, sigma, float(fr.GEOM.qz_vacuum), None)
    fit_data = fit_engine.FitData.prepare(data.q, data.intensity, data.sigma)
    out = dict(case=case)
    tick = time.perf_counter()
    try:
        found = fit_engine.fit(start_model(composition, data), fit_data, method="global", max_evaluations=budget,
                               seed=index)
        found = fit_engine.fit(found.model, fit_data, method="local")
        pops = [dict(R=c.value("R"), sR=c.value("sigma_R"), D=c.value("D"), s_nn=c.value("sigma_nn"))
                for c in sorted(found.model.components, key=lambda c: c.value("R"))]
        out["fit"] = dict(ok=True, populations=pops, chi2=float(found.chi2_reduced))
    except Exception as exc:  # noqa: BLE001 - one failing curve must not stop the study
        out["fit"] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    out["fit"]["seconds"] = time.perf_counter() - tick
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--out", default="search_baseline")
    a = ap.parse_args()
    fr.set_truth("born_qz", "si")
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(run, [(i, a.budget) for i in range(a.n)], chunksize=1))
    results.sort(key=lambda r: r["case"]["index"])
    table = {level: pr._errors(results, "fit", level) for level in ("model", "measured", "all")}
    ok = [r for r in results if r["fit"].get("ok")]
    summary = dict(n=len(results), failed=len(results) - len(ok), budget=a.budget, parameters=table,
                   median_seconds=float(np.median([r["fit"]["seconds"] for r in results])),
                   median_chi2={level: float(np.median([r["fit"]["chi2"] for r in ok
                                                        if level == "all" or r["case"]["level"] == level]))
                                for level in ("model", "measured", "all")})
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.out}.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    (HERE / f"{a.out}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for level in ("model", "measured"):
        row = table[level]
        print(level, {k: (None if row[k]["median_abs"] is None else round(row[k]["median_abs"], 4),
                          None if row[k]["p90_abs"] is None else round(row[k]["p90_abs"], 4)) for k in pr.KEYS},
              "chi2", round(summary["median_chi2"][level], 2))
    print("seconds per curve", round(summary["median_seconds"], 1), "failed", summary["failed"])


if __name__ == "__main__":
    main()
