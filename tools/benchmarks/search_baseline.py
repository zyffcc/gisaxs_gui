"""The baseline without a network: Fitting's *Search the ranges, then refine* on the held-out curves.

The same curves as posterior_recovery.py on the selected recovery_cases set (independent forward model yoneda_truth.py), the
same model (cut_model, sizes by number, LMA, the Born term at the cut's q_z, the beam tail free) and the true
composition given, but no proposal: every population starts at Fitting's defaults and is searched over Fitting's
default ranges (differential evolution, fit_engine.fit(method="global")), then refined by least squares. --budget is
the number of model evaluations of the search (Fitting's own default: 4000).
usage: python tools/benchmarks/search_baseline.py [--set fit1d] [--seed0 SEED] [--n 200] [--workers 8]
           [--budget 4000] [--out NAME]
Writes tools/benchmarks/NAME_summary.json and every curve to results/NAME.json. NAME defaults to search_baseline_SET.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
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
from case_streams import SETS  # noqa: E402
from src.gimap.features.fitting.domain import fit_engine, fit_model as fm  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import infer  # noqa: E402

BUDGET = 4000
RECOVERY_SETS = tuple(name for name in SETS if name not in ("extended", "legacy-extended"))


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


def run(index, *, budget, set_name, seed0) -> dict:
    case = fr.draw(index, seed0, set_name=set_name)
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fit1d", choices=RECOVERY_SETS)
    ap.add_argument("--seed0", type=int, help="override the named set's root seed")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--out", help="output basename (default: search_baseline_SET)")
    a = ap.parse_args(argv)
    seed0 = SETS[a.set][1] if a.seed0 is None else a.seed0
    if min(a.n, a.workers, a.budget) < 1 or seed0 < 0:
        ap.error("--n, --workers and --budget must be positive and --seed0 nonnegative")
    name = a.out or f"search_baseline_{a.set}"
    code = dict(pr._code_state(), source_sha256=pr._source_digest())
    fr.set_truth("born_qz", "si")
    worker = partial(run, budget=a.budget, set_name=a.set, seed0=seed0)
    if a.workers == 1:
        results = list(map(worker, range(a.n)))
    else:
        with Pool(min(a.workers, a.n)) as pool:
            results = list(pool.imap_unordered(worker, range(a.n)))
    results.sort(key=lambda r: r["case"]["index"])
    table = {level: pr._errors(results, "fit", level) for level in ("model", "measured", "all")}
    ok = [r for r in results if r["fit"].get("ok")]
    chi2 = {}
    for level in ("model", "measured", "all"):
        values = [r["fit"]["chi2"] for r in ok if level == "all" or r["case"]["level"] == level]
        chi2[level] = float(np.median(values)) if values else None
    summary = dict(set=a.set, seed0=seed0, stream_scheme="legacy-offset" if a.set.startswith("legacy-") else "seedsequence-v1",
                   code=code, n=len(results), failed=len(results) - len(ok), budget=a.budget, parameters=table,
                   median_seconds=float(np.median([r["fit"]["seconds"] for r in results])),
                   median_chi2=chi2)
    (HERE / "results").mkdir(exist_ok=True)
    archive = dict(summary=summary, results=results)
    (HERE / "results" / f"{name}.json").write_text(json.dumps(archive, indent=1, default=float), encoding="utf-8")
    (HERE / f"{name}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for level in ("model", "measured"):
        row = table[level]
        print(level, {k: (None if row[k]["median_abs"] is None else round(row[k]["median_abs"], 4),
                          None if row[k]["p90_abs"] is None else round(row[k]["p90_abs"], 4)) for k in pr.KEYS},
              "chi2", round(chi2[level], 2) if chi2[level] is not None else None)
    print("seconds per curve", round(summary["median_seconds"], 1), "failed", summary["failed"])


if __name__ == "__main__":
    main()
