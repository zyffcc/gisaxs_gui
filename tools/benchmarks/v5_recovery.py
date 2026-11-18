"""How well 1D Predict (Workflow V5, the "AI proposal") recovers the parameters of fit1d's held-out curves.

The curves, the levels of physics and the scoring of fit1d_recovery.py (its draw() and curve(), unchanged), for the
retirement of V5: the true composition is given (V5's conditional path, as in its own validation), its default options
otherwise (neural proposals, then four numerical steps), the q > 0 half (the curve has one). V5's sizes are Gaussian
by intensity and its σD/D is (σnn/D)/√(2π): R and σR/R are compared in V5's convention, the truth's (Gaussian by
number) converted to its intensity-weighted mean and relative spread (cut_model.conventions.intensity_from_number);
σD is converted to σnn.
Needs TensorFlow (requirements-tensorflow.txt). V5 was retired in 0.2.0: run this in a checkout of commit 06e7395
(git worktree add <folder> 06e7395); its summary stays here as the record the retirement was decided on.
usage: python tools/benchmarks/v5_recovery.py [--n 200] [--workers 4] [--seed0 20261116]
Writes tools/benchmarks/v5_recovery_summary.json and every curve to results/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
GIMAP = HERE.parents[1]
for p in (str(HERE), str(GIMAP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fit1d_recovery as fr  # noqa: E402

TYPES = {"sphere": 1, "vc": 3}
POWER = {"sphere": 6, "vertical_cylinder": 4}
SEED0 = [20261116]


def _init(seed0):
    SEED0[0] = seed0


def run(index: int) -> dict:
    from src.gimap.features.fitting.application.workflow_v5 import bundled_workflow, default_options, validate_options
    from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import run_workflow_job

    case = fr.draw(index, SEED0[0])
    I, s = fr.curve(case)
    out = dict(case=case)
    tick = time.perf_counter()
    try:
        options = validate_options({**default_options(), "method": "model", "q_unit": "nm^-1", "side": "positive",
                                    "components": [TYPES[p["shape"]] for p in case["populations"]]})
        with tempfile.TemporaryDirectory() as folder:
            value = run_workflow_job({"options": options, "output_dir": folder, "model_path": str(bundled_workflow()),
                                      "q": fr.Q.tolist(), "intensity": I.tolist(), "sigma": s.tolist()},
                                     lambda *args: None, lambda: False)
        best = sorted(value["candidates"], key=lambda r: r.get("best_log_rmse") or math.inf)[0]
        pops = []
        for c in sorted(best["components"], key=lambda c: c["params"]["R"]):
            p = c["params"]
            sigma_d = p.get("sigma_D")
            pops.append(dict(R=p["R"], sR=p["sigma_R"], D=p.get("D") if sigma_d is not None else None,
                             s_nn=None if sigma_d is None else math.sqrt(2 * math.pi) * sigma_d))
        out["fit"] = dict(ok=True, populations=pops, log_rmse=best.get("best_log_rmse"))
    except Exception as exc:  # noqa: BLE001 - one failing curve must not stop the study
        out["fit"] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
    out["fit"]["seconds"] = time.perf_counter() - tick
    return out


def _intensity_truth(population: dict) -> dict:
    from src.gimap.features.fitting.domain.cut_model.conventions import intensity_from_number

    family = "sphere" if population["shape"] == "sphere" else "vertical_cylinder"
    radius, spread = intensity_from_number(population["R"], population["sR"], POWER[family])
    return {**population, "R": radius, "sR": spread}


def summary(results):
    table = {}
    for level in ("model", "measured", "all"):
        errs = {k: [] for k in fr.KEYS}
        for r in results:
            if level != "all" and r["case"]["level"] != level:
                continue
            got = r.get("fit") or {}
            if not got.get("ok") or len(got["populations"]) != len(r["case"]["populations"]):
                continue
            truth = [_intensity_truth(p) for p in sorted(r["case"]["populations"], key=lambda p: p["R"])]
            for t, g in zip(truth, got["populations"]):
                for k in fr.KEYS:
                    if g.get(k) is not None and np.isfinite(g[k]):
                        errs[k].append((g[k] - t[k]) / t[k] if k in ("R", "D") else g[k] - t[k])
        table[f"fit/{level}"] = {k: dict(n=len(v), median_abs=float(np.median(np.abs(v))) if v else None,
                                         p90_abs=float(np.percentile(np.abs(v), 90)) if v else None,
                                         bias=float(np.median(v)) if v else None) for k, v in errs.items()}
    seconds = [r["fit"]["seconds"] for r in results if r["fit"].get("ok")]
    return dict(parameters=table, median_seconds=float(np.median(seconds)) if seconds else None, n=len(results),
                failed=int(sum(not r["fit"].get("ok") for r in results)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=20261116)
    ap.add_argument("--out", default="v5_recovery")
    a = ap.parse_args()
    results = []
    with Pool(min(a.workers, a.n), initializer=_init, initargs=(a.seed0,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, range(a.n)), 1):
            results.append(r)
            print(f"[{i}/{a.n}] {r['case']['composition']:13s} {r['case']['level']:8s} "
                  f"{'ok' if r['fit'].get('ok') else 'ERR ' + r['fit'].get('error', '')[:80]} {r['fit']['seconds']:.1f}s",
                  flush=True)
    s = summary(results)
    s["seed0"] = a.seed0
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / f"{a.out}_summary.json").write_text(json.dumps(s, indent=1))
    (HERE / "results" / f"{a.out}.json").write_text(json.dumps(dict(summary=s, results=results), indent=1, default=str))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
