"""fit1d on the GALAXI Yoneda cut, measured and simulated with BornAgain's model of the frame (Physics and Methods 9.12).

fit1d's sizes are number-weighted, so its R and σ_R/R compare directly with BornAgain's log-normal radii
(median 5.75 nm, σ 0.4: number mean 6.24 nm, relative spread 0.42; galaxi_bornagain_model.py) and its D, w/D with
the radial paracrystal (53.6 nm, 0.196). Variants:
  born1     one sphere population, Born at the vacuum q_z of the cut (α_i 0.463°, α_f 0.18°, λ 0.134 nm)
  dwba1     one sphere population, four-term DWBA on an effective Si substrate at the same angles (BornAgain's
            stack is HMDSO / PTFE / Si; fit1d has one substrate)
  auto3     composition chosen by fit1d's BIC rule among 1 to 3 sphere populations, Born
Each on the measured cut and on BornAgain's model of it (counts of the measurement, Poisson noise, 3 seeds).
usage: python tools/benchmarks/galaxi_fit1d.py
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KIT = ROOT / "utils" / "gisaxs1d_gui_kit"
for path in (str(HERE), str(ROOT), str(KIT / "app")):
    if path not in sys.path:
        sys.path.insert(0, path)

import galaxi_fit as gf  # noqa: E402

WAVELENGTH_NM, ALPHA_I, ALPHA_F = 0.134, 0.463, 0.18
K = 2 * math.pi / WAVELENGTH_NM
QZ = K * (math.sin(math.radians(ALPHA_I)) + math.sin(math.radians(ALPHA_F)))
SI = dict(substrate_delta=5.7816e-6, substrate_beta=1.0229e-7)
SIGMA_LN = 0.4
TRUTH = dict(R_number_mean=5.75 * math.exp(SIGMA_LN**2 / 2), sR=math.sqrt(math.exp(SIGMA_LN**2) - 1), D=53.6,
             w_rel=10.5 / 53.6)
VARIANTS = {
    "born1": dict(composition="sphere", qz=QZ),
    "dwba1": dict(composition="sphere", dwba=dict(wavelength_nm=WAVELENGTH_NM, alpha_i_deg=ALPHA_I,
                                                  alpha_f_deg=ALPHA_F, **SI)),
    "auto3": dict(composition="auto:sphere:3", qz=QZ),
}


def run(job):
    variant, source, q, intensity, sigma = job
    fit1d = importlib.import_module("research.fit1d.api").fit1d
    try:
        out = fit1d(q, intensity, sigma, mode="gisaxs", policy="auto", workers=1, seed=0,
                    model_dir=str(KIT / "models" / "general"), **VARIANTS[variant])
    except Exception as exc:  # a variant fit1d refuses is reported, not fatal
        return dict(variant=variant, source=source, error=f"{type(exc).__name__}: {exc}")
    best = out["results"][0]
    pops = [dict(R=p.get("R_nm"), R_number_mean=p.get("R_number_mean_nm"), sR=p.get("sigma_R_rel"), D=p.get("D_nm"),
                 w_rel=p.get("w_rel"), R_err=p.get("R_nm_err"), D_err=p.get("D_nm_err"), D_range=p.get("D_nm_range"),
                 weight=p.get("weight") or p.get("fraction"))
            for p in best["populations"]]
    flags = best.get("flags") if isinstance(best.get("flags"), dict) else {}
    return dict(variant=variant, source=source, composition=best.get("composition_text"),
                chi2=best.get("chi2_per_point"), populations=pops, flags=flags,
                in_set=[r.get("composition_text") for r in out["results"] if r.get("in_candidate_set")])


def main() -> None:
    q, measured, sigma = gf.curve()
    model = importlib.import_module("galaxi_bornagain_model")
    simulated = model.simulate(q, lognormal=True)
    scale = model.scale_background(simulated, measured, sigma, q > 0.3)
    expected = scale["scale"] * simulated + scale["background"]
    sources = {"measured": (measured, sigma)}
    for seed in (1, 2, 3):
        counts = np.random.default_rng(seed).poisson(expected * 5) / 5.0  # 5 pixels per point, as measured
        sources[f"bornagain_{seed}"] = (counts, np.sqrt(np.maximum(counts * 5, 1.0)) / 5.0)
    jobs = [(v, s, q, i, e) for v in VARIANTS for s, (i, e) in sources.items()]
    with Pool(min(10, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run, jobs))
    (HERE / "galaxi_fit1d.json").write_text(json.dumps(dict(truth=TRUTH, qz=QZ, fits=rows), indent=1, default=str),
                                            encoding="utf-8")
    print("truth (by number):", {k: round(v, 3) for k, v in TRUTH.items()})
    for r in sorted(rows, key=lambda r: (r["variant"], r["source"])):
        if "error" in r:
            print(f"{r['variant']:6s} {r['source']:12s} {r['error']}")
            continue
        text = "; ".join(f"R {p['R']:.2f} (number {p['R_number_mean'] if p['R_number_mean'] is None else round(p['R_number_mean'], 2)})"
                         f" sR {p['sR']:.2f} D {p['D'] if p['D'] is None else round(p['D'], 1)} w {p['w_rel'] if p['w_rel'] is None else round(p['w_rel'], 2)}"
                         for p in r["populations"])
        print(f"{r['variant']:6s} {r['source']:12s} {r['composition']:14s} chi2 {r['chi2']:.2f}  {text}")


if __name__ == "__main__":
    main()
