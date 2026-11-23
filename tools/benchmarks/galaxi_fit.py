"""GIMaP Fitting on the measured GALAXI cut, against BornAgain's reference model of the same frame.

BornAgain r23, rawEx/fit/scatter2d/hardcode_galaxi.py: Ag spheres R = 5.75 nm (the log-normal sigma 0.4 is defined but
commented out, so the spheres are monodisperse), 18.5 nm deep in HMDSO on PTFE on Si; radial paracrystal, peak distance
D = 53.6 nm, Gaussian disorder 10.5 nm (s_nn/D = 0.196), size-spacing coupling kappa = 17.5; DWBA of the layer stack.

GIMaP: the horizontal cut of Analyze at the Yoneda band (tests/data/external/gisaxs_galaxi/gimap_analysis), the mean of
both halves, EIA (P at q_z = 0), beam tail free, Search then Refine. Models:
  lma        sphere (R, sigma_R) x S(D, sigma_D)
  da         the same, Coupling DA
  mono       lma with sigma_R/R held at its lower limit 0.01 (BornAgain's monodisperse spheres)
  lma_large  lma + a large sphere population without S
  The same with "paths": the two-path sphere at the cut's angles (alpha_i 0.463 deg, alpha_f 0.18 deg at the Yoneda row
  607.5, lambda 0.134 nm; |r| and arg r free), so P is taken at q_z = k_i + k_f (q_z R ~ 3) instead of 0.
  "fixS": D = 53.6 nm and s_nn/D = 0.196 held at BornAgain's values.
usage: python tools/benchmarks/galaxi_fit.py
"""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CUT = REPO / "tests/data/external/gisaxs_galaxi/gimap_analysis/galaxi_data_fit_input.dat"
REFERENCE = dict(R=5.75, sR=0.0, D=53.6, s_nn=10.5 / 53.6)
SEEDS = (1729, 11, 23)


def curve():
    sys.path.insert(0, str(REPO))
    from src.gimap.shared.signed_cut import mean_of_halves
    q, I, s = np.loadtxt(CUT, usecols=(0, 1, 2), unpack=True)
    q = q * 10.0  # 1/A -> 1/nm
    pos, neg = q > 0, q < 0
    order = np.argsort(-q[neg])
    m = mean_of_halves((q[pos], I[pos], s[pos]), (-q[neg][order], I[neg][order], s[neg][order]), grid="positive")
    return m.q, m.intensity, m.fit_sigma(np.ones(m.q.size, dtype=bool))


def run(job):
    variant, seed = job
    paths, fixed_s = "paths" in variant, "fixS" in variant
    sys.path.insert(0, str(REPO))
    from src.gimap.features.fitting.domain import fit_model as fm, fit_engine as fe
    q, I, s = curve()
    main = fm.new_component("sphere", radius=6.0, distance=50.0)
    if variant.startswith("da"):
        main = replace(main, coupling="da")
    comps = [main]
    if "large" in variant:
        comps.append(fm.new_component("sphere", radius=20.0, structure=False))
    model = fm.FitModel(components=tuple(comps))
    if "mono" in variant:
        model = model.with_parameter((0, "sigma_R"), value=0.01, free=False)
    for key in ("res_width", "res_exponent"):
        model = model.with_parameter(("globals", key), free=True)
    if paths:
        for key, value in (("alpha_i", 0.463), ("alpha_f", 0.18), ("wavelength", 0.134)):
            model = model.with_parameter(("globals", key), value=value)
        model = model.with_parameter(("globals", "refl_amp"), value=0.5, free=True)
        model = model.with_parameter(("globals", "refl_phase"), free=True)
    if fixed_s:
        model = model.with_parameter((0, "D"), value=REFERENCE["D"], free=False)
        model = model.with_parameter((0, "sigma_D"), value=REFERENCE["s_nn"] / np.sqrt(2 * np.pi), free=False)
    data = fe.FitData.prepare(q, I, s)
    res = fe.fit(model, data, method="global", max_evaluations=12000, seed=seed)
    res = fe.fit(res.model, data, method="local")
    c = res.model.components[0]
    number = fm.number_size(c)
    return dict(variant=variant, seed=seed, chi2=res.chi2_reduced, refl_amp=res.model.globals["refl_amp"].value, R=c.value("R"), sR=c.value("sigma_R"),
                D=c.value("D"), s_nn=float(np.sqrt(2 * np.pi) * c.value("sigma_D")),
                R_number=None if number is None else number[0], points=int(q.size), q_range=[float(q.min()), float(q.max())])


if __name__ == "__main__":
    variants = ("lma", "da", "mono", "lma_large", "lma_fixS", "mono_fixS",
                "lma_paths", "mono_paths", "lma_paths_fixS", "mono_paths_fixS", "lma_large_paths")
    jobs = [(v, sd) for v in variants for sd in SEEDS]
    with Pool(12) as pool:
        res = list(pool.imap_unordered(run, jobs))
    (HERE / "galaxi_fit.json").write_text(json.dumps(dict(reference=REFERENCE, fits=res), indent=1))
    print("reference", REFERENCE)
    for r in sorted(res, key=lambda r: (r["variant"], r["chi2"])):
        print(f"{r['variant']:9s} seed {r['seed']:5d} chi2 {r['chi2']:6.3f}  R {r['R']:.2f} sR {r['sR']:.3f} "
              f"(number {r['R_number'] if r['R_number'] is None else round(r['R_number'], 2)})  D {r['D']:.1f}  "
              f"s_nn {r['s_nn']:.3f}  |r| {r['refl_amp']:.2f}")
