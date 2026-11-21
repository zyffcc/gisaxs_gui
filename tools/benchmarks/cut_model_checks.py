"""cut_model, GIMaP's one 1D forward model, against the models it replaces and an independent one (Physics and Methods 9.2).

References, each on q = 0.02–3 nm⁻¹ (600 points):
  fitting   Fitting's earlier formulas (scattering_model, coherent_terms): on their own default rules, and on fine
            rules of the same formulas (the integral they approximate)
  v5        1D Predict V5's float64 forward (modules/Fitting_1D_Model/Workflow_v5/conditional_v2/numpy_forward64.py;
            V5 was retired in 0.2.0, so this part is skipped from then on: run it in a checkout of commit 06e7395)
  fit1d     the fit1d kit's tabulated physics (number weights, Hosemann paracrystal; utils/gisaxs1d_gui_kit)
  truth     tools/benchmarks/yoneda_truth.py (no code shared), its size rule refined to 1500 nodes: Born at the cut's
            q_z and the four-term DWBA on Si, LMA and DA
For every case: the largest and the median relative deviation |a/b − 1| over q. Curves with an arbitrary absolute
scale (truth) are compared after one scale. Also the time of 1000 parameter sets × 500 q per family on one core.
usage: python tools/benchmarks/cut_model_checks.py [--out PATH] [--force]
  --out PATH   where the summary is written (default: the stored cut_model_checks.json next to this script)
  --force      overwrite the stored file also when an optional comparison (v5, fit1d) was skipped: without it such a
               run refuses, because it would replace the stored record by one with fewer rows
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import functools  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STORED = HERE / "cut_model_checks.json"
ROOT = HERE.parents[1]
KIT = ROOT / "utils" / "gisaxs1d_gui_kit"
V5 = ROOT / "modules" / "Fitting_1D_Model" / "Workflow_v5" / "conditional_v2"
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.gimap.features.fitting.domain import coherent_terms as ct  # noqa: E402
from src.gimap.features.fitting.domain import scattering_model as sm  # noqa: E402
from src.gimap.features.fitting.domain.cut_model import Geometry, population_terms  # noqa: E402
from src.gimap.features.fitting.domain.cut_model.conventions import SQRT_2PI  # noqa: E402
from src.gimap.features.fitting.domain.cut_model.structure import paracrystal  # noqa: E402

Q = np.linspace(0.02, 3.0, 600)
WAVELENGTH, ALPHA_I = 0.1045, 0.40
SI = dict(delta=3.28e-6 * (WAVELENGTH / 0.1023) ** 2, beta=5.0e-8 * (WAVELENGTH / 0.1023) ** 4)
RESULTS = []
SKIPPED = []
"""The optional comparisons that did not run here (their old code is not in this checkout)."""


def record(name, reference, new, ref, scaled=False, note=""):
    new, ref = np.asarray(new, float), np.asarray(ref, float)
    if scaled:
        ref = ref * float(np.exp(np.median(np.log(new / ref))))
    ratio = np.abs(new / ref - 1.0)
    RESULTS.append(dict(name=name, reference=reference, max_rel=float(ratio.max()), median_rel=float(np.median(ratio)),
                        q_at_max=float(Q[np.argmax(ratio)]) if new.size == Q.size else None, note=note))
    print(f"{reference:8s} {name:58s} max {ratio.max():.2e}  median {np.median(ratio):.2e}", flush=True)


def fitting_checks():
    for radius, spread in ((2.0, 0.05), (5.0, 0.2), (10.0, 0.3), (20.0, 0.4)):
        new = population_terms("sphere", Q, radius=radius, radius_spread=spread, law="intensity_gauss")[0]
        record(f"sphere R={radius} σ={spread}, old default rule (25 nodes)", "fitting", new,
               sm.sphere_form_factor_pd(Q, radius, radius * spread))
        record(f"sphere R={radius} σ={spread}, old formula on 20001 nodes", "fitting", new,
               sm.sphere_form_factor_pd(Q, radius, radius * spread, n_samples=20001))
    for radius, spread in ((3.0, 0.1), (8.0, 0.3)):
        new = population_terms("vertical_cylinder", Q, radius=radius, radius_spread=spread, law="number_gauss")[0]
        old = sm.vertical_cylinder_form_factor_pd(Q, radius, spread, n_samples=20000, nsig=4.0)
        zero = sm.vertical_cylinder_form_factor_pd(np.array([1e-9]), radius, spread, n_samples=20000, nsig=4.0)
        record(f"vertical cylinder R={radius} σ={spread} (number), old formula fine, ±4σ", "fitting", new, old / zero)
    q_cyl = Q[::2]
    for radius, spread, height, h_spread in ((3.0, 0.1, 10.0, 0.1), (5.0, 0.3, 30.0, 0.2)):
        new = population_terms("cylinder", q_cyl, radius=radius, radius_spread=spread, height=height,
                               height_spread=h_spread, law="intensity_gauss")[0]
        for orient in (24, 800):
            old = sm.cylinder_form_factor_pd(q_cyl, radius, radius * spread, height, height * h_spread,
                                             n_R=13 if orient == 24 else 201, n_h=13 if orient == 24 else 201,
                                             n_orient=orient)
            ratio = np.abs(new / old - 1)
            RESULTS.append(dict(name=f"random cylinder R={radius} h={height}, old rule {orient} orientations",
                                reference="fitting", max_rel=float(ratio.max()), median_rel=float(np.median(ratio)),
                                q_at_max=float(q_cyl[np.argmax(ratio)]), note="old orientation rule converges as 1/n"))
            print(f"fitting  random cylinder R={radius} h={height} old {orient} orientations   max {ratio.max():.2e}")
    radius, spread, distance, sigma_d = 5.0, 0.2, 18.0, 0.1
    s_old = sm.structure_factor_1d(Q, distance, distance * sigma_d)
    record("paracrystal, σ_nn = √(2π) σ_D", "fitting", paracrystal(Q, distance, SQRT_2PI * sigma_d)[0], s_old)
    p_old = sm.sphere_form_factor_pd(Q, radius, radius * spread, n_samples=20001)
    c_old = ct.sphere_coherent_pd(Q, radius, radius * spread, n_samples=20001)
    new = population_terms("sphere", Q, radius=radius, radius_spread=spread, law="intensity_gauss", distance=distance,
                           distance_spread=SQRT_2PI * sigma_d, coupling="da")[0]
    record("sphere DA, old formula on 20001 nodes", "fitting", new, p_old + c_old * (s_old - 1))
    geo = Geometry("two_path", WAVELENGTH, ALPHA_I, 0.15, 0.8, -0.3)
    paths = (geo.k_i + geo.k_f, geo.k_i - geo.k_f, geo.k_f, 0.8, -0.3)
    new = population_terms("sphere", Q, radius=radius, radius_spread=spread, law="intensity_gauss", geometry=geo)[0]
    record("two-path sphere |r| 0.8, old formula on 20001 nodes", "fitting", new,
           sm.sphere_two_path_pd(Q, radius, radius * spread, *paths, n_samples=20001))


def v5_checks():
    if not (V5 / "numpy_forward64.py").is_file():
        print("V5 not here: skipped")
        SKIPPED.append("v5")
        return
    sys.path.insert(0, str(V5))
    component = importlib.import_module("numpy_forward64").component
    for typ, family in ((1, "sphere"), (3, "vertical_cylinder")):
        for radius, spread in ((4.0, 0.0), (3.0, 0.05)):
            new = population_terms(family, Q, radius=radius, radius_spread=spread, law="intensity_gauss",
                                   distance=15.0, distance_spread=SQRT_2PI * 0.15)[0]
            old = component(Q, typ, radius, spread, 10, 0.2, 15.0, 0.15, True)  # relative widths, σ_D/D
            record(f"{family} R={radius} σ={spread} + paracrystal", "v5", new, old)


def fit1d_checks():
    if not (KIT / "tablepack.py").is_file():
        print("fit1d kit not here: skipped")
        SKIPPED.append("fit1d")
        return
    tables = Path(os.environ.setdefault("GISAXS1D_TABLES", str(Path(tempfile.gettempdir()) / "gisaxs1d_tables")))
    for path in (str(KIT), str(KIT / "app")):
        if path not in sys.path:
            sys.path.insert(0, path)
    if not all((tables / f"ff_{t}_number_v2.npy").is_file() for t in (1, 2, 3)):
        tables.mkdir(parents=True, exist_ok=True)  # by name, as GIMaP loads the kit (not a package of requirements.txt)
        importlib.import_module("tablepack").unpack(str(KIT / "tables_packed"), str(tables))
    tabfit = importlib.import_module("research.fast_inference.tabfit")
    geo = Geometry("born", WAVELENGTH, ALPHA_I, 0.15)
    phys = tabfit.TabPhysics(Q, qz=geo.qz_born, weighting="number", sf="hosemann")
    for radius, spread in ((2.0, 0.1), (5.0, 0.2), (10.0, 0.35)):
        new = population_terms("sphere", Q, radius=radius, radius_spread=spread, geometry=geo)[0]
        record(f"sphere R={radius} σ={spread}, Born at q_z={geo.qz_born:.3f}", "fit1d", new, phys.radial(1, radius, spread))
        new = population_terms("vertical_cylinder", Q, radius=radius, radius_spread=spread, geometry=geo)[0]
        record(f"vertical cylinder R={radius} σ={spread}", "fit1d", new, phys.radial(3, radius, spread))
    new = population_terms("cylinder", Q, radius=3.0, radius_spread=0.15, height=12.0, height_spread=0.2, geometry=geo)[0]
    record("random cylinder R=3 h=12, Born at q_z", "fit1d", new,
           phys.combine(2, phys.radial(2, 3.0, 0.15), phys.height(12.0, 0.2)))
    record("paracrystal D=20 σ_nn/D=0.2", "fit1d", paracrystal(Q, 20.0, 0.2)[0], phys.structure(20.0, 0.2))
    dwba = Geometry("dwba", WAVELENGTH, ALPHA_I, None, delta=SI["delta"], beta=SI["beta"])
    phys = tabfit.TabPhysics(Q, qz=dwba.qz_born, weighting="number", sf="hosemann",
                             dwba=tabfit.YonedaDWBA(WAVELENGTH, ALPHA_I, dwba.exit_angle_deg, SI["delta"], SI["beta"]))
    for radius, spread in ((3.0, 0.15), (6.0, 0.25)):
        new = population_terms("sphere", Q, radius=radius, radius_spread=spread, geometry=dwba)[0]
        record(f"sphere R={radius} σ={spread}, four-term DWBA on Si", "fit1d", new, phys.radial(1, radius, spread))


def truth_checks():
    import yoneda_truth as yt  # noqa: PLC0415
    yt.size_rule = functools.partial(yt.size_rule, n=1500)
    yt_geometry = yt.Geometry(wavelength_nm=WAVELENGTH, alpha_i_deg=ALPHA_I)
    cases = (("sphere", dict(R=4.0, sR=0.2, D=14.0, s_nn=0.2)), ("sphere", dict(R=6.0, sR=0.3, D=20.0, s_nn=0.25)),
             ("vertical_cylinder", dict(R=4.0, sR=0.2, D=14.0, s_nn=0.2, cyl_h=8.0)))
    for scattering, mode in (("born_qz", "born"), ("dwba", "dwba")):
        geo = Geometry(mode, WAVELENGTH, ALPHA_I, yt_geometry.alpha_f, delta=SI["delta"], beta=SI["beta"])
        for family, f in cases:
            for coupling in ("lma", "da"):
                film = yt.Film(coupling=coupling, scattering=scattering, **f)
                ref = yt.particle_intensity(Q, yt.Case(yt_geometry, film))
                new = population_terms(family, Q, radius=f["R"], radius_spread=f["sR"], distance=f["D"],
                                       distance_spread=f["s_nn"], coupling=coupling, geometry=geo)[0]
                record(f"{family} R={f['R']} σ={f['sR']} {coupling.upper()} {scattering}", "truth", new, ref, scaled=True)


def timing():
    rng = np.random.default_rng(0)
    q = np.geomspace(0.01, 2.0, 500)
    out = {}
    for family in ("sphere", "vertical_cylinder", "cylinder"):
        radius = np.exp(rng.uniform(math.log(1.0), math.log(15.0), 1000))
        start = time.perf_counter()
        population_terms(family, q, radius=radius, radius_spread=rng.uniform(0.05, 0.4, 1000),
                         height=radius * rng.uniform(1, 4, 1000), height_spread=0.15,
                         distance=radius * rng.uniform(2.5, 6, 1000), distance_spread=0.2)
        out[family] = time.perf_counter() - start
        print(f"timing   {family:20s} 1000 x 500 q: {out[family]:.2f} s", flush=True)
    return out


def refuses(out: Path, skipped, force: bool) -> str:
    """Why this run must not write ``out`` ('' when it may): it would replace the stored record by one without the
    rows of the comparisons skipped here."""
    if force or not skipped or not STORED.exists() or Path(out).resolve() != STORED.resolve():
        return ""
    return (f"Not written: {' and '.join(skipped)} skipped here, so {STORED.name} would lose those rows. "
            "Give --out PATH for this run's summary, or --force to overwrite the stored file.")


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=STORED, help="where the summary is written (default: the stored file)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the stored file also when an optional comparison was skipped")
    arguments = parser.parse_args(argv)
    fitting_checks()
    v5_checks()
    fit1d_checks()
    truth_checks()
    seconds = timing()
    summary = dict(q=[float(Q[0]), float(Q[-1]), int(Q.size)], wavelength_nm=WAVELENGTH, alpha_i_deg=ALPHA_I, si=SI,
                   checks=RESULTS, seconds_1000x500_one_core=seconds, skipped=list(SKIPPED))
    refusal = refuses(arguments.out, SKIPPED, arguments.force)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written: {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
