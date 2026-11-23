"""Predict parameters (the proposal network and its posterior) on the GALAXI Yoneda cut, measured and simulated with
BornAgain's model of the frame: the counterpart of galaxi_fit1d.py (Physics and Methods 9.13).

The same cut, angles and BornAgain sources as galaxi_fit1d.py (BornAgain's log-normal radii: number mean 6.24 nm,
relative spread 0.42; radial paracrystal 53.6 nm, σnn/D 0.196). Variants:
  born1        one sphere population with a paracrystal, Born at the vacuum q_z of the cut, Gaussian sizes by number
  born1_lognormal  the same with log-normal sizes (the network's samples, weighted and refined in that law)
  auto3        the composition among one to three sphere populations (with or without a paracrystal)
  paths1, paths_auto3  born1 and auto3 with both angles of the cut: spheres with the exit-reflected path and a free
               effective reflection coefficient (two paths, as with the angles in Model)
Fitting has no four-term DWBA, so fit1d's dwba1 has no closer counterpart. Needs BornAgain (the simulated sources).
usage: python tools/benchmarks/galaxi_posterior.py [--network PATH/NAME]   (default: the network that comes with GIMaP)
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import galaxi_fit as gf  # noqa: E402
from galaxi_fit1d import ALPHA_F, ALPHA_I, K, QZ, TRUTH  # noqa: E402
from posterior_recovery import _code_state  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import network as posterior_network  # noqa: E402
from src.gimap.features.fitting.domain.posterior import refine  # noqa: E402
from src.gimap.features.fitting.domain.posterior.flow import Network  # noqa: E402

SPHERE = E.composition_index([("sphere", True)])
QZ_MINUS = K * (math.sin(math.radians(ALPHA_I)) - math.sin(math.radians(ALPHA_F)))
VARIANTS = {
    "born1": dict(compositions=[SPHERE]),
    "born1_lognormal": dict(compositions=[SPHERE], law="lognormal"),
    "auto3": dict(allowed=E.select(("sphere",), 3)),
    "paths1": dict(compositions=[SPHERE], qz_minus=QZ_MINUS),
    "paths_auto3": dict(allowed=E.select(("sphere",), 3), qz_minus=QZ_MINUS),
}


def row(solution) -> dict:
    values = {name: dict(best=v.best, median=v.median, low68=v.low68, high68=v.high68, error=v.error)
              for name, v in solution.values.items()}
    r = refine.reflection(solution.model)
    return dict(composition=solution.name, probability=solution.probability, chi2=solution.chi2_reduced,
                flags=list(solution.flags), ess=solution.ess, values=values, reflection=[abs(r), math.atan2(r.imag, r.real)],
                law=solution.model.components[0].law if solution.model.components else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network")
    a = ap.parse_args()
    network = Network.load(a.network) if a.network else posterior_network.load()
    q, measured, sigma = gf.curve()
    model = importlib.import_module("galaxi_bornagain_model")
    simulated = model.simulate(q, lognormal=True)
    scale = model.scale_background(simulated, measured, sigma, q > 0.3)
    expected = scale["scale"] * simulated + scale["background"]
    sources = {"measured": (measured, sigma)}
    for seed in (1, 2, 3):
        counts = np.random.default_rng(seed).poisson(expected * 5) / 5.0  # 5 pixels per point, as galaxi_fit1d.py
        sources[f"bornagain_{seed}"] = (counts, np.sqrt(np.maximum(counts * 5, 1.0)) / 5.0)
    rows = []
    for variant, options in VARIANTS.items():
        for source, (intensity, error) in sources.items():
            try:
                solutions = refine.predict(network, q, intensity, error, qz=QZ, seed=0, **options)
                rows.append(dict(variant=variant, source=source, best=row(solutions[0]),
                                 others=[row(s) for s in solutions[1:4]], seconds=solutions[0].seconds))
            except Exception as exc:  # noqa: BLE001 - a failing variant is reported, not fatal
                rows.append(dict(variant=variant, source=source, error=f"{type(exc).__name__}: {exc}"))
            best = rows[-1].get("best") or {}
            values = best.get("values") or {}
            print(variant, source, best.get("composition"), "chi2/N", best.get("chi2"),
                  "R", (values.get("R_1") or {}).get("best"), "D", (values.get("D_1") or {}).get("best"), flush=True)
    payload = dict(code=_code_state(), truth=TRUTH, qz=QZ, network=network.meta.get("sha256"), fits=rows)
    (HERE / "galaxi_posterior.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
