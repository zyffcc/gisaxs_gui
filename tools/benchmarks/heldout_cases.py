"""The second held-out set of the 1D benchmarks: fit1d_recovery.py's draw on a fresh seed range, widened.

Same ranges, levels of physics, curve and counts as fit1d_recovery.draw/curve (the independent truth model
yoneda_truth.py), and in addition three populations (a third size scale 1.8–3 times from the second) and the decoupling
approximation (DA) in half of the cases. Compositions cycle through COMPOSITIONS; ``index`` and ``seed0`` fix a case.
Used by posterior_recovery.py and fit1d_recovery.py with --set extended (seed0 20261201 for the frozen set).
"""
from __future__ import annotations

import numpy as np

import fit1d_recovery as fr
import yoneda_truth as yt

COMPOSITIONS = {"sphere": ("sphere",), "vc": ("vc",), "sphere,sphere": ("sphere", "sphere"),
                "sphere,vc": ("sphere", "vc"), "sphere,sphere,vc": ("sphere", "sphere", "vc"),
                "sphere,vc,vc": ("vc", "vc", "sphere")}
SEED0 = 20261201


def draw(index: int, seed0: int = SEED0) -> dict:
    rng = np.random.default_rng(seed0 + index)
    names = list(COMPOSITIONS)
    comp = names[index % len(names)]
    level = "model" if (index // len(names)) % 2 == 0 else "measured"
    coupling = "da" if (index // (2 * len(names))) % 2 else "lma"
    pops = []
    for k, shape in enumerate(COMPOSITIONS[comp]):
        if k == 0:
            radius = float(np.exp(rng.uniform(np.log(2.0), np.log(8.0))))
        else:  # another size scale, 1.8–3 times above the largest so far (within 10 nm) or below the smallest
            ratio = float(np.exp(rng.uniform(np.log(1.8), np.log(3.0))))
            largest, smallest = max(p["R"] for p in pops), min(p["R"] for p in pops)
            radius = largest * ratio if largest * ratio <= 10.0 else smallest / ratio
            radius = float(max(radius, 1.2))
        pops.append(dict(shape=shape, R=radius, sR=float(rng.uniform(0.1, 0.35)),
                         D=float(2 * radius * rng.uniform(1.3, 2.5)), s_nn=float(rng.uniform(0.1, 0.3)),
                         share=1.0 if k == 0 else float(rng.uniform(0.3, 1.0))))
    peak = float(10 ** rng.uniform(3, 5))
    return dict(index=index, composition=comp, level=level, coupling=coupling, populations=pops, peak_counts=peak,
                background=float(10 ** rng.uniform(-3, -1.5)), seed=seed0 + index)


def curve(case: dict):
    """fit1d_recovery.curve with the case's coupling."""
    inst = yt.Instrument(peak_counts=0.0)
    if case["level"] == "measured":
        inst = yt.replace(inst, sigma_g=0.005, central=30.0, eta=0.03, gamma=0.004, rough=1.0, xi=50.0, hurst=0.5)
    total, peak_of_particles = np.zeros_like(fr.Q), None
    for k, p in enumerate(case["populations"]):
        film = yt.Film(R=p["R"], sR=p["sR"], D=p["D"], s_nn=p["s_nn"], coupling=case.get("coupling", "lma"),
                       scattering="born_qz", cyl_h=2.0 * p["R"] if p["shape"] == "vc" else 0.0)
        own = yt.measured_curve(fr.Q, yt.Case(fr.GEOM, film, inst if k == 0 else yt.replace(inst, central=0.0,
                                                                                           rough=0.0)))
        particles = own["parts"]["particles"]
        if peak_of_particles is None:
            peak_of_particles = float(particles.max())
            total += own["I"]
        else:
            total += own["I"] * p["share"] * peak_of_particles / float(particles.max())
    total += case["background"] * float(total.max())
    scale = case["peak_counts"] / float(total[fr.Q > 0.1].max())
    rng = np.random.default_rng(case["seed"])
    counts = rng.poisson(total * scale).astype(float)
    return counts, np.sqrt(np.maximum(counts, 1.0))


__all__ = ["COMPOSITIONS", "SEED0", "curve", "draw"]
