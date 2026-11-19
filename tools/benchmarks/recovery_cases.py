"""The held-out curves of the 1D recovery benchmarks (fit1d's 200 and the posterior's): their draw and their curve
from the independent truth model ``yoneda_truth.py``. Shared by ``posterior_recovery.py``, ``heldout_cases.py`` and
``fit1d_recovery.py``; needs neither the fit1d kit nor GIMaP."""
from __future__ import annotations

import numpy as np

import yoneda_truth as yt

GEOM = yt.Geometry(wavelength_nm=0.1045, alpha_i_deg=0.40)       # P03, Yoneda row of Si
Q = np.linspace(0.02, 2.0, 397)                                  # nm^-1
COMPOSITIONS = {"sphere": ("sphere",), "vc": ("vc",), "sphere,sphere": ("sphere", "sphere"),
                "sphere,vc": ("sphere", "vc")}
# What the particles sit on. 'ps100': 100 nm polystyrene on Si (1.05 g/cm³: δ 1.67e-6, β 2e-9 at 0.1045 nm); the cut
# stays at the Yoneda angle of Si, so the vacuum q_z of the cut is the same on both.
SUBSTRATES = {"si": {}, "ps100": dict(film_delta=1.67e-6, film_beta=2e-9, film_thickness=100.0)}
TRUTH = dict(scattering="born_qz", substrate="si")
"""The truth's physics: the frozen held-out sets are the Born term at the cut's q_z ('born_qz') on Si; 'dwba' adds
the waves the surface reflects (the four terms of yoneda_truth). Set with set_truth()."""


def set_truth(scattering: str = "born_qz", substrate: str = "si") -> None:
    if scattering not in ("born_qz", "dwba") or substrate not in SUBSTRATES:
        raise ValueError(f"Unknown truth {scattering!r} on {substrate!r}")
    TRUTH.update(scattering=scattering, substrate=substrate)


def truth_geometry() -> yt.Geometry:
    return yt.replace(GEOM, **SUBSTRATES[TRUTH["substrate"]])


def draw(index, seed0=20261116):
    """One held-out case: composition, populations, counts and the level of physics."""
    rng = np.random.default_rng(seed0 + index)
    names = list(COMPOSITIONS)
    comp = names[index % len(names)]
    level = "model" if (index // len(names)) % 2 == 0 else "measured"
    shapes = COMPOSITIONS[comp]
    pops = []
    for k, shape in enumerate(shapes):
        if k == 0:
            R = float(np.exp(rng.uniform(np.log(2.0), np.log(8.0))))
        else:                                       # a second size scale, at least 1.8 times apart
            ratio = float(np.exp(rng.uniform(np.log(1.8), np.log(3.0))))
            R = pops[0]["R"] * ratio if pops[0]["R"] * ratio <= 10.0 else pops[0]["R"] / ratio
            R = float(max(R, 1.2))
        pops.append(dict(shape=shape, R=R, sR=float(rng.uniform(0.1, 0.35)),
                         D=float(2 * R * rng.uniform(1.3, 2.5)), s_nn=float(rng.uniform(0.1, 0.3)),
                         share=1.0 if k == 0 else float(rng.uniform(0.3, 1.0))))
    peak = float(10 ** rng.uniform(3, 5))
    return dict(index=index, composition=comp, level=level, populations=pops, peak_counts=peak,
                background=float(10 ** rng.uniform(-3, -1.5)), seed=seed0 + index)


def curve(case):
    """The measured curve of a case: populations summed without noise, then background and Poisson counts."""
    inst = yt.Instrument(peak_counts=0.0)
    if case["level"] == "measured":
        inst = yt.replace(inst, sigma_g=0.005, central=30.0, eta=0.03, gamma=0.004, rough=1.0, xi=50.0, hurst=0.5)
    total, peak_of_particles = np.zeros_like(Q), None
    for k, p in enumerate(case["populations"]):
        film = yt.Film(R=p["R"], sR=p["sR"], D=p["D"], s_nn=p["s_nn"], coupling="lma", scattering=TRUTH["scattering"],
                       cyl_h=2.0 * p["R"] if p["shape"] == "vc" else 0.0)
        own = yt.measured_curve(Q, yt.Case(truth_geometry(), film,
                                           inst if k == 0 else yt.replace(inst, central=0.0, rough=0.0)))
        particles = own["parts"]["particles"]
        if peak_of_particles is None:
            peak_of_particles = float(particles.max())
            total += own["I"]
        else:                                        # the second population at a share of the first one's peak
            total += own["I"] * p["share"] * peak_of_particles / float(particles.max())
    total += case["background"] * float(total.max())
    scale = case["peak_counts"] / float(total[Q > 0.1].max())
    rng = np.random.default_rng(case["seed"])
    counts = rng.poisson(total * scale).astype(float)
    return counts, np.sqrt(np.maximum(counts, 1.0))


__all__ = ["COMPOSITIONS", "GEOM", "Q", "SUBSTRATES", "TRUTH", "curve", "draw", "set_truth", "truth_geometry"]
