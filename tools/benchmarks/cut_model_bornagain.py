"""cut_model against BornAgain on a horizontal cut at the Yoneda angle of Si (Physics and Methods 9.2).

Particles stand on a Si substrate (BornAgain: ``deposit2D`` on the vacuum layer above Si), four-term DWBA,
λ 0.1045 nm, α_i 0.40°, the detector row at the critical angle of Si. The cut's q_∥ = √(q_x² + q_y²) of every
pixel is passed to cut_model (mode ``dwba``, the same δ and β). Cases, each for a sphere and an upright cylinder
(height 6 nm):
  one size      R = 4 nm
  sizes         a number-weighted Gaussian (σ_R/R = 0.2): BornAgain gets a Mixture of the very radii and number
                abundances of cut_model's own quadrature, so the check is on the amplitudes and their sum, not on a
                quadrature
  paracrystal   R = 4 nm with a radial paracrystal, D = 14 nm, Gaussian neighbour distribution σ_nn = 2.8 nm,
                no damping (one size: LMA and DA are the same)
BornAgain's average layer materials are switched off (cut_model's DWBA, like fit1d's and yoneda_truth's, uses the
reflection of the bare substrate); one more row shows the sphere paracrystal with BornAgain's default.
Compared after one scale (BornAgain's absolute units are not cut_model's): the largest and the median relative
deviation over q_∥ = 0.05–2.5 nm⁻¹. Needs BornAgain 24 (the GIMaP release environment has it).
usage: python tools/benchmarks/cut_model_bornagain.py   (writes cut_model_bornagain.json next to itself)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.gimap.features.fitting.domain.cut_model import Geometry, population_terms, sizes  # noqa: E402
from src.gimap.features.fitting.domain.cut_model.shapes import RADIUS_POWER  # noqa: E402

WAVELENGTH, ALPHA_I = 0.1045, 0.40
DELTA, BETA = 3.28e-6 * (WAVELENGTH / 0.1023) ** 2, 5.0e-8 * (WAVELENGTH / 0.1023) ** 4
GEOMETRY = Geometry("dwba", WAVELENGTH, ALPHA_I, None, delta=DELTA, beta=BETA)
HEIGHT = 6.0
N_PHI = 500


def simulate(particles, structure=None, average_materials=False):
    """BornAgain's cut: ``particles`` = [(family, radius, abundance)], one row at the Yoneda angle; (q_∥, I).

    ``average_materials``: BornAgain's default, the layer holding the particles gets their volume-averaged index,
    which changes the reflection the DWBA terms carry; off, the particles sit on the bare substrate as in cut_model."""
    import bornagain as ba

    vacuum, silicon = ba.Vacuum(), ba.RefractiveMaterial("Si", (0.5, 0.5, 0.5), DELTA, BETA)
    gold = ba.RefractiveMaterial("Au", (0.9, 0.8, 0.2), 4.9e-5, 4.9e-6)

    def particle(family, radius):
        shape = ba.Sphere(radius * ba.nm) if family == "sphere" else ba.Cylinder(radius * ba.nm, HEIGHT * ba.nm)
        return ba.Particle(gold, shape)

    if len(particles) == 1:
        content = particle(*particles[0][:2])
    else:
        content = ba.Mixture()
        for family, radius, abundance in particles:
            content.addParticle(particle(family, radius), float(abundance))
    if structure is None:
        layout = ba.Dilute2D(0.001, content)
    else:
        distance, sigma_nn = structure
        layout = ba.RadialParacrystal(content, distance * ba.nm, 0.0)
        layout.setProbabilityDistribution(ba.Profile1DGauss(sigma_nn * ba.nm))
    top = ba.Layer(vacuum)
    top.deposit2D(layout)
    sample = ba.Sample()
    sample.addLayer(top)
    sample.addLayer(ba.Layer(silicon))
    k = 2 * math.pi / WAVELENGTH
    af = math.radians(GEOMETRY.exit_angle_deg)
    phi_max = math.asin(2.6 / (k * math.cos(af)))
    detector = ba.SphericalDetector(N_PHI, 1e-6, phi_max, 1, af - 1e-7, af + 1e-7)
    beam = ba.Beam(1.0, WAVELENGTH * ba.nm, ALPHA_I * ba.deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    simulation.options().setUseAvgMaterials(average_materials)
    result = simulation.simulate()
    image = np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float).reshape(-1)
    phi = 1e-6 + (np.arange(N_PHI) + 0.5) * (phi_max - 1e-6) / N_PHI
    q_y = k * math.cos(af) * np.sin(phi)
    q_x = k * (math.cos(af) * np.cos(phi) - math.cos(math.radians(ALPHA_I)))
    return np.hypot(q_x, q_y), image


def compare(name, q, ba_curve, ours):
    keep = (q >= 0.05) & (q <= 2.5) & (ba_curve > 0)
    ratio = ours[keep] / ba_curve[keep]
    ratio = ratio / np.exp(np.median(np.log(ratio)))
    deviation = np.abs(ratio - 1.0)
    row = dict(name=name, max_rel=float(deviation.max()), median_rel=float(np.median(deviation)),
               q_at_max=float(q[keep][np.argmax(deviation)]), points=int(keep.sum()))
    print(f"{name:45s} max {row['max_rel']:.2e}  median {row['median_rel']:.2e}  at q {row['q_at_max']:.3f}", flush=True)
    return row


def main():
    rows = []
    for family in ("sphere", "vertical_cylinder"):
        q, curve = simulate([(family, 4.0, 1.0)])
        rows.append(compare(f"{family} one size", q, curve,
                            population_terms(family, q, radius=4.0, radius_spread=0.0, geometry=GEOMETRY)[0]))
        radii, weights = sizes.nodes("number_gauss", 4.0, 0.2, power=RADIUS_POWER[family], panels=3)
        q, curve = simulate([(family, r, w) for r, w in zip(radii[0], weights[0])])
        rows.append(compare(f"{family} sizes σ_R/R 0.2 ({radii.shape[1]} radii)", q, curve,
                            population_terms(family, q, radius=4.0, radius_spread=0.2, geometry=GEOMETRY, panels=3)[0]))
        q, curve = simulate([(family, 4.0, 1.0)], structure=(14.0, 2.8))
        rows.append(compare(f"{family} one size, radial paracrystal", q, curve,
                            population_terms(family, q, radius=4.0, radius_spread=0.0, distance=14.0,
                                             distance_spread=0.2, geometry=GEOMETRY)[0]))
    # BornAgain's default: the dense film of the paracrystal (volume fraction ~17 %) changes the reflection
    q, curve = simulate([("sphere", 4.0, 1.0)], structure=(14.0, 2.8), average_materials=True)
    default_avg = compare("sphere paracrystal, BornAgain default (average materials)", q, curve,
                          population_terms("sphere", q, radius=4.0, radius_spread=0.0, distance=14.0,
                                           distance_spread=0.2, geometry=GEOMETRY)[0])
    out = dict(wavelength_nm=WAVELENGTH, alpha_i_deg=ALPHA_I, alpha_f_deg=GEOMETRY.exit_angle_deg, si=dict(delta=DELTA, beta=BETA),
               cylinder_height_nm=HEIGHT, average_materials=False, checks=rows,
               bornagain_default_average_materials=default_avg)
    (HERE / "cut_model_bornagain.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
