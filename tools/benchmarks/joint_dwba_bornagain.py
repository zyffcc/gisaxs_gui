"""The four-term DWBA level of Fitting and the vertical cut of a joint fit against BornAgain (Physics and Methods 9.17,
9.18).

Particles stand on a substrate of known index (BornAgain: ``deposit2D`` on the vacuum layer above it, average layer
materials off, as ``cut_model_bornagain.py``), λ 0.1045 nm, α_i 0.40°. The substrate's δ and β are those of GIMaP's
table (``domain.substrates``: Si at this λ); BornAgain gets the same numbers.

* Horizontal cut at the Yoneda row of Si: Fitting's model with the substrate known (``fit_model.evaluate``, level
  *Four-term DWBA*) for spheres (one size; sizes by number, σ_R/R = 0.2, BornAgain given the very radii and number
  weights of the quadrature; a radial paracrystal D = 14 nm, σ_nn = 2.8 nm) and upright cylinders (H = 6 nm).
* Vertical cut at φ_f = asin(0.3 nm⁻¹ / k) (q_y ≈ 0.3 nm⁻¹), α_f from 0.02° to 1.2°: ``vertical_cut`` for spheres
  (one size, sizes, a paracrystal) and upright cylinders (one size; radii and heights by number, σ = 0.15 each),
  each point at BornAgain's own q_∥ = √(q_x² + q_y²); and once with the cut's single q_∥ = q_y (the model as a fit uses
  it), which shows what leaving out q_x along the cut costs.

Compared after one scale: the largest and the median relative deviation, and where the largest lies.
usage: python tools/benchmarks/joint_dwba_bornagain.py   (writes joint_dwba_bornagain.json next to itself)
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

from src.gimap.features.fitting.domain import fit_model as fm  # noqa: E402
from src.gimap.features.fitting.domain.cut_model import sizes  # noqa: E402
from src.gimap.features.fitting.domain.cut_model.structure import paracrystal  # noqa: E402
from src.gimap.features.fitting.domain.substrates import Substrate  # noqa: E402
from src.gimap.features.fitting.domain.vertical_cut import VerticalCut, cylinder_averages, sphere_averages  # noqa: E402

WAVELENGTH, ALPHA_I = 0.1045, 0.40
SUBSTRATE = Substrate("Si")
DELTA, BETA = SUBSTRATE.optical(WAVELENGTH)
ALPHA_C = SUBSTRATE.critical_angle_deg(WAVELENGTH)
HEIGHT = 6.0
K = 2 * math.pi / WAVELENGTH
Q_Y = 0.3
N_PHI, N_ALPHA = 500, 300
ALPHA_RANGE = (0.02, 1.2)


def sample_of(particles, structure=None):
    import bornagain as ba

    vacuum, substrate = ba.Vacuum(), ba.RefractiveMaterial("Si", (0.5, 0.5, 0.5), DELTA, BETA)
    gold = ba.RefractiveMaterial("Au", (0.9, 0.8, 0.2), 4.9e-5, 4.9e-6)

    def particle(family, radius, height):
        shape = ba.Sphere(radius * ba.nm) if family == "sphere" else ba.Cylinder(radius * ba.nm, height * ba.nm)
        return ba.Particle(gold, shape)

    if len(particles) == 1:
        content = particle(*particles[0][:3])
    else:
        content = ba.Mixture()
        for family, radius, height, abundance in particles:
            content.addParticle(particle(family, radius, height), float(abundance))
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
    sample.addLayer(ba.Layer(substrate))
    return sample


def run(sample, detector):
    import bornagain as ba

    beam = ba.Beam(1.0, WAVELENGTH * ba.nm, ALPHA_I * ba.deg)
    simulation = ba.ScatteringSimulation(beam, sample, detector)
    simulation.options().setUseAvgMaterials(False)
    result = simulation.simulate()
    return np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float)


def horizontal(particles, structure=None):
    """BornAgain's row at the Yoneda angle: (q_∥, I)."""
    import bornagain as ba

    af = math.radians(ALPHA_C)
    phi_max = math.asin(2.6 / (K * math.cos(af)))
    detector = ba.SphericalDetector(N_PHI, 1e-6, phi_max, 1, af - 1e-7, af + 1e-7)
    image = run(sample_of(particles, structure), detector).reshape(-1)
    phi = 1e-6 + (np.arange(N_PHI) + 0.5) * (phi_max - 1e-6) / N_PHI
    q_y = K * math.cos(af) * np.sin(phi)
    q_x = K * (math.cos(af) * np.cos(phi) - math.cos(math.radians(ALPHA_I)))
    return np.hypot(q_x, q_y), image


def vertical(particles, structure=None):
    """BornAgain's column at φ_f = asin(Q_Y/k): (α_f in degrees, q_∥ of each pixel, I)."""
    import bornagain as ba

    phi = math.asin(Q_Y / K)
    lo, hi = (math.radians(a) for a in ALPHA_RANGE)
    detector = ba.SphericalDetector(1, phi - 1e-7, phi + 1e-7, N_ALPHA, lo, hi)
    image = run(sample_of(particles, structure), detector).reshape(-1)
    alpha = lo + (np.arange(N_ALPHA) + 0.5) * (hi - lo) / N_ALPHA
    q_y = K * np.cos(alpha) * math.sin(phi)
    q_x = K * (np.cos(alpha) * math.cos(phi) - math.cos(math.radians(ALPHA_I)))
    return np.degrees(alpha), np.hypot(q_x, q_y), image


def compare(name, x, reference, ours, low=None, high=None):
    keep = reference > 0
    if low is not None:
        keep &= (x >= low) & (x <= high)
    ratio = ours[keep] / reference[keep]
    ratio = ratio / np.exp(np.median(np.log(ratio)))
    deviation = np.abs(ratio - 1.0)
    row = dict(name=name, max_rel=float(deviation.max()), median_rel=float(np.median(deviation)),
               x_at_max=float(x[keep][np.argmax(deviation)]), points=int(keep.sum()))
    print(f"{name:60s} max {row['max_rel']:.2e}  median {row['median_rel']:.2e}  at {row['x_at_max']:.4g}", flush=True)
    return row


def fitting_model(family, radius, spread=0.0, structure=None):
    """Fitting's model of the horizontal cut, the substrate known (Si of the table), at the Yoneda row."""
    component = fm.new_component(family, radius=radius, distance=None if structure is None else structure[0],
                                 structure=structure is not None)
    model = fm.FitModel((component,), substrate=SUBSTRATE)
    values = {(0, "sigma_R"): spread, ("globals", "alpha_i"): ALPHA_I, ("globals", "alpha_f"): ALPHA_C,
              ("globals", "wavelength"): WAVELENGTH}
    if structure is not None:
        values[(0, "sigma_nn")] = structure[1] / structure[0]
    return model.with_values(values)


def vertical_model(alpha_deg, q_par, terms):
    """The vertical cut at each point's q_∥ (``q_par`` an array) or at one (a number): Σ over ``terms`` of
    (weight, family, radius, height) one-size averages, weighted as BornAgain's Mixture by number."""
    alpha_deg = np.asarray(alpha_deg, float)
    q_par = np.broadcast_to(np.asarray(q_par, float), alpha_deg.shape)
    sin_i = math.sin(math.radians(ALPHA_I))
    qz = K * (sin_i + np.sin(np.radians(alpha_deg)))
    out = np.zeros(alpha_deg.size)
    for point in range(alpha_deg.size):
        cut = VerticalCut(float(q_par[point]), WAVELENGTH, ALPHA_I, DELTA, BETA)
        paths = cut.paths(qz[point:point + 1])
        for weight, family, radius, height in terms:
            if family == "sphere":
                form, _c = sphere_averages(cut.q_parallel, *paths, "number_gauss", radius, 0.0, False)
                out[point] += weight * radius**6 * form[0]
            else:
                form, _c = cylinder_averages(cut.q_parallel, *paths, "number_gauss", radius, 0.0, height, 0.0, False)
                out[point] += weight * radius**4 * height**2 * form[0]
    return out


def main():
    rows = []
    q, curve = horizontal([("sphere", 4.0, HEIGHT, 1.0)])
    rows.append(compare("horizontal: sphere one size", q, curve, fm.evaluate(fitting_model("sphere", 4.0), q), 0.05, 2.5))
    radii, weights = sizes.nodes("number_gauss", 4.0, 0.2, power=6, panels=3)
    q, curve = horizontal([("sphere", r, HEIGHT, w) for r, w in zip(radii[0], weights[0])])
    from src.gimap.features.fitting.domain.cut_model import population_terms

    geometry = fm.grazing(fitting_model("sphere", 4.0))
    rows.append(compare(f"horizontal: sphere sizes σ_R/R 0.2 ({radii.shape[1]} radii)", q, curve,
                        population_terms("sphere", q, radius=4.0, radius_spread=0.2, geometry=geometry, panels=3)[0],
                        0.05, 2.5))
    q, curve = horizontal([("sphere", 4.0, HEIGHT, 1.0)], structure=(14.0, 2.8))
    rows.append(compare("horizontal: sphere one size, radial paracrystal", q, curve,
                        fm.evaluate(fitting_model("sphere", 4.0, structure=(14.0, 2.8)), q), 0.05, 2.5))
    q, curve = horizontal([("vertical_cylinder", 4.0, HEIGHT, 1.0)])
    rows.append(compare("horizontal: upright cylinder one size", q, curve,
                        fm.evaluate(fitting_model("vertical_cylinder", 4.0), q), 0.05, 2.5))

    alpha, q_par, curve = vertical([("sphere", 4.0, HEIGHT, 1.0)])
    rows.append(compare("vertical: sphere one size", alpha, curve, vertical_model(alpha, q_par, [(1.0, "sphere", 4.0, HEIGHT)])))
    rows.append(compare("vertical: sphere one size, one q_∥ = q_y (as fitted)", alpha, curve,
                        vertical_model(alpha, Q_Y, [(1.0, "sphere", 4.0, HEIGHT)])))
    alpha, q_par, curve = vertical([("sphere", r, HEIGHT, w) for r, w in zip(radii[0], weights[0])])
    rows.append(compare(f"vertical: sphere sizes σ_R/R 0.2 ({radii.shape[1]} radii)", alpha, curve,
                        vertical_model(alpha, q_par, [(w, "sphere", r, HEIGHT) for r, w in zip(radii[0], weights[0])])))
    alpha, q_par, curve = vertical([("sphere", 4.0, HEIGHT, 1.0)], structure=(14.0, 2.8))
    ours = vertical_model(alpha, q_par, [(1.0, "sphere", 4.0, HEIGHT)]) * paracrystal(q_par, 14.0, 0.2)[0]
    rows.append(compare("vertical: sphere one size, radial paracrystal", alpha, curve, ours))
    alpha, q_par, curve = vertical([("vertical_cylinder", 4.0, HEIGHT, 1.0)])
    rows.append(compare("vertical: upright cylinder one size", alpha, curve,
                        vertical_model(alpha, q_par, [(1.0, "cylinder", 4.0, HEIGHT)])))
    c_radii, c_weights = sizes.nodes("number_gauss", 4.0, 0.15, power=4, panels=1)
    heights, h_weights = sizes.nodes("number_gauss", HEIGHT, 0.15, power=2, panels=1)
    pairs = [(r, h, w * v) for r, w in zip(c_radii[0], c_weights[0]) for h, v in zip(heights[0], h_weights[0])]
    alpha, q_par, curve = vertical([("vertical_cylinder", r, h, w) for r, h, w in pairs])
    rows.append(compare(f"vertical: upright cylinder, radii and heights σ 0.15 ({len(pairs)} particles)", alpha, curve,
                        vertical_model(alpha, q_par, [(w, "cylinder", r, h) for r, h, w in pairs])))
    out = dict(wavelength_nm=WAVELENGTH, alpha_i_deg=ALPHA_I, substrate="Si (GIMaP's table)", delta=DELTA, beta=BETA,
               horizontal_alpha_f_deg=ALPHA_C, vertical_q_y=Q_Y, vertical_alpha_f_deg=list(ALPHA_RANGE),
               cylinder_height_nm=HEIGHT, average_materials=False, checks=rows)
    (HERE / "joint_dwba_bornagain.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
