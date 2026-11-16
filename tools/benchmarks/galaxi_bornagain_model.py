"""BornAgain's reference model of the GALAXI frame, on the Yoneda cut that GIMaP analyses.

1. BornAgain 24.1 simulates its own GALAXI model (rawEx/fit/scatter2d/hardcode_galaxi.py of r23: Ag spheres R = 5.75 nm,
   monodisperse, 18.5 nm deep in HMDSO 18.5 nm / PTFE 22.1 nm / Si, radial paracrystal D = 53.6 nm, disorder 10.5 nm,
   kappa 17.5, domain 20 um; roughness 1.1 and 2.3 nm) on the band of the measured cut: rows 605.5–609.5 of the frame
   (alpha_f 0.169–0.192 deg, the Yoneda row 607.5 at 0.18 deg), averaged, at the measured q_y.
2. How well that model describes the measured cut: one scale and one background, on q_y > 0.3 nm^-1 (the beam tail,
   absent from the model, is negligible there) and on the whole cut.
3. What GIMaP Fitting recovers from it: the simulated cut, scaled to the measured counts, plus the measured background,
   with Poisson noise (3 seeds), fitted with the models of galaxi_fit.py. Needs BornAgain (24.1).
usage: python tools/benchmarks/galaxi_bornagain_model.py
"""
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import galaxi_fit as gb  # noqa: E402

WAVELENGTH_NM, ALPHA_I = 0.134, 0.463
PIXEL_DEG = math.degrees(math.atan(0.172 / 1730.0))
ALPHA_F = 0.18
ROWS = 5


def simulate(q, lognormal=False):
    """The cut of the model; ``lognormal``: the log-normal radii (median 5.75 nm, sigma 0.4) that r23 defines but
    comments out, 21 radii uniform in ln R over +-3 sigma, weighted by number (a Mixture: BornAgain then couples them
    by DA, with kappa)."""
    import bornagain as ba
    vac = ba.Vacuum()
    si = ba.RefractiveMaterial("Si", (0.5, 0.5, 0.5), 5.7816e-6, 1.0229e-7)
    ag = ba.RefractiveMaterial("Ag", (0.8, 0.8, 0.8), 2.2475e-5, 1.6152e-6)
    ptfe = ba.RefractiveMaterial("PTFE", (0.3, 0.6, 0.3), 5.20509e-6, 1.9694e-8)
    hmdso = ba.RefractiveMaterial("HMDSO", (0.3, 0.3, 0.6), 2.0888e-6, 1.3261e-8)
    def particle(radius):
        p = ba.Particle(ag, ba.Sphere(radius * ba.nm))
        p.translate(ba.R3(0, 0, -18.5 * ba.nm))
        return p

    if lognormal:
        sigma = 0.4
        x = np.linspace(-3 * sigma, 3 * sigma, 21)
        sphere = ba.Mixture()
        for value, weight in zip(5.75 * np.exp(x), np.exp(-0.5 * (x / sigma) ** 2)):
            sphere.addParticle(particle(float(value)), float(weight))
    else:
        sphere = particle(5.75)
    structure = ba.RadialParacrystal(sphere, 53.6 * ba.nm, 1e6 * ba.nm)  # 24.1: the structure holds the particle
    structure.setKappa(17.5)
    structure.setDomainSize(2e4 * ba.nm)
    structure.setProbabilityDistribution(ba.Profile1DGauss(10.5 * ba.nm))
    rough_hmdso = ba.Roughness(ba.SelfAffineFractalModel(1.1 * ba.nm, 0.3, 5 * ba.nm), ba.TanhTransient())
    rough_ptfe = ba.Roughness(ba.SelfAffineFractalModel(2.3 * ba.nm, 0.3, 5 * ba.nm), ba.TanhTransient())
    hmdso_layer = ba.Layer(hmdso, 18.5 * ba.nm, rough_hmdso)
    hmdso_layer.deposit2D(structure)  # on the bottom of the layer, as the r23 translation by -18.5 nm
    sample = ba.Sample()
    sample.addLayer(ba.Layer(vac))
    sample.addLayer(hmdso_layer)
    sample.addLayer(ba.Layer(ptfe, 22.1 * ba.nm, rough_ptfe))
    sample.addLayer(ba.Layer(si))
    k = 2 * math.pi / WAVELENGTH_NM
    af = math.radians(ALPHA_F)
    phi_max = math.asin(min(1.0, q.max() * 1.02 / (k * math.cos(af))))
    n = 1200
    half = math.radians(PIXEL_DEG * ROWS / 2)
    detector = ba.SphericalDetector(n, 1e-6, phi_max, ROWS, af - half, af + half)
    beam = ba.Beam(1.2e7, WAVELENGTH_NM * ba.nm, ALPHA_I * ba.deg)
    result = ba.ScatteringSimulation(beam, sample, detector).simulate()
    image = np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float)
    image = image.reshape(ROWS, n) if image.size == ROWS * n else image.reshape(n, ROWS).T
    line = image.mean(axis=0)
    phi = 1e-6 + (np.arange(n) + 0.5) * (phi_max - 1e-6) / n
    q_line = k * math.cos(af) * np.sin(phi)
    return np.interp(q, q_line, line)


def scale_background(model, data, sigma, keep):
    a = np.column_stack([model[keep], np.ones(keep.sum())]) / sigma[keep, None]
    coef, *_ = np.linalg.lstsq(a, data[keep] / sigma[keep], rcond=None)
    resid = (data[keep] - (coef[0] * model[keep] + coef[1])) / sigma[keep]
    return dict(scale=float(coef[0]), background=float(coef[1]), chi2_red=float(np.sum(resid**2) / (keep.sum() - 2)),
                points=int(keep.sum()))


def fit_job(job):
    variant, seed, q, intensity, sigma = job
    import galaxi_fit as g
    g.curve = lambda: (q, intensity, sigma)
    out = g.run((variant, 1729))
    out["noise_seed"] = seed
    return out


if __name__ == "__main__":
    q, measured, sigma = gb.curve()
    describe = {}
    for name, lognormal in (("r23_as_written", False), ("r23_lognormal", True)):
        y = simulate(q, lognormal)
        describe[name] = dict(high_q=scale_background(y, measured, sigma, q > 0.3),
                              whole=scale_background(y, measured, sigma, q > 0))
    print(json.dumps(describe, indent=1))
    model = simulate(q, True)
    fit = describe["r23_lognormal"]["high_q"]
    expected = fit["scale"] * model + fit["background"]
    jobs = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        counts = rng.poisson(expected * 5) / 5.0  # 5 pixels per point, as the measured cut
        s = np.sqrt(np.maximum(counts * 5, 1.0)) / 5.0
        for variant in ("lma", "da", "lma_paths", "da_paths", "lma_paths_fixS"):
            jobs.append((variant, seed, q, counts, s))
    with Pool(12) as pool:
        fits = list(pool.imap_unordered(fit_job, jobs))
    sigma_ln = 0.4  # log-normal moments: by number, and weighted by R^6 (the intensity weights of Fitting's sphere)
    truth = dict(gb.REFERENCE, R_number_mean=5.75 * math.exp(sigma_ln**2 / 2),
                 R_intensity_mean=5.75 * math.exp(6 * sigma_ln**2 + sigma_ln**2 / 2),
                 sR_relative=math.sqrt(math.exp(sigma_ln**2) - 1))
    out = dict(describe=describe, truth=truth, fits=fits,
               curve=dict(q=q.tolist(), measured=measured.tolist(), model=(fit["scale"] * model + fit["background"]).tolist()))
    (HERE / "galaxi_bornagain_model.json").write_text(json.dumps(out, indent=1))
    print("BornAgain model on the measured cut:", describe)
    for r in sorted(fits, key=lambda r: (r["variant"], r["noise_seed"])):
        print(f"{r['variant']:16s} noise {r['noise_seed']} chi2 {r['chi2']:6.3f}  R {r['R']:.2f} sR {r['sR']:.3f}  "
              f"D {r['D']:.1f}  s_nn {r['s_nn']:.3f}  |r| {r['refl_amp']:.2f}")
