"""How well a joint fit of a horizontal and a vertical cut gives the height of upright cylinders when BornAgain's cut
holds what GIMaP's four-term DWBA on a bare substrate leaves out (referee round 5, M8).

``joint_dwba_bornagain.py`` checks the vertical cut forward, against BornAgain with the average layer off. Here
BornAgain simulates both cuts of one sample and GIMaP fits them together as a user would (Fitting ▸ Fit Several
Curves Together: ``application.joint_fit.run_joint_fit``), the substrate Si known, the vertical cut in the four-term
DWBA on it, R, σR/R, h, σh/h, D and σnn/D shared, each curve with its own scale and background.

The sample: upright cylinders, R = 4 nm and H = 6 nm, each Gaussian by number with σ/mean = 0.15 and independent
(BornAgain: a Mixture of 16 × 16 cylinders on Gauss–Legendre nodes over ±4σ, the span of GIMaP's Gaussian law;
BornAgain treats a Mixture in the decoupling approximation, so GIMaP's model is in the DA too), a radial paracrystal
of mean distance D and σnn/D = 0.2, on Si; λ 0.1045 nm, α_i 0.40°. BornAgain's radial paracrystal carries the
density of a hexagonal lattice, 2/(√3 D²), which sets the average layer: the coverage is that density times π⟨R²⟩.
Particles of Au or Si (δ, β of GIMaP's table at this λ).

* Horizontal cut: the row α_f = α_c(Si), q_∥ = √(q_x² + q_y²) from 0.05 to 2.5 nm⁻¹ (471 points).
* Vertical cut: the column at φ_f = asin(0.3 nm⁻¹/k), α_f from 0.02° to 1.2° (300 points), its vacuum
  q_z = k (sin α_i + sin α_f); fitted at the one q_∥ = 0.3 nm⁻¹.

Cases (``CASES``): (a) the average layer off, no roughness (the baseline of the forward check) at D = 14, 24.4 and
12.2 nm; (b) BornAgain's average layer on (``setUseAvgMaterials``; the particles' layer cut into 24 slices, a graded
profile converged to 10⁻³; BornAgain's default of one slice averages up to the tallest cylinder) at coverage 10 %
(D = 24.4 nm), 30 % (14 nm) and 40 % (12.2 nm), Si and Au particles; (c) a rough substrate, σ = 0.5 and 1 nm
(BornAgain's ``Roughness``: K-correlation, Hurst 0.7, lateral correlation 25 nm, error-function transient), once as
BornAgain gives it (the reflection coefficients reduced and the substrate's own diffuse scattering added) and once
with only the reduced reflection (the bare rough substrate's cut, without particles, subtracted); (d) the angular
resolution: a Gaussian detector resolution of σ = 0.01° in α_f and φ_f, and σ = 0.02° with a beam divergence of
σ = 0.01° in α_i (BornAgain's ``ResolutionFunction2DGaussian`` and ``BeamGrazingAngle`` distribution); (e) heights
correlated with radii, H = 1.5 R for every cylinder (GIMaP takes them independent); (f) everything at once: dense Au
(40 %) with the average layer, σ = 1 nm and the coarser resolution.

Each case is fitted noise-free (counts as expected, σ = √counts: the bias itself) and with Poisson noise (``--seeds``
draws), the curves scaled to ``PEAK`` counts at their maximum on a background of ``BACKGROUND`` counts; once with the
whole vertical cut and once with it from 2α_c(Si) = 0.30° (``WINDOWS``), the same counts. Starts near but not at
the truth (``START``). Reported: R, h, D with their 1σ, χ²_r, the relative bias of h, R and D.

BornAgain's average layer is checked first (``uniform_layer_check``): for one size in one slice its vertical cut has
the shape of the four-term DWBA in a uniform layer of the average index on Si, written here from the layer's own
reflection and transmission.
usage: python tools/benchmarks/vertical_height_bias.py [--seeds 10] [--workers 2] [--cases NAME ...] [--out PATH]
       (writes vertical_height_bias.json next to itself; needs BornAgain 24; about 3 minutes with two workers)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.gimap.features.fitting.application import joint_fit as japp  # noqa: E402
from src.gimap.features.fitting.domain import fit_model as fm  # noqa: E402
from src.gimap.features.fitting.domain.joint_fit import SHARED, default_shared  # noqa: E402
from src.gimap.features.fitting.domain.substrates import Substrate, optical_constants  # noqa: E402
from tools.benchmarks.posterior_recovery import _code_state  # noqa: E402

WAVELENGTH, ALPHA_I = 0.1045, 0.40
SUBSTRATE = Substrate("Si")
DELTA, BETA = SUBSTRATE.optical(WAVELENGTH)
ALPHA_C = SUBSTRATE.critical_angle_deg(WAVELENGTH)
K = 2 * math.pi / WAVELENGTH
Q_Y = 0.3
RADIUS, HEIGHT, SPREAD, SIGMA_NN = 4.0, 6.0, 0.15, 0.2
NODES = 16
"""Gauss–Legendre nodes of R and of H in BornAgain's Mixture, over mean·(1 ± 4σ/mean)."""
SLICES = 24
HURST, CORRELATION_LENGTH = 0.7, 25.0
N_PHI, Q_RANGE = 500, (0.05, 2.5)
N_ALPHA, ALPHA_RANGE = 300, (0.02, 1.2)
PAD = 50
"""Pixels simulated beyond each end of a cut (the detector resolution needs them; the fit does not see them)."""
PEAK, BACKGROUND = 1e4, 2.0
WINDOWS = {"full": ALPHA_RANGE[0], "from_2ac": 2.0 * ALPHA_C}
"""The lowest α_f (°) of the vertical cut fitted: all of it, or from twice α_c(Si) (above the Yoneda peak, where the
effective layer and the resolution change the cut most)."""
START = {"R": 4.3, "sigma_R": 0.2, "h": 6.6, "sigma_h": 0.2, "D": 1.07, "sigma_nn": 0.25}
"""Start of every fit: R and h in nm, D as a factor of the true D, the spreads relative."""


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    label: str
    distance: float = 14.0
    material: str = "Au"
    average: bool = False
    roughness: float = 0.0
    roughness_part: str = "all"
    """``all``: BornAgain's cut; ``r``: the bare rough substrate's cut subtracted (the reduced reflection only)."""
    resolution: float = 0.0
    """Detector resolution σ (°) in α_f and φ_f."""
    divergence: float = 0.0
    """Beam divergence σ (°) in α_i."""
    correlated: bool = False
    """H = (HEIGHT/RADIUS)·R for every cylinder."""


DILUTE, DENSE = 24.4, 12.2
CASES = (
    Case("a_off_D14", "a", "average layer off, smooth (D 14 nm)"),
    Case("a_off_D24", "a", "average layer off, smooth (D 24.4 nm)", distance=DILUTE),
    Case("a_off_D12", "a", "average layer off, smooth (D 12.2 nm)", distance=DENSE),
    Case("b_avg_10_Si", "b", "average layer, 10 % Si", distance=DILUTE, material="Si", average=True),
    Case("b_avg_10_Au", "b", "average layer, 10 % Au", distance=DILUTE, average=True),
    Case("b_avg_30_Au", "b", "average layer, 30 % Au (D 14 nm)", average=True),
    Case("b_avg_40_Si", "b", "average layer, 40 % Si", distance=DENSE, material="Si", average=True),
    Case("b_avg_40_Au", "b", "average layer, 40 % Au", distance=DENSE, average=True),
    Case("c_rough_05", "c", "roughness 0.5 nm (r and diffuse)", roughness=0.5),
    Case("c_rough_10", "c", "roughness 1.0 nm (r and diffuse)", roughness=1.0),
    Case("c_rough_05_r", "c", "roughness 0.5 nm (r only)", roughness=0.5, roughness_part="r"),
    Case("c_rough_10_r", "c", "roughness 1.0 nm (r only)", roughness=1.0, roughness_part="r"),
    Case("d_res_001", "d", "resolution σ 0.01°", resolution=0.01),
    Case("d_res_002_div", "d", "resolution σ 0.02°, divergence σ 0.01°", resolution=0.02, divergence=0.01),
    Case("e_correlated", "e", "H = 1.5 R (correlated)", correlated=True),
    Case("f_all", "f", "40 % Au, average layer, roughness 1.0 nm, resolution 0.02° + divergence 0.01°",
         distance=DENSE, average=True, roughness=1.0, resolution=0.02, divergence=0.01),
)


def gauss_legendre(mean: float, rel: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and number weights (sum 1) of a Gaussian by number over mean·(1 ± 4 rel)."""
    x, w = np.polynomial.legendre.leggauss(NODES)
    weights = w * np.exp(-0.5 * (4.0 * x) ** 2)
    return mean * (1.0 + 4.0 * rel * x), weights / weights.sum()


def population(case: Case) -> list[tuple[float, float, float]]:
    """(radius, height, number weight) of every cylinder of BornAgain's Mixture."""
    radii, r_weights = gauss_legendre(RADIUS, SPREAD)
    if case.correlated:
        return [(r, HEIGHT / RADIUS * r, w) for r, w in zip(radii, r_weights)]
    heights, h_weights = gauss_legendre(HEIGHT, SPREAD)
    return [(r, h, w * v) for r, w in zip(radii, r_weights) for h, v in zip(heights, h_weights)]


def coverage(case: Case) -> float:
    """BornAgain's density of a radial paracrystal, 2/(√3 D²), times π⟨R²⟩."""
    radii, weights = gauss_legendre(RADIUS, SPREAD)
    return 2.0 / (math.sqrt(3.0) * case.distance**2) * math.pi * float(weights @ radii**2)


def sample_of(case: Case, particles: bool = True):
    import bornagain as ba

    substrate = ba.RefractiveMaterial("Si", (0.5, 0.5, 0.5), DELTA, BETA)
    top = ba.Layer(ba.Vacuum())
    if particles:
        material = ba.RefractiveMaterial(case.material, (0.9, 0.8, 0.2), *optical_constants(case.material, WAVELENGTH))
        mixture = ba.Mixture()
        for radius, height, weight in population(case):
            mixture.addParticle(ba.Particle(material, ba.Cylinder(radius * ba.nm, height * ba.nm)), float(weight))
        structure = ba.RadialParacrystal(mixture, case.distance * ba.nm, 0.0)
        structure.setProbabilityDistribution(ba.Profile1DGauss(SIGMA_NN * case.distance * ba.nm))
        top.deposit2D(structure)
        if case.average:
            top.setNumberOfSlices(SLICES)
    sample = ba.Sample()
    sample.addLayer(top)
    if case.roughness > 0:
        roughness = ba.Roughness(ba.SelfAffineFractalModel(case.roughness * ba.nm, HURST, CORRELATION_LENGTH * ba.nm),
                                 ba.ErfTransient())
        sample.addLayer(ba.Layer(substrate, roughness))
    else:
        sample.addLayer(ba.Layer(substrate))
    return sample


def simulate(case: Case, detector, shape: tuple[int, int], particles: bool = True) -> np.ndarray:
    """BornAgain's image, ``shape`` = (rows α_f, columns φ_f)."""
    import bornagain as ba

    if case.resolution > 0:
        detector.setResolutionFunction(ba.ResolutionFunction2DGaussian(1e-6, math.radians(case.resolution)))
    beam = ba.Beam(1.0, WAVELENGTH * ba.nm, ALPHA_I * ba.deg)
    simulation = ba.ScatteringSimulation(beam, sample_of(case, particles), detector)
    if case.divergence > 0:
        simulation.addParameterDistribution(ba.ParameterDistribution.BeamGrazingAngle, ba.DistributionGaussian(
            ALPHA_I * ba.deg, math.radians(case.divergence), 15, 3.0))
    simulation.options().setUseAvgMaterials(case.average)
    result = simulation.simulate()
    return np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float).reshape(shape)


def horizontal_cut(case: Case, particles: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(q_∥, I) of the row at α_c(Si). The detector's φ_f reaches below 0 so that a resolution sees both sides. With a
    resolution the row is the middle one of three, 2·10⁻⁴ rad apart in α_f (the φ_f smearing is what changes its
    shape: an upright cylinder's height factor is one number along the row)."""
    import bornagain as ba

    af = math.radians(ALPHA_C)
    phi_max = math.asin(2.6 / (K * math.cos(af)))
    pitch = phi_max / N_PHI
    lo, hi = -PAD * pitch, phi_max + PAD * pitch
    n = N_PHI + 2 * PAD
    rows, step = (3, 2e-4) if case.resolution > 0 else (1, 2e-7)
    detector = ba.SphericalDetector(n, lo, hi, rows, af - 0.5 * rows * step, af + 0.5 * rows * step)
    curve = simulate(case, detector, (rows, n), particles)[rows // 2]
    phi = lo + (np.arange(n) + 0.5) * pitch
    q_y = K * math.cos(af) * np.sin(phi)
    q_x = K * (math.cos(af) * np.cos(phi) - math.cos(math.radians(ALPHA_I)))
    q = np.hypot(q_x, q_y)
    keep = (phi > 0) & (q >= Q_RANGE[0]) & (q <= Q_RANGE[1])
    return q[keep], curve[keep]


def vertical_cut(case: Case, particles: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(α_f in degrees, vacuum q_z, I) of the column at φ_f = asin(Q_Y/k); ``PAD`` pixels beyond each end simulated
    and dropped. With a resolution the column is the middle one of three, 2·10⁻⁴ rad apart (BornAgain's resolution
    needs more than one column)."""
    import bornagain as ba

    phi = math.asin(Q_Y / K)
    lo, hi = (math.radians(a) for a in ALPHA_RANGE)
    pitch = (hi - lo) / N_ALPHA
    lo, hi = lo - PAD * pitch, hi + PAD * pitch
    n = N_ALPHA + 2 * PAD
    columns, step = (3, 2e-4) if case.resolution > 0 else (1, 2e-7)
    detector = ba.SphericalDetector(columns, phi - 0.5 * columns * step, phi + 0.5 * columns * step, n, lo, hi)
    curve = simulate(case, detector, (n, columns), particles)[PAD:PAD + N_ALPHA, columns // 2]
    alpha = (lo + (np.arange(n) + 0.5) * pitch)[PAD:PAD + N_ALPHA]
    qz = K * (math.sin(math.radians(ALPHA_I)) + np.sin(alpha))
    return np.degrees(alpha), qz, curve


def bornagain_cuts(case: Case) -> dict:
    tick = time.perf_counter()
    q, horizontal = horizontal_cut(case)
    alpha, qz, vertical = vertical_cut(case)
    if case.roughness > 0 and case.roughness_part == "r":
        horizontal = horizontal - horizontal_cut(case, particles=False)[1]
        vertical = vertical - vertical_cut(case, particles=False)[2]
    return dict(q=q, horizontal=horizontal, alpha=alpha, qz=qz, vertical=vertical,
                seconds=time.perf_counter() - tick)


def uniform_layer_check() -> dict:
    """BornAgain's vertical cut with the average layer (one size, Au, D = 12.2 nm, one slice: a uniform layer of
    height H) against the four-term DWBA in a uniform layer of the average index on Si: the fields in the layer are
    A e^{−ik_L z} + B e^{ik_L z}, B = r_LS A, matched to vacuum at z = H, and the amplitude is Σ ∫₀^H of the four
    products of the incident and the exit field. Shapes compared after one scale."""
    import bornagain as ba

    case = Case("check", "check", "check", distance=DENSE, average=True)
    delta_p, beta_p = optical_constants("Au", WAVELENGTH)
    fraction = 2.0 / (math.sqrt(3.0) * DENSE**2) * math.pi * RADIUS**2
    alpha_deg, qz, _curve = vertical_cut(replace(case, average=False))

    def column(average: bool) -> np.ndarray:
        material = ba.RefractiveMaterial("Au", (0.9, 0.8, 0.2), delta_p, beta_p)
        structure = ba.RadialParacrystal(ba.Particle(material, ba.Cylinder(RADIUS * ba.nm, HEIGHT * ba.nm)),
                                         DENSE * ba.nm, 0.0)
        structure.setProbabilityDistribution(ba.Profile1DGauss(SIGMA_NN * DENSE * ba.nm))
        top = ba.Layer(ba.Vacuum())
        top.deposit2D(structure)
        sample = ba.Sample()
        sample.addLayer(top)
        sample.addLayer(ba.Layer(ba.RefractiveMaterial("Si", (0.5, 0.5, 0.5), DELTA, BETA)))
        phi = math.asin(Q_Y / K)
        lo, hi = math.radians(ALPHA_RANGE[0]), math.radians(ALPHA_RANGE[1])
        detector = ba.SphericalDetector(1, phi - 1e-7, phi + 1e-7, N_ALPHA, lo, hi)
        simulation = ba.ScatteringSimulation(ba.Beam(1.0, WAVELENGTH * ba.nm, ALPHA_I * ba.deg), sample, detector)
        simulation.options().setUseAvgMaterials(average)
        result = simulation.simulate()
        return np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float).reshape(-1)

    def fields(angle_deg, delta_l, beta_l):
        s = np.sin(np.radians(angle_deg))
        k0 = K * s + 0j
        k_l = K * np.sqrt(s * s - 2.0 * delta_l + 2j * beta_l + 0j)
        k_s = K * np.sqrt(s * s - 2.0 * DELTA + 2j * BETA + 0j)
        r_ls = (k_l - k_s) / (k_l + k_s)
        down, up = np.exp(-1j * k_l * HEIGHT), np.exp(1j * k_l * HEIGHT)
        a = 2.0 * k0 / (k0 * (down + r_ls * up) - k_l * (-down + r_ls * up))
        return a, r_ls * a, k_l

    def layer_model(delta_l, beta_l):
        a_i, b_i, k_i = fields(np.array([ALPHA_I]), delta_l, beta_l)
        a_f, b_f, k_f = fields(alpha_deg, delta_l, beta_l)
        amplitude = 0j
        for coefficient, q in ((a_i * a_f, k_i + k_f), (a_i * b_f, k_i - k_f), (b_i * a_f, k_f - k_i),
                               (b_i * b_f, -(k_i + k_f))):
            amplitude = amplitude + coefficient * (1.0 - np.exp(-1j * q * HEIGHT)) / (1j * q)
        return np.abs(amplitude) ** 2

    out = {}
    for name, average, (delta_l, beta_l) in (("average layer off", False, (0.0, 0.0)),
                                             ("average layer on", True, (fraction * delta_p, fraction * beta_p))):
        ratio = column(average) / layer_model(delta_l, beta_l)
        ratio /= np.median(ratio)
        out[name] = float(np.max(np.abs(ratio - 1.0)))
    out["coverage"] = fraction
    return out


def counts(curve: np.ndarray, seed) -> tuple[np.ndarray, np.ndarray]:
    """(counts, σ): ``PEAK`` at the maximum on ``BACKGROUND``; ``seed`` None: as expected, σ = √counts."""
    expected = PEAK * curve / curve.max() + BACKGROUND
    if seed is None:
        return expected, np.sqrt(expected)
    drawn = np.random.default_rng(seed).poisson(expected).astype(float)
    return drawn, np.sqrt(np.maximum(drawn, 1.0))


def page_model(case: Case) -> fm.FitModel:
    """What a user sets on the page: upright cylinders with their height (a vertical cut is among the curves),
    Gaussian by number, the paracrystal in the DA (BornAgain's Mixture), no beam tail (the cuts start at 0.05 nm⁻¹
    and hold none)."""
    component = replace(fm.new_component("vertical_cylinder", radius=START["R"], distance=START["D"] * case.distance,
                                         law="gaussian", heights=True), coupling="da")
    model = fm.FitModel((component,)).with_values({
        (0, "sigma_R"): START["sigma_R"], (0, "h"): START["h"], (0, "sigma_h"): START["sigma_h"],
        (0, "sigma_nn"): START["sigma_nn"]})
    return model.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)


def fit(task) -> dict:
    """One joint fit: the case's cuts as counts (``seed``), the vertical cut from ``WINDOWS[window]``."""
    case, cuts, seed, window = task
    y_h, s_h = counts(cuts["horizontal"], seed)
    y_v, s_v = counts(cuts["vertical"], None if seed is None else seed + 1000)
    specs = [
        japp.CurveSpec("horizontal", cuts["q"], y_h, s_h, kind="horizontal", alpha_i=ALPHA_I, alpha_f=ALPHA_C,
                       wavelength=WAVELENGTH),
        japp.CurveSpec("vertical", cuts["qz"], y_v, s_v, kind="vertical", alpha_i=ALPHA_I,
                       wavelength=WAVELENGTH, q_parallel=Q_Y),
    ]
    model = page_model(case)
    result = japp.run_joint_fit(specs, model, SUBSTRATE, default_shared(model), method="local",
                                min_exit_angle_ac=0.0 if window == "full" else 2.0)
    fitted = result.models[0]
    row = dict(case=case.name, window=window, seed=seed, chi2_reduced=result.chi2_reduced,
               chi2_per_point={"horizontal": result.curve_chi2[0], "vertical": result.curve_chi2[1]},
               points=list(result.points), converged=result.converged, evaluations=result.evaluations,
               seconds=result.seconds, at_bounds=[japp.key_name(key) for key in result.at_bounds])
    for key in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn"):
        row[key] = float(fitted.get((0, key)).value)
        row[key + "_err"] = float(result.errors.get((SHARED, (0, key)), math.nan))
    return row


def truth(case: Case) -> dict:
    return {"R": RADIUS, "h": HEIGHT, "D": case.distance}


def summary(case: Case, rows: list[dict]) -> dict:
    """Per parameter: the noise-free fit (its value, 1σ and relative bias) and the Poisson draws (mean, standard
    deviation, mean relative bias, mean 1σ reported); χ²_r of both. The noise-free fit's 1σ is scaled by its own
    χ²_r (as every fit's), which is small where the model holds: the draws' 1σ is the one a user sees."""
    exact = [r for r in rows if r["seed"] is None]
    noisy = [r for r in rows if r["seed"] is not None]
    out = {}
    for key, value in truth(case).items():
        entry = {}
        if exact:
            entry.update(noise_free=exact[0][key], noise_free_err=exact[0][key + "_err"],
                         noise_free_bias=exact[0][key] / value - 1.0)
        if noisy:
            values = np.array([r[key] for r in noisy])
            entry.update(seeds_mean=float(values.mean()), seeds_sd=float(values.std(ddof=1)) if values.size > 1 else None,
                         seeds_mean_bias=float(values.mean() / value - 1.0),
                         seeds_mean_err=float(np.nanmean([r[key + "_err"] for r in noisy])))
        out[key] = entry
    out["chi2_reduced"] = dict(noise_free=exact[0]["chi2_reduced"] if exact else None,
                               seeds_mean=float(np.mean([r["chi2_reduced"] for r in noisy])) if noisy else None)
    out["at_bounds"] = sorted({name for r in rows for name in r["at_bounds"]})
    out["not_converged"] = sum(not r["converged"] for r in rows)
    return out


def table_row(case: Case, window: str, s: dict) -> dict:
    """The numbers of the README's table (biases in %)."""
    def pct(value):
        return None if value is None or not math.isfinite(value) else round(100.0 * value, 2)

    h, r = s["h"], s["R"]
    return dict(case=case.name, window=window, label=case.label, coverage=round(coverage(case), 3),
                h_bias_noise_free=pct(h.get("noise_free_bias")), h_bias_seeds=pct(h.get("seeds_mean_bias")),
                h_sd_seeds=h.get("seeds_sd"), h_err_seeds=h.get("seeds_mean_err"),
                R_bias_noise_free=pct(r.get("noise_free_bias")), R_bias_seeds=pct(r.get("seeds_mean_bias")),
                D_bias_noise_free=pct(s["D"].get("noise_free_bias")),
                chi2_reduced_noise_free=s["chi2_reduced"]["noise_free"], chi2_reduced_seeds=s["chi2_reduced"]["seeds_mean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="Poisson draws per case (besides the noise-free fit)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cases", nargs="*", help="names of CASES (default: all)")
    parser.add_argument("--out", type=Path, default=HERE / "vertical_height_bias.json")
    args = parser.parse_args(argv)
    cases = [c for c in CASES if not args.cases or c.name in args.cases]
    check = uniform_layer_check()
    print(f"BornAgain's average layer against a uniform-layer DWBA (one size): largest shape deviation "
          f"{check['average layer on']:.1e} (layer off {check['average layer off']:.1e})", flush=True)
    simulated = {}
    for case in cases:
        simulated[case.name] = bornagain_cuts(case)
        print(f"BornAgain {case.name:16s} {simulated[case.name]['seconds']:.1f} s", flush=True)
    seeds = [None] + list(range(1, args.seeds + 1))
    tasks = [(case, simulated[case.name], seed, window) for case in cases for window in WINDOWS for seed in seeds]
    tick = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(fit, tasks, chunksize=4))
    else:
        rows = [fit(task) for task in tasks]
    out_cases, table = [], []
    for case in cases:
        summaries = {}
        for window in WINDOWS:
            mine = [r for r in rows if r["case"] == case.name and r["window"] == window]
            summaries[window] = summary(case, mine)
            table.append(table_row(case, window, summaries[window]))
            t = table[-1]
            sd = math.nan if t["h_sd_seeds"] is None else 100 * t["h_sd_seeds"] / HEIGHT
            err = math.nan if t["h_err_seeds"] is None else 100 * t["h_err_seeds"] / HEIGHT
            seeds_bias = math.nan if t["h_bias_seeds"] is None else t["h_bias_seeds"]
            seeds_chi2 = math.nan if t["chi2_reduced_seeds"] is None else t["chi2_reduced_seeds"]
            print(f"{case.name:16s} {window:8s} cov {t['coverage']:.2f}  h {t['h_bias_noise_free']:+6.1f} % "
                  f"(draws {seeds_bias:+6.1f} ± {sd:.1f} %, 1σ {err:.1f} %)  R {t['R_bias_noise_free']:+5.1f} %  "
                  f"D {t['D_bias_noise_free']:+5.1f} %  chi2_r {t['chi2_reduced_noise_free']:.2f} / {seeds_chi2:.2f}",
                  flush=True)
        out_cases.append(dict(**asdict(case), coverage=coverage(case), truth=truth(case), summary=summaries,
                              fits=[r for r in rows if r["case"] == case.name],
                              bornagain_seconds=simulated[case.name]["seconds"]))
    out = dict(
        code=_code_state(),
        wavelength_nm=WAVELENGTH, alpha_i_deg=ALPHA_I, substrate="Si (GIMaP's table)", delta=DELTA, beta=BETA,
        horizontal_alpha_f_deg=ALPHA_C, horizontal_q_range=list(Q_RANGE), vertical_q_y=Q_Y,
        vertical_alpha_f_deg=list(ALPHA_RANGE), windows_alpha_f_min_deg=WINDOWS,
        truth=dict(R=RADIUS, H=HEIGHT, sigma_R_over_R=SPREAD, sigma_H_over_H=SPREAD, sigma_nn_over_D=SIGMA_NN,
                   bornagain_nodes=f"{NODES} x {NODES} Gauss-Legendre over +-4 sigma (correlated: {NODES})",
                   interference="BornAgain RadialParacrystal, Gaussian neighbour distance; DA (Mixture)",
                   density="2/(sqrt(3) D^2) (BornAgain's radial paracrystal)"),
        average_layer_slices=SLICES, uniform_layer_check=check, roughness_model=dict(model="SelfAffineFractalModel (K-correlation)", hurst=HURST,
                                                          lateral_correlation_nm=CORRELATION_LENGTH,
                                                          transient="ErfTransient"),
        counts=dict(peak=PEAK, background=BACKGROUND, seeds=args.seeds),
        fit=dict(call="application.joint_fit.run_joint_fit, method local (Refine)", substrate="Si known",
                 shared="R, sigma_R, h, sigma_h, D, sigma_nn", coupling="da", law="gaussian (by number)",
                 upright_height_coherence="squared number-average of the complex axial amplitude / mean axial intensity",
                 beam_tail="off", vertical_q_parallel=Q_Y, start=START),
        fit_seconds=time.perf_counter() - tick, table=table, cases=out_cases)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
