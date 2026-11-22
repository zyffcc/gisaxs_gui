"""Recovery of known crystallite textures and the orientation fit to the public pygid P08 frame."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.analyze.application.indexing import phase_from_cif
from src.gimap.features.analyze.domain.crystal import fold_profile, reflections

CIF = ROOT / "tests" / "data" / "crystal"
EXTERNAL = ROOT / "tests" / "data" / "external"
# A non-basal fibre in MAPbI3, a basal fibre in PbI2. Each ring has an independent scale.
TEXTURES = {
    "MAPbI3_tetragonal": ((1, 1, 0), 16., 0.38, ((0.980, 1.015), (1.398, 1.426), (2.202, 2.252))),
    "PbI2": ((0, 0, 1), 22., 0.29, ((0.89, 0.91), (1.580, 1.605), (1.813, 1.846))),
}
SEED = 41027


def _binned_density(psi, width):
    """Direct integration of the two spherical Gaussian lobes, independent of the fitting kernel."""
    sigma = np.radians(width) / np.sqrt(8. * np.log(2.))
    norm = quad(lambda angle: np.sin(angle) * np.exp(-0.5 * (angle / sigma)**2), 0., np.pi)[0]
    az_nodes, az_weights = roots_legendre(96)
    azimuth = np.cos(np.pi * (az_nodes + 1.))
    bin_nodes, bin_weights = roots_legendre(16)
    chi = np.radians(np.arange(90)[:, None] + 0.5 * (bin_nodes + 1.))
    dot = np.cos(chi)[..., None] * np.cos(psi) + np.sin(chi)[..., None] * np.sin(psi) * azimuth
    gamma = np.arccos(np.clip(dot, -1., 1.))
    lobes = np.exp(-0.5 * (gamma / sigma)**2) + np.exp(-0.5 * ((np.pi - gamma) / sigma)**2)
    return ((lobes @ az_weights) * np.sin(chi)) @ bin_weights / (4. * norm)


@lru_cache(maxsize=8)
def expected_rings(name, axis=None, family_reweight=()):
    """Poisson means on each side of a ring and the bins surviving the geometry cut."""
    default_axis, width, phi, windows = TEXTURES[name]
    phase = phase_from_cif(CIF / (name + ".cif")).phase()
    director = phase.cell.plane_normal(default_axis if axis is None else axis)
    found = reflections(phase, max(high for _, high in windows))
    chi = np.arange(90) + 0.5
    isotropic = (np.cos(np.radians(chi - 0.5)) - np.cos(np.radians(chi + 0.5))) / np.radians(1.)
    rings = []
    for index, (low, high) in enumerate(windows):
        selected = (found.q >= low) & (found.q <= high)
        q = float(np.average(found.q[selected], weights=found.f2[selected]))
        cosines = np.abs(found.q_vectors[selected] @ director) / found.q[selected]
        unique, membership = np.unique(np.round(cosines, 10), return_inverse=True)
        intensity = found.f2[selected].copy()
        for indices, factor in family_reweight:
            equivalents = {tuple(sign * (indices @ operation.rotation))
                           for operation in phase.operations for sign in (-1, 1)}
            same = [tuple(hkl) in equivalents for hkl in found.hkl[selected]]
            intensity[same] *= factor
        weights = np.bincount(membership, weights=intensity)
        oriented = sum(w * _binned_density(np.arccos(np.clip(c, 0., 1.)), width)
                       for c, w in zip(unique, weights)) / weights.sum()
        scale = (650., 400., 800.)[index]
        background = 45.
        before_folding = background + scale * ((1. - phi) * oriented + phi * isotropic) / np.sin(np.radians(chi))
        # qz = q cos χ = k (sin αf + sin αi); include a detector gap and a missing wedge.
        min_exit = np.sin(np.radians(0.3))
        sin_exit = q * np.cos(np.radians(chi)) / (2. * np.pi / 1.0) - np.sin(np.radians(0.15))
        measured = (sin_exit >= min_exit) & (chi >= 5.) & ~((chi >= 42.) & (chi < 45.))
        rings.append((q, low, high, before_folding, measured, background))
    return phase, rings


def synthetic(name, seed, *, second_axis=None, share=0., family_reweight=()):
    from src.gimap.features.analyze.domain.crystal.texture import TextureRing

    phase, expected = expected_rings(name, family_reweight=family_reweight)
    second = expected_rings(name, second_axis, family_reweight)[1] if second_axis else None
    rng = np.random.default_rng(seed)
    chi = np.arange(90) + 0.5
    rings = []
    for i, (q, low, high, mean, measured, background) in enumerate(expected):
        if second is not None:
            mean = (1. - share) * mean + share * second[i][3]
        observed = rng.poisson(mean, size=(2, len(mean))).astype(float)
        observed[:, ~measured] = np.nan
        flat = observed.ravel()
        profile = fold_profile(np.r_[-chi, chi], flat, np.sqrt(np.maximum(flat, 1.)), background)
        rings.append(TextureRing(i, q, low, high, profile))
    return phase, rings


def coverage(name, trials):
    from src.gimap.features.analyze.domain.crystal.texture import fit_crystal_orientation

    axis, width, phi, _ = TEXTURES[name]
    counts = {"axis": 0, "includes_true_axis": 0, "determined": 0, "correct_and_determined": 0,
              "fwhm_deg": 0, "fraction_iso": 0}
    fits = []
    for seed in range(SEED, SEED + trials):
        fit = fit_crystal_orientation(*synthetic(name, seed))
        correct = fit.get("axes") == [list(axis)]
        counts["axis"] += correct
        counts["includes_true_axis"] += list(axis) in fit.get("axes", [])
        counts["determined"] += bool(fit.get("determined"))
        counts["correct_and_determined"] += correct and bool(fit.get("determined"))
        for key, truth, err in (("fwhm_deg", width, "fwhm_err_deg"), ("fraction_iso", phi, "fraction_iso_err")):
            estimate, sigma = fit.get(key), fit.get(err)
            if estimate is not None and sigma is not None:
                counts[key] += abs(estimate - truth) <= sigma
        fits.append({key: fit.get(key) for key in ("axes", "fwhm_deg", "fwhm_err_deg", "fraction_iso",
                                                   "fraction_iso_err", "delta_bic", "determined")})
    return {"phase": name, "truth": {"axis": axis, "fwhm_deg": width, "fraction_iso": phi}, "trials": trials,
            "successes": counts, "first_seed": SEED, "fits": fits}


def public_frame(*, with_phases=True):
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application.pipeline import PipelineOptions

    info = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))["giwaxs_p08_mapi"]
    cifs = tuple(str(CIF / (name + ".cif")) for name in TEXTURES) if with_phases else ()
    options = PipelineOptions(technique="giwaxs", calibration=str(EXTERNAL / info["poni"]),
                              incidence_deg=info["incidence_deg"], phases=cifs)
    with tempfile.TemporaryDirectory() as tmp:
        session = open_headless_session(EXTERNAL / info["frame"], saved_profiles=False, out_dir=tmp)
        try:
            report = session.run_pipeline(options)
        finally:
            session.close()
    if not report.get("ok"):
        raise RuntimeError(f"P08 automatic analysis failed: {report.get('failed')}")
    if with_phases and not report.get("crystal_orientation"):
        raise RuntimeError("P08 automatic analysis did not return a crystal orientation block.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("synthetic", "public", "all"), default="all")
    parser.add_argument("--trials", default=24, type=int, help="independent Poisson draws per phase")
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()
    if arguments.trials < 1:
        parser.error("--trials must be positive")
    results = {}
    if arguments.part in ("public", "all"):
        with tempfile.TemporaryDirectory() as home:
            os.environ["GIMAP_HOME"] = home
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            if os.name == "nt":
                os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
            results["public"] = public_frame()
    if arguments.part in ("synthetic", "all"):
        results["synthetic"] = [coverage(name, arguments.trials) for name in TEXTURES]
        for item in results["synthetic"]:
            print(item["phase"], item["successes"], "of", item["trials"])
    if "public" in results:
        for fit in results["public"].get("crystal_orientation", []):
            print(fit["phase"], fit.get("axes"), "FWHM", fit.get("fwhm_deg"), "phi", fit.get("fraction_iso"),
                  "delta BIC", fit.get("delta_bic"), "Hermans", fit.get("herman_crystal"))
            for ring in fit.get("rings", []):
                print(ring["q"], ring["reduced_chi2"], ring["herman_measured"])
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
