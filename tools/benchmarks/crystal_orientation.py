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
MIXTURE = "MAPbI3 sharp + broad"
MIXED_WIDTHS = (8., 36.)
SHARP_SHARE = 0.30  # of the oriented material
# Known synthetic film constants at the simulated wavelength, independent of CIF optics.
DELTA, BETA = 4.2e-6, 3.1e-7


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
def expected_rings(name, axis=None, family_reweight=(), width_deg=None):
    """Poisson means on each side of a ring and the bins surviving the geometry cut."""
    default_axis, width, phi, windows = TEXTURES[name]
    if width_deg is not None:
        width = width_deg
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

    with_optics = name == MIXTURE
    if with_optics:
        from src.gimap.features.analyze.domain.crystal.film_optics import exit_transmission

        name = "MAPbI3_tetragonal"
        phase, expected = expected_rings(name, width_deg=MIXED_WIDTHS[0])
        second = expected_rings(name, width_deg=MIXED_WIDTHS[1])[1]
        share = 1. - SHARP_SHARE
    else:
        phase, expected = expected_rings(name, family_reweight=family_reweight)
        second = expected_rings(name, second_axis, family_reweight)[1] if second_axis else None
    rng = np.random.default_rng(seed)
    chi = np.arange(90) + 0.5
    rings = []
    for i, (q, low, high, mean, measured, background) in enumerate(expected):
        if second is not None:
            mean = (1. - share) * mean + share * second[i][3]
        transmission = None
        if with_optics:
            # Each synthetic detector bin samples its central exit angle. Evaluate the
            # truth with the real/imaginary parts of the transmitted normal wavevector.
            u = q * np.cos(np.radians(chi)) / (2. * np.pi) - np.sin(np.radians(0.15))
            measured = (u > 0.) & (chi >= 5.) & ~((chi >= 42.) & (chi < 45.))
            vacuum = np.maximum(u, 0.)
            real_sq = vacuum**2 - 2*DELTA
            modulus = np.hypot(real_sq, 2*BETA)
            real = np.sqrt((modulus + real_sq) / 2)
            truth = 4 * vacuum**2 / (vacuum**2 + 2*vacuum*real + modulus)
            mean = background + (mean - background) * truth
            transmission = exit_transmission(u, DELTA, BETA)
        observed = rng.poisson(mean, size=(2, len(mean))).astype(float)
        observed[:, ~measured] = np.nan
        flat = observed.ravel()
        profile = fold_profile(np.r_[-chi, chi], flat, np.sqrt(np.maximum(flat, 1.)), background)
        rings.append(TextureRing(i, q, low, high, profile, transmission=transmission))
    return phase, rings


def coverage(name, trials):
    from src.gimap.features.analyze.domain.crystal.texture import fit_crystal_orientation

    mixed = name == MIXTURE
    axis, width, phi, _ = TEXTURES["MAPbI3_tetragonal" if mixed else name]
    widths = MIXED_WIDTHS if mixed else (width,)
    fractions = [(1.-phi)*share for share in (SHARP_SHARE, 1.-SHARP_SHARE)] if mixed else [1.-phi]
    counts = {"axis": 0, "includes_true_axis": 0, "determined": 0, "correct_and_determined": 0,
              "width_within_1sigma": [0]*len(widths), "fraction_within_1sigma": [0]*len(widths),
              "fraction_iso": 0}
    fits = []
    for seed in range(SEED, SEED + trials):
        fit = fit_crystal_orientation(*synthetic(name, seed))
        correct = fit.get("axes") == [list(axis)] * len(widths)
        counts["axis"] += correct
        counts["includes_true_axis"] += list(axis) in fit.get("axes", [])
        counts["determined"] += bool(fit.get("determined"))
        counts["correct_and_determined"] += correct and bool(fit.get("determined"))
        if fit.get("fraction_iso_err") is not None:
            counts["fraction_iso"] += abs(fit["fraction_iso"] - phi) <= fit["fraction_iso_err"]
        if correct:
            for i, pop in enumerate(sorted(fit["populations"], key=lambda p: p["fwhm_deg"])):
                for target, key, err, counter in ((widths[i], "fwhm_deg", "fwhm_err_deg", "width_within_1sigma"),
                                                  (fractions[i], "fraction", "fraction_err", "fraction_within_1sigma")):
                    sigma = pop[err]
                    counts[counter][i] += sigma is not None and abs(pop[key] - target) <= sigma
        fits.append({key: fit.get(key) for key in ("axes", "populations", "fraction_iso", "reduced_chi2",
                                                   "fraction_iso_err", "delta_bic", "determined")})
    return {"phase": name, "truth": {"axis": axis, "fwhm_deg": widths, "fractions": fractions,
                                     "fraction_iso": phi, "exit_transmission": mixed,
                                     "film_index": {"delta": DELTA, "beta": BETA} if mixed else None}, "trials": trials,
            "successes": counts, "first_seed": SEED, "fits": fits}


def sector_diagnostics(analysis, strongest, maps):
    """Independent signed-sector check of the radial peak and its flanks on the public frame."""
    low, high = strongest["q_window"]
    keep = analysis.valid & maps.above_horizon & (maps.q >= low-.04) & (maps.q <= high+.04)
    q, chi, y = maps.q[keep], maps.chi_deg[keep], analysis.data[keep].astype(float)
    edges = np.arange(low-.045, high+.045, .002)
    centres = (edges[:-1] + edges[1:])/2
    search = (centres >= low-.015) & (centres <= high+.015)
    rows = []
    for start, stop in ((4, 8), (70, 75), (79, 81), (82, 84), (84, 86), (86, 88)):
        for sign in (-1, 1):
            inside = (sign * chi >= start) & (sign * chi < stop)
            count = np.histogram(q[inside], edges)[0]
            total = np.histogram(q[inside], edges, weights=y[inside])[0]
            profile = np.divide(total, count, where=count > 0, out=np.full(len(count), np.nan))
            band = inside & (q >= low) & (q <= high)
            flank = inside & (((q > low-.035) & (q < low-.015)) | ((q > high+.015) & (q < high+.035)))
            if band.any() and flank.any():
                background = float(y[flank].mean())
                area = (float(np.sum((profile[search] - background) * np.diff(edges)[search]))
                        if np.isfinite(profile[search]).all() else None)
                rows.append({"chi_limits_deg": sorted((sign*start, sign*stop)), "ring_mean": float(y[band].mean()),
                             "flank_mean": background, "wider_net_area": area, "pixels": int(band.sum()),
                             "radial_maximum_q": float(centres[search][np.nanargmax(profile[search])])})
    return {"q_window": [low, high], "radial_bin_width": .002, "sectors": rows,
            "source": "measured pixels before orientation polarization and background subtraction"}


def public_diagnostics(report, page, cifs):
    """Repeat the independent-width fit on the original exit cut and inspect signed raw sectors."""
    from src.gimap.features.analyze.application.crystal_orientation import _pixels, _profile
    from src.gimap.features.analyze.domain.crystal.texture import TextureRing, fit_crystal_orientation
    from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps

    model = page.view_model
    analysis = model.state.analysis
    maps = giwaxs_maps(analysis.shape, analysis.geometry)
    pixels = _pixels(analysis, model.frames)
    fits = report["crystal_orientation"]
    if fits[0]["rings"]:
        report["benchmark_signed_sectors"] = sector_diagnostics(analysis, fits[0]["rings"][0], maps)
    old_cut = []
    residuals = []
    polar = np.abs(maps.chi_deg)
    for cif, fit in zip(cifs, fits):
        rings = []
        for row in fit["rings"]:
            profile, _, _ = _profile(maps, pixels, analysis, row["q_window"], row["q"], {"available": False})
            rings.append(TextureRing(row["ring_id"], row["q"], *row["q_window"], profile))
            chi = np.asarray(row["chi_deg"])
            use = np.asarray(row["fit_mask"])
            difference = np.asarray(row["observed"], float)[use] - np.asarray(row["fitted"])[use]
            z = difference / np.asarray(row["sigma"], float)[use]
            worst = row["largest_residual"]
            angle = worst["chi_deg"]
            low, high = row["q_window"]
            mask = analysis.valid & maps.above_horizon & (maps.q >= low) & (maps.q <= high)
            sector = maps.q[mask & (polar >= angle-.5) & (polar < angle+.5)]
            covered = [float(sector.min()), float(sector.max())] if sector.size else None
            chi2 = z @ z
            residuals.append({"phase": fit["phase"], "q": row["q"], "peak_residual_chi_deg": angle,
                              "peak_residual_sigma": worst["observed_minus_fit_sigma"],
                              "sampled_q_span_in_peak_residual_bin": covered,
                              "chi2_fraction_above_80_deg": float(np.sum(z[chi[use] > 80]**2)/chi2) if chi2 else None})
        stage = fit_crystal_orientation(phase_from_cif(cif).phase(), rings)
        old_cut.append({"phase": fit["phase"], "populations": stage["populations"], "points": stage.get("points", 0),
                        "reduced_chi2": stage.get("reduced_chi2"), "fraction_iso": stage.get("fraction_iso"),
                        "rings": [{k: r[k] for k in ("q", "points", "reduced_chi2")} for r in stage["rings"]]})
    report["benchmark_independent_widths_original_cut"] = old_cut
    report["benchmark_ring_residuals"] = residuals


def public_frame(*, with_phases=True, diagnostics=False, incidence_deg=None):
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application.pipeline import PipelineOptions

    info = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))["giwaxs_p08_mapi"]
    cifs = tuple(str(CIF / (name + ".cif")) for name in TEXTURES) if with_phases else ()
    options = PipelineOptions(technique="giwaxs", calibration=str(EXTERNAL / info["poni"]),
                              incidence_deg=info["incidence_deg"] if incidence_deg is None else incidence_deg, phases=cifs)
    with tempfile.TemporaryDirectory() as tmp:
        session = open_headless_session(EXTERNAL / info["frame"], saved_profiles=False, out_dir=tmp)
        try:
            report = session.run_pipeline(options)
            if diagnostics and report.get("crystal_orientation"):
                public_diagnostics(report, session.page, cifs)
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
    parser.add_argument("--public-incidence-deg", type=float, help="test a specified incidence angle on the public frame")
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
            results["public"] = public_frame(diagnostics=True, incidence_deg=arguments.public_incidence_deg)
    if arguments.part in ("synthetic", "all"):
        results["synthetic"] = [coverage(name, arguments.trials) for name in (*TEXTURES, MIXTURE)]
        for item in results["synthetic"]:
            print(item["phase"], item["successes"], "of", item["trials"])
    if "public" in results:
        for fit in results["public"].get("crystal_orientation", []):
            print(fit["phase"], fit.get("populations"), "phi", fit.get("fraction_iso"),
                  "delta BIC", fit.get("delta_bic"), "Hermans", fit.get("herman_crystal"))
            for ring in fit.get("rings", []):
                print(ring["q"], ring["reduced_chi2"], ring["herman_measured"])
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
