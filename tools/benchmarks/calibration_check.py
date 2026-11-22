"""Geometry Calibration checked: the public silver behenate frame against pyFAI, and the coverage of GIMaP's 1σ on
simulated frames of known geometry.

Part 1 (``--part real``, needs pyFAI): GIMaP's calibration of the public pyFAI frame
``tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf`` (AgBh, λ = 1.0 Å, Pilatus 1M, 172 µm pixels; the route of the
Geometry Calibration window: ``RunCalibration``, standard ``agbh``, tilt ``auto``) and pyFAI's own calibration of the
same frame with the same calibrant, wavelength and pixel size, made without a person: pyFAI's ``Recalibration``
(``pyFAI.gui.cli_calibration``, the class behind ``pyFAI-recalib --no-gui --no-interactive``) picks control points
along every ring (``extract_cpt``, massif, 1 point per degree of azimuth, numpy seed fixed) from the shipped
``Pilatus1M.poni`` (rot3 set to 0), and ``GeometryRefinement.curve_fit`` refines dist, poni1, poni2, rot1, rot2 with
rot3 = 0 and λ fixed; points and refinement are repeated from the newest geometry until it stops moving. pyFAI's own
uncertainty is that of ``curve_fit`` (s²(JᵀJ)⁻¹ of the 2θ residuals); a jackknife over its six rings is
reported as well. Both geometries are given in both conventions (GIMaP: direct beam in pixels with pixel corners on
integers, distance from the sample to the direct beam; pyFAI: PONI), with their differences in units of the combined σ
and the RMS ring-point residuals of each geometry on both point sets. The shipped ``.poni`` is reported for reference.

Part 2 is the definition of what GIMaP minimises and which σ it reports (``RESIDUAL_DEFINITION``, in the JSON).

Part 3 (``--part coverage``, code in ``calibration_coverage.py``, no pyFAI needed): AgBh frames simulated on the
Pilatus 1M (its module gaps from the public frame) with a known direct beam, distance 1.5-3 m, rot1 and rot2 drawn
in ±3°, λ chosen so that 3 to 9 rings lie on the detector, Gaussian rings (2×2 sub-pixels, intensity falling with
the order and varying with azimuth) on a falling background with Poisson noise, a beam stop and its arm. The pixels'
2θ come from pyFAI's formula (``shared.geometry.detector_tilt.tilted_rays``), not from the ``lab_rays`` path GIMaP
fits with. GIMaP's engine calibrates each frame (standard and energy given, a distance estimate within ±15 %); the
coverage of its reported 1σ (and of s²(JᵀJ)⁻¹ on the same ring points, for comparison) is counted by the number of
matched rings. Each frame is calibrated by the shipped engine, which polishes beam-centre proposals for both
transmission and grazing-incidence images.

usage: python tools/benchmarks/calibration_check.py [--part real|coverage|all] [--cases 300] [--workers 4]
       [--seed 20261122] [--json tools/benchmarks/calibration_check.json]
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse  # noqa: E402
import gzip  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_coverage import (  # noqa: E402
    AGBH_Q1, FRAME, GIMAP_KEYS, HC_KEV_A, PIXEL_M, coverage, engine_to_gimap, gimap_spread,
    coverage_summary, ls_sigma_gimap, named, record_fit_points, three_ring_set, three_ring_summary)
from src.gimap.features.calibration.application import LoadCalibrationImage, RunCalibration  # noqa: E402
from src.gimap.features.calibration.application.headless import describe_calibration, with_line_checks  # noqa: E402
from src.gimap.features.calibration.domain.requests import CalibrationRequest  # noqa: E402
from src.gimap.features.calibration.domain.tilt_fit import tilt_residual  # noqa: E402
from src.gimap.features.calibration.infrastructure.adapters import (  # noqa: E402
    LegacyCalibrationRunnerAdapter, LocalCalibrationImageAdapter)
from src.gimap.shared.geometry import DetectorGeometry  # noqa: E402
from src.gimap.shared.geometry.detector_tilt import DetectorTilt  # noqa: E402
from src.gimap.shared.geometry.poni import tilt_from_geometry  # noqa: E402

PONI = FRAME.with_suffix(".poni")
WAVELENGTH_A = 1.0  # stated in Pilatus1M.poni
PONI_KEYS = ("dist_mm", "poni1_mm", "poni2_mm", "rot1_deg", "rot2_deg")
RESIDUAL_DEFINITION = {
    "gimap_minimises": "scipy.optimize.least_squares(loss='soft_l1', f_scale=2.0) over the ring points i: cost = "
                       "sum_i f^2 (sqrt(1 + r_i^2 / f^2) - 1), f = 2 px (r_i^2/2 for |r_i| << 2 px, f |r_i| for "
                       "|r_i| >> 2 px); free: beam x, y, L, rot1, rot2 (tilt_fit.fit_tilted_geometry). Before it, the "
                       "centre and L alone are fitted to ring pixels with the untilted residual (Physics and Methods "
                       "3.3, eq. 3.8) and the same loss.",
    "gimap_residual": "r_i = L (tan 2theta_i - tan 2theta_ring(i)) / pbar, in pixels: L the distance from the sample to "
                      "the direct beam (m), 2theta_i the scattering angle of point i for the trial geometry (beam centre, "
                      "L, rot1, rot2; GIMaP's lab_rays, pyFAI's rotation convention, rot3 fixed), 2theta_ring from "
                      "q = 2 pi n / 58.38 A and lambda, pbar = (pixel_x + pixel_y)/2. Untilted, r_i is the radial pixel "
                      "distance of the point from its ring. (tilt_fit.tilt_residual)",
    "gimap_points": "one point per azimuthal sector (about 2 px of arc, 36-720 a ring) of each matched ring: the "
                    "intensity-weighted mean position of the valid pixels within +-max(3, min(10, 1.2 % of the radius)) "
                    "px of the ring (weights: signal above the sector's lower quartile, signal = log(1 + counts) minus a "
                    "smooth background); a sector is kept when its brightest pixel stands above that quartile by at "
                    "least 0.3 of the ring's median such height; points taken again about the tilted rings, 4 passes "
                    "at most (tilt_fit.tilted_ring_points, refine_tilt)",
    "gimap_bounds": "beam centre within 4 % of the larger image side (12-60 px) of the start, L within +-25 % (and "
                    "the user's range), |rot1|, |rot2| <= 30 deg",
    "gimap_sigma_reported": "4 or more contributing rings: jackknife over rings, sigma^2 = (n-1)/n sum_k (theta_(k) - "
                            "mean)^2, theta_(k) the same robust fit with ring k left out (unbounded, from the full fit, "
                            "same points); 2 rings: s^2 (J^T J)^-1, s^2 = sum r_i^2 / (N - p), forward-difference J "
                            "(optimizer.estimate_uncertainty). With 3 rings the two-ring refits hold both tilts at "
                            "the full fit. When the full fit refined the tilts, their variance comes from its "
                            "five-parameter covariance, and x, y and L take the larger of the jackknife and covariance "
                            "standard errors. With fixed tilt use the jackknife for x, y and L.",
    "pyfai_minimises": "sum_i (2theta_i - 2theta_ring(i))^2 in radians, plain least squares "
                       "(GeometryRefinement.residu2; curve_fit = Levenberg-Marquardt); its sigma is curve_fit's "
                       "s^2 (J^T J)^-1",
}


def to_poni(values) -> np.ndarray:
    """GIMaP's (x, y, L mm, rot1°, rot2°) as pyFAI's PONI (dist mm, poni1 mm, poni2 mm, rot1°, rot2°)."""
    x, y, distance, rot1, rot2 = map(float, values)
    tilt = tilt_from_geometry(DetectorGeometry(PIXEL_M, PIXEL_M, distance * 1e-3, x, y, 1.0,
                                               rot1_rad=math.radians(rot1), rot2_rad=math.radians(rot2)))
    return np.array([tilt.distance_m * 1e3, tilt.poni1_m * 1e3, tilt.poni2_m * 1e3, rot1, rot2])


def from_poni(values) -> np.ndarray:
    """pyFAI's PONI (dist mm, poni1 mm, poni2 mm, rot1°, rot2°; rot3 = 0) as GIMaP's direct beam, distance to it."""
    dist, poni1, poni2, rot1, rot2 = map(float, values)
    tilt = DetectorTilt(dist * 1e-3, poni1 * 1e-3, poni2 * 1e-3, math.radians(rot1), math.radians(rot2), 0.0,
                        PIXEL_M, PIXEL_M)
    x, y = tilt.direct_beam_px()
    return np.array([x, y, tilt.beam_distance_m * 1e3, rot1, rot2])


def index_params(values) -> np.ndarray:
    """GIMaP's convention → the parameters of ``tilt_residual`` (centre counted from pixel centres, radians)."""
    x, y, distance, rot1, rot2 = map(float, values)
    return np.array([x - 0.5, y - 0.5, distance, math.radians(rot1), math.radians(rot2)])


def jackknife_sigma(replicates) -> np.ndarray:
    values = np.asarray(replicates, float)
    n = values.shape[0]
    return np.sqrt((n - 1) / n * np.sum((values - values.mean(axis=0)) ** 2, axis=0))


def point_rms(points, values, wavelength=WAVELENGTH_A) -> float:
    """RMS (px) of GIMaP's residual r_i of the geometry ``values`` (GIMaP's convention) on ``points`` = (xs, ys, qs)."""
    xs, ys, qs = points
    residual = tilt_residual(xs, ys, qs, wavelength, PIXEL_M, PIXEL_M)(index_params(values))
    return float(np.sqrt(np.mean(residual ** 2)))


# ---------------------------------------------------------------- part 1: the public frame, GIMaP and pyFAI

def gimap_real() -> dict:
    image = LoadCalibrationImage(LocalCalibrationImageAdapter(ask_dataset=False))(FRAME)
    request = CalibrationRequest(image=image, energy_kev=HC_KEV_A / WAVELENGTH_A, standard_key="agbh",
                                 pixel_size_x_m=PIXEL_M, pixel_size_y_m=PIXEL_M)
    start = time.perf_counter()
    with record_fit_points() as fit_points:
        result = RunCalibration(LegacyCalibrationRunnerAdapter())(request, None, None)
    seconds = time.perf_counter() - start
    summary = with_line_checks(describe_calibration(result), image)
    candidate = result.selected_candidate
    points = fit_points[id(candidate)]
    replicates, covariance = gimap_spread(points, candidate, WAVELENGTH_A)
    values = np.array([candidate.center_x_px + 0.5, candidate.center_y_px + 0.5, candidate.distance_mm,
                       candidate.rot1_deg, candidate.rot2_deg])
    reported = np.array([candidate.center_x_sigma_px, candidate.center_y_sigma_px, candidate.distance_sigma_mm,
                         candidate.rot1_sigma_deg, candidate.rot2_sigma_deg], float)
    gimap_replicates = np.array([engine_to_gimap(item) for item in replicates])
    repeated = jackknife_sigma(gimap_replicates)
    return dict(
        values=values, sigma=reported, sigma_poni=jackknife_sigma([to_poni(item) for item in gimap_replicates]),
        sigma_ls=ls_sigma_gimap(covariance), points=points, seconds=seconds,
        record=dict(matched_rings=candidate.matched_ring_count, orders=[r.theoretical_index + 1 for r in candidate.matched_rings],
                    uncertainty_method=candidate.uncertainty_method, tilt_refined=candidate.tilt_refined,
                    point_rms_px=candidate.point_rms_px, untilted_point_rms_px=candidate.untilted_point_rms_px,
                    ring_points=int(points[0].size), chosen_by=summary.get("chosen_by"), seconds=seconds,
                    jackknife_repeated_here_max_relative_difference=float(np.max(np.abs(repeated / reported - 1.0)))),
    )


class PyFAIRun:
    """pyFAI's calibration of the frame: Recalibration's control points, GeometryRefinement.curve_fit."""

    def __init__(self):
        logging.disable(logging.WARNING)
        import importlib  # pyFAI is not a dependency of GIMaP: imported by name (as pyfai_comparison.py)
        get_calibrant = importlib.import_module("pyFAI.calibrant").get_calibrant
        GeometryRefinement = importlib.import_module("pyFAI.geometryRefinement").GeometryRefinement
        Recalibration = importlib.import_module("pyFAI.gui.cli_calibration").Recalibration
        AzimuthalIntegrator = importlib.import_module("pyFAI.integrator.azimuthal").AzimuthalIntegrator
        self.GeometryRefinement, self.AzimuthalIntegrator = GeometryRefinement, AzimuthalIntegrator
        self.calibrant = get_calibrant("AgBh")
        self.calibrant.wavelength = WAVELENGTH_A * 1e-10
        self.recal = Recalibration(dataFiles=[str(FRAME)], detector="Pilatus1M", wavelength=WAVELENGTH_A * 1e-10,
                                   calibrant="AgBh")
        self.recal.gui = self.recal.interactive = self.recal.keep = False
        self.recal.ai = self.shipped()
        self.recal.detector = self.recal.ai.detector
        with contextlib.redirect_stdout(io.StringIO()):
            self.recal.preprocess()
        self.recal.basename = None  # write no .npt / .poni beside the frame

    def shipped(self):
        start = self.AzimuthalIntegrator.sload(str(PONI))
        start.wavelength = WAVELENGTH_A * 1e-10
        start.rot3 = 0.0
        return start

    def refinement(self, data, values):
        dist, poni1, poni2, rot1, rot2 = values
        return self.GeometryRefinement(np.asarray(data, np.float64), dist=dist, poni1=poni1, poni2=poni2, rot1=rot1,
                                       rot2=rot2, rot3=0.0, detector=self.recal.detector,
                                       calibrant=self.calibrant, wavelength=WAVELENGTH_A * 1e-10)

    def curve_fit(self, data, values):
        geometry = self.refinement(data, values)
        with contextlib.redirect_stdout(io.StringIO()):
            error, _ = geometry.curve_fit(with_rot=True)
            error, _ = geometry.curve_fit(with_rot=True)
        return np.array([geometry.dist, geometry.poni1, geometry.poni2, geometry.rot1, geometry.rot2]), error

    def calibrate(self, start, cycles=16, seed=20261122):
        self.recal.ai = start
        history = []
        for _ in range(cycles):
            np.random.seed(seed)  # massif shuffles its candidate pixels
            with contextlib.redirect_stdout(io.StringIO()):
                self.recal.extract_cpt("massif", 1.0)
            data = np.array(self.recal.data, dtype=np.float64)
            source = self.recal.ai
            values, error = self.curve_fit(data, (source.dist, source.poni1, source.poni2, source.rot1, source.rot2))
            self.recal.ai = self.refinement(data, values)
            history.append(values)
            if len(history) > 1 and np.all(np.abs(history[-1] - history[-2]) < [1e-7, 1e-6, 1e-6, 1e-5, 1e-5]):
                break
        return data, values, error, np.asarray(history)

    def delta_two_theta(self, values, rows, columns, orders) -> np.ndarray:
        geometry = self.refinement(np.c_[rows, columns, np.asarray(orders) - 1], values)
        return geometry.tth(rows, columns, [*values, 0.0]) - geometry.calc_2th(np.asarray(orders) - 1)

    def covariance(self, data, values) -> np.ndarray:
        """s²(JᵀJ)⁻¹ of the 2θ residuals at the solution (what curve_fit's pcov is), central differences."""
        rows, columns, orders = data[:, 0], data[:, 1], data[:, 2].astype(int) + 1
        steps = np.array([1e-7, 1e-7, 1e-7, 1e-8, 1e-8])
        base = self.delta_two_theta(values, rows, columns, orders)
        jacobian = np.column_stack([(self.delta_two_theta(values + steps * axis, rows, columns, orders)
                                     - self.delta_two_theta(values - steps * axis, rows, columns, orders)) / (2 * steps[i])
                                    for i, axis in enumerate(np.eye(5))])
        return np.linalg.inv(jacobian.T @ jacobian) * float(base @ base) / (base.size - 5)


def poni_units(values) -> np.ndarray:
    """pyFAI's (m, m, m, rad, rad) → (mm, mm, mm, °, °)."""
    return np.asarray(values, float) * np.array([1e3, 1e3, 1e3, 180 / math.pi, 180 / math.pi])


def propagate(function, values, covariance) -> np.ndarray:
    """1σ of ``function(values)`` from the covariance of ``values`` (numerical Jacobian)."""
    values = np.asarray(values, float)
    steps = np.maximum(np.abs(values) * 1e-7, 1e-9)
    columns = [(function(values + steps * axis) - function(values - steps * axis)) / (2 * steps[i])
               for i, axis in enumerate(np.eye(values.size))]
    jacobian = np.column_stack(columns)
    return np.sqrt(np.diag(jacobian @ covariance @ jacobian.T))


def real_frame() -> dict:
    import importlib
    pyFAI = importlib.import_module("pyFAI")
    gimap = gimap_real()
    run = PyFAIRun()
    data, values, error, history = run.calibrate(run.shipped())
    shipped = run.shipped()
    start_b = run.shipped()
    start_b.dist, start_b.rot1, start_b.rot2 = start_b.dist * 1.03, 0.0, 0.0
    _, values_b, _, history_b = run.calibrate(start_b)
    covariance = run.covariance(data, values)
    poni_mm = poni_units(values)
    cov_mm = covariance * np.outer(poni_units(np.ones(5)), poni_units(np.ones(5)))
    sigma_cf = poni_units([error[key] for key in ("dist", "poni1", "poni2", "rot1", "rot2")])
    rows, columns, orders = data[:, 0], data[:, 1], data[:, 2].astype(int) + 1
    replicates = [poni_units(run.curve_fit(data[orders != order], values)[0]) for order in np.unique(orders)]
    pyfai_points = (columns, rows, AGBH_Q1 * orders)
    pyfai = dict(poni=poni_mm, gimap=from_poni(poni_mm), sigma_poni=np.sqrt(np.diag(cov_mm)),
                 sigma_gimap=propagate(from_poni, poni_mm, cov_mm), sigma_poni_jk=jackknife_sigma(replicates),
                 sigma_gimap_jk=jackknife_sigma([from_poni(item) for item in replicates]))
    shipped_mm = poni_units([shipped.dist, shipped.poni1, shipped.poni2, shipped.rot1, shipped.rot2])
    geometries = {"gimap": gimap["values"], "pyfai": pyfai["gimap"], "shipped_poni": from_poni(shipped_mm)}
    point_sets = {"gimap_points": gimap["points"], "pyfai_points": pyfai_points}
    rms_px = {f"{name}_on_{points}": point_rms(point_sets[points], values_)
              for name, values_ in geometries.items() for points in point_sets}
    gx, gy, gq = gimap["points"]
    gimap_orders = np.rint(gq / AGBH_Q1).astype(int)
    rms_2theta = {f"{name}_on_{points}_urad": 1e6 * float(np.sqrt(np.mean(run.delta_two_theta(
        to_poni(values_) * np.array([1e-3, 1e-3, 1e-3, math.pi / 180, math.pi / 180]), *args) ** 2)))
        for name, values_ in geometries.items()
        for points, args in (("gimap_points", (gy, gx, gimap_orders)), ("pyfai_points", (rows, columns, orders)))}
    fit2d = run.refinement(data, values).getFit2D()

    def compare(keys, g_values, g_sigma, p_values, p_sigma, p_sigma_jk):
        rows_ = {}
        for i, key in enumerate(keys):
            delta = g_values[i] - p_values[i]
            rows_[key] = named(("gimap", "gimap_sigma", "pyfai", "pyfai_sigma_curve_fit", "pyfai_sigma_jackknife",
                                "difference", "difference_over_combined_sigma_curve_fit",
                                "difference_over_combined_sigma_jackknife"),
                               (g_values[i], g_sigma[i], p_values[i], p_sigma[i], p_sigma_jk[i], delta,
                                delta / math.hypot(g_sigma[i], p_sigma[i]), delta / math.hypot(g_sigma[i], p_sigma_jk[i])))
        return rows_

    radii = sorted(gimap["record"]["orders"])
    return {
        "pyfai_version": pyFAI.version,
        "frame": "tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf", "wavelength_angstrom": WAVELENGTH_A,
        "pixel_um": PIXEL_M * 1e6, "calibrant": "AgBh, d001 = 58.38 A (GIMaP agbh and pyFAI AgBh.D)",
        "gimap": {**gimap["record"], "sigma_method_reported": gimap["record"]["uncertainty_method"],
                  "values": named(GIMAP_KEYS, gimap["values"]), "sigma": named(GIMAP_KEYS, gimap["sigma"]),
                  "values_poni": named(PONI_KEYS, to_poni(gimap["values"])),
                  "sigma_poni_jackknife": named(PONI_KEYS, gimap["sigma_poni"]),
                  "sigma_least_squares_same_points": named(GIMAP_KEYS, gimap["sigma_ls"])},
        "pyfai": {"route": "Recalibration.extract_cpt('massif', 1.0) from the shipped .poni (rot3 = 0), then "
                           "GeometryRefinement.curve_fit(with_rot=True), repeated until the geometry stops moving",
                  "cycles": int(len(history)), "control_points": int(len(data)),
                  "points_per_order": {int(o): int((orders == o).sum()) for o in np.unique(orders)},
                  "values_poni": named(PONI_KEYS, pyfai["poni"]), "sigma_poni_curve_fit": named(PONI_KEYS, sigma_cf),
                  "sigma_poni_covariance_here": named(PONI_KEYS, pyfai["sigma_poni"]),
                  "sigma_poni_jackknife": named(PONI_KEYS, pyfai["sigma_poni_jk"]),
                  "values_gimap_convention": named(GIMAP_KEYS, pyfai["gimap"]),
                  "sigma_gimap_convention_curve_fit": named(GIMAP_KEYS, pyfai["sigma_gimap"]),
                  "sigma_gimap_convention_jackknife": named(GIMAP_KEYS, pyfai["sigma_gimap_jk"]),
                  "fit2d_direct_beam_px": [fit2d["centerX"], fit2d["centerY"]], "fit2d_direct_distance_mm": fit2d["directDist"],
                  "last_8_cycles_spread_gimap_convention": named(GIMAP_KEYS, np.std([from_poni(poni_units(item)) for item in history[-8:]], axis=0)),
                  "second_start": {"start": "shipped .poni, distance x 1.03, rot1 = rot2 = 0", "cycles": int(len(history_b)),
                                   "values_gimap_convention": named(GIMAP_KEYS, from_poni(poni_units(values_b))),
                                   "difference_from_first_over_sigma_curve_fit": named(
                                       PONI_KEYS, (poni_units(values_b) - poni_mm) / sigma_cf, 3)}},
        "shipped_poni": {"values_poni": named(PONI_KEYS, shipped_mm), "values_gimap_convention": named(GIMAP_KEYS, from_poni(shipped_mm)),
                         "note": "a pixel-to-q test geometry, not a calibration of this image: its ring radii are "
                                 "about 3 % smaller than measured (see the RMS residuals)",
                         "difference_from_pyfai_over_sigma_curve_fit": named(PONI_KEYS, (shipped_mm - poni_mm) / sigma_cf, 2)},
        "compare_gimap_convention": compare(GIMAP_KEYS, gimap["values"], gimap["sigma"], pyfai["gimap"],
                                            pyfai["sigma_gimap"], pyfai["sigma_gimap_jk"]),
        "compare_poni": compare(PONI_KEYS, to_poni(gimap["values"]), gimap["sigma_poni"], pyfai["poni"],
                                pyfai["sigma_poni"], pyfai["sigma_poni_jk"]),
        "rms_residual_px": {key: round(value, 4) for key, value in rms_px.items()},
        "rms_residual_two_theta": {key: round(value, 3) for key, value in rms_2theta.items()},
        "gimap_orders": radii,
    }


def frames_file(summary: Path) -> Path:
    """Every simulated frame's record, beside the summary in ``per_cut/`` (gzip), as the Predict runs keep theirs."""
    return summary.parent / "per_cut" / f"{summary.stem}_frames.json.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("real", "coverage", "three", "summarize", "all"), default="all",
                    help="coverage: the main simulated set and the three-ring set; three: the three-ring set alone; "
                         "summarize: the coverage summaries again from the stored frames")
    ap.add_argument("--cases", type=int, default=300, help="frames of the main simulated set")
    ap.add_argument("--three-ring-cases", type=int, default=150, help="frames with only orders 1-3 drawn")
    ap.add_argument("--workers", type=int, default=4, help="processes (at most 4)")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--json", type=Path, default=Path(__file__).with_suffix(".json"))
    a = ap.parse_args()
    workers = max(1, min(4, a.workers))
    out = json.loads(a.json.read_text(encoding="utf-8")) if a.json.exists() else {}
    out["residual_definition"] = RESIDUAL_DEFINITION
    command = "python tools/benchmarks/calibration_check.py " + " ".join(sys.argv[1:])
    if a.part in ("real", "all"):
        out["real_frame"] = {"command": command, **real_frame()}
        print(json.dumps({k: out["real_frame"][k] for k in ("compare_gimap_convention", "rms_residual_px")}, indent=1))
    path = frames_file(a.json)
    frames = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
    if a.part in ("coverage", "all"):
        meta, records = coverage(a.cases, workers, a.seed)
        frames["main"] = {"meta": {"command": command, **meta}, "frames": records}
    if a.part in ("coverage", "three", "all"):
        meta, records = three_ring_set(a.three_ring_cases, workers, a.seed + 1_000_000)
        frames["three_ring_set"] = {"meta": {"command": command, **meta}, "frames": records}
    if a.part != "real" and frames:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(frames, default=float).encode("utf-8"), mtime=0))
        out["coverage"] = {"frames_file": path.relative_to(a.json.parent).as_posix()}
        if "main" in frames:
            out["coverage"].update(coverage_summary(frames["main"]["meta"], frames["main"]["frames"]))
        if "three_ring_set" in frames:
            out["coverage"]["three_ring_set"] = three_ring_summary(frames["three_ring_set"]["meta"],
                                                                   frames["three_ring_set"]["frames"])
        print(json.dumps(out["coverage"], indent=1))
    a.json.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
