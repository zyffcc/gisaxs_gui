"""Part 3 of ``calibration_check.py``: the coverage of GIMaP's calibration 1σ on simulated AgBh frames of known
geometry, and the helpers both parts share (the leave-one-ring-out fits of ``optimizer.estimate_uncertainty``
repeated on the points used by the engine).

Run through ``calibration_check.py --part coverage``; pyFAI is not needed for this part.
"""
from __future__ import annotations

import contextlib
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.calibration.application.headless import describe_calibration, with_line_checks  # noqa: E402
from src.gimap.features.calibration.domain import engine as engine_module  # noqa: E402
from src.gimap.features.calibration.domain.engine import CalibrationEngine  # noqa: E402
from src.gimap.features.calibration.domain.models import THREE_RING_UNCERTAINTY_METHOD, DetectorImage  # noqa: E402
from src.gimap.features.calibration.domain.standards import AGBH_Q1  # noqa: E402
from src.gimap.features.calibration.domain.tilt_fit import tilt_residual  # noqa: E402
from src.gimap.shared.geometry import DetectorGeometry  # noqa: E402
from src.gimap.shared.geometry.detector_tilt import tilted_rays  # noqa: E402
from src.gimap.shared.geometry.poni import tilt_from_geometry  # noqa: E402

FRAME = ROOT / "tests" / "data" / "external" / "saxs_agbh_pyfai" / "Pilatus1M.edf"
HC_KEV_A = 12.398419843320026
PIXEL_M = 172e-6
D001_A = 2 * math.pi / AGBH_Q1  # 58.38 Å in GIMaP and in pyFAI's AgBh.D
GIMAP_KEYS = ("beam_x_px", "beam_y_px", "distance_mm", "rot1_deg", "rot2_deg")
"""GIMaP's convention: direct beam (pixel corners on integers, row 0 at the top), sample to direct beam, rotations."""
CLASSES = (("3", 3, 3), ("4-6", 4, 6), (">=7", 7, 99))
SIMULATION = (
    "Pilatus 1M (981 x 1043 px of 172 um, the gaps of the public frame); direct beam uniform in the central 80 % of "
    "the detector; L uniform in 1.5-3 m; rot1, rot2 uniform in +-3 deg, rot3 = 0; lambda (0.5-2 A) chosen so that "
    "about 3, 4-6 or 7-9 rings lie on the detector (a third of the frames each); every order on the detector drawn "
    "as a Gaussian ring of 1.5-3 px sigma (2x2 sub-pixels; the public frame's rings are 5.5-6 px FWHM), first ring "
    "2000-30000 counts, order^-(1.5-2.5), azimuthal modulation up to 30 %; background falling as 1/(1 + (r/r0)^2), "
    "r0 0.4-1 ring spacing, 1/10-1/60 of the first ring at its radius, plus 1-10 counts; Poisson counts; a beam stop "
    "of 8-20 px radius and its arm (1 % of the counts). Engine: standard agbh, energy and pixel size given, a "
    "distance estimate within +-15 %, tilt auto. 2theta of every sub-pixel from pyFAI's formula "
    "(detector_tilt.tilted_rays), not from lab_rays, which GIMaP fits with.")

def named(keys, values, digits=6) -> dict:
    return {key: (None if value is None or not np.isfinite(value) else round(float(value), digits))
            for key, value in zip(keys, values)}


@contextlib.contextmanager
def record_fit_points():
    """Keep the final fit's points for each candidate, including when its extraction stopped at the pass limit."""
    points_by_candidate = {}
    estimate = engine_module.estimate_uncertainty

    def record(points, candidate, *args):
        points_by_candidate[id(candidate)] = points
        return estimate(points, candidate, *args)

    with patch.object(engine_module, "estimate_uncertainty", record):
        yield points_by_candidate


def gimap_spread(points, candidate, wavelength):
    """The ring points and the leave-one-ring-out fits of ``optimizer.estimate_uncertainty`` (tilted candidate),
    repeated here to have the replicates; and s²(JᵀJ)⁻¹ of the same points. Parameters in the engine's terms."""
    centre = np.array([candidate.center_x_px, candidate.center_y_px, candidate.distance_mm,
                       math.radians(candidate.rot1_deg), math.radians(candidate.rot2_deg)])
    xs, ys, qs = points
    scale = np.array([1.0, 1.0, max(1.0, 1e-3 * centre[2]), 1e-3, 1e-3])
    rings = np.unique(qs)
    # The shipped record distinguishes the conditional fallback from a full joint jackknife.
    # Repeat the indicated fit independently on the recorded points.
    fixed = centre[3:] if rings.size == 3 and candidate.uncertainty_method == THREE_RING_UNCERTAINTY_METHOD else None
    start = centre[:3] if fixed is not None else centre
    replicates = []
    for ring in rings:
        keep = qs != ring
        if np.unique(qs[keep]).size < 2:
            continue
        residual = tilt_residual(xs[keep], ys[keep], qs[keep], wavelength, PIXEL_M, PIXEL_M, fixed_tilt=fixed)
        fit = least_squares(residual, start, loss="soft_l1", f_scale=2.0,
                            max_nfev=40 if fixed is not None else 80, x_scale=scale[:start.size])
        replicates.append(np.r_[fit.x, fixed] if fixed is not None else fit.x)
    residual = tilt_residual(xs, ys, qs, wavelength, PIXEL_M, PIXEL_M)
    steps = np.array([1e-3, 1e-3, max(1e-6, 1e-6 * centre[2]), 1e-7, 1e-7])
    values = residual(centre)
    jacobian = np.column_stack([(residual(centre + steps * axis) - values) / steps[i]
                                for i, axis in enumerate(np.eye(5))])
    covariance = np.linalg.inv(jacobian.T @ jacobian) * float(values @ values) / max(1, values.size - 5)
    return np.asarray(replicates), covariance


def engine_to_gimap(values) -> np.ndarray:
    """Engine parameters (index centre, mm, radians) → GIMaP's convention (canonical centre, mm, degrees)."""
    return np.array([values[0] + 0.5, values[1] + 0.5, values[2], math.degrees(values[3]), math.degrees(values[4])])


def ls_sigma_gimap(covariance) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return np.array([sigma[0], sigma[1], sigma[2], math.degrees(sigma[3]), math.degrees(sigma[4])])


def gap_mask() -> np.ndarray:
    from src.gimap.shared.detector_io import load_detector_image
    return np.asarray(load_detector_image(FRAME).data) < 0


def _reach(cx, cy, shape, share=0.3) -> float:
    """The radius at which less than ``share`` of a circle about (cx, cy) lies on the detector."""
    rows, columns = shape
    angle = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    for radius in np.arange(20.0, 2500.0, 5.0):
        x, y = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
        if np.mean((x >= 0) & (x < columns) & (y >= 0) & (y < rows)) < share:
            return float(radius)
    return 2500.0


def simulate(seed: int, target: int, gaps: np.ndarray, max_order: int | None = None):
    """Counts of an AgBh frame of known geometry with about ``target`` rings on the detector, and that geometry; with
    ``max_order`` the orders above it are not drawn (too weak to be seen)."""
    rng = np.random.default_rng(seed)
    rows, columns = gaps.shape
    while True:
        cx, cy = rng.uniform(0.1, 0.9) * columns, rng.uniform(0.1, 0.9) * rows
        distance = rng.uniform(1.5, 3.0)
        spacing = _reach(cx, cy, gaps.shape) / (target + rng.uniform(0.25, 0.85))
        wavelength = 2 * D001_A * math.sin(math.atan(spacing * PIXEL_M / distance) / 2)
        if 0.5 <= wavelength <= 2.0 and spacing >= 60:
            break
    rot1, rot2 = np.radians(rng.uniform(-3.0, 3.0, 2))
    truth = DetectorGeometry(PIXEL_M, PIXEL_M, distance, cx, cy, wavelength, rot1_rad=rot1, rot2_rad=rot2)
    tilt = tilt_from_geometry(truth)
    width_px, first = rng.uniform(1.5, 3.0), math.exp(rng.uniform(math.log(2000), math.log(30000)))
    falloff, contrast = rng.uniform(1.5, 2.5), rng.uniform(10.0, 60.0)
    knee, floor = rng.uniform(0.4, 1.0) * spacing, rng.uniform(1, 10)
    level = first / contrast * (1 + (spacing / knee) ** 2)  # the background at the first ring is first / contrast
    modulation, phase = rng.uniform(0.0, 0.3), rng.uniform(0.0, math.pi)
    index_rows, index_columns = np.arange(rows, dtype=float)[:, None], np.arange(columns, dtype=float)[None, :]
    expected = np.zeros(gaps.shape)
    for dy in (-0.25, 0.25):
        for dx in (-0.25, 0.25):
            t1, t2, t3 = tilted_rays(tilt, index_rows + dy, index_columns + dx)
            two_theta = np.arctan2(np.hypot(t1, t2), t3)
            chi = np.arctan2(t1, t2)
            radius = distance * np.tan(two_theta) / PIXEL_M
            sub = level / (1 + (radius / knee) ** 2) + floor
            texture = 1 + modulation * np.cos(2 * (chi - phase))
            top = float(two_theta.max())
            for order in range(1, 60):
                ring = 2 * math.asin(order * wavelength / (2 * D001_A))
                if ring > top + 0.01 or (max_order and order > max_order):
                    break
                sigma = width_px * PIXEL_M * math.cos(ring) ** 2 / distance
                near = np.abs(two_theta - ring) < 8 * sigma
                sub[near] += first * order ** -falloff * texture[near] * np.exp(-0.5 * ((two_theta[near] - ring) / sigma) ** 2)
            expected += sub / 4
    yy, xx = np.mgrid[0:rows, 0:columns]
    shadow = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) < rng.uniform(8.0, 20.0)
    arm = rng.uniform(-math.pi, math.pi)
    along = (xx + 0.5 - cx) * math.cos(arm) + (cy - yy - 0.5) * math.sin(arm)
    across = -(xx + 0.5 - cx) * math.sin(arm) + (cy - yy - 0.5) * math.cos(arm)
    shadow |= (along > 0) & (np.abs(across) < rng.uniform(3, 7))
    expected[shadow] *= 0.01
    counts = rng.poisson(expected).astype(np.float32)
    counts[gaps] = -1
    values = np.array([cx, cy, distance * 1e3, math.degrees(rot1), math.degrees(rot2)])
    return counts, values, dict(wavelength_angstrom=wavelength, ring_spacing_px=spacing, ring_width_px=width_px,
                                first_ring_counts=first, background_counts=level)


def calibrated(image, truth, wavelength, estimate) -> dict:
    start = time.perf_counter()
    try:
        with record_fit_points() as fit_points:
            result = CalibrationEngine().calibrate(image, energy_kev=HC_KEV_A / wavelength, standard_key="agbh",
                                                   estimated_distance_mm=estimate, pixel_size_x_m=PIXEL_M,
                                                   pixel_size_y_m=PIXEL_M)
    except Exception as error:  # a failure is a result of the benchmark
        return {"failed": f"{type(error).__name__}: {error}", "identified": False}
    summary = with_line_checks(describe_calibration(result), image)
    candidate = result.selected_candidate
    fit = np.array([candidate.center_x_px + 0.5, candidate.center_y_px + 0.5, candidate.distance_mm,
                    candidate.rot1_deg, candidate.rot2_deg])
    headless_beam = summary["beam_center_px"]
    headless_distance = summary["distance_mm"]
    headless_identified = bool(
        summary["tilt_refined"] and summary["uncertainty"]["distance_mm"] is not None
        and all(value is not None for value in summary["tilt_uncertainty_deg"])
        and abs(headless_distance / truth[2] - 1) < 0.05
        and np.hypot(*(np.asarray(headless_beam) - truth[:2])) < 5
    )
    record = dict(seconds=round(time.perf_counter() - start, 2), matched_rings=candidate.matched_ring_count,
                  orders=[ring.theoretical_index + 1 for ring in candidate.matched_rings],
                  method=candidate.uncertainty_method, tilt_refined=candidate.tilt_refined,
                  chosen_by=summary["chosen_by"], fit=named(GIMAP_KEYS, fit),
                  headless_beam_center_px=headless_beam, headless_distance_mm=headless_distance,
                  headless_identified=headless_identified,
                  identified=bool(abs(fit[2] / truth[2] - 1) < 0.05 and np.hypot(*(fit[:2] - truth[:2])) < 5))
    if not candidate.tilt_refined or candidate.distance_sigma_mm is None:
        return {**record, "failed": "tilt not refined (fewer than 3 matched rings) or no uncertainty",
                "identified": False}
    if candidate.rot1_sigma_deg is None or candidate.rot2_sigma_deg is None:
        return {**record, "failed": "no tilt uncertainty from the full fit", "identified": False}
    sigma = np.array([candidate.center_x_sigma_px, candidate.center_y_sigma_px, candidate.distance_sigma_mm,
                      candidate.rot1_sigma_deg, candidate.rot2_sigma_deg], float)
    replicates, covariance = gimap_spread(fit_points[id(candidate)], candidate, wavelength)
    values = np.array([engine_to_gimap(params) for params in replicates])
    n = len(values)
    sigma_from_points = np.sqrt((n - 1) / n * np.sum((values - values.mean(axis=0)) ** 2, axis=0))
    sigma_ls = ls_sigma_gimap(covariance)
    if candidate.uncertainty_method == THREE_RING_UNCERTAINTY_METHOD:
        sigma_from_points = np.r_[np.maximum(sigma_from_points[:3], sigma_ls[:3]), sigma_ls[3:]]
    record.update(sigma=named(GIMAP_KEYS, sigma), sigma_ls=named(GIMAP_KEYS, sigma_ls),
                  jackknife_fits=n, sigma_check_max_relative_difference=float(np.max(
                      np.abs(sigma_from_points - sigma) / np.maximum(sigma, 1e-12))))
    if candidate.matched_ring_count == 3:
        record["jackknife_two_ring_fits"] = [named(GIMAP_KEYS, engine_to_gimap(item), 4) for item in replicates]
    return record


def run_case(job) -> dict:
    index, seed, target, max_order = job
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    counts, truth, setup = simulate(seed, target, gap_mask(), max_order)
    estimate = truth[2] * np.random.default_rng(seed + 7).uniform(0.85, 1.15)
    tilt = math.degrees(math.acos(math.cos(math.radians(truth[3])) * math.cos(math.radians(truth[4]))))
    record = dict(case=index, seed=seed, target_rings=target, orders_drawn=max_order or "all",
                  truth=named(GIMAP_KEYS, truth), estimate_mm=estimate,
                  tilt_deg=round(tilt, 3), **{key: round(value, 4) for key, value in setup.items()})
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "simulated_agbh.npy"
        path.write_bytes(b"simulated AgBh frame")
        image = DetectorImage(counts, None, path, pixel_size_x_m=PIXEL_M, pixel_size_y_m=PIXEL_M)
        record["as_shipped"] = calibrated(image, truth, setup["wavelength_angstrom"], estimate)
    return record


def coverage_table(records, name: str, sigma_key: str) -> dict:
    table = {}
    for label, low, high in CLASSES:
        chosen = [r for r in records if low <= r[name]["matched_rings"] <= high]
        if not chosen:
            continue
        row = {"cases": len(chosen)}
        for key in GIMAP_KEYS:
            ratio = np.array([abs(r[name]["fit"][key] - r["truth"][key]) / r[name][sigma_key][key] for r in chosen])
            row[key] = {"within_1sigma": round(float(np.mean(ratio <= 1)), 3),
                        "within_2sigma": round(float(np.mean(ratio <= 2)), 3),
                        "median_error_over_sigma": round(float(np.median(ratio)), 3)}
        if sigma_key == "sigma" and all(r[name]["method"] == "ring jackknife" for r in chosen):
            dof = np.array([r[name]["jackknife_fits"] - 1 for r in chosen])
            row["student_t_expected"] = {"within_1sigma": round(float(np.mean(2 * student_t.cdf(1, dof) - 1)), 3),
                                         "within_2sigma": round(float(np.mean(2 * student_t.cdf(2, dof) - 1)), 3),
                                         "median_error_over_sigma": round(float(np.mean(student_t.ppf(0.75, dof))), 3)}
        table[label] = row
    return table


def identified_share(records, name: str) -> str:
    return f"{sum(r[name]['identified'] for r in records)}/{len(records)}"


def headless_recovery(records, name: str) -> dict:
    if not any("headless_identified" in r[name] for r in records):
        return {}
    return {"headless_identified": f"{sum(r[name].get('headless_identified', False) for r in records)}/{len(records)}"}


def not_identified(records, name: str) -> dict:
    """How many frames were not identified, by reason."""
    reasons: dict[str, int] = {}
    for r in records:
        if not r[name]["identified"]:
            reason = r[name].get("failed", "wrong geometry (L 5 % or more off, or the beam 5 px or more)")
            reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


def calibration_times(records, name: str) -> dict:
    """Elapsed time per frame inside a worker, including the headless line check, excluding simulation."""
    seconds = [r[name]["seconds"] for r in records if "seconds" in r[name]]
    if not seconds:
        return {}
    return {"timed": len(seconds), "mean": round(float(np.mean(seconds)), 3),
            "median": round(float(np.median(seconds)), 3)}


def engine_summary(records, name: str) -> dict:
    good = [r for r in records if r[name]["identified"]]
    three = [r for r in good if r[name]["matched_rings"] == 3]
    many = [r for r in good if r[name]["matched_rings"] >= 7]
    targets = (("3", 3, 3), ("4-6", 4, 6), ("7-9", 7, 9))
    tilts = (("<1", 0, 1), ("1-2", 1, 2), ("2-3", 2, 3), (">=3", 3, 99))
    return {
        "identified": identified_share(records, name),
        **headless_recovery(records, name),
        "calibration_and_line_check_seconds_per_frame": calibration_times(records, name),
        "identified_by_target_rings": {label: identified_share([r for r in records if low <= r["target_rings"] <= high],
                                                               name) for label, low, high in targets},
        "identified_by_tilt_deg": {label: identified_share([r for r in records if low <= r["tilt_deg"] < high], name)
                                   for label, low, high in tilts},
        "not_identified": not_identified(records, name),
        "line_check_switched_candidate": sum(str(r[name].get("chosen_by")).startswith("line positions")
                                             for r in records),
        "coverage_reported_sigma": coverage_table(good, name, "sigma"),
        "coverage_least_squares_sigma": coverage_table(good, name, "sigma_ls"),
        "within_1sigma_with_covariance_floor_7plus": {
            key: round(sum(abs(r[name]["fit"][key] - r["truth"][key]) <=
                           max(r[name]["sigma"][key], r[name]["sigma_ls"][key]) for r in many) / len(many), 3)
            for key in GIMAP_KEYS
        } if many else {},
        "three_rings_cases": len(three),
    }


def run_cases(jobs, workers: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = []
        for record in pool.map(run_case, jobs):
            records.append(record)
            if len(records) % 25 == 0 or len(records) == len(jobs):
                print(f"{len(records)}/{len(jobs)} frames, {identified_share(records, 'as_shipped')} identified",
                      flush=True)
        return records


def three_ring_set(cases: int, workers: int, seed0: int) -> tuple[dict, list[dict]]:
    """Frames on which only the first three orders are drawn: the jackknife with exactly three rings."""
    start = time.perf_counter()
    records = run_cases([(i, seed0 + i, 3, 3) for i in range(cases)], workers)
    return {"cases": cases, "seed0": seed0, "workers": workers, "seconds": round(time.perf_counter() - start, 1)}, records


def three_ring_summary(meta: dict, records: list[dict]) -> dict:
    summary = {**meta, "set_up": "as the main set with about 3 rings on the detector, but orders above 3 not drawn"}
    name = "as_shipped"
    good = [r for r in records if r[name]["identified"] and r[name]["matched_rings"] == 3]
    example = good[0] if good else None
    summary[name] = {
        "identified": identified_share(records, name), "identified_with_3_rings": len(good),
        **headless_recovery(records, name),
        "calibration_and_line_check_seconds_per_frame": calibration_times(records, name),
        "not_identified": not_identified(records, name),
        "coverage_reported_sigma": coverage_table(good, name, "sigma").get("3"),
        "coverage_least_squares_sigma": coverage_table(good, name, "sigma_ls").get("3"),
        "example": None if example is None else {
            "truth": example["truth"], "fit": example[name]["fit"], "sigma": example[name]["sigma"],
            "jackknife_two_ring_fits": example[name].get("jackknife_two_ring_fits")},
    }
    return summary


def coverage(cases: int, workers: int, seed0: int) -> tuple[dict, list[dict]]:
    targets = [(3,), (4, 5, 6), (7, 8, 9)]
    jobs = [(i, seed0 + i, targets[i % 3][(i // 3) % len(targets[i % 3])], None) for i in range(cases)]
    start = time.perf_counter()
    records = run_cases(jobs, workers)
    return {"cases": cases, "seed0": seed0, "workers": workers, "seconds": round(time.perf_counter() - start, 1)}, records


def coverage_summary(meta: dict, records: list[dict]) -> dict:
    return {
        **meta, "set_up": SIMULATION,
        "identified_means": "tilt refined with a 1 sigma, |L - L_true| < 5 % and the beam within 5 px of the truth",
        "engines": {"as_shipped": "CalibrationEngine as in this checkout, with guided ring extraction and fitted "
                                  "candidate ranking by the supported and missing rings and the point residual"},
        "three_rings_jackknife": "Leave one of three rings out and refit all five parameters. If any replicate is "
                                 "unconverged, physically inadmissible, or has a rank-deficient normalized Jacobian, "
                                 "hold tilt fixed in all three replicates. In that fallback, use full-fit covariance "
                                 "for the tilts and its standard errors as floors on x, y and L.",
        "as_shipped": engine_summary(records, "as_shipped"),
    }
