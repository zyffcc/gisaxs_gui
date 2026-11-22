"""Calibration candidate numerical refinement."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .geometry_model import q_to_ring_radius_m
from .models import THREE_RING_UNCERTAINTY_METHOD, CalibrationCandidate
from .preprocessing import AnalysisImage
from .tilt_fit import TILT_LIMIT_DEG, tilt_residual


def _ring_points(
    analysis: AnalysisImage,
    candidate: CalibrationCandidate,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_x: list[np.ndarray] = []
    points_y: list[np.ndarray] = []
    point_q: list[np.ndarray] = []
    height, width = analysis.signal.shape
    mean_pixel = 0.5 * (pixel_size_x_m + pixel_size_y_m)
    for match in candidate.matched_rings:
        radius = match.predicted_radius_px
        half_width = max(3.0, min(10.0, 0.012 * radius))
        x0, x1 = max(0, int(candidate.center_x_px - radius - half_width)), min(width, int(candidate.center_x_px + radius + half_width) + 1)
        y0, y1 = max(0, int(candidate.center_y_px - radius - half_width)), min(height, int(candidate.center_y_px + radius + half_width) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        physical = np.sqrt(((xx - candidate.center_x_px) * pixel_size_x_m) ** 2 + ((yy - candidate.center_y_px) * pixel_size_y_m) ** 2)
        annulus = (np.abs(physical / mean_pixel - radius) <= half_width) & analysis.valid[y0:y1, x0:x1]
        values = analysis.signal[y0:y1, x0:x1][annulus]
        if values.size < 12:
            continue
        cutoff = np.percentile(values, 70)
        selected = annulus & (analysis.signal[y0:y1, x0:x1] >= cutoff)
        ys, xs = np.nonzero(selected)
        if xs.size > 800:
            take = np.linspace(0, xs.size - 1, 800, dtype=int)
            xs, ys = xs[take], ys[take]
        points_x.append(xs.astype(float) + x0)
        points_y.append(ys.astype(float) + y0)
        point_q.append(np.full(xs.size, match.q_inv_angstrom, dtype=float))
    if not points_x:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(points_x), np.concatenate(points_y), np.concatenate(point_q)


def _ring_residual(xs, ys, qs, wavelength_angstrom, pixel_size_x_m, pixel_size_y_m):
    mean_pixel = 0.5 * (pixel_size_x_m + pixel_size_y_m)

    def residual(params: np.ndarray) -> np.ndarray:
        cx, cy, distance = params
        observed_m = np.sqrt(((xs - cx) * pixel_size_x_m) ** 2 + ((ys - cy) * pixel_size_y_m) ** 2)
        return (observed_m - q_to_ring_radius_m(qs, wavelength_angstrom, distance)) / mean_pixel

    return residual


def _identifiable_two_ring_fit(fit, full_distance: float) -> bool:
    """A two-ring subset must constrain all five parameters inside the joint fit's physical range."""
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return False
    if not 0.75 * full_distance < fit.x[2] < 1.25 * full_distance:
        return False
    if np.any(np.abs(fit.x[3:]) >= math.radians(TILT_LIMIT_DEG)):
        return False
    # Unit-length columns remove the dependence on the chosen units of distance and tilt.
    jac = fit.jac
    norms = np.linalg.norm(jac, axis=0)
    if jac.shape[0] < jac.shape[1] or not np.all(np.isfinite(norms) & (norms > 0)):
        return False
    singular = np.linalg.svd(jac / norms, compute_uv=False)
    return singular[-1] > np.sqrt(np.finfo(float).eps) * singular[0]


def _ring_jackknife(points, rings, make, free, fixed_tilt):
    """Standard errors on the final points. A two-ring joint fit may have too little angular support."""
    xs, ys, qs = points
    scale = np.array([1.0, 1.0, max(1.0, 1e-3 * free[2]), 1e-3, 1e-3])[:free.size]
    check_subset = rings.size == 3 and free.size == 5
    estimates = []
    for ring in rings:
        keep = qs != ring
        residual = make(xs[keep], ys[keep], qs[keep], fixed_tilt)
        try:
            fit = least_squares(residual, free, loss="soft_l1", f_scale=2.0,
                                max_nfev=40 if free.size == 3 else 80, x_scale=scale)
            if check_subset and not _identifiable_two_ring_fit(fit, free[2]):
                return None
        except (ValueError, np.linalg.LinAlgError):
            if check_subset:
                return None
            raise
        estimates.append(fit.x)
    values = np.asarray(estimates)
    n = rings.size
    return np.sqrt((n - 1) / n * np.sum((values - values.mean(axis=0)) ** 2, axis=0))


def estimate_uncertainty(
    points: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    candidate: CalibrationCandidate,
    wavelength_angstrom: float,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> CalibrationCandidate:
    """1σ of the beam centre and the distance of a refined candidate (and of rot1, rot2 when the tilt was fitted).

    The jackknife leaves out each of n rings in turn, keeping the points of the final fit.
    Three rings permit a joint jackknife when all two-ring subsets constrain tilt. Otherwise
    repeat with fixed tilts, using the full-fit covariance for the tilt errors and a floor on
    the centre and distance errors. Two rings use the full-fit covariance alone."""
    if points is None:
        return candidate
    xs, ys, qs = points
    rot3 = math.radians(candidate.rot3_deg)
    rotations = np.radians([candidate.rot1_deg, candidate.rot2_deg])
    centre = np.array([candidate.center_x_px, candidate.center_y_px, candidate.distance_mm])
    if candidate.tilt_refined:
        centre = np.r_[centre, rotations]

    def make(xs, ys, qs, fixed_tilt):
        return tilt_residual(xs, ys, qs, wavelength_angstrom, pixel_size_x_m, pixel_size_y_m, rot3,
                             fixed_tilt=fixed_tilt)

    if xs.size < 30:
        return candidate
    rings = np.unique(qs)
    if rings.size >= 3:
        sigma = _ring_jackknife(points, rings, make, centre, None if candidate.tilt_refined else rotations)
        method = "ring jackknife"
        if sigma is None:
            sigma = _ring_jackknife(points, rings, make, centre[:3], rotations)
            full_sigma = _least_squares_sigma(make(xs, ys, qs, None), centre)
            if full_sigma is not None:
                # Fixed tilts in the jackknife omit the centre/distance dependence on tilt.
                sigma = np.r_[np.maximum(sigma, full_sigma[:3]), full_sigma[3:]]
                method = THREE_RING_UNCERTAINTY_METHOD
        _set_sigma(candidate, sigma, method)
    else:
        sigma = _least_squares_sigma(make(xs, ys, qs, None if candidate.tilt_refined else rotations), centre)
        if sigma is not None:
            _set_sigma(candidate, sigma, "least squares")
    return candidate


def _least_squares_sigma(residual, centre: np.ndarray) -> np.ndarray | None:
    """Standard errors from the raw residuals at the full fit, s² (JᵀJ)⁻¹."""
    steps = np.array([1e-3, 1e-3, max(1e-6, 1e-6 * centre[2]), 1e-7, 1e-7])[:centre.size]
    values = residual(centre)
    jacobian = np.column_stack([(residual(centre + steps * axis) - values) / steps[i]
                                for i, axis in enumerate(np.eye(centre.size))])
    dof = max(1, values.size - centre.size)
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * float(values @ values) / dof
    except np.linalg.LinAlgError:
        return None
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _set_sigma(candidate: CalibrationCandidate, sigma: np.ndarray, method: str) -> None:
    candidate.center_x_sigma_px, candidate.center_y_sigma_px, candidate.distance_sigma_mm = map(float, sigma[:3])
    if sigma.size == 5:
        candidate.rot1_sigma_deg, candidate.rot2_sigma_deg = (math.degrees(float(value)) for value in sigma[3:])
    else:
        candidate.rot1_sigma_deg = candidate.rot2_sigma_deg = None
    candidate.uncertainty_method = method


def refine_candidate(
    analysis: AnalysisImage,
    candidate: CalibrationCandidate,
    wavelength_angstrom: float,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    distance_range_mm: tuple[float, float] | None = None,
) -> CalibrationCandidate:
    if candidate.matched_ring_count < 2:
        candidate.warnings.append("Only one ring is matched; center and distance remain ambiguous.")
        return candidate
    xs, ys, qs = _ring_points(analysis, candidate, pixel_size_x_m, pixel_size_y_m)
    if xs.size < 30:
        candidate.warnings.append("Too few experimental ring points were available for joint refinement.")
        return candidate
    residual = _ring_residual(xs, ys, qs, wavelength_angstrom, pixel_size_x_m, pixel_size_y_m)

    if candidate.center_method == "direct beamstop":
        center_window = 5.0
    elif candidate.center_method == "concentric arc gradients":
        # Partial-arc voting localizes an off-detector center to a broad
        # histogram cell. Give the joint ring fit room to reach the sub-pixel
        # solution without constraining it to the detector bounds.
        center_window = max(40.0, min(140.0, 0.08 * max(analysis.signal.shape)))
    else:
        center_window = max(12.0, min(60.0, 0.04 * max(analysis.signal.shape)))
    initial = np.array([candidate.center_x_px, candidate.center_y_px, candidate.distance_mm])
    low_mm, high_mm = distance_range_mm or (1.0, np.inf)   # the user's range is a hard limit of the fit too
    initial[2] = float(np.clip(initial[2], low_mm, high_mm))
    lower = np.array([initial[0] - center_window, initial[1] - center_window, max(1.0, low_mm, initial[2] * 0.85)])
    upper = np.array([initial[0] + center_window, initial[1] + center_window, min(high_mm, initial[2] * 1.15)])
    if not lower[2] < upper[2]:
        lower[2], upper[2] = initial[2] - 1e-6, initial[2] + 1e-6
    fit = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=2.0, max_nfev=80)
    candidate.center_x_px, candidate.center_y_px, candidate.distance_mm = map(float, fit.x)
    return candidate
