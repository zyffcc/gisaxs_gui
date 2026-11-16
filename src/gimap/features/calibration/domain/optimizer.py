"""Calibration candidate numerical refinement."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .geometry_model import q_to_ring_radius_m
from .models import CalibrationCandidate
from .preprocessing import AnalysisImage


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


def estimate_uncertainty(
    analysis: AnalysisImage,
    candidate: CalibrationCandidate,
    wavelength_angstrom: float,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> CalibrationCandidate:
    """1σ of the beam centre and the distance of a refined candidate.

    The points along one ring are not independent (a ring that is slightly off moves all of them together), so the
    least-squares covariance of the points says too little. With three or more matched rings the error is the
    jackknife over rings: the fit is repeated with each ring left out, and
    σ² = (n − 1)/n · Σ (θ_i − θ̄)². With two rings it is the least-squares covariance s² (JᵀJ)⁻¹, s² the mean square
    residual per point, which is a lower bound. Nothing is set when the ring points are too few."""
    if candidate.matched_ring_count < 2:
        return candidate
    xs, ys, qs = _ring_points(analysis, candidate, pixel_size_x_m, pixel_size_y_m)
    if xs.size < 30:
        return candidate
    centre = np.array([candidate.center_x_px, candidate.center_y_px, candidate.distance_mm])
    rings = np.unique(qs)
    if rings.size >= 3:
        estimates = []
        for ring in rings:
            keep = qs != ring
            if np.unique(qs[keep]).size < 2:
                continue
            residual = _ring_residual(xs[keep], ys[keep], qs[keep], wavelength_angstrom, pixel_size_x_m, pixel_size_y_m)
            fit = least_squares(residual, centre, loss="soft_l1", f_scale=2.0, max_nfev=40)
            estimates.append(fit.x)
        if len(estimates) >= 3:
            values = np.asarray(estimates)
            n = values.shape[0]
            sigma = np.sqrt((n - 1) / n * np.sum((values - values.mean(axis=0)) ** 2, axis=0))
            candidate.center_x_sigma_px, candidate.center_y_sigma_px, candidate.distance_sigma_mm = map(float, sigma)
            candidate.uncertainty_method = "ring jackknife"
            return candidate
    residual = _ring_residual(xs, ys, qs, wavelength_angstrom, pixel_size_x_m, pixel_size_y_m)
    values = residual(centre)
    step = np.array([1e-3, 1e-3, max(1e-6, 1e-6 * centre[2])])
    jacobian = np.column_stack([(residual(centre + step * axis) - values) / step[i]
                                for i, axis in enumerate(np.eye(3))])
    dof = max(1, values.size - 3)
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * float(values @ values) / dof
    except np.linalg.LinAlgError:
        return candidate
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    candidate.center_x_sigma_px, candidate.center_y_sigma_px, candidate.distance_sigma_mm = map(float, sigma)
    candidate.uncertainty_method = "least squares"
    return candidate


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
    mean_pixel = 0.5 * (pixel_size_x_m + pixel_size_y_m)

    def residual(params: np.ndarray) -> np.ndarray:
        cx, cy, distance = params
        observed_m = np.sqrt(((xs - cx) * pixel_size_x_m) ** 2 + ((ys - cy) * pixel_size_y_m) ** 2)
        predicted_m = q_to_ring_radius_m(qs, wavelength_angstrom, distance)
        return (observed_m - predicted_m) / mean_pixel

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
