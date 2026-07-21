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


def refine_candidate(
    analysis: AnalysisImage,
    candidate: CalibrationCandidate,
    wavelength_angstrom: float,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
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
    lower = np.array([initial[0] - center_window, initial[1] - center_window, max(1.0, initial[2] * 0.85)])
    upper = np.array([initial[0] + center_window, initial[1] + center_window, initial[2] * 1.15])
    fit = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=2.0, max_nfev=80)
    candidate.center_x_px, candidate.center_y_px, candidate.distance_mm = map(float, fit.x)
    return candidate
