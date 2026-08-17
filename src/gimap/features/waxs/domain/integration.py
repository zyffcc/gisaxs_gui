"""WAXS 1D integration and cut pure calculations。"""

from __future__ import annotations

import numpy as np

from .geometry import compute_q_maps, q_range_mask


def _binned_mean(x, y, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), bins + 1)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1e-9
    indices = np.digitize(x, edges) - 1
    valid_bins = (indices >= 0) & (indices < bins)
    indices = indices[valid_bins]
    y = y[valid_bins]
    sums = np.bincount(indices, weights=y, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    means = np.divide(
        sums,
        counts,
        out=np.full(bins, np.nan, dtype=float),
        where=counts > 0,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    finite = np.isfinite(means)
    return centers[finite], means[finite]


def integrate_image(
    image: np.ndarray,
    geometry: dict,
    integration: dict,
    mask_min: float,
    mask_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float)
    valid = (
        np.isfinite(arr)
        & (arr >= mask_min)
        & (arr <= mask_max)
        & q_range_mask(arr.shape, geometry)
    )
    qr, qz = compute_q_maps(arr.shape, geometry)
    mode = integration.get("mode", "radial")
    bins = int(integration.get("bins", 500))
    axis_mode = integration.get("x_axis", "q")

    if mode == "azimuthal":
        yy, xx = np.indices(arr.shape, dtype=float)
        x_values = np.degrees(
            np.arctan2(yy - geometry["center_y"], xx - geometry["center_x"])
        )
    elif axis_mode == "pixel":
        yy, xx = np.indices(arr.shape, dtype=float)
        x_values = np.sqrt(
            (xx - geometry["center_x"]) ** 2
            + (yy - geometry["center_y"]) ** 2
        )
    elif axis_mode == "2theta":
        yy, xx = np.indices(arr.shape, dtype=float)
        radius_m = np.sqrt(
            ((xx - geometry["center_x"]) * geometry["pixel_x"] * 1e-6) ** 2
            + ((yy - geometry["center_y"]) * geometry["pixel_y"] * 1e-6) ** 2
        )
        x_values = np.degrees(
            np.arctan(radius_m / (geometry["distance"] * 1e-3))
        )
    else:
        x_values = np.sqrt(qr**2 + qz**2)

    x = x_values[valid]
    y = arr[valid]
    if x.size == 0:
        raise RuntimeError("No valid pixels in the selected integration region.")
    return _binned_mean(x, y, bins)


def line_cut_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    mask_min: float,
    mask_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float)
    image_height, image_width = arr.shape[:2]
    x0 = max(0, int(np.floor(center_x - width / 2.0)))
    x1 = min(image_width, int(np.ceil(center_x + width / 2.0)))
    y0 = max(0, int(np.floor(center_y - height / 2.0)))
    y1 = min(image_height, int(np.ceil(center_y + height / 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Line cut region is empty.")
    region = arr[y0:y1, x0:x1].copy()
    region[(region < mask_min) | (region > mask_max)] = np.nan
    if width >= height:
        y = np.nanmean(region, axis=0)
        x = np.arange(x0, x1, dtype=float)
    else:
        y = np.nanmean(region, axis=1)
        x = np.arange(y0, y1, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        raise RuntimeError("No valid pixels in the selected line cut.")
    return x[finite], y[finite]


def normalize_angle_deg(angle):
    return (np.asarray(angle) + 360.0) % 360.0


def angle_between(angle: np.ndarray, start: float, end: float) -> np.ndarray:
    if start <= end:
        return (angle >= start) & (angle <= end)
    return (angle >= start) | (angle <= end)


def circle_cut_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    bins: int,
    *,
    mode: str,
    mask_min: float,
    mask_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float)
    start = normalize_angle_deg(float(start_angle))
    end = normalize_angle_deg(float(end_angle))
    inner = min(float(inner_radius), float(outer_radius))
    outer = max(float(inner_radius), float(outer_radius))
    height, width = arr.shape[:2]
    cx = float(center_x)
    cy = float(center_y)
    x0 = max(0, int(np.floor(cx - outer)))
    x1 = min(width, int(np.ceil(cx + outer)) + 1)
    y0 = max(0, int(np.floor(cy - outer)))
    y1 = min(height, int(np.ceil(cy + outer)) + 1)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Circle cut region is outside the image.")
    region = arr[y0:y1, x0:x1]
    yy, xx = np.indices(region.shape, dtype=np.float32)
    xx += float(x0)
    yy += float(y0)
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    angle = normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
    valid = (
        np.isfinite(region)
        & (region >= mask_min)
        & (region <= mask_max)
        & (radius >= inner)
        & (radius <= outer)
        & angle_between(angle, start, end)
    )
    if not np.any(valid):
        raise RuntimeError("No valid pixels in the selected circle cut.")
    x_values = angle if mode == "azimuthal" else radius
    return _binned_mean(x_values[valid], region[valid], int(bins))


def smooth_curve(y: np.ndarray, window: int = 7) -> np.ndarray:
    if y.size < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")
