"""WAXS/GIWAXS detector geometry pure calculations。"""

from __future__ import annotations

import numpy as np


UNSET_Q_LIMIT = -121.0


def compute_q_maps(
    shape: tuple[int, int], geometry: dict
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape[:2]
    x_center = float(geometry["center_x"])
    y_center = float(geometry["center_y"])
    distance = float(geometry["distance"])
    pixel_x = float(geometry["pixel_x"])
    pixel_y = float(geometry["pixel_y"])
    wavelength = float(geometry["wavelength"])
    incidence = float(geometry["incidence"]) * np.pi / 180.0

    yy, xx = np.indices((height, width), dtype=float)
    qr_pix = (xx + 1.0) - x_center
    y_c = height - y_center
    qz_pix = (height - y_c) - (yy + 1.0)
    qr_m = qr_pix * pixel_x * 1e-6
    qz_m = qz_pix * pixel_y * 1e-6
    distance_m = distance * 1e-3
    theta_f = np.arctan(qr_m / distance_m) / 2.0
    alpha_f = np.arctan(qz_m / np.sqrt(distance_m**2 + qr_m**2))
    qx = 2 * np.pi / wavelength * (
        np.cos(2 * theta_f) * np.cos(alpha_f) - np.cos(incidence)
    )
    qy = 2 * np.pi / wavelength * np.sin(2 * theta_f) * np.cos(alpha_f)
    qz = 2 * np.pi / wavelength * (np.sin(alpha_f) + np.sin(incidence))
    qr = np.sign(qy) * np.sqrt(qx**2 + qy**2)
    return qr, qz


def q_range_mask(shape: tuple[int, int], geometry: dict) -> np.ndarray:
    qr, qz = compute_q_maps(shape, geometry)
    mask = np.ones(shape[:2], dtype=bool)
    for key, op, grid in (
        ("qr_min", np.greater_equal, qr),
        ("qr_max", np.less_equal, qr),
        ("qz_min", np.greater_equal, qz),
        ("qz_max", np.less_equal, qz),
    ):
        value = float(geometry.get(key, UNSET_Q_LIMIT))
        if value != UNSET_Q_LIMIT:
            mask &= op(grid, value)
    return mask


def cut_image_by_q_range(
    image: np.ndarray, geometry: dict
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    source = np.asarray(image)
    qr, qz = compute_q_maps(source.shape, geometry)
    cut = np.where(q_range_mask(source.shape, geometry), source, np.nan)
    finite_qr = qr[np.isfinite(qr)]
    finite_qz = qz[np.isfinite(qz)]
    extent = None
    if finite_qr.size and finite_qz.size:
        extent = (
            float(np.nanmin(finite_qr)),
            float(np.nanmax(finite_qr)),
            float(np.nanmin(finite_qz)),
            float(np.nanmax(finite_qz)),
        )
    return cut, extent
