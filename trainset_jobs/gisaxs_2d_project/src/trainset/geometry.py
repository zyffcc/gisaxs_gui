from __future__ import annotations

from typing import Any, Dict

import numpy as np


def roi_to_spherical_ranges(config: Dict[str, Any]) -> Dict[str, float]:
    """Convert a rectangular detector ROI to BornAgain phi/alpha limits.

    This follows the conversion used by ML_GISAXS_Yuxin/simulate_tool.ipynb,
    but uses an explicit ROI (x, y, width, height) instead of ambiguous edge crops.
    """
    detector = config["detector"]
    roi = config["roi"]
    ai_deg = float(config["beam"]["grazing_angle_deg"])
    ai_rad = np.deg2rad(ai_deg)
    distance = float(detector["distance_mm"])
    px = float(detector["pixel_size_x_mm"])
    py = float(detector["pixel_size_y_mm"])

    width_mm = float(roi["width"]) * px
    height_mm = float(roi["height"]) * py
    u0 = (float(detector["beam_center_x_px"]) - float(roi["x"])) * px
    v0 = (float(detector["beam_center_y_px"]) - float(roi["y"])) * py

    phi_1 = -np.rad2deg(np.arctan(u0 / (distance * np.cos(ai_rad))))
    phi_2 = np.rad2deg(np.arctan(-(width_mm - u0) / (distance * np.cos(ai_rad))))
    alpha_min = np.rad2deg(np.arctan((-v0) / distance) - ai_rad)
    alpha_max = np.rad2deg(np.arctan((height_mm - v0) / distance) - ai_rad)
    return {
        "phi_min_deg": float(min(phi_1, phi_2)),
        "phi_max_deg": float(max(phi_1, phi_2)),
        "alpha_min_deg": float(min(alpha_min, alpha_max)),
        "alpha_max_deg": float(max(alpha_min, alpha_max)),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "beam_center_roi_x_mm": u0,
        "beam_center_roi_y_mm": v0,
    }


def q_vectors(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    detector = config["detector"]
    beam = config["beam"]
    nx, ny = int(detector["pixels_x"]), int(detector["pixels_y"])
    px, py = float(detector["pixel_size_x_mm"]), float(detector["pixel_size_y_mm"])
    distance = float(detector["distance_mm"])
    theta_in = np.deg2rad(float(beam["grazing_angle_deg"]))
    wavelength = float(beam["wavelength_nm"])
    center_x = float(detector["beam_center_x_px"]) * px
    center_y = float(detector["beam_center_y_px"]) * py + distance * np.tan(theta_in)
    x = np.arange(nx, dtype=np.float64) * px - center_x
    y = np.arange(ny, dtype=np.float64) * py - center_y
    xx, yy = np.meshgrid(x, y)
    theta_sc = np.arctan2(yy, distance)
    psi = np.arctan2(xx, distance)
    k0 = 2.0 * np.pi / wavelength
    qx = k0 * (np.cos(theta_sc) * np.cos(psi) - np.cos(theta_in))
    qy = k0 * np.cos(theta_sc) * np.sin(psi)
    qz = np.flipud(k0 * (np.sin(theta_sc) + np.sin(theta_in)))
    qr = np.copysign(np.sqrt(qx**2 + qy**2), qy)
    return {"qx": qx, "qy": qy, "qz": qz, "qr": qr}
