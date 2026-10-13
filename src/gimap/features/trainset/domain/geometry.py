"""Trainset detector/q-space pure geometry。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.gimap.shared.geometry.legacy_conventions import trainset_q_vectors


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
    center_x = float(detector["beam_center_x_px"])
    center_y = float(detector["beam_center_y_px"])
    left_x = float(roi["x"])
    right_x = left_x + float(roi["width"] - 1)
    top_y = float(roi["y"])
    bottom_y = top_y + float(roi["height"] - 1)

    # One explicit display convention is used everywhere:
    # detector x grows to the right, detector y grows downward, and the top of
    # the displayed image has the larger exit angle/qz.  BornAgain returns
    # alpha in ascending order, so simulation.py flips its result vertically
    # exactly once to recover this detector-display orientation.
    phi_left = np.rad2deg(np.arctan(((left_x - center_x) * px) / (distance * np.cos(ai_rad))))
    phi_right = np.rad2deg(np.arctan(((right_x - center_x) * px) / (distance * np.cos(ai_rad))))
    alpha_top = np.rad2deg(np.arctan(((center_y - top_y) * py) / distance) - ai_rad)
    alpha_bottom = np.rad2deg(np.arctan(((center_y - bottom_y) * py) / distance) - ai_rad)
    u0 = (center_x - left_x) * px
    v0 = (center_y - top_y) * py
    return {
        "phi_min_deg": float(min(phi_left, phi_right)),
        "phi_max_deg": float(max(phi_left, phi_right)),
        "alpha_min_deg": float(min(alpha_top, alpha_bottom)),
        "alpha_max_deg": float(max(alpha_top, alpha_bottom)),
        "alpha_top_deg": float(alpha_top),
        "alpha_bottom_deg": float(alpha_bottom),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "beam_center_roi_x_mm": u0,
        "beam_center_roi_y_mm": v0,
    }


def q_vectors(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Detector q grids in nm⁻¹; the physics lives in shared.geometry."""
    return trainset_q_vectors(config)
