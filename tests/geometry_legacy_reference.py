"""Frozen copies of the q-map code that existed before ``src.gimap.shared.geometry``.

These are verbatim copies (only renamed) of the implementations as of
commit c4788ed.  They are the reference for the pixel-by-pixel equivalence
tests in ``test_shared_geometry_equivalence.py`` and must not be edited:
changing them would silently change what "unchanged behaviour" means.
"""

from __future__ import annotations

import numpy as np


def trainset_q_vectors(config):
    detector = config["detector"]
    beam = config["beam"]
    nx, ny = int(detector["pixels_x"]), int(detector["pixels_y"])
    px, py = float(detector["pixel_size_x_mm"]), float(detector["pixel_size_y_mm"])
    distance = float(detector["distance_mm"])
    theta_in = np.deg2rad(float(beam["grazing_angle_deg"]))
    wavelength = float(beam["wavelength_nm"])
    center_x = float(detector["beam_center_x_px"])
    center_y = float(detector["beam_center_y_px"])
    x = (np.arange(nx, dtype=np.float64) - center_x) * px
    y = (center_y - np.arange(ny, dtype=np.float64)) * py
    xx, yy = np.meshgrid(x, y)
    alpha_f = np.arctan2(yy, distance) - theta_in
    psi = np.arctan2(xx, distance)
    k0 = 2.0 * np.pi / wavelength
    qx = k0 * (np.cos(alpha_f) * np.cos(psi) - np.cos(theta_in))
    qy = k0 * np.cos(alpha_f) * np.sin(psi)
    qz = k0 * (np.sin(alpha_f) + np.sin(theta_in))
    qr = np.copysign(np.sqrt(qx**2 + qy**2), qy)
    return {"qx": qx, "qy": qy, "qz": qz, "qr": qr}
