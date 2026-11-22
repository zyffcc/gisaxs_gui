"""What the cursor readout says at a point of the detector, the q map or the cake: every coordinate at once.

At a detector position (GIMaP pixel coordinates) the numbers come from the one pixel→q mapping
(``shared/geometry/q_mapping``): q and its components (q_y, q_z, and q_∥ for grazing incidence), the scattering
angle 2θ (between the ray and the direct beam), the azimuth χ, the exit angle α_f (grazing incidence) and
d = 2π/q. χ is the convention of the technique: for GISAXS and GIWAXS atan2(q_∥, q_z), 0° along the surface normal
(``domain/giwaxs``); in transmission atan2(q_z, q_y), 0° to the right of the beam and 90° up
(``domain/transmission``). On the q map and on the cake the point is given in q and the rest follows from it
(2θ from |q| and λ; α_f from q_z = k (sin α_f + sin α_i), without refraction).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry, lab_rays, scattering_vectors

from ..domain import TRANSMISSION


def _two_theta_from_q(q: float, wavelength: float) -> Optional[float]:
    ratio = q * wavelength / (4.0 * math.pi)
    return math.degrees(2.0 * math.asin(ratio)) if 0.0 <= ratio <= 1.0 else None


def _spacing(q: float) -> Optional[float]:
    return 2.0 * math.pi / q if q > 1e-9 else None


def rod_rotation_deg(geometry: DetectorGeometry, first, second) -> float:
    """Rotation nearest the current rot3 that puts the picked ridge in the sample incidence plane.

    Turning rot3 by δ rotates (right, up) about the beam. The horizontal component of the
    difference in the rolled basis must vanish. A ridge has no arrow, hence the correction is modulo 180°.
    The two q_y readouts separately test that the line also passes through the direct beam.
    """
    dx = (second[0] - first[0]) * geometry.pixel_size_x_m
    dy = (first[1] - second[1]) * geometry.pixel_size_y_m
    _, right, up = lab_rays(geometry, dx, dy)
    # Δright after an additional roll δ is right·cos δ − up·sin δ.
    correction = (math.degrees(math.atan2(right, up)) - geometry.sample_roll_deg + 90.0) % 180.0 - 90.0
    return (math.degrees(geometry.rot3_rad) + correction + 180.0) % 360.0 - 180.0


def detector_point(geometry: DetectorGeometry, x_px: float, y_px: float, kind: Optional[str]) -> dict:
    """The coordinates of a detector position (see the module docstring); ``alpha_f_deg`` and ``q_parallel`` are
    ``None`` in transmission."""
    x = np.array([(float(x_px) - geometry.beam_center_x_px) * geometry.pixel_size_x_m])
    y = np.array([(geometry.beam_center_y_px - float(y_px)) * geometry.pixel_size_y_m])
    along, right, up = lab_rays(geometry, x, y)
    two_theta = math.degrees(math.atan2(float(np.hypot(right, up)[0]), float(along[0])))
    if kind == TRANSMISSION:
        k = geometry.wavevector_inv_angstrom
        radius = float(np.sqrt(along**2 + right**2 + up**2)[0])
        qy, qz = k * float(right[0]) / radius, k * float(up[0]) / radius
        q = 2.0 * k * math.sin(math.radians(two_theta) / 2.0)
        return {"q": q, "qy": qy, "qz": qz, "q_parallel": None, "two_theta_deg": two_theta,
                "chi_deg": math.degrees(math.atan2(qz, qy)), "alpha_f_deg": None, "d_angstrom": _spacing(q)}
    vectors = scattering_vectors(right, up, along, geometry.incidence_rad, geometry.wavevector_inv_angstrom,
                                 critical_rad=geometry.critical_rad, sample_roll_rad=geometry.sample_roll_rad)
    q = float(vectors.q[0])
    q_parallel, qz = float(vectors.q_parallel[0]), float(vectors.qz[0])
    sin_exit = float(vectors.sin_exit[0])
    return {"q": q, "qy": float(vectors.qy[0]), "qz": qz, "q_parallel": q_parallel, "two_theta_deg": two_theta,
            "chi_deg": math.degrees(math.atan2(q_parallel, qz)),
            "alpha_f_deg": math.degrees(math.asin(max(-1.0, min(1.0, sin_exit)))), "d_angstrom": _spacing(q)}


def map_point(geometry: DetectorGeometry, horizontal: float, qz: float, kind: Optional[str]) -> dict:
    """The coordinates of a point of the q map (``horizontal``: q_∥ for GIWAXS, q_y for GISAXS and transmission)."""
    q = math.hypot(horizontal, qz)
    alpha_f = None
    if kind != TRANSMISSION:
        sin_exit = qz / geometry.wavevector_inv_angstrom - math.sin(geometry.incidence_rad)
        alpha_f = math.degrees(math.asin(sin_exit)) if -1.0 <= sin_exit <= 1.0 else None
    chi = math.degrees(math.atan2(qz, horizontal)) if kind == TRANSMISSION else math.degrees(math.atan2(horizontal, qz))
    return {"q": q, "horizontal": horizontal, "qz": qz, "two_theta_deg": _two_theta_from_q(q, geometry.wavelength_angstrom),
            "chi_deg": chi, "alpha_f_deg": alpha_f, "d_angstrom": _spacing(q)}


def cake_point(geometry: DetectorGeometry, q: float, chi: float) -> dict:
    return {"q": q, "chi_deg": chi, "two_theta_deg": _two_theta_from_q(q, geometry.wavelength_angstrom),
            "d_angstrom": _spacing(q)}


__all__ = ["cake_point", "detector_point", "map_point"]
