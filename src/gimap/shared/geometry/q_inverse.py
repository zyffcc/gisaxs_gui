"""Reciprocal space back to the detector: where a grazing-incidence (q_y, q_z) falls, and the angles of it.

The inverse of :func:`q_mapping.grazing_q_region` (the ``EXACT`` model, q in Å⁻¹, canonical pixel coordinates:
pixel ``(i, j)`` covers ``[j, j + 1] x [i, i + 1]``, row 0 at the top):

* q_z fixes the exit angle alone: ``sin α_f = q_z/k − sin α_i`` (with refraction in a film of critical angle α_c,
  ``sin² α_f = (q_z/k − Re√(sin² α_i − sin² α_c))² + sin² α_c``, above α_c only);
* q_y = k cos α_f sin 2θ_f, so the scattered direction in the sample frame is ``(along, q_y/k, sin α_f)`` with
  ``along = +√(1 − (q_y/k)² − sin² α_f)`` (forward);
* turned back by α_i into the laboratory and met with the detector plane (tilted or not, ``lab_rays``).

Positions the scattered ray cannot reach (|q| too large, below α_c, behind the detector) are NaN. The 1D
conversions used to type a cut in angles (``alpha_f_of_qz``, ``two_theta_f_of_qy`` …) are exact: α_f depends on
q_z alone; 2θ_f of a q_y is the one on the sample horizon (α_f = 0, q_y = k sin 2θ_f).
"""

from __future__ import annotations

import math

import numpy as np

from .detector_geometry import DetectorGeometry
from .q_mapping import lab_rays, scattering_vectors


def q_of_positions(geometry: DetectorGeometry, x_px, y_px) -> tuple[np.ndarray, np.ndarray]:
    """``(q_y, q_z)`` in Å⁻¹ at continuous detector positions (arrays of any matching shape)."""
    x = (np.asarray(x_px, dtype=np.float64) - geometry.beam_center_x_px) * geometry.pixel_size_x_m
    y = (geometry.beam_center_y_px - np.asarray(y_px, dtype=np.float64)) * geometry.pixel_size_y_m
    along, right, up = lab_rays(geometry, *np.broadcast_arrays(x, y))
    q = scattering_vectors(right, up, along, geometry.incidence_rad, geometry.wavevector_inv_angstrom,
                           critical_rad=geometry.critical_rad)
    return np.asarray(q.qy, dtype=np.float64), np.asarray(q.qz, dtype=np.float64)


def qy_on_horizon(geometry: DetectorGeometry, columns) -> np.ndarray:
    """q_y where each detector column meets α_f = 0 (NaN when it has no unique crossing)."""
    columns = np.asarray(columns, dtype=np.float64)
    return q_of_positions(geometry, columns, geometry.horizon_row(columns))[0]


def exit_sine_of_qz(geometry: DetectorGeometry, qz) -> np.ndarray:
    """sin α_f of the rays with this q_z (NaN where none has it)."""
    k = geometry.wavevector_inv_angstrom
    qz = np.asarray(qz, dtype=np.float64)
    sin_i = math.sin(geometry.incidence_rad)
    if geometry.refraction:
        sin_c2 = math.sin(geometry.critical_rad) ** 2
        exit_term = qz / k - math.sqrt(max(sin_i * sin_i - sin_c2, 0.0))
        with np.errstate(invalid="ignore"):
            sine = np.where(exit_term > 0, np.sqrt(np.maximum(exit_term, 0.0) ** 2 + sin_c2), np.nan)
    else:
        sine = qz / k - sin_i
    return np.where(np.abs(sine) <= 1.0, sine, np.nan)


def qz_of_exit_sine(geometry: DetectorGeometry, sine) -> np.ndarray:
    """q_z (Å⁻¹) of the rays leaving at sin α_f (``refracted_qz`` with refraction)."""
    k = geometry.wavevector_inv_angstrom
    sine = np.asarray(sine, dtype=np.float64)
    sin_i = math.sin(geometry.incidence_rad)
    if geometry.refraction:
        sin_c2 = math.sin(geometry.critical_rad) ** 2
        exit_term = np.where(sine > 0, np.sqrt(np.maximum(sine * sine - sin_c2, 0.0)), 0.0)
        return k * (exit_term + math.sqrt(max(sin_i * sin_i - sin_c2, 0.0)))
    return k * (sine + sin_i)


def alpha_f_of_qz(geometry: DetectorGeometry, qz) -> np.ndarray:
    """Exit angle α_f (degrees) of a q_z."""
    return np.degrees(np.arcsin(exit_sine_of_qz(geometry, qz)))


def qz_of_alpha_f(geometry: DetectorGeometry, alpha_f_deg) -> np.ndarray:
    return qz_of_exit_sine(geometry, np.sin(np.radians(np.asarray(alpha_f_deg, dtype=np.float64))))


def two_theta_f_of_qy(geometry: DetectorGeometry, qy) -> np.ndarray:
    """In-plane exit angle 2θ_f (degrees, the sign of q_y) of a q_y on the sample horizon (α_f = 0)."""
    ratio = np.asarray(qy, dtype=np.float64) / geometry.wavevector_inv_angstrom
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(np.where(np.abs(ratio) <= 1.0, ratio, np.nan)))


def qy_of_two_theta_f(geometry: DetectorGeometry, two_theta_f_deg) -> np.ndarray:
    return geometry.wavevector_inv_angstrom * np.sin(np.radians(np.asarray(two_theta_f_deg, dtype=np.float64)))


def horizon_qz(geometry: DetectorGeometry) -> float:
    """q_z of the sample horizon (α_f = 0; with refraction, of α_f = α_c)."""
    sine = math.sin(geometry.critical_rad) if geometry.refraction else 0.0
    return float(qz_of_exit_sine(geometry, sine))


def positions_of_q(geometry: DetectorGeometry, qy, qz) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` continuous detector coordinates where the ray of each ``(q_y, q_z)`` meets the detector."""
    k = geometry.wavevector_inv_angstrom
    qy, qz = np.broadcast_arrays(np.asarray(qy, dtype=np.float64), np.asarray(qz, dtype=np.float64))
    normal = exit_sine_of_qz(geometry, qz)
    across = qy / k
    with np.errstate(invalid="ignore"):
        along = np.sqrt(1.0 - across * across - normal * normal)  # NaN where no ray has these q
    cos_i, sin_i = math.cos(geometry.incidence_rad), math.sin(geometry.incidence_rad)
    ray = np.stack([along * cos_i - normal * sin_i, across, along * sin_i + normal * cos_i])  # (along beam, right, up)
    return positions_of_rays(geometry, ray)


def positions_of_rays(geometry: DetectorGeometry, ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intersect laboratory rays (along beam, right, up) with the detector. NaN if the ray misses its plane."""
    columns, rows_up = (np.asarray(axis, dtype=np.float64) for axis in geometry.in_plane_axes())
    normal_vector = np.cross(columns, rows_up)
    origin = np.array([geometry.distance_m, 0.0, 0.0])
    facing = np.tensordot(normal_vector, ray, axes=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = float(origin @ normal_vector) / facing
        distance = np.where(np.isfinite(distance) & (distance > 0), distance, np.nan)
    point = ray * distance - origin.reshape((3,) + (1,) * (ray.ndim - 1))
    x = np.tensordot(columns, point, axes=1)
    y = np.tensordot(rows_up, point, axes=1)
    return geometry.beam_center_x_px + x / geometry.pixel_size_x_m, geometry.beam_center_y_px - y / geometry.pixel_size_y_m


def q_per_pixel(geometry: DetectorGeometry) -> float:
    """The change of q from one pixel to the next near the direct beam, k·p/D (Å⁻¹): the width of one bin of a sector
    or line cut, so a bin never splits a pixel and holds about one pixel's step."""
    return geometry.wavevector_inv_angstrom * min(geometry.pixel_size_x_m, geometry.pixel_size_y_m) / geometry.distance_m


__all__ = [
    "alpha_f_of_qz", "exit_sine_of_qz", "horizon_qz", "positions_of_q", "positions_of_rays", "q_of_positions", "q_per_pixel", "qy_on_horizon",
    "qy_of_two_theta_f", "qz_of_alpha_f", "qz_of_exit_sine", "two_theta_f_of_qy",
]
