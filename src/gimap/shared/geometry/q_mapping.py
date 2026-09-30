"""Pixel → reciprocal-space mapping: the one implementation all features use.

The mapping is split in two steps so that every historical pixel convention can
be expressed exactly while the physics exists only once:

1. *Displacements*: where the centre of each pixel sits on the detector,
   relative to the direct beam (``X`` to the right, ``Y`` upward, both in the
   same length unit as the sample–detector distance ``D``).
2. *Physics*: from ``(X, Y, D)``, the grazing angle and |k| to the scattering
   vector in the sample frame (``qx`` along the beam, ``qy`` in the surface
   plane perpendicular to it, ``qz`` along the surface normal).

:class:`ExitAngleModel` names the formulas in use.  ``EXACT`` is the rigorous
result for a flat detector perpendicular to the direct beam and a sample tilted
by alpha_i.  The three others reproduce, unchanged, the approximations that the
Fitting, Trainset and WAXS features have historically used; they exist so those
features can share this code without changing a single computed value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .detector_geometry import DetectorGeometry


class ExitAngleModel(str, Enum):
    """How the scattered direction is derived from a detector position."""

    EXACT = "exact"
    """Rigorous: ray direction ``(D, X, Y)/R`` rotated into the tilted sample frame."""

    HORIZON_SHIFT = "horizon_shift"
    """Fitting (legacy): alpha_f = atan((Y − D·tan alpha_i)/D), 2θf = atan(X/D)."""

    SUBTRACT_INCIDENCE = "subtract_incidence"
    """Trainset (legacy): alpha_f = atan(Y/D) − alpha_i, 2θf = atan(X/D)."""

    NO_INCIDENCE_OFFSET = "no_incidence_offset"
    """WAXS (legacy): alpha_f = atan(Y/√(D² + X²)), 2θf = atan(X/D); the beam centre
    is treated as the horizon, alpha_i only enters through k_i."""


@dataclass(frozen=True)
class GrazingQMap:
    """Scattering-vector components for every pixel, row 0 at the top."""

    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    q_parallel: np.ndarray
    """In-plane magnitude √(qx² + qy²), signed like qy (the usual "qr" axis)."""

    @property
    def q(self) -> np.ndarray:
        return np.sqrt(self.qx**2 + self.qy**2 + self.qz**2)


@dataclass(frozen=True)
class TransmissionMap:
    """Ring coordinates for SAXS/WAXS in transmission (alpha_i ignored)."""

    q: np.ndarray
    two_theta_deg: np.ndarray
    chi_deg: np.ndarray
    """Azimuth of the pixel around the direct beam: 0° to the right, 90° up."""


# -- step 1: displacements ------------------------------------------------------------


def pixel_center_displacements(
    shape: tuple[int, int], geometry: DetectorGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` in metres of every pixel centre relative to the direct beam."""
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise ValueError(f"Image shape must be positive, got {shape!r}.")
    return region_displacements(geometry, (0, rows), (0, columns))


def region_displacements(
    geometry: DetectorGeometry, rows: tuple[int, int], columns: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` in metres for the pixel centres of ``rows[0]:rows[1]``, ``columns[0]:columns[1]``.

    Identical to slicing :func:`pixel_center_displacements`, without building
    the full-frame arrays (large detectors have > 10⁷ pixels).
    """
    row_start, row_stop = int(rows[0]), int(rows[1])
    column_start, column_stop = int(columns[0]), int(columns[1])
    if row_stop <= row_start or column_stop <= column_start:
        raise ValueError(f"Empty pixel region rows={rows!r}, columns={columns!r}.")
    x = (
        np.arange(column_start, column_stop, dtype=np.float64) + 0.5 - geometry.beam_center_x_px
    ) * geometry.pixel_size_x_m
    y = (
        geometry.beam_center_y_px - (np.arange(row_start, row_stop, dtype=np.float64) + 0.5)
    ) * geometry.pixel_size_y_m
    return np.meshgrid(x, y)


# -- step 2: physics ------------------------------------------------------------------


def exit_directions(
    x: np.ndarray,
    y: np.ndarray,
    distance: float,
    incidence_rad: float,
    model: ExitAngleModel = ExitAngleModel.EXACT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vector of the scattered ray in the sample frame.

    Returns ``(cos αf·cos 2θf, cos αf·sin 2θf, sin αf)``; ``x``, ``y`` and
    ``distance`` must share one length unit.
    """
    model = ExitAngleModel(model)
    if model is ExitAngleModel.EXACT:
        radius = np.sqrt(distance**2 + x**2 + y**2)
        cos_i, sin_i = np.cos(incidence_rad), np.sin(incidence_rad)
        along_beam = (distance * cos_i + y * sin_i) / radius
        across_beam = x / radius
        normal = (y * cos_i - distance * sin_i) / radius
        return along_beam, across_beam, normal
    if model is ExitAngleModel.HORIZON_SHIFT:
        alpha_f = np.arctan2(y - distance * np.tan(incidence_rad), distance)
        two_theta_f = np.arctan2(x, distance)
    elif model is ExitAngleModel.SUBTRACT_INCIDENCE:
        alpha_f = np.arctan2(y, distance) - incidence_rad
        two_theta_f = np.arctan2(x, distance)
    else:  # NO_INCIDENCE_OFFSET
        two_theta_f = np.arctan(x / distance)
        alpha_f = np.arctan(y / np.sqrt(distance**2 + x**2))
    cos_f = np.cos(alpha_f)
    return cos_f * np.cos(two_theta_f), cos_f * np.sin(two_theta_f), np.sin(alpha_f)


def signed_parallel(qx: np.ndarray, qy: np.ndarray, *, zero_at_axis: bool = False) -> np.ndarray:
    """√(qx² + qy²) carrying the sign of qy.

    ``zero_at_axis=True`` reproduces the WAXS convention ``sign(qy)·|q∥|``,
    which returns exactly 0 where qy == 0; otherwise ``copysign`` is used.
    """
    magnitude = np.sqrt(qx**2 + qy**2)
    if zero_at_axis:
        return np.sign(qy) * magnitude
    return np.copysign(magnitude, qy)


def scattering_vectors(
    x: np.ndarray,
    y: np.ndarray,
    distance: float,
    incidence_rad: float,
    wavevector: float,
    model: ExitAngleModel = ExitAngleModel.EXACT,
    *,
    zero_at_axis: bool = False,
) -> GrazingQMap:
    """q = k_f − k_i in the sample frame, in units of ``wavevector``."""
    along_beam, across_beam, normal = exit_directions(x, y, distance, incidence_rad, model)
    qx = wavevector * (along_beam - np.cos(incidence_rad))
    qy = wavevector * across_beam
    qz = wavevector * (normal + np.sin(incidence_rad))
    return GrazingQMap(qx, qy, qz, signed_parallel(qx, qy, zero_at_axis=zero_at_axis))


# -- public entry points on DetectorGeometry ---------------------------------------------


def grazing_q_map(
    shape: tuple[int, int],
    geometry: DetectorGeometry,
    model: ExitAngleModel = ExitAngleModel.EXACT,
) -> GrazingQMap:
    """GISAXS/GIWAXS q for every pixel of an image of ``shape``, in Å⁻¹."""
    x, y = pixel_center_displacements(shape, geometry)
    return scattering_vectors(
        x,
        y,
        geometry.distance_m,
        geometry.incidence_rad,
        geometry.wavevector_inv_angstrom,
        model,
    )


def grazing_q_region(
    geometry: DetectorGeometry,
    rows: tuple[int, int],
    columns: tuple[int, int],
    model: ExitAngleModel = ExitAngleModel.EXACT,
) -> GrazingQMap:
    """:func:`grazing_q_map` restricted to a rectangular pixel region, in Å⁻¹."""
    x, y = region_displacements(geometry, rows, columns)
    return scattering_vectors(
        x,
        y,
        geometry.distance_m,
        geometry.incidence_rad,
        geometry.wavevector_inv_angstrom,
        model,
    )


def transmission_map(shape: tuple[int, int], geometry: DetectorGeometry) -> TransmissionMap:
    """|q|, 2θ and χ for transmission SAXS/WAXS, in Å⁻¹ and degrees."""
    x, y = pixel_center_displacements(shape, geometry)
    two_theta = np.arctan2(np.hypot(x, y), geometry.distance_m)
    q = 2.0 * geometry.wavevector_inv_angstrom * np.sin(two_theta / 2.0)
    return TransmissionMap(q, np.degrees(two_theta), np.degrees(np.arctan2(y, x)))
