"""Pixel → reciprocal-space mapping: the one implementation all features use.

The mapping is split in two steps so that every historical pixel convention can
be expressed exactly while the physics exists only once:

1. *Displacements*: where the centre of each pixel sits on the detector,
   relative to the direct beam (``X`` to the right, ``Y`` upward, both in the
   same length unit as the sample–detector distance ``D``), and from them the
   laboratory vector from the sample to the pixel (:func:`lab_rays`: ``(D, X, Y)``
   for a detector perpendicular to the beam, turned by the detector's rotations
   otherwise).
2. *Physics*: from that vector, the grazing angle and |k| to the scattering
   vector in the sample frame (``qx`` along the beam, ``qy`` in the surface
   plane perpendicular to it, ``qz`` along the surface normal).

:class:`ExitAngleModel` names the formulas.  ``EXACT`` is the rigorous result for
a flat detector perpendicular to the direct beam and a sample tilted by alpha_i;
it is the only one Analyze uses (and so Fitting, which fits Analyze's curves).
``SUBTRACT_INCIDENCE`` reproduces, unchanged, the approximation Trainset's
training data were made with (``legacy_conventions``).  ``HORIZON_SHIFT`` and
``NO_INCIDENCE_OFFSET`` are the approximations of the former Cut & Fitting
detector part and WAXS page, both removed: no feature uses them, and they are
kept so the tests can state how far those approximations were from ``EXACT``
(docs/architecture/geometry.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .detector_geometry import DetectorGeometry


class ExitAngleModel(str, Enum):
    """How the scattered direction is derived from a detector position."""

    EXACT = "exact"
    """Rigorous: ray direction ``(D, X, Y)/R`` rotated into the tilted sample frame."""

    HORIZON_SHIFT = "horizon_shift"
    """Former Cut & Fitting detector part (removed; tests only):
    alpha_f = atan((Y − D·tan alpha_i)/D), 2θf = atan(X/D)."""

    SUBTRACT_INCIDENCE = "subtract_incidence"
    """Trainset (legacy, still used by ``legacy_conventions``): alpha_f = atan(Y/D) − alpha_i, 2θf = atan(X/D)."""

    NO_INCIDENCE_OFFSET = "no_incidence_offset"
    """Former WAXS page (removed; tests only): alpha_f = atan(Y/√(D² + X²)), 2θf = atan(X/D);
    the beam centre is treated as the horizon, alpha_i only enters through k_i."""


@dataclass(frozen=True)
class GrazingQMap:
    """Scattering-vector components for every pixel, row 0 at the top."""

    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    q_parallel: np.ndarray
    """In-plane magnitude √(qx² + qy²), signed like qy (the usual "qr" axis)."""
    sin_exit: Optional[np.ndarray] = None
    """sin α_f of the scattered ray in vacuum (the exact model; ``None`` for the legacy ones)."""

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


def lab_rays(geometry: DetectorGeometry, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(along the beam, right, up)``: the laboratory vector from the sample to the detector points at in-plane
    displacements ``(x, y)`` from the direct beam (metres, from :func:`region_displacements`).

    A detector perpendicular to the beam gives ``(D, x, y)``. A tilted one (``geometry.rot1_rad`` …, pyFAI's
    rotations) turns the detector plane about the direct beam: ``D·ê_beam + x·ĉ + y·r̂`` with ``ĉ``, ``r̂`` the
    directions of its columns and of its rows upward (``DetectorGeometry.in_plane_axes``). This is pyFAI's pixel
    position written relative to the direct beam."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not geometry.tilted:
        return np.full(np.broadcast(x, y).shape, float(geometry.distance_m)), x, y
    (ax, rx, ux), (ay, ry, uy) = geometry.in_plane_axes()
    return geometry.distance_m + x * ax + y * ay, x * rx + y * ry, x * ux + y * uy


def geometry_rays(
    geometry: DetectorGeometry, rows: tuple[int, int], columns: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`lab_rays` of the pixel centres of a rectangular region."""
    x, y = region_displacements(geometry, rows, columns)
    return lab_rays(geometry, x, y)


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
    ``distance`` must share one length unit. ``distance`` may be an array: the
    along-beam component of every ray of a tilted detector (:func:`lab_rays`).
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


def refracted_qz(sin_exit, sin_incidence: float, critical_rad: float, wavevector: float) -> np.ndarray:
    """q_z inside a film of critical angle α_c: k (Re√(sin²α_f − sin²α_c) + Re√(sin²α_i − sin²α_c)).

    The in-plane components of k are kept at the surface, so only q_z changes. Below α_c a wave in the film is
    evanescent and its term is 0 (α_f below 0 too); q_z then no longer tells the exit angle, so the analyses leave
    those pixels out (``sin_exit < sin α_c``)."""
    sin_c2 = math.sin(critical_rad) ** 2
    sin_exit = np.asarray(sin_exit, dtype=np.float64)
    exit_term = np.where(sin_exit > 0, np.sqrt(np.maximum(sin_exit * sin_exit - sin_c2, 0.0)), 0.0)
    return wavevector * (exit_term + math.sqrt(max(sin_incidence * sin_incidence - sin_c2, 0.0)))


def scattering_vectors(
    x: np.ndarray,
    y: np.ndarray,
    distance: float,
    incidence_rad: float,
    wavevector: float,
    model: ExitAngleModel = ExitAngleModel.EXACT,
    *,
    zero_at_axis: bool = False,
    critical_rad: float = 0.0,
) -> GrazingQMap:
    """q = k_f − k_i in the sample frame, in units of ``wavevector``; with ``critical_rad`` above 0, q_z is the one
    inside the film (:func:`refracted_qz`)."""
    along_beam, across_beam, normal = exit_directions(x, y, distance, incidence_rad, model)
    qx = wavevector * (along_beam - np.cos(incidence_rad))
    qy = wavevector * across_beam
    if critical_rad > 0:
        qz = refracted_qz(normal, math.sin(incidence_rad), critical_rad, wavevector)
    else:
        qz = wavevector * (normal + np.sin(incidence_rad))
    return GrazingQMap(qx, qy, qz, signed_parallel(qx, qy, zero_at_axis=zero_at_axis), normal)


# -- public entry points on DetectorGeometry ---------------------------------------------


def grazing_q_map(
    shape: tuple[int, int],
    geometry: DetectorGeometry,
    model: ExitAngleModel = ExitAngleModel.EXACT,
) -> GrazingQMap:
    """GISAXS/GIWAXS q for every pixel of an image of ``shape``, in Å⁻¹."""
    along, x, y = lab_rays(geometry, *pixel_center_displacements(shape, geometry))
    return scattering_vectors(
        x,
        y,
        along,
        geometry.incidence_rad,
        geometry.wavevector_inv_angstrom,
        model,
        critical_rad=geometry.critical_rad,
    )


def grazing_q_region(
    geometry: DetectorGeometry,
    rows: tuple[int, int],
    columns: tuple[int, int],
    model: ExitAngleModel = ExitAngleModel.EXACT,
) -> GrazingQMap:
    """:func:`grazing_q_map` restricted to a rectangular pixel region, in Å⁻¹."""
    along, x, y = geometry_rays(geometry, rows, columns)
    return scattering_vectors(
        x,
        y,
        along,
        geometry.incidence_rad,
        geometry.wavevector_inv_angstrom,
        model,
        critical_rad=geometry.critical_rad,
    )


def transmission_q(along, x, y, wavevector) -> tuple[np.ndarray, np.ndarray]:
    """Vacuum |q| and 2θ (radians) from laboratory rays relative to the direct beam."""
    two_theta = np.arctan2(np.hypot(x, y), along)
    return 2.0 * wavevector * np.sin(two_theta / 2.0), two_theta


def transmission_map(shape: tuple[int, int], geometry: DetectorGeometry) -> TransmissionMap:
    """|q|, 2θ and χ for transmission SAXS/WAXS, in Å⁻¹ and degrees.

    χ is the azimuth on the detector, 0° to the right and 90° up; it is not the GIWAXS χ of Analyze
    (``analyze/domain/giwaxs.py``: 0° along the surface normal).
    """
    along, x, y = lab_rays(geometry, *pixel_center_displacements(shape, geometry))
    q, two_theta = transmission_q(along, x, y, geometry.wavevector_inv_angstrom)
    return TransmissionMap(q, np.degrees(two_theta), np.degrees(np.arctan2(y, x)))
