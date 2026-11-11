"""The rotations of a pyFAI ``.poni`` that GIMaP's flat geometry leaves out: |q| and direction separately.

GIMaP's geometry (``DetectorGeometry``) is a flat detector perpendicular to the direct beam. A ``.poni`` file
describes the detector by pyFAI's distance, PONI and three rotations. When one is read
(``assistant/domain/calibration_files.parse_poni``), GIMaP keeps the direct beam (where the ray along the beam
meets the detector) and the distance to it (:meth:`DetectorTilt.flat_geometry`), and drops the rotations.
:class:`DetectorTilt` keeps what was dropped, and :func:`flat_detector_q_error` says how far |q| of a flat geometry is
from |q| of the tilted one. What the tilt alone costs is that comparison with the flat geometry made from the same
``.poni``, not with the geometry in use: a beam centre moved afterwards (the GISAXS symmetry axis, a centre set by
hand in Analyze) is no part of the tilt.

pyFAI's convention (``pyFAI.geometry.Geometry``): a pixel centre lies at ``p1 = (i + ½)·pixel1`` (rows) and
``p2 = (j + ½)·pixel2`` (columns) from the corner of pixel (0, 0), and its vector from the sample, in the
laboratory frame (axis 3 along the beam), is ``R3·R2·R1`` applied to ``(p1 − poni1, p2 − poni2, L)``::

    t1 = p1 c2 c3 + p2 (c3 s1 s2 − c1 s3) − L (c1 c3 s2 + s1 s3)
    t2 = p1 c2 s3 + p2 (c1 c3 + s1 s2 s3) − L (−c3 s1 + c1 s2 s3)
    t3 = p1 s2 − p2 c2 s1 + L c1 c2                                    (cᵢ = cos rotᵢ, sᵢ = sin rotᵢ)

so ``cos 2θ = t3 / |t|``. The flat geometry's ray is ``(D, X, Y)`` with ``X, Y`` from
``q_mapping.region_displacements`` (the one GIMaP uses), so ``cos 2θ = D / √(D² + X² + Y²)``. With
``|q| = 4π sin θ / λ`` the relative error ``sin θ_flat / sin θ_tilted − 1`` depends on neither λ nor αi. A rotation
about the beam (rot3) moves no pixel's angle to the beam: ``t3`` and ``t1² + t2²`` do not depend on it, so it
leaves |q| unchanged (it turns χ by rot3).

Pure numpy: no Qt, no feature imports.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .detector_geometry import DetectorGeometry
from .q_mapping import region_displacements

TILT_LIMIT_DEG = 1.0
"""A dropped tilt above this is an open point of the automatic analysis (and a message in Analyze)."""
Q_ERROR_LIMIT = 0.005
"""…and so is a largest relative |q| error above this (0.5 %), whatever the tilt."""
ROTATION_REVIEW_DEG = 1.0
"""Review an omitted beam-axis rotation above 1°, independently of |q|. Like TILT_LIMIT_DEG this is a reminder,
not an acceptable scientific error bound: every nonzero rotation is recorded and explained, including smaller
ones. It is an exact angular contribution to χ, whereas qy/qz errors depend on the pixel and sample geometry."""
ROTATION_ZERO_TOLERANCE_DEG = 1e-10
"""Only suppress floating-point residue when reducing full turns; not an orientation-accuracy tolerance."""
NEAR_BEAM_PX = 10.0
"""Pixels whose centre lies closer than this (in pixels) to the direct beam are left out: there |q| tends to zero
and a relative error says nothing."""
_BLOCK_PIXELS = 1 << 20
"""Pixels compared at once: a 9-megapixel frame is done in blocks of rows, never all at once."""


@dataclass(frozen=True)
class DetectorTilt:
    """What a ``.poni`` says that GIMaP's flat geometry leaves out: pyFAI's own distance (sample to PONI), PONI,
    rotations (radians) and pixel sizes (metres). ``pixel1``/``poni1`` are along the rows, ``2`` along the columns."""

    distance_m: float
    poni1_m: float
    poni2_m: float
    rot1_rad: float
    rot2_rad: float
    rot3_rad: float
    pixel1_m: float
    pixel2_m: float

    @property
    def tilt_deg(self) -> float:
        """Angle between the detector normal and the beam: ``arccos(cos rot1 · cos rot2)``."""
        cosine = math.cos(self.rot1_rad) * math.cos(self.rot2_rad)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def rotations_deg(self) -> tuple[float, float, float]:
        """rot1, rot2, rot3 in degrees."""
        return math.degrees(self.rot1_rad), math.degrees(self.rot2_rad), math.degrees(self.rot3_rad)

    @property
    def beam_distance_m(self) -> float:
        """The distance from the sample to the direct beam on the detector, ``L / (cos rot1 · cos rot2)``."""
        return self.distance_m / (math.cos(self.rot1_rad) * math.cos(self.rot2_rad))

    def direct_beam_px(self) -> tuple[float, float]:
        """``(x, y)``: where the ray along the beam meets the detector, in GIMaP's pixel coordinates (pixel corners on
        integers, row 0 at the top) — ``x = (poni2 − L tan rot1) / pixel2``, ``y = (poni1 + L tan rot2 / cos rot1) /
        pixel1`` (pyFAI's ``Geometry.getFit2D`` conversion)."""
        distance = self.distance_m
        return (
            (self.poni2_m - distance * math.tan(self.rot1_rad)) / self.pixel2_m,
            (self.poni1_m + distance * math.tan(self.rot2_rad) / math.cos(self.rot1_rad)) / self.pixel1_m,
        )

    def flat_geometry(self, wavelength_angstrom: float = 1.0, incidence_deg: float = 0.0) -> DetectorGeometry:
        """GIMaP's flat geometry made from this ``.poni``: its direct beam, the distance to it and its pixel sizes
        (``parse_poni`` gives the same numbers). |q| errors relative to the tilted geometry depend on neither λ nor αi."""
        x, y = self.direct_beam_px()
        return DetectorGeometry(self.pixel2_m, self.pixel1_m, self.beam_distance_m, x, y, wavelength_angstrom, incidence_deg)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: Optional[Mapping[str, Any]]) -> Optional["DetectorTilt"]:
        """The tilt written by :meth:`to_dict`; ``None`` for nothing, or for values that do not describe one."""
        if not values:
            return None
        try:
            return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class FlatDetectorError:
    """The omitted normal tilt and rotations, and largest/RMS relative |q| error (fractions: 0.063 is 6.3 %) over
    valid pixels (``None`` errors when there were none). Rot3 direction review is independent of these errors."""

    tilt_deg: float
    rotations_deg: tuple[float, float, float]
    """rot1, rot2, rot3 of the ``.poni`` in degrees."""
    q_error_max: Optional[float]
    q_error_rms: Optional[float]
    pixels: int

    @property
    def q_needs_attention(self) -> bool:
        """Above ``TILT_LIMIT_DEG`` or ``Q_ERROR_LIMIT``: a point to check, not only a line among the decisions."""
        return self.tilt_deg > TILT_LIMIT_DEG or (self.q_error_max or 0.0) > Q_ERROR_LIMIT

    @property
    def beam_axis_rotation_deg(self) -> float:
        """The omitted Rot3 modulo full turns, in [-180°, 180°]; raw rotations remain in ``rotations_deg``."""
        angle = math.remainder(self.rotations_deg[2], 360.0)
        return 0.0 if abs(angle) <= ROTATION_ZERO_TOLERANCE_DEG else angle

    @property
    def direction_needs_attention(self) -> bool:
        return abs(self.beam_axis_rotation_deg) > ROTATION_REVIEW_DEG

    @property
    def needs_attention(self) -> bool:
        """Either magnitude/normal-tilt or beam-axis direction needs review; neither blocks reduction."""
        return self.q_needs_attention or self.direction_needs_attention

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_tilt_deg": self.tilt_deg, "dropped_rotations_deg": dict(zip(("rot1", "rot2", "rot3"), self.rotations_deg)),
            "dropped_beam_axis_rotation_deg": self.beam_axis_rotation_deg,
            "q_error_max": self.q_error_max, "q_error_rms": self.q_error_rms, "pixels_compared": self.pixels,
            "needs_attention": self.needs_attention, "q_needs_attention": self.q_needs_attention,
            "direction_needs_attention": self.direction_needs_attention,
            "limits": {"tilt_deg": TILT_LIMIT_DEG, "q_error": Q_ERROR_LIMIT, "near_beam_px": NEAR_BEAM_PX,
                       "beam_axis_rotation_deg": ROTATION_REVIEW_DEG,
                       "rotation_zero_tolerance_deg": ROTATION_ZERO_TOLERANCE_DEG},
        }


def tilted_rays(tilt: DetectorTilt, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t1, t2, t3)``: the vector from the sample to the pixel centres (``rows``, ``columns``: broadcastable index
    arrays) in pyFAI's laboratory frame, t3 along the beam."""
    p1 = (np.asarray(rows, dtype=np.float64) + 0.5) * tilt.pixel1_m - tilt.poni1_m
    p2 = (np.asarray(columns, dtype=np.float64) + 0.5) * tilt.pixel2_m - tilt.poni2_m
    distance = tilt.distance_m
    c1, s1 = math.cos(tilt.rot1_rad), math.sin(tilt.rot1_rad)
    c2, s2 = math.cos(tilt.rot2_rad), math.sin(tilt.rot2_rad)
    c3, s3 = math.cos(tilt.rot3_rad), math.sin(tilt.rot3_rad)
    t1 = p1 * c2 * c3 + p2 * (c3 * s1 * s2 - c1 * s3) - distance * (c1 * c3 * s2 + s1 * s3)
    t2 = p1 * c2 * s3 + p2 * (c1 * c3 + s1 * s2 * s3) - distance * (-c3 * s1 + c1 * s2 * s3)
    t3 = p1 * s2 - p2 * c2 * s1 + distance * c1 * c2
    return t1, t2, t3


def tilted_cos_two_theta(tilt: DetectorTilt, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``cos 2θ`` of the pixel centres in pyFAI's tilted geometry."""
    t1, t2, t3 = tilted_rays(tilt, rows, columns)
    return t3 / np.sqrt(t1**2 + t2**2 + t3**2)


def _sin_theta(along: np.ndarray, across_squared: np.ndarray) -> np.ndarray:
    """``sin θ`` of a ray with ``along`` the beam and ``across_squared`` = the square of the rest, without the
    cancellation of ``√((1 − cos 2θ)/2)`` near the beam: ``2 sin²θ = across² / (R (R + along))``."""
    radius = np.sqrt(along**2 + across_squared)
    return np.sqrt(across_squared / (2.0 * radius * (radius + along)))


def relative_q_error_map(
    tilt: DetectorTilt, geometry: DetectorGeometry, shape: tuple[int, int], *, near_beam_px: float = NEAR_BEAM_PX,
) -> np.ndarray:
    """``sin θ_flat / sin θ_tilted − 1`` of every pixel (float32): |q| of ``geometry`` (a flat detector; for what the
    tilt alone costs, ``tilt.flat_geometry()``) against |q| of the tilted ``.poni`` geometry. NaN within
    ``near_beam_px`` of the direct beam of ``geometry``. Done in blocks of rows, so a large frame never needs more
    than a few arrays of a block."""
    rows, columns = int(shape[0]), int(shape[1])
    out = np.full((rows, columns), np.nan, dtype=np.float32)
    step = max(1, _BLOCK_PIXELS // max(1, columns))
    column_index = np.arange(columns, dtype=np.float64)[None, :]
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        x, y = region_displacements(geometry, (start, stop), (0, columns))
        flat = _sin_theta(np.full(x.shape, geometry.distance_m), x**2 + y**2)
        t1, t2, t3 = tilted_rays(tilt, np.arange(start, stop, dtype=np.float64)[:, None], column_index)
        tilted = _sin_theta(t3, t1**2 + t2**2)
        use = (np.hypot(x / geometry.pixel_size_x_m, y / geometry.pixel_size_y_m) >= float(near_beam_px)) & (tilted > 0.0)
        block = out[start:stop]
        block[use] = flat[use] / tilted[use] - 1.0
    return out


def flat_detector_q_error(
    tilt: DetectorTilt,
    geometry: DetectorGeometry,
    shape: tuple[int, int],
    valid: Optional[np.ndarray] = None,
    *,
    near_beam_px: float = NEAR_BEAM_PX,
    error_map: Optional[np.ndarray] = None,
) -> FlatDetectorError:
    """The largest ``|ε|`` of :func:`relative_q_error_map` and its root mean square over the ``valid`` pixels (all
    when ``None``) farther than ``near_beam_px`` from the direct beam. ``error_map``: that map, when already made
    for this tilt, geometry and shape (a series of frames differs only in its valid pixels)."""
    rows, columns = int(shape[0]), int(shape[1])
    mask = None if valid is None else np.asarray(valid, dtype=bool)
    if mask is not None and mask.shape != (rows, columns):
        raise ValueError(f"The valid mask is {mask.shape}; the frame is {(rows, columns)}.")
    if error_map is None:
        error_map = relative_q_error_map(tilt, geometry, shape, near_beam_px=near_beam_px)
    values = error_map[np.isfinite(error_map) & (mask if mask is not None else True)].astype(np.float64)
    count = int(values.size)
    return FlatDetectorError(
        tilt_deg=tilt.tilt_deg, rotations_deg=tilt.rotations_deg(),
        q_error_max=float(np.abs(values).max()) if count else None,
        q_error_rms=math.sqrt(float(np.mean(values**2))) if count else None,
        pixels=count,
    )


__all__ = [
    "DetectorTilt", "FlatDetectorError", "NEAR_BEAM_PX", "Q_ERROR_LIMIT", "TILT_LIMIT_DEG", "ROTATION_REVIEW_DEG",
    "ROTATION_ZERO_TOLERANCE_DEG", "flat_detector_q_error",
    "relative_q_error_map", "tilted_cos_two_theta", "tilted_rays",
]
