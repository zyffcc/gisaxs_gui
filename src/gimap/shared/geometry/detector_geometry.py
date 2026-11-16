"""The single description of where the detector sits relative to beam and sample."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return number


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


@dataclass(frozen=True)
class DetectorGeometry:
    """A flat detector, perpendicular to the direct beam or tilted, plus the grazing angle.

    Conventions (the same everywhere in GIMaP from now on):

    * Pixel ``(row i, column j)`` of the analysis array covers the continuous
      detector area ``[j, j + 1] x [i, i + 1]``; its centre is ``(j + 0.5, i + 0.5)``.
      Row 0 is the top of the displayed image, x grows to the right and y grows
      downward.  This is the pixel-corner convention used by pyFAI and by an
      ``imshow(..., extent=(0, width, height, 0))`` display.
    * ``beam_center_x_px`` / ``beam_center_y_px`` locate the *direct* (transmitted)
      beam in those coordinates.  For an untilted detector that is pyFAI's PONI
      expressed in pixels; for a tilted one it is not (a ``.poni`` file is converted
      to the direct beam when read, ``assistant/domain/calibration_files.parse_poni``).
    * ``incidence_deg`` is the grazing angle alpha_i between beam and sample surface;
      0 means transmission geometry.
    * ``rot1_rad``, ``rot2_rad``, ``rot3_rad`` are pyFAI's three detector rotations (as in a ``.poni``); all 0 is a
      detector perpendicular to the beam. With them the detector plane turns about the direct beam, which stays at
      ``beam_center_x_px``, ``beam_center_y_px`` and ``distance_m`` from the sample; where each pixel then lies is
      ``q_mapping.lab_rays``.
    * ``critical_angle_deg`` is the critical angle α_c of the film the beam enters; above 0 the grazing q_z is the
      one inside the film (refraction, ``q_mapping.refracted_qz``), 0 (the default) the kinematic q_z in vacuum. It
      belongs to the sample, not to the instrument: a profile keeps 0 and Analyze sets it per measurement.
    * Lengths are metres, the wavelength is in Å, so q comes out in Å⁻¹.
    """

    pixel_size_x_m: float
    pixel_size_y_m: float
    distance_m: float
    beam_center_x_px: float
    beam_center_y_px: float
    wavelength_angstrom: float
    incidence_deg: float = 0.0
    rot1_rad: float = 0.0
    rot2_rad: float = 0.0
    rot3_rad: float = 0.0
    critical_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixel_size_x_m", _positive("pixel_size_x_m", self.pixel_size_x_m))
        object.__setattr__(self, "pixel_size_y_m", _positive("pixel_size_y_m", self.pixel_size_y_m))
        object.__setattr__(self, "distance_m", _positive("distance_m", self.distance_m))
        object.__setattr__(
            self, "wavelength_angstrom", _positive("wavelength_angstrom", self.wavelength_angstrom)
        )
        object.__setattr__(self, "beam_center_x_px", _finite("beam_center_x_px", self.beam_center_x_px))
        object.__setattr__(self, "beam_center_y_px", _finite("beam_center_y_px", self.beam_center_y_px))
        incidence = _finite("incidence_deg", self.incidence_deg)
        if not -90.0 < incidence < 90.0:
            raise ValueError(f"incidence_deg must be within (-90, 90), got {incidence!r}.")
        object.__setattr__(self, "incidence_deg", incidence)
        for name in ("rot1_rad", "rot2_rad", "rot3_rad"):
            angle = _finite(name, getattr(self, name))
            if name != "rot3_rad" and not abs(angle) < math.pi / 2:
                raise ValueError(f"{name} must be within (-90°, 90°), got {angle!r} rad.")
            object.__setattr__(self, name, angle)
        critical = _finite("critical_angle_deg", self.critical_angle_deg)
        if not 0.0 <= critical < 90.0:
            raise ValueError(f"critical_angle_deg must be within [0, 90), got {critical!r}.")
        object.__setattr__(self, "critical_angle_deg", critical)

    @property
    def tilted(self) -> bool:
        """Whether the detector has any rotation (pyFAI's rot1, rot2 or rot3)."""
        return bool(self.rot1_rad or self.rot2_rad or self.rot3_rad)

    @property
    def tilt_deg(self) -> float:
        """Angle between the detector normal and the beam, ``arccos(cos rot1 · cos rot2)``."""
        cosine = math.cos(self.rot1_rad) * math.cos(self.rot2_rad)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def in_plane_axes(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The laboratory vectors ``(along the beam, right, up)`` of one metre along the detector's columns (to the
        right of the image) and along its rows upward. Untilted: ``(0, 1, 0)`` and ``(0, 0, 1)``.

        pyFAI places a pixel at ``M · (p1 − poni1, p2 − poni2, L)`` with ``M = R3 R2 R1`` in its frame (1 down the
        rows, 2 along the columns, 3 along the beam); a step along the columns is ``M[:, 1]``, a step up the rows
        ``−M[:, 0]``; GIMaP's laboratory axes are (3, 2, −1)."""
        c1, s1 = math.cos(self.rot1_rad), math.sin(self.rot1_rad)
        c2, s2 = math.cos(self.rot2_rad), math.sin(self.rot2_rad)
        c3, s3 = math.cos(self.rot3_rad), math.sin(self.rot3_rad)
        down = (c2 * c3, c2 * s3, s2)                                      # M[:, 0] in pyFAI's (1, 2, 3)
        right = (c3 * s1 * s2 - c1 * s3, c1 * c3 + s1 * s2 * s3, -c2 * s1)  # M[:, 1]
        lab = lambda v, sign: (sign * v[2], sign * v[1], -sign * v[0])     # noqa: E731
        return lab(right, 1.0), lab(down, -1.0)

    @property
    def wavevector_inv_angstrom(self) -> float:
        """|k| = 2π/λ in Å⁻¹."""
        return 2.0 * math.pi / self.wavelength_angstrom

    @property
    def incidence_rad(self) -> float:
        return math.radians(self.incidence_deg)

    def with_beam_center(self, x_px: float, y_px: float) -> "DetectorGeometry":
        return replace(self, beam_center_x_px=float(x_px), beam_center_y_px=float(y_px))

    def with_incidence(self, incidence_deg: float) -> "DetectorGeometry":
        return replace(self, incidence_deg=float(incidence_deg))

    @property
    def refraction(self) -> bool:
        """Whether q_z is corrected for refraction in a film (``critical_angle_deg`` above 0)."""
        return self.critical_angle_deg > 0.0

    @property
    def critical_rad(self) -> float:
        return math.radians(self.critical_angle_deg)

    def with_critical_angle(self, critical_angle_deg: float) -> "DetectorGeometry":
        return replace(self, critical_angle_deg=float(critical_angle_deg))

    def with_rotations(self, rot1_rad: float, rot2_rad: float, rot3_rad: float) -> "DetectorGeometry":
        return replace(self, rot1_rad=float(rot1_rad), rot2_rad=float(rot2_rad), rot3_rad=float(rot3_rad))

    def cropped(self, left_px: int = 0, top_px: int = 0) -> "DetectorGeometry":
        """Geometry of a sub-image whose top-left corner is at ``(left_px, top_px)``."""
        return replace(
            self,
            beam_center_x_px=self.beam_center_x_px - float(left_px),
            beam_center_y_px=self.beam_center_y_px - float(top_px),
        )

    # -- positions of characteristic features at the beam-centre column -----------

    def row_for_exit_angle(self, alpha_f_deg: float) -> float:
        """Continuous row coordinate where the exit angle equals ``alpha_f_deg``.

        Evaluated on the beam-centre column. For a detector perpendicular to the
        beam the relation is ``Y = D·tan(alpha_f + alpha_i)`` above the direct beam;
        a tilted detector is solved for the row numerically.
        """
        angle = math.radians(float(alpha_f_deg)) + self.incidence_rad
        if not -math.pi / 2 < angle < math.pi / 2:
            raise ValueError("The requested exit angle does not reach the detector.")
        height_m = self.distance_m * math.tan(angle)
        row = self.beam_center_y_px - height_m / self.pixel_size_y_m
        if not self.tilted:
            return row
        target = math.radians(float(alpha_f_deg))
        for _ in range(60):  # Newton on the row, with a numerical slope; converges in a few steps
            here = self.exit_angle_at(self.beam_center_x_px, row)
            step = 1e-3
            slope = (self.exit_angle_at(self.beam_center_x_px, row + step) - here) / step
            if slope == 0.0:
                break
            change = (here - target) / slope
            row -= change
            if abs(change) < 1e-9:
                break
        return row

    def exit_angle_at(self, x_px: float, y_px: float) -> float:
        """Exit angle alpha_f (radians) of the ray to a continuous detector position."""
        (ax, rx, ux), (ay, ry, uy) = self.in_plane_axes()
        x = (float(x_px) - self.beam_center_x_px) * self.pixel_size_x_m
        y = (self.beam_center_y_px - float(y_px)) * self.pixel_size_y_m
        along, up = self.distance_m + x * ax + y * ay, x * ux + y * uy
        right = x * rx + y * ry
        radius = math.sqrt(along * along + right * right + up * up)
        normal = (up * math.cos(self.incidence_rad) - along * math.sin(self.incidence_rad)) / radius
        return math.asin(max(-1.0, min(1.0, normal)))

    def horizon_row(self) -> float:
        """Row of the sample horizon (alpha_f = 0)."""
        return self.row_for_exit_angle(0.0)

    def specular_row(self) -> float:
        """Row of the specularly reflected beam (alpha_f = alpha_i)."""
        return self.row_for_exit_angle(self.incidence_deg)
