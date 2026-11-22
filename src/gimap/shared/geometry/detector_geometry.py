"""The single description of where the detector sits relative to beam and sample."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .orientation import orientation_number


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


def roll_transverse(right, up, roll_rad: float):
    """Lab (right, up) in the rolled sample's transverse axes. Invert by passing −ψ.

    Positive ψ leans the surface normal toward lab right, as in the Sputter client.
    The zero branch preserves the original floating-point operations.
    """
    if roll_rad == 0.0:
        return right, up
    cosine, sine = math.cos(roll_rad), math.sin(roll_rad)
    return cosine * right - sine * up, sine * right + cosine * up


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
    * ``orientation`` is pyFAI's detector orientation. Relative to native orientation 3,
      1 reverses rows and columns, 2 reverses rows and 4 reverses columns before rotation.
    * ``sample_roll_deg`` tilts the surface normal about the incident beam toward lab right
      for positive angles. In lab (beam, right, up), n = (−sin αi, cos αi sin ψ, cos αi cos ψ).
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
    orientation: int = 3
    sample_roll_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "orientation", orientation_number(self.orientation))
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
        object.__setattr__(self, "sample_roll_deg", _finite("sample_roll_deg", self.sample_roll_deg))
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
    def requires_projection(self) -> bool:
        """Whether the native, unrotated image-to-laboratory axes need transforming."""
        return self.tilted or self.orientation != 3

    @property
    def grazing_requires_projection(self) -> bool:
        """Whether the detector coordinates differ from an upright native grazing geometry."""
        return self.requires_projection or self.sample_roll_deg != 0.0

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
        return (lab(right, -1.0 if self.orientation in (1, 4) else 1.0),
                lab(down, 1.0 if self.orientation in (1, 2) else -1.0))

    @property
    def wavevector_inv_angstrom(self) -> float:
        """|k| = 2π/λ in Å⁻¹."""
        return 2.0 * math.pi / self.wavelength_angstrom

    @property
    def incidence_rad(self) -> float:
        return math.radians(self.incidence_deg)

    @property
    def sample_roll_rad(self) -> float:
        return math.radians(self.sample_roll_deg)

    def with_sample_roll(self, sample_roll_deg: float) -> "DetectorGeometry":
        return replace(self, sample_roll_deg=sample_roll_deg)

    def rolled_detector_axes(self):
        """Detector column and upward row axes in (beam, sample transverse, rolled up)."""
        return tuple((along, *roll_transverse(right, up, self.sample_roll_rad))
                     for along, right, up in self.in_plane_axes())

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
        for a tilted detector the ray on this column is intersected with the cone of constant exit angle.
        """
        angle = math.radians(float(alpha_f_deg)) + self.incidence_rad
        if not -math.pi / 2 < angle < math.pi / 2:
            raise ValueError("The requested exit angle does not reach the detector.")
        height_m = self.distance_m * math.tan(angle)
        if not self.tilted and not self.sample_roll_deg:
            sign = 1 if self.orientation in (1, 2) else -1
            return self.beam_center_y_px + sign * (height_m / self.pixel_size_y_m)
        # At the beam column the ray is D*(e_beam + t*row_up). Solve n·r = |r| sin α_f;
        # squaring also yields the opposite exit angle, checked on the unsquared ray below.
        target = math.radians(float(alpha_f_deg))
        if target == 0:
            row = self.horizon_row()
            if math.isfinite(row):
                return row
        _col, (ax, _, up) = self.rolled_detector_axes()
        si, ci = math.sin(self.incidence_rad), math.cos(self.incidence_rad)
        b = up * ci - ax * si
        s2 = math.sin(target) ** 2
        aa, bb, cc = b * b - s2, -2 * (si * b + s2 * ax), si * si - s2
        roots = []
        if abs(aa) > 1e-16:
            disc = bb * bb - 4 * aa * cc
            if disc >= 0:
                root = math.sqrt(disc)
                roots = [(-bb + root) / (2 * aa), (-bb - root) / (2 * aa)]
        elif bb != 0:
            roots = [-cc / bb]
        for t in sorted(roots, key=abs):
            row = self.beam_center_y_px - t * self.distance_m / self.pixel_size_y_m
            if abs(self.exit_angle_at(self.beam_center_x_px, row) - target) < 1e-9:
                return row
        raise ValueError("The exit angle does not cross the beam-centre column.")

    def exit_angle_at(self, x_px: float, y_px: float) -> float:
        """Exit angle alpha_f (radians) of the ray to a continuous detector position."""
        (ax, rx, ux), (ay, ry, uy) = self.in_plane_axes()
        x = (float(x_px) - self.beam_center_x_px) * self.pixel_size_x_m
        y = (self.beam_center_y_px - float(y_px)) * self.pixel_size_y_m
        along, up = self.distance_m + x * ax + y * ay, x * ux + y * uy
        right = x * rx + y * ry
        radius = math.sqrt(along * along + right * right + up * up)
        _, up = roll_transverse(right, up, self.sample_roll_rad)
        normal = (up * math.cos(self.incidence_rad) - along * math.sin(self.incidence_rad)) / radius
        return math.asin(max(-1.0, min(1.0, normal)))

    def horizon_plane(self) -> tuple[float, float, float]:
        """The horizon in pixels relative to the direct beam: A·dx + B·dy = C (the sample normal dotted with the ray)."""
        (ax, _, ux), (ay, _, uy) = self.rolled_detector_axes()
        si, ci = math.sin(self.incidence_rad), math.cos(self.incidence_rad)
        return ((ux * ci - ax * si) * self.pixel_size_x_m,
                (ay * si - uy * ci) * self.pixel_size_y_m, self.distance_m * si)

    def horizon_line(self) -> tuple[tuple[float, float], float] | None:
        """A point on α_f = 0 and the angle of its detector trace in (column, row) coordinates."""
        if not self.grazing_requires_projection:
            return (self.beam_center_x_px, self.horizon_row()), 0.0
        a, b, c = self.horizon_plane()
        norm2 = a * a + b * b
        if norm2 == 0.0:
            return None
        point = self.beam_center_x_px + c * a / norm2, self.beam_center_y_px + c * b / norm2
        return point, math.degrees(math.atan2(a, -b))

    def horizon_row(self, x_px: float | None = None) -> float:
        """Row of α_f = 0 at this column (default: direct beam); NaN for a vertical horizon."""
        if not self.tilted and not self.sample_roll_deg:
            return self.row_for_exit_angle(0.0)
        a, b, c = self.horizon_plane()
        if abs(b) <= 1e-14 * self.pixel_size_y_m:
            return math.nan
        dx = 0.0 if x_px is None else x_px - self.beam_center_x_px
        return self.beam_center_y_px + (c - a * dx) / b

    def specular_row(self) -> float:
        """Row of the specularly reflected beam (alpha_f = alpha_i)."""
        if not self.grazing_requires_projection:
            return self.row_for_exit_angle(self.incidence_deg)
        from .q_inverse import positions_of_rays
        import numpy as np

        across, up = roll_transverse(0.0, math.sin(2 * self.incidence_rad), -self.sample_roll_rad)
        ray = np.array([math.cos(2 * self.incidence_rad), across, up])
        return float(positions_of_rays(self, ray)[1])
