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
    """A flat detector perpendicular to the direct beam, plus the grazing angle.

    Conventions (the same everywhere in GIMaP from now on):

    * Pixel ``(row i, column j)`` of the analysis array covers the continuous
      detector area ``[j, j + 1] x [i, i + 1]``; its centre is ``(j + 0.5, i + 0.5)``.
      Row 0 is the top of the displayed image, x grows to the right and y grows
      downward.  This is the pixel-corner convention used by pyFAI and by an
      ``imshow(..., extent=(0, width, height, 0))`` display.
    * ``beam_center_x_px`` / ``beam_center_y_px`` locate the *direct* (transmitted)
      beam in those coordinates, i.e. pyFAI's PONI expressed in pixels.
    * ``incidence_deg`` is the grazing angle alpha_i between beam and sample surface;
      0 means transmission geometry.
    * Lengths are metres, the wavelength is in Å, so q comes out in Å⁻¹.
    """

    pixel_size_x_m: float
    pixel_size_y_m: float
    distance_m: float
    beam_center_x_px: float
    beam_center_y_px: float
    wavelength_angstrom: float
    incidence_deg: float = 0.0

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

        Evaluated in the plane of incidence (the beam-centre column), where the
        exact flat-detector relation is ``Y = D·tan(alpha_f + alpha_i)`` above
        the direct beam.
        """
        angle = math.radians(float(alpha_f_deg)) + self.incidence_rad
        if not -math.pi / 2 < angle < math.pi / 2:
            raise ValueError("The requested exit angle does not reach the detector.")
        height_m = self.distance_m * math.tan(angle)
        return self.beam_center_y_px - height_m / self.pixel_size_y_m

    def horizon_row(self) -> float:
        """Row of the sample horizon (alpha_f = 0)."""
        return self.row_for_exit_angle(0.0)

    def specular_row(self) -> float:
        """Row of the specularly reflected beam (alpha_f = alpha_i)."""
        return self.row_for_exit_angle(self.incidence_deg)
