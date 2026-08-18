"""Pure geometry used to present and manually refine calibration rings."""

from __future__ import annotations

from dataclasses import dataclass

from .models import CalibrationCandidate, CalibrationResult
from .scientific_kernel import STANDARDS, distance_from_ring_radius, q_to_ring_radius_m


@dataclass(frozen=True)
class TheoreticalRingOverlay:
    width_px: float
    height_px: float
    matched: bool


def theoretical_ring_overlays(
    candidate: CalibrationCandidate,
    result: CalibrationResult,
) -> tuple[TheoreticalRingOverlay, ...]:
    standard = STANDARDS.get(candidate.standard_key)
    if standard is None:
        return ()
    radii_m = q_to_ring_radius_m(
        standard.q_values_inv_angstrom,
        result.wavelength_angstrom,
        candidate.distance_mm,
    )
    matched_indices = {match.theoretical_index for match in candidate.matched_rings}
    return tuple(
        TheoreticalRingOverlay(
            width_px=float(2.0 * radius_m / result.pixel_size_x_m),
            height_px=float(2.0 * radius_m / result.pixel_size_y_m),
            matched=index in matched_indices,
        )
        for index, radius_m in enumerate(radii_m)
    )


def manual_ring_distance(
    result: CalibrationResult,
    experimental_radius_px: float,
    theoretical_q_inv_angstrom: float,
) -> float:
    return distance_from_ring_radius(
        experimental_radius_px,
        theoretical_q_inv_angstrom,
        result.wavelength_angstrom,
        0.5 * (result.pixel_size_x_m + result.pixel_size_y_m),
    )
