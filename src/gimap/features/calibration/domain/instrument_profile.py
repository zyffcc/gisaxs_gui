"""Turn a calibration result into a reusable instrument profile."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.conventions import (
    canonical_from_index_center,
    index_center_from_canonical,
)

from .manual_refinement import geometry_change_is_significant
from .models import CalibrationCandidate, CalibrationResult

ROTATION_NOTE_LIMIT_DEG = 0.05
"""Detector tilt the flat, perpendicular DetectorGeometry cannot represent."""


def calibrated_shape(result: CalibrationResult) -> Optional[tuple[int, int]]:
    shape = (result.metadata or {}).get("image_shape")
    try:
        rows, columns = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (rows, columns) if rows > 0 and columns > 0 else None


def calibrated_geometry(result: CalibrationResult, *, incidence_deg: float = 0.0) -> DetectorGeometry:
    """Canonical geometry of the selected candidate (numpy index centre → +0.5 px)."""
    candidate = result.selected_candidate
    center_x, center_y = canonical_from_index_center(candidate.center_x_px, candidate.center_y_px)
    return DetectorGeometry(
        pixel_size_x_m=float(result.pixel_size_x_m),
        pixel_size_y_m=float(result.pixel_size_y_m),
        distance_m=float(candidate.distance_mm) * 1e-3,
        beam_center_x_px=center_x,
        beam_center_y_px=center_y,
        wavelength_angstrom=float(result.wavelength_angstrom),
        incidence_deg=float(incidence_deg),
    )


def profile_name_for(result: CalibrationResult) -> str:
    """Default profile name: detector name and frame size, e.g. ``PILATUS 2M 1679×1475``."""
    name = " ".join(str(result.detector_name or "").split()) or "Detector"
    shape = calibrated_shape(result)
    return f"{name} {shape[0]}×{shape[1]}" if shape else name


def calibration_source(result: CalibrationResult) -> str:
    candidate = result.selected_candidate
    source = (
        f"calibration {candidate.standard_key} · {result.calibration_timestamp} · "
        f"{result.source_image}"
    )
    if abs(float(candidate.detector_rotation_deg)) > ROTATION_NOTE_LIMIT_DEG:
        source += f" · detector rotation {candidate.detector_rotation_deg:.2f}° not modelled"
    return source


def profile_from_calibration(
    result: CalibrationResult, existing: Optional[InstrumentProfile] = None
) -> InstrumentProfile:
    """New profile, or ``existing`` updated in place keeping its sample angle.

    The grazing angle is a property of the measurement, not of the transmission
    calibration, so an updated profile keeps the αi it already had.
    """
    source = calibration_source(result)
    if existing is not None:
        geometry = calibrated_geometry(result, incidence_deg=existing.geometry.incidence_deg)
        updated = existing.updated(geometry, source=source)
        shape = calibrated_shape(result)
        if shape is not None and updated.detector_shape is None:
            updated = replace(updated, detector_shape=shape)
        return updated
    return InstrumentProfile(
        name=profile_name_for(result),
        geometry=calibrated_geometry(result),
        detector_name=result.detector_name,
        detector_shape=calibrated_shape(result),
        source=source,
    )


def profile_change_is_significant(
    profile: InstrumentProfile, candidate: CalibrationCandidate
) -> bool:
    """Whether ``candidate`` moves the profile's beam centre > 10 px or its distance > 5 %."""
    geometry = profile.geometry
    center_x, center_y = index_center_from_canonical(
        geometry.beam_center_x_px, geometry.beam_center_y_px
    )
    return geometry_change_is_significant(
        {
            "distance": geometry.distance_m * 1e3,
            "beam_center_x": center_x,
            "beam_center_y": center_y,
        },
        candidate,
    )


__all__ = [
    "calibrated_geometry",
    "calibrated_shape",
    "profile_change_is_significant",
    "profile_from_calibration",
    "profile_name_for",
]
