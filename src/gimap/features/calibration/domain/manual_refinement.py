"""Pure/manual calibration candidate refinement rules."""

from __future__ import annotations

import copy

from .models import CalibrationCandidate, CalibrationResult


MANUAL_REFINEMENT_WARNING = (
    "Geometry was manually adjusted after automatic calibration."
)


def preview_manual_candidate(
    candidate: CalibrationCandidate,
    *,
    enabled: bool,
    center_x_px: float,
    center_y_px: float,
    distance_mm: float,
) -> CalibrationCandidate:
    preview = copy.deepcopy(candidate)
    if enabled:
        preview.center_x_px = center_x_px
        preview.center_y_px = center_y_px
        preview.distance_mm = distance_mm
    return preview


def commit_manual_refinement(
    result: CalibrationResult,
    *,
    enabled: bool,
    center_x_px: float,
    center_y_px: float,
    distance_mm: float,
) -> CalibrationCandidate:
    candidate = result.selected_candidate
    if enabled:
        candidate.center_x_px = center_x_px
        candidate.center_y_px = center_y_px
        candidate.distance_mm = distance_mm
        if MANUAL_REFINEMENT_WARNING not in candidate.warnings:
            candidate.warnings.append(MANUAL_REFINEMENT_WARNING)
    return candidate


def select_calibration_candidate(
    result: CalibrationResult,
    index: int,
) -> CalibrationCandidate:
    candidate = result.candidates[index]
    result.selected_candidate = candidate
    return candidate


def geometry_change_is_significant(
    current_geometry: dict[str, float],
    candidate: CalibrationCandidate,
) -> bool:
    current_distance = current_geometry["distance"]
    current_x = current_geometry["beam_center_x"]
    current_y = current_geometry["beam_center_y"]
    return (
        abs(current_distance - candidate.distance_mm)
        / max(abs(current_distance), 1.0)
        > 0.05
        or abs(current_x - candidate.center_x_px) > 10.0
        or abs(current_y - candidate.center_y_px) > 10.0
    )
