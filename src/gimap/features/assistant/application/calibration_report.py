"""Calibration diagnostics shared by assistant tools and reports."""

import math

from src.gimap.shared.numbers import format_number

from ..domain import assessment


def calibration_rms(result, format_text=str.format):
    """Describe the automatic fit on its ring points, with a radius fallback for older records."""
    final, initial = result.get("point_rms_px"), result.get("initial_point_rms_px")
    if final is not None:
        if initial is not None:
            return format_text("Ring-point RMS {before} px before refinement, {after} px after",
                               before=format_number(initial), after=format_number(final))
        return format_text("Ring-point RMS {rms} px", rms=format_number(final))
    radius = result.get("rms_residual_px")
    if radius is not None:
        return format_text("Ring-radius RMS {rms} px", rms=format_number(radius))
    return ""


def geometry_estimates(geometry, calibration):
    """Final distance and beam centre. A refined centre may differ from the calibration."""
    record = calibration or {}
    errors = record.get("uncertainty") or {}

    def shown(value, original, sigma, *, pixel=False):
        if sigma is not None and value is not None and original is not None and math.isclose(value, original, rel_tol=1e-12):
            return format_number(value, sigma)
        return f"{value:.2f}" if pixel and value is not None else format_number(value)

    distance = shown(geometry.get("distance_mm"), record.get("distance_mm"), errors.get("distance_mm"))
    centre = geometry.get("beam_center_px") or (None, None)
    source = record.get("beam_center_px") or (None, None)
    sigmas = errors.get("beam_center_px") or (None, None)
    return distance, [shown(value, original, sigma, pixel=True) for value, original, sigma in zip(centre, source, sigmas)]


def calibration_quality(result):
    """The assessment, standard, line check and fit residuals of the calibration used by a run."""
    if not result:
        return None
    if result.get("from_file"):
        return {"assessment": "a saved calibration file (not re-checked against a standard image)", "warnings": []}
    check = result.get("line_check") or {}
    return {
        "assessment": assessment(result),
        "standard": result.get("standard"),
        "lines_checked": check.get("lines_checked"),
        "mean_line_q_error_percent": None if check.get("mean_relative") is None else round(100 * check["mean_relative"], 3),
        "matched_rings": result.get("matched_rings"),
        "initial_point_rms_px": result.get("initial_point_rms_px"),
        "point_rms_px": result.get("point_rms_px"),
        "rms_residual_px": result.get("rms_residual_px"),
        "warnings": list(result.get("warnings") or ()),
    }


def fitted_rings(result):
    rings = f"{result.get('matched_rings')} rings"
    residual = calibration_rms(result)
    return f"{rings}, {residual}" if residual else rings
