"""Choose a grazing angle and keep its frame, calibration and independent observations together."""

from __future__ import annotations

import math

from ..domain.mask_bitmap import bitmap_record
from .detector_response import with_profile_response
from .record_values import plain

_CALIBRATION = ("pixel_size_x_m", "pixel_size_y_m", "distance_m", "beam_center_x_px", "beam_center_y_px",
                "wavelength_angstrom", "rot1_rad", "rot2_rad", "rot3_rad", "orientation")
# Corrections affecting the observed pixels used to find an angle. Exit-angle cuts,
# film absorption, Lorentz and mirror filling do not enter the measurement.
_PIXEL_INPUTS = ("dark_path", "dark_frame", "dark_scale", "background_path", "background_frame", "background_scale",
                 "flat_field_path", "flat_field_floor", "minimum", "maximum", "gap_guard_px", "mask_path",
                 "mask_shapes", "bad_pixels", "solid_angle", "polarization", "background_dark_scale",
                 "background_already_dark_subtracted", "flat_already_dark_subtracted", "counts_per_adu", "offset_adu")


def incidence_choice(reference_deg, measurement_dict, *, locked=False):
    """Return the frame-derived override, if any, and why it was selected or withheld.

    Published or typed values normally carry no uncertainty. Compare against three
    reported standard errors of the frame measurement, conditional on its calibration.
    """
    if locked:
        return None, "user_override"
    measured, error = measurement_dict.get("deg"), measurement_dict.get("error_deg")
    if measured is None or error is None or not (0 < measured < 90 and math.isfinite(error) and error > 0):
        return None, "unresolved"
    if reference_deg is None or reference_deg <= 0:
        return measured, "measured"
    if abs(measured - reference_deg) > 3 * error:
        return measured, "disagreement"
    return None, "agreement"


def _calibration(geometry):
    return {name: getattr(geometry, name) for name in _CALIBRATION}


def nominal_incidence(analysis):
    return _nominal(getattr(analysis, "request", None), analysis.resolution.profile, analysis.metadata)


def _nominal(request, profile, header):
    if request is not None and request.incidence_deg is not None:
        return request.incidence_deg, "entered"
    header = header or {}
    if header.get("incidence_deg") is not None:
        return header["incidence_deg"], header.get("incidence_source") or "the image header"
    if profile and profile.geometry.incidence_deg > 0:
        return profile.geometry.incidence_deg, "profile"
    return None, "missing"


def frame_incidence_record(analysis, result_dict, *, origin, reference_deg, reference_source, applied_deg, decision):
    angle_references = {key: ref for key, ref in analysis.correction_files.items()
                        if key != analysis.corrections.film_cif_path}
    return plain(dict(result_dict, origin=origin, path=str(analysis.path), frame_index=analysis.frame_index,
                      summed_frames=analysis.summed_frames, calibration=_calibration(analysis.geometry),
                      technique=analysis.kind, nominal_at_measurement_deg=nominal_incidence(analysis)[0],
                      frame_files=analysis.frame_files, calibration_files=analysis.geometry_files,
                      correction_files=angle_references, corrections=analysis.corrections,
                      mask=bitmap_record(analysis.drawn_mask, analysis.shape) if analysis.drawn_mask is not None else None,
                      reference_deg=reference_deg,
                      reference_source=reference_source, applied_deg=applied_deg, decision=decision))


def same_measurement_frame(record, request):
    return (record.get("path"), record.get("frame_index"), record.get("summed_frames", [])) == (
        str(request.path), request.frame_index, [[str(path), index] for path, index in request.summed_frames])


def selected_incidence_record(request, resolution, metadata, frame_files, correction_files=None):
    """Evaluate a saved observation against the selected frame, optics, and current nominal value."""
    record = request.incidence_measurement if request else None
    geometry = resolution.geometry
    if not record or geometry is None or record.get("calibration") != _calibration(geometry):
        return None
    if record.get("frame_files") and frame_files != record["frame_files"]:
        return None
    if record.get("correction_files") and correction_files != record["correction_files"]:
        return None
    previous_corrections = record.get("corrections")
    if previous_corrections:
        current = with_profile_response(request.corrections, resolution.profile)
        if any(plain(getattr(current, key)) != previous_corrections.get(key) for key in _PIXEL_INPUTS):
            return None
    if record.get("technique") and request.mode not in ("auto", record["technique"]):
        return None
    if record.get("roll_deg") is not None and not math.isclose(geometry.sample_roll_deg, record["roll_deg"], abs_tol=1e-8):
        return None
    if record.get("technique") == "gisaxs" and record.get("critical_angle_source") != "run option":
        if record.get("yoneda_material") != request.yoneda_material:
            return None
        if record.get("yoneda") and not request.yoneda_material:
            critical = request.yoneda_critical_deg or request.corrections.critical_angle_deg or geometry.critical_angle_deg
            if critical != record.get("critical_angle_deg"):
                return None
    reference, source = _nominal(request, resolution.profile, metadata)
    previous = record.get("override_deg", record.get("nominal_at_measurement_deg", record["reference_deg"]))
    if (reference or 0) != (previous or 0):
        applied, decision = incidence_choice(reference, record, locked=request.incidence_locked)
        return dict(record, override_deg=reference, override_source=source, applied_deg=applied, decision=decision)
    if request.incidence_locked and record["decision"] != "user_override":
        return dict(record, decision="user_override", applied_deg=None)
    return record


def checked_incidence_record(record):
    """Reject incomplete estimates before restoring a scientific setting from disk."""
    if record is None:
        return None
    message = "Invalid incidence measurement."
    if not isinstance(record, dict) or not isinstance(record.get("method"), str):
        raise ValueError(message)
    if (not isinstance(record.get("calibration"), dict) or not isinstance(record.get("path"), str)
            or record.get("decision") not in ("user_override", "unresolved", "measured", "disagreement", "agreement")
            or "reference_deg" not in record):
        raise ValueError(message)
    try:
        for key in ("roll_deg", "reference_deg", "nominal_at_measurement_deg", "override_deg", "critical_angle_deg"):
            if record.get(key) is not None and not math.isfinite(record[key]):
                raise ValueError(message)
        for key in (None, "specular", "horizon", "yoneda"):
            estimate = record if key is None else record.get(key)
            if estimate is None or (key is None and estimate.get("deg") is None and estimate.get("error_deg") is None):
                continue
            if not (math.isfinite(estimate["deg"]) and math.isfinite(estimate["error_deg"]) and estimate["error_deg"] > 0):
                raise ValueError(message)
        applied = record.get("applied_deg")
        if applied is not None and not (0 < applied < 90 and applied == record.get("deg")):
            raise ValueError(message)
    except (TypeError, KeyError, OverflowError):
        raise ValueError(message) from None
    return dict(record)
