"""Geometry calibration without the dialog, described as plain data.

Another component (the assistant) fits an image of a standard or reads a saved
calibration and gets the geometry and its quality as a dict; nothing is saved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from src.gimap.shared.geometry.poni import tilt_from_geometry

import numpy as np

from ..domain import (
    STANDARDS,
    CalibrationRequest,
    CalibrationResult,
    best_by_lines,
    calibrated_geometry,
    calibrated_shape,
    check_lines,
    standard_display_name,
    standard_options,
)
from .use_cases import ImportCalibration, LoadCalibrationImage, RunCalibration


def _describe_candidate(result: CalibrationResult) -> dict:
    candidate = result.selected_candidate
    geometry = calibrated_geometry(result)
    return {
        "standard": candidate.standard_key,
        "standard_name": standard_display_name(candidate.standard_key),
        "distance_mm": float(candidate.distance_mm),
        "beam_center_px": [float(geometry.beam_center_x_px), float(geometry.beam_center_y_px)],
        "initial_point_rms_px": candidate.initial_point_rms_px,
        "point_rms_px": candidate.point_rms_px,
        "untilted_point_rms_px": candidate.untilted_point_rms_px,
        "matched_rings": int(candidate.matched_ring_count),
        "rms_residual_px": float(candidate.rms_residual_px),
        "uncertainty": {"beam_center_px": [candidate.center_x_sigma_px, candidate.center_y_sigma_px],
                        "distance_mm": candidate.distance_sigma_mm, "method": candidate.uncertainty_method or None},
        "confidence": candidate.confidence,
        "score": float(candidate.score),
        "warnings": list(candidate.warnings),
        "rotation_deg": float(candidate.detector_rotation_deg),
        "rotations_deg": [float(candidate.rot1_deg), float(candidate.rot2_deg), float(candidate.rot3_deg)],
        "tilt_refined": bool(candidate.tilt_refined),
        "tilt_uncertainty_deg": [candidate.rot1_sigma_deg, candidate.rot2_sigma_deg],
        "detector_tilt": tilt_from_geometry(geometry).to_dict() if geometry.tilted else None,
    }


def describe_calibration(result: CalibrationResult) -> dict:
    """The selected geometry (canonical beam centre: first pixel's centre at 0.5) and its quality."""
    alternatives = [_describe_candidate(replace(result, selected_candidate=c)) for c in result.candidates
                    if c != result.selected_candidate]
    shape = calibrated_shape(result)
    return {
        **_describe_candidate(result),
        "energy_kev": float(result.energy_kev),
        "wavelength_angstrom": float(result.wavelength_angstrom),
        "pixel_size_um": [float(result.pixel_size_x_m) * 1e6, float(result.pixel_size_y_m) * 1e6],
        "detector": result.detector_name,
        "shape": list(shape) if shape else None,
        "energy_source": (result.metadata or {}).get("energy_source"),
        "source_image": result.source_image,
        "calibrated_at": result.calibration_timestamp,
        "alternatives": alternatives[:3],
    }


@dataclass(frozen=True)
class HeadlessCalibration:
    load_image: LoadCalibrationImage
    run_calibration: RunCalibration
    import_calibration: ImportCalibration

    def standards(self) -> dict[str, str]:
        return {standard.key: standard.display_name for standard in standard_options()}

    def calibrate(
        self,
        path: str,
        *,
        standard: str,
        energy_kev: float,
        distance_mm: Optional[float] = None,
        pixel_size_m: Optional[float] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        tilt: str = "auto",
    ) -> dict:
        image = self.load_image(Path(path))
        request = CalibrationRequest(
            image=image,
            energy_kev=float(energy_kev),
            standard_key=standard,
            tilt=tilt,
            estimated_distance_mm=distance_mm,
            pixel_size_x_m=pixel_size_m,
            pixel_size_y_m=pixel_size_m,
        )
        summary = describe_calibration(self.run_calibration(request, None, cancelled))
        return with_line_checks(summary, image)

    def read_result(self, path: str) -> dict:
        # Checked on the image it was made on: all the frames of a sum, not the first one alone.
        imported = self.import_calibration(Path(path), summed=True)
        summary = describe_calibration(imported.result)
        return with_line_checks(summary, imported.image) if imported.image is not None else summary


def with_line_checks(summary: dict, image) -> dict:
    """Check the fit and its alternatives against the standard's lines; keep the most accurate one.

    The engine ranks by ring matching in pixels; the geometry that puts the
    standard's lines closest to their q is the better calibration, so it becomes
    the result when it differs (``chosen_by`` says so).
    """
    data = np.asarray(image.data, dtype=np.float64)
    valid = np.isfinite(data) & (data >= 0)
    if image.mask is not None:
        valid &= ~np.asarray(image.mask, dtype=bool)
    candidates = [summary, *summary["alternatives"]]
    checks = []
    for candidate in candidates:
        standard = STANDARDS.get(candidate["standard"])
        if standard is None:
            checks.append(None)
            continue
        checks.append(check_lines(
            data, valid,
            center_x_px=candidate["beam_center_px"][0], center_y_px=candidate["beam_center_px"][1],
            distance_mm=candidate["distance_mm"],
            pixel_size_x_m=summary["pixel_size_um"][0] * 1e-6, pixel_size_y_m=summary["pixel_size_um"][1] * 1e-6,
            wavelength_angstrom=summary["wavelength_angstrom"], q_lines=standard.q_values_inv_angstrom,
            rotations_deg=candidate.get("rotations_deg") or (0.0, 0.0, 0.0),
        ))
    for candidate, check in zip(candidates, checks):
        candidate["line_check"] = check
    best = best_by_lines(checks)
    if best is None or best == 0:
        summary["chosen_by"] = "ring fit" if best is None else "ring fit, confirmed by the line positions"
        return summary
    winner = candidates[best]
    # RMS and uncertainty belong to the same solution as the reported centre.
    summary["alternatives"] = [{key: candidate[key] for key in winner}
                               for index, candidate in enumerate(candidates) if index != best]
    summary.update(winner)
    summary["chosen_by"] = "line positions (the engine's first choice put the standard's lines further from their q)"
    return summary


__all__ = ["HeadlessCalibration", "describe_calibration", "with_line_checks"]
