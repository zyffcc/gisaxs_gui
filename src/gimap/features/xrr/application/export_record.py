"""The JSON record written next to an exported XRR curve, and the default export name.

The record holds every setting that determines qz and the ROI position (series and θ, energy,
distance, pixel size, beam centre, direction), the ROI radius and aggregation, where each geometry
value came from, and the formulas. Values are the ones the extraction used, in the units named by
their keys.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..domain import HC_KEV_ANGSTROM
from .models import ExportXrrCurveRequest

RECORD_FORMAT = "gimap-xrr-curve-record"
RECORD_VERSION = 1


def record_path_for(csv_path: str | Path) -> Path:
    """``<name>.json`` next to ``<name>.csv`` (never the CSV itself)."""
    path = Path(csv_path)
    record = path.with_suffix(".json")
    return path.with_suffix(".record.json") if record == path else record


def default_export_path(source_path: str | Path) -> Path:
    """``<source folder>/<source stem>_xrr.csv``; for a folder of CBF frames, inside that folder."""
    source = Path(source_path)
    if source.is_dir():
        return source / f"{source.name}_xrr.csv"
    return source.with_name(f"{source.stem}_xrr.csv")


def xrr_export_record(
    request: ExportXrrCurveRequest,
    csv_path: str | Path,
    *,
    created: str | None = None,
) -> dict[str, Any]:
    points = request.result.points
    record: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "format_version": RECORD_VERSION,
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": Path(csv_path).name,
        "points": len(points),
        "columns": {
            "index": "position in the series (0-based)",
            "source": "file the frame was read from",
            "frame": "frame within that file (0-based)",
            "theta_deg": "sample angle θ (°)",
            "qz_inv_angstrom": "specular qz (Å⁻¹)",
            "intensity": "ROI intensity (detector counts; empty: no valid pixel)",
            "roi_center_x_px": "ROI centre, detector column (px)",
            "roi_center_y_px": "ROI centre, detector row (px)",
            "valid_pixels": "pixels summed or averaged",
        },
    }
    if points:
        record["theta_deg"] = {"first": points[0].theta_deg, "last": points[-1].theta_deg}
    settings = request.settings
    if settings is not None:
        series, geometry, extraction = settings.series, settings.geometry, settings.extraction
        linear = series.angle_mode == "linear"
        record["series"] = {
            "source_path": str(series.source_path),
            "source_kind": series.source_kind,
            "cbf_pattern": series.pattern,
            "angle_mode": series.angle_mode,
            "theta_start_deg": series.theta_start_deg if linear else None,
            "theta_step_deg": series.theta_step_deg if linear else None,
            "angle_dataset_path": "" if linear else series.angle_dataset_path,
        }
        record["geometry"] = {
            "distance_m": geometry.distance_m,
            "energy_kev": geometry.energy_kev,
            "wavelength_angstrom": geometry.wavelength_angstrom,
            "pixel_size_x_m": geometry.pixel_size_x_m,
            "pixel_size_y_m": geometry.pixel_size_y_m,
            "beam_center_x_px": geometry.beam_center_x_px,
            "beam_center_y_px": geometry.beam_center_y_px,
            "vertical_direction": geometry.vertical_direction,
            "pixel_convention": "numpy indices of the loaded frame: x = column, y = row, row 0 at the top",
            "sources": dict(request.geometry_sources),
        }
        calibration = request.calibration
        if calibration is not None and "calibration" in request.geometry_sources.values():
            # The applied Geometry Calibration the values marked "calibration" were taken from.
            record["geometry"]["last_calibration"] = {
                "source_image": calibration.source_image,
                "timestamp": calibration.timestamp,
                "detector": calibration.detector,
                "image_shape": list(calibration.image_shape) if calibration.image_shape else None,
            }
        record["extraction"] = {
            "roi_radius_px": extraction.radius_px,
            "aggregation": extraction.aggregation,
        }
    record["formulas"] = {
        "qz": f"qz = 4·π·sin(θ)/λ, λ = hc/E, hc = {HC_KEV_ANGSTROM} keV·Å",
        "roi_center": (
            "x = beam_center_x_px; y = beam_center_y_px + vertical_direction · distance_m · "
            "tan(2θ) / pixel_size_y_m (the detector is fixed, the sample turns by θ)"
        ),
        "intensity": (
            "sum or mean of the finite, unmasked pixels within roi_radius_px of the ROI centre "
            "(radius 0: the nearest pixel)"
        ),
    }
    return record


__all__ = ["default_export_path", "record_path_for", "xrr_export_record"]
