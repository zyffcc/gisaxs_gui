"""The Analyze set-up as a file: everything needed to process new raw data the same way.

A settings file is JSON (``{"format": "gimap-analyze-settings", "version": 1, …}``) with the
mode, the instrument profile (its name and its geometry, so the file also works on another
computer), αi, a beam-centre override, frame summing, the corrections (gap guard, hot/dead
pixels, masks drawn or loaded, mirror filling, valid range, background), the GIWAXS cuts (band
widths, bins, x axis, ring window, custom sector, q box, cut regions), the GISAXS cut positions
and, optionally, the batch-export choices. Paths are stored as they are; a mask or background
file that no longer exists is reported when the file is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from src.gimap.shared.geometry import InstrumentProfile

from ..domain import (
    Corrections,
    GisaxsCutSettings,
    GiwaxsSettings,
    MaskShape,
    QBox,
    Sector,
    X_AXIS_Q,
    X_AXIS_TWO_THETA,
)
from .cut_regions import region_to_dict, regions_from_dicts
from .models import AUTO, MODES
from .use_cases import SOFTWARE

SETTINGS_FORMAT = "gimap-analyze-settings"
SETTINGS_VERSION = 1


@dataclass(frozen=True)
class AnalyzeSettings:
    mode: str = AUTO
    profile_name: Optional[str] = None
    profile: Optional[InstrumentProfile] = None
    incidence_deg: Optional[float] = None
    beam_center: Optional[tuple[float, float]] = None
    sum_count: int = 1
    corrections: Corrections = field(default_factory=Corrections)
    giwaxs: GiwaxsSettings = field(default_factory=GiwaxsSettings)
    gisaxs: GisaxsCutSettings = field(default_factory=GisaxsCutSettings)
    export: Optional[dict] = None
    """Batch-export choices (``BatchChoices.to_dict``), when saved with them."""


def _pair(value) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    first, second = value
    return float(first), float(second)


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


def settings_record(settings: AnalyzeSettings) -> dict[str, Any]:
    corrections, giwaxs, gisaxs = settings.corrections, settings.giwaxs, settings.gisaxs
    sector, box = giwaxs.sector, giwaxs.box
    record: dict[str, Any] = {
        "format": SETTINGS_FORMAT,
        "version": SETTINGS_VERSION,
        "software": SOFTWARE,
        "mode": settings.mode,
        "profile_name": settings.profile_name,
        "profile": settings.profile.to_dict() if settings.profile is not None else None,
        "incidence_deg": settings.incidence_deg,
        "beam_center_px": list(settings.beam_center) if settings.beam_center is not None else None,
        "sum_frames": int(settings.sum_count),
        "corrections": {
            "gap_guard_px": int(corrections.gap_guard_px),
            "bad_pixels": bool(corrections.bad_pixels),
            "mirror_fill": bool(corrections.mirror_fill),
            "minimum": corrections.minimum,
            "maximum": corrections.maximum,
            "background_path": corrections.background_path,
            "background_frame": int(corrections.background_frame),
            "background_scale": float(corrections.background_scale),
            "mask_path": corrections.mask_path,
            "mask_shapes": [
                {"kind": shape.kind, "points": [list(point) for point in shape.points]} for shape in corrections.mask_shapes
            ],
            "solid_angle": bool(corrections.solid_angle),
            "polarization": corrections.polarization,
            "film_thickness_nm": corrections.film_thickness_nm,
            "attenuation_length_um": corrections.attenuation_length_um,
        },
        "giwaxs": {
            "in_plane_half_width_deg": giwaxs.in_plane_half_width_deg,
            "out_of_plane_half_width_deg": giwaxs.out_of_plane_half_width_deg,
            "chi_q_window": list(giwaxs.chi_q_window) if giwaxs.chi_q_window is not None else None,
            "bins": giwaxs.bins,
            "x_axis": giwaxs.x_axis,
            "sector": None if sector is None else {
                "chi_min_deg": sector.chi_min_deg, "chi_max_deg": sector.chi_max_deg,
                "q_min": sector.q_min, "q_max": sector.q_max,
            },
            "box": None if box is None else {"q_parallel": list(box.q_parallel), "qz": list(box.qz)},
            "regions": [region_to_dict(region) for region in giwaxs.regions],
        },
        "gisaxs": {
            "horizontal_row": gisaxs.horizontal_row,
            "horizontal_half_height_px": gisaxs.horizontal_half_height_px,
            "vertical_column": gisaxs.vertical_column,
            "vertical_half_width_px": gisaxs.vertical_half_width_px,
        },
    }
    if settings.export is not None:
        record["export"] = dict(settings.export)
    return record


def settings_from_record(record: Any) -> AnalyzeSettings:
    """``AnalyzeSettings`` from a settings file (``ValueError`` says what is wrong)."""
    if not isinstance(record, Mapping) or record.get("format") != SETTINGS_FORMAT:
        raise ValueError("This is not a GIMaP Analyze settings file.")
    try:
        mode = record.get("mode") if record.get("mode") in MODES else AUTO
        profile = InstrumentProfile.from_dict(record["profile"]) if record.get("profile") else None
        values = record.get("corrections") or {}
        corrections = Corrections(
            background_path=values.get("background_path") or None,
            background_frame=int(values.get("background_frame", 0)),
            background_scale=float(values.get("background_scale", 1.0)),
            minimum=_optional_float(values.get("minimum")),
            maximum=_optional_float(values.get("maximum")),
            gap_guard_px=int(values.get("gap_guard_px", 0)),
            mask_path=values.get("mask_path") or None,
            mask_shapes=tuple(
                MaskShape(str(item["kind"]), tuple(tuple(point) for point in item["points"]))
                for item in values.get("mask_shapes") or ()
            ),
            mirror_fill=bool(values.get("mirror_fill", False)),
            bad_pixels=bool(values.get("bad_pixels", True)),
            solid_angle=bool(values.get("solid_angle", False)),
            polarization=_optional_float(values.get("polarization")),
            film_thickness_nm=_optional_float(values.get("film_thickness_nm")),
            attenuation_length_um=_optional_float(values.get("attenuation_length_um")),
        )
        values = record.get("giwaxs") or {}
        sector = values.get("sector")
        box = values.get("box")
        x_axis = values.get("x_axis") if values.get("x_axis") in (X_AXIS_Q, X_AXIS_TWO_THETA) else X_AXIS_Q
        giwaxs = GiwaxsSettings(
            in_plane_half_width_deg=float(values.get("in_plane_half_width_deg", 10.0)),
            out_of_plane_half_width_deg=float(values.get("out_of_plane_half_width_deg", 10.0)),
            chi_q_window=_pair(values.get("chi_q_window")),
            bins=None if values.get("bins") in (None, 0) else int(values["bins"]),
            x_axis=x_axis,
            sector=None if not sector else Sector(
                float(sector["chi_min_deg"]), float(sector["chi_max_deg"]),
                _optional_float(sector.get("q_min")), _optional_float(sector.get("q_max")),
            ),
            box=None if not box else QBox(_pair(box["q_parallel"]), _pair(box["qz"])),
            regions=tuple(regions_from_dicts(values.get("regions") or ())),
        )
        values = record.get("gisaxs") or {}
        gisaxs = GisaxsCutSettings(
            horizontal_row=_optional_float(values.get("horizontal_row")),
            horizontal_half_height_px=float(values.get("horizontal_half_height_px", 2.5)),
            vertical_column=_optional_float(values.get("vertical_column")),
            vertical_half_width_px=float(values.get("vertical_half_width_px", 5.0)),
        )
        export = record.get("export")
        return AnalyzeSettings(
            mode=mode,
            profile_name=record.get("profile_name") or (profile.name if profile is not None else None),
            profile=profile,
            incidence_deg=_optional_float(record.get("incidence_deg")),
            beam_center=_pair(record.get("beam_center_px")),
            sum_count=max(1, int(record.get("sum_frames", 1))),
            corrections=corrections,
            giwaxs=giwaxs,
            gisaxs=gisaxs,
            export=dict(export) if isinstance(export, Mapping) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"The settings file is not valid: {exc}") from exc


__all__ = ["AnalyzeSettings", "SETTINGS_FORMAT", "settings_from_record", "settings_record"]
