"""What identifies a report's inputs and settings, for every way a report is written.

The command line (``app/headless_record.py``) and the window's Save Report (``presentation/guided_analysis.py``)
attach the same keys to a report, from the record Analyze writes beside its curve exports
(``<stem>_analysis.json``: ``analysis_record``):

- ``software_version``, ``created`` (UTC), ``options`` (the command line's; ``None`` in the window),
- ``inputs``: every file the result was made from (the frame's, calibration, mask, background, flat field), each by
  name, path, size and SHA-256,
- ``mask``: which mask was applied or that none was, and the bad pixels left out apart from it,
- ``analysis_record``: geometry, instrument profile, corrections, error model, binning.

``record_lines`` and ``mask_lines`` are the two sections of the Markdown report that say so in words (Record,
Mask), so a report saved as a web page or as Markdown names the release, its input by SHA-256, where the
calibration comes from (the file and the instrument profile), the mask and the corrections, as report.json does.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.gimap.shared.file_digest import file_record
from src.gimap.shared.version import SOFTWARE

MASK_IMAGE = "mask image: its non-zero pixels are left out"
MASK_SHAPES = "GIMaP masks file: {count} drawn shape(s) left out"
FILES_LISTED = 6
"""A report lists at most this many frame files by name and SHA-256; report.json has every one."""


def utc_now() -> str:
    """The time a record is written: UTC, ISO 8601, to the second (as ``<stem>_analysis.json``)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def profile_source(actor: str, origin: str, path: Optional[str] = None) -> str:
    """The ``source`` of an instrument profile a tool saves: who chose the geometry and from what, with the full
    path of the calibration file or standard image when there is one (``origin`` alone is only its name)."""
    text = f"{actor}: {origin}"
    return f"{text} [{path}]" if path and str(path) not in text else text


def file_geometry(path: str, info: dict, pixel_um: Optional[float]) -> dict:
    """The inspected file in the calibration record format used throughout a batch."""
    if info.get("kind") == "GIMaP calibration":
        return {key: value for key, value in info.items() if key != "kind"}
    return {
        "distance_mm": info["distance_mm"],
        "beam_center_px": [info["beam_center_x_px"], info["beam_center_y_px"]],
        "wavelength_angstrom": info.get("wavelength_angstrom"),
        "energy_kev": info.get("energy_kev"),
        "pixel_size_um": [pixel_um, pixel_um] if pixel_um else None,
        "source_image": path,
        "shape": None,
        "from_file": True,
        "detector_tilt": info.get("detector_tilt"),
        "sample_roll_deg": info.get("sample_roll_deg", 0.0),
        "sample_roll_measurement": info.get("sample_roll_measurement"),
    }


def exact_calibration(calibration: Optional[dict], used: Optional[dict]) -> Optional[dict]:
    """``calibration`` (a saved file as ``inspect_file`` answered: four significant digits) with the distance, centre,
    wavelength and tilt as they were applied (``used``: ``RunResults.geometry_used``), when both are the same
    geometry. The record then holds the numbers q was computed with, as the CSV headers do."""
    if not calibration or not used or not calibration.get("from_file"):
        return calibration
    exact = dict(calibration)
    centre = [used.get("beam_center_x_px"), used.get("beam_center_y_px")]
    for key, value in (("distance_mm", used.get("distance_mm")), ("wavelength_angstrom", used.get("wavelength_angstrom"))):
        old = calibration.get(key)
        if value is not None and old and abs(float(value) - float(old)) <= 1e-3 * abs(float(old)):
            exact[key] = value
    old = calibration.get("beam_center_px") or [None, None]
    if None not in centre and None not in old and all(abs(float(a) - float(b)) <= 0.05 for a, b in zip(centre, old)):
        exact["beam_center_px"] = centre
    if calibration.get("detector_tilt") and used.get("detector_tilt"):
        exact["detector_tilt"] = dict(used["detector_tilt"])
    if "sample_roll_deg" in used:
        exact["sample_roll_deg"] = used["sample_roll_deg"]
        exact["sample_roll_measurement"] = used.get("sample_roll_measurement")
    return exact


def _existing(path: Any) -> Optional[dict]:
    return file_record(path) if path and os.path.isfile(str(path)) else None


def mask_record(record: Optional[dict], mask_path: Optional[str] = None) -> dict:
    """Which mask was applied, or that none was; apart from it, the bad pixels found automatically.
    ``mask_path``: the masks file the session applied (its shapes are in the record, its path is not)."""
    record = record or {}
    corrections = record.get("corrections") or {}
    shapes = corrections.get("mask_shapes") or []
    sources = corrections.get("mask_sources") or []
    path = mask_path or corrections.get("mask_path") or next(iter(sources), None)
    image = bool(corrections.get("mask_path"))
    if shapes and (image or any(edit.get("kind") not in ("rectangle", "polygon")
                                or edit.get("mode") == "include" for edit in shapes)):
        kind = f"user mask with {len(shapes) + int(image)} ordered edit(s)"
    else:
        kind = MASK_IMAGE if image else (MASK_SHAPES.format(count=len(shapes)) if shapes else None)
    return {
        "applied": bool(image or shapes),
        "file": _existing(path) if (image or shapes) else None,
        "kind": kind,
        "masked_pixels": record.get("drawn_mask_pixels"),
        "bad_pixels_left_out": record.get("bad_pixels_left_out"),
    }


def input_files(report: dict, record: Optional[dict]) -> dict:
    """Every file the result was made from, each by name, path, size and SHA-256: the frame's files (the module
    or data files of a source, the frames summed with it), and the calibration, mask, background and flat-field
    file when one was used (``None``: none)."""
    record = record or {}
    frames = list(record.get("source_files") or [record.get("source_file") or report.get("frame")])
    frames += [item.get("file") for item in record.get("summed_frames") or ()]
    seen: list[str] = []
    for path in frames:
        if path and str(path) not in seen:
            seen.append(str(path))
    corrections = record.get("corrections") or {}
    return {
        "frame": [file_record(path) for path in seen],
        "calibration": _existing((report.get("calibration") or {}).get("source_image")),
        "mask": _existing(corrections.get("mask_path")),
        "mask_sources": [file_record(source) for source in corrections.get("mask_sources") or ()],
        "background": _existing(corrections.get("background_path")),
        "flat_field": _existing(corrections.get("flat_field_path")),
    }


def run_record(
    report: dict, record: Optional[dict], *, mask_path: Optional[str] = None, options: Optional[dict] = None,
    created: Optional[str] = None,
) -> dict:
    """The keys the module docstring lists, for ``report`` and the analysis record of its frame as analysed now."""
    created = created or utc_now()
    if record is not None:
        record["created"] = created  # one time for the whole record
    inputs = input_files(report, record)
    mask = mask_record(record, mask_path)
    if mask["applied"] and inputs["mask"] is None:  # a masks .json: its shapes were applied, the path is the session's
        inputs["mask"] = mask["file"]
    return {
        "software_version": SOFTWARE, "created": created, "options": options, "inputs": inputs, "mask": mask,
        "analysis_record": record,
    }


def mask_lines(mask: dict) -> list[str]:
    """What a report says about masks (``report["mask"]``): the mask file applied, or that none was, and apart
    from it what is always left out."""
    file = mask.get("file") or {}
    if mask.get("applied"):
        pixels = "" if mask.get("masked_pixels") is None else f"; {mask['masked_pixels']} pixels masked"
        digest = f"; SHA-256 {file['sha256']}" if file.get("sha256") else ""
        path = file.get("path") or file.get("file")
        label = f"`{path}`" if path else "Analyze user mask"
        lines = [f"- mask applied: {label} ({mask.get('kind')}{pixels}{digest})"]
    else:
        lines = ["- mask applied: none (no mask file was given)"]
    bad = mask.get("bad_pixels_left_out")
    found = "not searched" if bad is None else ", ".join(f"{count} {name.replace('_', ' ')}" for name, count in bad.items())
    return lines + [f"- left out apart from a mask: the detector's gaps (pixels without data) and bad pixels found "
                    f"automatically ({found})"]


def _file_line(what: str, file: dict) -> str:
    size = "" if file.get("bytes") is None else f", {file['bytes']} bytes"
    return f"- {what}: `{file.get('path') or file.get('file')}` (SHA-256 {file.get('sha256') or 'not readable'}{size})"


def _corrections(record: dict) -> str:
    """The corrections of the analysis record in one line: what was applied, and that the rest was not."""
    corrections = record.get("corrections") or {}
    parts = []
    background = corrections.get("background_path")
    parts.append(
        f"background `{Path(str(background)).name}` × {float(corrections.get('background_scale') or 1.0):g} subtracted"
        if background else "no background subtracted")
    low, high = corrections.get("minimum"), corrections.get("maximum")
    if low is not None or high is not None:
        parts.append(f"valid intensities {'—' if low is None else f'{low:g}'} to {'—' if high is None else f'{high:g}'}")
    flat = corrections.get("flat_field")
    if flat:
        parts.append(f"divided by the flat field `{flat.get('file')}` ({flat.get('masked_pixels')} pixels left out)")
    if corrections.get("gap_guard_px"):
        parts.append(f"{corrections['gap_guard_px']} px next to detector gaps left out")
    if corrections.get("mirror_fill"):
        parts.append("gaps filled from the mirror image")
    parts.append("bad pixels searched and left out" if corrections.get("bad_pixels", True) else "bad pixels not searched")
    intensity = record.get("intensity_corrections") or {}
    applied = [name.replace("_", " ") for name in ("solid_angle", "polarization", "film_absorption") if intensity.get(name)]
    parts.append("intensity corrections: " + (", ".join(applied) if applied else "none (solid angle, polarisation "
                                              "and film absorption off)"))
    critical = (record.get("geometry") or {}).get("critical_angle_deg")
    if critical:
        parts.append(f"q_z refracted (αc = {float(critical):g}°)")
    return "; ".join(parts)


def record_lines(report: dict) -> list[str]:
    """The Record section of a report: the release, each input file with its SHA-256, where the calibration comes
    from (the file with its path, and the instrument profile), the corrections and the error model. Empty for a
    report without a record (one made before its frame was analysed)."""
    if not report.get("software_version"):
        return []
    record = report.get("analysis_record") or {}
    inputs = report.get("inputs") or {}
    lines = [f"- written by {report['software_version']}"]  # the time is in the JSON record: two equal runs, one text
    frames = inputs.get("frame") or []
    lines += [_file_line("input", file) for file in frames[:FILES_LISTED]]
    if len(frames) > FILES_LISTED:
        lines.append(f"- input: {len(frames) - FILES_LISTED} more file(s), each with its SHA-256 in the JSON record")
    profile = record.get("instrument_profile") or {}
    source = (report.get("geometry") or {}).get("source")
    calibration = inputs.get("calibration")
    if calibration:
        lines.append(_file_line("calibration file", calibration))
    elif source:
        lines.append(f"- calibration: {source}")
    if profile:
        made = f", saved {profile['updated_at']}" if profile.get("updated_at") else ""
        lines.append(f"- instrument profile: '{profile.get('name')}' (source: {profile.get('source') or 'not recorded'}{made})")
    elif not calibration and not source:
        lines.append("- calibration: none (no geometry)")
    if inputs.get("background"):
        lines.append(_file_line("background file", inputs["background"]))
    if inputs.get("flat_field"):
        lines.append(_file_line("flat-field file", inputs["flat_field"]))
    if record:
        lines.append(f"- corrections: {_corrections(record)}")
        if record.get("error_model"):
            lines.append(f"- σ of the curves: {str(record['error_model']).replace('_', ' ')}")
    return lines


__all__ = [
    "FILES_LISTED", "MASK_IMAGE", "MASK_SHAPES", "exact_calibration", "input_files", "mask_lines", "mask_record", "profile_source",
    "record_lines", "run_record", "utc_now",
]
