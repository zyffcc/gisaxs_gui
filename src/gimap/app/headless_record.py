"""The record and the files of a run without a window (``tools/gimap_agent.py``, the MCP server).

One vocabulary with the window's records (Physics and Methods, chapter 14):

- ``report.json`` holds the release (``software_version``), the UTC time it was written (``created``), the
  options as they were given (``options``, the notes among them), every input by path, size and SHA-256
  (``inputs``: the frame's files, and the calibration, mask and background files when one was used;
  ``shared.file_digest.file_record``), which mask was applied or that none was (``mask``), and
  ``analysis_record``: the record Analyze writes beside its curve exports (``<stem>_analysis.json``: geometry,
  corrections, error model, binning, pixels left out). A model fit names its network, the SHA-256 of the
  network's weights and its seed (``gisaxs.fit.method``) beside the solutions with their intervals.
- every CSV starts with the same ``#`` lines: the release, each input file with its SHA-256, the frame and the
  geometry (``provenance_lines``).
- ``summary.json`` is one object: the release, the time, the options, what the arguments stood for and the
  frame reports (``frames``; before 0.2.0 the file was that list).

Two runs on the same input with the same options differ only in ``created`` and in the durations
(``seconds``): nothing else here depends on the clock.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from src.gimap.shared.file_digest import file_record
from src.gimap.shared.version import SOFTWARE

from ..features.assistant.application import fit_curve_table, fit_solutions_csv, pipeline_markdown, ring_overlays
from ..features.assistant.application.report_language import ENGLISH
from .headless_sources import batch_summary

CURVES = ("radial", "in_plane", "out_of_plane", "azimuthal", "horizontal", "vertical")
SUMMARY_LEAVES_OUT = ("tables", "analysis_record", "options")
"""Parts of a frame's report that stay in its own report.json (``options`` is the summary's own)."""
MADE_BY_THE_RUN = "automatic analysis:"
"""How the source of an instrument profile begins when the automatic analysis chose its geometry."""
MASK_IMAGE = "mask image: its non-zero pixels are left out"
MASK_SHAPES = "GIMaP masks file: {count} drawn shape(s) left out"


def utc_now() -> str:
    """The time a record is written: UTC, ISO 8601, to the second (as ``<stem>_analysis.json``)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def analysis_record(session) -> Optional[dict]:
    """The record Analyze writes beside its exports, for the frame as it is analysed now (``None``: no frame)."""
    from ..features.analyze.application.export_record import analysis_metadata

    analysis = session.page.view_model.state.analysis
    if analysis is None:
        return None
    record = analysis_metadata(analysis)
    profile = record.get("instrument_profile")
    if profile and str(profile.get("source") or "").startswith(MADE_BY_THE_RUN):
        # A profile the run made itself is as old as the run: its clock time would differ between two equal runs.
        record["instrument_profile"] = {key: value for key, value in profile.items() if key != "updated_at"}
    return record


def _existing(path: Any) -> Optional[dict]:
    return file_record(path) if path and os.path.isfile(str(path)) else None


def mask_record(session, record: Optional[dict]) -> dict:
    """Which mask was applied, or that none was; apart from it, the bad pixels found automatically."""
    record = record or {}
    corrections = record.get("corrections") or {}
    shapes = corrections.get("mask_shapes") or []
    path = getattr(session, "mask", None) or corrections.get("mask_path")
    image = bool(corrections.get("mask_path"))
    return {
        "applied": bool(image or shapes),
        "file": _existing(path) if (image or shapes) else None,
        "kind": MASK_IMAGE if image else (MASK_SHAPES.format(count=len(shapes)) if shapes else None),
        "masked_pixels": record.get("drawn_mask_pixels"),
        "bad_pixels_left_out": record.get("bad_pixels_left_out"),
    }


def input_files(report: dict, record: Optional[dict]) -> dict:
    """Every file the result was made from, each by name, path, size and SHA-256: the frame's files (the module
    or data files of a source, the frames summed with it), and the calibration, mask and background file when
    one was used (``None``: none)."""
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
        "background": _existing(corrections.get("background_path")),
    }


def run_record(session, report: dict, *, options: Optional[dict] = None, created: Optional[str] = None) -> dict:
    """What identifies a report's inputs and settings (the keys the module docstring lists)."""
    created = created or utc_now()
    record = analysis_record(session)
    if record is not None:
        record["created"] = created  # one time for the whole record
    inputs = input_files(report, record)
    mask = mask_record(session, record)
    if mask["applied"] and inputs["mask"] is None:  # a masks .json: its shapes were applied, the path is the session's
        inputs["mask"] = mask["file"]
    return {
        "software_version": SOFTWARE, "created": created, "options": options, "inputs": inputs, "mask": mask,
        "analysis_record": record,
    }


def provenance_lines(session) -> str:
    """The lines above the column names of every CSV the command line writes: the release, each input file by name
    and SHA-256, and the geometry, so a curve says where it came from also without report.json beside it."""
    from ..features.analyze.application.export_record import analysis_metadata

    analysis = session.page.view_model.state.analysis
    lines = [f"{SOFTWARE} (command line); settings and decisions in report.json"]
    if analysis is not None:
        record = analysis_metadata(analysis)
        lines += [f"input: {item['file']} sha256 {item['sha256']}" for item in record.get("input_sha256") or ()]
        lines.append(f"frame: {record.get('frame')}")
        if (geometry := analysis.geometry) is not None:
            lines.append(
                f"geometry: distance {geometry.distance_m * 1e3:.4f} mm, direct beam ({geometry.beam_center_x_px:.3f}, "
                f"{geometry.beam_center_y_px:.3f}) px, wavelength {geometry.wavelength_angstrom:.6g} A, "
                f"alpha_i {geometry.incidence_deg:.4g} deg, detector tilt {geometry.tilt_deg:.4g} deg" + (
                    f", q_z refracted, alpha_c {geometry.critical_angle_deg:.4g} deg" if geometry.refraction else ""))
    return "\n".join(lines)


def write_outputs(
    session, report: dict, folder: Path, *, options: Optional[dict] = None, created: Optional[str] = None,
    language: str = ENGLISH,
) -> list[Path]:
    """report.json, report.md, the curves as CSV (GISAXS: the cuts and the fit) and the q-map as PNG. ``report``
    gets the record of the run (``run_record``), so a summary made of it names the same inputs."""
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    report.update(run_record(session, report, options=options, created=created))
    written = [folder / "report.json", folder / "report.md"]
    written[0].write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    written[1].write_text(pipeline_markdown(report, language=language), encoding="utf-8")
    provenance = provenance_lines(session)
    for key in CURVES:
        curve = session.curve(key)
        if curve is None or len(curve["x"]) == 0:
            continue
        name = "azimuthal_last_ring.csv" if key == "azimuthal" else f"{key}.csv"
        with open(folder / name, "w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(
                stream, np.column_stack([curve["x"], curve["y"], curve["sigma"]]), delimiter=",",
                header=f"{provenance}\n{curve['x_label']}, I, sigma", comments="# ",
            )
        written.append(folder / name)
    written += _write_fit(((report.get("gisaxs") or {}).get("fit")), folder, provenance)
    png = session.preview_png(1100, rings=ring_overlays(report))
    if png:
        (folder / "qmap.png").write_bytes(png)
        written.append(folder / "qmap.png")
    return written


def _write_fit(fit: Optional[dict], folder: Path, provenance: str = "") -> list[Path]:
    """GISAXS: the fitted curve (data and the fit the run reports, the most probable) and the solutions table,
    both under the ``#`` lines of the curve tables."""
    import numpy as np

    if not fit:
        return []
    written = []
    header, columns = fit_curve_table(fit)
    if len(columns):
        written.append(folder / "fit_curve.csv")
        with open(written[-1], "w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(stream, columns, delimiter=",", header=f"{provenance}\n{header}" if provenance else header,
                       comments="# ")
    if fit.get("solutions"):
        written.append(folder / "fit_solutions.csv")
        with open(written[-1], "w", encoding="utf-8", newline="\n") as stream:
            stream.write(fit_solutions_csv(fit, provenance))
    return written


def write_summary(
    out: Path, reports: Sequence[dict], *, options: Optional[dict] = None, notes: Sequence[str] = (),
    created: Optional[str] = None, language: str = ENGLISH,
) -> None:
    """``summary.md`` and ``summary.json`` of a batch. ``notes``: what patterns and folders among the arguments
    stood for (``headless_inputs.expand_frames``)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(batch_summary(reports, notes=notes, language=language), encoding="utf-8")
    summary = {
        "software_version": SOFTWARE, "created": created or utc_now(), "options": options,
        "arguments": list(notes),
        "frames": [{key: value for key, value in report.items() if key not in SUMMARY_LEAVES_OUT} for report in reports],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


__all__ = [
    "CURVES", "analysis_record", "input_files", "mask_record", "provenance_lines", "run_record", "utc_now",
    "write_outputs", "write_summary",
]
