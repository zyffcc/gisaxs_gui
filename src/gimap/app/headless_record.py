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
from pathlib import Path
from typing import Optional, Sequence

from src.gimap.shared.version import SOFTWARE

from ..features.assistant.application import fit_curve_table, fit_solutions_csv, pipeline_markdown, ring_overlays
from ..features.assistant.application import run_record as records
from ..features.assistant.application.report_language import ENGLISH
from ..features.assistant.application.run_record import MASK_IMAGE, MASK_SHAPES, input_files, utc_now
from .headless_sources import batch_summary

CURVES = ("radial", "in_plane", "out_of_plane", "azimuthal", "horizontal", "vertical")
SUMMARY_LEAVES_OUT = ("tables", "analysis_record", "options")
"""Parts of a frame's report that stay in its own report.json (``options`` is the summary's own)."""
MADE_BY_THE_RUN = "automatic analysis:"
"""How the source of an instrument profile begins when the automatic analysis chose its geometry."""


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


def mask_record(session, record: Optional[dict]) -> dict:
    """Which mask was applied, or that none was; apart from it, the bad pixels found automatically."""
    return records.mask_record(record, getattr(session, "mask", None))


def run_record(session, report: dict, *, options: Optional[dict] = None, created: Optional[str] = None) -> dict:
    """What identifies a report's inputs and settings (the keys the module docstring lists; the window's Save
    Report attaches the same, ``assistant/application/run_record.py``)."""
    return records.run_record(report, analysis_record(session), mask_path=getattr(session, "mask", None),
                              options=options, created=created)


def provenance_lines(session) -> str:
    """The lines above the column names of every CSV the command line writes: the release, each input file by name
    and SHA-256, and the geometry, so a curve says where it came from also without report.json beside it."""
    from ..features.analyze.application.export_record import analysis_metadata
    from ..features.analyze.application.export_record import provenance_lines as provenance

    analysis = session.page.view_model.state.analysis
    if analysis is None:
        return f"{SOFTWARE} (command line); settings and decisions in report.json"
    return "\n".join(provenance(analysis_metadata(analysis), "command line", "report.json"))


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
