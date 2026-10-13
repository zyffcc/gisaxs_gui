"""What the automatic analysis is doing, in words, and how it stops.

Every tool call of ``StandardPipeline`` belongs to a phase (reading the frame, geometry, frames,
peaks …). ``step_text`` says in one sentence what a call does; ``SLOW`` names the calls that can
take long, so a person who pressed Stop knows the run ends only after them. A stop request is
checked between two calls: ``PipelineStopped`` ends the run there, and the report keeps what was
found so far (``report["stopped"]`` names the step that was not started). A run also ends, as a
failure, when Analyze no longer shows the file it started on (``FrameChanged``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

READ, GEOMETRY, FRAMES = "Reading the frame", "Geometry", "Frames of the series"
PEAKS, SECTORS, RINGS = "Peaks", "In-plane and out-of-plane", "Orientation and size of the rings"
YONEDA, SPACING, MODEL = "Yoneda cut and symmetry", "Spacing", "Model fit"
PHASES = {
    "giwaxs": (READ, GEOMETRY, FRAMES, PEAKS, SECTORS, RINGS),
    "gisaxs": (READ, GEOMETRY, FRAMES, YONEDA, SPACING, MODEL),
    "geometry": (READ, GEOMETRY),
}
TOOL_PHASE = {
    "get_status": READ,
    "inspect_file": GEOMETRY,
    "find_calibration_files": GEOMETRY,
    "calibrate_geometry": GEOMETRY,
    "use_geometry": GEOMETRY,
    "set_incidence_angle": GEOMETRY,
    "set_frame": FRAMES,
    "find_peaks": PEAKS,
    "compare_sectors": SECTORS,
    "ring_orientation": RINGS,
    "crystallite_size": RINGS,
    "refine_beam_center_symmetry": YONEDA,
    "choose_halves": YONEDA,
    "set_halves": YONEDA,
    "in_plane_spacing": SPACING,
    "fit_horizontal_cut": MODEL,
}
TOOL_TEXT = {
    "get_status": "Reading the frame and its header",
    "inspect_file": "Reading a calibration file",
    "find_calibration_files": "Looking for calibration files and standards near the data",
    "calibrate_geometry": "Fitting the geometry to the rings of the standard",
    "use_geometry": "Applying the geometry",
    "set_incidence_angle": "Setting the incidence angle αi",
    "set_frame": "Choosing (and summing) the frames of the series",
    "set_measurement_mode": "Switching between GISAXS and GIWAXS",
    "find_peaks": "Finding the peaks of I(q)",
    "compare_sectors": "Comparing the in-plane and out-of-plane sectors",
    "ring_orientation": "Orientation of the ring at q = {q}",
    "crystallite_size": "Crystallite size (Scherrer) at q = {q}",
    "refine_beam_center_symmetry": "Finding the symmetry axis of the horizontal cut",
    "choose_halves": "Checking both halves of the horizontal cut",
    "set_halves": "Choosing the halves of the cut",
    "in_plane_spacing": "In-plane spacing from the correlation peak",
    "fit_horizontal_cut": "Fitting form-factor models to the horizontal cut",
}
SLOW = {
    "calibrate_geometry": "can take up to a minute",
    "fit_horizontal_cut": "can take up to half a minute",
    "find_calibration_files": "a few seconds",
    "set_frame": "a few seconds when many frames are summed",
}


class PipelineStopped(Exception):
    """A stop was requested; raised before the next tool call starts."""


class FrameChanged(RuntimeError):
    """Analyze no longer shows the file a run started on (Analyze cleared, a project opened).

    Raised before anything acts on the other file; the run ends there (``report["failed"]``) and its
    report stays the one of the file it started on (``report["frame"]``).
    """


def same_file(first, second) -> bool:
    """Whether two paths name the same file (case and separators as the system compares them)."""
    if not first or not second:
        return False
    return os.path.normcase(os.path.abspath(str(first))) == os.path.normcase(os.path.abspath(str(second)))


def frame_changed(start, now) -> FrameChanged:
    """The failure of a run on ``start`` while Analyze now shows ``now`` (``None``: no frame)."""
    shown = Path(str(now)).name if now else "no frame"
    return FrameChanged(f"The frame changed during the run: it started on {Path(str(start)).name}, Analyze now shows {shown}")


def phase_of(tool: str) -> Optional[str]:
    return TOOL_PHASE.get(tool)


def step_text(
    tool: str, arguments: Optional[Mapping[str, Any]] = None, translate: Callable[[str], str] = lambda text: text,
) -> str:
    """One sentence of what a tool call does; ``translate`` turns the English template into the interface language."""
    text = TOOL_TEXT.get(tool, tool.replace("_", " ").capitalize())
    arguments = arguments or {}
    if "{q}" in text:
        q = arguments.get("q_center")
        if isinstance(q, (int, float)):
            return translate(text).format(q=f"{float(q):.4g} Å⁻¹")
        text = text.replace(" at q = {q}", "")
    return translate(text)


__all__ = [
    "FrameChanged", "PHASES", "PipelineStopped", "SLOW", "TOOL_PHASE", "TOOL_TEXT", "frame_changed", "phase_of",
    "same_file", "step_text",
]
