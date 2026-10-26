"""The raw detector frames of one time window of a result, and the geometry the result was made with (pure Python).

The model sums the CBF frames of each time window; its binned files (``binned/chunk_NNNNN.json``) hold one row per
window: its id (``frame``, as in the result's tables), the times of its first and last frame, how many frames it
summed (``raw_frames``) and their file names in order (``raw_members``). GIMaP records the scan folder and the request
a result was predicted from (``gimap_request.json``). The frames of a window are the files of those names in that scan
folder: Sputter ▸ Show in Analyze shows their sum.

The beam centre is the one given for Predict, in GIMaP's pixel coordinates (pixel corners on integers, row 0 at the
top, as Analyze shows it); ``None`` when the model estimated it itself. That estimate is kept apart
(``estimated_beam``, in the same coordinates): it is named, never set in Analyze.

Problems are English templates (``WINDOW_TEMPLATES``) the page says in its language.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .structure import number

HC_KEV_ANGSTROM = 12.398419843320026
"""λ (Å) = hc / E (keV)."""

NO_SCAN = "{folder} does not say which scan it was predicted from (gimap_request.json): its frames cannot be shown."
SCAN_MISSING = "The scan folder of this result is not there: {folder}"
NO_WINDOWS = "{folder} has no binned windows (binned/chunk_*.json): its frames cannot be shown."
UNKNOWN_WINDOW = "Window {window} is not among the binned windows of {folder}."
NO_MEMBERS = "The binned windows of {folder} do not name the frames of window {window} (raw_members)."
FRAMES_MISSING = "{missing} of the {count} frames of window {window} are not in {folder} (first missing: {name})."
WINDOW_TEMPLATES = (NO_SCAN, SCAN_MISSING, NO_WINDOWS, UNKNOWN_WINDOW, NO_MEMBERS, FRAMES_MISSING)


@dataclass(frozen=True)
class SputterGeometry:
    """The geometry a result was predicted with (``gimap_request.json``); ``None`` where it does not say."""

    sdd_mm: Optional[float] = None
    alpha_i_deg: Optional[float] = None
    wavelength_nm: Optional[float] = None
    energy_ev: Optional[float] = None
    beam_col: Optional[float] = None
    """Beam centre given for Predict (GIMaP pixel coordinates); ``None``: the model estimated it."""
    beam_row: Optional[float] = None
    estimated_beam: Optional[tuple[float, float]] = None
    """The model's own estimate of the beam centre (GIMaP pixel coordinates), when it made one."""

    @property
    def wavelength_angstrom(self) -> Optional[float]:
        if self.wavelength_nm:
            return 10.0 * self.wavelength_nm
        if self.energy_ev:
            return HC_KEV_ANGSTROM / (self.energy_ev / 1000.0)
        return None

    @property
    def beam_center(self) -> Optional[tuple[float, float]]:
        """The beam centre given for Predict, ``(x, y)``; ``None`` when the model estimated it."""
        if self.beam_col is None or self.beam_row is None:
            return None
        return float(self.beam_col), float(self.beam_row)


@dataclass(frozen=True)
class WindowFrames:
    """One time window of a result: the raw frames it summed, in order, and the geometry of the result."""

    window: str
    frames: tuple
    scan_folder: Path
    geometry: SputterGeometry
    time_s: Optional[float] = None
    first_time_s: Optional[float] = None
    """Time of the window's first frame (s, from the start of the scan)."""
    last_time_s: Optional[float] = None
    detector: str = ""
    """The detector the first frame's header names ("" when it does not)."""

    @property
    def count(self) -> int:
        return len(self.frames)


def geometry_of(request: dict, estimated_beam: Optional[tuple[float, float]] = None) -> SputterGeometry:
    """The geometry of a ``gimap_request.json`` request (numbers that are missing or not finite are ``None``)."""
    request = request if isinstance(request, dict) else {}
    values = {key: number(request.get(key)) for key in ("sdd_mm", "alpha_i_deg", "wavelength_nm", "energy_ev",
                                                         "beam_col", "beam_row")}
    return SputterGeometry(**values, estimated_beam=estimated_beam)


def _plain_name(name) -> bool:
    return isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name


def window_frames(row: dict, scan_folder: Path, geometry: SputterGeometry, *, folder) -> WindowFrames:
    """The frames of one row of the binned files: the files of the scan folder it names (``raw_members``), in order.
    A row that names none, or a name that is no plain file name, is refused (``ValueError``)."""
    window = str(row.get("frame") or "")
    members = row.get("raw_members")
    if not isinstance(members, list) or not members or not all(_plain_name(name) for name in members):
        raise ValueError(NO_MEMBERS.format(folder=folder, window=window))
    return WindowFrames(window, tuple(Path(scan_folder) / name for name in members), Path(scan_folder), geometry,
                        time_s=number(row.get("time_s")), first_time_s=number(row.get("first_time_s")),
                        last_time_s=number(row.get("last_time_s")))


__all__ = ["FRAMES_MISSING", "HC_KEV_ANGSTROM", "NO_MEMBERS", "NO_SCAN", "NO_WINDOWS", "SCAN_MISSING", "SputterGeometry",
           "UNKNOWN_WINDOW", "WINDOW_TEMPLATES", "WindowFrames", "geometry_of", "window_frames"]
