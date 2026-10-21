"""What a detector source is, said in the interface language.

A source is what one row of the file list stands for. For a NeXus / HDF5 source the frame source describes
it (``view_model.describe(path)``, a ``NexusLayout``): its summary (“11 module files · 240 frames ·
stitched”), its warnings, the orientation its frames are shown in. The reader makes these as English
templates with their values; here they are filled in the interface language (``trf``). Numbers, file names
and dataset paths are never translated.

Frames are counted from 1 on screen (“frame 4 of 12”); file names keep the 0-based ``_frame0003``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.detector_io import DETECTOR_SUFFIXES
from src.gimap.shared.detector_io.nexus_layout import AS_STORED, FRAME_GRID, P03_LAMBDA
from src.gimap.shared.detector_io.nexus_modules import NOT_A_MODULE, SAME_POSITION

from ..application.ports import FRAME_OF, source_name

FILE_FILTER = f"Detector frames ({' '.join('*' + suffix for suffix in DETECTOR_SUFFIXES)});;All files (*)"
"""The Open dialog's filter: every suffix the reader accepts."""
FORMATS = "CBF, NeXus/HDF5, TIFF, EDF"
ORIENTATIONS = {
    AS_STORED: "Image as stored in the file",
    P03_LAMBDA: "P03 Lambda: transposed and flipped (rotated by 90°)",
}
"""How the frames of a NeXus / HDF5 source are shown (``NexusLayout.orientation``)."""
NOTES = (SAME_POSITION, NOT_A_MODULE, FRAME_GRID)
"""Warnings of a layout that only say how the files were understood: shown with the source, but they do not
put the Data step and the status line in the warning state as the others do (frames or parts left out, a
file still written, a dataset chosen among several)."""
ROW_FRAMES = "{name} · {frames} frames"
"""A row of the file list for a source of several frames."""
FRAMES_TOTAL = "of {n}"
"""Beside the frame box: “Frame [4] of 12”."""
FILES_AND_MORE = "{files} and {n} more"
NEW_FRAMES = "{name}: {n} new frames"
NEW_FRAME = "{name}: 1 new frame"
NAME_LIMIT = 30
"""Characters of a file name inside a sentence of the narrow Data step (a name cannot wrap)."""
LISTED_FILES = 12
"""Files of a source a tooltip names before it says “and n more”."""
LISTED_FOLDERS = 3
"""Subfolders a notice names before it says “and n more”."""
_LONG = re.compile(rf"\S{{{NAME_LIMIT + 1},}}")


def frame_text(frame_index: int, frame_count: int) -> str:
    """“frame 4 of 12” for the 0-based ``frame_index``, in the interface language."""
    return trf(FRAME_OF, i=int(frame_index) + 1, n=int(frame_count))


def new_frames_text(name: str, count: int) -> str:
    """“scan_00007: 6 new frames” (the status line of the folder watch)."""
    return trf(NEW_FRAME, name=name) if int(count) == 1 else trf(NEW_FRAMES, name=name, n=int(count))


def short_name(name: str, limit: int = NAME_LIMIT) -> str:
    """``name`` with its middle left out when it is longer than ``limit`` (the scan number at its end stays)."""
    name = str(name)
    if len(name) <= limit:
        return name
    head = (limit - 1) // 2
    return f"{name[:head]}…{name[-(limit - 1 - head):]}"


def names_text(names, limit: int) -> str:
    """``a, b, c`` — the first ``limit`` names, then “and n more”."""
    names = list(names)
    shown, more = ", ".join(names[:limit]), len(names) - limit
    return trf(FILES_AND_MORE, files=shown, n=more) if more > 0 else shown


def summary_text(layout) -> str:
    """What the source consists of (``NexusLayout.summary``), e.g. “11 module files · 240 frames · stitched”."""
    template, values = layout.summary()
    return trf(template, **values)


def orientation_text(layout) -> str:
    return tr(ORIENTATIONS.get(layout.orientation, ORIENTATIONS[AS_STORED]))


def warning_texts(layout, *, notes: Optional[bool] = None, narrow: bool = False, without=()) -> list[str]:
    """The layout's warnings in the interface language; ``notes=False``: only those that mean data are left
    out or may be wrong, ``notes=True``: only the notes on how the files were understood, ``None``: all.
    ``narrow``: for a narrow panel, where a file name or a dataset path cannot wrap: each one longer than
    ``NAME_LIMIT`` is shortened in the middle (say the whole sentence in a tooltip). ``without``: templates
    left out (what the line they go into says in other words already)."""
    if layout is None:
        return []
    return [trf(template, **(_narrow(values) if narrow else values)) for template, values in layout.warnings
            if (notes is None or (template in NOTES) == notes) and template not in without]


def _narrow(values: dict) -> dict:
    return {key: _LONG.sub(lambda found: short_name(found.group()), value) if isinstance(value, str) else value
            for key, value in values.items()}


def row_text(path, layout=None, frames: int = 1) -> str:
    """A row of the file list: the source's name, and its frames when it holds several."""
    name = source_name(path, layout)
    return trf(ROW_FRAMES, name=name, frames=int(frames)) if int(frames) > 1 else name


def row_tooltip(path, layout=None) -> str:
    """The tooltip of a row: the path; for a NeXus / HDF5 source also what it consists of, the files by name
    and its warnings."""
    lines = [str(path)]
    if layout is None:
        return lines[0]
    lines.append(summary_text(layout))
    members = [Path(member).name for member in tuple(layout.members) + tuple(layout.pending)]
    if len(members) > 1:
        lines.append(names_text(members, LISTED_FILES))
    lines.extend(warning_texts(layout))
    return "\n".join(lines)


__all__ = [
    "FILE_FILTER", "FORMATS", "FRAMES_TOTAL", "LISTED_FOLDERS", "NAME_LIMIT", "NEW_FRAME", "NEW_FRAMES", "NOTES",
    "ORIENTATIONS", "ROW_FRAMES", "frame_text", "names_text", "new_frames_text", "orientation_text", "row_text",
    "row_tooltip", "short_name", "summary_text", "warning_texts",
]
