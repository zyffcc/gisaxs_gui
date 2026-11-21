"""Where the geometry of every frame of a batch comes from, said before Batch Export writes anything.

The frames are grouped by folder. A folder whose frames have a ``.poni`` beside them (``geometry_files.sibling_poni``)
uses it; the folder of the frame on screen uses what that frame uses (a profile picked or matched, the ``.poni``
beside it); in another folder each frame takes the profile picked when it is for frames of its size, else the
profile that matches its detector and size, and a frame without one gives no q curves. When no frame can have a
geometry at all (none on screen, no ``.poni`` beside any frame, no instrument profile saved), Batch Export is off.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Sequence

from .geometry_files import sibling_poni

FOLDER_PONI = "{n} frame(s) in {folder}: geometry from {poni} beside them."
FOLDER_SHOWN = "{n} frame(s) in {folder}: {source}, as the frame on screen."
FOLDER_PROFILE = ("{n} frame(s) in {folder}: the profile picked or the one that fits their detector and size, read "
                  "frame by frame; a frame without one gives no q curves.")
NO_GEOMETRY = "No geometry: q curves cannot be made. Load a .poni, enter the geometry or calibrate first."


def batch_geometry_lines(
    files: Sequence, current_path, current_source: Optional[str], *, have_profiles: bool,
    find_poni: Callable = sibling_poni,
) -> tuple[list[tuple[str, dict]], bool]:
    """``([(template, values), …], possible)``: one sentence per folder of ``files`` (English templates and their
    values, for the page to translate) and whether any frame can have a geometry. ``current_source``: what the frame
    on screen uses (``None``: it has no geometry)."""
    folders: "OrderedDict[Path, list]" = OrderedDict()
    for path in files:
        folders.setdefault(Path(path).parent, []).append(Path(path))
    current_folder = Path(current_path).parent if current_path is not None else None
    lines, possible = [], bool(current_source) or have_profiles
    for folder, paths in folders.items():
        poni = find_poni(paths[0])
        values = {"n": len(paths), "folder": folder.name or str(folder)}
        if poni is not None:
            lines.append((FOLDER_PONI, {**values, "poni": poni.name}))
            possible = True
        elif folder == current_folder and current_source:
            lines.append((FOLDER_SHOWN, {**values, "source": current_source}))
        else:
            lines.append((FOLDER_PROFILE, values))
    if len(lines) == 1 and lines[0][0] == FOLDER_SHOWN:
        lines = []  # one folder, as the frame on screen: the set-up line names its geometry already
    return lines, possible


__all__ = ["FOLDER_PONI", "FOLDER_PROFILE", "FOLDER_SHOWN", "NO_GEOMETRY", "batch_geometry_lines"]
