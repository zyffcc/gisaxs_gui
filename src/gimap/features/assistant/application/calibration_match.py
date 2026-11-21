"""Does a calibration file the automatic search found belong to the frame?

A ``.poni`` the person named, or one lying beside the frame, is theirs. One the search picked up in another folder
(a sibling data set, a folder further up) may come from another experiment: it is adopted only when nothing the two
files say about themselves contradicts it: another detector shape and it is skipped; another energy and it is used
(the distance and the beam centre do not depend on the energy) and the run says so among its decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..domain.calibration_files import folder_place
from .calibration_tools import file_key

ENERGY_TOLERANCE = 2e-3
"""Relative difference of the calibration's and the frame's energy beyond which they are different set-ups (the
tolerance the pipeline uses elsewhere for "measured at another energy")."""
OTHER_SHAPE = (
    "{name} lies in another folder and was made for a detector of {shape} pixels, the frame has {frame_shape}: "
    "another experiment's calibration"
)
OTHER_ENERGY = (
    "{name} was found by the search in another folder than the frame's and was made at {energy} keV, while the "
    "frame was measured at {frame_energy} keV: its distance and beam centre are used with the frame's energy."
)
OTHER_ENERGY_DECISION = "{name}: made at another energy than the frame's"
_MAX_SHAPE = re.compile(r'"max_shape"\s*:\s*(\[[^\]]*\])')


def poni_shape(text: str) -> Optional[tuple[int, int]]:
    """The detector shape (rows, columns) a ``.poni`` states in ``Detector_config``; ``None`` when it states none."""
    found = _MAX_SHAPE.search(text or "")
    if not found:
        return None
    try:
        rows, columns = (int(value) for value in json.loads(found.group(1)))
    except (TypeError, ValueError):
        return None
    return (rows, columns) if rows > 0 and columns > 0 else None


def beside_frame(frame_path: str, path: str) -> bool:
    """``path`` lies in the frame's folder or in a folder below it."""
    if not frame_path:
        return False
    frame_folder = file_key(Path(frame_path).resolve().parent)
    return folder_place(frame_folder, file_key(Path(path).resolve()))[0] == 0


def searched_mismatch(
    path: str, info: dict, frame: dict, text: Optional[str] = None,
) -> Optional[str]:
    """Why a calibration file found by the search in ANOTHER folder than the frame's must not be adopted, in words;
    ``None`` when nothing contradicts it (or it lies beside the frame).

    ``info`` is ``inspect_file``'s payload, ``frame`` the status of the open frame, ``text`` the ``.poni``'s text."""
    if beside_frame(str(frame.get("path") or ""), path):
        return None
    name = Path(path).name
    shape, frame_shape = info.get("shape") or poni_shape(text or ""), frame.get("shape")
    if shape and frame_shape and sorted(int(v) for v in shape) != sorted(int(v) for v in frame_shape):
        return OTHER_SHAPE.format(name=name, shape=list(shape), frame_shape=list(frame_shape))
    return None


def searched_other_energy(path: str, info: dict, frame: dict, frame_energy_kev: Optional[float]) -> Optional[str]:
    """What to say when a calibration file the search found in another folder was made at another energy than the
    frame's: its distance and beam centre do not depend on the energy and are used, but it is the mark of another
    set-up, so the run writes it among its decisions (next to the one that says where the search found the file)
    instead of adopting it silently. ``None`` otherwise."""
    energy = info.get("energy_kev")
    if beside_frame(str(frame.get("path") or ""), path) or not (energy and frame_energy_kev):
        return None
    if abs(energy - frame_energy_kev) <= ENERGY_TOLERANCE * frame_energy_kev:
        return None
    return OTHER_ENERGY.format(name=Path(path).name, energy=f"{energy:.4g}", frame_energy=f"{frame_energy_kev:.4g}")


__all__ = ["ENERGY_TOLERANCE", "OTHER_ENERGY", "OTHER_ENERGY_DECISION", "OTHER_SHAPE", "beside_frame", "poni_shape",
           "searched_mismatch", "searched_other_energy"]
