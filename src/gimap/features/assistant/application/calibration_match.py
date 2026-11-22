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

from src.gimap.shared.geometry.orientation import KNOWN_DETECTORS, detector_shape

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
NOT_THIS_FRAME = (
    "the calibration does not match this frame: {name} was made for a detector of {shape} pixels{detector}, "
    "the frame has {frame_shape}"
)
SEARCHED_ITEM = "calibration found by the search"
SEARCHED_WHY = (
    "Nobody named the calibration: the search found {path} ({place}). A file from another folder may belong to "
    "another experiment or detector, and nothing in it is checked against a standard image of this measurement."
)
SEARCHED_HINT = "Name the calibration file of this measurement (this path when it is the right one)."
_MAX_SHAPE = re.compile(r'"max_shape"\s*:\s*(\[[^\]]*\])')


def calibration_shape(info: dict, text: Optional[str] = None) -> Optional[tuple[int, int]]:
    """The detector shape a calibration file states: its own record, a ``.poni``'s ``max_shape``, else the shape of
    the detector it names."""
    shape = info.get("shape") or poni_shape(text or "") or detector_shape(info.get("detector"))
    return tuple(int(value) for value in shape) if shape else None


def shape_mismatch(path: str, info: dict, frame: dict, text: Optional[str] = None) -> Optional[str]:
    """Why a calibration file — given, noted or found — is not this frame's: it states (or names a detector of)
    another shape than the frame's; ``None`` when it states none or the shapes agree (either orientation)."""
    shape, frame_shape = calibration_shape(info, text), frame.get("shape")
    if not shape or not frame_shape:
        return None
    frame_shape = tuple(int(v) for v in frame_shape)
    # On a flipped axis the extent sets the beam coordinate. A transposed shape has different extents.
    oriented = (info.get("detector_tilt") or {}).get("orientation", 3) != 3
    if shape == frame_shape or (not oriented and sorted(shape) == sorted(frame_shape)):
        return None
    named = f" ({info['detector']})" if info.get("detector") and not (info.get("shape") or poni_shape(text or "")) else ""
    return NOT_THIS_FRAME.format(name=Path(path).name, shape=list(shape), detector=named, frame_shape=list(frame_shape))


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
    shape, frame_shape = calibration_shape(info, text), frame.get("shape")
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


__all__ = ["ENERGY_TOLERANCE", "KNOWN_DETECTORS", "NOT_THIS_FRAME", "OTHER_ENERGY", "OTHER_ENERGY_DECISION", "OTHER_SHAPE",
           "SEARCHED_HINT", "SEARCHED_ITEM", "SEARCHED_WHY", "beside_frame", "calibration_shape", "detector_shape",
           "poni_shape", "searched_mismatch", "searched_other_energy", "shape_mismatch"]
