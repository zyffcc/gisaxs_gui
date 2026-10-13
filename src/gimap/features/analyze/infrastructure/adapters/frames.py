"""Detector frames for Analyze, read through the shared detector_io loader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from src.gimap.shared.detector_io import (
    DetectorImage,
    detect_nxs_frame_count,
    load_detector_image,
    nxs_series_paths,
)

SUPPORTED_SUFFIXES = (".cbf", ".nxs", ".tif", ".tiff", ".edf")
_NATURAL = re.compile(r"(\d+)")


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.casefold() for part in _NATURAL.split(path.name)]


def readable_read_error(path: Path, exc: BaseException) -> str:
    """What went wrong reading a frame, in words (the library's own message only as a last resort)."""
    path = Path(path)
    name = path.name
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return f"{name} is not there any more (moved or deleted)."
    except OSError as error:
        return f"{name} cannot be opened: {error.strerror or error}."
    if size == 0:
        return f"{name} is empty (0 bytes) — it may still be being written, or the copy failed."
    if isinstance(exc, PermissionError):
        return f"{name} cannot be read: no permission (is it open in another program?)."
    if isinstance(exc, IndexError) and path.suffix.lower() == ".nxs":
        return f"{name} has fewer frames than asked for ({exc})."
    text = str(exc)
    lowered = text.lower()
    if "could not identify" in lowered or "unknown" in lowered and "format" in lowered:
        return f"{name} is not a detector image GIMaP can read (CBF, TIFF, EDF or NeXus)."
    if path.suffix.lower() in (".nxs", ".h5", ".hdf5") and ("unable to" in lowered or "truncated" in lowered
                                                             or "address" in lowered or "signature" in lowered):
        return f"{name} looks damaged or incomplete (the NeXus/HDF5 file could not be opened) — if it is still being written, try again later."
    if isinstance(exc, (AttributeError, TypeError, ValueError, KeyError, EOFError, OSError)):
        return f"{name} could not be read: the file looks damaged, incomplete or of another kind ({type(exc).__name__}: {text})."
    return f"{name} could not be read ({type(exc).__name__}: {text})."


class FrameReadError(ValueError):
    """A frame file could not be read; the message says why in words."""


class DetectorIoFrameSource:
    def load(self, path: str | Path, frame_index: int = 0) -> DetectorImage:
        try:
            return load_detector_image(path, frame_idx=int(frame_index))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:  # noqa: BLE001 - every reader failure becomes one readable message
            raise FrameReadError(readable_read_error(Path(path), exc)) from exc

    def settled_size(self, path: str | Path) -> int:
        """Bytes on disk for the frame, all modules of an NXS series included."""
        path = Path(path)
        paths = nxs_series_paths(path) if path.suffix.lower() == ".nxs" else [path]
        return sum(member.stat().st_size for member in paths)

    def frame_count(self, path: str | Path) -> int:
        path = Path(path)
        if path.suffix.lower() != ".nxs":
            return 1
        return int(detect_nxs_frame_count(path))

    def expand(self, paths: Sequence[str | Path]) -> list[Path]:
        """Supported frames in the given files and folders (folders not recursive).

        A multi-module NeXus series (``…_m01.nxs`` … ``…_m11.nxs``) is one
        stitched frame, so only its first module file is listed.
        """
        found: list[Path] = []
        for item in paths:
            path = Path(item).expanduser()
            if path.is_dir():
                candidates = [
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
                ]
            elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                candidates = [path]
            else:
                candidates = []
            found.extend(candidates)
        unique: dict[str, Path] = {}
        for path in found:
            if path.suffix.lower() == ".nxs":
                series = nxs_series_paths(path)
                path = series[0] if series else path
            resolved = path.resolve()
            unique.setdefault(str(resolved).casefold(), resolved)
        return sorted(unique.values(), key=lambda path: (_natural_key(path.parent), _natural_key(path)))


__all__ = ["DetectorIoFrameSource", "SUPPORTED_SUFFIXES"]
