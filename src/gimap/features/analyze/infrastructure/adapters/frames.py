"""Detector frames for Analyze, read through the shared detector_io loader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

from src.gimap.shared.detector_io import (
    DETECTOR_SUFFIXES,
    BeingWritten,
    DetectorImage,
    NexusLayout,
    cached_layout,
    detect_nxs_frame_count,
    group_sources,
    is_hdf5,
    last_known_layout,
    load_detector_image,
)
from src.gimap.shared.detector_io.nexus_files import left_alone
from src.gimap.shared.detector_io.nexus_layout import ERROR_TEMPLATES, FRAME_BEYOND

SUPPORTED_SUFFIXES = DETECTOR_SUFFIXES
SKIPPED_FOLDERS = ("gimap_analysis",)
"""Subfolders never searched for frames: what GIMaP wrote itself (hidden folders are skipped too)."""
NO_FRAMES_YET = "{name} holds no frames yet."
NO_FILTER = "{name} is compressed with a filter this installation cannot read: install the Python package hdf5plugin."
_NATURAL = re.compile(r"(\d+)")


def _sentence(template: str) -> re.Pattern:
    """A pattern that matches every text made from ``template``."""
    return re.compile(".+?".join(re.escape(part) for part in re.split(r"\{\w+\}", template)), re.DOTALL)


_OWN_SENTENCES = tuple(_sentence(template) for template in ERROR_TEMPLATES if template != FRAME_BEYOND)
"""The reader's sentences that say what is wrong in words already (``nexus_layout.ERROR_TEMPLATES``): a file its
writer holds or left unfinished, a file without a detector image, a dataset that holds no frames."""
_NO_FRAMES = _sentence(FRAME_BEYOND.replace("{frames}", "0"))  # “frame 1 of 0”


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
    text = str(exc)
    if isinstance(exc, BeingWritten) or any(pattern.fullmatch(text) for pattern in _OWN_SENTENCES):
        return text  # the reader's own sentence: the file its writer holds, no image, a dataset that holds no frames
    if isinstance(exc, PermissionError):
        return f"{name} cannot be read: no permission (is it open in another program?)."
    if isinstance(exc, IndexError) and is_hdf5(path):
        if _NO_FRAMES.fullmatch(text):  # the file is there, its frames are not written yet
            return NO_FRAMES_YET.format(name=name)
        return f"{name} has fewer frames than asked for ({exc})."
    lowered = text.lower()
    if "could not identify" in lowered or "unknown" in lowered and "format" in lowered or "unsupported detector image" in lowered:
        return f"{name} is not a detector image GIMaP can read (CBF, TIFF, EDF or NeXus)."
    if is_hdf5(path) and ("filter" in lowered and "not registered" in lowered or "plugin" in lowered):
        return NO_FILTER.format(name=name)
    # (The library raises RuntimeError for metadata that contradict each other: “Link iteration failed …”.)
    if is_hdf5(path) and (isinstance(exc, RuntimeError) or "unable to" in lowered or "truncated" in lowered
                          or "address" in lowered or "signature" in lowered):
        return f"{name} looks damaged or incomplete (the NeXus/HDF5 file could not be opened) — if it is still being written, try again later."
    if isinstance(exc, (AttributeError, TypeError, ValueError, KeyError, EOFError, OSError)):
        return f"{name} could not be read: the file looks damaged, incomplete or of another kind ({type(exc).__name__}: {text})."
    return f"{name} could not be read ({type(exc).__name__}: {text})."


class FrameReadError(ValueError):
    """A frame file could not be read; the message says why in words."""


def _frames_in(folder: Path) -> list[Path]:
    """The frame files a folder holds itself; when it holds none, those of its subfolders, one level down
    (a scan folder whose detectors write into ``lmbdp03/``, ``pilatus/`` …). One listing per folder."""
    found: list[Path] = []
    below: list[str] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name.lower().endswith(SUPPORTED_SUFFIXES):
                    if entry.is_file():
                        found.append(Path(entry.path))
                elif not entry.name.startswith(".") and entry.name.casefold() not in SKIPPED_FOLDERS and entry.is_dir():
                    below.append(entry.path)
    except OSError:
        return []
    if found:
        return found
    for subfolder in sorted(below):
        try:
            with os.scandir(subfolder) as entries:
                found.extend(Path(entry.path) for entry in entries
                             if entry.name.lower().endswith(SUPPORTED_SUFFIXES) and entry.is_file())
        except OSError:
            continue
    return found


class DetectorIoFrameSource:
    def load(self, path: str | Path, frame_index: int = 0) -> DetectorImage:
        try:
            return load_detector_image(path, frame_idx=int(frame_index))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:  # noqa: BLE001 - every reader failure becomes one readable message
            raise FrameReadError(readable_read_error(Path(path), exc)) from exc

    def describe(self, path: str | Path) -> Optional[NexusLayout]:
        """What a NeXus / HDF5 source consists of (``None``: a frame of another format). Asking again costs
        ``os.stat`` only while its files are unchanged. A source that cannot be read now is described as it was
        when it was last read (``None`` when it never was): reading a frame of it says why."""
        if not is_hdf5(path):
            return None
        try:
            return cached_layout(path)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:  # noqa: BLE001 - what was known before stays known
            return last_known_layout(path)

    def parts(self, path: str | Path) -> tuple[Path, ...]:
        """Every file the source of ``path`` stands for, the one that names it first: the module files of a
        set (also one that cannot be read yet), a master and every data file it links to (also those behind a
        data file that is missing: ``NexusLayout.behind``). None of them is a row of its own beside it. Another
        format, or a file never read: itself."""
        layout = self.describe(path)
        if layout is None:
            return (Path(path),)
        return tuple(dict.fromkeys([layout.path, *layout.members, *layout.pending, *layout.behind]))

    def settled_size(self, path: str | Path) -> int:
        """Bytes on disk of the source: every module file of a set, a master with its data files.

        ``BeingWritten`` (an ``OSError``) while its writer holds one of its files: the folder watch asks again
        at its next poll. While a module file of a set cannot be read and may still be being written or copied,
        an ``OSError`` whose text says why (held by its writer, damaged or incomplete). A module file that has
        stood unreadable for a while is not waited for: the set is measured as it is (it says what it lacks, and
        is never whole). A file that cannot be read for another reason counts with its own size (it is listed,
        and reading it says why)."""
        path = Path(path)
        members: Sequence[Path] = (path,)
        if is_hdf5(path):
            try:
                layout = cached_layout(path)
            except BeingWritten:
                raise
            except Exception:  # noqa: BLE001
                layout = None
            if layout is not None:
                changing = [part for part in layout.pending if not left_alone(part)]
                if changing:
                    raise self._why_unread(changing[0])
                members = (*layout.members, *layout.pending)
        return sum(member.stat().st_size for member in members)

    @staticmethod
    def _why_unread(path: Path) -> OSError:
        """Why a module file of a set cannot be read now, as the error the folder watch says and waits on."""
        try:
            cached_layout(path)
        except BeingWritten as error:  # (also a file its writer left unfinished)
            return error
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as error:  # noqa: BLE001 - damaged, or still being copied
            return OSError(readable_read_error(path, error))
        return BeingWritten(path)  # (readable by now: the set is put together again at the next look)

    def frame_count(self, path: str | Path) -> int:
        """Frames the source holds now (for a module set: those every module has). While it cannot be read,
        the count of the last time it could. When it never could: the reader's ``OSError`` / ``ValueError``, and
        a ``FrameReadError`` (a ``ValueError``) with a readable sentence for whatever else the library raises
        (``RuntimeError`` for a damaged or half-written file), so that callers handle one kind of failure."""
        path = Path(path)
        if not is_hdf5(path):
            return 1
        try:
            return int(detect_nxs_frame_count(path))
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as exc:  # noqa: BLE001 - as ``load``: every reader failure is handled the same way
            known = last_known_layout(path)
            if known is not None and known.kind != "no_image":
                return int(known.frames)
            if isinstance(exc, (OSError, ValueError)):
                raise
            raise FrameReadError(readable_read_error(path, exc)) from exc

    def expand(self, paths: Sequence[str | Path]) -> list[Path]:
        """The detector sources in the given files and folders, sorted; one path stands for each source.

        A folder gives the frames it holds itself, or, when it holds none, those one level down. Files that
        are one source are one entry (``group_sources``): the module files of a detector (``…_m01.nxs`` …)
        become the first of them, the data files of a master the master (also the data files behind one that
        is missing, when their master is listed). An HDF5 file without a detector image is left out of a folder;
        one named itself is kept, and reading it says what it is.
        """
        in_folders: list[Path] = []
        named: list[Path] = []
        for item in paths:
            path = Path(item).expanduser()
            if path.is_dir():
                in_folders.extend(_frames_in(path))
            elif path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                named.append(path)
        unique: dict[str, Path] = {}
        for path in group_sources(in_folders) + group_sources(named, keep_without_image=True):
            source = path if is_hdf5(path) else path.resolve()  # an HDF5 source keeps the folder its files are in
            unique.setdefault(str(source).casefold(), source)
        return sorted(unique.values(), key=lambda path: (_natural_key(path.parent), _natural_key(path)))


__all__ = ["DetectorIoFrameSource", "FrameReadError", "NO_FILTER", "NO_FRAMES_YET", "SUPPORTED_SUFFIXES",
           "readable_read_error"]
