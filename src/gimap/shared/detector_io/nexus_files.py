"""Opening NeXus / HDF5 files for reading, and the file-system facts the layout cache is built on."""

from __future__ import annotations

import functools
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py

try:  # registers the bitshuffle / LZ4 / Zstd / Blosc filters with HDF5 (Eiger, MAX IV files)
    import hdf5plugin  # noqa: F401
except ImportError:  # optional: only files compressed with those filters need it
    hdf5plugin = None

HDF5_SUFFIXES = (".nxs", ".nx", ".nx5", ".h5", ".hdf5", ".hdf")
# English templates of sentences a NeXus / HDF5 read fails with (presentation translates them).
BEING_WRITTEN = "{name} is being written by another program and cannot be read until it is closed."
LEFT_UNFINISHED = ("{name} was left open for writing and has not changed for a while: the program that wrote it seems "
                   "to have ended without closing it. It cannot be read as it is; running h5clear -s on a copy of it "
                   "makes readable what was stored in it.")
UNFINISHED_AFTER_S = 300.0
"""A file that is marked as open for writing and has not changed for this long is taken as left by a writer
that ended without closing it (the mark stays in the file for good). Nothing in the file tells a writer that
died from one that waits, so time decides, and the sentence says "seems". Five minutes: longer than a detector
keeps a file open between two frames (a long exposure, a pause for a motor), short beside a night of waiting."""
NO_IMAGE = "{name} holds no detector image."
FRAME_BEYOND = "frame {frame} of {frames}"  # IndexError: frame is 1-based
_SETTLE_NS = 2_000_000_000  # a folder untouched for this long has no change hiding behind its timestamp
_LISTINGS: dict[str, tuple[int, int, tuple[str, ...]]] = {}  # folder -> (listed at, folder mtime, names)
_BURST = threading.local()


def is_hdf5(path: str | Path) -> bool:
    return Path(path).suffix.lower() in HDF5_SUFFIXES


class BeingWritten(OSError):
    """The file is open for writing by another program and cannot be read until it is closed."""

    def __init__(self, path: Any):
        super().__init__(os.fspath(path))
        self.path = Path(os.fspath(path))

    def __str__(self) -> str:
        return BEING_WRITTEN.format(name=self.path.name)


class LeftUnfinished(BeingWritten):
    """The file is marked as open for writing but has not changed for a while: its writer ended without
    closing it.  It stays unreadable until the mark is cleared (``h5clear -s``); GIMaP never does that."""

    def __str__(self) -> str:
        return LEFT_UNFINISHED.format(name=self.path.name)


def left_alone(path: str | Path) -> bool:
    """The file has not changed for ``UNFINISHED_AFTER_S``: whoever wrote it is taken to have ended."""
    try:
        return time.time() - os.stat(path).st_mtime >= UNFINISHED_AFTER_S
    except OSError:
        return False


def open_nexus(path: str | Path) -> h5py.File:
    """Open an HDF5 file for reading: the one place that does.

    Without file locking first (a writer that holds the lock, GPFS/NFS without locks); a file that
    its writer keeps open is read in SWMR mode when the writer allows that; otherwise
    ``BeingWritten``, or ``LeftUnfinished`` (a ``BeingWritten``) when the file has not changed for a
    while.  Every other failure is the library's own ``OSError``.  The file is never changed.
    """
    name = os.fspath(path)
    try:
        return _unlocked(name)
    except OSError as error:
        text = str(error).lower()
        if "already open for write" not in text and "unable to lock" not in text:
            raise
    try:
        return _unlocked(name, swmr=True)
    except OSError as error:
        raise (LeftUnfinished if left_alone(name) else BeingWritten)(name) from error


def _unlocked(name: str, **options: Any) -> h5py.File:
    try:
        return h5py.File(name, "r", locking=False, **options)
    except ValueError as error:  # an HDF5 library too old to open without locking
        if "locking" not in str(error).lower():
            raise
        return h5py.File(name, "r", **options)


@functools.lru_cache(maxsize=256)
def _real_folder(folder: str) -> str:
    return os.path.realpath(folder)


def _canonical(path: str | Path) -> Path:
    path = Path(os.path.expanduser(os.fspath(path)))
    return Path(_real_folder(os.path.abspath(path.parent)), path.name)


def _stamp(path: str | Path) -> tuple:
    try:
        stat = os.stat(path)
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), 0, -1)


def _signature(paths: Iterable[str | Path]) -> tuple:
    return tuple(_stamp(path) for path in paths)


class one_listing_per_folder:
    """Within the block (of this thread) every folder is listed once: one answer for a whole file list."""

    def __enter__(self) -> None:
        _BURST.listings = {}

    def __exit__(self, *_error: Any) -> None:
        _BURST.listings = None


def _folder_names(folder: str) -> tuple[tuple[str, ...], Optional[int]]:
    """``(file names in the folder, its mtime when no later change can hide behind that timestamp)``.

    A listing is used again while the folder's timestamp stands and it was taken well after that
    change: a file created in the same tick of the clock as an earlier listing is then not missed.
    """
    burst = getattr(_BURST, "listings", None)
    if burst is not None and folder in burst:
        return burst[folder]
    try:
        stamp = os.stat(folder).st_mtime_ns
    except OSError:
        return (), None
    known = _LISTINGS.get(folder)
    if known is None or known[1] != stamp or known[0] - stamp <= _SETTLE_NS:
        try:
            names = tuple(sorted(os.listdir(folder)))
        except OSError:
            names = ()
        if known is not None and known[2] == names:
            names = known[2]  # the same listing as before: what was worked out for it stays valid
        known = _LISTINGS[folder] = (time.time_ns(), stamp, names)
    answer = known[2], (stamp if known[0] - stamp > _SETTLE_NS else None)
    if burst is not None:
        burst[folder] = answer
    return answer
