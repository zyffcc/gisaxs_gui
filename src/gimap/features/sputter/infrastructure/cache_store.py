"""Where Analyze keeps its import caches, and the caches on disk (``application/storage.py`` says what they are).

**The local folder** (``local_data_dir``): ``%LOCALAPPDATA%\\GIMaP`` on Windows, ``~/Library/Caches/GIMaP`` on macOS,
``$XDG_CACHE_HOME/gimap`` (``~/.cache/gimap``) elsewhere; with ``GIMAP_HOME`` set, ``<GIMAP_HOME>/cache``. Not the user
data folder: on Windows that is the roaming profile, which on many institute computers is a slow network home with a
quota, and an import is gigabytes. Imports GIMaP made before in the user data folder stay where they are: Analyze
uses one found there (``ClientPackageRunner.roots``), and they are listed and can be removed like the others.

**A cache** is a folder ``<root>/sputter_client/<package version>/data/<mask>/<label>`` (only GIMaP's own labels,
``gui_…``), with ``<label>.trace`` and ``<label>.gimap.json`` beside it. Its last use is the time of the marker
(``touch`` sets it), else of the folder.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable

from ..application import LABEL_PREFIX
from ..application.storage import PLACE_LOCAL, PLACE_USER, ImportCache

STORAGE_FOLDER = "sputter_client"
IMPORT_MARKER = ".gimap.json"
TRACE_SUFFIX = ".trace"


def local_data_dir(environ=None, system: str = "", home=None) -> Path:
    """GIMaP's local folder for large files it can make again (the import caches of Analyze). ``environ``, ``system``
    (``win32``, ``darwin``, ``linux``) and ``home``: this computer's unless given."""
    environ = os.environ if environ is None else environ
    system = system or ("win32" if os.name == "nt" else sys.platform)
    home = Path(home) if home is not None else Path.home()
    override = str(environ.get("GIMAP_HOME", "")).strip()
    if override:
        return Path(override).expanduser() / "cache"
    if system == "win32":
        return Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / "GIMaP"
    if system == "darwin":
        return home / "Library" / "Caches" / "GIMaP"
    return Path(environ.get("XDG_CACHE_HOME") or home / ".cache") / "gimap"


def folder_size(folder: Path) -> int:
    """Bytes of every file under ``folder`` (links not followed; 0 when it is not there)."""
    total, pending = 0, [Path(folder)]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class ImportCacheStore:
    def __init__(self, roots: Callable[[], list]):
        """``roots()``: [(folder, place)] — the folders that hold ``sputter_client`` (local first, then the user data
        folder), each listed once."""
        self._roots = roots

    def roots(self) -> list:
        seen, found = set(), []
        for folder, place in self._roots():
            key = os.path.normcase(os.path.abspath(str(folder)))
            if key not in seen:
                seen.add(key)
                found.append((Path(folder), place))
        return found

    def data_folders(self) -> list:
        """(data folder, place) of every package version and mask."""
        found = []
        for root, place in self.roots():
            base = root / STORAGE_FOLDER
            try:
                versions = [path for path in base.iterdir() if path.is_dir()] if base.is_dir() else []
            except OSError:
                continue
            for version in sorted(versions):
                data = version / "data"
                try:
                    found += [(path, place) for path in sorted(data.iterdir()) if path.is_dir()] if data.is_dir() else []
                except OSError:
                    continue
        return found

    def caches(self) -> list:
        found = []
        for data, place in self.data_folders():
            try:
                entries = [(path.name, path.is_dir()) for path in data.iterdir()]
            except OSError:
                continue
            labels = set()
            for name, is_folder in entries:  # a label's folder, its details' folder or its marker
                if is_folder and name.endswith(TRACE_SUFFIX):
                    name = name[: -len(TRACE_SUFFIX)]
                elif not is_folder:
                    name = name[: -len(IMPORT_MARKER)] if name.endswith(IMPORT_MARKER) else ""
                if name.startswith(LABEL_PREFIX):
                    labels.add(name)
            for label in sorted(labels):
                found.append(self._cache(data, label, place))
        return found

    def _cache(self, data: Path, label: str, place: str) -> ImportCache:
        paths = tuple(path for path in (data / label, data / (label + TRACE_SUFFIX), data / (label + IMPORT_MARKER))
                      if path.exists())
        size = sum(folder_size(path) if path.is_dir() else _size(path) for path in paths)
        marker = data / (label + IMPORT_MARKER)
        used = _mtime(marker) if marker.exists() else max((_mtime(path) for path in paths), default=0.0)
        return ImportCache(label, data, size, used, place if place in (PLACE_LOCAL, PLACE_USER) else PLACE_LOCAL, paths)

    def remove(self, cache: ImportCache) -> None:
        for path in (Path(cache.folder) / cache.label, Path(cache.folder) / (cache.label + TRACE_SUFFIX)):
            if path.is_dir():
                shutil.rmtree(path)
        (Path(cache.folder) / (cache.label + IMPORT_MARKER)).unlink(missing_ok=True)

    @staticmethod
    def folder_size(folder: Path) -> int:
        return folder_size(folder)

    @staticmethod
    def touch(data: Path, label: str) -> None:
        """The import is used now (its marker's time; nothing when it has none yet)."""
        marker = Path(data) / (label + IMPORT_MARKER)
        try:
            if marker.is_file():
                os.utime(marker, None)
        except OSError:
            pass


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


__all__ = ["IMPORT_MARKER", "ImportCacheStore", "STORAGE_FOLDER", "TRACE_SUFFIX", "folder_size", "local_data_dir"]
