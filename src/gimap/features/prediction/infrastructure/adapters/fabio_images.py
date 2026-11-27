"""Fabio detector image repository。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.gimap.shared.detector_io.completeness import edf_written

from ...application import FolderEntry, IndexedPredictionFile, LoadedPredictionImage
from ...domain import extract_frame_index


def _frame_files(folder: Path, allowed) -> list[Path]:
    """The files of ``folder`` whose suffix (case-folded) is in ``allowed``, by name, as resolving each would name
    them: one listing (``os.scandir``) and the folder resolved once, not two or three calls to the file system per
    file — a folder of a long run holds tens of thousands of frames."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(root)
    resolved = root.resolve()
    found = []
    with os.scandir(root) as listing:
        for item in listing:
            if os.path.splitext(item.name)[1].casefold() not in allowed:
                continue
            try:
                if not item.is_file():
                    continue
                link = item.is_symlink()
            except OSError:
                continue
            found.append(Path(item.path).resolve() if link else resolved / item.name)
    found.sort(key=lambda path: path.name.casefold())
    return found


class LocalPredictionFileCatalog:
    def stack_paths(self, start_path: Path, count: int) -> tuple[Path, ...]:
        start = Path(start_path)
        requested = max(1, int(count))
        if requested == 1:
            return (start,)
        suffix = start.suffix.casefold()  # a stack is frames of one format: that of its first frame
        files = sorted(
            (
                path
                for path in start.parent.iterdir()
                if path.is_file() and path.suffix.casefold() == suffix
            ),
            key=lambda path: path.name,
        )
        try:
            index = files.index(start)
        except ValueError as exc:
            raise FileNotFoundError(f"Stack start file is not in its folder: {start}") from exc
        selected = tuple(files[index:index + requested])
        if len(selected) != requested:
            raise ValueError(
                f"Requested {requested} stack files from {start.name}, found {len(selected)}"
            )
        return selected

    def numbered_files(
        self, folder: Path, suffix: str | tuple[str, ...] = ".cbf"
    ) -> tuple[IndexedPredictionFile, ...]:
        suffixes = tuple(item.casefold() for item in ((suffix,) if isinstance(suffix, str) else suffix))
        indexed = []
        for path in _frame_files(Path(folder), set(suffixes)):
            index = extract_frame_index(path.name, suffixes)
            if index is not None:
                indexed.append(IndexedPredictionFile(path, index))
        return tuple(indexed)

    def compatible_files(
        self, folder: Path, suffixes: tuple[str, ...]
    ) -> tuple[Path, ...]:
        allowed = {
            suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
            for suffix in suffixes
        }
        return tuple(_frame_files(Path(folder), allowed))

    def look(self, folder: Path, suffixes: tuple[str, ...], skip: frozenset = frozenset()) -> tuple[FolderEntry, ...]:
        """Every frame file of ``folder`` but those in ``skip`` (frames the watch has taken), with its size and
        time, and whether an EDF says it is whole (Watch for new frames). One directory listing (``os.scandir`` gives
        the sizes with it), and only the files not skipped are opened (an EDF's header): a folder of a long run holds
        tens of thousands of frames. Paths as ``compatible_files`` gives them. ``OSError`` when the folder cannot be
        listed (a share that dropped, a folder moved away)."""
        root = Path(folder).resolve()
        allowed = {suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}" for suffix in suffixes}
        entries = []
        prefix = os.path.join(str(root), "")  # (``os.path.join(root, name)`` without its cost for every name)
        with os.scandir(root) as listing:
            for item in listing:
                suffix = _suffix(item.name).casefold()
                if suffix not in allowed:
                    continue
                path = prefix + item.name
                if path in skip:
                    continue
                try:
                    if not item.is_file():
                        continue
                    state = item.stat()
                except OSError:  # its writer holds it: seen at the next look
                    continue
                complete = edf_written(Path(path)) if suffix == ".edf" and state.st_size else None
                entries.append(FolderEntry(path, (state.st_size, state.st_mtime_ns), complete))
        return tuple(entries)


def _suffix(name: str) -> str:
    """``os.path.splitext(name)[1]`` of a file name (no folder): the last dot on, when something else than dots comes
    before it. A watch asks for every file of the folder at every look: 52 000 names took 0.13 s, now 0.05 s."""
    dot = name.rfind(".")
    return "" if dot < 0 or not name[:dot].strip(".") else name[dot:]


class FabioPredictionImageRepository:
    def load(self, paths: tuple[Path, ...]) -> LoadedPredictionImage:
        if not paths:
            raise ValueError("At least one detector image path is required")
        import fabio

        summed = None
        for path in paths:
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError(source)
            image = np.asarray(fabio.open(str(source)).data, dtype=np.float32)
            if image.ndim != 2 or image.size == 0:
                raise ValueError(f"Detector image must be a non-empty 2D array: {source}")
            if summed is None:
                summed = np.array(image, dtype=np.float32, copy=True)
            else:
                if image.shape != summed.shape:
                    raise ValueError(
                        f"Detector stack shape mismatch: {image.shape} != {summed.shape}"
                    )
                summed += image
        return LoadedPredictionImage(summed, tuple(Path(path) for path in paths))
