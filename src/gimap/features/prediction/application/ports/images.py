"""Prediction detector image storage port。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..frame_watch import FolderEntry
from ..models import IndexedPredictionFile, LoadedPredictionImage


class PredictionImageRepository(Protocol):
    def load(self, paths: tuple[Path, ...]) -> LoadedPredictionImage: ...


class PredictionFileCatalog(Protocol):
    def stack_paths(self, start_path: Path, count: int) -> tuple[Path, ...]: ...

    def numbered_files(
        self, folder: Path, suffix: str | tuple[str, ...] = ".cbf"
    ) -> tuple[IndexedPredictionFile, ...]: ...

    def compatible_files(
        self, folder: Path, suffixes: tuple[str, ...]
    ) -> tuple[Path, ...]: ...

    def look(
        self, folder: Path, suffixes: tuple[str, ...], skip: frozenset = frozenset()
    ) -> tuple[FolderEntry, ...]: ...
