"""Fabio detector image repository。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...application import IndexedPredictionFile, LoadedPredictionImage
from ...domain import extract_cbf_index


class LocalPredictionFileCatalog:
    def stack_paths(self, start_path: Path, count: int) -> tuple[Path, ...]:
        start = Path(start_path)
        requested = max(1, int(count))
        if requested == 1:
            return (start,)
        files = sorted(
            (
                path
                for path in start.parent.iterdir()
                if path.is_file() and path.suffix.casefold() == ".cbf"
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
        self, folder: Path, suffix: str = ".cbf"
    ) -> tuple[IndexedPredictionFile, ...]:
        root = Path(folder)
        normalized_suffix = suffix.casefold()
        if not root.is_dir():
            raise NotADirectoryError(root)
        indexed = []
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.suffix.casefold() != normalized_suffix:
                continue
            index = extract_cbf_index(path.name)
            if index is not None:
                indexed.append(IndexedPredictionFile(path.resolve(), index))
        return tuple(indexed)

    def compatible_files(
        self, folder: Path, suffixes: tuple[str, ...]
    ) -> tuple[Path, ...]:
        root = Path(folder)
        if not root.is_dir():
            raise NotADirectoryError(root)
        allowed = {
            suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
            for suffix in suffixes
        }
        return tuple(
            path.resolve()
            for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
            if path.is_file() and path.suffix.casefold() in allowed
        )


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
