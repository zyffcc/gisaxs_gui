"""Prediction file discovery and sequence commands."""

from __future__ import annotations

from pathlib import Path


class PredictionFileViewModel:
    def __init__(self, *, numbered_files, files, sequence_rules, on_error):
        self._numbered_files = numbered_files
        self._files = files
        self._sequence_rules = sequence_rules
        self._on_error = on_error

    def discover_numbered_files(self, folder: Path, suffix: str = ".cbf"):
        try:
            return self._numbered_files.execute(Path(folder), suffix)
        except Exception as exc:
            self._on_error(str(exc))
            return ()

    def discover_files(self, folder: Path, suffixes: tuple[str, ...]):
        try:
            return self._files.execute(Path(folder), suffixes)
        except Exception as exc:
            self._on_error(str(exc))
            return ()

    def file_index(self, file_name: str):
        return self._sequence_rules.file_index(file_name)

    def index_range(self, text: str):
        return self._sequence_rules.index_range(text)

    def complete_batches(self, paths, batch_size: int):
        return self._sequence_rules.complete_batches(paths, batch_size)
