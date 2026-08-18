"""Framework-neutral in-situ record persistence commands."""

from __future__ import annotations

from pathlib import Path

from .ports.insitu import InSituRecordRepository


class ManageInSituRecords:
    def __init__(self, repository: InSituRecordRepository):
        self._repository = repository

    def cache_directory(self) -> Path:
        return self._repository.cache_directory()

    def session_path(self) -> Path:
        return self._repository.session_path()

    def ensure_directory(self) -> Path:
        return self._repository.ensure_directory()

    def reset(self) -> None:
        self._repository.reset()

    def append(self, record) -> None:
        self._repository.append(dict(record))

    def load(self) -> list[dict[str, object]]:
        return self._repository.load()

    def export_csv(self, path: Path, rows) -> Path:
        if not rows:
            raise ValueError("In-situ record export requires at least one row")
        return self._repository.export_csv(
            Path(path),
            tuple(dict(row) for row in rows),
        )
