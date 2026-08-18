"""Fitting log commands."""

from __future__ import annotations

from pathlib import Path

from .ports.logs import FittingLogRepository


class SaveFittingLog:
    def __init__(self, repository: FittingLogRepository):
        self._repository = repository

    def execute(self, path: Path, content: str) -> Path:
        if not str(content):
            raise ValueError("Fitting log is empty")
        return self._repository.save(Path(path), str(content))
