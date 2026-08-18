"""Fitting parameter file commands."""

from __future__ import annotations

from pathlib import Path

from .ports.parameter_files import FittingParameterFileRepository


class ManageFittingParameterFiles:
    def __init__(self, repository: FittingParameterFileRepository):
        self._repository = repository

    def save_snapshot(self, path: Path, values) -> Path:
        return self._repository.save_snapshot(Path(path), dict(values))

    def load_snapshot(self, path: Path) -> dict[str, object]:
        values = self._repository.load_snapshot(Path(path))
        if not isinstance(values, dict):
            raise ValueError("Fitting parameter snapshot must contain an object")
        return values

    def export_model_parameters(self, source: Path, destination: Path) -> Path:
        return self._repository.copy(Path(source), Path(destination))

    def import_model_parameters(self, source: Path, destination: Path) -> Path:
        return self._repository.copy(Path(source), Path(destination))
