"""Framework-neutral project parameter file commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .ports import ProjectParametersRepository


@dataclass(frozen=True)
class LoadProjectParameters:
    repository: ProjectParametersRepository

    def execute(self, path: str | Path) -> dict:
        return self.repository.load(path)


@dataclass(frozen=True)
class SaveProjectParameters:
    repository: ProjectParametersRepository

    def execute(self, path: str | Path, values: Mapping) -> Path:
        return self.repository.save(path, values)
