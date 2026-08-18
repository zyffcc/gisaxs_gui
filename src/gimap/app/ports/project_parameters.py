"""Port for the application-wide parameter snapshot file."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol


class ProjectParametersRepository(Protocol):
    def load(self, path: str | Path) -> dict: ...

    def save(self, path: str | Path, values: Mapping) -> Path: ...
