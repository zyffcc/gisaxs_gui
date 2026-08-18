"""Fitting parameter snapshot storage port."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol


class FittingParameterFileRepository(Protocol):
    def save_snapshot(self, path: Path, values: Mapping[str, object]) -> Path: ...

    def load_snapshot(self, path: Path) -> dict[str, object]: ...

    def copy(self, source: Path, destination: Path) -> Path: ...
