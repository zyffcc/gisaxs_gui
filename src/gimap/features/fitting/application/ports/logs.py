"""Fitting user-visible log storage port."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FittingLogRepository(Protocol):
    def save(self, path: Path, content: str) -> Path: ...
