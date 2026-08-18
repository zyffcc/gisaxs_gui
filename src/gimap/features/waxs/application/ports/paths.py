"""WAXS workspace path operations required by application commands."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class WaxsPathPort(Protocol):
    def normalize(self, path: str | Path) -> str: ...

    def current_directory(self) -> str: ...

    def is_directory(self, path: str | Path) -> bool: ...
