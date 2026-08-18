"""Framework-neutral user-interface preference persistence port."""

from __future__ import annotations

from typing import Any, Protocol


class UserPreferencesRepository(Protocol):
    """Flat key/value preferences stored independently from scientific settings."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...

    def save(self) -> None: ...
