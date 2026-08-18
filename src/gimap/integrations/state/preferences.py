"""Adapters for the flat legacy ``user_settings.json`` preference store."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class InMemoryUserPreferencesRepository:
    """File-free preferences for tests and headless feature composition."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._values = deepcopy(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return deepcopy(self._values.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self._values[key] = deepcopy(value)

    def save(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._values)


class LegacyUserPreferencesRepository:
    """Expose the existing UserSettings object through the application port."""

    def __init__(self, manager):
        self._manager = manager

    def get(self, key: str, default: Any = None) -> Any:
        return self._manager.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._manager.set(key, value)

    def save(self) -> None:
        self._manager.save_settings()
