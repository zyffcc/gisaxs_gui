"""In-memory UserPreferencesRepository (the real one lives in ``user_store``)."""

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

    def reset(self) -> None:
        self._values = {}
