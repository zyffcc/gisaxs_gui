"""SessionRepository 的内存与本地 JSON 实现。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class InMemorySessionRepository:
    def __init__(self, initial: dict[str, Any] | None = None):
        self._state = deepcopy(initial)

    def load(self) -> dict[str, Any] | None:
        return deepcopy(self._state)

    def save(self, state: dict[str, Any]) -> None:
        self._state = deepcopy(state)

    def clear(self) -> None:
        self._state = None


class JsonSessionRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Session JSON must contain an object.")
        return payload

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
