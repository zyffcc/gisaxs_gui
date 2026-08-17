"""AI candidate 持久化边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class CandidateRepository(Protocol):
    def load(self, output_dir: Path) -> tuple[dict[str, Any], ...]:
        ...
