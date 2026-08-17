"""BornAgain version value object。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class BornAgainVersion:
    major: int
    minor: int
    patch: int = 0
    raw: str = ""

    @classmethod
    def parse(cls, value) -> "BornAgainVersion":
        raw = str(value or "").strip()
        match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", raw)
        if match is None:
            raise ValueError(f"Cannot parse BornAgain version: {raw or '<empty>'}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3) or 0),
            raw=raw,
        )

    @property
    def supported(self) -> bool:
        return (self.major, self.minor) == (24, 1)

    def __str__(self) -> str:
        return self.raw or f"{self.major}.{self.minor}.{self.patch}"
