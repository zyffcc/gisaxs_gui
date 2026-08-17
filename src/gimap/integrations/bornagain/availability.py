"""BornAgain runtime availability model。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .version import BornAgainVersion


AvailabilityState = Literal["available", "not_installed", "broken", "unsupported"]


@dataclass(frozen=True)
class BornAgainAvailability:
    state: AvailabilityState
    message: str
    version: BornAgainVersion | None = None
    module_path: str = ""

    @property
    def available(self) -> bool:
        return self.state == "available"

    @classmethod
    def from_dict(cls, payload: dict) -> "BornAgainAvailability":
        raw_version = payload.get("version")
        version = BornAgainVersion.parse(raw_version) if raw_version else None
        return cls(
            state=payload["state"],
            message=str(payload.get("message", "")),
            version=version,
            module_path=str(payload.get("module_path", "")),
        )
