"""Port for named detector geometries (instrument profiles)."""

from __future__ import annotations

from typing import Optional, Protocol

from src.gimap.shared.geometry import InstrumentProfile


class InstrumentProfileRepository(Protocol):
    def load_all(self) -> list[InstrumentProfile]: ...

    def save(self, profile: InstrumentProfile) -> None: ...

    def delete(self, name: str) -> bool: ...

    def find(self, name: str) -> Optional[InstrumentProfile]: ...

    def match(
        self, *, detector_name: Optional[str], shape: Optional[tuple[int, int]]
    ) -> Optional[InstrumentProfile]: ...
