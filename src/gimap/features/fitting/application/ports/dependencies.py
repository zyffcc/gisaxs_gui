"""Optional Fitting runtime availability port."""

from __future__ import annotations

from typing import Protocol


class FittingDependencyAvailabilityPort(Protocol):
    def is_available(self, distribution: str) -> bool: ...
