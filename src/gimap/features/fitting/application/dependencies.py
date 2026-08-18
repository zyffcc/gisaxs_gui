"""Optional Fitting dependency query."""

from __future__ import annotations

from .ports.dependencies import FittingDependencyAvailabilityPort


class CheckFittingDependency:
    def __init__(self, availability: FittingDependencyAvailabilityPort):
        self._availability = availability

    def execute(self, distribution: str) -> bool:
        name = str(distribution).strip()
        if not name:
            raise ValueError("Dependency name is required")
        return self._availability.is_available(name)
