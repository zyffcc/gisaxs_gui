"""Import metadata adapter for optional Fitting dependencies."""

from __future__ import annotations

import importlib.util


class ImportlibFittingDependencyAvailabilityAdapter:
    def is_available(self, distribution: str) -> bool:
        try:
            return importlib.util.find_spec(str(distribution)) is not None
        except (ImportError, ValueError):
            return False
