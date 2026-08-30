"""Port for portable WAXS JSON configurations."""

from pathlib import Path
from typing import Any, Protocol


class WaxsConfigurationPort(Protocol):
    def load(self, path: Path) -> dict[str, Any]: ...

    def save(self, path: Path, values: dict[str, Any]) -> Path: ...
