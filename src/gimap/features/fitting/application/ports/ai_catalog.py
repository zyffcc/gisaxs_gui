"""Port for AI fitting profiles and model discovery metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol


class AiFittingCatalogPort(Protocol):
    @property
    def default_profile_name(self) -> str: ...

    def profile_names(self) -> tuple[str, ...]: ...

    def profile(self, name: str): ...

    def default_model_directories(self, root: Path) -> tuple[Path, ...]: ...

    def discover_models(self, directories: Iterable[Path]) -> tuple: ...

    def discover_model(self, path: Path) -> tuple: ...
