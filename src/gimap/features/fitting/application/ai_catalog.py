"""Application API for AI fitting profiles and model artifacts."""

from __future__ import annotations

from pathlib import Path

from .ports import AiFittingCatalogPort


class AiFittingCatalog:
    def __init__(self, catalog: AiFittingCatalogPort):
        self._catalog = catalog

    @property
    def default_profile_name(self) -> str:
        return self._catalog.default_profile_name

    def profile_names(self) -> tuple[str, ...]:
        return self._catalog.profile_names()

    def has_profile(self, name: str) -> bool:
        return str(name) in self.profile_names()

    def profile(self, name: str | None = None):
        return self._catalog.profile(name or self.default_profile_name)

    def default_model_directories(self, root: Path | None = None):
        """The built-in model folders, inside the GIMaP folder unless another root is given."""
        return self._catalog.default_model_directories(None if root is None else Path(root))

    def resolve_model_directory(self, path: Path | str) -> Path:
        """A stored model folder as an absolute path (a relative one is inside the GIMaP folder)."""
        return self._catalog.resolve_model_directory(path)

    def discover_models(self, directories):
        return self._catalog.discover_models(tuple(Path(path) for path in directories))

    def discover_model(self, path: Path):
        return self._catalog.discover_model(Path(path))
