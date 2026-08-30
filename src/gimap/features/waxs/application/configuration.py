"""WAXS portable configuration use cases."""

from pathlib import Path

from .ports import WaxsConfigurationPort


class LoadWaxsConfiguration:
    def __init__(self, repository: WaxsConfigurationPort):
        self._repository = repository

    def execute(self, path: Path) -> dict:
        return self._repository.load(Path(path))


class SaveWaxsConfiguration:
    def __init__(self, repository: WaxsConfigurationPort):
        self._repository = repository

    def execute(self, path: Path, values: dict) -> Path:
        return self._repository.save(Path(path), dict(values))
