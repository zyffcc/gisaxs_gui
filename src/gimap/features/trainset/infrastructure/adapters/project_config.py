"""Existing YAML/JSON trainset config repository adapter。"""

from pathlib import Path


class LocalTrainsetConfigRepository:
    def load(self, path: Path):
        from .configuration import load_project_config

        return load_project_config(path)

    def save(self, config, path: Path):
        from .configuration import save_project_config

        return save_project_config(config, path)
