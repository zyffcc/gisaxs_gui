"""The model folder each user chose for a prediction module, kept in the user data folder.

``modules/<module>/module.yaml`` belongs to the program: it is tracked in git and shared by everyone who
uses the installation. ``Import model...`` therefore never writes the chosen folder there. The choice is
kept in the user preferences (``settings.json`` in the user data folder, as the other per-user choices)
under one key, a mapping from the module's id (its folder name when it has none) to the folder. At load
time it overrides ``model.model_path`` of ``module.yaml``, which stays the default, for example for a
module that ships its model beside it or a ``module.yaml`` that already names a model folder.
"""

from __future__ import annotations

from typing import Any

PREFERENCES_KEY = "prediction.model_paths"


def _clean(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(path)
        for key, path in value.items()
        if str(key).strip() and isinstance(path, str) and path.strip()
    }


class PreferencesModelPathStore:
    """Model folders chosen by this user, in the user preferences (survives restarts and updates)."""

    def __init__(self, preferences, key: str = PREFERENCES_KEY):
        self._preferences = preferences
        self._key = key

    def _paths(self) -> dict[str, str]:
        try:
            return _clean(self._preferences.get(self._key, {}))
        except Exception:  # an unreadable store must not stop the module list
            return {}

    def get(self, module_key: str) -> str:
        return self._paths().get(str(module_key), "")

    def set(self, module_key: str, model_path: str) -> None:
        paths = self._paths()
        paths[str(module_key)] = str(model_path)
        self._preferences.set(self._key, paths)
        self._preferences.save()


class InMemoryModelPathStore:
    """Choices kept for this session only (tests, or a repository built without user preferences)."""

    def __init__(self, initial: dict[str, str] | None = None):
        self._paths = _clean(initial or {})

    def get(self, module_key: str) -> str:
        return self._paths.get(str(module_key), "")

    def set(self, module_key: str, model_path: str) -> None:
        self._paths[str(module_key)] = str(model_path)
