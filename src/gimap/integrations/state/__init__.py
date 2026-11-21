"""AppContext state port adapters: the per-user store, session and profiles.

The names load on first use (PEP 562): the launcher reads where the user folder is (``user_store``) before
the heavy imports, and the instrument profiles pull in NumPy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "InMemoryInstrumentProfileRepository": ".instrument_profiles",
    "JsonInstrumentProfileRepository": ".instrument_profiles",
    "InMemoryUserPreferencesRepository": ".preferences",
    "JsonProjectParametersRepository": ".project_parameters",
    "InMemorySessionRepository": ".session",
    "JsonSessionRepository": ".session",
    "InMemorySettingsRepository": ".settings",
    "JsonSettingsRepository": ".settings",
    "StorePreferencesRepository": ".user_store",
    "StoreSettingsRepository": ".user_store",
    "UserStore": ".user_store",
    "explicit_home": ".user_store",
    "migrate_legacy_files": ".user_store",
    "migrate_once": ".user_store",
    "user_data_dir": ".user_store",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    owner = _EXPORTS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(owner, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
