"""The single per-user store: settings, preferences and where user files live.

Everything a user changes is kept in one versioned JSON document in the user
data folder (``%APPDATA%/GIMaP`` on Windows, ``~/.config/gimap`` elsewhere,
or ``$GIMAP_HOME``), next to the instrument profiles and the last session.
The program folder therefore stays read-only and unchanged by use, and
unpacking a new release keeps the user's settings.

Files written before this store existed (``config/user_parameters.json``,
``config/user_settings.json``, ``config/instrument_profiles.json``,
``.gimap_cache/session.json``) are imported once and never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .default_settings import default_preferences, default_settings

SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[4]

SETTINGS_FILE = "settings.json"
PROFILES_FILE = "instrument_profiles.json"
SESSION_FILE = "session.json"
MODEL_PARAMETERS_FILE = "model_parameters.json"

OBSOLETE_PREFERENCES = frozenset(
    {
        "enable_adaptive_scaling",
        "window_width",
        "window_height",
        "font_adjustment",
        "visual_font_scale",
        "responsive_layout_mode",
        "responsive_resize_on_start",
        "responsive_font_enabled",
        "auto_detect_monitor_dpi",
        "adaptive_layout_enabled",
        "manual_screen_resolution",
        "layout_target_mode",
        "layout_target_custom",
        "auto_fit_layout_target",
    }
)
"""Window-scaling preferences of the removed custom scaling code."""


def user_data_dir() -> Path:
    override = os.environ.get("GIMAP_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "GIMaP"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "gimap"


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge; values from ``overlay`` win, nested mappings are merged."""
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write via a temporary file and rename, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_json_object(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _get_nested(values: Mapping[str, Any], key: str, default: Any) -> Any:
    current: Any = values
    for segment in key.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return default
        current = current[segment]
    return deepcopy(current)


def _set_nested(values: dict[str, Any], key: str, value: Any) -> None:
    segments = key.split(".")
    current = values
    for segment in segments[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            child = {}
            current[segment] = child
        current = child
    current[segments[-1]] = deepcopy(value)


class UserStore:
    """``{"schema_version", "settings": {section: {...}}, "preferences": {...}}``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.settings: dict[str, dict[str, Any]] = {}
        self.preferences: dict[str, Any] = {}
        self.migrated_from: Optional[dict] = None
        self.reload()

    def reload(self) -> None:
        payload = _read_json_object(self.path) if self.path.is_file() else None
        if payload is not None and int(payload.get("schema_version", 0)) > SCHEMA_VERSION:
            raise ValueError(
                f"{self.path} was written by a newer GIMaP (schema {payload['schema_version']})."
            )
        payload = payload or {}
        stored_settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
        stored_preferences = (
            payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {}
        )
        self.settings = deep_merge(default_settings(), stored_settings)
        self.preferences = deep_merge(default_preferences(), stored_preferences)
        self.migrated_from = payload.get("migrated_from")

    def save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "settings": self.settings,
            "preferences": self.preferences,
        }
        if self.migrated_from:
            payload["migrated_from"] = self.migrated_from
        atomic_write_json(self.path, payload)

    def reset_settings(self) -> None:
        self.settings = default_settings()

    def reset_preferences(self) -> None:
        self.preferences = default_preferences()


class StoreSettingsRepository:
    """``SettingsRepository`` port over the ``settings`` part of a :class:`UserStore`."""

    def __init__(self, store: UserStore):
        self.store = store

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return _get_nested(self.store.settings.get(section, {}), key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        _set_nested(self.store.settings.setdefault(section, {}), key, value)

    def get_section(self, section: str) -> dict[str, Any]:
        return deepcopy(self.store.settings.get(section, {}))

    def update_section(self, section: str, values: dict[str, Any]) -> None:
        self.store.settings.setdefault(section, {}).update(deepcopy(dict(values)))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.store.settings)

    def reload(self) -> None:
        self.store.reload()

    def save(self) -> None:
        self.store.save()

    def reset(self) -> None:
        self.store.reset_settings()


class StorePreferencesRepository:
    """``UserPreferencesRepository`` port over the ``preferences`` part of a store."""

    def __init__(self, store: UserStore):
        self.store = store

    def get(self, key: str, default: Any = None) -> Any:
        return deepcopy(self.store.preferences.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self.store.preferences[key] = deepcopy(value)

    def save(self) -> None:
        self.store.save()

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.store.preferences)

    def reset(self) -> None:
        self.store.reset_preferences()


def migrate_legacy_files(
    data_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    """Import the pre-store files once; returns what was imported.

    Nothing is imported when the store already exists, and legacy files are
    only read (they stay where they are, untouched).
    """
    data_dir = Path(data_dir)
    imported: list[str] = []
    store_path = data_dir / SETTINGS_FILE
    if not store_path.exists():
        legacy_parameters = project_root / "config" / "user_parameters.json"
        legacy_preferences = project_root / "config" / "user_settings.json"
        settings = _read_json_object(legacy_parameters) or {}
        preferences = {
            key: value
            for key, value in (_read_json_object(legacy_preferences) or {}).items()
            if key not in OBSOLETE_PREFERENCES
        }
        if settings or preferences:
            sources = [str(path) for path in (legacy_parameters, legacy_preferences) if path.is_file()]
            payload = {
                "schema_version": SCHEMA_VERSION,
                "settings": {key: value for key, value in settings.items() if isinstance(value, dict)},
                "preferences": preferences,
                "migrated_from": {
                    "files": sources,
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            }
            atomic_write_json(store_path, payload)
            imported.extend(sources)
    for legacy, name in (
        (project_root / "config" / PROFILES_FILE, PROFILES_FILE),
        (project_root / ".gimap_cache" / SESSION_FILE, SESSION_FILE),
        (project_root / "config" / MODEL_PARAMETERS_FILE, MODEL_PARAMETERS_FILE),
    ):
        target = data_dir / name
        if legacy.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, target)
            imported.append(str(legacy))
    return imported


__all__ = [
    "MODEL_PARAMETERS_FILE",
    "OBSOLETE_PREFERENCES",
    "PROFILES_FILE",
    "SCHEMA_VERSION",
    "SESSION_FILE",
    "SETTINGS_FILE",
    "StorePreferencesRepository",
    "StoreSettingsRepository",
    "UserStore",
    "atomic_write_json",
    "deep_merge",
    "migrate_legacy_files",
    "user_data_dir",
]
