"""Analyze choices remembered between sessions (settings section ``analyze``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..application import MODES

SECTION = "analyze"


@dataclass(frozen=True)
class AnalyzePreferences:
    mode: str = "auto"
    profile_name: Optional[str] = None
    incidence_deg: Optional[float] = None
    last_folder: str = ""
    auto_export: bool = False


def load_preferences(settings: Any, profile_names: Sequence[str]) -> AnalyzePreferences:
    """Stored choices, ignoring values that no longer make sense (deleted profile, bad mode)."""
    if settings is None:
        return AnalyzePreferences()
    try:
        values = dict(settings.get_section(SECTION) or {})
    except Exception:
        return AnalyzePreferences()
    mode = values.get("mode") if values.get("mode") in MODES else "auto"
    profile = values.get("profile_name")
    profile = profile if profile in set(profile_names) else None
    incidence = values.get("incidence_deg")
    try:
        incidence = None if incidence is None else float(incidence)
    except (TypeError, ValueError):
        incidence = None
    return AnalyzePreferences(
        mode=mode,
        profile_name=profile,
        incidence_deg=incidence,
        last_folder=str(values.get("last_folder") or ""),
        auto_export=bool(values.get("auto_export", False)),
    )


def save_preferences(settings: Any, preferences: AnalyzePreferences) -> None:
    if settings is None:
        return
    settings.update_section(
        SECTION,
        {
            "mode": preferences.mode,
            "profile_name": preferences.profile_name,
            "incidence_deg": preferences.incidence_deg,
            "last_folder": preferences.last_folder,
            "auto_export": preferences.auto_export,
        },
    )
    settings.save()


__all__ = ["AnalyzePreferences", "SECTION", "load_preferences", "save_preferences"]
