"""The assistant's remembered choices (settings section ``assistant``)."""

from __future__ import annotations

from typing import Any, Optional

from src.gimap.app.presentation.i18n import current_language

from ..application import (
    BACKEND_CLAUDE_CODE,
    DEFAULT_EFFORT,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    GOALS,
    LANGUAGES,
    PERMISSION_PREVIEW,
)

SECTION = "assistant"
DEFAULTS: dict[str, Any] = {
    "backend": BACKEND_CLAUDE_CODE,
    "code_cli": "",
    "code_model": "",
    "model": DEFAULT_MODEL,
    "effort": DEFAULT_EFFORT,
    "permission": PERMISSION_PREVIEW,  # new users see the AI's changes as cards first
    "allow_images": False,
    "max_turns": DEFAULT_MAX_TURNS,
    "language": "English",
    "goals": list(GOALS),
    "ring_q": 0.0,
    "standing_instructions": "",
    # Other AI providers (BACKEND_PROVIDER): the preset key, the model per provider key, and
    # the person's own address per provider key (local servers, Azure, custom services).
    "provider": "deepseek",
    "provider_models": {},
    "provider_urls": {},
}


def read(settings, key: str) -> Any:
    if settings is None:
        return DEFAULTS[key]
    value = settings.get(SECTION, key, DEFAULTS[key])
    return DEFAULTS[key] if value is None else value


def report_language(settings, interface_language: Optional[str] = None) -> str:
    """The report language to preselect: the one chosen before, else 中文 when the interface is in Chinese."""
    stored = settings.get(SECTION, "language", None) if settings is not None else None
    if stored in LANGUAGES:
        return str(stored)
    if interface_language is None:
        interface_language = current_language()
    return "中文" if interface_language == "zh" else DEFAULTS["language"]


def write(settings, key: str, value: Any) -> None:
    if settings is None:
        return
    settings.set(SECTION, key, value)
    settings.save()


__all__ = ["DEFAULTS", "SECTION", "read", "report_language", "write"]
