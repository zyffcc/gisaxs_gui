"""The user's theme, font size and interface language: applied live and kept in the preferences."""

from __future__ import annotations

from src.gimap.app.ports import UserPreferencesRepository

from .engine import apply_theme, theme_manager
from .tokens import DEFAULT_FONT_PT, DEFAULT_MODE, normalized_font_pt, normalized_mode

THEME_KEY = "appearance.theme"
FONT_KEY = "appearance.font_pt"


class Appearance:
    def __init__(self, preferences: UserPreferencesRepository):
        self.preferences = preferences

    @property
    def mode(self) -> str:
        return normalized_mode(self.preferences.get(THEME_KEY, DEFAULT_MODE))

    @property
    def font_pt(self) -> float:
        return normalized_font_pt(self.preferences.get(FONT_KEY, DEFAULT_FONT_PT))

    @property
    def language(self) -> str:
        from ..i18n import LANGUAGE_KEY, normalized_language

        return normalized_language(self.preferences.get(LANGUAGE_KEY, "en"))

    def apply_saved(self) -> None:
        apply_theme(self.mode, self.font_pt)

    def apply_language(self, roots=()) -> str:
        """The saved language on ``roots`` now (and on windows and menus when they are shown)."""
        from ..i18n import apply_language

        return apply_language(self.language, roots)

    def set_language(self, language: str, roots=()) -> str:
        from ..i18n import LANGUAGE_KEY, apply_language, normalized_language

        language = normalized_language(language)
        self._remember(LANGUAGE_KEY, language)
        return apply_language(language, roots)

    def set_theme(self, mode: str) -> None:
        mode = normalized_mode(mode)
        self._remember(THEME_KEY, mode)
        if theme_manager().mode != mode or not theme_manager().applied:
            apply_theme(mode, self.font_pt)

    def set_font_pt(self, value: float) -> None:
        size = normalized_font_pt(value)
        self._remember(FONT_KEY, size)
        if theme_manager().font_pt != size or not theme_manager().applied:
            apply_theme(self.mode, size)

    def change_font(self, delta: float) -> None:
        self.set_font_pt(self.font_pt + float(delta))

    def reset_font(self) -> None:
        self.set_font_pt(DEFAULT_FONT_PT)

    def _remember(self, key: str, value) -> None:
        self.preferences.set(key, value)
        save = getattr(self.preferences, "save", None)
        if callable(save):
            save()


__all__ = ["Appearance", "FONT_KEY", "THEME_KEY"]
