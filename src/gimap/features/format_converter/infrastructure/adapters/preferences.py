"""The Format Converter's last input folder, in the user preferences."""

from __future__ import annotations

from pathlib import Path

LAST_FOLDER_KEY = "format_converter.last_folder"


class PreferencesConverterFolderAdapter:
    """``ConverterInputFolderPort`` over the ``UserPreferencesRepository`` of the app context."""

    def __init__(self, preferences, key: str = LAST_FOLDER_KEY):
        self.preferences = preferences
        self.key = key

    def last_folder(self) -> str:
        try:
            folder = str(self.preferences.get(self.key, "") or "")
        except Exception:
            return ""
        return folder if folder and Path(folder).is_dir() else ""

    def remember(self, path: str | Path) -> None:
        if not path:
            return
        location = Path(path)
        folder = location if location.is_dir() else location.parent
        try:
            self.preferences.set(self.key, str(folder))
            self.preferences.save()
        except OSError:
            pass


__all__ = ["LAST_FOLDER_KEY", "PreferencesConverterFolderAdapter"]
