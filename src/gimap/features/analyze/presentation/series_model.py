"""Series part of the Analyze view model: what a Series map is built with, and its own exports.

``series_built_with`` is taken when a map (or a Batch Export that fills it) starts: the numbers of the
listed frames, the Sum N, the cut of the series and the settings, so that the map still says what it was
built with after the frame on screen, the list or the settings changed. The exports of the map itself, of
its stages and of the peak followed through it are in ``export_model.py``.
"""

from __future__ import annotations

from pathlib import Path

from ..application import SeriesMap, settings_record
from ..application.series_map import export_series_curve, export_series_trace

VOLATILE = ("software", "format", "version", "export")
"""Fields of a settings record that do not change what a map holds."""
PROFILE_VOLATILE = ("source", "updated_at")
"""… and of its instrument profile: when and how it was saved (its geometry does change every q)."""


def settings_key(record: dict) -> dict:
    """A settings record without what does not change a map (two maps built alike compare equal)."""
    key = {name: value for name, value in dict(record or {}).items() if name not in VOLATILE}
    if isinstance(key.get("profile"), dict):
        key["profile"] = {name: value for name, value in key["profile"].items() if name not in PROFILE_VOLATILE}
    return key


class SeriesModelMixin:
    """Needs ``_export``, ``_profiles``, ``current_settings`` (with the pin: ``view_model.py``), ``frame_numbers``,
    ``listed_frames``, ``pin_series_cuts``, ``series_cut`` and ``series_cut_text`` (``frames_model.py``)."""

    def series_built_with(self) -> dict:
        """What a series run starting now is built with: ``numbers`` (``frame_numbers``), ``group`` (the Sum N),
        ``cut`` (``series_cut``), ``pinned`` (one cut for every frame), ``cut_shown`` (``series_cut_text``, for the
        interface), ``cut_text`` (the same sentence in English, for the exports), ``settings`` (the settings
        record) and ``listed`` (frames)."""
        template, values = self.series_cut_text()
        return {
            "numbers": self.frame_numbers(), "group": int(self.sum_count), "cut": self.series_cut(),
            "pinned": bool(self.pin_series_cuts),
            "cut_shown": (template, dict(values)), "cut_text": template.format(**values) if template else "",
            "settings": self._series_settings(), "listed": int(self.listed_frames()),
        }

    def _series_settings(self) -> dict:
        """The settings record of a series run started now. The profile as it is stored now: one saved again
        under its name (Geometry ▸ entered by hand) is what the next analysis uses, also before the frame on
        screen is analysed again."""
        record = settings_record(self.current_settings())
        profile = record.get("profile") or {}
        stored = self._profiles.find(profile["name"]) if profile.get("name") and self._profiles is not None else None
        if stored is not None:
            record["profile"] = stored.to_dict()
        return record


    def series_settings_changed(self, built: dict) -> bool:
        """The settings in force are no longer those ``built`` (``series_built_with``) was taken with."""
        before = settings_key((built or {}).get("settings") or {})
        return bool(before) and before != settings_key(self._series_settings())

    def write_series_curve(self, series: SeriesMap, row: int, path: Path) -> Path:
        """The curve of one row of the map as a table."""
        return export_series_curve(self._export.writer, series, int(row), Path(path))

    def write_series_trace(self, series: SeriesMap, name: str, values, path: Path, *, what: str) -> Path:
        """One value per frame (the trace shown under the map) as a table."""
        return export_series_trace(self._export.writer, series, name, values, Path(path), what=what)


__all__ = ["SeriesModelMixin", "settings_key"]
