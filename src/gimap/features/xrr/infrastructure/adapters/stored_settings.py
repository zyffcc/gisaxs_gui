"""XRR adapters over the shared settings and preferences: the last calibration, the last folder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...application import XrrCalibrationGeometry

LAST_FOLDER_KEY = "xrr.last_folder"


def _number(value: Any, *, positive: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _shape(value: Any) -> tuple[int, int] | None:
    try:
        rows, columns = (int(item) for item in value)
    except (TypeError, ValueError):
        return None
    return (rows, columns) if rows > 0 and columns > 0 else None


class SettingsXrrGeometryAdapter:
    """Reads what Geometry Calibration's Apply stored (``detector.*`` in mm, µm and numpy pixel
    indices, ``beam.energy_kev``); only after a calibration was applied (``system.geometry_calibration``),
    so the built-in detector defaults are never shown as a calibration."""

    def __init__(self, settings):
        self.settings = settings

    def last_calibration(self) -> XrrCalibrationGeometry | None:
        info = self.settings.get("system", "geometry_calibration", None)
        if not isinstance(info, dict):
            return None

        def detector(key: str, *, positive: bool = False) -> float | None:
            return _number(self.settings.get("detector", key, None), positive=positive)

        geometry = XrrCalibrationGeometry(
            distance_mm=detector("distance", positive=True),
            energy_kev=_number(self.settings.get("beam", "energy_kev", None), positive=True),
            pixel_size_x_um=detector("pixel_size_x", positive=True),
            pixel_size_y_um=detector("pixel_size_y", positive=True),
            beam_center_x_px=detector("beam_center_x"),
            beam_center_y_px=detector("beam_center_y"),
            source_image=str(info.get("source_image") or ""),
            timestamp=str(info.get("timestamp") or ""),
            image_shape=_shape(info.get("image_shape")),
            detector=str(info.get("detector") or ""),
        )
        values = (
            geometry.distance_mm, geometry.energy_kev, geometry.pixel_size_x_um,
            geometry.pixel_size_y_um, geometry.beam_center_x_px, geometry.beam_center_y_px,
        )
        return geometry if any(value is not None for value in values) else None


class PreferencesXrrFolderAdapter:
    """The folder of the last series the XRR window read, in the user preferences."""

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


__all__ = ["LAST_FOLDER_KEY", "PreferencesXrrFolderAdapter", "SettingsXrrGeometryAdapter"]
