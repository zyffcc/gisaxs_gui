"""Use cases for loading and saving fitting detector settings."""

from __future__ import annotations

from src.gimap.app.ports import SettingsRepository

from ..domain.detector_settings import DetectorSettings
from ..domain.detector_q_grid import normalize_horizontal_q_axis


class LoadDetectorSettings:
    def __init__(self, repository: SettingsRepository):
        self._repository = repository

    def execute(self) -> DetectorSettings:
        get = self._repository.get
        return DetectorSettings(
            distance=float(get("fitting", "detector.distance", 2000.0)),
            grazing_angle=float(get("beam", "grazing_angle", 0.4)),
            wavelength=float(get("beam", "wavelength", 0.015)),
            beam_center_x=float(get("fitting", "detector.beam_center_x", 50.0)),
            beam_center_y=float(get("fitting", "detector.beam_center_y", 50.0)),
            pixel_size_x=float(get("fitting", "detector.pixel_size_x", 172.0)),
            pixel_size_y=float(get("fitting", "detector.pixel_size_y", 172.0)),
            show_q_axis=bool(get("fitting", "detector.show_q_axis", False)),
            horizontal_q_axis=normalize_horizontal_q_axis(
                get("fitting", "detector.horizontal_q_axis", "qy")
            ),
        )


class SaveDetectorSettings:
    def __init__(self, repository: SettingsRepository):
        self._repository = repository

    def execute(self, settings: DetectorSettings) -> None:
        detector_values = {
            "distance": settings.distance,
            "beam_center_x": settings.beam_center_x,
            "beam_center_y": settings.beam_center_y,
            "pixel_size_x": settings.pixel_size_x,
            "pixel_size_y": settings.pixel_size_y,
            "show_q_axis": settings.show_q_axis,
            "horizontal_q_axis": settings.horizontal_q_axis,
        }
        for key, value in detector_values.items():
            self._repository.set("fitting", f"detector.{key}", value)
        self._repository.set("beam", "grazing_angle", settings.grazing_angle)
        self._repository.set("beam", "wavelength", settings.wavelength)
        self._repository.save()
