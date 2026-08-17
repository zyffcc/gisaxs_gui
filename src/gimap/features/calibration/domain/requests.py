"""Calibration 的 framework-neutral 请求对象。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import DetectorImage


@dataclass(frozen=True)
class CalibrationRequest:
    image: DetectorImage
    energy_kev: float
    standard_key: str = "auto"
    estimated_distance_mm: float | None = None
    distance_range_mm: tuple[float, float] = (30.0, 10_000.0)
    pixel_size_x_m: float | None = None
    pixel_size_y_m: float | None = None
    subtract_background: bool = True

    def algorithm_options(self) -> dict:
        return {
            "energy_kev": self.energy_kev,
            "standard_key": self.standard_key,
            "estimated_distance_mm": self.estimated_distance_mm,
            "distance_range_mm": self.distance_range_mm,
            "pixel_size_x_m": self.pixel_size_x_m,
            "pixel_size_y_m": self.pixel_size_y_m,
            "subtract_background": self.subtract_background,
        }
