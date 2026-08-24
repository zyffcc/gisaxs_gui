"""Stable X-ray reflectometry geometry and ROI extraction primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


HC_KEV_ANGSTROM = 12.398419843320026


@dataclass(frozen=True)
class SpecularGeometry:
    """Fixed beam/detector geometry for a sample-angle scan."""

    distance_m: float
    energy_kev: float
    pixel_size_x_m: float
    pixel_size_y_m: float
    beam_center_x_px: float
    beam_center_y_px: float
    vertical_direction: int = -1

    def __post_init__(self) -> None:
        if self.distance_m <= 0:
            raise ValueError("Detector distance must be greater than zero.")
        if self.energy_kev <= 0:
            raise ValueError("Beam energy must be greater than zero.")
        if self.pixel_size_x_m <= 0 or self.pixel_size_y_m <= 0:
            raise ValueError("Detector pixel sizes must be greater than zero.")
        if self.vertical_direction not in (-1, 1):
            raise ValueError("Specular direction must be either -1 or +1.")

    @property
    def wavelength_angstrom(self) -> float:
        return HC_KEV_ANGSTROM / self.energy_kev


@dataclass(frozen=True)
class RoiMeasurement:
    intensity: float
    valid_pixels: int
    center_x_px: float
    center_y_px: float
    bounds: tuple[int, int, int, int]


def qz_from_theta(theta_deg: float, energy_kev: float) -> float:
    """Return specular qz in inverse Angstrom for sample angle theta."""

    if energy_kev <= 0:
        raise ValueError("Beam energy must be greater than zero.")
    theta = math.radians(float(theta_deg))
    wavelength = HC_KEV_ANGSTROM / float(energy_kev)
    return float(4.0 * math.pi * math.sin(theta) / wavelength)


def specular_pixel(theta_deg: float, geometry: SpecularGeometry) -> tuple[float, float]:
    """Locate the reflected beam for a fixed detector and rotating sample.

    Rotating the sample by ``theta`` moves the specular reflection through a
    detector scattering angle of ``2 theta`` relative to the direct beam.
    """

    scattering_angle = math.radians(2.0 * float(theta_deg))
    offset_y = geometry.distance_m * math.tan(scattering_angle) / geometry.pixel_size_y_m
    return (
        float(geometry.beam_center_x_px),
        float(geometry.beam_center_y_px + geometry.vertical_direction * offset_y),
    )


def extract_circular_roi(
    image: np.ndarray,
    invalid_mask: np.ndarray | None,
    center_x_px: float,
    center_y_px: float,
    radius_px: int,
    aggregation: str,
) -> RoiMeasurement:
    """Extract one finite circular ROI without copying the full detector image."""

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D detector image, got shape {values.shape}.")
    radius = int(radius_px)
    if radius < 0:
        raise ValueError("ROI radius cannot be negative.")
    mode = str(aggregation).casefold()
    if mode not in {"sum", "mean"}:
        raise ValueError("ROI aggregation must be 'sum' or 'mean'.")

    height, width = values.shape
    if radius == 0:
        x0 = int(math.floor(center_x_px + 0.5))
        y0 = int(math.floor(center_y_px + 0.5))
        x1, y1 = x0 + 1, y0 + 1
    else:
        x0 = math.floor(center_x_px - radius)
        x1 = math.ceil(center_x_px + radius) + 1
        y0 = math.floor(center_y_px - radius)
        y1 = math.ceil(center_y_px + radius) + 1
    clipped = (
        max(0, x0),
        max(0, y0),
        min(width, x1),
        min(height, y1),
    )
    left, top, right, bottom = clipped
    if left >= right or top >= bottom:
        return RoiMeasurement(float("nan"), 0, center_x_px, center_y_px, clipped)

    region = values[top:bottom, left:right]
    yy, xx = np.ogrid[top:bottom, left:right]
    if radius == 0:
        selected = np.ones(region.shape, dtype=bool)
    else:
        selected = (xx - center_x_px) ** 2 + (yy - center_y_px) ** 2 <= radius**2
    selected &= np.isfinite(region)
    if invalid_mask is not None:
        mask = np.asarray(invalid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("Detector invalid mask must have the same shape as the image.")
        selected &= ~mask[top:bottom, left:right]
    finite_values = np.asarray(region[selected], dtype=np.float64)
    if finite_values.size == 0:
        intensity = float("nan")
    elif mode == "mean":
        intensity = float(np.mean(finite_values))
    else:
        intensity = float(np.sum(finite_values))
    return RoiMeasurement(intensity, int(finite_values.size), center_x_px, center_y_px, clipped)


__all__ = [
    "HC_KEV_ANGSTROM",
    "RoiMeasurement",
    "SpecularGeometry",
    "extract_circular_roi",
    "qz_from_theta",
    "specular_pixel",
]
