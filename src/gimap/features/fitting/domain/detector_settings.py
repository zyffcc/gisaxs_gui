"""Typed detector settings used by the fitting presentation and application."""

from __future__ import annotations

from dataclasses import dataclass


HC_KEV_NM = 1.239841984


def wavelength_to_energy(wavelength_nm: float) -> float:
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive.")
    return HC_KEV_NM / float(wavelength_nm)


def energy_to_wavelength(energy_kev: float) -> float:
    if energy_kev <= 0:
        raise ValueError("Energy must be positive.")
    return HC_KEV_NM / float(energy_kev)


@dataclass(frozen=True)
class DetectorSettings:
    distance: float = 2000.0
    grazing_angle: float = 0.4
    wavelength: float = 0.015
    beam_center_x: float = 50.0
    beam_center_y: float = 50.0
    pixel_size_x: float = 172.0
    pixel_size_y: float = 172.0
    show_q_axis: bool = False

    @property
    def energy(self) -> float:
        return wavelength_to_energy(self.wavelength)
