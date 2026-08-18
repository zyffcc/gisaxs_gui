"""Stable X-ray energy/wavelength conversion used by detector readers."""

from __future__ import annotations

import math


HC_KEV_ANGSTROM = 12.398419843320026


def energy_to_wavelength(energy_kev: float) -> float:
    energy = float(energy_kev)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("X-ray energy must be greater than zero.")
    return HC_KEV_ANGSTROM / energy
