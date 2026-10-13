"""Intensity corrections of a GIWAXS frame: solid angle, polarisation, absorption in the film.

What a pixel records is the true scattered intensity times a factor that depends on where the pixel
is; the corrected intensity is ``I / factor``. Solid angle and polarisation are 1 at the direct beam,
the film absorption at the specular exit angle, so corrected values stay close to counts per pixel.
They change only intensities, never q:

* **Solid angle** — a flat detector perpendicular to the direct beam, pixel at ``(X, Y)`` from the
  beam, distance ``D``, ``R² = D² + X² + Y²``: the pixel sees ``Ω ∝ D/R³ = cos³2θ / D²``; factor
  ``cos³2θ``.
* **Polarisation** — ``P = ½ [1 + cos²2θ − f · cos 2φ · sin²2θ]`` (pyFAI's convention), ``φ`` the
  azimuth on the detector from the horizontal, ``f`` the polarisation factor: about 0.95–0.99 for a
  synchrotron (horizontal), 0 for an unpolarised laboratory source.
* **Absorption in the film** — a film of thickness ``t`` and attenuation length ``L`` (1/µ), beam in
  at ``αi``, out at ``αf``: the scattering from depth ``z`` is attenuated by ``exp(−z·s/L)`` with
  ``s = 1/sin αi + 1/sin αf``, so the film scatters ``∝ (1 − exp(−t·s/L)) / s``; factor relative to
  ``αf = αi``. Refraction is not included (valid well above the critical angle); pixels at or below
  the sample horizon have no factor (they are left out).

For photon counts the variance of a corrected pixel is ``value / factor`` (see ``BinnedMean.add``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import exit_directions, region_displacements

BLOCK_ROWS = 512


def has_intensity_corrections(corrections) -> bool:
    return bool(
        getattr(corrections, "solid_angle", False)
        or getattr(corrections, "polarization", None) is not None
        or film_absorption_on(corrections)
    )


def film_absorption_on(corrections) -> bool:
    thickness = getattr(corrections, "film_thickness_nm", None)
    length = getattr(corrections, "attenuation_length_um", None)
    return bool(thickness and length and thickness > 0 and length > 0)


def _film(sin_exit: np.ndarray, sin_in: float, thickness_over_length: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):  # below the horizon: masked after
        path = 1.0 / sin_in + 1.0 / sin_exit
        return -np.expm1(-thickness_over_length * path) / path


def intensity_factor(
    shape: tuple[int, int],
    geometry: DetectorGeometry,
    *,
    solid_angle: bool = False,
    polarization: Optional[float] = None,
    film_thickness_nm: Optional[float] = None,
    attenuation_length_um: Optional[float] = None,
) -> np.ndarray:
    """The factor of every pixel (float32; NaN where it is not defined)."""
    rows, columns = int(shape[0]), int(shape[1])
    factor = np.ones((rows, columns), dtype=np.float32)
    distance = float(geometry.distance_m)
    film = bool(film_thickness_nm and attenuation_length_um and film_thickness_nm > 0 and attenuation_length_um > 0)
    sin_in = math.sin(geometry.incidence_rad)
    if film and sin_in <= 0:
        raise ValueError("The film absorption needs a grazing angle αi above 0.")
    ratio = float(film_thickness_nm or 0.0) * 1e-9 / (float(attenuation_length_um or 1.0) * 1e-6)
    reference = _film(np.array(sin_in), sin_in, ratio) if film else None
    for start in range(0, rows, BLOCK_ROWS):
        stop = min(rows, start + BLOCK_ROWS)
        x, y = region_displacements(geometry, (start, stop), (0, columns))
        in_plane = x * x + y * y
        cos_2theta = distance / np.sqrt(distance * distance + in_plane)
        block = np.ones_like(cos_2theta)
        if solid_angle:
            block *= cos_2theta**3
        if polarization is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                cos_2phi = np.where(in_plane > 0, (x * x - y * y) / in_plane, 0.0)
            sin2 = 1.0 - cos_2theta**2
            block *= 0.5 * (1.0 + cos_2theta**2 - float(polarization) * cos_2phi * sin2)
        if film:
            sin_exit = exit_directions(x, y, distance, geometry.incidence_rad)[2]
            relative = _film(sin_exit, sin_in, ratio) / reference
            block *= np.where(sin_exit > 0, relative, np.nan)
        factor[start:stop] = block
    return factor


def describe_intensity_corrections(corrections) -> Optional[dict]:
    """What was corrected and how, for the JSON record (``None``: nothing)."""
    if not has_intensity_corrections(corrections):
        return None
    record: dict = {"applied_as": "I / factor (1 at the direct beam; film absorption 1 at af = ai); q unchanged"}
    if corrections.solid_angle:
        record["solid_angle"] = "cos^3(2theta), flat detector perpendicular to the direct beam"
    if corrections.polarization is not None:
        record["polarization"] = {
            "factor": float(corrections.polarization),
            "formula": "0.5*(1 + cos^2(2theta) - f*cos(2phi)*sin^2(2theta)), phi from the horizontal (pyFAI)",
        }
    if film_absorption_on(corrections):
        record["film_absorption"] = {
            "thickness_nm": float(corrections.film_thickness_nm),
            "attenuation_length_um": float(corrections.attenuation_length_um),
            "formula": "(1 - exp(-t/L*(1/sin ai + 1/sin af))) / (1/sin ai + 1/sin af), relative to af = ai; "
                       "no refraction",
        }
    return record


__all__ = ["describe_intensity_corrections", "film_absorption_on", "has_intensity_corrections", "intensity_factor"]
