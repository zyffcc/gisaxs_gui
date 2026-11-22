"""Intensity corrections of a GIWAXS frame: solid angle, polarisation, absorption in the film, Lorentz factor.

What a pixel records is the true scattered intensity times a factor that depends on where the pixel
is; the corrected intensity is ``I / factor``. Solid angle and polarisation are 1 at the direct beam,
the film absorption at the specular exit angle, so corrected values stay close to counts per pixel.
They change only intensities, never q:

* **Solid angle** — a flat detector, tilted or not (the rotations of a ``.poni`` are part of the
  geometry: ``q_mapping.lab_rays``), pixel at the vector ``t`` from the sample, direct beam at the
  distance ``D``: the pixel sees ``Ω ∝ 1/|t|³``; factor ``(D/|t|)³``, its solid angle relative to
  the pixel at the direct beam. Perpendicular to the beam (no rotations), with the pixel at
  ``(X, Y)`` from the beam, ``|t|² = D² + X² + Y²`` and the factor is ``cos³2θ``; on a tilted
  detector ``D/|t|`` is not ``cos 2θ`` and the factor is not ``cos³2θ``.
* **Polarisation** — ``P = ½ [1 + cos²2θ − f · cos 2φ · sin²2θ]`` (pyFAI's convention), ``2θ`` and
  ``φ`` those of the pixel's ray (of the tilted detector when it is tilted), ``φ`` the
  azimuth from the horizontal, ``f`` the polarisation factor: about 0.95–0.99 for a
  synchrotron (horizontal), 0 for an unpolarised laboratory source.
* **Absorption in the film** — a film of thickness ``d`` and attenuation length ``L`` (1/µ), beam in
  at ``αi``, out at ``αf``: the scattering from depth ``z`` is attenuated by ``exp(−z·s/L)`` with
  ``s = 1/sin αi + 1/sin αf``, so the film scatters ``∝ (1 − exp(−d·s/L)) / s``; factor relative to
  ``αf = αi``. Refraction is not included (valid well above the critical angle); pixels at or below
  the sample horizon have no factor (they are left out).
* **Lorentz factor of a fibre texture** (GIXSGUI, Jiang 2015, Eq. 5 with S = cos αi cos αf) — crystallites
  oriented at random about the surface normal: a reciprocal-lattice point at the polar angle χ from the normal
  lies on a circle about q_z, and the Ewald sphere cuts that circle at a rate set by the component of its motion
  along the scattered ray, ``k cos αi cos αf |sin 2θf|`` (``2θf`` the in-plane exit angle, ``αf`` the exit angle).
  This Jacobian weights detector sums for crystalline peak areas. It vanishes in inverse form throughout
  the plane of incidence (qy = 0). Ring orientation uses pixel means before this correction.

For photon counts the variance of a corrected pixel is ``value / factor`` (see ``BinnedMean.add``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import exit_directions, geometry_rays

BLOCK_ROWS = 512
LORENTZ_FORMULA = ("L = 1/(cos ai * cos af * |sin 2theta_f|), fibre texture about the surface normal (GIXSGUI, "
                   "Jiang 2015, Eq. 5, S = cos ai cos af); 1/L = cos(ai) |qy| / k")


def has_intensity_corrections(corrections) -> bool:
    return bool(
        getattr(corrections, "solid_angle", False)
        or getattr(corrections, "polarization", None) is not None
        or film_absorption_on(corrections)
        or getattr(corrections, "lorentz", False)
    )


def film_absorption_on(corrections) -> bool:
    thickness = getattr(corrections, "film_thickness_nm", None)
    length = getattr(corrections, "attenuation_length_um", None)
    return bool(thickness and length and thickness > 0 and length > 0)


def film_absorption_factor(sin_exit, sin_in: float, optical_depth: float) -> np.ndarray:
    """Integrated incident and exit path attenuation, relative to the specular direction."""
    if sin_in <= 0. or optical_depth < 0. or np.isnan(optical_depth):
        raise ValueError("Film absorption needs positive incidence and nonnegative optical depth.")
    sin_exit = np.asarray(sin_exit, dtype=float)
    valid = (sin_exit > 0.) & (sin_exit <= 1.)
    path = 1. / sin_in + 1. / np.where(valid, sin_exit, 1.)
    reference = 2. / sin_in
    if optical_depth == 0.:
        factor = np.ones_like(path)
    else:
        factor = (reference / path) * np.expm1(-optical_depth * path) / np.expm1(-optical_depth * reference)
    return np.where(valid, factor, np.nan)


def polarization_factor(along, x, y, polarization: float) -> np.ndarray:
    """pyFAI convention; x is the horizontal lab coordinate of the scattered ray."""
    # Equivalent to ½[1+cos²(2θ)−f cos(2φ) sin²(2θ)], also defined on the direct beam.
    return 1.0 - 0.5 * ((1 + polarization) * x*x + (1 - polarization) * y*y) / (along*along + x*x + y*y)


def intensity_factor(
    shape: tuple[int, int],
    geometry: DetectorGeometry,
    *,
    solid_angle: bool = False,
    polarization: Optional[float] = None,
    film_thickness_nm: Optional[float] = None,
    attenuation_length_um: Optional[float] = None,
    lorentz: bool = False,
    lorentz_band_px: float = 2.0,
) -> np.ndarray:
    """The factor of every pixel (float32; NaN or infinite where it is not defined)."""
    rows, columns = int(shape[0]), int(shape[1])
    factor = np.ones((rows, columns), dtype=np.float32)
    beam = float(geometry.distance_m)
    film = bool(film_thickness_nm and attenuation_length_um and film_thickness_nm > 0 and attenuation_length_um > 0)
    sin_in, cos_in = math.sin(geometry.incidence_rad), math.cos(geometry.incidence_rad)
    if film and sin_in <= 0:
        raise ValueError("The film absorption needs a grazing angle αi above 0.")
    ratio = float(film_thickness_nm or 0.0) * 1e-9 / (float(attenuation_length_um or 1.0) * 1e-6)
    for start in range(0, rows, BLOCK_ROWS):
        stop = min(rows, start + BLOCK_ROWS)
        along, x, y = geometry_rays(geometry, (start, stop), (0, columns))
        in_plane = x * x + y * y
        radius = np.sqrt(along * along + in_plane)
        cos_2theta = along / radius
        block = np.ones_like(cos_2theta)
        if solid_angle:  # (distance to the direct beam / distance to the pixel)³, also on a tilted detector
            block *= (beam / radius) ** 3
        if polarization is not None:
            block *= polarization_factor(along, x, y, float(polarization))
        if film or lorentz:
            _along, across, sin_exit = exit_directions(x, y, along, geometry.incidence_rad, sample_roll_rad=geometry.sample_roll_rad)
        if film:
            block *= film_absorption_factor(sin_exit, sin_in, ratio)
        if lorentz:
            # across = q_y/k. One horizontal pixel at the direct beam subtends p_x/D radians.
            minimum = float(lorentz_band_px) * geometry.pixel_size_x_m / beam
            with np.errstate(divide="ignore"):
                block *= np.where(np.abs(across) >= minimum, 1.0 / (cos_in * np.abs(across)), np.inf)
        factor[start:stop] = block
    return factor


def describe_intensity_corrections(corrections) -> Optional[dict]:
    """What was corrected and how, for the JSON record (``None``: nothing)."""
    if not has_intensity_corrections(corrections):
        return None
    record: dict = {"applied_as": "I / factor (1 at the direct beam; film absorption 1 at af = ai; the Lorentz "
                                  "factor not normalised); q unchanged"}
    if corrections.solid_angle:
        record["solid_angle"] = "(D/r)^3, D the distance to the direct beam and r to the pixel: cos^3(2theta) for a detector perpendicular to the beam"
    if corrections.polarization is not None:
        record["polarization"] = {
            "factor": float(corrections.polarization),
            "formula": "0.5*(1 + cos^2(2theta) - f*cos(2phi)*sin^2(2theta)), phi from the horizontal (pyFAI)",
        }
    if film_absorption_on(corrections):
        record["film_absorption"] = {
            "thickness_nm": float(corrections.film_thickness_nm),
            "attenuation_length_um": float(corrections.attenuation_length_um),
            "formula": "(1 - exp(-d/L*(1/sin ai + 1/sin af))) / (1/sin ai + 1/sin af), relative to af = ai; "
                       "no refraction",
        }
    if getattr(corrections, "lorentz", False):
        record["lorentz"] = {
            "formula": LORENTZ_FORMULA,
            "excluded_qy_half_width_px": float(corrections.lorentz_band_px),
            "exclusion": "|qy|/k < half_width_px * pixel_size_x_m / distance_m; qy=0 always excluded",
            "radial": "sum of corrected pixel intensities per q bin",
            "orientation": "pixel means before the Lorentz correction; N(chi) proportional to I(chi) sin(chi)",
        }
    return record


__all__ = ["LORENTZ_FORMULA", "describe_intensity_corrections", "film_absorption_on", "has_intensity_corrections",
           "intensity_factor"]
