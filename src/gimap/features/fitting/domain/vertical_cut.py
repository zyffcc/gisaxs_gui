"""A vertical cut of a GISAXS frame: the intensity along the exit angle at one in-plane q, in the four-term DWBA of
particles on a substrate of known refractive index (Physics and Methods 9.18). Used by the fit of several curves
together (``joint_fit``), where it constrains what a horizontal cut does not see: the height of upright cylinders
and the vertical extent of the form factor.

Every point of the cut is an exit angle α_f (q_z = k (sin α_i + sin α_f) in vacuum, k = 2π/λ), with its own path
wave vectors and reflection coefficient (Renaud, Lazzari & Leroy 2009; IsGISAXS, Lazzari 2002):

    A = F(q_∥, k_i + k_f) + r_i F(q_∥, k_f − k_i) + r_f F(q_∥, k_i − k_f) + r_i r_f F(q_∥, −(k_i + k_f))
    k_i = k sin α_i,  k_f = k sin α_f,  r = (k_z − k_z,s)/(k_z + k_z,s),  k_z,s = k √(sin²α − 2δ + 2iβ)

for particles standing on the substrate (their base at z = 0):

    sphere            F(q, Q) = V f(√(q² + Q²) R) e^{−iQR},          f(x) = 3 (sin x − x cos x)/x³,  V ∝ R³
    upright cylinder  F(q, Q) = V [2J₁(qR)/(qR)] sinc(QH/2) e^{−iQH/2},  V ∝ R² H

averaged over the sizes by number (``cut_model.sizes``; R and H independent):

    P̂(α_f) = ⟨V² |A|²⟩ / ⟨V²⟩,    C(α_f) = |⟨V A⟩|² / ⟨V²⟩
    T = P̂ · S(q_∥)  (LMA)   or   P̂ + C (S(q_∥) − 1)  (DA)

The interference function is that of the in-plane q of the cut, the same at every point. The averages are direct
composite Gauss–Legendre rules whose panels follow the largest |q| = √(q_∥² + Q²) of the cut (``sizes.panel_counts``),
without the envelope of the horizontal cut's tables. The cut's q_∥ is that of its column band on the horizon; over the
cut it changes with α_f by k (cos α_f − cos α_i), less than 10⁻² nm⁻¹ at α_f < 1.5°, which is not modelled; nor is a
q resolution. Random cylinders and core–shell spheres are not available in a vertical cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from . import model_library
from .cut_model import sizes
from .cut_model.shapes import disc_amplitude, sinc_half, sphere_amplitude
from .cut_model.structure import paracrystal

FAMILIES = ("sphere", "vertical_cylinder")
CHUNK = 2_000_000
"""Complex elements of the largest temporary (nodes × points) evaluated at once."""


@dataclass(frozen=True)
class VerticalCut:
    """Where a vertical cut is taken: the in-plane q of its columns (nm⁻¹), λ (nm), α_i (°) and the substrate's δ, β
    at that λ. ``critical_angle_deg``: the curve's q_z is the one refracted in a film of this critical angle
    (Analyze's Corrections); 0: the vacuum q_z."""
    q_parallel: float
    wavelength_nm: float
    alpha_i_deg: float
    delta: float
    beta: float
    critical_angle_deg: float = 0.0

    def __post_init__(self):
        if not (self.wavelength_nm > 0 and self.alpha_i_deg > 0):
            raise ValueError("A vertical cut needs the wavelength and the incidence angle")
        if not (self.delta > 0 and self.beta >= 0):
            raise ValueError("A vertical cut needs the substrate (δ > 0, β ≥ 0)")
        if not self.q_parallel >= 0:
            raise ValueError("The in-plane q of a vertical cut must be ≥ 0")

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.wavelength_nm

    def exit_angles_deg(self, qz) -> np.ndarray:
        """α_f (°) of every q_z (nm⁻¹); NaN where the q_z has no exit angle above the horizon (or, refracted, none
        above the film's critical angle)."""
        qz = np.asarray(qz, float)
        sin_i = math.sin(math.radians(self.alpha_i_deg))
        if self.critical_angle_deg > 0:
            sin_c2 = math.sin(math.radians(self.critical_angle_deg)) ** 2
            incoming = math.sqrt(max(sin_i**2 - sin_c2, 0.0))
            outgoing = qz / self.k - incoming
            sin_f = np.sqrt(np.where(outgoing > 0, outgoing**2 + sin_c2, np.nan))
        else:
            sin_f = qz / self.k - sin_i
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(sin_f) & (sin_f > 0) & (sin_f < 1)
            return np.where(ok, np.degrees(np.arcsin(np.where(ok, sin_f, 0.5))), np.nan)

    def fresnel(self, alpha_deg) -> np.ndarray:
        """Reflection coefficient of the bare substrate at the grazing angles ``alpha_deg`` (as
        ``cut_model.Geometry.fresnel``)."""
        s = np.sin(np.radians(np.asarray(alpha_deg, float)))
        kz0 = self.k * s
        ks = self.k * np.sqrt((s * s - 2.0 * self.delta) + 2j * self.beta)
        return (kz0 - ks) / (kz0 + ks)

    def paths(self, qz) -> tuple[np.ndarray, np.ndarray]:
        """(coefficients (4, N) complex, Q (4, N)) of the four DWBA paths at every q_z."""
        alpha_f = self.exit_angles_deg(qz)
        if np.any(~np.isfinite(alpha_f)):
            raise ValueError("A point of the vertical cut has no exit angle above the horizon")
        k_i = self.k * math.sin(math.radians(self.alpha_i_deg))
        k_f = self.k * np.sin(np.radians(alpha_f))
        r_i = complex(self.fresnel(self.alpha_i_deg))
        r_f = self.fresnel(alpha_f)
        ones = np.ones_like(k_f, dtype=complex)
        coefficients = np.stack([ones, r_i * ones, r_f, r_i * r_f])
        wave = np.stack([k_i + k_f, k_f - k_i, k_i - k_f, -(k_i + k_f)])
        return coefficients, wave


def _rule(law: str, mean: float, spread: float, power: float, q_top: float) -> tuple[np.ndarray, np.ndarray]:
    panels = int(sizes.panel_counts(q_top, sizes.extent(law, mean, spread, power))[0])
    radii, weights = sizes.nodes(law, mean, spread, power=power, panels=panels)
    return radii[0], weights[0]


def _blocks(nodes: int, points: int):
    step = max(1, CHUNK // max(nodes * 4, 1))
    return [slice(a, min(a + step, points)) for a in range(0, points, step)]


def sphere_averages(q_parallel: float, coefficients, wave, law: str, radius: float, spread: float,
                    coherent: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """(P̂, C or None) of spheres on the four paths at every point (module docstring)."""
    q_top = float(np.max(np.hypot(q_parallel, wave))) if wave.size else 0.0
    radii, weights = _rule(law, radius, spread, 6.0, q_top)
    v2 = weights * radii**6
    norm = v2.sum()
    points = wave.shape[1]
    form = np.empty(points)
    coh = np.empty(points) if coherent else None
    for cols in _blocks(radii.size, points):
        amplitude = 0j
        for c, q_z in zip(coefficients[:, cols], wave[:, cols]):
            x = np.hypot(q_parallel, q_z)[None, :] * radii[:, None]
            amplitude = amplitude + c[None, :] * sphere_amplitude(x) * np.exp(-1j * q_z[None, :] * radii[:, None])
        form[cols] = v2 @ (np.abs(amplitude) ** 2) / norm
        if coherent:
            coh[cols] = np.abs((weights * radii**3) @ amplitude) ** 2 / norm
    return form, coh


def cylinder_averages(q_parallel: float, coefficients, wave, law: str, radius: float, spread: float, height: float,
                      height_spread: float, coherent: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """(P̂, C or None) of upright cylinders: the radius part at q_∥ times the height part of the four paths."""
    radii, r_weights = _rule(law, radius, spread, 4.0, q_parallel)
    disc = disc_amplitude(q_parallel * radii)
    r4 = r_weights * radii**4
    radial_sq = float(r4 @ disc**2 / r4.sum())
    radial = float((r_weights * radii**2) @ disc / math.sqrt(r4.sum()))
    q_top = float(np.max(np.abs(wave))) if wave.size else 0.0
    heights, h_weights = _rule(law, height, height_spread, 2.0, q_top)
    h2 = h_weights * heights**2
    points = wave.shape[1]
    form = np.empty(points)
    coh = np.empty(points) if coherent else None
    for cols in _blocks(heights.size, points):
        amplitude = 0j
        for c, q_z in zip(coefficients[:, cols], wave[:, cols]):
            x = q_z[None, :] * heights[:, None]
            amplitude = amplitude + c[None, :] * sinc_half(x) * np.exp(-0.5j * x)
        form[cols] = radial_sq * (h2 @ (np.abs(amplitude) ** 2)) / h2.sum()
        if coherent:
            coh[cols] = radial**2 * np.abs((h_weights * heights) @ amplitude) ** 2 / h2.sum()
    return form, coh


def structure_value(interference: str, values: Mapping[str, float], q_parallel: float) -> Optional[float]:
    """S(q_∥) of the component's interference function (None: no structure factor in effect)."""
    q = np.array([max(q_parallel, 1e-9)])
    if interference == "hard_sphere":
        if not (values["R_hs"] > 0 and values["phi"] > 0):
            return None
        return float(np.ravel(model_library.hard_sphere(q, values["R_hs"], values["phi"]))[0])
    if not (values["D"] > 0 and values["sigma_nn"] > 0):
        return None
    if interference == "hex2d":
        return float(np.ravel(model_library.hexagonal_paracrystal(q, values["D"], values["sigma_nn"]))[0])
    return float(paracrystal(q, values["D"], values["sigma_nn"])[0, 0])


def component_term(family: str, law: str, interference: str, coupling: str, structure: bool,
                   values: Mapping[str, float], qz, cut: VerticalCut,
                   cached: Optional[Callable] = None) -> np.ndarray:
    """T(q_z) (unit scale up to the cut's own factor) of one component on a vertical cut. ``values``: its parameters
    by key (an upright cylinder needs ``h`` and ``sigma_h``); ``cached(key, compute)``: the fit's cache."""
    if family not in FAMILIES:
        raise ValueError("A vertical cut takes spheres and upright cylinders (the four-term DWBA of particles on the "
                         "substrate); this model has another family")
    cached = cached or (lambda _key, compute: compute())
    qz = np.abs(np.asarray(qz, float))
    factor = structure_value(interference, values, cut.q_parallel) if structure else None
    coherent = factor is not None and coupling == "da"
    paths = cached(("vertical paths", cut), lambda: cut.paths(qz))
    if family == "sphere":
        key = ("vertical", family, law, values["R"], values["sigma_R"], cut, coherent)
        form, coh = cached(key, lambda: sphere_averages(cut.q_parallel, *paths, law, values["R"], values["sigma_R"],
                                                         coherent))
    else:
        if "h" not in values:
            raise ValueError("Upright cylinders in a vertical cut need their height h (and σh/h)")
        key = ("vertical", family, law, values["R"], values["sigma_R"], values["h"], values["sigma_h"], cut, coherent)
        form, coh = cached(key, lambda: cylinder_averages(cut.q_parallel, *paths, law, values["R"], values["sigma_R"],
                                                           values["h"], values["sigma_h"], coherent))
    if factor is None:
        return np.asarray(form, float)
    return np.asarray(form + coh * (factor - 1.0) if coherent else form * factor, float)


__all__ = ["FAMILIES", "VerticalCut", "component_term", "cylinder_averages", "sphere_averages", "structure_value"]
