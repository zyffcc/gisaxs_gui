"""Models for manual fitting beyond the proposal network's families (Physics and Methods, section 9.14).

Refine and Search can use them; Predict parameters proposes only the sphere and cylinder compositions it was trained
on. Each function is unit-scale like ``cut_model`` (a form factor is 1 at q = 0, a structure factor → 1 at large q).
q in nm⁻¹, lengths in nm.

* ``core_shell``: spheres of core radius R (size distribution by ``law``) with a shell of fixed thickness t; the
  core's scattering contrast is ``ratio`` times the shell's (0: a hollow shell, 1: a homogeneous sphere of radius
  R + t). Amplitude A(q) = R_s³ f(q R_s) + (ratio − 1) R³ f(q R), R_s = R + t, f(x) = 3 (sin x − x cos x)/x³
  (Guinier & Fournet 1955; SasView's ``core_shell_sphere`` with ratio = (ρ_core − ρ_solvent)/(ρ_shell − ρ_solvent)).
  P̂ = ⟨A²⟩/⟨A(0)²⟩ and, for the decoupling approximation, C = ⟨A⟩²/⟨A(0)²⟩ (number averages over R).
* ``hard_sphere``: the Percus–Yevick structure factor of hard spheres of radius R_HS at volume fraction φ
  (Percus & Yevick 1958, solved by Wertheim 1963 and Thiele 1963; the closed form of Kinning & Thomas 1984, as in
  SasView's ``hardsphere``): S = 1 / (1 + 24 φ I(2 q R_HS)), I(x) = ∫₀¹ (α s² + β s³ + γ s⁵) sin(x s)/(x s) ds,
  α = (1 + 2φ)²/(1 − φ)⁴, β = −6φ (1 + φ/2)²/(1 − φ)⁴, γ = φ α / 2. S(0) = (1 − φ)⁴/(1 + 2φ)².
* ``hexagonal_paracrystal``: the 2D paracrystal of a hexagonal lattice (nearest-neighbour distance D, lattice angle
  120°) with an isotropic Gaussian of standard deviation ω = σ·D for each in-plane component of a neighbour's
  displacement, infinite domains, averaged over the in-plane orientation ξ of the domains (Hosemann & Bagchi 1962;
  Renaud, Lazzari & Leroy 2009, Surf. Sci. Rep. 64, 255; BornAgain's ``Paracrystal2D`` with integration over ξ,
  Pospelov et al. 2020, J. Appl. Cryst. 53, 262):
  S(q) = (1/π) ∫₀^π s(q D cos ξ) s(q D cos(ξ + 2π/3)) dξ,  s(θ) = Re[(1 + φ)/(1 − φ)] = (1 − p²)/(1 − 2p cos θ + p²),
  φ = p e^{iθ}, p = exp(−q² ω²/2). The ξ grid follows the narrowest peak of s (midpoint rule, periodic in ξ).
"""

from __future__ import annotations

import math

import numpy as np

from .cut_model import sizes
from .cut_model.model import size_averages
from .cut_model.shapes import sphere_amplitude, upright_height_coherence
from .cut_model.structure import paracrystal

CHUNK = 1_000_000
"""Elements of the largest temporary evaluated at once."""
SHELL_POWER = 6
"""V² ∝ R^6 of the core radius: how ``intensity_gauss`` turns intensity weights into number weights (an approximation
for a core–shell particle, whose V² depends on the shell too)."""
SERIES_BELOW = 1.0
"""x = 2 q R_HS below which the hard-sphere integral is summed as its Taylor series (no cancellation)."""
SERIES_TERMS = 14
XI_POINTS_PER_WIDTH = 3.0
"""Points of the orientation grid per half-width of the narrowest peak of s (midpoint rule: error ~ e^(−2π·3))."""
XI_MIN, XI_MAX = 64, 2**18


# -- core–shell spheres ---------------------------------------------------------------------------------------


def core_shell(q, radius: float, spread: float, thickness: float, ratio: float, law: str = "number_gauss",
               coherent: bool = False):
    """(P̂, C or None) of core–shell spheres, each (1, len(q)); the core radius follows ``law``."""
    q = np.abs(np.asarray(q, float))
    if thickness < 0:
        raise ValueError("A shell thickness must be >= 0")
    q_max = float(q.max()) if q.size else 0.0
    span = float(sizes.extent(law, radius, spread, SHELL_POWER)[0])
    panels = sizes.panel_count(q_max, span)
    radii, weights = sizes.nodes(law, radius, spread, power=SHELL_POWER, panels=panels)
    radii, weights = radii[0], weights[0]
    outer = radii + thickness
    contrast = ratio - 1.0
    forward = outer**3 + contrast * radii**3
    norm = float(np.sum(weights * forward**2))
    if not norm > 0:
        raise ValueError("This core–shell particle scatters nothing at q = 0 (its contrasts cancel)")
    p_out = np.empty(q.size)
    c_out = np.empty(q.size) if coherent else None
    step = max(1, CHUNK // max(radii.size, 1))
    for start in range(0, q.size, step):
        part = q[start:start + step][None, :]
        amplitude = (outer[:, None] ** 3 * sphere_amplitude(part * outer[:, None])
                     + contrast * radii[:, None] ** 3 * sphere_amplitude(part * radii[:, None]))
        p_out[start:start + step] = weights @ (amplitude * amplitude) / norm
        if coherent:
            c_out[start:start + step] = (weights @ amplitude) ** 2 / norm
    return p_out[None, :], (None if c_out is None else c_out[None, :])


# -- Percus–Yevick hard spheres --------------------------------------------------------------------------------


def _py_coefficients(fraction: float) -> tuple[float, float, float]:
    alpha = (1.0 + 2.0 * fraction) ** 2 / (1.0 - fraction) ** 4
    beta = -6.0 * fraction * (1.0 + 0.5 * fraction) ** 2 / (1.0 - fraction) ** 4
    return alpha, beta, 0.5 * fraction * alpha


def hard_sphere(q, radius: float, fraction: float) -> np.ndarray:
    """S(q) of hard spheres of radius ``radius`` at volume fraction ``fraction`` (Percus–Yevick); (1, len(q))."""
    q = np.abs(np.asarray(q, float))
    if radius <= 0 or fraction <= 0:
        return np.ones((1, q.size))
    if fraction >= 1:
        raise ValueError("A hard-sphere volume fraction must be below 1")
    alpha, beta, gamma = _py_coefficients(fraction)
    x = 2.0 * q * radius
    integral = np.empty_like(x)
    small = x < SERIES_BELOW
    if np.any(small):  # Σ_k (−1)^k x^{2k}/(2k+1)! [α/(2k+3) + β/(2k+4) + γ/(2k+6)]
        x2, term, total = x[small] ** 2, np.ones(int(small.sum())), np.zeros(int(small.sum()))
        for k in range(SERIES_TERMS):
            total += term * (alpha / (2 * k + 3) + beta / (2 * k + 4) + gamma / (2 * k + 6))
            term = -term * x2 / ((2 * k + 2) * (2 * k + 3))
        integral[small] = total
    large = ~small
    if np.any(large):
        z = x[large]
        s, c = np.sin(z), np.cos(z)
        integral[large] = (alpha * (s - z * c) / z**3
                           + beta * (2.0 * z * s + (2.0 - z * z) * c - 2.0) / z**4
                           + gamma * (-(z**4) * c + 4.0 * ((3.0 * z * z - 6.0) * c + (z**3 - 6.0 * z) * s + 6.0)) / z**6)
    return (1.0 / (1.0 + 24.0 * fraction * integral))[None, :]


# -- the 2D hexagonal paracrystal --------------------------------------------------------------------------------


def _kernel(theta: np.ndarray, log_p: np.ndarray) -> np.ndarray:
    """(1 − p²)/(1 − 2p cos θ + p²) without cancellation near p = 1 (as ``cut_model.structure.paracrystal``)."""
    p = np.exp(log_p)
    return -np.expm1(2.0 * log_p) / (np.expm1(log_p) ** 2 + 4.0 * p * np.sin(0.5 * theta) ** 2)


def xi_points(q, distance: float, spread: float) -> np.ndarray:
    """Points of the orientation grid for every q: ``XI_POINTS_PER_WIDTH`` per half-width (in ξ) of the narrowest
    peak of s, (1 − p)/(q D), rounded up to a power of 2."""
    q = np.abs(np.asarray(q, float))
    width = -np.expm1(-0.5 * (q * spread * distance) ** 2) / np.maximum(q * distance, 1e-300)
    need = np.ceil(XI_POINTS_PER_WIDTH * math.pi / np.maximum(width, 1e-300))
    need = np.clip(need, XI_MIN, XI_MAX)
    return np.minimum(2 ** np.ceil(np.log2(need)), XI_MAX).astype(int)


def hexagonal_paracrystal(q, distance: float, spread: float) -> np.ndarray:
    """S(q) of the orientation-averaged 2D hexagonal paracrystal (see the module); (1, len(q)). q = 0 is taken at
    q D = 10⁻⁶ (the product form rises as 1/(q D) towards q = 0)."""
    q = np.abs(np.asarray(q, float))
    if distance <= 0 or spread <= 0:
        return np.ones((1, q.size))
    q = np.maximum(q, 1e-6 / distance)
    log_p = -0.5 * (q * spread * distance) ** 2
    counts = xi_points(q, distance, spread)
    out = np.empty(q.size)
    for count in np.unique(counts):
        where = np.flatnonzero(counts == count)
        xi = (np.arange(count) + 0.5) * (math.pi / count)
        first, second = np.cos(xi), np.cos(xi + 2.0 * math.pi / 3.0)
        rows = max(1, CHUNK // int(count))
        for start in range(0, where.size, rows):
            index = where[start:start + rows]
            x = (q[index] * distance)[:, None]
            lp = log_p[index][:, None]
            out[index] = np.mean(_kernel(x * first[None, :], lp) * _kernel(x * second[None, :], lp), axis=1)
    return out[None, :]


# -- one component of a manual model ------------------------------------------------------------------------------


def component_term(family: str, law: str, kind: str, coupling: str, structure: bool, values: dict, q, cached,
                   geometry) -> np.ndarray:
    """T(q) (unit scale) of a component whose family or interference function is of this module (``fit_model``
    evaluates the others itself). ``values``: its parameters by key; ``cached(key, compute)``: the fit's cache;
    ``geometry``: the cut's (core–shell spheres are evaluated at q_z = 0, the effective interface)."""
    if not structure:
        structured = False
    elif kind == "hard_sphere":
        structured = values["R_hs"] > 0 and values["phi"] > 0
    else:
        structured = values["D"] > 0 and values["sigma_nn"] > 0
    coherent = structured and coupling == "da"
    if family == "core_shell_sphere":
        key = ("core-shell", law, values["R"], values["sigma_R"], values["t_shell"], values["core_contrast"], coherent)
        form, average = cached(key, lambda: core_shell(q, values["R"], values["sigma_R"], values["t_shell"],
                                                       values["core_contrast"], law, coherent))
    else:
        height, height_spread = (values["h"], values["sigma_h"]) if family == "cylinder" else (10.0, 0.0)
        key = ("sizes", family, law, values["R"], values["sigma_R"], height, height_spread,
               geometry if family != "vertical_cylinder" else None, coherent)
        form, average = cached(key, lambda: size_averages(
            family, q, radius=values["R"], radius_spread=values["sigma_R"], law=law, height=height,
            height_spread=height_spread, coherent=coherent, geometry=geometry))
        if coherent and family == "vertical_cylinder" and values.get("sigma_h", 0.0) > 0:
            height_key = ("upright height coherence", law, values["h"], values["sigma_h"], geometry)
            fraction = cached(height_key, lambda: upright_height_coherence(law, values["h"], values["sigma_h"], geometry))
            average = average * fraction
    if not structured:
        return np.asarray(form[0], dtype=float)
    if kind == "hard_sphere":
        factor = cached(("PY", values["R_hs"], values["phi"]), lambda: hard_sphere(q, values["R_hs"], values["phi"]))
    elif kind == "hex2d":
        factor = cached(("hex2d", values["D"], values["sigma_nn"]),
                        lambda: hexagonal_paracrystal(q, values["D"], values["sigma_nn"]))
    else:
        factor = cached(("S", values["D"], values["sigma_nn"]), lambda: paracrystal(q, values["D"], values["sigma_nn"]))
    out = form + average * (factor - 1.0) if coherent else form * factor
    return np.asarray(out[0], dtype=float)


__all__ = ["component_term", "core_shell", "hard_sphere", "hexagonal_paracrystal", "xi_points"]
