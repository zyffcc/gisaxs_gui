"""The intensity of a 1D GISAXS cut: GIMaP's one forward model for Fitting, the proposal network and refinement.

    I(q) = BG + k · [ Σ_j c_j T_j(q) + A / (1 + (|q|/w)^ν) ]

* T_j is population j's unit-scale term (P̂ and C of ``shapes``, S of ``structure``):
  LMA  T = P̂ · S  (every size among neighbours of its own size), DA  T = P̂ + C (S − 1) (sizes and positions
  independent); without a structure factor T = P̂. With a Gaussian q resolution σ_q > 0, every T_j is convolved with
  it (``resolution``); the beam tail and BG are not.
* c_j is the forward intensity of population j (T(0) = 1 for every family in the effective interface
  approximation), A, w, ν the beam tail of the direct and reflected beam, BG a constant, k an overall factor.

q in nm⁻¹ (|q| is used), lengths in nm. ``population_terms`` evaluates B parameter sets of one family at once (the
training simulator); ``evaluate`` one model (Fitting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from . import resolution, shapes, sizes
from .geometry import Geometry
from .structure import paracrystal

COUPLINGS = ("lma", "da")
MAX_KEY = sizes.MAX_PANELS + 1
"""Groups rows by (radius panels, height panels)."""


@dataclass(frozen=True)
class Population:
    family: str
    radius: float
    radius_spread: float = 0.0
    law: str = "number_gauss"
    height: float = 10.0
    height_spread: float = 0.0
    distance: float = 0.0
    """Mean neighbour distance D of the paracrystal; 0: no structure factor."""
    distance_spread: float = 0.0
    """σ_nn/D, the relative standard deviation of the neighbour distance (Hosemann)."""
    coupling: str = "lma"
    scale: float = 1.0

    def __post_init__(self):
        if self.family not in shapes.FAMILIES:
            raise ValueError(f"Unknown particle family {self.family!r}")
        if self.coupling not in COUPLINGS:
            raise ValueError(f"Unknown coupling {self.coupling!r}")


@dataclass(frozen=True)
class Settings:
    background: float = 0.0
    tail_amplitude: float = 0.0
    tail_width: float = 0.02
    tail_exponent: float = 3.0
    q_resolution: float = 0.0
    k: float = 1.0
    geometry: Geometry = field(default_factory=Geometry)


def tail(q, width: float, exponent: float) -> np.ndarray:
    """1 / (1 + (|q|/w)^ν); 1 everywhere when w ≤ 0 or ν ≤ 0."""
    q = np.abs(np.asarray(q, float))
    if width <= 0 or exponent <= 0:
        return np.ones_like(q)
    return 1.0 / (1.0 + np.power(q / width, exponent))


def _column(value, rows: int) -> np.ndarray:
    return np.broadcast_to(np.atleast_1d(np.asarray(value, float)), (rows,)).astype(float)


def population_terms(family: str, q, *, radius, radius_spread, law: str = "number_gauss", height=10.0,
                     height_spread=0.0, distance=0.0, distance_spread=0.0, coupling: str = "lma",
                     geometry: Geometry = Geometry(), panels: Optional[int] = None) -> np.ndarray:
    """T(q) of B populations of one family (parameters scalars or arrays of length B); result (B, len(q)).

    ``panels``: the size rule's panels for every row (default: enough for the largest q, per row,
    ``sizes.panel_counts``; rows needing the same rule are evaluated together)."""
    if family not in shapes.FAMILIES:
        raise ValueError(f"Unknown particle family {family!r}")
    if coupling not in COUPLINGS:
        raise ValueError(f"Unknown coupling {coupling!r}")
    q = np.abs(np.asarray(q, float))
    radius = np.atleast_1d(np.asarray(radius, float))
    rows = radius.size
    spread = _column(radius_spread, rows)
    height, h_spread = _column(height, rows), _column(height_spread, rows)
    distance, d_spread = _column(distance, rows), _column(distance_spread, rows)
    structured = (distance > 0) & (d_spread > 0)
    coherent = coupling == "da" and bool(np.any(structured))
    terms = geometry.sphere_terms()
    qz = {"sphere": max(abs(z) for _c, z in terms), "cylinder": geometry.random_cylinder_qz()}.get(family, 0.0)
    q_eff = float(np.hypot(q.max(), qz)) if q.size else 0.0
    if panels is not None:
        r_counts = h_counts = np.full(rows, int(panels))
    else:
        r_counts = sizes.panel_counts(q_eff, sizes.extent(law, radius, spread, shapes.RADIUS_POWER[family]))
        h_counts = (sizes.panel_counts(q_eff, sizes.extent(law, height, h_spread, shapes.HEIGHT_POWER))
                    if family == "cylinder"
                    else np.zeros(rows, int))
    form = np.empty((rows, q.size))
    coh = np.empty_like(form) if coherent else None
    keys = r_counts * MAX_KEY + h_counts
    for key in np.unique(keys):
        group = np.flatnonzero(keys == key)
        count = int(r_counts[group[0]])
        radii, weights = sizes.nodes(law, radius[group], spread[group], power=shapes.RADIUS_POWER[family],
                                     panels=count)
        if family == "sphere":
            part, part_c = shapes.sphere(q, radii, weights, terms, coherent)
        elif family == "vertical_cylinder":
            part, part_c = shapes.vertical_cylinder(q, radii, weights, coherent)
        else:
            heights, h_weights = sizes.nodes(law, height[group], h_spread[group], power=shapes.HEIGHT_POWER,
                                             panels=int(h_counts[group[0]]))
            part, part_c = shapes.cylinder(q, radii, weights, heights, h_weights, qz, coherent)
        form[group] = part
        if coherent:
            coh[group] = part_c
    if not np.any(structured):
        return form
    s = paracrystal(q, distance, d_spread)
    out = form + coh * (s - 1.0) if coherent else form * s
    return np.where(structured[:, None], out, form)


def population_term(population: Population, q, geometry: Geometry = Geometry()) -> np.ndarray:
    """T(q) of one population (unit scale)."""
    return population_terms(
        population.family, q, radius=population.radius, radius_spread=population.radius_spread, law=population.law,
        height=population.height, height_spread=population.height_spread, distance=population.distance,
        distance_spread=population.distance_spread, coupling=population.coupling, geometry=geometry)[0]


def unit_terms(populations: Sequence[Population], settings: Settings, q,
               term: Optional[Callable[[Population, np.ndarray], np.ndarray]] = None) -> list[np.ndarray]:
    """T_j(q) of every population, convolved with the q resolution when σ_q > 0.

    ``term(population, q)``: how one population is evaluated (default ``population_term`` in the settings'
    geometry); Fitting passes a cached one."""
    q = np.abs(np.asarray(q, float))
    term = term or (lambda population, grid: population_term(population, grid, settings.geometry))
    if settings.q_resolution > 0 and q.size:
        return resolution.smeared(q, settings.q_resolution, lambda grid: [term(p, grid) for p in populations])
    return [term(p, q) for p in populations]


def evaluate(populations: Sequence[Population], settings: Settings, q, *, parts: bool = False, term=None):
    """I(q); with ``parts``, also ``{"populations": [k·c_j·T_j], "tail": k·A·tail, "background": BG}``."""
    q = np.abs(np.asarray(q, float))
    curves = [settings.k * p.scale * t for p, t in zip(populations, unit_terms(populations, settings, q, term))]
    beam = settings.k * settings.tail_amplitude * tail(q, settings.tail_width, settings.tail_exponent)
    total = settings.background + beam + (np.sum(curves, axis=0) if curves else 0.0)
    total = np.broadcast_to(total, q.shape).astype(float)
    if parts:
        return total, {"populations": curves, "tail": beam, "background": np.full_like(q, settings.background)}
    return total


def size_moments(population: Population) -> dict:
    """Mean radius and relative spread by number and by intensity (V²) (``sizes.size_report``)."""
    return sizes.size_report(population.law, population.radius, population.radius_spread,
                             shapes.RADIUS_POWER[population.family])


__all__ = ["COUPLINGS", "Population", "Settings", "evaluate", "population_term", "population_terms", "size_moments",
           "tail", "unit_terms"]
