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

from . import envelope, resolution, shapes, sizes, tables
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


def size_averages(family: str, q, *, radius, radius_spread, law: str = "number_gauss", height=10.0,
                  height_spread=0.0, coherent: bool = False, geometry: Geometry = Geometry(),
                  panels: Optional[int] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """(P̂, C or None) of B populations of one family (parameters scalars or arrays of length B), each (B, len(q)).

    ``panels``: the size rule's panels for every row (default: enough for the largest q, per row,
    ``sizes.panel_counts``; rows needing the same rule are evaluated together)."""
    if family not in shapes.FAMILIES:
        raise ValueError(f"Unknown particle family {family!r}")
    q = np.abs(np.asarray(q, float))
    radius, spread = sizes._arrays(radius, radius_spread)  # refuses a mean ≤ 0 or a width < 0
    rows = radius.size
    height, h_spread = _column(height, rows), _column(height_spread, rows)
    terms = geometry.sphere_terms()
    qz = {"sphere": max(abs(z) for _c, z in terms), "cylinder": geometry.random_cylinder_qz()}.get(family, 0.0)
    form = np.empty((rows, q.size))
    coh = np.empty_like(form) if coherent else None
    fast = _tabulated_rows(family, law, spread, h_spread, terms, coherent) if panels is None else np.zeros(rows, bool)
    if fast.any():
        index = np.flatnonzero(fast)
        if family == "cylinder":
            part, part_c = tables.cylinder(q, law, radius[index], spread[index], height[index], h_spread[index], qz,
                                           coherent)
        elif family == "sphere" and len(terms) == 2:  # two paths: two Born terms and their cross term
            (c_1, q_1), (c_2, q_2) = terms
            a, b = np.hypot(q, q_1), np.hypot(q, q_2)
            if abs(q_2) > abs(q_1):
                (c_1, q_1, a), (c_2, q_2, b) = (c_2, q_2, b), (c_1, q_1, a)
            first, _m = tables.averages(law, "sphere", a[None, :] * radius[index, None], spread[index])
            second, _m = tables.averages(law, "sphere", b[None, :] * radius[index, None], spread[index])
            cross = tables.sphere_cross(law, q, q_1, q_2, radius[index], spread[index])
            part = abs(c_1) ** 2 * first + abs(c_2) ** 2 * second + 2.0 * np.real(c_1 * np.conj(c_2) * cross)
            part_c = None
        else:
            kind = "sphere" if family == "sphere" else "disc"
            xbar = np.hypot(q, qz)[None, :] * radius[index, None]
            part, mean = tables.averages(law, kind, xbar, spread[index], coherent)
            part_c = None if mean is None else mean**2
        form[index] = part
        if coherent:
            coh[index] = part_c
        if fast.all():
            return form, coh
    rest = np.flatnonzero(~fast)
    radius, spread, height, h_spread = radius[rest], spread[rest], height[rest], h_spread[rest]
    rows = rest.size
    q_top = float(q.max()) if q.size else 0.0
    power = shapes.RADIUS_POWER[family]
    extent_r = sizes.extent(law, radius, spread, power)
    extent_h = sizes.extent(law, height, h_spread, shapes.HEIGHT_POWER) if family == "cylinder" else np.zeros(rows)
    # beyond q_cut a row's size average goes over to its envelope's (``envelope``): its quadrature only has to resolve
    # BLEND · q_cut
    cut_r = envelope.q_cut(law, radius, spread, power) if panels is None else np.full(rows, np.inf)
    cut_h = (envelope.q_cut(law, height, h_spread, shapes.HEIGHT_POWER) if family == "cylinder" and panels is None
             else np.full(rows, np.inf))
    if panels is not None:
        r_counts = h_counts = np.full(rows, int(panels))
    else:
        reach = envelope.BLEND
        r_counts = sizes.panel_counts(np.hypot(np.minimum(q_top, reach * cut_r), qz), extent_r)
        h_counts = (sizes.panel_counts(np.hypot(np.minimum(q_top, reach * cut_h), qz), extent_h)
                    if family == "cylinder" else np.zeros(rows, int))
    keys = r_counts * MAX_KEY + h_counts
    for key in np.unique(keys):
        group = np.flatnonzero(keys == key)
        radii, weights = sizes.nodes(law, radius[group], spread[group], power=power, panels=int(r_counts[group[0]]))
        if family == "cylinder":
            heights, h_weights = sizes.nodes(law, height[group], h_spread[group], power=shapes.HEIGHT_POWER,
                                             panels=int(h_counts[group[0]]))
            part, part_c = shapes.cylinder(
                q, radii, weights, heights, h_weights, qz, coherent, cut_r[group], cut_h[group],
                sizes.nodes(law, radius[group], spread[group], power=power, panels=ENVELOPE_PANELS),
                sizes.nodes(law, height[group], h_spread[group], power=shapes.HEIGHT_POWER, panels=ENVELOPE_PANELS))
        else:
            part, part_c = _direct_and_envelope(family, q, radii, weights, terms, coherent, cut_r[group], law,
                                                radius[group], spread[group], power, qz)
        form[rest[group]] = part
        if coherent:
            coh[rest[group]] = part_c
    return form, coh


def _tabulated_rows(family, law, spread, h_spread, terms, coherent) -> np.ndarray:
    """Rows the tables of ``tables`` evaluate: their widths covered, and for spheres one path or two paths without
    the coherent term (the coherent term of a path at q_z ≠ 0 carries a phase the tables do not; four DWBA paths are
    averaged directly)."""
    ok = tables.usable(law, spread)
    if family == "cylinder":
        return ok & tables.usable(law, h_spread)
    if family == "sphere" and (len(terms) > 2 or (coherent and any(z != 0.0 for _c, z in terms))):
        return np.zeros_like(ok)
    return ok


ENVELOPE_PANELS = 16
"""Panels of the size rule of an envelope average (smooth in R; more when two DWBA paths beat)."""


def _direct_and_envelope(family, q, radii, weights, terms, coherent, cut, law, radius, spread, power, qz):
    """Spheres or upright cylinders of one quadrature group: resolved up to each row's ``cut``, its envelope from
    ``BLEND`` · ``cut`` on, a smoothstep between (``envelope``)."""
    def evaluate(columns, nodes_, node_weights, envelope_part):
        if family == "sphere":
            return shapes.sphere(q[columns], nodes_, node_weights, terms, coherent, envelope_part)
        return shapes.vertical_cylinder(q[columns], nodes_, node_weights, coherent, envelope_part)

    step = envelope.blend(q, cut)
    if not np.any(step > 0):
        return evaluate(slice(None), radii, weights, False)
    form = np.zeros((radii.shape[0], q.size))
    coh = np.zeros_like(form) if coherent else None
    resolved = np.any(step < 1, axis=0)
    if resolved.any():
        part, part_c = evaluate(resolved, radii, weights, False)
        form[:, resolved] = (1.0 - step[:, resolved]) * part
        if coherent:
            coh[:, resolved] = (1.0 - step[:, resolved]) * part_c
    outer = np.any(step > 0, axis=0)
    slow = max((abs(a - b) for _c, a in terms for _d, b in terms), default=0.0)  # beating of two DWBA paths
    count = max(ENVELOPE_PANELS, int(np.max(sizes.panel_counts(slow, sizes.extent(law, radius, spread, power)))))
    env_nodes, env_weights = sizes.nodes(law, radius, spread, power=power, panels=count)
    part, _zero = evaluate(outer, env_nodes, env_weights, True)
    form[:, outer] += step[:, outer] * part
    return form, coh


def couple(form, coherent, q, distance, distance_spread, coupling: str = "lma") -> np.ndarray:
    """T from the size averages: P̂·S (LMA) or P̂ + C (S − 1) (DA); rows without a structure factor keep P̂."""
    if coupling not in COUPLINGS:
        raise ValueError(f"Unknown coupling {coupling!r}")
    rows = form.shape[0]
    distance, d_spread = _column(distance, rows), _column(distance_spread, rows)
    structured = (distance > 0) & (d_spread > 0)
    if not np.any(structured):
        return form
    s = paracrystal(q, distance, d_spread)
    out = form + coherent * (s - 1.0) if coupling == "da" else form * s
    return np.where(structured[:, None], out, form)


def population_terms(family: str, q, *, radius, radius_spread, law: str = "number_gauss", height=10.0,
                     height_spread=0.0, distance=0.0, distance_spread=0.0, coupling: str = "lma",
                     geometry: Geometry = Geometry(), panels: Optional[int] = None) -> np.ndarray:
    """T(q) of B populations of one family (parameters scalars or arrays of length B); result (B, len(q))."""
    if coupling not in COUPLINGS:
        raise ValueError(f"Unknown coupling {coupling!r}")
    rows = np.atleast_1d(np.asarray(radius, float)).size
    structured = (_column(distance, rows) > 0) & (_column(distance_spread, rows) > 0)
    form, coh = size_averages(family, q, radius=radius, radius_spread=radius_spread, law=law, height=height,
                              height_spread=height_spread, coherent=coupling == "da" and bool(np.any(structured)),
                              geometry=geometry, panels=panels)
    return couple(form, coh, np.abs(np.asarray(q, float)), distance, distance_spread, coupling)


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


__all__ = ["COUPLINGS", "Population", "Settings", "couple", "evaluate", "population_term", "population_terms",
           "size_averages", "size_moments", "tail", "unit_terms"]
