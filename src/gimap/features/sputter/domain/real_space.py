"""Real-space picture of one solution of the analysis package (pure Python): where its grains sit, in nm.

A solution of a grain model family gives the grain radius R, the centre distance D, the log-normal width of the radius
distribution, the height H of the metal above the substrate and the thickness of the continuous layer under the grains
(``base_nm``; 0 for separate islands). The picture shows them at their size:

- top view: a square of ``FIELD_DISTANCES`` x D; the grains on a hexagonal arrangement of distance D, each shifted at
  random by about ``SHIFT`` x D (an arrangement without long-range order, as the model's interference function has),
  each with a radius drawn from the log-normal distribution of median R. A base layer covers the whole field.
- side view: a cut ``SIDE_DISTANCES`` x D long through one row of grains: the base layer, and on it each grain as a cap
  of its radius and of the height H - base, scaled with its radius.

The random numbers come from one seed and are drawn in the same order for every solution, so moving through the windows
or the candidates changes only the sizes, never the dice. A solution without a radius or a distance (a family that has
no grain arrangement, e.g. metal in a porous layer) has no grain picture: ``real_space`` returns None.

Coalescence: grains that touch form a cluster (``clusters``); a cluster that connects two opposite edges of the
top view has percolated (``arrangement``: islands, coalescing or network). The picture draws a cluster as one
piece.

A porous TiO₂ layer (``porous_space``) is drawn from where the package says its metal sits: TiO₂
particles of the model's radius, the pore metal as a Gaussian in depth, the cap layer on a share of the particle
tops, and the clusters on top (``layer_particles``, ``pore_density``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Optional

POROUS_SKETCH_KEYS = ("particle_nm", "pore_depth_nm", "pore_width_nm", "cap_cover", "cap_thickness_nm",
                      "island_amount_nm", "cap_amount_nm", "pore_amount_nm", "cluster_height_nm",
                      "cluster_distance_nm", "cluster_cover")
SKETCH_KEYS = ("base_nm",) + POROUS_SKETCH_KEYS
"""Values of a solution the picture needs besides the shown quantities."""
TOUCH = 1e-9
"""Two grains belong to one cluster when their centres are no farther apart than the sum of their radii."""
LAYER_PARTICLES = 6
"""Width of the cross-section of a porous layer, in particle diameters."""
PARTICLE_GAP = 0.06
"""Distance between neighbouring TiO₂ particles of the picture beyond touching, as a fraction of their diameter:
the pores between them stay visible (illustration)."""
FIELD_DISTANCES = 7.0
SIDE_DISTANCES = 4.0
SHIFT = 0.12
"""The random shift of a grain from its place on the arrangement, as a fraction of the distance D (illustration)."""
SPREAD_LIMIT = 2.5
"""Radii are drawn from the log-normal distribution within this many widths of the median."""
SEED = 7


@dataclass(frozen=True)
class Grain:
    x: float
    y: float
    """Centre in nm (y: 0 in the side view)."""
    radius: float
    height: float
    """Height of the cap above the base layer, nm."""


@dataclass(frozen=True)
class RealSpace:
    radius: float
    distance: float
    width: float
    height: float
    """Top of the metal above the substrate (base layer + grains), nm."""
    base: float
    """Thickness of the continuous layer under the grains, nm (0: separate islands)."""

    @property
    def grain_height(self) -> float:
        return max(self.height - self.base, 0.0)

    @property
    def field(self) -> float:
        """Side of the square of the top view, nm."""
        return FIELD_DISTANCES * self.distance

    @property
    def side(self) -> float:
        """Length of the side view, nm."""
        return SIDE_DISTANCES * self.distance


def _positive(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def real_space(values: dict) -> Optional[RealSpace]:
    """The picture of a solution's values (``R_nm``, ``D_nm``, ``width``, ``H_nm``, ``base_nm``), or None without a
    radius and a distance. Without a height the grains are drawn as hemispheres on the base."""
    radius, distance = _positive(values.get("R_nm")), _positive(values.get("D_nm"))
    if radius is None or distance is None:
        return None
    width = _positive(values.get("width")) or 0.0
    base = _positive(values.get("base_nm")) or 0.0
    height = _positive(values.get("H_nm"))
    height = base + radius if height is None else max(height, base)
    return RealSpace(radius, distance, width, height, base)


def _draws(rng: random.Random, count: int) -> list:
    """``count`` (shift x, shift y, size) triples of standard normal numbers, the size clipped to ±SPREAD_LIMIT."""
    return [(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), max(-SPREAD_LIMIT, min(SPREAD_LIMIT, rng.gauss(0.0, 1.0))))
            for _ in range(count)]


def _grain(space: RealSpace, x: float, y: float, size: float) -> Grain:
    radius = space.radius * math.exp(space.width * size)
    return Grain(x, y, radius, space.grain_height * radius / space.radius)


def top_grains(space: RealSpace, seed: int = SEED) -> list:
    """The grains of the top view (some reach over the edge of the square: the view clips them)."""
    rows = int(math.ceil(FIELD_DISTANCES / (math.sqrt(3.0) / 2.0))) + 2
    columns = int(math.ceil(FIELD_DISTANCES)) + 2
    draws = _draws(random.Random(seed), rows * columns)
    pitch, step = space.distance, space.distance * math.sqrt(3.0) / 2.0
    grains = []
    for row in range(rows):
        for column in range(columns):
            dx, dy, size = draws[row * columns + column]
            x = (column - 0.5 + 0.5 * (row % 2)) * pitch + SHIFT * pitch * dx
            y = (row - 0.5) * step + SHIFT * pitch * dy
            grains.append(_grain(space, x, y, size))
    return grains


def side_grains(space: RealSpace, seed: int = SEED + 1) -> list:
    """The grains of the side view, along x (y = 0)."""
    count = int(math.ceil(SIDE_DISTANCES)) + 2
    draws = _draws(random.Random(seed), count)
    return [_grain(space, (index - 0.5) * space.distance + SHIFT * space.distance * dx, 0.0, size)
            for index, (dx, _dy, size) in enumerate(draws)]


def clusters(grains: list) -> list:
    """The cluster of each grain (an index, the same for grains that touch directly or through others)."""
    parent = list(range(len(grains)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, a in enumerate(grains):
        for second in range(first + 1, len(grains)):
            b = grains[second]
            reach = a.radius + b.radius + TOUCH
            if abs(a.x - b.x) <= reach and abs(a.y - b.y) <= reach and \
                    (a.x - b.x) ** 2 + (a.y - b.y) ** 2 <= reach ** 2:
                parent[root(second)] = root(first)
    return [root(index) for index in range(len(grains))]


def spanning(grains: list, labels: list, side: float) -> set:
    """The clusters that connect two opposite edges of the square of side ``side`` (left to right or bottom to top)."""
    edges: dict = {}
    for grain, label in zip(grains, labels):
        touched = edges.setdefault(label, set())
        for name, reached in (("left", grain.x - grain.radius <= 0.0), ("right", grain.x + grain.radius >= side),
                              ("bottom", grain.y - grain.radius <= 0.0), ("top", grain.y + grain.radius >= side)):
            if reached:
                touched.add(name)
    return {label for label, touched in edges.items() if {"left", "right"} <= touched or {"bottom", "top"} <= touched}


def visible(space: RealSpace, grains: list) -> list:
    """The grains of the top view that reach into its square."""
    return [grain for grain in grains if -grain.radius < grain.x < space.field + grain.radius
            and -grain.radius < grain.y < space.field + grain.radius]


def arrangement(space: RealSpace, grains: list) -> str:
    """``islands`` (no two grains of the top view touch), ``network`` (a cluster of touching grains crosses the
    view: percolated) or ``coalescing`` (grains touch, no cluster crosses)."""
    inside = visible(space, grains)
    labels = clusters(inside)
    if len(set(labels)) == len(inside):
        return "islands"
    return "network" if spanning(inside, labels, space.field) else "coalescing"


@dataclass(frozen=True)
class PorousSpace:
    """Where the metal of a solution in a porous TiO₂ layer sits (the package's ``porous_geometry``), nm."""

    particle: float
    """Radius of the TiO₂ particles."""
    island_radius: Optional[float]
    island_height: Optional[float]
    island_distance: Optional[float]
    width: float
    pore_depth: Optional[float]
    """Depth of the pore metal below the particle tops (centre of a Gaussian), None when the model has none."""
    pore_width: float
    cap_cover: float
    """Area fraction of the particle tops under the cap layer (0: no cap)."""
    cap_thickness: float
    island_amount: float
    cap_amount: float
    pore_amount: float
    """Equivalent thicknesses (nm of dense metal) of the islands, the cap and the pores."""
    deposited: float = 0.0
    """The metal deposited up to this window (nm)."""

    @property
    def depth_shown(self) -> float:
        """Depth of the cross-section: four particle diameters, or deeper to show the pore metal and its width."""
        deep = 8.0 * self.particle
        if self.pore_depth is not None:
            deep = max(deep, self.pore_depth + 3.0 * self.pore_width + self.particle)
        return 2.0 * self.particle * math.ceil(deep / (2.0 * self.particle))

    @property
    def side(self) -> float:
        return LAYER_PARTICLES * 2.0 * self.particle


def porous_space(values: dict) -> Optional[PorousSpace]:
    """The porous picture of a solution, or None when the package gave no porous geometry for it."""
    particle = _positive(values.get("particle_nm"))
    if particle is None or values.get("pore_amount_nm") is None:
        return None

    def number(key: str) -> float:
        return _positive(values.get(key)) or 0.0

    radius = _positive(values.get("R_nm"))
    height = _positive(values.get("cluster_height_nm")) or radius
    return PorousSpace(particle, radius, height, _positive(values.get("cluster_distance_nm")), number("width"),
                       _positive(values.get("pore_depth_nm")), number("pore_width_nm"), min(number("cap_cover"), 1.0),
                       number("cap_thickness_nm"), number("island_amount_nm"), number("cap_amount_nm"),
                       number("pore_amount_nm"), number("amount_nm"))


def layer_particles(space: PorousSpace, seed: int = SEED + 2) -> list:
    """The TiO₂ particles of the cross-section as (``Grain``, capped): centre and radius, y negative below the
    tops. The top row is capped in a fixed random order, as many particles as the cap's area fraction says (at least
    one when there is a cap)."""
    radius, pitch = space.particle, 2.0 * space.particle * (1.0 + PARTICLE_GAP)
    step = pitch * math.sqrt(3.0) / 2.0
    rows = int(math.ceil(space.depth_shown / step)) + 1
    columns = LAYER_PARTICLES + 2
    rng = random.Random(seed)
    draws = [(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.random()) for _ in range(rows * columns)]
    particles, top = [], []
    for row in range(rows):
        for column in range(columns):
            dx, dy, order = draws[row * columns + column]
            x = (column - 0.5 + 0.5 * (row % 2)) * pitch + 0.04 * pitch * dx
            y = -radius - row * step + (0.04 * pitch * dy if row else 0.0)
            particles.append(Grain(x, y, radius, 0.0))
            if row == 0 and 0.0 <= x <= space.side:
                top.append((order, len(particles) - 1))
    capped = set()
    if space.cap_cover > 0 and space.cap_thickness > 0 and top:
        count = max(1, int(round(space.cap_cover * len(top))))
        capped = {index for _order, index in sorted(top)[:count]}
    return [(particle, index in capped) for index, particle in enumerate(particles)]


def pore_density(space: PorousSpace, depth: float) -> float:
    """Relative density of the pore metal at ``depth`` below the tops (1 at its centre; 0 without a pore depth)."""
    if space.pore_depth is None:
        return 0.0
    width = max(space.pore_width, 1e-6)
    return math.exp(-0.5 * ((depth - space.pore_depth) / width) ** 2)


def drawn_coverage(space: RealSpace, grains: list) -> float:
    """The share of the top view's square under the grains as drawn (overlaps counted once on a fine grid)."""
    count, side = 200, space.field
    covered = 0
    for row in range(count):
        y = (row + 0.5) * side / count
        near = [grain for grain in grains if abs(grain.y - y) <= grain.radius]
        for column in range(count):
            x = (column + 0.5) * side / count
            if any((grain.x - x) ** 2 + (grain.y - y) ** 2 <= grain.radius ** 2 for grain in near):
                covered += 1
    return covered / (count * count)


__all__ = ["FIELD_DISTANCES", "Grain", "POROUS_SKETCH_KEYS", "PorousSpace", "RealSpace", "SHIFT",
           "SIDE_DISTANCES", "SKETCH_KEYS", "arrangement", "clusters", "drawn_coverage", "layer_particles",
           "pore_density", "porous_space", "real_space", "side_grains", "spanning", "top_grains", "visible"]
