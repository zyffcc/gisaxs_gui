"""Real-space picture of one solution of the analysis package (Python and NumPy): where its grains sit, in nm.

A solution of a grain model family gives the grain radius R, the centre distance D, the log-normal width of the radius
distribution, the height H of the metal above the substrate and the thickness of the continuous layer under the grains
(``base_nm``; 0 for separate islands). The picture shows them at their size:

- top view: a square of ``FIELD_DISTANCES`` x D; the grains on a hexagonal arrangement of distance D, each shifted at
  random by about ``SHIFT`` x D (an arrangement without long-range order, as the model's interference function has),
  each with a radius drawn from the log-normal distribution of median R. A base layer covers the whole field:
  the film is closed (``RealSpace.continuous``) and the grains are relief on it.
- side view: a cut ``SIDE_DISTANCES`` x D long through one row of grains: the base layer, and on it each grain as a cap
  of its radius and of the height H - base, scaled with its radius.

The random numbers come from one seed and are drawn in the same order for every solution, so moving through the windows
or the candidates changes only the sizes, never the dice. A solution without a radius or a distance (a family that has
no grain arrangement, e.g. metal in a porous layer) has no grain picture: ``real_space`` returns None.

Coalescence: grains that touch form a cluster (``clusters``); a cluster that connects two opposite edges of the
top view has percolated (``arrangement``: islands, coalescing or network). The picture draws a cluster as one
piece.
A connected film - a continuous base, or grains joined into a network across the view - is drawn as the film itself:
``height_map`` gives the height of the metal over the top view (base plus the highest cap, 0 in the holes) and
``profile`` the cut along one line for the side view, instead of grain outlines.

A porous TiO₂ layer (``porous_space``) is drawn from where the package says its metal sits: TiO₂
particles of the model's radius, the pore metal as a Gaussian in depth, the cap layer on a share of the particle
tops, and the clusters on top (``layer_particles``, ``pore_density``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import random
from typing import Optional

import numpy as np

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
    field_nm: Optional[float] = None
    """Side of the top view fixed for a whole series (``grown``), nm; None: ``FIELD_DISTANCES`` x D."""
    cover: Optional[float] = None
    """The solution's coverage (area share of the metal seen from above), when it gives one."""
    channels: bool = False
    """A growing film (``grown``): the continuous layer lies only under the metal footprint, the rest are channels
    down to the substrate (holes of the film)."""

    @property
    def continuous(self) -> bool:
        """A closed film: the grains sit on a continuous layer of the metal."""
        return self.base > 0

    @property
    def grain_height(self) -> float:
        return max(self.height - self.base, 0.0)

    @property
    def field(self) -> float:
        """Side of the square of the top view, nm."""
        return self.field_nm if self.field_nm else FIELD_DISTANCES * self.distance

    @property
    def side(self) -> float:
        """Length of the side view, nm (a growing series: the cut across the whole top view)."""
        return self.field_nm if self.field_nm else SIDE_DISTANCES * self.distance


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
    cover = _positive(values.get("coverage"))
    return RealSpace(radius, distance, width, height, base, cover=min(cover, 1.0) if cover else None)


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


# -- a series drawn as one growing film ---------------------------------------------------------------------------
MAX_GROWN = 1200
"""At most this many grains in the field of a growing series: the field is made smaller when the densest window
(the smallest D) would need more."""
NUCLEUS_TRIES = 40
"""Random places tried for a new nucleus; the one farthest from the grains already there is taken."""


def series_field(spaces) -> Optional[float]:
    """Side of the top view shared by every window of a series, nm: ``FIELD_DISTANCES`` x the largest D, smaller when
    the smallest D would put more than ``MAX_GROWN`` grains into it."""
    drawn = [space for space in spaces if space is not None]
    if not drawn:
        return None
    densest = 2.0 / (math.sqrt(3.0) * min(space.distance for space in drawn) ** 2)
    return min(FIELD_DISTANCES * max(space.distance for space in drawn), math.sqrt(MAX_GROWN / densest))


def _count(space: RealSpace, field: float) -> int:
    """Grains of the hexagonal number density of the distance D in the field."""
    return max(1, int(round(2.0 / (math.sqrt(3.0) * space.distance ** 2) * field * field)))


ROUND_UP = 0.45
"""Below this coverage merging islands round up into one (coalescence of small islands); above it they stay joined as
they are and grow together into a connected film whose open parts are channels."""
COVER_MAX = 0.97
"""The largest coverage drawn (a film always keeps some channels or pits to show)."""
CLOSED_COVER = 0.85
"""A window with a continuous layer (base > 0) is drawn with at least this share of metal: the model's layer is
laterally homogeneous (its density below 1 stands for voids and channels the package does not export), so the
channels of the picture take at most the rest (illustration)."""
COVER_GRID = 128
"""Points per side of the grid the drawn coverage is measured on."""
NEAREST = (8, 32, 128)
"""Grains looked up per grid point, stage by stage, before the exact comparison with all (``_cover_scale``)."""


def _cover_scale(x, y, z, width: float, field: float, cover: float) -> float:
    """The common radius scale s (radius of grain i = s exp(width z_i)) at which the union of the grains covers the
    share ``cover`` of the field: a grid point g is covered from s = min_i |g - x_i| exp(-width z_i) on."""
    axis = (np.arange(COVER_GRID) + 0.5) * field / COVER_GRID
    gx, gy = (a.ravel() for a in np.meshgrid(axis, axis))
    weight = np.exp(-width * np.asarray(z, float))
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    need = np.full(gx.size, np.inf)
    rest = np.arange(gx.size)
    # 2026-11-24 (a slow Sputter page: this was most of the time of opening a result): the nearest grains of every grid
    # point first (a k-d tree), k at a time; a farther grain can only come first where the k-th distance times the
    # smallest weight is below the best found, and only those points go on. The same scale as comparing every grain
    if len(xs):
        from scipy.spatial import cKDTree

        tree, grid, smallest = cKDTree(np.column_stack((xs, ys))), np.column_stack((gx, gy)), float(weight.min())
        for k in NEAREST:
            if k >= len(xs) or not rest.size:
                break
            distance, index = tree.query(grid[rest], k=k)
            need[rest] = np.minimum(need[rest], (distance * weight[index]).min(axis=1))
            rest = rest[distance[:, -1] * smallest < need[rest]]
    for first in range(0, len(xs), 256):                     # in blocks: grid points x grains
        sl = slice(first, first + 256)
        d = np.hypot(gx[rest, None] - xs[None, sl], gy[rest, None] - ys[None, sl]) * weight[None, sl]
        need[rest] = np.minimum(need[rest], d.min(axis=1))
    return float(np.quantile(need, min(max(cover, 0.0), COVER_MAX)))


@lru_cache(maxsize=8)
def _growth(spaces: tuple, seed: int) -> tuple:
    """The grains of every window of a series as one growing film: (x, y, z, s) per window (None where the window has
    no picture); grain i has the radius s exp(width z_i). Window by window, in order:
      - while the film is open (drawn coverage below ``ROUND_UP``), the number density of the solution (1 / D^2) sets
        the number of islands: new nuclei appear at free places when it asks for more (nucleation), and when it asks
        for fewer the pairs of islands closest to touching merge into one round island of their joint area at their
        area-weighted centre (coalescence of small islands);
      - from then on no island disappears or moves: they grow together into connected clusters and a film, and what
        stays open are the channels between them;
      - the drawn coverage is the solution's coverage (or, without one, that of grains of radius R at the distance D)
        and never falls: the scale s is set so that the union of the grains covers that share of the field.
    A grain keeps its place and its size draw z (a standard normal number): nothing is drawn again from one window to
    the next."""
    field = series_field(spaces)
    rng = random.Random(seed)
    x, y, z = np.zeros(0), np.zeros(0), np.zeros(0)
    drawn, states = 0.0, []
    for space in spaces:
        if space is None:
            states.append(None)
            continue
        target = _count(space, field)
        if drawn < ROUND_UP and target > len(x):              # nucleation: new grains at free places
            new_x, new_y, new_z = list(x), list(y), list(z)
            for _ in range(target - len(x)):
                tries = [(rng.uniform(0.0, field), rng.uniform(0.0, field)) for _ in range(NUCLEUS_TRIES)]
                if new_x:
                    px, py = np.asarray(new_x), np.asarray(new_y)
                    room = [float(np.min((px - a) ** 2 + (py - b) ** 2)) for a, b in tries]
                    a, b = tries[int(np.argmax(room))]
                else:
                    a, b = tries[0]
                new_x.append(a)
                new_y.append(b)
                new_z.append(max(-SPREAD_LIMIT, min(SPREAD_LIMIT, rng.gauss(0.0, 1.0))))
            x, y, z = np.asarray(new_x), np.asarray(new_y), np.asarray(new_z)
        while drawn < ROUND_UP and len(x) > target:           # coalescence of small islands: they round up
            r = space.radius * np.exp(space.width * z)
            gap = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2) - (r[:, None] + r[None, :])
            first, second = np.triu_indices(len(x), 1)
            order = np.argsort(gap[first, second], kind="stable")
            used, merges = set(), []
            for pair in order:
                i, j = int(first[pair]), int(second[pair])
                if i in used or j in used:
                    continue
                used.update((i, j))
                merges.append((i, j))
                if len(merges) >= len(x) - target:
                    break
            keep = np.ones(len(x), bool)
            for i, j in merges:
                wi, wj = r[i] ** 2, r[j] ** 2                 # area-weighted centre, joint area
                x[i], y[i] = (wi * x[i] + wj * x[j]) / (wi + wj), (wi * y[i] + wj * y[j]) / (wi + wj)
                if space.width > 0:
                    z[i] = min(math.log(math.sqrt(wi + wj) / space.radius) / space.width, 3 * SPREAD_LIMIT)
                keep[j] = False
            x, y, z = x[keep], y[keep], z[keep]
        cover = space.cover
        if cover is None:                                     # coverage of grains of radius R at the distance D
            cover = min(math.pi * space.radius ** 2 * 2.0 / (math.sqrt(3.0) * space.distance ** 2), COVER_MAX)
        if space.base > 0:                                    # a closed film: channels only between the metal
            cover = max(cover, CLOSED_COVER)
        cover = max(cover, drawn)                             # the film never opens up again
        scale = _cover_scale(x, y, z, space.width, field, cover)
        drawn = cover
        states.append((x.copy(), y.copy(), z.copy(), scale))
    return tuple(states)


def grown(spaces, index: int, seed: int = SEED):
    """Window ``index`` of a series (one ``RealSpace`` or None per window, in time or thickness order) drawn as one
    growing film: (space with the series' fixed field, grains of the top view), or None when that window has no
    picture. The grains of a window are those of the window before, grown, merged or joined by new nuclei
    (``_growth``)."""
    spaces = tuple(spaces)
    if not 0 <= index < len(spaces) or spaces[index] is None:
        return None
    xs, ys, zs, scale = _growth(spaces, seed)[index]
    space = replace(spaces[index], field_nm=series_field(spaces), channels=True)
    grains = []
    for a, b, c in zip(xs, ys, zs):
        size = math.exp(space.width * float(c))
        grains.append(Grain(float(a), float(b), scale * size, space.grain_height * size))
    return space, visible(space, grains)


@lru_cache(maxsize=16)
def _ceiling(spaces: tuple, seed: int) -> float:
    top = 0.0
    for index, space in enumerate(spaces):
        drawn = grown(spaces, index, seed) if space is not None else None
        if drawn is None:
            continue
        space, grains = drawn
        top = max(top, space.height, space.radius, *(space.base + grain.height for grain in grains))
    return top


def series_ceiling(spaces, seed: int = SEED) -> float:
    """The highest metal of a series drawn as one growing film (``grown``): over every window with a picture, its top
    (``RealSpace.height``), its radius and the base plus the tallest of its grown grains, nm (0 without any picture).
    The side view keeps it as its height in every window, so the picture keeps its size while the film grows."""
    return _ceiling(tuple(spaces), seed)


def cut(grains: list, y: float) -> list:
    """The grains crossed by the line y of the top view, as caps of the side view: the chord half-width and the cap
    height on that line."""
    row = []
    for grain in grains:
        dy = y - grain.y
        if abs(dy) < grain.radius:
            share = math.sqrt(max(1.0 - (dy / grain.radius) ** 2, 0.0))
            row.append(Grain(grain.x, 0.0, grain.radius * share, grain.height * share))
    return sorted(row, key=lambda grain: grain.x)


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


MAP_POINTS = 220
"""Points per side of the height map of a connected film (top view)."""


def cap_height(grain: Grain, dx, dy):
    """Height of a grain's cap above the base at the offsets (dx, dy) from its centre (0 outside it): a half
    spheroid of the grain's radius and height. Works on numbers and on NumPy arrays."""
    r2 = (np.asarray(dx, float) ** 2 + np.asarray(dy, float) ** 2) / max(grain.radius, 1e-9) ** 2
    return grain.height * np.sqrt(np.clip(1.0 - r2, 0.0, None))


def height_map(space: RealSpace, grains: list, points: int = MAP_POINTS):
    """(x, y, h): the height of the metal above the substrate on a square grid over the top view - the base plus the
    highest cap at each point; 0 where neither covers (a hole of a connected network). For a film whose grains have
    joined, the picture of the film itself instead of grain outlines."""
    axis = (np.arange(points) + 0.5) * space.field / points
    x, y = np.meshgrid(axis, axis)
    caps = np.zeros_like(x)
    for grain in grains:
        reach = grain.radius
        cols = np.flatnonzero(np.abs(axis - grain.x) <= reach)
        rows = np.flatnonzero(np.abs(axis - grain.y) <= reach)
        if not len(cols) or not len(rows):
            continue
        block = np.ix_(rows, cols)
        caps[block] = np.maximum(caps[block], cap_height(grain, x[block] - grain.x, y[block] - grain.y))
    covered = caps > 0
    if space.channels:                                       # a growing film: the layer only under the footprint
        inside = np.zeros_like(x, dtype=bool)
        for grain in grains:
            cols = np.flatnonzero(np.abs(axis - grain.x) <= grain.radius)
            rows = np.flatnonzero(np.abs(axis - grain.y) <= grain.radius)
            if len(cols) and len(rows):
                block = np.ix_(rows, cols)
                inside[block] |= (x[block] - grain.x) ** 2 + (y[block] - grain.y) ** 2 <= grain.radius ** 2
        return axis, axis, np.where(inside, space.base + caps, 0.0)
    height = np.where(covered, space.base + caps, space.base)
    return axis, axis, height


def profile(space: RealSpace, grains: list, y: float, points: int = 600):
    """(x, h): the height of the metal along the line y of the top view (the side view of a connected film)."""
    x = (np.arange(points) + 0.5) * space.side / points
    caps = np.zeros_like(x)
    inside = np.zeros_like(x, dtype=bool)
    for grain in grains:
        dy = y - grain.y
        if abs(dy) < grain.radius:
            caps = np.maximum(caps, cap_height(grain, x - grain.x, dy))
            inside |= (x - grain.x) ** 2 + dy ** 2 <= grain.radius ** 2
    if space.channels:                                       # a growing film: channels down to the substrate
        return x, np.where(inside, space.base + caps, 0.0)
    return x, np.where(caps > 0, space.base + caps, space.base)


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


__all__ = ["CLOSED_COVER", "COVER_MAX", "FIELD_DISTANCES", "Grain", "MAX_GROWN", "ROUND_UP", "cut", "grown", "series_field", "POROUS_SKETCH_KEYS", "PorousSpace", "RealSpace", "SHIFT",
           "SIDE_DISTANCES", "SKETCH_KEYS", "arrangement", "clusters", "drawn_coverage", "layer_particles",
           "pore_density", "porous_space", "real_space", "side_grains", "spanning", "top_grains", "visible"]
