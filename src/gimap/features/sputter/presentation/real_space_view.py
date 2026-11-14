"""The chosen solution of a window drawn in real space, in place of the two images (Matplotlib).

**Grains** (a flat substrate or a polymer film). Left a side view: a cut through one row of grains — the substrate (a
polymer film is drawn, its thickness not to scale), the continuous base layer and the grains as caps of their radius
and height, on one nm scale in both directions. Right the top view of a square of 7 distances D: the grains (log-normal
radii around R, arranged at the distance D without long-range order), the base layer under them as a tint. Grains that
touch are drawn as one piece (an outline around the cluster only); a cluster that crosses the view — the film has
percolated — is drawn darker, and a line says whether the grains are separate, coalescing or a connected network.

**A porous TiO₂ layer.** Left a cross-section of the layer to scale: TiO₂ particles of the model's radius, the metal in
the pores as a tint whose strength follows its Gaussian in depth, the cap layer on the share of the particle tops it
covers. Right the top of one capped particle, enlarged: the cap at its thickness and the clusters at their radius,
height and distance. Below, where the metal is, as equivalent thicknesses of dense metal.

The positions are drawn at random once (``domain/real_space.py``); the sizes are the solution's.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from ..application import (FILM_SUBSTRATES, arrangement, clusters, layer_particles, pore_density, porous_space,
                           real_space, side_grains, spanning, top_grains, visible)
from .sketch_view import INK, METAL_COLOURS, POLYMER, POLYMER_EDGE, SUBSTRATE
from .texts import text

SIDE_TITLE = "side view (cut)"
TOP_TITLE = "top view"
X_LABEL = "x (nm)"
Y_LABEL = "y (nm)"
HEIGHT_LABEL = "height (nm)"
POLYMER_NOTE = "polymer (thickness not to scale)"
BASE_NOTE = "continuous layer {thickness} nm"
DRAWN_NOTE = ("Grains placed at random around the distance D; radius, size distribution, height and base layer are "
              "those of the solution: {name}")
ARRANGEMENTS = {"islands": "Separate islands",
                "coalescing": "Coalescing: neighbouring grains touch",
                "network": "Connected network across the view (percolated)"}
ON_LAYER = "Continuous layer of {thickness} nm; the grains on it: {grains}"
GRAINS_ON_LAYER = {"islands": "separate", "coalescing": "touching their neighbours",
                   "network": "connected across the view"}
LAYER_TITLE = "cross-section of the TiO₂ layer"
ZOOM_TITLE = "top of a particle (enlarged)"
DEPTH_LABEL = "depth below the particle tops (nm)"
PORE_NOTE = "pore metal around {depth} nm (width {width} nm)"
CAP_NOTE = "cap on {cover} of the tops, {thickness} nm"
NO_CAP = "no cap layer"
POROUS_NOTE = ("Of the {deposited} nm deposited, as dense-metal thickness: islands {islands} nm · cap {cap} nm · "
               "pores {pores} nm. TiO₂ particles of {particle} nm radius; positions drawn at random: {name}")
NO_GRAINS = ("This solution has no arrangement to draw: the package gave neither a grain radius and distance nor where "
             "the metal sits in a porous layer.")
TEMPLATES = (SIDE_TITLE, TOP_TITLE, X_LABEL, Y_LABEL, HEIGHT_LABEL, POLYMER_NOTE, BASE_NOTE, DRAWN_NOTE,
             *ARRANGEMENTS.values(), ON_LAYER, *GRAINS_ON_LAYER.values(), LAYER_TITLE, ZOOM_TITLE, DEPTH_LABEL, PORE_NOTE, CAP_NOTE, NO_CAP,
             POROUS_NOTE, NO_GRAINS)
LIGHT = 0.45
"""Opacity of the base layer's tint in the top view (the grains on it stay fully opaque)."""
TIO2, TIO2_EDGE = "#f4efe4", "#b3a487"
"""TiO₂ particles (an off-white ceramic, so that silver, gold and copper all stand out)."""
CONTINUOUS = 0.3
"""A base layer at least this thick (nm) is named in the line about the arrangement."""


def cap_outline(x: float, base: float, radius: float, height: float, points: int = 48) -> np.ndarray:
    """The outline of a grain in the side view: half an ellipse of half-width ``radius`` and height ``height``."""
    angle = np.linspace(0.0, np.pi, points)
    return np.column_stack((x + radius * np.cos(angle), base + max(height, 0.0) * np.sin(angle)))


def _mix(first: str, second: str, share: float) -> tuple:
    from matplotlib.colors import to_rgb

    a, b = to_rgb(first), to_rgb(second)
    return tuple((1 - share) * u + share * v for u, v in zip(a, b))


def _number(value: float) -> str:
    return f"{value:.3g}"


def _union(axes, patches_of, fills: list, edge: str, zorder: float) -> None:
    """Patches drawn as one piece where they overlap: outlines first, then the fills over every inner outline."""
    from matplotlib.collections import PatchCollection

    axes.add_collection(PatchCollection(patches_of(), facecolor="none", edgecolor=edge, linewidth=1.4, zorder=zorder))
    axes.add_collection(PatchCollection(patches_of(), facecolor=fills, edgecolor="none", zorder=zorder + 0.1))


class RealSpacePicture:
    def __init__(self, layout):
        self.canvas = None
        self.figure = None
        self._last = None
        self._axes: Optional[tuple] = None  # (left view, right view) while a solution is drawn
        self.drawn = None
        """The ``RealSpace`` or ``PorousSpace`` drawn now (None: a message or nothing)."""
        self.arrangement = ""
        """``islands``, ``coalescing`` or ``network`` for grains drawn now ("" otherwise)."""
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # matplotlib missing: the values stay
            layout.addWidget(QLabel(text("Plotting is unavailable: {error}", error=exc)))
            return
        self.figure = Figure(figsize=(9, 3.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(220)
        layout.addWidget(self.canvas, 1)

    def show(self, values: dict, metal: str, substrate: str, name: str) -> None:
        """The solution with ``values`` (its quantities and the picture's own keys), shown as ``name``."""
        self._last = ("space", dict(values), metal, substrate, name)
        self._draw()

    def show_message(self, message: str) -> None:
        self._last = ("message", message)
        self._draw()

    def redraw(self) -> None:
        """Again, in the colours of the theme and the language of the interface."""
        if self._last is not None:
            self._draw()

    # -- drawing -------------------------------------------------------------------------------------------------
    def _draw(self) -> None:
        if self.canvas is None:
            return
        from src.gimap.app.presentation.theme.figures import restyle_figure
        from src.gimap.shared.figures import cjk_fallback

        figure = self.figure
        figure.clear()
        self._axes, self.drawn, self.arrangement = None, None, ""
        if self._last[0] == "message":
            figure.text(.5, .5, self._last[1], ha="center", va="center", fontsize=10, wrap=True)
        else:
            values, metal, substrate, name = self._last[1:]
            grains, porous = real_space(values), porous_space(values)
            if porous is not None:
                self._porous(porous, metal, name)
            elif grains is not None:
                self._grains(grains, metal, substrate, name)
            else:
                figure.text(.5, .5, tr(NO_GRAINS), ha="center", va="center", fontsize=10, wrap=True)
        restyle_figure(figure)
        cjk_fallback(figure, self._texts())
        self.canvas.draw_idle()

    def _texts(self) -> list:
        """Every text of the picture (the titles stand at the left: ``axes.title`` is the centred one)."""
        from matplotlib.text import Text

        return [item for item in self.figure.findobj(Text) if item.get_text()]

    def _frame(self, axes, title: str, x_label: str, y_label: str) -> None:
        axes.set_title(tr(title), fontsize=9, loc="left")
        axes.set_xlabel(tr(x_label), fontsize=8)
        axes.set_ylabel(tr(y_label), fontsize=8)
        axes.tick_params(labelsize=7)

    # -- grains on a flat substrate or a polymer film -------------------------------------------------------------
    def _grains(self, space, metal: str, substrate: str, name: str) -> None:
        from matplotlib.patches import Circle, Polygon, Rectangle

        fill, edge = METAL_COLOURS.get(metal, METAL_COLOURS["Ag"])
        deep = _mix(fill, edge, 0.45)
        polymer = substrate in FILM_SUBSTRATES
        figure = self.figure
        side = figure.add_axes([0.07, 0.2, 0.52, 0.66])
        top = figure.add_axes([0.66, 0.2, 0.3, 0.66])
        self._axes, self.drawn = (side, top), space

        # top view first: which clusters cross it (they are drawn darker in both views)
        grains = visible(space, top_grains(space))
        labels = clusters(grains)
        crossing = spanning(grains, labels, space.field)
        self.arrangement = arrangement(space, grains)

        # side view: substrate (or polymer film), base layer, grains (touching caps as one piece)
        row = side_grains(space)
        tallest = max([space.base + grain.height for grain in row] + [space.height, space.radius])
        below = max(0.3 * tallest, 0.15 * space.side / 4.0)
        side.add_patch(Rectangle((0, -below), space.side, below, facecolor=POLYMER if polymer else SUBSTRATE,
                                 edgecolor=POLYMER_EDGE if polymer else "none", lw=.6, zorder=1))
        side.text(space.side * .985, -below * .5, tr(POLYMER_NOTE) if polymer else substrate, ha="right",
                  va="center", fontsize=7, color=INK, zorder=6)
        if space.base > 0:
            side.add_patch(Rectangle((0, 0), space.side, space.base, facecolor=fill, edgecolor=edge, lw=.6, zorder=2))
            side.text(space.side * .015, space.base * .5, text(BASE_NOTE, thickness=_number(space.base)), ha="left",
                      va="center", fontsize=7, color=INK, zorder=6)
        _union(side, lambda: [Polygon(cap_outline(grain.x, space.base, grain.radius, grain.height), closed=True)
                              for grain in row], [fill] * len(row), edge, 3)
        side.set_xlim(0, space.side)
        side.set_ylim(-below, tallest * 1.25)
        side.set_aspect("equal", adjustable="box")
        self._frame(side, SIDE_TITLE, X_LABEL, HEIGHT_LABEL)

        # top view: the base layer as a tint, the grains on it, a cluster as one piece
        field = space.field
        top.add_patch(Rectangle((0, 0), field, field, facecolor=POLYMER if polymer else SUBSTRATE, lw=0, zorder=1))
        if space.base > 0:
            top.add_patch(Rectangle((0, 0), field, field, facecolor=fill, alpha=LIGHT, lw=0, zorder=2))
        _union(top, lambda: [Circle((grain.x, grain.y), grain.radius) for grain in grains],
               [deep if label in crossing else fill for label in labels], edge, 3)
        top.set_xlim(0, field)
        top.set_ylim(0, field)
        top.set_aspect("equal", adjustable="box")
        self._frame(top, TOP_TITLE, X_LABEL, Y_LABEL)
        words = tr(ARRANGEMENTS[self.arrangement])
        if space.base >= CONTINUOUS:  # a closed film: what is said is the layer, then the grains on it
            words = text(ON_LAYER, thickness=_number(space.base), grains=tr(GRAINS_ON_LAYER[self.arrangement]))
        figure.text(0.96, 0.93, words, ha="right", va="bottom", fontsize=8.5, fontweight="bold")
        figure.text(0.07, 0.03, text(DRAWN_NOTE, name=name), ha="left", va="bottom", fontsize=7.5)

    # -- a porous TiO₂ layer ----------------------------------------------------------------------------------------
    def _porous(self, space, metal: str, name: str) -> None:
        from matplotlib.patches import Circle, Polygon, Wedge

        fill, edge = METAL_COLOURS.get(metal, METAL_COLOURS["Ag"])
        figure = self.figure
        layer = figure.add_axes([0.07, 0.2, 0.36, 0.66])
        zoom = figure.add_axes([0.53, 0.2, 0.43, 0.66])
        self._axes, self.drawn = (layer, zoom), space

        # cross-section to scale: the pore metal as a tint along depth, the particles over it, the cap on some tops
        side, deep = space.side, space.depth_shown
        depths = np.linspace(0.0, deep, 400)
        strength = np.array([pore_density(space, depth) for depth in depths])
        if strength.max() > 0:
            colour = np.zeros((len(depths), 1, 4))
            colour[:, 0, :3] = _mix(fill, edge, 0.7)
            colour[:, 0, 3] = 0.85 * strength
            layer.imshow(colour, extent=(0, side, -deep, 0), origin="upper", aspect="auto", zorder=1,
                         interpolation="bilinear")
        particles = layer_particles(space)
        layer.add_collection(_collection([Circle((p.x, p.y), p.radius) for p, _capped in particles], TIO2, TIO2_EDGE))
        if space.cap_cover > 0 and space.cap_thickness > 0:
            for particle, capped in particles:
                if capped:
                    layer.add_patch(Wedge((particle.x, particle.y), particle.radius + space.cap_thickness, 20, 160,
                                          width=space.cap_thickness, facecolor=fill, edgecolor=edge, lw=.5, zorder=3))
        if space.pore_depth is not None:
            layer.axhline(-space.pore_depth, color=edge, lw=.8, ls="--", zorder=4)
            layer.text(side * .98, -space.pore_depth, text(PORE_NOTE, depth=_number(space.pore_depth),
                                                           width=_number(space.pore_width)),
                       ha="right", va="bottom", fontsize=7, color=INK, zorder=5,
                       bbox=dict(facecolor="white", alpha=.7, lw=0, pad=1))
        from matplotlib.ticker import FuncFormatter

        layer.set_xlim(0, side)
        layer.set_ylim(-deep, max(0.15 * space.particle, 1.3 * space.cap_thickness + 5.0))
        layer.set_aspect("equal", adjustable="box")
        layer.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _number(abs(value)) if value <= 0 else ""))
        self._frame(layer, LAYER_TITLE, X_LABEL, DEPTH_LABEL)

        # one particle top, enlarged: the cap at its thickness, the clusters at their size
        radius = space.island_radius or 0.0
        distance = space.island_distance or (3.0 * radius if radius else 0.0)
        width = max(10.0 * distance, 12.0 * radius, 4.0 * space.cap_thickness, 20.0)
        centre, rp = width / 2.0, space.particle
        zoom.add_patch(Circle((centre, -rp), rp, facecolor=TIO2, edgecolor=TIO2_EDGE, lw=.8, zorder=1))
        cap = space.cap_thickness if space.cap_cover > 0 else 0.0

        def surface(x: float) -> float:
            return -rp + math.sqrt(max(rp * rp - (x - centre) ** 2, 0.0)) + cap

        if cap > 0:
            xs = np.linspace(0.0, width, 120)
            outer = [(x, surface(x)) for x in xs]
            inner = [(x, surface(x) - cap) for x in xs[::-1]]
            zoom.add_patch(Polygon(outer + inner, closed=True, facecolor=fill, edgecolor=edge, lw=.6, zorder=2))
        if radius > 0 and distance > 0:
            height = space.island_height or radius
            count = int(width / distance) + 2
            rng = np.random.default_rng(11)
            shifts, sizes = rng.normal(0.0, 0.12, count), np.clip(rng.normal(0.0, 1.0, count), -2.5, 2.5)
            caps = []
            for index in range(count):
                x = (index + 0.5) * distance + shifts[index] * distance
                r = radius * math.exp(space.width * sizes[index])
                caps.append(cap_outline(x, surface(x), r, height * r / radius))
            _union(zoom, lambda: [Polygon(outline, closed=True) for outline in caps], [fill] * len(caps), edge, 3)
        top_height = max(cap + 2.5 * (space.island_height or radius or 1.0), 3.0)
        zoom.set_xlim(0, width)
        zoom.set_ylim(-max(0.35 * width, 4.0), top_height)
        zoom.set_aspect("equal", adjustable="box")
        self._frame(zoom, ZOOM_TITLE, X_LABEL, HEIGHT_LABEL)
        cap_words = text(CAP_NOTE, cover=f"{space.cap_cover:.0%}", thickness=_number(space.cap_thickness)) if cap > 0 \
            else tr(NO_CAP)
        figure.text(0.96, 0.93, cap_words, ha="right", va="bottom", fontsize=8.5, fontweight="bold")
        figure.text(0.07, 0.03, text(POROUS_NOTE, islands=_number(space.island_amount), cap=_number(space.cap_amount),
                                     pores=_number(space.pore_amount), particle=_number(space.particle),
                                     deposited=_number(space.deposited), name=name),
                    ha="left", va="bottom", fontsize=7.5)


def _collection(patches: list, face: str, edge_colour: str):
    from matplotlib.collections import PatchCollection

    return PatchCollection(patches, facecolor=face, edgecolor=edge_colour, linewidth=.6, zorder=2)


__all__ = ["ARRANGEMENTS", "GRAINS_ON_LAYER", "RealSpacePicture", "TEMPLATES", "cap_outline"]
