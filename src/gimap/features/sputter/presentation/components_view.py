"""The model image of the solution shown split into its physical terms, and their shares over the scan (Matplotlib).

Left, one small image per term — grains and islands, large clusters, the film fluctuation or network, the substrate
seen through the metal, fluorescence, the beam-path background … — the largest first, at most ``MAX_MAPS``; each is
the term's share of the model per cell times the model image, on the cells of the result and with the one colour
scale of the measured image (``pattern_view.colour_limits``): a term that contributes little looks dark. Its title
gives its share of the model counts. Right, the share of every term over the scan for the solution shown in each
window (logarithmic), the window shown marked.

A result of an older package has no terms: the picture says so in one line.
"""

from __future__ import annotations

import math
import textwrap
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from .pattern_view import (COLOUR_MAP, X_LABEL, Y_LABEL, cell_edges, colour_limits, decade_ticks, raster_cells,
                           shown)
from .texts import text

COLOURS = {"grains": "#1f77b4", "large_clusters": "#e08a1e", "film": "#2a9d5c", "voids": "#8c564b",
           "buried": "#9467bd", "domains": "#e377c2", "base_roughness": "#bcbd22", "clusters_in_pores": "#1f77b4",
           "clusters_on_cap": "#e08a1e", "pore_network": "#2a9d5c", "substrate": "#7f7f7f",
           "fluorescence": "#d62728", "background": "#17becf"}
"""The colour of each term in the chart of shares: fixed per term, readable in both themes."""
MAX_MAPS = 6
TITLE_CHARACTERS = 28
"""A term's name wraps onto a second line beyond this many characters (its image is narrow)."""
SMALLEST_SHARE = 0.001
"""Terms below this share of the model counts get no image (they are in the chart)."""
MAP_TITLE = "{name} · {share}"
SHARES_TITLE = "Share of the model counts"
NO_COMPONENTS = ("This result does not split the model into its physical terms: analyze it again with the current "
                 "analysis package.")
MORE_TERMS = "Also: {names}"
TEMPLATES = (MAP_TITLE, SHARES_TITLE, NO_COMPONENTS, MORE_TERMS)


def percent(share: float) -> str:
    """A share as a percentage: one decimal, two significant digits below 1 %."""
    value = 100.0 * share
    return f"{value:.1f} %" if abs(value) >= 1 else f"{value:.2g} %"


class ComponentsPicture:
    def __init__(self, layout):
        self.canvas = None
        self.figure = None
        self._last = None
        self.titles: tuple = ()
        """The titles of the term images as drawn."""
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

    def show(self, patterns, label: str, shares: dict, names: dict, series: Optional[tuple] = None,
             current: Optional[int] = None, x_label: str = "") -> None:
        """The terms of entry ``label`` (``shares``: term -> share of its counts; ``names``: term -> shown name) and
        ``series`` (positions, {term: shares}) over the scan with window ``current`` marked."""
        self._last = ("terms", patterns, label, dict(shares), dict(names), series, current, x_label)
        self._draw()

    def show_message(self, message: str) -> None:
        self._last = ("message", message)
        self._draw()

    def redraw(self) -> None:
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
        self.titles = ()
        if self._last[0] == "message":
            figure.text(.5, .5, self._last[1], ha="center", va="center", fontsize=10, wrap=True)
        else:
            self._terms(*self._last[1:])
        restyle_figure(figure)
        texts = list(figure.texts)
        for axes in figure.axes:
            texts += [axes.title, axes.xaxis.label, axes.yaxis.label]
            legend = axes.get_legend()
            if legend is not None:
                texts += list(legend.get_texts())
        cjk_fallback(figure, texts)
        self.canvas.draw_idle()

    def _terms(self, patterns, label, shares, names, series, current, x_label) -> None:
        figure = self.figure
        images = dict(patterns.components(label)) if patterns is not None else {}
        if not images:
            figure.text(.5, .5, tr(NO_COMPONENTS), ha="center", va="center", fontsize=10, wrap=True)
            return
        order = sorted(images, key=lambda key: -abs(shares.get(key, 0.0)))
        mapped = [key for key in order if shares.get(key, 0.0) >= SMALLEST_SHARE][:MAX_MAPS]
        rest = [key for key in order if key not in mapped and abs(shares.get(key, 0.0)) >= SMALLEST_SHARE]
        columns = min(3, max(1, len(mapped)))
        rows = max(1, math.ceil(len(mapped) / columns))
        grid = figure.add_gridspec(rows, columns + 2, left=0.07, right=0.80, bottom=0.17, top=0.86, wspace=0.14,
                                   hspace=0.6, width_ratios=[1.0] * columns + [0.25, 1.35])  # a gap, then the chart
        model = patterns.model(label)
        limits = colour_limits(patterns.measured, model)
        log_q = bool(np.all(patterns.q > 0))
        x = cell_edges(patterns.q, positive=log_q)
        y = cell_edges(patterns.af)
        cols = raster_cells(x, int(np.clip(3 * patterns.q.size, 150, 600)), log=log_q)
        rws = raster_cells(y, int(np.clip(2 * patterns.af.size, 120, 450)))
        extent = (math.log10(x[0]), math.log10(x[-1]), y[0], y[-1]) if log_q else (x[0], x[-1], y[0], y[-1])
        titles, first = [], None
        for index, key in enumerate(mapped):
            axes = figure.add_subplot(grid[index // columns, index % columns], sharex=first, sharey=first)
            first = first or axes
            if limits is not None:
                from matplotlib.colors import LogNorm

                axes.imshow(shown(images[key], limits[0])[np.ix_(rws, cols)], origin="lower", extent=extent,
                            aspect="auto", interpolation="nearest", norm=LogNorm(*limits), cmap=COLOUR_MAP)
            share = percent(shares.get(key, 0.0)).replace(" ", " ")  # the number keeps its unit on wrapping
            title = textwrap.fill(text(MAP_TITLE, name=names.get(key, key), share=share), TITLE_CHARACTERS)
            axes.set_title(title, fontsize=8)
            titles.append(title)
            axes.set_xlim(extent[0], extent[1])
            axes.set_ylim(extent[2], extent[3])
            axes.tick_params(labelsize=7)
            if index % columns:
                axes.tick_params(labelleft=False)
            else:
                axes.set_ylabel(tr(Y_LABEL), fontsize=8)
            if index // columns == rows - 1:
                axes.set_xlabel(X_LABEL, fontsize=8)
            else:
                axes.tick_params(labelbottom=False)
        if log_q and first is not None:
            majors, labels, _minors = decade_ticks(extent[0], extent[1])
            first.set_xticks(majors)
            first.set_xticklabels(labels)
        self.titles = tuple(titles)
        if rest:
            figure.text(0.07, 0.02, text(MORE_TERMS, names=", ".join(
                f"{names.get(key, key)} {percent(shares.get(key, 0.0))}" for key in rest)), fontsize=7)
        if series is not None:
            self._chart(figure.add_subplot(grid[:, columns + 1]), series, current, names, x_label)

    def _chart(self, axes, series, current, names, x_label) -> None:
        positions, shares = series
        for key, values in shares.items():
            values = np.asarray(values, dtype=float)
            if not np.any(np.isfinite(values) & (values >= SMALLEST_SHARE)):
                continue
            axes.plot(positions, np.where(values > 0, values, np.nan), "-o", markersize=3, linewidth=1.6,
                      color=COLOURS.get(key, "#7f7f7f"), label=names.get(key, key))
        axes.set_yscale("log")
        axes.set_ylim(SMALLEST_SHARE, 1.5)
        if current is not None and 0 <= current < len(positions):
            axes.axvline(positions[current], color="#d62728", linewidth=1.0, alpha=0.6)
        axes.set_title(tr(SHARES_TITLE), fontsize=8)
        axes.set_xlabel(x_label, fontsize=8)
        axes.tick_params(labelsize=7)
        if axes.lines:  # beside the chart: never over a line
            axes.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0,
                        handlelength=1.4)


__all__ = ["COLOURS", "ComponentsPicture", "MAX_MAPS", "TEMPLATES", "percent"]
