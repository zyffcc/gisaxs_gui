"""The measured image of a time window beside the model image of the chosen solution (Matplotlib).

Both images are drawn on the cells of the result (exit angle against q_y; neither needs to be evenly spaced), on the
same axes — q_y on a logarithmic axis when every q_y is positive — and with one logarithmic colour scale. Its limits
are the 1 % and 99.7 % points of the positive measured count rate, so they do not move when another solution is
chosen and a model that is too bright or too dark shows as such. Cells without data stay empty in both.

Each image is drawn as a picture: every point of the picture takes the value of the data cell it lies in
(``raster_cells``), so cells keep their true extent and no value is interpolated or changed. That is fast enough to
follow the window slider: while the cells stay the same (the windows of one result), another window or another
solution only puts new values into the two pictures and the colour scale. The axes keep their place from one window
to the next (the margins are fixed, in pixels).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from .texts import text

MEASURED_TITLE = "Measured image"
MODEL_TITLE = "Model image: {name}"
MODEL_MISSING = "Model image"
X_LABEL = "q_y (nm⁻¹)"
Y_LABEL = "Exit angle αf (°)"
COLOUR_LABEL = "Count rate"
LEVEL_PERCENTILES = (1.0, 99.7)
COLOUR_MAP = "viridis"
TEMPLATES = (MEASURED_TITLE, MODEL_TITLE, MODEL_MISSING, Y_LABEL, COLOUR_LABEL)
MARGINS = {"left": 68, "right": 82, "top": 24, "bottom": 54, "between": 16, "bar": 12, "bar_gap": 10}
"""The room around the two images, in pixels: the axis labels, the titles and the colour bar keep their place."""
PICTURE_ROWS = (240, 900)
PICTURE_COLUMNS = (300, 1200)
"""The least and the most rows and columns of the picture an image is drawn as (three and six per data cell)."""


def cell_edges(centres: np.ndarray, *, positive: bool = False) -> np.ndarray:
    """The edges of the cells whose centres are ``centres`` (halfway between neighbours; ``positive``: the first
    edge stays above zero, for a logarithmic axis)."""
    centres = np.asarray(centres, dtype=float)
    middle = 0.5 * (centres[1:] + centres[:-1])
    edges = np.concatenate(([2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]))
    if positive and edges[0] <= 0:
        edges[0] = 0.5 * min(centres[0], edges[1])
    return edges


def colour_limits(measured: np.ndarray, fallback: Optional[np.ndarray] = None) -> Optional[tuple]:
    """(low, high) of the logarithmic colour scale: the 1 % and 99.7 % points of the positive measured values (of
    ``fallback`` when nothing was measured); None when there is nothing positive to show."""
    for values in (measured, fallback):
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            low, high = (float(value) for value in np.percentile(values, LEVEL_PERCENTILES))
            return (low, high) if high > low else (low, low * 10.0)
    return None


def shown(values: np.ndarray, low: float) -> np.ma.MaskedArray:
    """``values`` for the logarithmic scale: cells without data masked, zero counts at the lowest colour."""
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.ma.masked_invalid(np.where(values < low, low, values))


def raster_cells(edges: np.ndarray, count: int, *, log: bool = False) -> np.ndarray:
    """For ``count`` picture points evenly spaced between the first and the last edge (evenly in log₁₀ when
    ``log``): the index of the data cell each point lies in."""
    edges = np.asarray(edges, dtype=float)
    low, high = (math.log10(edges[0]), math.log10(edges[-1])) if log else (edges[0], edges[-1])
    points = low + (np.arange(count) + 0.5) * (high - low) / count
    points = 10.0 ** points if log else points
    return np.clip(np.searchsorted(edges, points, side="right") - 1, 0, edges.size - 2)


def decade_ticks(low: float, high: float) -> tuple:
    """Ticks of a logarithmic axis drawn in log₁₀ units from ``low`` to ``high``: (major positions, their labels,
    minor positions). Whole decades when the axis holds two or more, else 1, 2 and 5 of each decade."""
    decades = range(math.floor(low) - 1, math.ceil(high) + 1)
    inside = [k for k in decades if low <= k <= high]
    if len(inside) >= 2:
        majors, labels = [float(k) for k in inside], [f"$10^{{{k}}}$" for k in inside]
    else:
        values = [m * 10.0 ** k for k in decades for m in (1, 2, 5)]
        values = [value for value in values if low <= math.log10(value) <= high]
        majors, labels = [math.log10(value) for value in values], [f"{value:g}" for value in values]
    minors = [math.log10(m * 10.0 ** k) for k in decades for m in range(2, 10)]
    return majors, labels, [u for u in minors if low <= u <= high and all(abs(u - major) > 1e-9 for major in majors)]


class PatternPair:
    def __init__(self, layout):
        self.canvas = None
        self.figure = None
        self._last = None
        self._axes: Optional[tuple] = None  # (measured, model, colour bar) while the two images are drawn
        self._drawn: Optional[dict] = None  # the cells, the two pictures and their colour scale as drawn
        self.titles = ("", "")
        """The titles of the two images as drawn (measured, model)."""
        self.limits: Optional[tuple] = None
        """The limits of the colour scale as drawn."""
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
        self.canvas.mpl_connect("resize_event", lambda _event: self._place())
        layout.addWidget(self.canvas, 1)

    def show(self, patterns, label: str, name: str) -> None:
        """The measured image and the model of entry ``label`` (shown as ``name``); ``patterns`` None: nothing."""
        self._last = ("patterns", patterns, label, name)
        if not self._update(patterns, label, name):
            self._draw()

    def show_message(self, message: str) -> None:
        """A sentence in place of the images (a window without a solution, images that cannot be read)."""
        self._last = ("message", message)
        self._draw()

    def redraw(self) -> None:
        """Again, in the colours of the theme and the language of the interface."""
        if self._last is not None:
            self._draw()

    def model_values(self) -> Optional[np.ndarray]:
        """The model array drawn now (None when no model is drawn)."""
        if self._last is None or self._last[0] != "patterns" or self._last[1] is None:
            return None
        return self._last[1].model(self._last[2])

    # -- drawing -------------------------------------------------------------------------------------------------
    def _draw(self) -> None:
        if self.canvas is None:
            return
        from src.gimap.app.presentation.theme.figures import restyle_figure
        from src.gimap.shared.figures import cjk_fallback

        figure = self.figure
        figure.clear()
        self._axes, self._drawn, self.titles, self.limits = None, None, ("", ""), None
        if self._last[0] == "message":
            figure.text(.5, .5, self._last[1], ha="center", va="center", fontsize=10, wrap=True)
        elif self._last[1] is not None:
            self._images(*self._last[1:])
        restyle_figure(figure)
        cjk_fallback(figure, self._texts())
        self.canvas.draw_idle()

    def _texts(self) -> list:
        """The texts that are in the interface language: the message, the titles and the axis titles."""
        texts = list(self.figure.texts)
        for axes in self._axes or ():
            texts += [axes.title, axes.xaxis.label, axes.yaxis.label]
        return texts

    def _place(self) -> None:
        """The two images and the colour bar inside the canvas, with the margins in pixels."""
        if self._axes is None:
            return
        width, height = (max(1.0, value) for value in self.figure.get_size_inches() * self.figure.dpi)
        m = MARGINS
        room = width - m["left"] - m["right"] - m["between"]
        each, tall = max(10.0, room / 2.0), max(10.0, height - m["top"] - m["bottom"])
        bottom = m["bottom"] / height
        measured, model, bar = self._axes
        measured.set_position([m["left"] / width, bottom, each / width, tall / height])
        model.set_position([(m["left"] + each + m["between"]) / width, bottom, each / width, tall / height])
        bar.set_position([(width - m["right"] + m["bar_gap"]) / width, bottom, m["bar"] / width, tall / height])

    def _images(self, patterns, label: str, name: str) -> None:
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import FixedLocator

        model = patterns.model(label)
        limits = colour_limits(patterns.measured, model)
        figure = self.figure
        left = figure.add_axes([0.1, 0.2, 0.3, 0.7])
        right = figure.add_axes([0.5, 0.2, 0.3, 0.7], sharex=left, sharey=left)
        bar = figure.add_axes([0.9, 0.2, 0.02, 0.7])
        self._axes = (left, right, bar)
        self.titles = (tr(MEASURED_TITLE), text(MODEL_TITLE, name=name) if model is not None else tr(MODEL_MISSING))
        log_q = bool(np.all(patterns.q > 0))
        x = cell_edges(patterns.q, positive=log_q)
        y = cell_edges(patterns.af)
        columns = raster_cells(x, int(np.clip(6 * patterns.q.size, *PICTURE_COLUMNS)), log=log_q)
        rows = raster_cells(y, int(np.clip(3 * patterns.af.size, *PICTURE_ROWS)))
        extent = (math.log10(x[0]), math.log10(x[-1]), y[0], y[-1]) if log_q else (x[0], x[-1], y[0], y[-1])
        norm = LogNorm(*limits) if limits is not None else None  # one scale for both images
        images = []
        for axes, values, title in ((left, patterns.measured, self.titles[0]), (right, model, self.titles[1])):
            axes.set_title(title, fontsize=10)
            axes.set_xlabel(X_LABEL)
            if values is not None and norm is not None:
                picture = shown(values, limits[0])[np.ix_(rows, columns)]
                images.append(axes.imshow(picture, origin="lower", extent=extent, aspect="auto",
                                          interpolation="nearest", norm=norm, cmap=COLOUR_MAP))
            axes.set_xlim(extent[0], extent[1])
            axes.set_ylim(extent[2], extent[3])
        if log_q:  # a logarithmic q axis, drawn in log10 units
            majors, labels, minors = decade_ticks(extent[0], extent[1])
            left.set_xticks(majors)
            left.set_xticklabels(labels)
            left.xaxis.set_minor_locator(FixedLocator(minors))
        left.set_ylabel(tr(Y_LABEL))
        right.tick_params(labelleft=False)
        self.limits = limits
        if images:
            figure.colorbar(images[0], cax=bar, label=tr(COLOUR_LABEL))
        else:
            bar.set_visible(False)
        if len(images) == 2:  # the usual case: the next window or solution only changes the values
            self._drawn = {"q": patterns.q, "af": patterns.af, "rows": rows, "columns": columns, "images": images,
                           "norm": norm}
        self._place()

    def _update(self, patterns, label: str, name: str) -> bool:
        """Put another window's or solution's values into the two pictures drawn (False: they have to be drawn
        anew — nothing is drawn yet, the cells differ, or there is no model or nothing to show)."""
        drawn = self._drawn
        if drawn is None or patterns is None or self.canvas is None:
            return False
        model = patterns.model(label)
        limits = colour_limits(patterns.measured, model)
        if model is None or limits is None or not (np.array_equal(drawn["q"], patterns.q)
                                                   and np.array_equal(drawn["af"], patterns.af)):
            return False
        cells = np.ix_(drawn["rows"], drawn["columns"])
        if limits != self.limits:
            norm = drawn["norm"]
            with norm.callbacks.blocked():  # both limits first: the scale is never seen with one old and one new
                norm.vmin, norm.vmax = limits
            norm.callbacks.process("changed")  # the two images and the colour bar follow
            self.limits = limits
        for image, values in zip(drawn["images"], (patterns.measured, model)):
            image.set_data(shown(values, limits[0])[cells])
        self.titles = (self.titles[0], text(MODEL_TITLE, name=name))
        self._axes[1].set_title(self.titles[1], fontsize=10)
        from src.gimap.shared.figures import cjk_fallback

        cjk_fallback(self.figure, [self._axes[1].title])
        self.canvas.draw_idle()
        return True


__all__ = ["PatternPair", "TEMPLATES", "cell_edges", "colour_limits", "decade_ticks", "raster_cells", "shown"]
