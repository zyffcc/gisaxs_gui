"""The evolution of the solutions chosen for the windows of a result (Matplotlib).

Left, the lengths the result has (nm): grain radius R, grain height H, centre distance D and the gap between grains;
on a porous substrate the island radius (a relative size) and the vertical and lateral correlation lengths of the
metal network. Right, its shares: the coverage, or on a porous substrate the island, cap and film shares. Both
against the time in the scan — or, for a series of static samples, against what the series says its axis is (the
nominal thickness), with the range of a value where the package gives one.

Each point is the entry shown for its window (the user's choice, else the window's default); a quantity a window does
not have leaves a gap in its line; samples at one position are joined through their median. The window shown above is
marked on both plots (a line at its position, a ring around its points); a click selects the window nearest to it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from .texts import text

SIZE_SERIES = (("R_nm", "Radius R", "#1f77b4"), ("H_nm", "Height H", "#2a9d5c"), ("D_nm", "Distance D", "#9467bd"),
               ("gap_nm", "Gap", "#e08a1e"), ("depth_nm", "Vertical correlation", "#2a9d5c"),
               ("xi_nm", "Lateral correlation", "#9467bd"))
"""(quantity, name, colour) of the lines of the size plot: data colours, readable in both themes."""
SHARE_SERIES = (("coverage", "Coverage", "#17a2b8"), ("f_islands", "Island share", "#1f77b4"),
                ("cap_share", "Cap share", "#e08a1e"), ("film_share", "Film share", "#8c564b"))
"""(quantity, name, colour) of the lines of the share plot."""
RELATIVE_RADIUS = "Island radius (relative)"
"""The radius of a result on a porous substrate."""
MARK_COLOUR = "#d62728"
TIME_LABEL = "Time in the scan (s)"
NOMINAL_LABEL = "Nominal thickness"
SIZE_LABEL = "Size (nm)"
COVERAGE_LABEL = "Coverage"
SHARE_LABEL = "Share"
TEMPLATES = (TIME_LABEL, NOMINAL_LABEL, SIZE_LABEL, COVERAGE_LABEL, SHARE_LABEL, RELATIVE_RADIUS,
             *(name for _key, name, _colour in (*SIZE_SERIES, *SHARE_SERIES)))
MARGINS = {"left": 68, "right": 14, "top": 26, "bottom": 44, "between": 64}
"""The room around the two plots, in pixels (the left edge is that of the images above)."""
WIDTHS = (2.4, 1.0)
"""The share of the width of the size plot and of the share plot."""
WIDTHS_SHARES = (1.9, 1.5)
"""The same when the share plot holds several lines (their names stand above it)."""
SAME_POSITION = 3.0
"""Windows no farther apart than this (pixels along the axis) are told apart by the height of the click."""


def _finite(values) -> list:
    return [value for value in values if value is not None and math.isfinite(value)]


def axis_label(result) -> str:
    """What the horizontal axis is: the time in the scan, or what a series of static samples says (with its unit)."""
    if not result.static:
        return tr(TIME_LABEL)
    name = result.x_label.strip() or NOMINAL_LABEL
    shown = tr(name[:1].upper() + name[1:])  # the package writes it in lower case
    return f"{shown} ({result.x_unit})" if result.x_unit else shown


class EvolutionPlot:
    def __init__(self, layout, on_window: Optional[Callable[[int], None]] = None):
        """``on_window(index)``: the user clicked the plots nearest to window ``index``."""
        self.canvas = None
        self.figure = None
        self._on_window = on_window
        self._last: Optional[tuple] = None
        self._marks: list = []
        self._rings: list = []
        self._axes: Optional[tuple] = None
        self._drawn: tuple = ([], [])
        self._positions: list = []
        self.series: dict = {}
        """Quantity -> (positions, values) as drawn."""
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # matplotlib missing: the values stay
            layout.addWidget(QLabel(text("Plotting is unavailable: {error}", error=exc)))
            return
        self.figure = Figure(figsize=(9, 2.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(170)
        self.canvas.setCursor(Qt.PointingHandCursor)
        self.canvas.mpl_connect("button_press_event", self._clicked)
        self.canvas.mpl_connect("resize_event", lambda _event: self._place())
        layout.addWidget(self.canvas, 1)

    def draw(self, result, choices: Optional[dict], current: Optional[int]) -> None:
        """The chosen entries of every window of ``result`` (None: nothing), window ``current`` marked."""
        self._last = (result, dict(choices or {}), current)
        self._draw()

    def mark(self, current: Optional[int]) -> None:
        """Move the mark to window ``current`` (the lines stay)."""
        if self._last is None or self.canvas is None:
            return
        self._last = (*self._last[:2], current)
        self._show_mark(current)
        self.canvas.draw_idle()

    def redraw(self) -> None:
        """Again, in the colours of the theme and the language of the interface."""
        if self._last is not None:
            self._draw()

    def window_at(self, plot: int, x: float, y: Optional[float] = None) -> Optional[int]:
        """The window nearest to the data position (x, y) of plot ``plot`` (0: sizes, 1: shares): nearest along the
        axis, and among windows at one position the one whose point is nearest in height."""
        if self._axes is None or not self._positions or not 0 <= plot < len(self._axes):
            return None
        to_pixels = self._axes[plot].transData.transform
        click = to_pixels((x, 0.0 if y is None else y))
        along = [abs(to_pixels((position, 0.0))[0] - click[0]) for position in self._positions]
        near = [index for index, distance in enumerate(along) if distance <= min(along) + SAME_POSITION]
        if len(near) == 1 or y is None:
            return min(near, key=lambda index: along[index])

        def height(index: int) -> float:
            values = _finite([values[index] for values in self._drawn[plot]])
            return min((abs(to_pixels((self._positions[index], value))[1] - click[1]) for value in values),
                       default=math.inf)

        return min(near, key=lambda index: (height(index), along[index]))

    def _clicked(self, event) -> None:
        if self._on_window is None or self._axes is None or event.button != 1 or event.xdata is None:
            return
        if event.inaxes in self._axes:
            index = self.window_at(self._axes.index(event.inaxes), float(event.xdata), event.ydata)
            if index is not None:
                self._on_window(index)

    def _draw(self) -> None:
        if self.canvas is None:
            return
        from src.gimap.app.presentation.theme.figures import restyle_figure
        from src.gimap.shared.figures import cjk_fallback

        result, choices, current = self._last
        figure = self.figure
        figure.clear()
        self._marks, self._rings, self._axes, self.series = [], [], None, {}
        self._drawn, self._positions = ([], []), []
        if result is not None and result.windows:
            self._plots(result, choices)
            self._show_mark(current)
        restyle_figure(figure)
        cjk_fallback(figure, self._texts())
        self.canvas.draw_idle()

    def _texts(self) -> list:
        """The texts that are in the interface language: the axis titles and the legends."""
        texts = []
        for axes in self._axes or ():
            texts += [axes.xaxis.label, axes.yaxis.label]
            if axes.get_legend() is not None:
                texts += list(axes.get_legend().get_texts())
        return texts

    def _place(self) -> None:
        """The two plots inside the canvas, with the margins in pixels (they keep their place when values change)."""
        if self._axes is None:
            return
        width, height = (max(1.0, value) for value in self.figure.get_size_inches() * self.figure.dpi)
        m = MARGINS
        widths = WIDTHS_SHARES if len(self._drawn[1]) > 1 else WIDTHS
        room = max(20.0, width - m["left"] - m["right"] - m["between"])
        first = room * widths[0] / sum(widths)
        tall = max(10.0, height - m["top"] - m["bottom"]) / height
        sizes, shares = self._axes
        sizes.set_position([m["left"] / width, m["bottom"] / height, first / width, tall])
        shares.set_position([(m["left"] + first + m["between"]) / width, m["bottom"] / height,
                             (room - first) / width, tall])

    def _show_mark(self, current: Optional[int]) -> None:
        valid = current is not None and 0 <= current < len(self._positions)
        position = self._positions[current] if valid else math.nan
        for line in self._marks:
            line.set_xdata([position, position])
        for ring, drawn in zip(self._rings, self._drawn):
            heights = _finite([values[current] for values in drawn]) if valid else []
            ring.set_data([position] * len(heights), heights)

    def _line(self, axes, values: list, spread: tuple, colour: str, name: str) -> None:
        positions = self._positions
        distinct = sorted(set(positions))
        if len(distinct) == len(positions):
            axes.plot(positions, values, "-o", color=colour, markersize=3.5, linewidth=1.4, label=name)
        else:  # several samples at one position: every point, and a line through the medians
            medians = []
            for position in distinct:
                here = sorted(_finite([value for at, value in zip(positions, values) if at == position]))
                middle = len(here) // 2
                medians.append(math.nan if not here else here[middle] if len(here) % 2
                               else 0.5 * (here[middle - 1] + here[middle]))
            axes.plot(distinct, medians, "-", color=colour, linewidth=1.2)
            axes.plot(positions, values, "o", color=colour, markersize=4.5, label=name)
        lows, highs = spread
        if _finite(lows) and _finite(highs):
            below = [max(0.0, value - low) if math.isfinite(low) else math.nan for value, low in zip(values, lows)]
            above = [max(0.0, high - value) if math.isfinite(high) else math.nan for value, high in zip(values, highs)]
            axes.errorbar(positions, values, yerr=[below, above], fmt="none", ecolor=colour, elinewidth=1.0,
                          capsize=2.5, alpha=0.85)

    def _plots(self, result, choices: dict) -> None:
        sizes = self.figure.add_axes([0.08, 0.2, 0.55, 0.7])
        shares = self.figure.add_axes([0.72, 0.2, 0.25, 0.7])
        self._axes = (sizes, shares)
        self._positions = [result.position(window) for window in result.windows]
        porous = result.porous
        limits = ([], [])
        for plot, (axes, listed) in enumerate(((sizes, SIZE_SERIES), (shares, SHARE_SERIES))):
            for key, name, colour in listed:
                _positions, values = result.series(key, choices)
                self.series[key] = (self._positions, values)
                if not _finite(values):
                    continue
                spread = result.spread(key, choices)
                self._line(axes, values, spread, colour, tr(RELATIVE_RADIUS if porous and key == "R_nm" else name))
                self._drawn[plot].append(values)
                limits[plot].extend(_finite(values) + _finite(spread[0]) + _finite(spread[1]))
        several = len(self._drawn[1]) > 1
        sizes.set_ylabel(tr(SIZE_LABEL))
        shares.set_ylabel(tr(SHARE_LABEL if several or (self._drawn[1] and "coverage" not in self._shown(1))
                             else COVERAGE_LABEL))
        if limits[0]:
            sizes.set_ylim(bottom=min(0.0, min(limits[0])))
        if limits[1]:
            shares.set_ylim(0.0, max(1.0, 1.05 * max(limits[1])))
        for axes, drawn in ((sizes, self._drawn[0]), (shares, self._drawn[1] if several else [])):
            if drawn:  # above the plot: never over a line
                axes.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=max(1, len(drawn)), fontsize=8,
                            frameon=False, handlelength=1.4, columnspacing=1.2, borderaxespad=0.2)
        label = axis_label(result)
        for axes in (sizes, shares):
            axes.set_xlabel(label)
            axes.grid(True, alpha=0.25)
            self._marks.append(axes.axvline(math.nan, color=MARK_COLOUR, linewidth=1.2, linestyle="--"))
            self._rings.append(axes.plot([], [], "o", markersize=9, markerfacecolor="none",
                                         markeredgecolor=MARK_COLOUR, markeredgewidth=1.3, zorder=5)[0])
        self._place()

    def _shown(self, plot: int) -> list:
        """The quantities drawn in a plot."""
        listed = (SIZE_SERIES, SHARE_SERIES)[plot]
        return [key for key, _name, _colour in listed if key in self.series and _finite(self.series[key][1])]


__all__ = ["EvolutionPlot", "RELATIVE_RADIUS", "SHARE_SERIES", "SIZE_SERIES", "TEMPLATES", "axis_label"]
