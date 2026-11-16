"""The pictures of the live view while the frames are read (Matplotlib): the detector and the frames read last, and the
count rate of every frame read so far.

``FramesPicture``: left, the detector summed over the frames the automatic mask was made from (log₁₀ of the counts on
blocks of pixels, as the package sends it), the blocks the mask leaves out tinted red, the direct beam marked + and
the sample horizon dashed; right, the count rate of the chunk of frames read last on the grid of q_y (both sides of
the beam: one side has negative q_y) and exit angle, with a logarithmic colour scale from the 1 % and 99.7 % points
of its positive values. The cells keep their extent; the gap between the two sides stays empty. The next chunk only
puts new values into the right picture.

``TracePlot``: the count rate per pixel of every frame against its time in the scan — it rises once metal is
deposited —, and the start of the deposition when the import found it.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from .pattern_view import COLOUR_MAP, cell_edges, colour_limits
from .texts import text

DETECTOR_TITLE = "Detector, summed"
FRAMES_TITLE = "Frames {first}–{last} s"
FRAMES_WAITING = "Frames read"
X_LABEL = "q_y (nm⁻¹)"
Y_LABEL = "Exit angle αf (°)"
COLOUR_LABEL = "Count rate"
TRACE_X = "Time in the scan (s)"
TRACE_Y = "Count rate per pixel"
ONSET_LABEL = "Start of the deposition"
TEMPLATES = (DETECTOR_TITLE, FRAMES_TITLE, FRAMES_WAITING, Y_LABEL, COLOUR_LABEL, TRACE_X, TRACE_Y, ONSET_LABEL)
MASK_COLOUR = (0.86, 0.15, 0.15)
MASK_ALPHA = 0.45
"""How strongly the blocks the mask leaves out are tinted (the detector stays visible under them)."""
BEAM_COLOUR = "#00e5ff"
GAP_FACTOR = 5.0
"""Frames farther apart than this many times the usual spacing are not joined by the line (chunks of frames are read
in parallel and arrive out of order: the gap fills when its chunk arrives)."""
TRACE_COLOUR = "#1f77b4"
MARGINS = {"left": 12, "gap": 60, "right": 82, "top": 26, "bottom": 50, "bar": 12, "bar_gap": 10}
"""Room around the two pictures, in pixels (the detector has no axes: it is a picture of the detector)."""
DETECTOR_SHARE = 0.42
"""The share of the width the detector takes (it keeps its aspect inside it)."""
TRACE_MARGINS = {"left": 68, "right": 14, "top": 14, "bottom": 44}


def _canvas(layout, size: tuple, minimum: int):
    """(figure, canvas) in ``layout``; (None, None) with a sentence when Matplotlib is missing."""
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
    except Exception as exc:  # matplotlib missing: the steps still show
        layout.addWidget(QLabel(text("Plotting is unavailable: {error}", error=exc)))
        return None, None
    figure = Figure(figsize=size, dpi=100)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    canvas.setMinimumHeight(minimum)
    layout.addWidget(canvas, 1)
    return figure, canvas


def _finish(figure, canvas, texts) -> None:
    from src.gimap.app.presentation.theme.figures import restyle_figure
    from src.gimap.shared.figures import cjk_fallback

    restyle_figure(figure)
    cjk_fallback(figure, texts)
    canvas.draw_idle()


def side_edges(q: np.ndarray) -> tuple:
    """(edges along q_y, the columns of the values in the picture) for cells on both sides of the beam: the cells of
    each side keep their own edges and one empty column fills the gap between the sides."""
    q = np.asarray(q, dtype=float)
    negative, positive = q[q < 0], q[q > 0]
    if negative.size < 2 or positive.size < 2:
        return cell_edges(q), np.arange(q.size)
    left, right = cell_edges(negative), cell_edges(positive)
    gap = right[0] > left[-1]
    edges = np.concatenate((left, right if gap else right[1:]))
    columns = np.r_[np.arange(negative.size), negative.size + int(gap) + np.arange(positive.size)]
    return edges, columns


def with_gaps(times, rates) -> tuple:
    """(times, rates) with a NaN wherever two neighbouring frames are farther apart than GAP_FACTOR x the usual
    spacing: the line is not drawn across frames that have not been read yet."""
    times, rates = np.asarray(times, dtype=float), np.asarray(rates, dtype=float)
    if times.size < 3:
        return times, rates
    steps = np.diff(times)
    usual = float(np.median(steps))
    breaks = np.flatnonzero(steps > GAP_FACTOR * usual) + 1 if usual > 0 else np.array([], dtype=int)
    if not breaks.size:
        return times, rates
    return np.insert(times, breaks, np.nan), np.insert(rates, breaks, np.nan)


class FramesPicture:
    def __init__(self, layout):
        self.figure, self.canvas = _canvas(layout, (9, 3.2), 220)
        if self.canvas is not None:
            self.canvas.mpl_connect("resize_event", lambda _event: self._place())
        self._axes: Optional[tuple] = None
        self._mesh = None
        self._norm = None
        self._cells: Optional[tuple] = None
        self._last: tuple = (None, None)
        self._frames_title = ""
        self.titles = ("", "")

    def show(self, detector, frames, frames_title: str = "") -> None:
        """``detector``: the event of the summed detector (or None); ``frames``: the event of the chunk read last (or
        of the usable frames of a sample); ``frames_title``: the right picture's title (in the interface language) when
        the frames have no times."""
        same = self._last[0] is detector and self._axes is not None and frames_title == self._frames_title
        self._last, self._frames_title = (detector, frames), frames_title
        if not (same and self._update(frames)):
            self._draw()

    def redraw(self) -> None:
        if self._axes is not None or self._last != (None, None):
            self._draw()

    # -- drawing -------------------------------------------------------------------------------------------------
    def _draw(self) -> None:
        if self.canvas is None:
            return
        figure = self.figure
        figure.clear()
        detector_axes = figure.add_axes([0.0, 0.1, 0.4, 0.8])
        frames_axes = figure.add_axes([0.5, 0.15, 0.35, 0.75])
        bar = figure.add_axes([0.9, 0.15, 0.02, 0.75])
        self._axes = (detector_axes, frames_axes, bar)
        self._mesh, self._norm, self._cells = None, None, None
        self._detector(detector_axes, self._last[0])
        frames_axes.set_xlabel(X_LABEL)
        frames_axes.set_ylabel(tr(Y_LABEL))
        bar.set_visible(False)
        self._frames(self._last[1])
        self._place()
        _finish(figure, self.canvas, self._texts())

    def _texts(self) -> list:
        return [axes.title for axes in self._axes[:2]] + [self._axes[1].yaxis.label, self._axes[2].yaxis.label]

    def _detector(self, axes, event) -> None:
        axes.set_axis_off()
        axes.set_title(tr(DETECTOR_TITLE), fontsize=10)
        self.titles = (tr(DETECTOR_TITLE), self.titles[1])
        if event is None:
            return
        image = np.asarray(event.arrays.get("image"), dtype=float)
        if image.ndim != 2:
            return
        finite = image[np.isfinite(image)]
        low, high = (np.percentile(finite, (1.0, 99.8)) if finite.size else (0.0, 1.0))
        axes.imshow(np.ma.masked_invalid(image), cmap=COLOUR_MAP, vmin=low, vmax=max(high, low + 1e-6),
                    interpolation="nearest", origin="upper")
        masked = np.asarray(event.arrays.get("masked", np.zeros_like(image)), dtype=float)
        if masked.shape == image.shape:
            tint = np.zeros(image.shape + (4,))
            tint[..., :3] = MASK_COLOUR
            tint[..., 3] = np.clip(masked, 0.0, 1.0) * MASK_ALPHA
            axes.imshow(tint, interpolation="nearest", origin="upper")
        col, row, horizon = (event.number(key) for key in ("col", "row", "horizon_row"))
        if horizon is not None:
            axes.axhline(horizon, color="white", linewidth=0.9, linestyle="--", alpha=0.8)
        if col is not None and row is not None:
            axes.plot([col], [row], "+", color=BEAM_COLOUR, markersize=16, markeredgewidth=2.2)
        axes.set_xlim(-0.5, image.shape[1] - 0.5)
        axes.set_ylim(image.shape[0] - 0.5, -0.5)

    def _frames(self, event) -> None:
        from matplotlib.colors import LogNorm

        axes, bar = self._axes[1], self._axes[2]
        self.titles = (self.titles[0], self._frames_title or tr(FRAMES_WAITING))
        axes.set_title(self.titles[1], fontsize=10)
        cells = self._grid(event)
        if cells is None:
            return
        q_edges, af_edges, columns, rate = cells
        limits = colour_limits(rate)
        if limits is None:
            return
        self._norm = LogNorm(*limits)
        values = self._values(rate, columns, q_edges.size - 1, limits[0])
        self._mesh = axes.pcolormesh(q_edges, af_edges, values, cmap=COLOUR_MAP, norm=self._norm, shading="flat")
        self._cells = (q_edges, af_edges)
        axes.set_xlim(q_edges[0], q_edges[-1])
        axes.set_ylim(af_edges[0], af_edges[-1])
        self._title(event)
        bar.set_visible(True)
        self.figure.colorbar(self._mesh, cax=bar, label=tr(COLOUR_LABEL))

    @staticmethod
    def _grid(event) -> Optional[tuple]:
        if event is None:
            return None
        try:
            q = np.asarray(event.arrays["q"], dtype=float).ravel()
            af = np.asarray(event.arrays["af"], dtype=float).ravel()
            rate = np.asarray(event.arrays["rate"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        if rate.shape != (af.size, q.size) or q.size < 2 or af.size < 2:
            return None
        q_edges, columns = side_edges(q)
        return q_edges, cell_edges(af), columns, rate

    @staticmethod
    def _values(rate: np.ndarray, columns: np.ndarray, count: int, low: float) -> np.ma.MaskedArray:
        values = np.full((rate.shape[0], count), np.nan)
        values[:, columns] = rate
        with np.errstate(invalid="ignore"):
            return np.ma.masked_invalid(np.where(values < low, low, values))

    def _title(self, event) -> None:
        first, last = event.number("t_first"), event.number("t_last")
        if first is not None and last is not None and not self._frames_title:
            self.titles = (self.titles[0], text(FRAMES_TITLE, first=f"{first:.1f}", last=f"{last:.1f}"))
            self._axes[1].set_title(self.titles[1], fontsize=10)

    def _update(self, event) -> bool:
        """New values for the frames picture drawn (False: it has to be drawn anew)."""
        if self._mesh is None or self._cells is None:
            return False
        cells = self._grid(event)
        if cells is None or not (np.array_equal(cells[0], self._cells[0]) and np.array_equal(cells[1], self._cells[1])):
            return False
        limits = colour_limits(cells[3])
        if limits is None:
            return False
        with self._norm.callbacks.blocked():
            self._norm.vmin, self._norm.vmax = limits
        self._norm.callbacks.process("changed")
        self._mesh.set_array(self._values(cells[3], cells[2], cells[0].size - 1, limits[0]))
        self._title(event)
        from src.gimap.shared.figures import cjk_fallback

        cjk_fallback(self.figure, [self._axes[1].title])
        self.canvas.draw_idle()
        return True

    def _place(self) -> None:
        """The detector at the left (its own aspect), the frames and their colour bar at the right, margins in pixels."""
        if self._axes is None or self.figure is None:
            return
        width, height = (max(1.0, value) for value in self.figure.get_size_inches() * self.figure.dpi)
        m = MARGINS
        tall = max(10.0, height - m["top"] - m["bottom"])
        share = max(10.0, (width - m["left"]) * DETECTOR_SHARE)
        detector, frames, bar = self._axes
        detector.set_position([m["left"] / width, m["bottom"] / height, share / width, tall / height])
        start = m["left"] + share + m["gap"]
        room = max(10.0, width - start - m["right"])
        frames.set_position([start / width, m["bottom"] / height, room / width, tall / height])
        bar.set_position([(width - m["right"] + m["bar_gap"]) / width, m["bottom"] / height, m["bar"] / width,
                          tall / height])


class TracePlot:
    def __init__(self, layout):
        self.figure, self.canvas = _canvas(layout, (9, 2.4), 150)
        if self.canvas is not None:
            self.canvas.mpl_connect("resize_event", lambda _event: self._place())
        self._axes = None
        self._line = None
        self._onset = None
        self._last: tuple = ((), (), None)

    def show(self, times, rates, onset: Optional[float] = None) -> None:
        """The count rate of the frames read so far (in the order of time) and the start of the deposition."""
        self._last = (list(times), list(rates), onset)
        if self._line is not None and self._axes is not None:
            self._line.set_data(*with_gaps(*self._last[:2]))
            self._show_onset(onset)
            self._limits()
            self.canvas.draw_idle()
            return
        self._draw()

    def redraw(self) -> None:
        self._line = None
        self._draw()

    def _draw(self) -> None:
        if self.canvas is None:
            return
        figure = self.figure
        figure.clear()
        axes = figure.add_axes([0.08, 0.2, 0.9, 0.75])
        self._axes = axes
        times, rates, onset = self._last
        (self._line,) = axes.plot(*with_gaps(times, rates), "-", color=TRACE_COLOUR, linewidth=1.2)
        self._onset = axes.axvline(math.nan, color="#e08a1e", linewidth=1.2, linestyle="--", label=tr(ONSET_LABEL))
        self._show_onset(onset)
        axes.set_xlabel(tr(TRACE_X))
        axes.set_ylabel(tr(TRACE_Y))
        axes.grid(True, alpha=0.25)
        self._limits()
        self._place()
        _finish(figure, self.canvas, [axes.xaxis.label, axes.yaxis.label])

    def _show_onset(self, onset: Optional[float]) -> None:
        position = onset if onset is not None else math.nan
        self._onset.set_xdata([position, position])
        legend = self._axes.get_legend()
        if onset is not None and legend is None:
            self._axes.legend(loc="upper left", fontsize=8, frameon=False)
        elif onset is None and legend is not None:
            legend.remove()

    def _limits(self) -> None:
        times, rates, _onset = self._last
        finite = [rate for rate in rates if rate is not None and math.isfinite(rate)]
        if times:
            low, high = min(times), max(times)
            self._axes.set_xlim(low, high if high > low else low + 1.0)
        if finite:
            self._axes.set_ylim(0.0, max(finite) * 1.08 if max(finite) > 0 else 1.0)

    def _place(self) -> None:
        if self._axes is None:
            return
        width, height = (max(1.0, value) for value in self.figure.get_size_inches() * self.figure.dpi)
        m = TRACE_MARGINS
        self._axes.set_position([m["left"] / width, m["bottom"] / height,
                                 max(10.0, width - m["left"] - m["right"]) / width,
                                 max(10.0, height - m["top"] - m["bottom"]) / height])


__all__ = ["FramesPicture", "TEMPLATES", "TracePlot", "side_edges"]
