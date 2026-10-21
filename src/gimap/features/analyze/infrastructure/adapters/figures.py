"""Publication figures of Analyze results, rendered with Matplotlib (no pyplot state).

Format and size follow :mod:`src.gimap.shared.figures`: PNG/TIFF at 600 dpi or
SVG/PDF vectors by suffix, one journal column (8.5 cm) wide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.figures import (
    COLUMN_WIDTH_CM,
    break_at_gaps,
    publication_figure,
    save_figure,
    style_publication_axes,
)

CURVE_COLORS = ("#1f4e9c", "#d9480f", "#2b8a3e", "#7048e8", "#c92a2a", "#0b7285")
SIDE_COLORS = {1: "#1f4e9c", -1: "#d9480f", 0: "#2b8a3e"}
TALL_FRAME = 0.8
"""rows / columns from which the picture of a frame is laid out around the box of the frame.

The room for the frame is about as tall as wide (1.02 times with a title, 1.08 without). A wider frame fills the
width of the room: Matplotlib's constrained layout is right for it, and its colour bar is as tall as the room.
A frame about as tall as the room or taller does not fill the width: that layout then cuts off “y (pixel)”
(a square frame) or leaves the frame at the right of an empty picture, so the compressed layout is asked for,
which ties the colour bar to the box of the frame — 8 mm with two numbers if a 195 × 1475 strip were given it."""
BAR_FRACTION, BAR_PAD, BAR_ASPECT = 0.05, 0.02, 20.0
"""The colour bar: its width and its distance from the frame as fractions of the frame's width, and its length
over its width."""
BAR_WIDEST = 8.0
"""How many times :data:`BAR_FRACTION` and :data:`BAR_PAD` a tall frame may ask for (0.4 and 0.16 of its width).
Matplotlib sizes the bar from the room the frame has BEFORE the layout: a fraction near 1 leaves no room for the
axes (“axes sizes collapsed to zero”, and a crash of the renderer for a 500 × 1 frame)."""


def _whole_rows(bottom: float, top: float, rows: int) -> bool:
    """Rows of a map between ``bottom`` and ``top`` are centred on whole numbers, a whole number apart."""
    step = abs(float(top) - float(bottom)) / max(int(rows), 1)
    first = min(float(top), float(bottom)) + 0.5 * step
    return step >= 1.0 - 1e-9 and abs(step - round(step)) < 1e-6 and abs(first - round(first)) < 1e-6


class MatplotlibFigureWriter:
    def write_panels(self, panels, path: Path, *, x_label: str = "frame", title: str = "") -> Path:
        """Stacked panels sharing x: ``panels`` = ``[(y_label, [(name, x, y[, error]), …]), …]`` (a fit trend).

        A curve without any value is left out (no empty entry in the legend), ``error`` is drawn as bars, the
        values are written in full on the axis (no offset in a corner that hides which q a peak is at), and x —
        frame numbers — has whole-number ticks. Panels that all hold the one same series (a single peak) have no
        legend: its name is the title."""
        drawn = []
        for y_label, curves in panels:
            curves = [curve for curve in curves if np.isfinite(np.asarray(curve[2], dtype=float)).any()]
            if curves:
                drawn.append((y_label, curves))
        if not drawn:
            raise ValueError("Nothing to plot.")
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * (0.3 + 0.32 * len(drawn)))
        axes_list = figure.subplots(len(drawn), 1, sharex=True, squeeze=False)[:, 0]
        for axes, (y_label, curves) in zip(axes_list, drawn):
            for index, (name, x, y, *rest) in enumerate(curves):
                x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                error = np.asarray(rest[0], dtype=float) if rest and rest[0] is not None else None
                color = CURVE_COLORS[index % len(CURVE_COLORS)]
                if error is not None and np.isfinite(error).any():
                    axes.errorbar(x, y, yerr=np.where(np.isfinite(error), error, 0.0), marker="o", markersize=2.5, lw=0.8,
                                  elinewidth=0.6, capsize=1.2, color=color, label=name)
                else:
                    axes.plot(x, y, marker="o", markersize=2.5, lw=0.8, color=color, label=name)
            axes.set_ylabel(y_label)
            style_publication_axes(axes)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            low, high = axes.get_ylim()
            if 1e-3 <= max(abs(low), abs(high)) < 1e5:  # beyond these a power of ten on the axis is the clearer way
                axes.yaxis.set_major_formatter(formatter)
            if len(curves) > 1:
                axes.legend(fontsize=6, frameon=False)
        axes_list[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes_list[-1].set_xlabel(x_label)
        names = {str(curves[0][0]) for _label, curves in drawn if len(curves) == 1}
        if not title and len(names) == 1 and all(len(curves) == 1 for _label, curves in drawn):
            (name,) = names  # one series in every panel (one peak; the others gave no value): the figure says which
            title = name if name not in {label for label, _curves in drawn} else ""
        if title:
            axes_list[0].set_title(title, fontsize=8)
        return save_figure(figure, path)

    def write_image(
        self,
        image: np.ndarray,
        path: Path,
        *,
        extent: Optional[tuple[float, float, float, float]] = None,
        origin_upper: bool = True,
        x_label: str = "x (pixel)",
        y_label: str = "y (pixel)",
        colormap: str = "viridis",
        log_scale: bool = True,
        levels: Optional[tuple[float, float]] = None,
        title: str = "",
        y_ticks: Optional[Sequence[tuple[float, str]]] = None,
    ) -> Path:
        """``image`` as shown (already log-scaled when ``log_scale``), with a colour bar.

        Without ``extent`` it is a detector frame, drawn with square pixels; with it a map on its own axes.
        ``y_ticks``: ``(position, label)`` of every tick of the y axis (the numbers of the rows of a Series map
        whose rows are not evenly numbered). Without them a map whose rows are centred on whole numbers (frame
        numbers, 1 … N) has whole-number ticks: three rows are 1, 2, 3, not 1.0, 1.5, 2.0 …"""
        shown = np.ma.masked_invalid(np.asarray(image, dtype=float))
        rows, columns = (max(int(size), 1) for size in shown.shape[:2])
        around_the_frame = extent is None and rows >= TALL_FRAME * columns  # see TALL_FRAME
        figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * 0.85, fixed_aspect=around_the_frame)
        axes = figure.add_subplot()
        vmin, vmax = levels if levels is not None else (None, None)
        artist = axes.imshow(
            shown,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper" if origin_upper else "lower",
            aspect="equal" if extent is None else "auto",
            interpolation="nearest",
        )
        # The bar is BAR_FRACTION of the frame's width wide and BAR_ASPECT times as long, at most as long as the
        # frame. For a frame taller than wide that is too thin and too short (a fifth of the frame for 1500 × 300):
        # the fractions grow with rows / columns, so the bar is as wide as for a square frame and as tall as this
        # one. Beyond BAR_WIDEST the bar gets thinner instead, and stays as tall as the frame.
        taller = rows / columns if around_the_frame and rows > columns else 1.0
        wider = min(taller, BAR_WIDEST)
        colorbar = figure.colorbar(artist, ax=axes, fraction=BAR_FRACTION * wider, pad=BAR_PAD * wider,
                                   aspect=BAR_ASPECT * taller / wider)
        colorbar.set_label("log₁₀ I" if log_scale else "I", size=8)
        colorbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        if title:
            axes.set_title(title, fontsize=8)
        style_publication_axes(axes)
        if y_ticks is not None:
            axes.set_yticks([float(position) for position, _label in y_ticks])
            axes.set_yticklabels([str(label) for _position, label in y_ticks])
        elif extent is not None and _whole_rows(extent[2], extent[3], rows):
            from matplotlib.ticker import MaxNLocator

            axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        return save_figure(figure, path)

    def write_frame(
        self,
        data: np.ndarray,
        valid: Optional[np.ndarray],
        path: Path,
        *,
        log_scale: bool = True,
        colormap: str = "viridis",
        title: str = "",
        extent: Optional[tuple[float, float, float, float]] = None,
        origin_upper: bool = True,
        x_label: str = "x (pixel)",
        y_label: str = "y (pixel)",
        levels: Optional[tuple[float, float]] = None,
    ) -> Path:
        """A frame with ``levels`` (intensity) or automatic display levels (1–99.7 % of the valid pixels)."""
        shown = np.array(data, dtype=np.float64, copy=True)
        if valid is not None:
            shown[~np.asarray(valid, dtype=bool)] = np.nan
        if log_scale:
            with np.errstate(divide="ignore", invalid="ignore"):
                shown = np.where(shown > 0, np.log10(shown), np.nan)
        if levels is not None:
            low, high = (float(value) for value in levels)
            if log_scale:
                high = float(np.log10(high)) if high > 0 else 0.0
                low = float(np.log10(low)) if low > 0 else high - 6.0
            levels = (low, high)
        else:
            finite = shown[np.isfinite(shown)]
            levels = tuple(np.percentile(finite, (1.0, 99.7))) if finite.size else None
        return self.write_image(
            shown, path, extent=extent, origin_upper=origin_upper, x_label=x_label,
            y_label=y_label, colormap=colormap, log_scale=log_scale, levels=levels, title=title,
        )

    def write_curves(
        self,
        curves: Sequence[tuple[str, np.ndarray, np.ndarray]],
        path: Path,
        *,
        x_label: str,
        y_label: str,
        log_y: bool = True,
        log_x: bool = False,
        title: str = "",
        sides: Optional[Sequence[Optional[np.ndarray]]] = None,
        colors: Optional[Sequence[str]] = None,
    ) -> Path:
        """One line per curve (in ``colors`` when given, as on screen); a curve with ``sides`` draws each half in its own colour."""
        figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * 0.75)
        axes = figure.add_subplot()
        for index, (name, x, y) in enumerate(curves):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if log_y:
                y = np.where(y > 0, y, np.nan)
            if log_x:
                y = np.where(x > 0, y, np.nan)
            side = sides[index] if sides is not None and index < len(sides) else None
            if side is None:
                color = colors[index] if colors is not None and index < len(colors) else CURVE_COLORS[index % len(CURVE_COLORS)]
                axes.plot(*break_at_gaps(x, y), lw=0.9, color=color, label=name)
                continue
            for sign, label in ((1, "qy > 0"), (-1, "qy < 0"), (0, "mean")):
                keep = np.asarray(side) == sign
                if keep.any():
                    axes.plot(*break_at_gaps(x[keep], y[keep]), lw=0.9, color=SIDE_COLORS[sign], label=f"{name}, {label}")
        if log_y:
            axes.set_yscale("log")
        if log_x:
            axes.set_xscale("log")
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        if title:
            axes.set_title(title, fontsize=8)
        if len(axes.get_lines()) > 1:
            axes.legend(fontsize=7, frameon=False)
        style_publication_axes(axes)
        return save_figure(figure, path)


__all__ = ["MatplotlibFigureWriter"]
