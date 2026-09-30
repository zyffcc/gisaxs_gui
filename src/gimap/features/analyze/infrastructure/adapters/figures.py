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


class MatplotlibFigureWriter:
    def write_panels(self, panels, path: Path, *, x_label: str = "frame", title: str = "") -> Path:
        """Stacked panels sharing x: ``panels`` = ``[(y_label, [(name, x, y), …]), …]`` (a fit trend)."""
        panels = [panel for panel in panels if panel[1]]
        if not panels:
            raise ValueError("Nothing to plot.")
        figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * (0.3 + 0.32 * len(panels)))
        axes_list = figure.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
        for axes, (y_label, curves) in zip(axes_list, panels):
            for index, (name, x, y) in enumerate(curves):
                axes.plot(np.asarray(x, float), np.asarray(y, float), marker="o", markersize=2.5, lw=0.8,
                          color=CURVE_COLORS[index % len(CURVE_COLORS)], label=name)
            axes.set_ylabel(y_label)
            style_publication_axes(axes)
            if len(curves) > 1:
                axes.legend(fontsize=6, frameon=False)
        axes_list[-1].set_xlabel(x_label)
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
    ) -> Path:
        """``image`` as shown (already log-scaled when ``log_scale``), with a colour bar."""
        figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * 0.85)
        axes = figure.add_subplot()
        shown = np.ma.masked_invalid(np.asarray(image, dtype=float))
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
        colorbar = figure.colorbar(artist, ax=axes, fraction=0.05, pad=0.02)
        colorbar.set_label("log₁₀ I" if log_scale else "I", size=8)
        colorbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        if title:
            axes.set_title(title, fontsize=8)
        style_publication_axes(axes)
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
