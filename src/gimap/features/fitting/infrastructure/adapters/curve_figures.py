"""Publication figure of the Fitting curve plot (format and size as Analyze's figures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gimap.shared.figures import (
    COLUMN_WIDTH_CM,
    publication_figure,
    save_figure,
    style_publication_axes,
)

from ...application.models import ExportCurveFigureRequest


SUFFIXES = (".png", ".svg", ".pdf", ".tif", ".tiff")
"""The formats written (``shared/figures.save_figure``)."""


def _draw(axes, series, log_y: bool) -> None:
    x = np.asarray(series.x, dtype=float)
    y = np.asarray(series.y, dtype=float)
    if log_y:
        y = np.where(y > 0, y, np.nan)
    if series.style == "scatter" and series.error is not None:
        error = np.asarray(series.error, dtype=float)
        if log_y:  # the lower bar ends above 0 on a log axis
            lower = np.where(np.isfinite(y), np.minimum(error, 0.999 * y), np.nan)
            error = np.vstack([lower, error])
        axes.errorbar(x, y, yerr=error, fmt=series.marker, ms=1.6, color=series.color, ecolor=series.color,
                      elinewidth=0.4, capsize=0, label=series.label, alpha=0.9, mew=0)
    elif series.style == "scatter":
        axes.scatter(x, y, s=2.5 if series.marker == "o" else 6, color=series.color, label=series.label,
                     linewidths=0 if series.marker == "o" else 0.6, marker=series.marker)
    else:
        axes.plot(x, y, lw=0.9, color=series.color, label=series.label, ls="--" if series.dashed else "-")


class MatplotlibCurveFigureWriter:
    """Measured points as small markers (with error bars when σ is given), model curves as thin lines, one column
    wide; with residuals, a second panel under the curve on the same x. PNG, TIFF, SVG or PDF by the suffix."""

    def write(self, request: ExportCurveFigureRequest) -> Path:
        if Path(request.path).suffix.lower() not in SUFFIXES:
            raise ValueError("Choose a .png, .svg or .pdf file")
        if request.residuals:
            figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * 0.95)
            grid = figure.add_gridspec(2, 1, height_ratios=(3, 1))
            axes = figure.add_subplot(grid[0])
            below = figure.add_subplot(grid[1], sharex=axes)
        else:
            figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * 0.75)
            axes, below = figure.add_subplot(), None
        if request.band is not None:
            for panel in (axes, below):
                if panel is not None:
                    panel.axvspan(*request.band, color="#f97316", alpha=0.06, lw=0)
        for series in request.series:
            _draw(axes, series, request.log_y)
        if request.x_scale == "symlog":
            finite = np.abs(np.concatenate([np.asarray(s.x, float) for s in request.series]))
            finite = finite[np.isfinite(finite) & (finite > 0)]
            axes.set_xscale("symlog", linthresh=float(finite.min() * 0.5) if finite.size else 1e-6)
        elif request.x_scale == "log":
            axes.set_xscale("log")
        if request.log_y:
            axes.set_yscale("log")
            shown = np.concatenate([np.asarray(s.y, dtype=float) for s in request.series] or [np.ones(1)])
            shown = shown[np.isfinite(shown) & (shown > 0)]
            if shown.size:  # the points and curves decide the axis; an error bar below 0 runs off it
                axes.set_ylim(0.5 * shown.min(), 2.0 * shown.max())
        axes.set_ylabel(request.y_label)
        if len(request.series) > 1:
            axes.legend(fontsize=7, frameon=False, markerscale=3)
        style_publication_axes(axes)
        if below is None:
            axes.set_xlabel(request.x_label)
        else:
            for series in request.residuals:
                _draw(below, series, False)
            below.set_xlabel(request.x_label)
            below.set_ylabel(request.residual_label)
            axes.tick_params(labelbottom=False)
            style_publication_axes(below)
        return save_figure(figure, request.path)


__all__ = ["MatplotlibCurveFigureWriter"]
