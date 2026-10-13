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


class MatplotlibCurveFigureWriter:
    """Measured points as small markers, model curves as thin lines, one column wide."""

    def write(self, request: ExportCurveFigureRequest) -> Path:
        figure = publication_figure(COLUMN_WIDTH_CM, COLUMN_WIDTH_CM * 0.75)
        axes = figure.add_subplot()
        for series in request.series:
            x = np.asarray(series.x, dtype=float)
            y = np.asarray(series.y, dtype=float)
            if request.log_y:
                y = np.where(y > 0, y, np.nan)
            if series.style == "scatter":
                axes.scatter(x, y, s=2.5, color=series.color, label=series.label, linewidths=0)
            else:
                axes.plot(x, y, lw=0.9, color=series.color, label=series.label)
        if request.x_scale == "symlog":
            finite = np.abs(np.concatenate([np.asarray(s.x, float) for s in request.series]))
            finite = finite[np.isfinite(finite) & (finite > 0)]
            axes.set_xscale("symlog", linthresh=float(finite.min() * 0.5) if finite.size else 1e-6)
        elif request.x_scale == "log":
            axes.set_xscale("log")
        if request.log_y:
            axes.set_yscale("log")
        axes.set_xlabel(request.x_label)
        axes.set_ylabel(request.y_label)
        if len(request.series) > 1:
            axes.legend(fontsize=7, frameon=False, markerscale=3)
        style_publication_axes(axes)
        return save_figure(figure, request.path)


__all__ = ["MatplotlibCurveFigureWriter"]
