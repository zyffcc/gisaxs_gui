"""Saving a pyqtgraph plot as a picture: SVG, or PNG 1600 px wide — always on the light plot palette.

``export_plot(plot, path)`` takes a ``CurvePlot`` (or a pyqtgraph ``PlotItem`` / ``PlotWidget``). A figure goes
on white paper and slides, so a plot shown on the dark theme is exported with the light theme's plot colours:
its background, axes, ticks, axis labels, title and legend are restyled for the moment of the export and put
back afterwards (also when the export fails). The curves keep their colours. ``OSError`` when nothing was
written — a folder that is not there, no permission, a plot of size 0 — so a caller never says “Saved” then.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

EXPORT_WIDTH = 1600
"""Width in pixels of an exported PNG (the height keeps the plot's proportions)."""
LEGEND_ALPHA = 150
"""Opacity of the legend's background in an exported figure (as on screen)."""


def _parts(plot) -> tuple:
    """``(plot item, graphics view, legend or None)`` of a ``CurvePlot``, a ``PlotWidget`` or a ``PlotItem``."""
    import pyqtgraph as pg

    if isinstance(plot, pg.PlotWidget):
        item = plot.getPlotItem()
        return item, plot, item.legend
    if isinstance(plot, pg.PlotItem):
        scene = plot.scene()
        views = scene.views() if scene is not None else []
        return plot, (views[0] if views else None), plot.legend
    item, view = getattr(plot, "plot", None), getattr(plot, "plot_widget", None)
    if not isinstance(item, pg.PlotItem):
        raise TypeError(f"not a plot: {type(plot).__name__}")
    return item, view, getattr(plot, "legend", None) or item.legend


def _light_palette(item, view, legend) -> Callable[[], None]:
    """Restyle the plot with the light theme's plot colours; returns what puts the shown colours back."""
    import pyqtgraph as pg
    from PyQt5.QtGui import QColor

    from ..theme import LIGHT

    background, foreground, grid = (LIGHT[name] for name in ("plot_bg", "plot_fg", "plot_grid"))
    undo: list[Callable[[], None]] = []
    if view is not None:
        brush = view.backgroundBrush()
        view.setBackground(background)
        undo.append(lambda: view.setBackgroundBrush(brush))
    for side in ("left", "bottom", "right", "top"):
        axis = item.getAxis(side)
        pen, text_pen = axis.pen(), axis.textPen()
        axis.setPen(pg.mkPen(foreground))
        axis.setTextPen(pg.mkPen(foreground))
        # setPen before setTextPen: the label takes the text pen's colour, as when the theme styled it
        undo.append(lambda axis=axis, pen=pen, text_pen=text_pen: (axis.setPen(pen), axis.setTextPen(text_pen)))
    title = item.titleLabel
    if title is not None and title.text:
        color = title.opts.get("color")
        title.setText(title.text, color=foreground)
        undo.append(lambda: title.setText(title.text, color=color))
    if legend is not None:
        brush, frame, label_color = legend.brush(), legend.pen(), legend.labelTextColor()
        colors = [(label, label.opts.get("color")) for _sample, label in legend.items]
        if legend.items:  # an empty legend stays invisible
            fill = QColor(background)
            fill.setAlpha(LEGEND_ALPHA)
            legend.setBrush(fill)
            legend.setPen(pg.mkPen(grid))
        legend.setLabelTextColor(foreground)
        for label, _color in colors:
            label.setText(label.text, color=foreground)  # setLabelTextColor alone does not redraw a label

        def restore_legend() -> None:
            legend.setBrush(brush)
            legend.setPen(frame)
            if label_color is None:  # a legend that never had a colour of its own (``addLegend()``): none again
                legend.opts["labelTextColor"] = None
            else:
                legend.setLabelTextColor(label_color)
            for label, color in colors:
                label.setText(label.text, color=color)

        undo.append(restore_legend)

    def restore() -> None:
        for step in reversed(undo):  # every step, even when one fails: the screen gets back all it can
            try:
                step()
            except (RuntimeError, TypeError, ValueError):  # the plot is being torn down; a value pyqtgraph refuses
                pass

    return restore


def _write(item, path: Path, width: int) -> None:
    from pyqtgraph import exporters

    svg = path.suffix.lower() == ".svg"
    try:
        exporter = exporters.SVGExporter(item) if svg else exporters.ImageExporter(item)
        if not svg:
            exporter.parameters()["width"] = int(width)
        written = exporter.export(str(path))
    except OSError:
        raise
    except Exception as exc:  # pyqtgraph: a bare Exception for a plot of size 0; Qt: a format it cannot write
        raise OSError(str(exc) or type(exc).__name__) from exc
    if written is False or not path.is_file():  # QImage.save returns False when nothing was written
        from ..i18n import tr

        raise OSError(tr("could not write {path}").format(path=path))


def export_plot(plot, path, *, width: int = EXPORT_WIDTH) -> Path:
    """Write ``plot`` to ``path``: SVG for ``.svg``, else an image (PNG …) ``width`` pixels wide, on the light
    plot palette in either theme. Returns the path; ``OSError`` when nothing was written."""
    path = Path(path)
    item, view, legend = _parts(plot)
    restore = _light_palette(item, view, legend)
    try:
        _write(item, path, width)
    finally:
        restore()
    return path


__all__ = ["EXPORT_WIDTH", "export_plot"]
