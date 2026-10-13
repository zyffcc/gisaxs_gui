"""Theme colours for matplotlib figures shown on screen (tool windows, previews).

``theme_figure(figure, canvas)`` colours the figure and axes faces, the text, ticks, spines and
titles from the ``plot_bg``, ``plot_fg`` and ``plot_grid`` tokens, redraws the canvas and keeps
following theme switches until the canvas is destroyed.  ``Axes.clear()`` restores matplotlib's
label and title colours, so code that clears and redraws calls ``restyle_figure(figure)`` before
``draw_idle()``.  Only the screen copy changes: data, limits and colour maps stay as they are.
Code that writes a themed figure to a file does it inside ``with exported_colors(figure):``, so
the file gets matplotlib's own colours and the screen copy gets the theme's back afterwards.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from functools import partial

from matplotlib import rcParams
from matplotlib.colors import same_color

from .engine import theme_manager

_SLOT_ATTRIBUTE = "_gimap_theme_slot"
_THEMED_ATTRIBUTE = "_gimap_themed_text"


def figure_colors() -> dict[str, str]:
    """The current theme's plot colours as ``background``, ``foreground`` and ``grid``."""
    manager = theme_manager()
    return {
        "background": manager.color("plot_bg").name(),
        "foreground": manager.color("plot_fg").name(),
        "grid": manager.color("plot_grid").name(),
    }


def apply_figure_colors(figure, *, background: str, foreground: str, grid: str) -> None:
    """Colour faces, spines, ticks, labels, titles and legends; grids keep their on/off state."""
    figure.patch.set_facecolor(background)
    for text in figure.texts:
        _recolor_free_text(text, foreground)
    for axes in figure.axes:
        axes.set_facecolor(background)
        for spine in axes.spines.values():
            spine.set_edgecolor(foreground)
        axes.tick_params(which="both", colors=foreground, grid_color=grid)
        for text in (
            axes.xaxis.label,
            axes.yaxis.label,
            axes.xaxis.get_offset_text(),
            axes.yaxis.get_offset_text(),
            axes.title,
            getattr(axes, "_left_title", None),
            getattr(axes, "_right_title", None),
        ):
            if text is not None:
                text.set_color(foreground)
        for text in axes.texts:
            _recolor_free_text(text, foreground)
        legend = axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(grid)
            for text in legend.get_texts():
                text.set_color(foreground)


def restyle_figure(figure) -> None:
    """Give ``figure`` the colours of the current theme (no redraw)."""
    apply_figure_colors(figure, **figure_colors())


@contextmanager
def exported_colors(figure):
    """Matplotlib's own colours while ``figure`` is saved, then the theme's again (no redraw)."""
    apply_figure_colors(
        figure,
        background=rcParams["figure.facecolor"],
        foreground=rcParams["text.color"],
        grid=rcParams["grid.color"],
    )
    try:
        yield figure
    finally:
        restyle_figure(figure)


def theme_figure(figure, canvas) -> None:
    """Colour ``figure`` for the current theme, redraw ``canvas`` and follow theme switches.

    Calling it again only re-colours: the theme connection is made once per canvas and is
    removed when the canvas is destroyed.
    """
    restyle_figure(figure)
    canvas.draw_idle()
    if getattr(canvas, _SLOT_ATTRIBUTE, None) is not None:
        return
    manager = theme_manager()
    figure_ref = weakref.ref(figure)
    canvas_ref = weakref.ref(canvas)

    def follow_theme(*_args) -> None:
        current_figure, current_canvas = figure_ref(), canvas_ref()
        if current_figure is None or current_canvas is None:
            _disconnect(manager, follow_theme)
            return
        try:
            restyle_figure(current_figure)
            current_canvas.draw_idle()
        except RuntimeError:  # the Qt canvas is already gone
            _disconnect(manager, follow_theme)

    manager.changed.connect(follow_theme)
    setattr(canvas, _SLOT_ATTRIBUTE, follow_theme)
    canvas.destroyed.connect(partial(_disconnect, manager, follow_theme))


def _disconnect(manager, slot, *_args) -> None:
    try:
        manager.changed.disconnect(slot)
    except (TypeError, RuntimeError):  # already disconnected
        pass


def _recolor_free_text(text, foreground: str) -> None:
    """Re-colour text left at matplotlib's default colour (or themed before); keep chosen colours."""
    if getattr(text, _THEMED_ATTRIBUTE, False) or same_color(text.get_color(), rcParams["text.color"]):
        text.set_color(foreground)
        setattr(text, _THEMED_ATTRIBUTE, True)


__all__ = [
    "apply_figure_colors",
    "exported_colors",
    "figure_colors",
    "restyle_figure",
    "theme_figure",
]
