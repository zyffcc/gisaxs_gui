"""The screen theme and text fonts of Fitting's Matplotlib figures (1D Predict)."""

from __future__ import annotations


def apply_plot_theme(figure, *, background: str, foreground: str, grid: str) -> None:
    """Screen colours of a curve figure (the light/dark theme); exported figures keep their own."""
    figure.patch.set_facecolor(background)
    for axes in figure.axes:
        axes.set_facecolor(background)
        for spine in axes.spines.values():
            spine.set_edgecolor(foreground)
        axes.tick_params(which="both", colors=foreground)
        for text in (axes.xaxis.label, axes.yaxis.label, axes.title):
            text.set_color(foreground)
        axes.grid(True, color=grid, alpha=1.0)
        legend = axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(grid)
            for text in legend.get_texts():
                text.set_color(foreground)


def theme_figure(figure) -> None:
    """``apply_plot_theme`` with the plot tokens of the current theme; free text (a placeholder) too."""
    from src.gimap.app.presentation.theme import theme_manager

    manager = theme_manager()
    foreground = manager.color("plot_fg").name()
    apply_plot_theme(figure, background=manager.color("plot_bg").name(), foreground=foreground,
                     grid=manager.color("plot_grid").name())
    for text in figure.texts:
        text.set_color(foreground)


def follow_plot_theme(owner, figure, canvas):
    """Theme ``figure`` now and again, redrawn, after every theme switch until ``owner`` (a QObject) is destroyed."""
    from src.gimap.app.presentation.theme import theme_manager

    changed = theme_manager().changed

    def retheme(*_args) -> None:
        try:
            theme_figure(figure)
            canvas.draw_idle()
        except RuntimeError:  # the canvas is gone
            pass

    def disconnect(*_args) -> None:
        try:
            changed.disconnect(retheme)
        except TypeError:
            pass

    theme_figure(figure)
    changed.connect(retheme)
    owner.destroyed.connect(disconnect)
    return retheme


CJK_FONTS = ("Microsoft YaHei", "Noto Sans SC", "Noto Sans CJK SC", "SimHei", "PingFang SC", "Hiragino Sans GB",
             "PingFang HK", "Heiti TC", "Songti SC", "Arial Unicode MS", "WenQuanYi Zen Hei")
"""Matplotlib knows a font collection by its first face: on macOS “PingFang HK”, never “PingFang SC”, so the
fonts every Mac has are named too (as ``shared/figures.CJK_FONTS``)."""


def text_font_family() -> list:
    """The fonts of a figure's own sentences (a placeholder): Matplotlib's font, then a Chinese one when
    there is one — Matplotlib's has no Chinese glyphs and draws Chinese text as empty boxes. An English
    sentence looks the same with or without the second font."""
    try:
        from matplotlib import font_manager

        names = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001 - no font list: Matplotlib's own font
        return ["sans-serif"]
    fallback = next((name for name in CJK_FONTS if name in names), None)
    return ["DejaVu Sans", fallback] if fallback and "DejaVu Sans" in names else [fallback or "sans-serif"]


__all__ = ["apply_plot_theme", "follow_plot_theme", "text_font_family", "theme_figure"]
