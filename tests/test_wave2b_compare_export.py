"""Wave 2b, Compare: plots saved on the light palette in either theme (``export_plot``), legend-less marks on a
CurvePlot (``set_curves(legend=…)`` and ``Marker``), and names readable on the light theme."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtGui import QColor, QImage

from src.gimap.app.presentation.theme import DARK, LIGHT, apply_theme, theme_manager
from tests.test_assistant_gui import _app


class _Theme:
    def __enter__(self):
        self.mode, self.font_pt = theme_manager().mode, theme_manager().font_pt
        return self

    def __exit__(self, *_exc):
        if (theme_manager().mode, theme_manager().font_pt) != (self.mode, self.font_pt):
            apply_theme(self.mode, self.font_pt)


def _plot():
    from src.gimap.app.presentation.components.curve_plot import CurvePlot

    _app()
    plot = CurvePlot("test", log_y=False, sides=False)
    plot.resize(600, 400)
    plot.show()
    x = np.linspace(0.0, 1.0, 40)
    plot.set_curves([("a", x, x ** 2), ("b", x, x)], legend=[True, False])
    return plot


def _shown_colors(plot) -> tuple[str, str, list]:
    axis = plot.plot.getAxis("left")
    labels = [QColor(label.opts["color"]).name() for _sample, label in plot.legend.items]
    return plot.plot_widget.backgroundBrush().color().name(), axis.textPen().color().name(), labels


def test_a_dark_plot_is_saved_on_the_light_palette_and_shown_dark_again(tmp_path: Path) -> None:
    from src.gimap.app.presentation.components.plot_export import EXPORT_WIDTH, export_plot

    with _Theme():
        apply_theme("dark")
        plot = _plot()
        try:
            before = _shown_colors(plot)
            assert before[:2] == (DARK["plot_bg"], DARK["plot_fg"])
            written = export_plot(plot, tmp_path / "plot.png")
            image = QImage(str(written))
            assert image.width() == EXPORT_WIDTH and image.pixelColor(2, 2).name() == LIGHT["plot_bg"]
            assert _shown_colors(plot) == before  # the screen keeps the dark theme
            svg = export_plot(plot, tmp_path / "plot.svg").read_text(encoding="utf-8")
            assert LIGHT["plot_fg"] in svg and DARK["plot_bg"] not in svg and DARK["plot_fg"] not in svg
            assert _shown_colors(plot) == before
            # Nothing written: an OSError, and the theme is back all the same.
            with pytest.raises(OSError):
                export_plot(plot, tmp_path / "missing" / "plot.png")
            with pytest.raises(OSError):
                export_plot(plot, tmp_path / "missing" / "plot.svg")
            assert _shown_colors(plot) == before
            # A bare PlotItem (what Fitting and the Assistant hold) works too.
            export_plot(plot.plot, tmp_path / "item.png")
            assert QImage(str(tmp_path / "item.png")).pixelColor(2, 2).name() == LIGHT["plot_bg"]
        finally:
            plot.dispose()
            plot.deleteLater()


def test_a_bare_plot_item_with_a_default_legend_is_saved_and_put_back(tmp_path: Path) -> None:
    """Fitting and the Assistant hold a plain ``PlotItem`` whose legend never had a label colour of its own."""
    import pyqtgraph as pg

    from src.gimap.app.presentation.components.plot_export import export_plot

    _app()
    widget = pg.PlotWidget()
    try:
        widget.resize(400, 300)
        widget.show()
        item = widget.getPlotItem()
        legend = item.addLegend()
        item.setTitle("A title")
        item.plot([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], name="a")
        brush = widget.backgroundBrush().color().name()
        assert legend.labelTextColor() is None
        written = export_plot(item, tmp_path / "bare.png")
        assert QImage(str(written)).pixelColor(2, 2).name() == LIGHT["plot_bg"]
        assert legend.labelTextColor() is None and legend.items[0][1].opts["color"] is None  # as it was
        assert item.titleLabel.opts["color"] is None and widget.backgroundBrush().color().name() == brush
    finally:
        widget.close()
        widget.deleteLater()


def test_marks_can_be_left_out_of_the_legend_and_drawn_hollow() -> None:
    from src.gimap.app.presentation.components.curve_plot import Marker

    with _Theme():
        apply_theme("light")
        plot = _plot()
        try:
            assert [label.text for _sample, label in plot.legend.items] == ["a"]  # “b”: no entry, still drawn
            assert plot.curve_count() == 2 and [name for name, _x, _y in plot.figure_state()["curves"]] == ["a", "b"]
            apply_theme("dark")  # drawn again on a new theme: still one entry
            assert len(plot.legend.items) == 1
            plot.log_check.setChecked(True)  # and after a log toggle
            assert len(plot.legend.items) == 1
            plot.log_check.setChecked(False)
            x = np.array([0.5])
            plot.set_curves([("dots", x, x), ("start", x, x), ("end", x, x), ("cross", x, x)],
                            markers=[True, Marker("o", 9.0, hollow=True), Marker("s", 9.0), "x"],
                            legend=[True, False, False, False])
            items = plot._items
            assert len(plot.legend.items) == 1
            assert items[0].opts["symbolSize"] == 4 and items[0].opts["symbolPen"] is None  # measured dots as before
            assert items[1].opts["symbolSize"] == 9.0 and items[1].opts["symbolBrush"].color().alpha() == 0
            assert items[2].opts["symbol"] == "s" and items[2].opts["symbolBrush"] == plot._colors[2]
            assert items[3].opts["symbol"] == "x" and items[3].opts["symbolSize"] == 7.0
        finally:
            plot.dispose()
            plot.deleteLater()


def test_stage_and_series_names_reach_4_5_to_1_on_the_light_theme() -> None:
    from src.gimap.app.presentation.components.curve_plot import CURVE_COLORS
    from src.gimap.app.presentation.components.row_groups import (
        STAGE_COLORS, contrast, stage_color, stage_text_color, text_color,
    )

    _app()
    with _Theme():
        apply_theme("light")
        for color in (*STAGE_COLORS, *CURVE_COLORS):
            shown = text_color(color)
            for surface in (LIGHT["surface"], LIGHT["surface_alt"]):
                assert contrast(shown, surface) >= 4.5, (color, shown, surface)
            if contrast(color, LIGHT["surface_alt"]) >= 4.5:
                assert shown == color  # readable already: kept as it is
            else:  # darker in the same hue
                assert QColor(shown).lightness() < QColor(color).lightness()
                assert abs(QColor(shown).hue() - QColor(color).hue()) <= 4
        assert stage_text_color(1) == "#2563eb"
        assert contrast(stage_text_color(2), "#ffffff") >= 4.5 > contrast(STAGE_COLORS[1], "#ffffff")
        apply_theme("dark")
        assert stage_text_color(2) == QColor(STAGE_COLORS[1]).lighter(150).name()  # dark: as before
        assert [stage_color(stage) for stage in range(1, 9)] == list(STAGE_COLORS)  # strips and curves unchanged
