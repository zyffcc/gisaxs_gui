"""Supporting curves may be visible without setting the plot's limits or losing a user's zoom."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.components.curve_plot import CurvePlot, LineStyle
from src.gimap.app.presentation.theme import apply_theme


@pytest.fixture
def plot():
    widget = CurvePlot(log_y=False, log_x=False)
    widget.resize(640, 420)
    widget.show()
    QApplication.processEvents()
    yield widget
    widget.dispose()
    widget.close()
    widget.deleteLater()
    apply_theme("light", 9)


def _settle(plot):
    QApplication.processEvents()
    plot.plot.getViewBox().updateAutoRange()
    QApplication.processEvents()
    return np.asarray(plot.plot.viewRange())


@pytest.mark.parametrize("log_y", [False, True])
def test_supporting_curve_does_not_expand_limits_until_included(plot, log_y):
    plot.log_check.setChecked(log_y)
    x = np.linspace(1, 10, 101)
    primary = ("model", x, np.geomspace(10, 100, x.size))
    supporting = ("component", x * 1000, np.geomspace(1e-90, 1e9, x.size))
    plot.set_curves([primary])
    expected = _settle(plot)

    plot.set_curves([primary, supporting], auto_range=[True, False])
    np.testing.assert_allclose(_settle(plot), expected, rtol=1e-6, atol=1e-8)
    assert plot.curve_count() == 2
    np.testing.assert_array_equal(plot.figure_state()["curves"][1][2], supporting[2])
    plot.plot.setRange(xRange=(2, 3), yRange=(1, 2), padding=0)
    plot.reset_view()
    np.testing.assert_allclose(_settle(plot), expected, rtol=1e-6, atol=1e-8)

    plot.set_curves([primary, supporting], auto_range=[True, True])
    expanded = _settle(plot)
    assert expanded[0, 1] > 10000
    assert expanded[1, 1] > (9 if log_y else 1e9)
    if log_y:
        assert expanded[1, 0] < -80


@pytest.mark.parametrize("log_y", [False, True])
def test_showing_and_hiding_supporting_curves_preserves_manual_zoom(plot, log_y):
    plot.log_check.setChecked(log_y)
    x = np.linspace(1, 10, 101)
    primary = ("model", x, x * 10)
    supporting = ("component", x, x * 1e-30)
    plot.set_curves([primary])
    _settle(plot)
    zoom = [[3.0, 6.0], [1.5, 1.8] if log_y else [35.0, 60.0]]
    plot.plot.setRange(xRange=zoom[0], yRange=zoom[1], padding=0)
    for curves in ([primary, supporting], [primary]):
        plot.set_curves(curves, auto_range=[True, False], preserve_view=True)
        np.testing.assert_allclose(_settle(plot), zoom, atol=1e-12)
        assert plot.plot.getViewBox().autoRangeEnabled() == [False, False]
    plot.reset_view()
    assert _settle(plot)[0, 1] > 10


def test_preserving_a_view_keeps_auto_range_enabled(plot):
    x = np.linspace(1, 10, 101)
    main = ("model", x, 10 * x)
    plot.set_curves([main])
    expected = _settle(plot)
    before = list(plot.plot.getViewBox().autoRangeEnabled())
    plot.set_curves([main, ("tiny component", x, x * 1e-50)], auto_range=[True, False], preserve_view=True)
    np.testing.assert_allclose(_settle(plot), expected)
    assert plot.plot.getViewBox().autoRangeEnabled() == before


def test_theme_redraw_keeps_styles_bounds_policy_and_zoom(plot):
    x = np.linspace(1, 10, 101)
    styles = [LineStyle(width=2.4, z=3), LineStyle(width=1.25, style=Qt.DashLine, z=1)]
    plot.set_curves([("total", x, x * 10), ("component", x, x * 1e-40)],
                    line_styles=styles, auto_range=[True, False])
    _settle(plot)
    plot.plot.setRange(xRange=(3, 6), yRange=(35, 60), padding=0)
    for mode in ("dark", "light"):
        apply_theme(mode, 9)
        np.testing.assert_allclose(_settle(plot), [[3, 6], [35, 60]], atol=1e-12)
        for item, style in zip(plot._items, styles):
            assert item.opts["pen"].widthF() == style.width
            assert item.opts["pen"].style() == style.style
            assert item.zValue() == style.z
        assert plot._items[1] not in plot.plot.getViewBox().addedItems
    plot.log_check.setChecked(True)
    assert _settle(plot)[1, 0] > 0
    assert plot._items[1].opts["pen"].style() == Qt.DashLine


def test_unspecified_styles_and_bounds_keep_existing_defaults(plot):
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    plot.set_curves([("signed", x, np.abs(x))])
    assert plot._items[0].opts["pen"].widthF() == 1.6
    assert plot._items[0].opts["pen"].style() == Qt.SolidLine
    assert plot._items[0] in plot.plot.getViewBox().addedItems
    plot.set_side("folded")
    assert [item.opts["pen"].style() for item in plot._items] == [Qt.SolidLine, Qt.DashLine]


def test_explicit_styles_survive_folded_halves_and_apply_z_to_markers(plot):
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    plot.set_curves([("signed", x, np.abs(x)), ("points", x, np.abs(x))],
                    markers=[False, True], auto_range=[False, True],
                    line_styles=[LineStyle(style=Qt.DotLine), LineStyle(z=4)])
    plot.set_side("folded")
    assert all(item.opts["pen"].style() == Qt.DotLine for item in plot._items[:2])
    assert all(item not in plot.plot.getViewBox().addedItems for item in plot._items[:2])
    assert all(item.zValue() == 4 for item in plot._items[2:])


def test_svg_export_retains_component_dash_and_width(plot, tmp_path):
    from src.gimap.app.presentation.components.plot_export import export_plot

    x = np.linspace(1, 10, 101)
    plot.set_curves([("total", x, x * 10), ("component", x, x)],
                    line_styles=[LineStyle(width=2.4), LineStyle(width=1.25, style=Qt.DashLine)])
    _settle(plot)
    target = export_plot(plot, tmp_path / "components.svg")
    svg = target.read_text(encoding="utf-8")
    assert 'stroke-width="1.25"' in svg
    assert "stroke-dasharray=" in svg
    assert plot._items[1].opts["pen"].style() == Qt.DashLine
