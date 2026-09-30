"""Regression coverage for persistent Matplotlib projections and overlay ownership."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QGraphicsView
from matplotlib.figure import Figure
from src.gimap.features.fitting.presentation.curve_rendering import (
    CurvePlotSpec,
    CurveSeries,
    render_curve_plot,
)
from src.gimap.features.fitting.presentation.bindings.curve_plot import CurvePlotMixin
from src.gimap.features.fitting.presentation.bindings.fit_graphics_events import (
    FitGraphicsEventsMixin,
)


_TEST_APP = None


@pytest.fixture(scope="module")
def app():
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


def spec(value, roi=None):
    return CurvePlotSpec(
        (
            CurveSeries(np.arange(1.0, 4.0), np.full(3, value), "data", "blue"),
            CurveSeries(np.arange(1.0, 4.0), np.full(3, value + 1), "fit", "red", style="line"),
        ),
        "q",
        "I",
        "Curve",
        roi_bounds=roi,
    )


def test_curve_artist_identity_roi_lifecycle_and_autoscale():
    ax = Figure().add_subplot()
    render_curve_plot(ax, spec(1, (1, 2)))
    scatter, lines, legend = ax.collections[0], tuple(ax.lines), ax.get_legend()
    for i in range(2, 22):
        render_curve_plot(ax, spec(i, (1.1, 2.1)))
        assert ax.collections[0] is scatter
        assert tuple(ax.lines) == lines
        assert ax.get_legend() is legend
        assert ax.get_ylim()[1] >= i + 1
    render_curve_plot(ax, spec(2))
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    render_curve_plot(ax, CurvePlotSpec((), "q", "I", "Empty"))
    assert len(ax.lines) == len(ax.collections) == 0
    assert ax.get_legend() is None


class CurveHarness(CurvePlotMixin, FitGraphicsEventsMixin):
    def __init__(self):
        self.ui = SimpleNamespace(fitGraphicsView=QGraphicsView())
        self.errors = []
        self.status_updated = SimpleNamespace(emit=self.errors.append)
        self.value = 1

    def _has_valid_data(self):
        return True

    def _build_curve_plot_spec(self, mode):
        return spec(self.value)

    def _apply_fit_y_axis_limits(self, *args, **kwargs):
        pass


def test_curve_scene_canvas_reuse_and_clear_reopen(app):
    harness = CurveHarness()
    harness._update_GUI_image("normal")
    canvas, figure = harness._current_fit_canvas, harness._current_fit_figure
    artist = figure.axes[0].collections[0]
    for i in range(20):
        harness.value = i + 1
        harness._update_GUI_image("normal")
        app.processEvents()
        assert harness._current_fit_canvas is canvas
        assert harness._current_fit_figure is figure
        assert figure.axes[0].collections[0] is artist
        assert len(harness._curve_graphics_scene.items()) == 1
    assert not harness.errors
    harness._clear_fit_graphics_view()
    assert harness._current_fit_canvas is None
    harness._update_GUI_image("normal")
    assert harness._current_fit_canvas is not canvas
    assert len(harness._curve_graphics_scene.items()) == 1
    harness.ui.fitGraphicsView.close()

