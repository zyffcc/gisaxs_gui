"""Regression coverage for persistent Matplotlib projections and overlay ownership."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QGraphicsView
from matplotlib.figure import Figure
from src.gimap.features.waxs.presentation.image_viewer import ScatteringImageViewer
from src.gimap.features.fitting.presentation.curve_rendering import (
    CurvePlotSpec,
    CurveSeries,
    render_curve_plot,
)
from src.gimap.features.fitting.presentation.bindings.detector_display import DetectorDisplayMixin
from src.gimap.features.fitting.presentation.bindings.fit_graphics_events import (
    FitGraphicsEventsMixin,
)


_TEST_APP = None


@pytest.fixture(scope="module")
def app():
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


class DisplayModel:
    def prepare_display(self, data, **options):
        data = np.array(data, dtype=float, copy=True)
        data[(data < options["mask_min"]) | (data > options["mask_max"])] = np.nan
        if options["log_scale"]:
            data = np.log(np.maximum(data, 0.001))
        return np.flipud(data) if options["flip_vertical"] else data


def show(viewer, data, **kw):
    options = dict(
        log_scale=False,
        colormap="viridis",
        auto_scale=False,
        vmin=0.0,
        vmax=20.0,
        mask_min=-1e12,
        mask_max=1e12,
        flip_vertical=False,
        title="Detector",
    )
    options.update(kw)
    viewer.show_image(data, **options)


def test_waxs_reuses_canvas_image_colorbar_and_preserves_zoom(app):
    viewer = ScatteringImageViewer(view_model=DisplayModel())
    data = np.arange(80.0).reshape(8, 10)
    show(viewer, data)
    canvas, ax, artist, colorbar = viewer.canvas, viewer.ax, viewer.ax.images[0], viewer.colorbar
    ax.set_xlim(2, 4)
    for index in range(20):
        show(viewer, data + index, colormap="plasma", log_scale=bool(index % 2))
        app.processEvents()
        assert (viewer.canvas, viewer.ax, viewer.ax.images[0], viewer.colorbar) == (
            canvas,
            ax,
            artist,
            colorbar,
        )
        assert len(viewer.figure.axes) == 2
        assert len(ax.images) == 1
        assert ax.get_xlim() == (2, 4)
    np.testing.assert_equal(data, np.arange(80.0).reshape(8, 10))
    viewer.close()


def test_waxs_q_mesh_reuse_geometry_mutation_and_mode_switch(app):
    viewer = ScatteringImageViewer(view_model=DisplayModel())
    x, y = np.meshgrid([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], np.arange(4.0))
    data = np.ones_like(x)
    show(viewer, data, q_coordinates=(x, y))
    meshes = tuple(viewer.ax.collections)
    colorbar = viewer.colorbar
    for i in range(10):
        show(viewer, data * i, q_coordinates=(x.copy(), y.copy()))
        assert tuple(viewer.ax.collections) == meshes
        for mesh in meshes:
            np.testing.assert_equal(mesh.get_array(), i)
    x *= 2
    show(viewer, data, q_coordinates=(x, y))
    assert tuple(viewer.ax.collections) != meshes
    for _ in range(5):
        show(viewer, data)
        assert len(viewer.ax.collections) == 0
        assert len(viewer.ax.images) == 1
        show(viewer, data, q_coordinates=(x, y), flip_vertical=True)
        assert len(viewer.ax.collections) == 2
        assert len(viewer.ax.images) == 0
        assert viewer.colorbar is colorbar
        assert len(viewer.figure.axes) == 2
        viewer.canvas.draw()
    viewer.close()


def test_overlay_clear_does_not_remove_selector_or_image(app):
    viewer = ScatteringImageViewer(view_model=DisplayModel())
    show(viewer, np.ones((4, 4)))
    from matplotlib.widgets import RectangleSelector

    selector = RectangleSelector(viewer.ax, lambda *args: None, interactive=True)
    for _ in range(20):
        viewer._overlay_artists = viewer.ax.plot([0, 1], [0, 1])
        show(viewer, np.ones((4, 4)))
        assert len(viewer.ax.lines) == len([a for a in selector.artists if a in viewer.ax.lines])
        assert all(a in viewer.ax.get_children() for a in selector.artists)
    selector.disconnect_events()
    viewer.close()


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


class CurveHarness(DetectorDisplayMixin, FitGraphicsEventsMixin):
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


def test_waxs_shape_limits_and_1d_roundtrip(app):
    viewer = ScatteringImageViewer(view_model=DisplayModel())
    show(viewer, np.ones((4, 6)))
    artist = viewer.ax.images[0]
    show(viewer, np.ones((8, 12)), vmin=100, vmax=200)
    assert viewer.ax.images[0] is artist
    assert artist.get_clim() == (100, 200)
    assert viewer.ax.get_xlim() == (0, 12)
    assert viewer.ax.get_ylim() == (8, 0)
    (overlay,) = viewer.ax.plot([1000], [1000])
    assert viewer.ax.get_xlim() == (0, 12)
    viewer._overlay_artists = [overlay]
    canvas = viewer.canvas
    viewer.figure.clear()
    viewer.ax = viewer.figure.add_subplot(111)
    viewer.cax = None
    viewer.colorbar = None
    viewer.ax.plot([1, 2], [3, 4])
    show(viewer, np.ones((4, 6)))
    assert viewer.canvas is canvas
    assert len(viewer.figure.axes) == 2
    assert len(viewer.ax.images) == 1
    assert not viewer.ax.lines
    viewer.canvas.draw()
    viewer.close()


def test_legacy_plot_refresh_keeps_axes_and_removes_stale_fit():
    from src.gimap.features.fitting.presentation.bindings.plot_refresh import PlotRefreshMixin

    class Harness(PlotRefreshMixin):
        def _convert_q_values_for_display(self, values):
            return np.asarray(values)

        def _build_q_axis_label(self):
            return "q"

        def _get_x_axis_scale(self):
            return "linear"

        def _get_checkbox_state(self, name, default):
            return default

    harness = Harness()
    harness._current_fit_figure = Figure()
    ax = harness._current_fit_figure.add_subplot()
    harness._current_fit_canvas = SimpleNamespace(draw_idle=lambda: None)
    harness.current_cut_data = {"x": np.arange(3.0), "y": np.ones(3)}
    harness.fitting_data = {"x": np.arange(3.0), "y": np.full(3, 2)}
    harness._update_fitting_plot()
    scatter, line = ax.collections[0], ax.lines[0]
    for i in range(10):
        harness._update_fitting_plot()
        assert ax.collections[0] is scatter
        assert ax.lines[0] is line
    harness._update_fitting_plot_points_only()
    assert harness._current_fit_figure.axes == [ax]
    assert len(ax.lines) == 0
    assert ax.collections[0] is scatter


def test_q_mode_does_not_inherit_pixel_or_previous_geometry_bounds(app):
    viewer = ScatteringImageViewer(view_model=DisplayModel())
    data = np.ones((20, 30))
    show(viewer, data)
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 30), np.linspace(0.001, 0.01, 20))
    show(viewer, data, q_coordinates=(x, y))
    assert -0.04 < viewer.ax.get_xlim()[0] < 0 < viewer.ax.get_xlim()[1] < 0.04
    assert 0 < viewer.ax.get_ylim()[0] < viewer.ax.get_ylim()[1] < 0.02
    show(viewer, data, q_coordinates=(x / 10, y / 10))
    assert viewer.ax.get_xlim()[1] < 0.004
    assert viewer.ax.get_ylim()[1] < 0.002
    viewer.close()
