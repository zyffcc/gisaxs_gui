"""Offscreen interaction tests for the shared pyqtgraph detector projection."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPointF, QCoreApplication, QEvent

pytest.importorskip("pyqtgraph")
from src.gimap.app.presentation import ScientificImageViewer


_TEST_APP = None


@pytest.fixture(scope="module")
def app():
    global _TEST_APP
    _TEST_APP = QApplication.instance() or QApplication([])
    return _TEST_APP


@pytest.fixture
def viewer(app):
    widget = ScientificImageViewer()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def update(viewer, data, **kw):
    options = dict(
        intensity=data,
        extent=(0, data.shape[1], data.shape[0], 0),
        origin="upper",
        levels=(0, 10),
        colormap="viridis",
        log_scale=False,
    )
    options.update(kw)
    viewer.set_frame(data, **options)


def test_persistent_image_histogram_roi_zoom_and_read_only_data(app, viewer):
    data = np.arange(80.0, dtype=np.float32).reshape(8, 10)
    data[0, 0] = np.nan
    snapshot = data.copy()
    update(viewer, data)
    image, histogram, roi = viewer.image_item, viewer.histogram, viewer.roi
    viewer.view_box.setRange(xRange=(2, 4), yRange=(2, 4), padding=0)
    previous_range = viewer.view_box.viewRange()
    for i in range(20):
        update(viewer, data + i)
        app.processEvents()
        assert viewer.image_item is image
        assert viewer.histogram is histogram
        assert viewer.roi is roi
        np.testing.assert_allclose(viewer.view_box.viewRange(), previous_range)
    np.testing.assert_equal(data, snapshot)
    assert np.isnan(viewer.intensity_at(0.5, 0.5)[2])
    assert viewer.intensity_at(10.5, 0.5) is None
    assert viewer.intensity_at(1.5, 2.5) == (2, 1, data[2, 1] + 19)


def test_gisaxs_origin_mapping_full_resolution_readout_and_mask(viewer):
    raw = np.arange(24.0).reshape(4, 6)
    update(viewer, raw[::2, ::2], intensity=raw, extent=(-0.5, 5.5, -0.5, 3.5), origin="lower")
    assert viewer.intensity_at(0, 0) == (0, 0, 0.0)
    assert viewer.intensity_at(5, 3) == (3, 5, 23.0)
    point = viewer.image_item.mapToView(QPointF(0.5, 0.5))
    assert point.x() == 0.5
    assert point.y() == 0.5


def test_display_signals_do_not_echo_programmatic_refresh(viewer):
    emitted = []
    viewer.log_changed.connect(lambda value: emitted.append(("log", value)))
    viewer.levels_changed.connect(lambda a, b: emitted.append(("levels", a, b)))
    update(viewer, np.ones((8, 8)), log_scale=True)
    assert not emitted
    viewer.log_button.setChecked(False)
    viewer.histogram.setLevels(2, 5)
    viewer._levels_changed()
    assert ("log", False) in emitted
    assert ("levels", 2.0, 5.0) in emitted


def test_roi_requires_apply_and_q_fallback_disables_stale_data(viewer):
    update(viewer, np.ones((10, 10)))
    regions = []
    viewer.region_selected.connect(regions.append)
    viewer.roi_button.setChecked(True)
    viewer.roi.setPos((1, 2))
    viewer.roi.setSize((3, 4))
    assert not regions
    viewer.apply_button.click()
    assert regions == [(1.0, 4.0, 2.0, 6.0)]
    viewer.set_unavailable("Use q-space in Matplotlib")
    viewer.apply_button.click()
    assert len(regions) == 1
    assert viewer.intensity_at(2, 2) is None
    update(viewer, np.ones((20, 20)))
    assert viewer.apply_button.isEnabled()


def test_frame_playback_has_backpressure_and_stops_on_close(viewer):
    update(viewer, np.ones((8, 8)))
    frames = []
    viewer.frame_requested.connect(frames.append)
    viewer.set_frame_position(0, 3)
    viewer.show()
    viewer.play_button.setChecked(True)
    viewer._next_frame()
    viewer._next_frame()
    assert frames == [1]
    viewer.set_frame_position(1, 3)
    viewer._next_frame()
    assert frames == [1, 2]
    viewer.set_frame_position(2, 3)
    viewer._next_frame()
    assert frames == [1, 2, 0]
    viewer.close()
    assert not viewer._timer.isActive()
    assert not viewer._frame_pending


def test_waxs_adapter_routes_roi_and_levels_without_new_scientific_logic():
    from src.gimap.features.waxs.presentation.bindings.interactive_detector import (
        InteractiveDetectorMixin,
    )

    harness = InteractiveDetectorMixin()
    harness.coordinate_mode_combo = SimpleNamespace(currentText=lambda: "Pixel")
    results = []
    harness._on_roi_selected = lambda start, end: results.append(
        (start.xdata, end.xdata, start.ydata, end.ydata)
    )
    harness._interactive_region_selected((1.0, 4.0, 2.0, 6.0))
    assert results == [(1.0, 4.0, 2.0, 6.0)]


def test_fitting_adapter_routes_region_through_existing_selection():
    from src.gimap.features.fitting.presentation.bindings.interactive_detector import (
        InteractiveDetectorMixin,
    )

    harness = InteractiveDetectorMixin()
    harness._should_show_q_axis = lambda: False
    harness._preview_ax = object()
    results = []
    harness._on_main_preview_release = lambda event: results.append(
        (harness._main_preview_selection_start, event.xdata, event.ydata)
    )
    harness._set_main_preview_tool = lambda tool: results.append(tool)
    harness._interactive_region_selected((1.0, 4.0, 2.0, 6.0))
    assert results == [((1.0, 2.0), 4.0, 6.0), None]


def test_real_waxs_projection_reopen_log_mask_and_q_mode(app):
    from src.gimap.app import AppContext
    from src.gimap.integrations.state import (
        InMemorySettingsRepository,
        InMemorySessionRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.features.waxs.bootstrap import create_waxs_view_model
    from src.gimap.features.waxs.presentation.page import InSituProcessingWidget

    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    page = InSituProcessingWidget(view_model=create_waxs_view_model(context))
    page.current_image = np.arange(1.0, 65.0).reshape(8, 8)
    source = page.current_image.copy()
    page.current_file = "test.tif"
    page._sync_selection_defaults_to_image()
    page._open_interactive_detector()
    widget = page.viewer.interactive_window
    image_item = widget.image_item
    assert widget.intensity_at(0.5, 0.5) == (0, 0, 1.0)
    widget.log_button.setChecked(not page.display_log.isChecked())
    assert widget.log_button.isChecked() == page.display_log.isChecked()
    page._interactive_levels_changed(1.0, 5.0)
    assert page.viewer.ax.images[0].get_clim() == widget.histogram.getLevels()
    page.cut_type_combo.setCurrentText("Circle Cut")
    page.refresh_view()
    assert len(widget.overlay_item.getData()[0]) > 0
    widget.close()
    page._open_interactive_detector()
    assert widget is page.viewer.interactive_window
    assert widget.image_item is image_item
    page.coordinate_mode_combo.setCurrentText("q space")
    assert widget._intensity is None
    page.coordinate_mode_combo.setCurrentText("Pixel")
    assert widget._intensity is not None
    np.testing.assert_equal(page.current_image, source)
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    page.close()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_waxs_playback_retries_if_frame_arrives_before_loader_cleanup(viewer):
    from src.gimap.features.waxs.presentation.bindings.interactive_detector import (
        InteractiveDetectorMixin,
    )

    update(viewer, np.ones((8, 8)))
    viewer.set_frame_position(1, 3)
    harness = InteractiveDetectorMixin()
    harness.viewer = SimpleNamespace(interactive_window=viewer)
    harness._loader_thread = SimpleNamespace(isRunning=lambda: True)
    harness.frame_spin = SimpleNamespace(value=lambda: 2)
    harness.current_frame_count = 3
    harness.current_file = "frames.nxs"
    started = []
    harness._start_loader = lambda path, index: started.append((path, index))
    viewer.frame_requested.connect(harness._interactive_frame_requested)
    viewer._next_frame()
    assert not started
    assert not viewer._frame_pending
    harness._loader_thread = None
    viewer._next_frame()
    assert started == [("frames.nxs", 2)]
    assert viewer._frame_pending
