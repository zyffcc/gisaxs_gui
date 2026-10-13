"""Offscreen construction and navigation-state tests for the XRR Tools window."""

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

from src.gimap.features.xrr.application import (
    XrrExtractionProgress,
    XrrExtractionResult,
    XrrPoint,
)
from src.gimap.features.xrr.presentation.dialog import XrrSeriesDialog


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class FakeViewModel:
    def __init__(self):
        self.state = SimpleNamespace(inspection=None, result=None, error_message="", running=False)

    def cancel(self):
        return True

    def export(self, path):
        self.exported = Path(path)


def _point(index=0):
    return XrrPoint(
        sequence_index=index,
        source_name=f"frame_{index}.cbf",
        frame_index=0,
        theta_deg=0.1 * index,
        qz_inv_angstrom=0.01 * index,
        intensity=100.0 + index,
        roi_center_x_px=12.0,
        roi_center_y_px=18.0,
        valid_pixels=13,
    )


def test_xrr_dialog_builds_core_workflow_with_safe_numeric_inputs():
    _app()
    dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
    assert dialog.windowTitle() == "XRR Series Extractor"
    assert dialog.preview_tabs.count() == 2
    assert dialog.input_section.title_label.text() == "Series data"
    assert dialog.angle_section.title_label.text() == "Sample-angle series"
    assert dialog.geometry_section.title_label.text() == "Detector geometry"
    assert dialog.run_section.title_label.text() == "Extract XRR"
    assert dialog.radius_spin.property("gimapSafeWheelInput") is True
    assert dialog.job_status.pause_button.isHidden()
    dialog.close()


def test_streaming_progress_never_changes_user_selected_result_tab():
    _app()
    view_model = FakeViewModel()
    dialog = XrrSeriesDialog(view_model=view_model, app_context=object())
    point = _point(1)
    view_model.state.result = XrrExtractionResult((point,))
    dialog.preview_tabs.setCurrentIndex(1)

    dialog._on_extraction_progress(
        XrrExtractionProgress(
            completed=1,
            total=3,
            point=point,
            preview=np.ones((20, 30), dtype=np.float32),
            preview_shape=(200, 300),
        )
    )

    assert dialog.preview_tabs.currentIndex() == 1
    assert dialog.results_table.rowCount() == 1
    assert dialog.live_frame_label.text().startswith("1/3")
    dialog.close()


def test_xrr_layout_remains_available_at_required_viewports():
    app = _app()
    dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
    for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
        dialog.resize(width, height)
        dialog.show()
        app.processEvents()
        assert dialog.run_button.isVisible()
        assert dialog.preview_tabs.isVisible()
        run_bottom = dialog.run_button.mapTo(dialog, QPoint(0, dialog.run_button.height())).y()
        assert run_bottom <= dialog.height()
        assert dialog.controls_scroll.verticalScrollBar().maximum() >= 0
    dialog.close()


# -- Esc while extracting, pick-mode Esc, theme ---------------------------------------------------


class SlowViewModel(FakeViewModel):
    """Extraction that runs until released, like a long synthetic stack."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def extract(self, request, *, on_progress=None):
        self.release.wait(10)
        return XrrExtractionResult(())

    def cancel(self):
        self.cancelled.set()
        return True


def _settle(app, seconds=0.1):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)


def test_escape_during_extraction_waits_for_the_thread_before_closing(tmp_path):
    app = _app()
    view_model = SlowViewModel()
    dialog = XrrSeriesDialog(view_model=view_model, app_context=object())
    dialog.source_picker.set_path(str(tmp_path))
    dialog.show()
    events = []
    dialog.destroyed.connect(lambda *_: events.append("destroyed"))
    dialog._run_extraction()
    thread = dialog._extract_thread
    assert thread is not None and thread.isRunning()
    thread.finished.connect(lambda: events.append("finished"))

    QTest.keyClick(dialog, Qt.Key_Escape)
    _settle(app, 0.2)
    assert events == []
    assert view_model.cancelled.is_set()  # Esc asks the extraction to stop
    assert thread.isRunning()

    view_model.release.set()
    end = time.monotonic() + 10
    while "destroyed" not in events and time.monotonic() < end:
        _settle(app, 0.05)
    assert events == ["finished", "destroyed"]


def test_escape_in_pick_mode_only_leaves_pick_mode():
    app = _app()
    dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
    dialog.show()
    dialog.pick_center_button.setChecked(True)
    QTest.keyClick(dialog, Qt.Key_Escape)
    app.processEvents()
    assert not dialog.pick_center_button.isChecked()
    assert dialog.isVisible()
    assert dialog.pick_center_button.text() == "Pick direct-beam center"
    dialog.close()


def test_plots_use_the_theme_plot_background_and_follow_a_switch():
    from src.gimap.app.presentation.theme import apply_theme, theme_color

    _app()
    apply_theme("dark", 9)
    try:
        dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
        dialog.plotter.render_detector(
            np.ones((20, 30)),
            (200, 300),
            roi_center=(10.0, 12.0),
            radius_px=2,
            direct_center=(15.0, 100.0),
            title="frame",
        )
        dialog.plotter.render_curve(XrrExtractionResult((_point(1), _point(2))), log_y=True)
        for canvas in (dialog.plotter.live_canvas, dialog.plotter.curve_canvas):
            canvas.draw()
            pixel = np.asarray(canvas.buffer_rgba())[2, 2, :3]
            assert tuple(pixel) == theme_color("plot_bg").getRgb()[:3]
        foreground = theme_color("plot_fg").name()
        assert dialog.plotter.live_axis.title.get_color() == foreground
        apply_theme("light", 9)
        for canvas in (dialog.plotter.live_canvas, dialog.plotter.curve_canvas):
            canvas.draw()
            pixel = np.asarray(canvas.buffer_rgba())[2, 2, :3]
            assert tuple(pixel) == theme_color("plot_bg").getRgb()[:3]
        dialog.close()
    finally:
        apply_theme("light", 9)
