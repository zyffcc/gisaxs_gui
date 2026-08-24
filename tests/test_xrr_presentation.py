"""Offscreen construction and navigation-state tests for the XRR Tools window."""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

from src.gimap.app.presentation.menu_manager import MenuManager
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


def test_tools_menu_opens_one_modeless_xrr_dialog_without_workspace_navigation():
    _app()
    window = QMainWindow()
    window.active_workspace = 3
    created = []

    def factory(parent):
        dialog = QDialog(parent)
        dialog.setModal(False)
        created.append(dialog)
        return dialog

    manager = MenuManager(
        window,
        settings=object(),
        xrr_series_dialog_factory=factory,
    )
    manager.create_tools_menu()
    assert window.actionXrrSeriesExtractor.text() == "XRR Series Extractor..."
    manager.open_xrr_series_extractor()
    manager.open_xrr_series_extractor()
    assert len(created) == 1
    assert window.active_workspace == 3
    created[0].close()
