"""Repeated manual centre adjustments update Analyze while the editor stays open."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5 import sip
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.analyze.presentation import AnalyzePage
from src.gimap.app.presentation.i18n import apply_language, tr
from tests.test_analyze_frames_and_center import FakeFrames, _geometry, _symmetric_frame, _view_model


@pytest.fixture
def page(tmp_path, qt_application):
    del qt_application
    frames = FakeFrames({"a.cbf": [_symmetric_frame(), _symmetric_frame()], "b.cbf": [_symmetric_frame()]})
    model = _view_model(frames)
    model.set_mode("gisaxs")
    model.set_horizontal_band(40.0, 45.0)
    widget = AnalyzePage(model)
    widget.add_paths([tmp_path / "a.cbf", tmp_path / "b.cbf"])
    assert widget.tasks.wait(10)
    yield widget
    if not getattr(widget, "_geometry_test_disposed", False):
        widget.dispose()
    assert widget.tasks.wait(10)
    widget.close()


def _open(page, kind):
    if kind == "geometry":
        page._enter_geometry()
        return page._geometry_dialog, "center_x_spin", "center_y_spin"
    page._enter_center()
    return page._center_dialog, "x_spin", "y_spin"


def _wheel_up(spin):
    local = QPointF(spin.rect().center())
    event = QWheelEvent(
        local, QPointF(spin.mapToGlobal(spin.rect().center())), QPoint(), QPoint(0, 120),
        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
    )
    QApplication.sendEvent(spin, event)


def _visible(dialog):
    return not sip.isdeleted(dialog) and dialog.isVisible()


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_centre_step_controls_real_keyboard_and_wheel_movement(page, kind):
    dialog, x_name, y_name = _open(page, kind)
    x_spin, y_spin = getattr(dialog, x_name), getattr(dialog, y_name)
    assert dialog.isVisible() and not dialog.isModal()
    assert x_spin.decimals() == y_spin.decimals() == 4
    assert dialog.center_step_spin.minimum() == pytest.approx(0.0001)
    assert dialog.center_step_spin.decimals() == 4
    dialog.center_step_spin.setValue(0.0123)
    x_spin.setValue(200.8037)
    y_spin.setValue(110.1234)
    QTest.keyClick(x_spin, Qt.Key_Up)
    _wheel_up(y_spin)
    assert dialog.center() == pytest.approx((200.8160, 110.1357), abs=0.00001)
    QTest.keyClick(x_spin, Qt.Key_Down)
    assert x_spin.value() == pytest.approx(200.8037, abs=0.00001)
    # A pending adjustment is not applied to scientific geometry until Apply.
    assert page.view_model.state.analysis.geometry == _geometry()
    dialog.center_step_spin.setValue(0.0001)
    _wheel_up(y_spin)
    assert y_spin.value() == pytest.approx(110.1358, abs=0.00001)


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_repeated_apply_changes_curve_q_without_closing_or_saving_profile(page, kind):
    model = page.view_model
    saved = model._profiles.find("Test")
    data_before = model.state.analysis.data.copy()
    q_before = model.state.analysis.reduction.curve("horizontal").x.copy()
    dialog, x_name, y_name = _open(page, kind)
    x_spin, y_spin = getattr(dialog, x_name), getattr(dialog, y_name)
    for center in ((200.8037, 110.1234), (201.1256, 109.8765)):
        x_spin.setValue(center[0])
        y_spin.setValue(center[1])
        dialog.apply_button.click()
        assert page.tasks.wait(10)
        analysis = model.state.analysis
        assert dialog.isVisible()
        assert model.state.beam_center == pytest.approx(center, abs=0.00001)
        assert (analysis.geometry.beam_center_x_px, analysis.geometry.beam_center_y_px) == pytest.approx(center)
        q_after = analysis.reduction.curve("horizontal").x
        assert q_before.shape == q_after.shape and not np.allclose(q_before, q_after, atol=1e-10)
        np.testing.assert_array_equal(analysis.data, data_before)
        assert model._profiles.find("Test") == saved
        q_before = q_after.copy()
    dialog.close()
    assert model.state.beam_center == pytest.approx((201.1256, 109.8765))


def test_geometry_save_stores_full_geometry_and_keeps_adjustment_window_open(page):
    model = page.view_model
    dialog, _x, _y = _open(page, "geometry")
    dialog.center_x_spin.setValue(201.1256)
    dialog.center_y_spin.setValue(109.8765)
    dialog.distance_spin.setValue(2350.0)
    dialog.apply_button.click()
    assert page.tasks.wait(10)
    # Apply Centre changes only the session centre; the other entries wait for Save.
    assert model.state.analysis.geometry.distance_m == pytest.approx(2.0)
    assert model._profiles.find("Test").geometry == _geometry()
    dialog.save_button.click()
    assert page.tasks.wait(10)
    assert dialog.isVisible()
    saved = model._profiles.find("Test").geometry
    assert saved.distance_m == pytest.approx(2.35)
    assert (saved.beam_center_x_px, saved.beam_center_y_px) == pytest.approx((201.1256, 109.8765))
    assert model.state.analysis.geometry == saved
    assert model.state.beam_center is None


def test_blank_profile_name_warning_survives_language_changes(page):
    dialog, _x, _y = _open(page, "geometry")
    before = page.view_model._profiles.load_all()
    dialog.name_edit.clear()
    dialog.save_button.click()
    assert dialog.status_label.text() == "Enter a profile name before saving."
    assert dialog.status_label.property("gimapRole") == "warning"
    try:
        apply_language("zh", [page, dialog])
        assert dialog.status_label.text() == "请先输入配置名称再保存。"
        assert dialog.status_label.property("gimapRole") == "warning"
        assert page.view_model._profiles.load_all() == before
    finally:
        apply_language("en", [page, dialog])
    assert dialog.status_label.text() == "Enter a profile name before saving."


@pytest.mark.parametrize("failure", ["analysis", "save"])
def test_failure_feedback_retranslates_the_inner_error(page, monkeypatch, failure):
    dialog, _x, _y = _open(page, "geometry")
    error = "The horizontal cut has several similarly supported symmetry axes."
    if failure == "analysis":
        page._geometry_editor_failed(error)
    else:
        def fail(*_args, **_kwargs):
            raise ValueError(error)
        monkeypatch.setattr(page.view_model, "save_profile", fail)
        dialog.save_button.click()
    assert error in dialog.status_label.text()
    try:
        apply_language("zh", [page, dialog])
        assert tr(error) in dialog.status_label.text()
        assert error not in dialog.status_label.text()
        assert dialog.status_label.property("gimapRole") == "error"
    finally:
        apply_language("en", [page, dialog])
    assert error in dialog.status_label.text()


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_enter_applies_committed_four_decimal_text_and_keeps_editor_open(page, kind):
    dialog, x_name, y_name = _open(page, kind)
    x_spin = getattr(dialog, x_name)
    y_spin = getattr(dialog, y_name)
    x_spin.lineEdit().setText("200.1234")
    y_spin.setValue(110.2345)
    x_spin.setFocus()
    QTest.keyClick(x_spin, Qt.Key_Return)
    assert page.tasks.wait(10)
    assert dialog.isVisible()
    assert page.view_model.state.beam_center == pytest.approx((200.1234, 110.2345), abs=0.00001)


def test_step_preference_is_shared_between_open_editors_and_survives_reopen(page):
    geometry, _x, _y = _open(page, "geometry")
    center, _x, _y = _open(page, "center")
    geometry.center_step_spin.setValue(0.0123)
    assert center.center_step_spin.value() == pytest.approx(0.0123)
    assert center.x_spin.singleStep() == center.y_spin.singleStep() == pytest.approx(0.0123)
    center.center_step_spin.setValue(0.0001)
    assert geometry.center_step_spin.value() == pytest.approx(0.0001)
    assert page.view_model.settings.get("analyze", "center_step_px") == pytest.approx(0.0001)
    geometry.close()
    center.close()
    reopened, _x, _y = _open(page, "geometry")
    assert reopened.center_step_spin.value() == pytest.approx(0.0001)
    assert reopened.center_x_spin.singleStep() == pytest.approx(0.0001)


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_adjustment_window_is_reused_and_reopens_with_last_applied_centre(page, kind):
    dialog, x_name, y_name = _open(page, kind)
    assert _open(page, kind)[0] is dialog
    getattr(dialog, x_name).setValue(200.8037)
    getattr(dialog, y_name).setValue(110.1234)
    dialog.apply_button.click()
    assert page.tasks.wait(10)
    dialog.close()
    reopened, _x, _y = _open(page, kind)
    assert reopened.isVisible()
    assert reopened.center() == pytest.approx((200.8037, 110.1234), abs=0.00001)


@pytest.mark.parametrize("action", ["clear", "dispose"])
def test_page_lifecycle_closes_both_adjustment_windows(page, action):
    geometry, _x, _y = _open(page, "geometry")
    center, _x, _y = _open(page, "center")
    if action == "clear":
        page.clear_files()
    else:
        page.dispose()
        page._geometry_test_disposed = True
    assert not _visible(geometry) and not _visible(center)


def test_changing_frame_closes_old_editors_and_cannot_apply_old_detector_values(page):
    geometry, _x, _y = _open(page, "geometry")
    center, _x, _y = _open(page, "center")
    geometry.center_x_spin.setValue(250.1234)
    center.x_spin.setValue(270.1234)
    page._file_selected(1)
    assert page.tasks.wait(10)
    assert not _visible(geometry) and not _visible(center)
    assert page.view_model.state.analysis.path.name == "b.cbf"
    for dialog in (geometry, center):
        if not sip.isdeleted(dialog):
            dialog.apply_button.click()
    assert page.tasks.wait(10)
    assert page.view_model.state.beam_center is None
    assert page.view_model.state.analysis.geometry == _geometry()
