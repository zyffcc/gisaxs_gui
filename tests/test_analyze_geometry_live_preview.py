"""Opt-in centre previews recalculate curves safely during manual adjustment."""

from __future__ import annotations

from threading import Event

import numpy as np
import pytest
from PyQt5.QtTest import QTest

from tests.test_analyze_geometry_adjustment import _open, _wheel_up, page


def _settle(page):
    QTest.qWait(230)
    assert page.tasks.wait(10)
    # A completed reduction may release a later preview that was waiting for it.
    QTest.qWait(230)
    assert page.tasks.wait(10)


def _count_reductions(page, monkeypatch):
    analyze = page.view_model.analyze
    requests = []

    def counted(request):
        requests.append(request)
        return analyze(request)

    monkeypatch.setattr(page.view_model, "analyze", counted)
    return requests


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_preview_is_opt_in_and_apply_still_works_with_it_off(page, monkeypatch, kind):
    dialog, x_name, _y = _open(page, kind)
    requests = _count_reductions(page, monkeypatch)
    assert not dialog.live_preview_check.isChecked()
    x_spin = getattr(dialog, x_name)
    x_spin.stepUp()
    _settle(page)
    assert not requests
    assert page.view_model.state.beam_center is None
    dialog.apply_button.click()
    assert page.tasks.wait(10)
    assert len(requests) == 1
    assert dialog.isVisible()
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(x_spin.value())


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_custom_step_wheel_previews_update_q_without_changing_data_or_profile(page, monkeypatch, kind):
    model = page.view_model
    profile = model._profiles.find("Test")
    data_before = model.state.analysis.data.copy()
    q_before = model.state.analysis.reduction.curve("horizontal").x.copy()
    dialog, x_name, y_name = _open(page, kind)
    x_spin, y_spin = getattr(dialog, x_name), getattr(dialog, y_name)
    requests = _count_reductions(page, monkeypatch)
    dialog.live_preview_check.setChecked(True)
    dialog.center_step_spin.setValue(0.0123)
    _settle(page)
    assert not requests  # changing the movement step does not alter scientific geometry
    start = dialog.center()
    _wheel_up(x_spin)
    _settle(page)
    assert len(requests) == 1
    assert model.state.beam_center == pytest.approx((start[0] + 0.0123, start[1]), abs=0.00001)
    q_after = model.state.analysis.reduction.curve("horizontal").x
    assert not np.allclose(q_before, q_after, rtol=0.0, atol=1e-10)
    dialog.center_step_spin.setValue(0.0001)
    _wheel_up(y_spin)
    _settle(page)
    assert len(requests) == 2
    assert model.state.beam_center == pytest.approx((start[0] + 0.0123, start[1] + 0.0001), abs=0.00001)
    np.testing.assert_array_equal(model.state.analysis.data, data_before)
    assert model._profiles.find("Test") == profile
    assert dialog.isVisible()


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_valid_typed_centre_is_previewed_without_enter_or_apply(page, monkeypatch, kind):
    dialog, x_name, _y = _open(page, kind)
    requests = _count_reductions(page, monkeypatch)
    dialog.live_preview_check.setChecked(True)
    x_spin = getattr(dialog, x_name)
    x_spin.lineEdit().selectAll()
    QTest.keyClicks(x_spin.lineEdit(), "200.8037")
    _settle(page)
    assert len(requests) == 1
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(200.8037)
    assert dialog.isVisible()


@pytest.mark.parametrize("kind", ["geometry", "center"])
def test_rapid_adjustments_coalesce_into_one_reduction_of_the_latest_centre(page, monkeypatch, kind):
    dialog, x_name, _y = _open(page, kind)
    requests = _count_reductions(page, monkeypatch)
    dialog.live_preview_check.setChecked(True)
    x_spin = getattr(dialog, x_name)
    for value in (197.1234, 198.2345, 199.3456, 200.4567):
        x_spin.setValue(value)
    _settle(page)
    assert len(requests) == 1
    assert requests[0].beam_center[0] == pytest.approx(200.4567)
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(200.4567)


@pytest.mark.parametrize("action", ["uncheck", "close", "file", "frame", "clear", "dispose"])
def test_pending_preview_is_cancelled_when_its_context_is_left(page, monkeypatch, action):
    dialog, x_name, _y = _open(page, "geometry")
    requests = _count_reductions(page, monkeypatch)
    dialog.live_preview_check.setChecked(True)
    getattr(dialog, x_name).setValue(250.1234)
    if action == "uncheck":
        dialog.live_preview_check.setChecked(False)
    elif action == "close":
        dialog.close()
    elif action == "file":
        page._file_selected(1)
    elif action == "frame":
        page._frame_changed(2)
    elif action == "clear":
        page.clear_files()
    else:
        page.dispose()
        page._geometry_test_disposed = True
    _settle(page)
    assert page.view_model.state.beam_center is None
    assert not any(request.beam_center and request.beam_center[0] == 250.1234 for request in requests)


def test_busy_preview_waits_and_only_reduces_the_latest_pending_centre(page, monkeypatch):
    dialog, x_name, _y = _open(page, "geometry")
    dialog.live_preview_check.setChecked(True)
    x_spin = getattr(dialog, x_name)
    analyze = page.view_model.analyze
    started, release = Event(), Event()
    requests = []

    def blocked(request):
        requests.append(request)
        if len(requests) == 1:
            started.set()
            assert release.wait(10), "test did not release its preview worker"
        return analyze(request)

    monkeypatch.setattr(page.view_model, "analyze", blocked)
    x_spin.setValue(198.1234)
    QTest.qWait(230)
    assert started.wait(2)
    try:
        for value in (199.2345, 200.3456, 201.4567):
            x_spin.setValue(value)
        QTest.qWait(230)
        assert len(requests) == 1
        assert page.view_model.state.beam_center[0] == pytest.approx(198.1234)
    finally:
        release.set()
    _settle(page)
    assert len(requests) == 2
    assert requests[1].beam_center[0] == pytest.approx(201.4567)
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(201.4567)


def test_a_failed_preview_does_not_prevent_the_next_adjustment(page, monkeypatch):
    dialog, x_name, _y = _open(page, "center")
    dialog.live_preview_check.setChecked(True)
    analyze = page.view_model.analyze
    requests = []

    def fail_once(request):
        requests.append(request)
        if len(requests) == 1:
            raise ValueError("Synthetic preview failure")
        return analyze(request)

    monkeypatch.setattr(page.view_model, "analyze", fail_once)
    getattr(dialog, x_name).setValue(198.1234)
    _settle(page)
    assert len(requests) == 1
    getattr(dialog, x_name).setValue(200.8037)
    _settle(page)
    assert len(requests) == 2
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(200.8037)
    assert dialog.isVisible()


def test_two_live_editors_finish_at_the_most_recent_user_adjustment_without_a_loop(page, monkeypatch):
    geometry, _x, _y = _open(page, "geometry")
    center, _x, _y = _open(page, "center")
    requests = _count_reductions(page, monkeypatch)
    geometry.live_preview_check.setChecked(True)
    center.live_preview_check.setChecked(True)
    geometry.center_x_spin.setValue(199.1234)
    center.x_spin.setValue(200.8037)
    _settle(page)
    assert 1 <= len(requests) <= 2
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(200.8037)
    count = len(requests)
    _settle(page)
    assert len(requests) == count


def test_unchecking_during_a_busy_preview_drops_its_unsubmitted_newer_centre(page, monkeypatch):
    dialog, x_name, _y = _open(page, "center")
    dialog.live_preview_check.setChecked(True)
    analyze = page.view_model.analyze
    started, release = Event(), Event()
    requests = []

    def blocked(request):
        requests.append(request)
        started.set()
        assert release.wait(10), "test did not release its preview worker"
        return analyze(request)

    monkeypatch.setattr(page.view_model, "analyze", blocked)
    getattr(dialog, x_name).setValue(198.1234)
    QTest.qWait(230)
    assert started.wait(2)
    try:
        getattr(dialog, x_name).setValue(200.8037)
        QTest.qWait(230)
        dialog.live_preview_check.setChecked(False)
    finally:
        release.set()
    _settle(page)
    assert len(requests) == 1
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(198.1234)


def test_live_preview_keeps_the_persons_manual_curve_zoom(page):
    dialog, x_name, _y = _open(page, "geometry")
    dialog.live_preview_check.setChecked(True)
    boxes = [plot.plot.getViewBox() for plot in (page.top_plot, page.bottom_plot)]
    for box in boxes:
        box.setRange(xRange=(0.001, 0.02), yRange=(1.5, 3.0), padding=0)
    kept = [np.asarray(box.viewRange()).copy() for box in boxes]
    getattr(dialog, x_name).setValue(200.8037)
    _settle(page)
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(200.8037)
    for box, before in zip(boxes, kept):
        np.testing.assert_allclose(box.viewRange(), before, rtol=0.0, atol=1e-12)


def test_incomplete_input_cancels_a_pending_preview_until_the_value_is_valid(page, monkeypatch):
    dialog, x_name, _y = _open(page, "center")
    requests = _count_reductions(page, monkeypatch)
    dialog.live_preview_check.setChecked(True)
    spin = getattr(dialog, x_name)
    spin.setValue(200.8037)
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin.lineEdit(), "-")
    assert not spin.hasAcceptableInput()
    _settle(page)
    assert not requests
    assert page.view_model.state.beam_center is None
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin.lineEdit(), "201.1256")
    _settle(page)
    assert len(requests) == 1
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(201.1256)


def test_an_old_inflight_preview_cannot_replace_the_new_file_analysis(page, monkeypatch):
    dialog, x_name, _y = _open(page, "geometry")
    dialog.live_preview_check.setChecked(True)
    analyze = page.view_model.analyze
    started, release = Event(), Event()
    shown = []
    page.analysisShown.connect(lambda analysis: shown.append(analysis.path.name))

    def blocked_old(request):
        if request.path.name == "a.cbf":
            started.set()
            assert release.wait(10), "test did not release its preview worker"
        return analyze(request)

    monkeypatch.setattr(page.view_model, "analyze", blocked_old)
    getattr(dialog, x_name).setValue(198.1234)
    QTest.qWait(230)
    assert started.wait(2)
    try:
        page._file_selected(1)
        assert page.view_model.current_path.name == "b.cbf"
    finally:
        release.set()
    _settle(page)
    assert page.view_model.state.analysis.path.name == "b.cbf"
    assert shown and all(name == "b.cbf" for name in shown)


def test_immediate_save_cancels_queued_previews_and_keeps_the_saved_profile_centre(page, monkeypatch):
    geometry, _x, _y = _open(page, "geometry")
    center, _x, _y = _open(page, "center")
    requests = _count_reductions(page, monkeypatch)
    geometry.live_preview_check.setChecked(True)
    center.live_preview_check.setChecked(True)
    geometry.center_x_spin.setValue(200.8037)
    geometry.center_y_spin.setValue(110.1234)
    # A different point is still queued in the other editor when Save is pressed.
    center.x_spin.setValue(205.2345)
    geometry.save_button.click()
    _settle(page)
    assert len(requests) == 1
    assert page.view_model.state.beam_center is None
    saved = page.view_model._profiles.find("Test").geometry
    assert (saved.beam_center_x_px, saved.beam_center_y_px) == pytest.approx((200.8037, 110.1234))
    assert page.view_model.state.analysis.geometry == saved
    assert geometry.isVisible() and center.isVisible()


def test_a_newer_manual_apply_cancels_an_older_preview_from_the_other_editor(page, monkeypatch):
    geometry, _x, _y = _open(page, "geometry")
    center, _x, _y = _open(page, "center")
    profile_before = page.view_model._profiles.find("Test")
    requests = _count_reductions(page, monkeypatch)
    center.live_preview_check.setChecked(True)
    center.x_spin.setValue(198.1234)
    geometry.center_x_spin.setValue(201.1256)
    geometry.center_y_spin.setValue(109.8765)
    geometry.apply_button.click()
    _settle(page)
    assert len(requests) == 1
    assert page.view_model.state.beam_center == pytest.approx((201.1256, 109.8765))
    analysis = page.view_model.state.analysis
    assert (analysis.geometry.beam_center_x_px, analysis.geometry.beam_center_y_px) == pytest.approx((201.1256, 109.8765))
    assert page.view_model._profiles.find("Test") == profile_before
