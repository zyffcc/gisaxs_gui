"""A centre search runs off the GUI thread and never overwrites newer Analyze settings."""

from __future__ import annotations

from threading import Event, get_ident

import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.presentation import AnalyzePage
from src.gimap.features.analyze.presentation.bindings import beam_center
from src.gimap.features.analyze.presentation.bindings.beam_center import CenterDialog
from tests.test_analyze_frames_and_center import FakeFrames, _geometry, _symmetric_frame, _view_model


@pytest.fixture
def page(tmp_path, qt_application):
    del qt_application
    frames = FakeFrames({"a.cbf": [_symmetric_frame(), _symmetric_frame()], "b.cbf": [_symmetric_frame()]})
    model = _view_model(frames)
    model.set_mode("gisaxs")
    model.set_horizontal_band(40.0, 45.0)
    model.set_beam_center(24.0, _geometry().beam_center_y_px)
    widget = AnalyzePage(model)
    widget.add_paths([tmp_path / "a.cbf", tmp_path / "b.cbf"])
    assert widget.tasks.wait(10)
    yield widget
    if not getattr(widget, "_symmetry_test_disposed", False):
        widget.dispose()
    assert widget.tasks.wait(10)
    widget.close()


def _blocked_search(page, monkeypatch):
    result = page.view_model.find_center_x()
    started, release = Event(), Event()
    calls = []

    def find(analysis, *, cancelled):
        calls.append((analysis, cancelled, get_ident()))
        started.set()
        assert release.wait(10), "test did not release the worker"
        return result

    monkeypatch.setattr(page.view_model, "find_center_x", find)
    page.refine_center_x()
    assert started.wait(2)
    return result, release, calls


def test_centre_search_runs_in_background_then_applies_x_only(page, monkeypatch):
    notices = []
    monkeypatch.setattr(beam_center, "show_toast", lambda _parent, text, **_options: notices.append(text))
    profile_before = page.view_model._profiles.find("Test")
    main_thread = get_ident()
    result, release, calls = _blocked_search(page, monkeypatch)
    try:
        assert calls[0][0] is page.view_model.state.analysis
        assert calls[0][2] != main_thread
        assert page.view_model.state.beam_center[0] == 24.0
        assert page.symmetry_button.text() == "Cancel Centre Search"
        assert page.symmetry_center_action.isEnabled()
        assert "coarse" in page.status_text()
    finally:
        release.set()
    assert page.tasks.wait(10)
    assert page.view_model.state.beam_center == pytest.approx((result.x_px, _geometry().beam_center_y_px))
    assert page.view_model.state.analysis.geometry.beam_center_x_px == pytest.approx(result.x_px)
    assert page.view_model._profiles.find("Test") == profile_before
    assert page.symmetry_button.text() == "Make Left and Right Symmetric"
    assert len(notices) == 1
    assert f"{result.coarse_x_px:.4f} → refined {result.x_px:.4f}" in notices[0]
    assert "Numerical tolerance 0.0001 px" in notices[0]
    assert "not measured accuracy" in notices[0]


def test_second_click_cancels_search_and_drops_its_result(page, monkeypatch):
    result, release, calls = _blocked_search(page, monkeypatch)
    del result
    try:
        page.symmetry_button.click()
        assert calls[0][1]()
        assert page.symmetry_button.text() == "Make Left and Right Symmetric"
        assert page.status_text() == "Beam centre search cancelled."
    finally:
        release.set()
    assert page.tasks.wait(10)
    assert page.view_model.state.beam_center[0] == 24.0


@pytest.mark.parametrize("change", ["center", "band", "mask", "frame"])
def test_late_result_never_applies_to_changed_settings(page, monkeypatch, change):
    result, release, calls = _blocked_search(page, monkeypatch)
    del result, calls
    model = page.view_model
    try:
        if change == "center":
            model.set_beam_center(30.0, 108.0)
        elif change == "band":
            model.set_horizontal_band(50.0, 55.0)
        elif change == "mask":
            model.set_valid_range(10.0, None)
        else:
            model.set_frame(1)
        expected = model.state.beam_center
    finally:
        release.set()
    assert page.tasks.wait(10)
    assert model.state.beam_center == expected
    assert page.symmetry_button.text() == "Make Left and Right Symmetric"


@pytest.mark.parametrize("action", ["reanalyse", "clear", "dispose"])
def test_page_lifecycle_cancels_running_centre_search(page, monkeypatch, action):
    result, release, calls = _blocked_search(page, monkeypatch)
    del result
    try:
        if action == "reanalyse":
            page.view_model.set_horizontal_band(50.0, 55.0)
            page.run_analysis()
            assert not page.symmetry_button.isEnabled()
        elif action == "clear":
            page.clear_files()
        else:
            page.dispose()
            page._symmetry_test_disposed = True
        assert calls[0][1]()
    finally:
        release.set()
    assert page.tasks.wait(10)
    assert page.view_model.state.beam_center[0] == 24.0


def test_failed_search_restores_controls_and_translates_the_error(page, monkeypatch):
    def find(_analysis, *, cancelled):
        del cancelled
        raise ValueError("The horizontal cut has several similarly supported symmetry axes.")

    monkeypatch.setattr(page.view_model, "find_center_x", find)
    try:
        apply_language("zh", [page])
        page.refine_center_x()
        assert page.tasks.wait(10)
        assert page.status_level() == "warning"
        assert "水平切线存在多个可信程度相近的对称轴" in page.status_text()
        assert page.symmetry_button.text() == "使左右对称"
        assert "0.0001" in page.symmetry_button.toolTip()
        assert page.view_model.state.beam_center[0] == 24.0
    finally:
        apply_language("en", [page])


def test_coordinate_dialog_keeps_four_decimal_centre(qt_application):
    del qt_application
    dialog = CenterDialog((200.8037, 110.1234))
    try:
        assert dialog.x_spin.decimals() == 4
        assert dialog.center() == pytest.approx((200.8037, 110.1234), abs=0.00001)
    finally:
        dialog.close()
