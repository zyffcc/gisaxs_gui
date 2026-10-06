"""Fitting ▸ Single analysis: Undo covers the model, the fitting range and the left-out points (fitting-7)."""

from __future__ import annotations

import pytest

from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from tests.test_fit_page import _curve_file, _page, _wait
from tests.test_wave2b_fitting_series_list import _close


def _fitted_page(tmp_path):
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.method_buttons["local"].setChecked(True)
    page.run_fit()
    _wait(page)
    assert page.session.result_is_current()
    return page


def test_undo_after_leaving_a_point_out_takes_it_back_and_the_fit_holds_again(tmp_path) -> None:
    page = _fitted_page(tmp_path)
    fitted = page.session.model
    full = page.session.all_points()
    page.exclude_button.setChecked(True)
    page._clicked(float(full.q[100]), float(full.intensity[100]))  # a point left out
    assert page.left_out() == 1 and not page.session.result_is_current()
    assert page.undo_button.isEnabled()
    page.undo()  # the exclusion, not the fit
    assert page.left_out() == 0 and page.session.model == fitted
    assert page.session.result_is_current() and page.step_rail.state("fit") == "ok"
    assert page.model_editor.row((0, "R")).error.text().startswith("± ")  # the fit's errors are back
    assert "changed after this fit" not in page.warnings_label.text()
    page.redo()
    assert page.left_out() == 1 and not page.session.result_is_current()
    page.undo()
    page.undo()  # then the fit itself: the starting model
    assert page.session.model != fitted and page.left_out() == 0
    _close(page)


def test_a_box_and_include_all_are_one_step_each(tmp_path) -> None:
    from PyQt5.QtCore import QRectF

    page = _page()
    page.open_curve(_curve_file(tmp_path))
    full = page.session.all_points()
    xs, ys = page._shown(full.q[10:20], full.intensity[10:20])
    page._exclude_box(QRectF(float(xs.min()) - 1e-9, float(ys.min()) - 1e-9,
                             float(xs.max() - xs.min()) + 2e-9, float(ys.max() - ys.min()) + 2e-9))
    boxed = set(page.session.excluded)
    assert len(boxed) >= 10
    page.include_all()
    assert page.left_out() == 0
    page.undo()  # Include All taken back: the box's points out again, all at once
    assert page.session.excluded == boxed
    page.undo()  # the box taken back
    assert page.left_out() == 0 and not page.session.can_undo()
    _close(page)


def test_a_drag_or_typing_of_the_range_is_one_step(tmp_path, monkeypatch) -> None:
    from src.gimap.features.fitting.presentation.single import session as session_module

    clock = [100.0]
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock[0])
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    assert page.session.q_range is None
    for low in (0.20, 0.25, 0.30, 0.35):  # the band dragged on (and the spin boxes' arrows): within a second
        page._band_dragged(low, 1.5)
        clock[0] += 0.2
    assert page.session.q_range == pytest.approx((0.35, 1.5))
    clock[0] += 5.0  # a while later: a step of its own
    page.range_max_spin.setValue(1.2)  # typed (keyboard tracking off: one value)
    assert page.session.q_range == pytest.approx((0.35, 1.2))
    page.undo()
    assert page.session.q_range == pytest.approx((0.35, 1.5))
    assert page.range_max_spin.value() == pytest.approx(1.5)  # the spin boxes follow
    page.undo()  # the whole drag at once
    assert page.session.q_range is None and not page.session.can_undo()
    page.redo()
    assert page.session.q_range == pytest.approx((0.35, 1.5))
    page.whole_range_button.click()  # a button: a step of its own, the redo history gone
    assert page.session.q_range is None and not page.session.can_redo()
    page.undo()
    assert page.session.q_range == pytest.approx((0.35, 1.5))
    _close(page)


def test_a_burst_that_comes_back_where_it_started_leaves_nothing_to_undo() -> None:
    from src.gimap.features.fitting.presentation.single.session import FitSession

    session = FitSession()
    session.set_range((0.1, 1.0), coalesce=True)
    assert session.can_undo()
    session.set_range(None, coalesce=True)  # dragged back within the burst
    assert not session.can_undo() and not session.undo()


def test_undo_past_another_curve_brings_back_only_the_model(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_range((0.3, 1.5))
    cylinder = FitModel((new_component("cylinder", radius=4.0),))
    page.set_model(cylinder)
    page.open_curve(_curve_file(tmp_path, name="other_fit_input.dat"))
    assert page.session.q_range is None
    page.set_range((0.5, 1.0))
    page.undo()
    assert page.session.q_range is None and page.session.model == cylinder
    page.undo()  # the model before the cylinder; the range of the curve before is not this curve's
    assert page.session.model != cylinder and page.session.q_range is None
    _close(page)


def test_a_project_opened_is_a_new_start_with_nothing_to_undo(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_range((0.3, 1.5))
    page.include_all()
    state = page.project_state()
    other = _page()
    other.open_curve(_curve_file(tmp_path, name="other_fit_input.dat"))
    other.set_model(FitModel((new_component("cylinder", radius=4.0),)))
    other.set_range((0.4, 1.0))
    assert other.session.can_undo()
    other.apply_project_state(state)
    assert other.session.q_range == pytest.approx((0.3, 1.5))
    assert not other.session.can_undo() and not other.undo_button.isEnabled()  # not the work before the project
    other.set_range((0.5, 1.5))
    other.undo()
    assert other.session.q_range == pytest.approx((0.3, 1.5)) and other.session.model == page.session.model
    _close(page, other)


def test_restoring_the_last_session_is_not_a_step_and_the_tooltip_says_what_undo_covers(tmp_path) -> None:
    from PyQt5.QtWidgets import QApplication

    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_range((0.3, 1.2))
    _close(page)
    again = _page()
    again.preferences = page.preferences
    again._restore()
    for _ in range(5):
        QApplication.processEvents()
    assert again.session.q_range == pytest.approx((0.3, 1.2)) and not again.session.can_undo()
    tip = again.undo_button.toolTip()
    assert "fitting range" in tip and "left-out points" in tip and "Ctrl+Z" in tip
    _close(again)
