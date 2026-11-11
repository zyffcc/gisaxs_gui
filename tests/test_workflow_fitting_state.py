"""A curve's physical fit settings belong to that curve and its saved project/session."""

from copy import deepcopy

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.app.project import _close_curve
from src.gimap.features.fitting.presentation.single.page import FitPage
from tests.test_fit_page import _curve_file, _page
from tests.test_fit_page_fit1d import KIT_LOOKED_FOR
from tests.test_handover import _fit_input


@pytest.fixture
def page(monkeypatch):
    apply_language("en")
    monkeypatch.setattr(KIT_LOOKED_FOR, lambda: True)
    shown = _page()
    try:
        yield shown
    finally:
        shown.dispose()
        shown.close()
        apply_language("en")


def _set_nondefault_settings(page):
    for widget, value in ((page.fit1d_network, "au_np_si"), (page.fit1d_policy, "robust"),
                          (page.fit1d_mode, "saxs"), (page.fit1d_composition, "model")):
        index = widget.findData(value)
        assert index >= 0
        widget.setCurrentIndex(index)
    page.fit1d_qz.setValue(0.42)
    page.fit1d_use_qz.setChecked(True)
    page.fit1d_sigma_floor.setValue(3.25)


def _open_project(page, state):
    # Exactly the Single analysis part of app.project.apply, without building unrelated workspaces.
    _close_curve(page)
    return page.apply_project_state(deepcopy(state))


@pytest.mark.parametrize("destination", ["another", "same_path", "same_unchanged", "unknown"])
def test_ordinary_curve_open_does_not_reuse_another_cut_qz(page, tmp_path, destination):
    first = _fit_input(tmp_path / "first_fit_input.dat")
    assert page.open_curve(first)
    page.fit1d_qz.setValue(0.42)
    page.fit1d_use_qz.setChecked(True)
    if destination == "same_path":
        # Analyze can overwrite its existing fit-input file after the cut is moved.
        target = _fit_input(first, scale=2.0)
    elif destination == "same_unchanged":
        target = first
    elif destination == "unknown":
        target = _curve_file(tmp_path, name="external_curve.dat")
    else:
        target = _fit_input(tmp_path / "second_fit_input.dat")
    assert page.open_curve(target)
    assert page.fit1d_settings()["fit1d_qz"] is None
    assert not page.fit1d_use_qz.isChecked()


def test_range_and_display_changes_keep_the_current_cut_qz(page, tmp_path):
    assert page.open_curve(_fit_input(tmp_path / "cut_fit_input.dat"))
    page.fit1d_qz.setValue(0.42)
    page.fit1d_use_qz.setChecked(True)
    page.set_range((0.2, 1.4))
    page.terms_check.setChecked(True)
    page.show_step("model")
    page.show_step("fit")
    assert page.fit1d_settings()["fit1d_qz"] == pytest.approx(0.42)


def test_known_gisaxs_cut_does_not_inherit_saxs_physics(page, tmp_path):
    page.fit1d_mode.setCurrentIndex(page.fit1d_mode.findData("saxs"))
    assert page.open_curve(_fit_input(tmp_path / "gisaxs_fit_input.dat"))
    assert page.fit1d_settings()["fit1d_mode"] == "gisaxs"
    assert page.fit1d_settings()["fit1d_qz"] is None


def test_project_restores_all_fit1d_inputs_and_the_explicit_method(page, tmp_path):
    assert page.open_curve(_fit_input(tmp_path / "project_curve.dat"))
    _set_nondefault_settings(page)
    page.set_method("local")
    expected = page.fit1d_settings()
    project = page.project_state()
    page.fit1d_network.setCurrentIndex(page.fit1d_network.findData("general"))
    page.fit1d_policy.setCurrentIndex(page.fit1d_policy.findData("fast"))
    page.fit1d_mode.setCurrentIndex(page.fit1d_mode.findData("gisaxs"))
    page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("auto:vc:3"))
    page.fit1d_qz.setValue(0.85)
    page.fit1d_sigma_floor.setValue(7.0)
    page.set_method("search")
    assert _open_project(page, project) == []
    assert page.fit1d_settings() == expected
    assert page.session.native_settings == expected
    assert page.method() == "local"


@pytest.mark.parametrize("kind", ["old", "empty"])
def test_project_without_fit1d_settings_uses_initial_defaults(page, tmp_path, kind):
    defaults = page.fit1d_settings()
    assert page.open_curve(_fit_input(tmp_path / "old_project_curve.dat"))
    project = page.project_state() if kind == "old" else {}
    project.pop("fit1d_settings", None)
    _set_nondefault_settings(page)
    assert _open_project(page, project) == []
    assert page.fit1d_settings() == defaults
    assert page.session.native_settings == defaults


def _show_native_result(page):
    """A small successful job result, without depending on the optional numerical runner."""
    data = page.session.data()
    q = data.q[:4].tolist()
    measured = data.intensity[:4].tolist()
    page.session.native_result = {
        "summary": {"status": "ok", "chi2_per_point": 1.0, "n_points": 4,
                    "model": "general", "composition": "sphere"},
        "settings": page.fit1d_settings(),
        "curves": {"q_nm_inv": q, "model": measured, "residual": [0.0] * 4},
        "table": [], "warnings": [],
    }
    page.session.native_model = page.session.model
    page.session.native_selection = page.session.selection()
    page._fit1d_candidates = [{"rank": 1, "composition": "sphere", "chi2_per_point": 1.0,
                               "delta_bic": 0.0, "table": []}]
    page._render_all()
    assert page.session.native_is_current()
    assert page.residual_plot._source


def test_empty_project_removes_old_native_result_candidates_and_residuals(page, tmp_path):
    assert page.open_curve(_fit_input(tmp_path / "previous_curve.dat"))
    _show_native_result(page)
    assert _open_project(page, {}) == []
    assert page.session.curve is None
    state = {
        "native_result": page.session.native_result is not None,
        "native_model": page.session.native_model is not None,
        "native_selection": page.session.native_selection is not None,
        "candidates": len(page._fit1d_candidates),
        "candidate_rows": page.fit1d_candidates_table.rowCount(),
        "residual_curves": len(page.residual_plot._source),
        "main_curves": len(page.plot._source),
    }
    assert state == {key: False if key.startswith("native_") else 0 for key in state}


@pytest.mark.parametrize("incoming", [False, True])
def test_new_session_restores_only_the_saved_curves_fit_settings(page, tmp_path, incoming):
    saved_path = _fit_input(tmp_path / "saved_curve.dat")
    assert page.open_curve(saved_path)
    _set_nondefault_settings(page)
    page.set_method("local")
    page.set_range((0.2, 1.4))
    expected = page.fit1d_settings()
    reopened = FitPage(page.view_model, preferences=page.preferences)
    try:
        if incoming:
            # A Send to Fitting arrives before the deferred reopening of the last session.
            assert reopened.open_curve(_fit_input(tmp_path / "incoming_curve.dat"))
        for _ in range(5):
            QApplication.processEvents()
        if incoming:
            assert reopened.session.curve.name == "incoming_curve.dat"
            assert reopened.fit1d_settings()["fit1d_qz"] is None
            assert reopened.fit1d_settings()["fit1d_mode"] == "gisaxs"
        else:
            assert reopened.session.curve.path == str(saved_path)
            assert reopened.fit1d_settings() == expected
            assert reopened.method() == "local"
            np.testing.assert_allclose(reopened.session.q_range, (0.2, 1.4))
    finally:
        reopened.dispose()
        reopened.close()


@pytest.mark.parametrize("key,bad", [("fit1d_qz", "invalid"), ("fit1d_qz", float("nan")),
                                    ("fit1d_sigma_floor", {"invalid": 1}), ("fit1d_sigma_floor", 0.6)])
def test_invalid_saved_fit1d_value_is_reported_without_abandoning_the_project(page, tmp_path, key, bad):
    defaults = page.fit1d_settings()
    path = _fit_input(tmp_path / "saved_input.dat")
    assert page.open_curve(path)
    saved = page.project_state()
    saved["fit1d_settings"].update({key: bad, "fit1d_policy": "robust"})
    notes = _open_project(page, saved)
    assert len(notes) == 1 and "invalid" in notes[0]
    assert page.session.curve.path == str(path)
    assert page.fit1d_settings()[key] == defaults[key]
    assert page.fit1d_settings()["fit1d_policy"] == "robust"
