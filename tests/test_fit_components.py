"""Component display is optional; it must not change science or discard a chosen view."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import apply_language, tr
from src.gimap.features.fitting.application.single_fit import evaluate
from src.gimap.features.fitting.presentation.single.plotting import component_label
from tests.test_fit_page import _curve_file, _page, _truth


def _settle():
    for _ in range(4):
        QApplication.processEvents()


@pytest.fixture
def page(tmp_path):
    widget = _page()
    assert widget.open_curve(_curve_file(tmp_path))
    widget.set_model(_truth())
    _settle()
    yield widget
    apply_language("en", [widget])
    widget.dispose()
    widget.close()


def _curves(plot):
    return {name: (np.asarray(q).copy(), np.asarray(y).copy()) for name, q, y in plot.figure_state()["curves"]}


def _install_native(page, tmp_path):
    """A small synthetic native result: no optional runtime or fitting search is needed."""
    q = np.linspace(0.1, 1.0, 40)
    y = 100 + q
    path = tmp_path / "synthetic_components.dat"
    np.savetxt(path, np.column_stack([q, y, np.ones_like(q)]), header="q (nm^-1) I sigma")
    assert page.open_curve(path)
    tiny = 1e-20 * np.exp(-q)
    native = {
        "settings": dict(page.session.native_settings),
        "curves": {
            "q_nm_inv": q.tolist(), "intensity": y.tolist(), "sigma": np.ones_like(q).tolist(),
            "model": y.tolist(), "residual": np.zeros_like(q).tolist(),
            "components": {"pop1_sphere": (y - tiny).tolist(), "pop2_sphere": tiny.tolist(),
                           "background_B": np.zeros_like(q).tolist()},
        },
    }
    page.session.native_result = native
    page.session.native_model = page.session.model
    page.session.native_selection = page.session.selection()
    assert page.session.native_is_current()
    page._redraw()
    _settle()
    return native


def test_manual_contributions_are_additive_and_visibility_keeps_full_export(page, tmp_path):
    model = _truth().with_values({("globals", "k"): 2.4, ("globals", "background"): 7.0,
                                  ("globals", "res_amplitude"): 30.0})
    page.set_model(model)
    page.terms_check.setChecked(True)
    shown = _curves(page.plot)
    q, total = shown[tr("model")]
    expected_total, expected_parts = evaluate(model, q, parts=True)
    np.testing.assert_allclose(total, expected_total)
    np.testing.assert_allclose(sum(shown[component_label(name)][1] for name in expected_parts), total)
    for name, values in expected_parts.items():
        np.testing.assert_array_equal(shown[component_label(name)][1], values)
    # In particular the fitted constant is not scaled by k or normalized as a fraction.
    np.testing.assert_array_equal(shown[component_label("constant background")][1], np.full_like(q, 7.0))
    before = tmp_path / "all_terms.csv"
    after = tmp_path / "hidden_term.csv"
    assert page.export_data_dialog(before) == before
    colors = dict(zip(shown, page.plot.figure_state()["colors"]))
    page._component_actions["1: Sphere"].setChecked(False)
    hidden = _curves(page.plot)
    assert component_label("1: Sphere") not in hidden
    remaining_colors = dict(zip(hidden, page.plot.figure_state()["colors"]))
    assert all(remaining_colors[name] == colors[name] for name in hidden)
    assert page.session.model == model
    assert page.export_data_dialog(after) == after
    assert before.read_bytes() == after.read_bytes()
    page.terms_check.setChecked(False)
    page.terms_check.setChecked(True)
    assert not page._component_actions["1: Sphere"].isChecked()
    assert component_label("1: Sphere") not in _curves(page.plot)


@pytest.mark.parametrize("log_y", [False, True])
def test_component_switches_keep_manual_zoom_and_residuals(page, log_y):
    page.plot.log_check.setChecked(log_y)
    view = page.plot.plot.getViewBox()
    view.setRange(xRange=(0.3, 0.9), yRange=(0.8, 2.5) if log_y else (10.0, 80.0), padding=0)
    _settle()
    limits, auto = np.array(view.viewRange()), list(view.autoRangeEnabled())
    residual_items = tuple(page.residual_plot._items)
    residuals = _curves(page.residual_plot)
    for checked in (True, False, True):
        page.terms_check.setChecked(checked)
        _settle()
        np.testing.assert_allclose(view.viewRange(), limits, rtol=0, atol=1e-10)
        assert view.autoRangeEnabled() == auto
    page._component_actions["constant background"].setChecked(False)
    _settle()
    np.testing.assert_allclose(view.viewRange(), limits, rtol=0, atol=1e-10)
    assert tuple(page.residual_plot._items) == residual_items
    for name, (q, values) in _curves(page.residual_plot).items():
        np.testing.assert_array_equal(q, residuals[name][0])
        np.testing.assert_array_equal(values, residuals[name][1])


@pytest.mark.parametrize("log_y", [False, True])
def test_weak_native_components_affect_limits_only_when_requested(page, tmp_path, log_y):
    _install_native(page, tmp_path)
    page.plot.log_check.setChecked(log_y)
    page.plot.reset_view()
    _settle()
    view = page.plot.plot.getViewBox()
    limits = np.array(view.viewRange())
    page.terms_check.setChecked(True)
    page.plot.reset_view()
    _settle()
    # Reset's X padding can change slightly as Qt lays out differently sized Y tick labels.
    # Components share the measured q grid; they must not change its covered interval.
    np.testing.assert_allclose(view.viewRange()[1], limits[1], rtol=0, atol=1e-10)
    assert 0.0 < view.viewRange()[0][0] < 0.1 and 1.0 < view.viewRange()[0][1] < 1.1
    page.component_range_action.setChecked(True)
    _settle()
    expanded = np.array(view.viewRange())
    assert expanded[1, 0] < limits[1, 0] - (10 if log_y else 50)
    assert np.isfinite(expanded).all()
    page.component_range_action.setChecked(False)
    _settle()
    np.testing.assert_allclose(view.viewRange()[1], limits[1], rtol=0, atol=1e-10)


def test_native_arrays_and_export_keys_survive_display_selection(page, tmp_path):
    native = _install_native(page, tmp_path)
    before = copy.deepcopy(native)
    page.terms_check.setChecked(True)
    shown = _curves(page.plot)
    np.testing.assert_array_equal(shown[tr("fit1d v2 model")][1], native["curves"]["model"])
    for name, values in native["curves"]["components"].items():
        np.testing.assert_array_equal(shown[component_label(name)][1], values)
    # Zero intensity stays zero in the source. Only its logarithmic rendering is absent.
    zero_index = list(shown).index(component_label("background_B"))
    x, y = page.plot._items[zero_index].getData()
    assert y is None or not np.isfinite(y).any()
    page._component_actions["pop2_sphere"].setChecked(False)
    page._component_actions["background_B"].setChecked(False)
    assert native == before
    target = tmp_path / "native.csv"
    assert page.export_data_dialog(target) == target
    names = target.read_text().splitlines()[0]
    assert "pop2_sphere" in names and "background_B" in names
    record = json.loads(target.with_suffix(".json").read_text())
    assert record["native_result"] == before


def test_component_labels_follow_language_without_exposing_native_keys(page, tmp_path):
    native = _install_native(page, tmp_path)
    page.terms_check.setChecked(True)
    apply_language("zh", [page])
    page._redraw(preserve_view=True)
    assert page.terms_check.text() == "组分"
    assert page._component_actions["pop1_sphere"].text() == f"1: {tr('Sphere')}"
    assert page._component_actions["background_B"].text() == tr("constant background")
    assert "Sphere" not in page._component_actions["pop1_sphere"].text()
    assert component_label("low_q_term") == tr("low-q term") != "low-q term"
    assert component_label("2: Random cylinder") == f"2: {tr('Random cylinder')}"
    assert list(native["curves"]["components"]) == ["pop1_sphere", "pop2_sphere", "background_B"]


def test_language_switch_keeps_zoom_and_refreshes_residual_explanation(page):
    page.terms_check.setChecked(True)
    view = page.plot.plot.getViewBox()
    view.setRange(xRange=(0.3, 0.9), yRange=(0.8, 2.5), padding=0)
    _settle()
    limits = np.array(view.viewRange())
    before = page.residual_plot.toolTip()
    apply_language("zh", [page])
    page.refresh_language()
    _settle()
    np.testing.assert_allclose(view.viewRange(), limits, rtol=0, atol=1e-10)
    assert page.residual_plot.toolTip() == page._residuals_tip(True) != before
    assert page.residual_plot.plot.getAxis("left").toolTip() == page.residual_plot.toolTip()
    assert component_label("resolution peak") == "光束尾项"
