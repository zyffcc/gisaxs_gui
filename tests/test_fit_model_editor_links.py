"""The Model step's compact parameter lines, links and constraints, the new model choices, and the Uncertainty and
Report window of the single-curve page."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QGridLayout, QToolButton

from src.gimap.features.fitting.application.single_fit import FitModel, new_component
from tests.test_fit_page import _curve_file, _page, _wait


def _two(page):
    model = FitModel((new_component("sphere", radius=3.0, distance=12.0), new_component("sphere", radius=6.0)))
    page.set_model(model)
    return page.model_editor


def _grid_row(row):
    grid = row.value.parentWidget().layout().itemAt(1).layout()
    for index in range(grid.count()):
        if grid.itemAt(index).widget() is row.value:
            return grid
    raise AssertionError("value box not in the card's grid")


def test_a_parameter_is_one_line_with_value_error_range_fit_and_link():
    page = _page()
    try:
        editor = _two(page)
        page.show_step("model")
        page.show_ranges_check.setChecked(True)
        row = editor.row((0, "R"))
        grid = _grid_row(row)
        assert isinstance(grid, QGridLayout)
        lines = {grid.getItemPosition(grid.indexOf(widget))[0]
                 for widget in (row.name, row.value, row.error, row.lower, row.upper, row.free, row.link)}
        assert len(lines) == 1  # name, value, ±, min, max, fit and link on one line
        assert row.lower.isVisibleTo(page) and not row.unbounded.isVisibleTo(page)  # ∞ only for scales
        assert editor.row((0, "Int")).unbounded.isVisibleTo(page)
        page.show_ranges_check.setChecked(False)
        assert not row.lower.isVisibleTo(page) and row.link.isVisibleTo(page)
        combo = editor.findChild(QComboBox, "fitFamily_0")
        assert [combo.itemData(i) for i in range(combo.count())][-1] == "core_shell_sphere"
    finally:
        page.dispose()


def test_links_follow_edits_show_in_the_card_and_go_with_their_component():
    page = _page()
    try:
        editor = _two(page)
        messages = []
        editor.message.connect(lambda text, level: messages.append((text, level)))
        editor.link((1, "R"), "2*R_1")
        editor.link((1, "sigma_R"), "sigma_R_1")
        model = page.session.model
        assert model.get((1, "R")).value == pytest.approx(6.0) and not model.get((1, "R")).free
        row = editor.row((1, "R"))
        assert not row.value.isEnabled() and not row.free.isEnabled() and "2*R_1" in row.value.toolTip()
        editor.row((0, "R")).value.setValue(3.5)
        assert page.session.model.get((1, "R")).value == pytest.approx(7.0)  # the linked value follows
        unlink = editor.findChild(QToolButton, "fitUnlink_R_2")
        assert unlink is not None
        editor.link((0, "R"), "R_2")  # a circle
        assert messages and messages[-1][1] == "warning" and "Could not link" in messages[-1][0]
        editor.add_constraint("D_1 >= 2*R_1 + 2")
        assert page.session.model.inequalities[0].text == "D_1 >= 2*R_1 + 2"
        editor.add_constraint("D_9 >= 1")
        assert "Could not add the constraint" in messages[-1][0] and len(page.session.model.inequalities) == 1
        editor.findChild(QCheckBox, "fitContact").setChecked(False)
        assert not page.session.model.contact
        unlink.click()
        assert page.session.model.get((1, "R")).tie is None
        editor.findChild(QToolButton, "fitRemove_0").click()  # the links and constraints of the first component go
        model = page.session.model
        assert len(model.components) == 1 and not model.inequalities and model.get((0, "sigma_R")).tie is None
        page.undo()
        assert len(page.session.model.components) == 2 and page.session.model.get((1, "sigma_R")).tie == "sigma_R_1"
    finally:
        page.dispose()


def test_interference_functions_and_core_shell_spheres_in_the_model_step():
    page = _page()
    try:
        editor = _two(page)
        combo = editor.findChild(QComboBox, "fitInterference_0")
        combo.setCurrentIndex(combo.findData("hard_sphere"))
        model = page.session.model
        assert model.components[0].interference == "hard_sphere" and editor.row((0, "phi")) is not None
        assert editor.row((0, "D")) is None
        family = editor.findChild(QComboBox, "fitFamily_1")
        family.setCurrentIndex(family.findData("core_shell_sphere"))
        assert page.session.model.components[1].family == "core_shell_sphere"
        assert editor.row((1, "t_shell")) is not None and editor.row((1, "core_contrast")) is not None
        assert editor.findChild(QComboBox, "fitInterference_1").currentData() == "paracrystal"
    finally:
        page.dispose()


def test_after_a_fit_the_uncertainty_window_shows_the_matrix_a_profile_and_saves_the_report(tmp_path):
    page = _page()
    try:
        page.open_curve(_curve_file(tmp_path))
        page.show_uncertainty()
        assert "Fit first" in page.uncertainty_summary.text() and not page.profile_run.isEnabled()
        page.set_model(page.session.model.with_values({(0, "R"): 5.5, (0, "D"): 42.0}))
        page.run_fit("local")
        _wait(page)
        result = page.session.result
        count = len(result.covariance_paths)
        assert page.correlation_table.rowCount() == count == page.correlation_table.columnCount()
        assert page.correlation_table.item(0, 0).text() == "+1.00"
        assert page.uncertainty_values.rowCount() >= count
        assert "|r| > 0.9" in page.correlation_pairs.text() or "|r| = 0.9" in page.correlation_pairs.text()
        page.profile_parameter.setCurrentIndex(page.profile_parameter.findText("R_1", Qt.MatchStartsWith))
        page.profile_points.setValue(7)
        page.run_profile()
        end = time.monotonic() + 120
        while page.tasks.is_busy() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        QApplication.processEvents()
        assert "68 % interval" in page.profile_result.text() and page.profile_plot.curve_count() == 2
        saved = page.save_report_dialog(str(tmp_path / "report.html"))
        text = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert saved and text.startswith("<!doctype html>") and "R_1" in text and "χ²ᵣ" in text
        page.report_format.setCurrentIndex(page.report_format.findData("markdown"))
        assert page.report_view.toPlainText().startswith("# GIMaP fit report")
    finally:
        page.dispose()


def test_the_beam_tail_takes_the_curves_ranges_when_a_curve_opens_or_a_fit_starts(tmp_path):
    page = _page()
    try:
        page.open_curve(_curve_file(tmp_path))
        tail = page.session.model.globals
        data = page._data()
        assert tail["res_amplitude"].upper == pytest.approx(10 * np.max(data.intensity))
        assert tail["res_width"].lower == pytest.approx(0.3 * data.q.min())
        generic = FitModel((new_component("sphere", radius=5.5, distance=42.0),))
        page.set_model(generic)
        assert np.isinf(page.session.model.globals["res_amplitude"].upper)
        page.run_fit("local")
        _wait(page)
        assert page.session.result is not None
        assert page.session.model.globals["res_amplitude"].upper == pytest.approx(10 * np.max(data.intensity))
        assert any("beam tail's ranges were set" in line for line in page.log_view.toPlainText().splitlines())
    finally:
        page.dispose()
