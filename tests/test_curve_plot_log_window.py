"""Log x is a display choice: the draggable selection stays in data units."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.components import CurvePlot


@pytest.fixture
def plot():
    host = QWidget()
    curve = CurvePlot("", host, log_x=False)
    curve.set_labels("q (Å⁻¹)", "I (counts)")
    x = np.geomspace(0.01, 100.0, 100)
    curve.set_curves([("sample", x, np.full_like(x, 100.0))])
    yield curve
    curve.dispose()
    host.close()
    host.deleteLater()


def test_toggling_log_x_preserves_the_window_and_drag_emits_data_units(plot) -> None:
    selections = []
    plot.windowChanged.connect(lambda low, high: selections.append((low, high)))
    plot.show_window(0.1, 10.0)
    assert plot.x_window.getRegion() == pytest.approx((0.1, 10.0))

    plot.log_x_check.setChecked(True)
    assert plot.x_window.getRegion() == pytest.approx((-1.0, 1.0))
    assert selections == []

    plot.x_window.setRegion(tuple(np.log10((0.2, 5.0))))
    assert len(selections) == 1
    assert selections[0] == pytest.approx((0.2, 5.0))
    plot.log_x_check.setChecked(False)
    assert plot.x_window.getRegion() == pytest.approx((0.2, 5.0))
    assert len(selections) == 1

    plot.log_x_check.setChecked(True)
    plot.show_window(0.4, 4.0)
    assert plot.x_window.getRegion() == pytest.approx(tuple(np.log10((0.4, 4.0))))
    assert len(selections) == 1  # rendering a selection is not a new scientific edit


def test_log_x_respects_hidden_marks_and_hidden_windows(plot) -> None:
    plot.show_window(0.1, 10.0)
    plot.marks.set_visible("window", False)
    for checked in (True, False, True):
        plot.log_x_check.setChecked(checked)
        assert not plot.x_window.isVisible()
        assert plot.marks.wanted(plot.x_window)
        assert not plot.marks.is_visible("window")

    plot.marks.set_visible("window", True)
    assert plot.x_window.isVisible()
    plot.hide_window()
    for checked in (False, True):
        plot.log_x_check.setChecked(checked)
        assert not plot.x_window.isVisible()
        assert not plot.marks.wanted(plot.x_window)


@pytest.mark.parametrize("interval", [(0.0, 1.0), (-1.0, 1.0), (-2.0, -1.0)])
def test_an_unrepresentable_window_is_restored_unchanged_on_linear_x(plot, interval) -> None:
    selections = []
    plot.windowChanged.connect(lambda *bounds: selections.append(bounds))
    plot.show_window(*interval)
    plot.log_x_check.setChecked(True)
    assert not plot.x_window.isVisible()
    assert np.isfinite(plot.x_window.getRegion()).all()
    plot.marks.set_visible("window", False)
    plot.marks.set_visible("window", True)
    assert not plot.x_window.isVisible()  # Marks cannot show an invalid log interval

    plot.log_x_check.setChecked(False)
    assert plot.x_window.isVisible()
    assert plot.x_window.getRegion() == pytest.approx(interval)
    assert selections == []


@pytest.mark.parametrize("interval", [(np.nan, 1.0), (1.0, np.inf)])
def test_nonfinite_window_endpoints_never_reach_the_graphics_item(plot, interval) -> None:
    plot.show_window(*interval)
    assert not plot.x_window.isVisible()
    plot.log_x_check.setChecked(True)
    assert not plot.x_window.isVisible()
    assert np.isfinite(plot.x_window.getRegion()).all()


@pytest.mark.parametrize("region", [(400.0, 401.0), (-400.0, -399.0)])
def test_a_drag_outside_representable_data_restores_the_last_window(plot, region) -> None:
    selections = []
    plot.windowChanged.connect(lambda *bounds: selections.append(bounds))
    plot.log_x_check.setChecked(True)
    plot.show_window(0.1, 10.0)
    plot.x_window.setRegion(region)
    assert plot.x_window.getRegion() == pytest.approx((-1.0, 1.0))
    assert selections == []
