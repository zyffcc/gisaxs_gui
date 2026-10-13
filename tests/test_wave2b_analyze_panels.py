"""Analyze's Series panel and Batch Export dialog (wave 2b: analyze-4 part 2, analyze-15, compare-6, compare-14).

The Series controls take two rows (what the map stacks; then Build Map, Export, Send to Compare, Batch
Export, wrapping when narrower), so a map does not widen the panel past ~380 px; the Stages combo shows
“Auto (n)” whole. The trace plot's legend sits bottom right. “Change along the series” has a short title,
the share of the change in its y label and as a sentence in the tooltips. Batch Export opens as tall as its
options (at most 90 % of the screen), the curve list as tall as its rows.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontInfo
from PyQt5.QtWidgets import QApplication, QComboBox, QDialog

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import BatchChoices, SeriesCorrection, stages_of
from src.gimap.features.analyze.domain import series_map
from src.gimap.features.analyze.presentation.views.batch_export_view import CURVE_LIST_HEIGHT, SCREEN_SHARE
from src.gimap.features.analyze.presentation.views.series_view import (
    CHANGE_TITLE,
    SERIES_CONTROLS_WIDTH,
    TRACE_LEGEND_OFFSET,
)
from tests.test_analyze_workspace import _app, _context, _page


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


LONG_CURVES = ("Out-of-plane sector I(2θ)", "Horizontal cut I(qy)", "Region 12 I(q)", "In-plane sector I(q)")


def test_the_series_controls_wrap_and_never_need_more_than_380_px() -> None:
    page = _page(_context())
    controls = page.series_controls
    for title in LONG_CURVES:
        page.series_curve_combo.addItem(title, title)
    for button in (page.series_export_button, page.series_compare_button):
        button.show()
    controls.show()
    actions = (page.series_build_button, page.series_export_button, page.series_compare_button,
               page.series_batch_button)
    first_row = (page.series_curve_combo, page.series_step_spin)
    # The width is that of the first row: the actions add none (they wrap). With the application's font
    # (Segoe UI, 9 pt) that is about 365 px; the test fonts of an offscreen run differ, so it is measured.
    one_row = sum(widget.sizeHint().width() for widget in (*first_row, *actions))
    least = controls.minimumSizeHint().width()
    assert least <= sum(widget.minimumSizeHint().width() for widget in first_row) + 6 * 2 + 80  # + the caption
    assert least < 0.7 * one_row
    if "Segoe UI" in QFontInfo(QApplication.font()).family():
        assert least <= SERIES_CONTROLS_WIDTH
    # The frame and trace plots side by side when there is room, else one above the other: with a map shown,
    # the controls (not the plots, nor a running batch's panel) set the least width of the panel.
    for widget in (page.series_map_view, page.series_plots, page.batch_panel):
        widget.show()
    panel = page.splitter.widget(2)
    plots = page.series_plots
    assert plots.minimumSizeHint().width() == max(page.series_profile_plot.minimumSizeHint().width(),
                                                  page.series_trace_plot.minimumSizeHint().width())
    assert panel.minimumSizeHint().width() <= least + 30
    # Laid out at its least width (the right panel at its minimum): every control whole, inside the panel.
    page.resize(1600, 900)
    page.show()
    page.show_right("series")
    page.even_split.by_hand = True  # the sizes below stay
    sizes = page.splitter.sizes()
    page.splitter.setSizes([sizes[0], sum(sizes[1:]), 0])
    _app().processEvents()
    width = controls.width()
    assert width <= panel.width() and plots.stacked()
    assert page.series_trace_plot.y() > page.series_profile_plot.y()
    for widget in (*first_row, *actions):
        geometry = widget.geometry()
        assert geometry.left() >= 0 and geometry.right() < width, widget.objectName()
        assert geometry.width() >= widget.minimumSizeHint().width(), widget.objectName()
    assert page.series_build_button.y() > page.series_curve_combo.y()  # the actions on a row of their own
    page.splitter.setSizes([sizes[0], 300, sum(sizes[1:]) - 300])  # a wide panel: the plots side by side
    _app().processEvents()
    assert not plots.stacked() and page.series_trace_plot.y() == page.series_profile_plot.y()
    # The combo may be narrower than a name: its list shows every name whole.
    page._refresh_series_curves(None)  # (no frame: an empty list)
    assert page.series_curve_combo.view().minimumWidth() == 0
    assert page.series_stages_combo.sizeAdjustPolicy() == QComboBox.AdjustToContents
    page.series_stages_combo.addItem("Auto (4)", None)
    text = page.series_stages_combo.fontMetrics().horizontalAdvance("Auto (4)")
    assert page.series_stages_combo.sizeHint().width() > text + 10  # “Auto (4” was cut before
    page.dispose()
    page.close()


def _stages_map():
    q = np.linspace(0.3, 2.0, 120)
    curves = []
    for index in range(12):
        grow = 1.0 + 4.0 * min(index, 6) / 6.0
        curves.append((q, 50.0 + 400.0 * grow * np.exp(-((q - 1.1) / 0.05) ** 2) + 30.0 * np.exp(-q)))
    series = series_map(curves, [f"run_{index:03d}.tif" for index in range(12)], x_label="q (Å⁻¹)", curve="radial",
                        refs=[(Path(f"run_{index:03d}.tif"), 0) for index in range(12)])
    return series, stages_of(series)


def test_change_along_the_series_has_a_short_title_and_its_share_in_the_y_label() -> None:
    page = _page(_context())
    plot = page.series_trace_plot
    assert tuple(plot.legend.opts["offset"]) == TRACE_LEGEND_OFFSET  # bottom right, away from the rising curves
    series, stages = _stages_map()
    page._series_map, page._series_q = series, (1.0, 1.2)
    page._series_stages = stages
    page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("change"))
    page._show_stages()
    share = 100.0 * float(stages.explained[0])
    assert plot.title_label.text() == CHANGE_TITLE == "Change along the series"
    assert plot.plot.getAxis("left").labelText == f"component 1 ({share:.0f} %)"
    sentence = f"Main component: {share:.0f} % of the change"
    assert plot.title_label.toolTip() == sentence
    index = page.series_trace_combo.findData("change")
    assert page.series_trace_combo.itemData(index, Qt.ToolTipRole) == sentence
    assert plot.has_curves() and tuple(plot.legend.opts["offset"]) == TRACE_LEGEND_OFFSET
    page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("intensity"))
    assert plot.title_label.toolTip() == plot.title_label.text()  # the other traces: their own title
    page._clear_stages()
    assert page.series_trace_combo.itemData(index, Qt.ToolTipRole) is None
    page.dispose()
    page.close()


def _dialog(curves):
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    _app()
    return BatchExportDialog(
        frames=40, files=40, stem="run", settings_text="GIWAXS · profile “P08”", choices=BatchChoices(),
        destination=Path("."), subfolder=True, series=SeriesCorrection(), curve_options=curves,
    )


def test_batch_export_opens_as_tall_as_its_options_and_the_curve_list_as_its_rows() -> None:
    few = _dialog([("radial", "I(q)"), ("azimuthal", "I(χ)")])
    listing = few.curve_list
    rows = sum(listing.sizeHintForRow(row) for row in range(listing.count()))
    assert listing.height() == min(CURVE_LIST_HEIGHT, rows + 2 * listing.frameWidth() + 2) < CURVE_LIST_HEIGHT
    many = _dialog([(f"region{index}", f"Region {index} I(q)") for index in range(1, 15)])
    assert many.curve_list.height() == CURVE_LIST_HEIGHT

    few.show()  # its first size: as tall as the options, at most 90 % of the screen's free height
    _app().processEvents()
    room = int(SCREEN_SHARE * (few.screen() or QApplication.primaryScreen()).availableGeometry().height())
    assert few.height() <= max(room, few.minimumHeight()) and few.minimumHeight() <= max(room, 1)
    tall = few.fit_to_screen(room=4000)  # a tall screen: every group in view, no scroll bar
    _app().processEvents()
    assert tall == few.height() and few.options_scroll.verticalScrollBar().maximum() == 0
    assert few.width() >= 760
    short = few.fit_to_screen(room=700)
    assert short == 700 == few.height() < tall
    for dialog in (few, many):
        dialog.done(QDialog.Rejected)
