"""Fixes from the check of wave 2b Analyze.

* A frame without a geometry after one with curves: the plots forget the old curves, so Log I or a theme change
  does not draw them again (``clear_curves`` kept them for a redraw).
* Clear hides the panel of a batch or map that ended (it was about the files cleared); a clean Build Map closes
  its own panel (the map, a toast and the status line say it), so the map and its plots get the height.
* “Change along the series” of a two-frame map says it needs three frames (no stages are looked for).
* After a switch of the language, the “n / N” position's tooltip is in the new language.
* The trace combo shrinks (its text shortened) so the frame and trace plots stand side by side in a narrower panel.
* Messages of the domain shown in the status line (a region or a mask that is not valid) are translated.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.app.presentation.theme import apply_theme
from src.gimap.features.analyze.presentation.texts import REGION_Q_EMPTY, message_text
from src.gimap.features.analyze.presentation.views.series_view import CHANGE_EMPTY, CHANGE_FEW, ShrinkingCombo
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_assistant_calibration import save_tiff
from tests.test_review_analyze_run_and_clear import _frames, _profiles
from tests.test_series_map import _wait


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


def _listed(tmp_path: Path, count: int):
    paths = _frames(tmp_path, count)
    page = _page(_context(_profiles()))
    page.add_paths([str(path) for path in paths])
    _done(page)
    return page, paths


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _build_map(page) -> None:
    page.series_build_button.click()
    _wait(page, lambda: page._series_map is not None and not page.batch_running())


def test_a_frame_without_geometry_forgets_the_curves_of_the_frame_before(tmp_path: Path) -> None:
    page, _paths = _listed(tmp_path, 1)
    assert page.top_plot.has_curves()
    bare = save_tiff(tmp_path / "bare.tif", np.random.default_rng(1).poisson(5, (64, 64)).astype(np.int32))
    page.add_paths([str(bare)])
    _done(page)
    assert page.view_model.state.analysis.reduction is None
    assert not page.top_plot.has_curves() and not page.bottom_plot.has_curves()
    page.top_plot.log_check.toggle()  # a redraw of the curves it was given last
    assert not page.top_plot.has_curves()
    try:
        apply_theme("dark", 9)
        _app().processEvents()
        assert not page.top_plot.has_curves() and not page.bottom_plot.has_curves()
    finally:
        apply_theme("light", 9)  # the tests' theme (``conftest.py``)
    _close(page)


def test_clear_hides_the_panel_of_a_map_and_a_clean_build_map_closes_it(tmp_path: Path) -> None:
    page, _paths = _listed(tmp_path, 3)
    _build_map(page)
    assert page._series_map.rows == 3
    assert page.batch_panel.isHidden()  # the map says it: no “Series map done” panel left over its room
    page.batch_panel.show()  # e.g. a map that stopped, or a Batch Export that ended
    page.clear_files()
    assert page.batch_panel.isHidden()
    _close(page)


def test_change_along_a_two_frame_map_says_it_needs_three_frames(tmp_path: Path) -> None:
    page, _paths = _listed(tmp_path, 2)
    _build_map(page)
    page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("change"))
    assert page.series_trace_plot.empty_overlay.text == CHANGE_FEW
    assert not page.series_trace_plot.has_curves()
    page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("intensity"))
    assert page.series_trace_plot.empty_overlay.text == "" and page.series_trace_plot.has_curves()
    assert CHANGE_EMPTY != CHANGE_FEW
    _close(page)


def test_the_file_position_tip_follows_a_switch_of_the_language(tmp_path: Path, monkeypatch) -> None:
    template = "File {n} of {total} in the list — Page Up / Page Down show the previous / next one"
    monkeypatch.setitem(i18n.ZH, template, "列表中的第 {n} 个文件，共 {total} 个 — Page Up / Page Down 显示上一个 / 下一个")
    page, _paths = _listed(tmp_path, 3)
    assert page.file_position_label.toolTip().startswith("File 1 of 3")
    apply_language("zh", [page])
    page.refresh_language()
    assert page.file_position_label.toolTip().startswith("列表中的第 1 个文件，共 3 个")
    apply_language("en", [page])
    page.refresh_language()
    assert page.file_position_label.toolTip().startswith("File 1 of 3")
    _close(page)


def test_the_trace_combo_shrinks_so_the_plots_stand_side_by_side_sooner() -> None:
    page = _page(_context())
    combo = page.series_trace_combo
    assert isinstance(combo, ShrinkingCombo)
    assert combo.minimumSizeHint().width() < combo.sizeHint().width()  # whole while there is room
    assert combo.view().minimumWidth() >= combo.view().sizeHintForColumn(0)  # its list shows every name whole
    combo.setCurrentIndex(combo.findData("change"))
    combo.resize(combo.minimumSizeHint().width(), combo.sizeHint().height())
    assert not combo.grab().isNull()  # painted with the text shortened
    plots = page.series_plots
    trace, profile = page.series_trace_plot, page.series_profile_plot
    assert plots.side_by_side_width() == trace.minimumSizeHint().width() + profile.minimumSizeHint().width() + 6
    assert trace.minimumSizeHint().width() < trace.sizeHint().width()
    _close(page)


def test_messages_of_the_domain_in_the_status_line_are_translated(monkeypatch) -> None:
    from src.gimap.features.analyze.domain.regions import CutRegion

    monkeypatch.setitem(i18n.ZH, REGION_Q_EMPTY, "区域 {name}：q 范围为空。")
    monkeypatch.setitem(i18n.ZH, "A polygon mask needs at least three vertices.", "多边形掩膜至少需要三个顶点。")
    with pytest.raises(ValueError) as caught:
        CutRegion("Ring 1", (1.0, 1.0), (0.0, 90.0), True)
    assert message_text(caught.value) == "Region 'Ring 1': the q range is empty."
    apply_language("zh")
    assert message_text(caught.value) == "区域 'Ring 1'：q 范围为空。"
    page = _page(_context())
    page._shape_drawn("polygon", [(0.0, 0.0), (1.0, 1.0)])  # two corners: not a polygon
    assert page.status_text() == "多边形掩膜至少需要三个顶点。" and page.status_level() == "warning"
    _close(page)
