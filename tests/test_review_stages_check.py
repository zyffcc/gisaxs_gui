"""Checks after the stages group's cross-area work: the In-situ list forgets the stages of curves no longer
listed (no stage colour, “· odd” or Leave-out left from another folder), a running series keeps the stages
it started with, and Compare gives the reason a comparison failed in the interface language."""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, translate
from tests.test_compare import Q, _close, _idle, _map, _shown_page, _wait


def _stages_found(series, limit: float = 60.0) -> None:
    end = time.monotonic() + limit
    while series._series_stages is None and time.monotonic() < end:
        series.stage_tasks.wait(0.1)
        QApplication.processEvents()
    assert series._series_stages is not None


def test_another_folder_forgets_the_stages_of_the_one_before(tmp_path) -> None:
    from tests.test_fit_series import _pages, _series
    from tests.test_review_stages_cross import _odd_series

    first = tmp_path / "first"
    _odd_series(first)  # 12 curves, the 6th odd
    (tmp_path / "second").mkdir()
    second = _series(tmp_path / "second", radii=(5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6))
    single, series = _pages()
    try:
        series.open_series(first)
        _stages_found(series)
        assert series.is_odd_frame(5) and not series.skip_odd_check.isHidden()
        series.skip_odd_check.setChecked(True)
        assert series._without_odd(series._listed()) == [index for index in range(12) if index != 5]
        series.open_series(second)  # before its own stages are found: nothing of the first folder's
        listed = series._listed()
        assert series._series_stages is None and series.stage_of_frame(5) is None
        assert series.skip_odd_check.isHidden() and series._without_odd(listed) == listed  # Start fits all 7
        for position in range(series.frame_list.count()):
            item = series.frame_list.item(position)
            assert item.toolTip() == "" and "odd" not in item.text() and item.data(Qt.ForegroundRole) is None
    finally:
        series.dispose()
        single.dispose()


def test_a_search_while_the_series_runs_keeps_its_stages(tmp_path) -> None:
    from tests.test_fit_series import _pages
    from tests.test_review_stages_cross import _odd_series

    folder = tmp_path / "gimap_analysis"
    _odd_series(folder)
    single, series = _pages()
    try:
        series.open_series(folder)
        _stages_found(series)
        found = series._series_stages
        series.running = True  # a timer that fires during a run (“afresh at each stage” reads the stages)
        series._find_stages()
        assert series._series_stages is found and series.stage_of_frame(11) is not None
        assert series._stages_key is None  # found again once the run is over
        series.running = False
        series._schedule_stages()
        assert series._stages_timer.isActive()
    finally:
        series.dispose()
        single.dispose()


def test_the_reason_a_comparison_failed_is_in_the_interface_language() -> None:
    page = _shown_page(1400, 900)
    try:
        page.add_map(_map(1), "a")
        page.add_map(_map(2, x=Q + 10.0), "b")  # no q in common
        _wait(page, lambda: _idle(page) and page.failure_reason() != "")
        reason = page.failure_reason()
        assert reason == "The series share no common q range."
        assert page.status_label.text() == f"Could not compare: {reason}"
        apply_language("zh", [page])
        page.refresh_language()
        shown = translate(reason, "zh") or reason  # the Chinese once the table has it (zh_entries of this round)
        line = translate("Could not compare: {reason}", "zh").format(reason=shown)
        assert page.status_label.text() == line and page.summary_label.text() == line
        assert page.empty_state.message_label.text().startswith(shown + "\n")
        apply_language(DEFAULT_LANGUAGE, [page])
        page.refresh_language()
        assert page.status_label.text() == f"Could not compare: {reason}"
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(page)
