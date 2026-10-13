"""Release fixes of Analyze's Series tab (R14, R19, R29).

The card shown until there is a map never says “One frame” while several frames are listed (it says there is
no map yet, and that a Batch Export fills it). A map is named after the curve it holds — on screen, in the
finished line and in the saved figure — whatever the curve combo shows later: another choice not built yet,
another language, a frame of another kind.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.presentation.views.series_view import (
    SERIES_EMPTY_TEXT,
    SERIES_EMPTY_TITLE,
    SERIES_WAITING_TEXT,
    SERIES_WAITING_TITLE,
)
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_release_analyze_i18n import ZH
from tests.test_review_analyze_run_and_clear import _frames, _profiles
from tests.test_series_map import _wait


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _card(page) -> tuple[str, str]:
    return page.series_empty.title_label.text(), page.series_empty.message_label.text()


def _built(tmp_path: Path, count: int = 3):
    """A page with ``count`` synthetic GIWAXS frames listed and the map of their I(q) built."""
    paths = _frames(tmp_path, count)
    page = _page(_context(_profiles()))
    page.set_mode_choice("giwaxs")
    page.add_paths([str(path) for path in paths])
    _done(page)
    assert page.series_curve_combo.currentData() == "radial"
    page.build_series_map()
    _wait(page, lambda: page._series_map is not None and page._series_map.rows == count and not page.batch_running())
    _done(page)
    return page, paths


# -- the card (R14, R19) --------------------------------------------------------------------------------------------


def test_the_card_does_not_say_one_frame_while_several_are_listed(tmp_path: Path) -> None:
    paths = _frames(tmp_path, 3)
    page = _page(_context(_profiles()))
    try:
        assert _card(page) == (SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT) and SERIES_EMPTY_TITLE == "One frame"
        page.add_paths([str(paths[0])])
        _done(page)
        assert _card(page) == (SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT)  # one frame: it says so, and what to do
        assert page.series_controls.isHidden() and not page.series_empty.isHidden()

        page.add_paths([str(path) for path in paths[1:]])
        _done(page)
        assert "3 files listed" in page.series_info_label.text()
        assert _card(page) == (SERIES_WAITING_TITLE, SERIES_WAITING_TEXT) and SERIES_WAITING_TITLE == "No map yet"
        assert "One frame" not in " ".join(_card(page)) and "Batch Export" in SERIES_WAITING_TEXT
        assert not page.series_empty.isHidden() and not page.series_controls.isHidden()

        page.build_series_map()
        _wait(page, lambda: page._series_map is not None and not page.batch_running())
        assert page.series_empty.isHidden() and not page.series_map_view.isHidden()

        page.remove_file(2)
        page.remove_file(1)  # one file left: no map, and the card says “One frame” again
        _done(page)
        assert page._series_map is None and not page.series_empty.isHidden()
        assert _card(page) == (SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT)
    finally:
        _close(page)


def test_clear_puts_the_first_explanation_back(tmp_path: Path) -> None:
    """Clear with several files listed: the tab is as before the first file — the card explains what a series
    needs (not “No map yet … a Batch Export fills it”), and the controls and curves of the files cleared go."""
    paths = _frames(tmp_path, 3)
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(path) for path in paths])
        _done(page)
        assert _card(page)[0] == SERIES_WAITING_TITLE and not page.series_controls.isHidden()
        assert page.series_curve_combo.count() > 0
        page.clear_files()
        assert _card(page) == (SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT) and not page.series_empty.isHidden()
        assert page.series_controls.isHidden() and page.series_curve_combo.count() == 0
        assert page.series_info_label.text() == "" and not page.series_build_button.isEnabled()

        page.add_paths([str(path) for path in paths])  # … and with a map: the same after Clear
        _done(page)
        page.build_series_map()
        _wait(page, lambda: page._series_map is not None and not page.batch_running())
        _done(page)
        page.clear_files()
        assert page._series_map is None and page.series_map_view.isHidden()
        assert _card(page) == (SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT) and page.series_controls.isHidden()
    finally:
        _close(page)


def test_the_card_follows_the_language(tmp_path: Path, monkeypatch) -> None:
    for english, chinese in ZH.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    paths = _frames(tmp_path, 3)
    page = _page(_context(_profiles()))
    try:
        apply_language("zh", [page])
        page.add_paths([str(path) for path in paths])
        _done(page)
        assert _card(page) == (ZH[SERIES_WAITING_TITLE], ZH[SERIES_WAITING_TEXT])
        assert "只有一帧" not in " ".join(_card(page))
        apply_language("en", [page])
        page.refresh_language()
        assert _card(page) == (SERIES_WAITING_TITLE, SERIES_WAITING_TEXT)
    finally:
        _close(page)


def test_a_batch_export_starting_keeps_the_card_true(tmp_path: Path) -> None:
    """While a batch fills the map and no row has arrived, the card says there is no map yet."""
    from src.gimap.features.analyze.application import BatchChoices

    paths = _frames(tmp_path / "frames", 3)
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(path) for path in paths])
        _done(page)
        page.run_batch(tmp_path / "out", BatchChoices(tables=True, per_frame=False, speed="gentle"), stem="film")
        assert page.batch_running() and page._series_map is None
        assert _card(page)[0] == "No map yet" and not page.series_empty.isHidden()
        _wait(page, lambda: not page.batch_running())
        _done(page)
        assert page._series_map is not None and page.series_empty.isHidden()
    finally:
        _close(page)


# -- the map is named after the curve it holds (R29) -------------------------------------------------------------------


def test_the_map_keeps_the_name_of_its_curve_when_the_combo_changes(tmp_path: Path) -> None:
    page, _paths = _built(tmp_path, 3)
    try:
        combo, title = page.series_curve_combo, page.series_map_view.title_label
        built = combo.currentText()
        assert title.text() == f"{built} — 3 frames" and page._series_curve_name() == built
        assert page.status_text() == f"Series map: 3 frames of {built}"
        other = next(index for index in range(combo.count()) if combo.itemData(index) != "radial")
        combo.setCurrentIndex(other)  # another curve chosen, not built
        assert combo.currentText() != built and page._series_map.curve == "radial"

        page.remove_file(2)  # the map is drawn again without that file's row: still the curve it holds
        _done(page)
        assert page._series_map.rows == 2 and title.text() == f"{built} — 2 frames"
        assert page.series_map_view.display_state()["title"] == f"{built} — 2 frames"  # what Save Series Map writes

        apply_language("zh", [page])
        page.refresh_language()
        assert title.text() == f"{built} — 2 帧"  # (“I(q)” is a symbol: the same in Chinese)
        assert page.series_map_view.display_state()["title"] == f"{built} — 2 帧"
        apply_language("en", [page])
        page.refresh_language()
        assert title.text() == f"{built} — 2 frames" and page._series_map.curve == "radial"
    finally:
        _close(page)


def test_a_map_follows_the_language_while_the_frame_chosen_cannot_be_read(tmp_path: Path) -> None:
    """No frame is held then (its picture is off the screen), but the map built before is still shown."""
    page, _paths = _built(tmp_path, 3)
    try:
        broken = tmp_path / "broken_frame.cbf"
        broken.write_bytes(b"not a detector")
        page.add_paths([str(broken)])
        _done(page)
        assert page.view_model.state.analysis is None and page._series_map is not None
        title = page.series_map_view.title_label
        assert not page.series_map_view.isHidden() and title.text() == "I(q) — 3 frames"
        apply_language("zh", [page])
        page.refresh_language()
        row = page._series_row + 1
        assert title.text() == "I(q) — 3 帧" and page.series_profile_plot.title_label.text().startswith(f"第 {row} 帧")
        apply_language("en", [page])
        page.refresh_language()
        assert title.text() == "I(q) — 3 frames"
        assert page.series_profile_plot.title_label.text().startswith(f"Frame {row}:")
    finally:
        _close(page)


def test_a_translated_curve_name_follows_the_language_and_stays_the_built_one(tmp_path: Path) -> None:
    """A map of a curve whose name has Chinese (the in-plane sector) while the combo shows another curve."""
    paths = _frames(tmp_path, 3)
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        _done(page)
        combo = page.series_curve_combo
        sector = combo.findText("In-plane sector I(q)")
        assert sector >= 0
        combo.setCurrentIndex(sector)
        page.build_series_map()
        _wait(page, lambda: page._series_map is not None and not page.batch_running())
        _done(page)
        key = page._series_map.curve
        assert page.series_map_view.title_label.text() == "In-plane sector I(q) — 3 frames"
        combo.setCurrentIndex(combo.findData("radial"))
        apply_language("zh", [page])
        page.refresh_language()
        assert page._series_map.curve == key
        assert page.series_map_view.title_label.text() == "面内扇区 I(q) — 3 帧"
        apply_language("en", [page])
        page.refresh_language()
        assert page.series_map_view.title_label.text() == "In-plane sector I(q) — 3 frames"
    finally:
        _close(page)
