"""Analyze's file list (wave 2b: analyze-13, analyze-14).

One file comes off the list (Remove from List in its right-click menu, or Delete), and everything that
counted it follows: the file on screen (or its neighbour, when it was the one removed), the Series map
without its frames, Batch Export with one file less, its results forgotten. Refused while the automatic
analysis keeps its frame and while a batch runs. Delete on the mask and region lists removes the selected
mask or region. The previous / next file buttons are theme-coloured chevrons with “2 / 3” beside them.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.presentation.bindings import file_list
from src.gimap.features.analyze.presentation.bindings.file_list import BATCH_KEEPS_LIST, POSITION_TIP
from src.gimap.features.analyze.presentation.bindings.results_state import RUN_KEEPS_FRAME
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_review_analyze_run_and_clear import _frames, _profiles
from tests.test_series_map import _wait


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


def _listed(tmp_path: Path, count: int, *, mode: str | None = None):
    paths = _frames(tmp_path, count)
    page = _page(_context(_profiles()))
    if mode:
        page.set_mode_choice(mode)
    page.add_paths([str(path) for path in paths])
    _done(page)
    return page, paths


def _names(page) -> list[str]:
    return [page.file_list.item(row).text() for row in range(page.file_list.count())]


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


# -- the view model ----------------------------------------------------------------------------------------


def test_the_view_model_takes_one_file_off_and_keeps_the_one_shown() -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model

    _app()
    model = create_analyze_view_model(_context())
    files = [Path(f"C:/data/f{index}.tif") for index in range(5)]
    model.state.files.extend(files)
    model.select(2)
    model.state.frame_index = 3
    model.state.analysis = SimpleNamespace(path=files[2])
    model.remember_frame_count(files[0], 7)
    assert model.remove_file(9) is None and model.remove_file(-1) is None
    assert model.remove_file(0) == files[0]  # above the one shown: it keeps its file, one row up
    assert model.current_path == files[2] and model.state.current_index == 1 and model.state.frame_index == 3
    assert model.state.analysis is not None and str(files[0]).casefold() not in model._frame_counts
    assert model.remove_file(3) == files[4]  # below: nothing else changes
    assert model.current_path == files[2] and model.state.files == files[1:4]
    assert model.remove_file(1) == files[2]  # the one shown: the file that took its row, at its first frame
    assert model.current_path == files[3] and model.state.frame_index == 0 and model.state.analysis is None
    assert model.remove_file(1) == files[3]  # the last row: the one before
    assert model.current_path == files[1]
    assert model.remove_file(0) == files[1] and model.state.current_index == -1 and model.current_path is None


# -- the page ------------------------------------------------------------------------------------------------


def test_removing_files_keeps_the_frame_shown_and_every_count(tmp_path: Path) -> None:
    page, paths = _listed(tmp_path, 4)
    removed, cleared = [], []
    page.fileRemoved.connect(removed.append)
    page.filesCleared.connect(lambda: cleared.append(True))
    page.file_list.setCurrentRow(1)
    _done(page)
    shown = page.view_model.state.analysis
    assert shown.path.name == "film_1.tif"

    assert page.remove_file(3)  # below the frame shown: nothing is analysed again
    _done(page)
    assert page.view_model.state.analysis is shown and page.file_list.currentRow() == 1
    assert _names(page) == ["film_0.tif", "film_1.tif", "film_2.tif"]
    assert page.status_text() == "Removed film_3.tif from the list" and page.status_level() == "ok"
    assert "Files listed: 3" in page.data_info_label.text()
    assert page.data_batch_button.text() == "Batch Export 3 Frames…"
    assert page.file_position_label.text() == "2 / 3"
    assert len(page.view_model.batch_requests()) == 3

    assert page.remove_file(0)  # above it: the same file stays shown, one row up
    _done(page)
    assert page.view_model.state.analysis is shown and page.file_list.currentRow() == 0
    assert page.file_position_label.text() == "1 / 2"
    assert page.previous_file_button.isEnabled() is False and page.next_file_button.isEnabled()

    assert page.remove_file()  # the selected one, i.e. the one shown: the file that took its row is shown
    assert page.file_chip.full_text() == "film_2.tif"
    _done(page)
    assert page.view_model.state.analysis.path.name == "film_2.tif" and _names(page) == ["film_2.tif"]
    assert page.file_list.currentRow() == 0 and page.data_batch_button.isHidden()
    assert page.previous_file_button.isHidden() and page.file_position_label.isHidden()

    assert page.remove_file(0) and cleared == [True]  # the last one: as Clear
    assert page.file_list.count() == 0 and page.view_model.state.analysis is None
    assert page.status_text() == "Removed film_2.tif from the list"
    assert [path.name for path in removed] == ["film_3.tif", "film_0.tif", "film_1.tif", "film_2.tif"]
    assert not page.remove_file(0)  # nothing listed
    _close(page)


def test_a_file_cannot_leave_the_list_while_a_run_or_a_batch_uses_it(tmp_path: Path, monkeypatch) -> None:
    page, _paths = _listed(tmp_path, 3)
    page.automatic_started("Automatic analysis …")
    assert not page.file_list.isEnabled()
    assert not page.remove_file(1) and page.file_list.count() == 3
    assert page.status_text() == RUN_KEEPS_FRAME and page.status_level() == "warning"
    menu = {action.text(): action for action in page._file_menu(1).actions()}
    assert not menu["Remove from List"].isEnabled() and menu["Copy Path"].isEnabled()  # and Show in Folder
    page.automatic_finished("ok", "done")
    monkeypatch.setattr(page, "batch_running", lambda: True)
    assert not page.remove_file(1) and page.status_text() == BATCH_KEEPS_LIST
    monkeypatch.setattr(page, "batch_running", lambda: False)
    assert page.remove_file(1) and page.file_list.count() == 2
    _close(page)


def test_the_menu_of_a_file_and_a_right_press_that_shows_nothing_new(tmp_path: Path, monkeypatch) -> None:
    page, paths = _listed(tmp_path, 3)
    page.resize(1400, 900)
    page.show()
    QApplication.setActiveWindow(page)
    _done(page)
    menu = page._file_menu(2)
    texts = [action.text() for action in menu.actions() if not action.isSeparator()]
    assert texts == ["film_2.tif", "Remove from List", "Show in Folder", "Copy Path"]
    assert not menu.actions()[0].isEnabled()  # the file's name, a title
    opened = []
    monkeypatch.setattr(file_list.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    next(action for action in menu.actions() if action.text() == "Show in Folder").trigger()
    assert Path(opened[0]) == paths[2].parent
    next(action for action in menu.actions() if action.text() == "Copy Path").trigger()
    assert QApplication.clipboard().text() == str(paths[2]) and "film_2.tif" in page.status_text()

    # A right press on another file does not show it (no analysis); a left click does.
    page.show_step("data")
    _app().processEvents()
    viewport = page.file_list.viewport()
    centre = page.file_list.visualRect(page.file_list.model().index(2, 0)).center()
    QTest.mousePress(viewport, Qt.RightButton, Qt.NoModifier, centre)
    QTest.mouseRelease(viewport, Qt.RightButton, Qt.NoModifier, centre)
    _done(page)
    assert page.file_list.currentRow() == 0 and page.view_model.state.analysis.path.name == "film_0.tif"
    page._file_menu_requested(QPoint(-50, -50))  # no file under the pointer: no menu (and no error)

    next(action for action in menu.actions() if action.text() == "Remove from List").trigger()
    _done(page)
    assert _names(page) == ["film_0.tif", "film_1.tif"] and page.view_model.state.analysis.path.name == "film_0.tif"

    # Delete on the list removes the file selected.
    page.file_list.setFocus()
    QTest.keyClick(page.file_list, Qt.Key_Delete)
    _done(page)
    assert _names(page) == ["film_1.tif"] and page.view_model.state.analysis.path.name == "film_1.tif"
    _close(page)


def test_delete_removes_the_selected_mask_and_the_selected_added_region(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import RECTANGLE, MaskShape

    page, _paths = _listed(tmp_path, 1, mode="giwaxs")
    page.resize(1400, 900)
    page.show()
    QApplication.setActiveWindow(page)
    page.show_step("mask")
    for corner in (10.0, 50.0):
        page.view_model.add_mask_shape(MaskShape(RECTANGLE, ((corner, corner), (corner + 20, corner + 20))))
    page._refresh_mask_list()
    page.run_analysis()
    _done(page)
    page.mask_list.setFocus()
    page.mask_list.setCurrentRow(-1)
    QTest.keyClick(page.mask_list, Qt.Key_Delete)  # nothing selected: nothing removed (the button takes the last)
    assert len(page.view_model.state.corrections.mask_shapes) == 2
    page.mask_list.setCurrentRow(0)
    QTest.keyClick(page.mask_list, Qt.Key_Delete)
    _done(page)
    shapes = page.view_model.state.corrections.mask_shapes
    assert len(shapes) == 1 and shapes[0].points[0] == (50.0, 50.0)

    page.show_step("cuts")
    page._add_region("in_plane")
    _done(page)
    assert len(page.view_model.state.giwaxs.regions) == 1
    page.region_list.setFocus()
    page.region_list.setCurrentRow(0)  # the full ring: a standard row, never removed
    QTest.keyClick(page.region_list, Qt.Key_Delete)
    _done(page)
    assert len(page.view_model.state.giwaxs.regions) == 1
    page.region_list.setCurrentRow(page.region_list.count() - 1)  # the region added
    QTest.keyClick(page.region_list, Qt.Key_Delete)
    _done(page)
    assert page.view_model.state.giwaxs.regions == ()
    _close(page)


def test_the_series_map_loses_the_rows_of_a_removed_file(tmp_path: Path) -> None:
    page, _paths = _listed(tmp_path, 4, mode="giwaxs")
    page.series_build_button.click()
    _wait(page, lambda: page._series_map is not None and not page.batch_running())
    assert page._series_map.rows == 4
    assert page.remove_file(2)
    _done(page)
    series = page._series_map
    assert series.rows == 3 and all("film_2" not in label for label in series.labels)
    assert [Path(path).name for path, _frame in series.refs] == ["film_0.tif", "film_1.tif", "film_3.tif"]
    assert "3 frames ×" in page.series_info_label.text()
    assert page.remove_file(1)
    _done(page)
    assert page._series_map.rows == 2
    assert page.remove_file(1)  # one row left: no map
    _done(page)
    assert page._series_map is None and not page.series_empty.isHidden() and page.series_map_view.isHidden()
    _close(page)


def test_frames_summed_across_files_leave_the_map_of_the_earlier_list(tmp_path: Path) -> None:
    page, _paths = _listed(tmp_path, 3)
    page.view_model.set_sum_count(2)
    page._series_map = SimpleNamespace(rows=2)  # a map of two groups of two frames
    page._series_rows = []
    assert page.remove_file(2)
    _done(page)
    assert "Map of the earlier list (2 frames) — Build Map again without the removed file" in page.series_info_label.text()
    _close(page)


# -- ‹ 2 / 3 › ---------------------------------------------------------------------------------------------


def _icon_colour(button) -> QColor:
    image = button.icon().pixmap(32, 32).toImage()
    opaque = [image.pixelColor(x, y) for x in range(32) for y in range(32) if image.pixelColor(x, y).alpha() == 255]
    assert opaque, "the chevron is drawn"
    return opaque[0]


def test_the_file_stepper_has_chevrons_in_the_theme_colour_and_a_position(tmp_path: Path) -> None:
    from src.gimap.app.presentation.theme import apply_theme, theme_manager

    page, _paths = _listed(tmp_path, 3)
    try:
        for mode in ("light", "dark"):
            apply_theme(mode)
            for button in (page.previous_file_button, page.next_file_button):
                assert button.toolButtonStyle() == Qt.ToolButtonIconOnly and not button.icon().isNull()
                assert button.minimumWidth() >= 24 and button.minimumHeight() >= 24
                assert _icon_colour(button).name() == theme_manager().color("text").name()
    finally:
        apply_theme("light")
    page.file_list.setCurrentRow(1)
    _done(page)
    page.command_bar._apply(False)
    label = page.file_position_label
    assert label.text() == "2 / 3" and not label.isHidden() and label.property("gimapRole") == "muted"
    assert label.toolTip() == POSITION_TIP.format(n=2, total=3) and "Page Up" in label.toolTip()
    page.command_bar._apply(True)  # a narrow bar leaves the stepper out …
    assert label.isHidden() and page.previous_file_button.isHidden()
    page.command_bar._apply(False)  # … and shows it again
    assert not label.isHidden() and not page.next_file_button.isHidden()
    page.clear_files()
    assert label.isHidden() and page.previous_file_button.isHidden() and page.next_file_button.isHidden()
    _close(page)
