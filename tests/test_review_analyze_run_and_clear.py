"""Review fixes in Analyze: the automatic analysis keeps the frame it works on, Clear forgets everything of
the cleared files (results, a running analysis, the beam-centre chip, the curves), αi from a set-up is not
called "from your last session", files listed by folder watching mark an earlier Series map, and the
region editor is as tall as the page it shows."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.analyze.application import settings_record
from src.gimap.features.analyze.presentation.bindings.results_state import (
    RESULTS_ELSEWHERE,
    RESULTS_INTRO,
    RUN_KEEPS_FRAME,
    RUN_LISTED,
)
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, save_tiff
from tests.test_series_map import _ring_frame


def _profiles() -> InMemoryInstrumentProfileRepository:
    geometry = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2)
    return InMemoryInstrumentProfileRepository([InstrumentProfile("synthetic", geometry, None, SHAPE)])


def _frames(folder: Path, count: int, stem: str = "film") -> list[Path]:
    return [save_tiff(folder / f"{stem}_{index}.tif", _ring_frame(200.0, seed=index)) for index in range(count)]


def _file_controls(page) -> dict:
    return {
        "list": page.file_list.isEnabled(), "frame": page.frame_spin.isEnabled(), "open": page.open_files_button.isEnabled(),
        "folder": page.open_folder_action.isEnabled(), "clear": page.clear_button.isEnabled(),
        "previous": page.previous_file_button.isEnabled(), "next": page.next_file_button.isEnabled(),
    }


def test_the_automatic_analysis_keeps_the_frame_it_works_on(tmp_path: Path) -> None:
    paths = _frames(tmp_path, 3)
    page = _page(_context(_profiles()))
    page.add_paths([str(path) for path in paths])
    _done(page)
    assert page.file_list.currentRow() == 0

    page.automatic_started("Automatic analysis …")
    assert not any(_file_controls(page).values())  # nothing that shows another frame
    assert page.watch_button.isEnabled()  # watching only lists frames, and stays stoppable

    page.file_list.setCurrentRow(1)  # from code (the list itself is disabled)
    _done(page)
    assert page.file_list.currentRow() == 0 and page.view_model.state.current_index == 0
    assert page.view_model.state.analysis.path.name == "film_0.tif"
    assert page.status_text() == RUN_KEEPS_FRAME and page.status_level() == "warning"
    page._step_file(+1)  # PgDown
    page.frame_spin.setValue(2)
    _done(page)
    assert page.file_list.currentRow() == 0 and page.view_model.state.frame_index == 0

    more = _frames(tmp_path / "more", 2, stem="late")
    page.add_paths([str(path) for path in more])  # a drop, File ▸ Open, the Start page: listed, not shown
    _done(page)
    assert page.file_list.count() == 5 and page.file_list.currentRow() == 0
    assert page.status_text() == RUN_LISTED.format(n=2)
    page.add_paths([str(paths[2])])  # already listed: not shown either
    _done(page)
    assert page.file_list.currentRow() == 0 and page.status_text() == RUN_KEEPS_FRAME

    page.automatic_finished("ok", "Two rings in film_0.tif", path=paths[0])
    assert page.step_rail.state("results") == "ok" and page.status_level() == "ok"
    controls = _file_controls(page)
    assert controls.pop("previous") is False and all(controls.values())  # row 0: no previous file
    page.file_list.setCurrentRow(1)
    _done(page)
    assert page.view_model.state.analysis.path.name == "film_1.tif"
    page.dispose()
    page.close()


def test_results_of_another_file_are_never_shown_as_the_frame_on_screen(tmp_path: Path) -> None:
    paths = _frames(tmp_path, 2)
    page = _page(_context(_profiles()))
    page.add_paths([str(path) for path in paths])
    _done(page)
    page.file_list.setCurrentRow(1)
    _done(page)
    page.automatic_started("Automatic analysis …")
    page.automatic_finished("ok", "Rings in film_0.tif", path=paths[0])
    assert page.step_rail.state("results") == "pending"
    assert page.status_text() == RESULTS_ELSEWHERE.format(name="film_0.tif") and page.status_level() == "info"
    assert page.current_right() == "curves"
    page.file_list.setCurrentRow(0)
    _done(page)
    assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "Rings in film_0.tif"
    page.dispose()
    page.close()


def test_clear_stops_a_running_analysis_and_drops_its_results(tmp_path: Path) -> None:
    paths = _frames(tmp_path, 2)
    page = _page(_context(_profiles()))
    stopped = []
    page.set_automatic_analysis(run=lambda: None, find_geometry=lambda: None, stop=lambda: stopped.append(True))
    page.add_paths([str(paths[0])])
    _done(page)
    page.automatic_started("Automatic analysis …")
    page.clear_files()
    assert stopped == [True]
    assert _file_controls(page)["list"] and _file_controls(page)["open"]  # nothing is left to keep
    assert not page.run_pipeline_button.isEnabled()  # until the run has ended
    page.automatic_progress("Fitting form-factor models — R = 6.13 nm")  # its step in progress ends
    assert page.status_text() == "Ready"
    page.automatic_finished("ok", "GISAXS — R 6.13 nm", path=paths[0])  # the stopped run ends later
    assert page.run_pipeline_button.isEnabled() and page.stop_pipeline_button.isHidden()
    assert page.step_rail.state("results") == "pending" and page.step_intro["results"].text() == RESULTS_INTRO
    assert page.status_text() == "Ready" and page.results_for() is None
    page.add_paths([str(paths[0])])
    _done(page)
    assert page.step_rail.state("results") == "pending"  # opened again: nothing of the dropped run

    # A project opened while a run works (``apply_project_state`` clears first): its frames are shown.
    page.automatic_started("Automatic analysis …")
    record = settings_record(page.view_model.current_settings())
    page.apply_project_state({"files": [str(path) for path in paths], "current": 1, "settings": record})
    _done(page)
    assert stopped == [True, True]
    assert page.view_model.state.analysis.path.name == "film_1.tif"
    page.automatic_failed("No frame is open in Analyze.")
    assert page.step_rail.state("results") == "pending" and page.status_level() != "error"
    page.dispose()
    page.close()


def test_clear_leaves_nothing_of_the_last_frame(tmp_path: Path) -> None:
    paths = _frames(tmp_path, 1)
    page = _page(_context(_profiles()))
    page.add_paths([str(paths[0])])
    _done(page)
    page.automatic_started("Automatic analysis …")
    page.automatic_finished("ok", "Two rings in film_0.tif")
    assert page.current_right() == "results" and page.center_button.isEnabled()
    assert page.top_plot.curve_count() > 0
    page.clear_files()
    assert page.current_right() == "curves"
    assert page.step_rail.state("results") == "pending" and page.step_intro["results"].text() == RESULTS_INTRO
    assert page.center_button.text() == "Beam centre" and not page.center_button.isEnabled()
    assert page.top_plot.curve_count() == 0 and page.top_plot._source == [] and page.bottom_plot._source == []
    assert not page.top_plot.marks.wanted(page.top_plot.x_window)  # no I(χ) ring band
    assert page.top_plot.title_label.text() == ""
    page.top_plot.log_check.toggle()  # a Log (or theme) change draws nothing again
    assert page.top_plot.curve_count() == 0
    page.add_paths([str(paths[0])])  # the same file again: no results of before the Clear
    _done(page)
    assert page.step_rail.state("results") == "pending" and page.current_right() == "curves"
    page.dispose()
    page.close()


def test_alpha_i_from_a_set_up_is_not_called_the_last_sessions(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.analyze.presentation.bindings import incidence

    notices = []
    monkeypatch.setattr(incidence, "show_toast", lambda parent, text, **options: notices.append(text))
    context = _context(_profiles())
    first = _page(context)
    first.incidence_spin.setValue(0.3)  # remembered for the next session
    first.dispose()
    first.close()

    second = _page(context)
    assert second.view_model.state.incidence_deg == 0.3 and second._incidence_notice_pending
    settings = tmp_path / "setup.json"
    record = settings_record(replace(second.view_model.current_settings(), incidence_deg=0.25))
    settings.write_text(json.dumps(record), encoding="utf-8")
    assert second.load_settings(settings)  # no frame listed: nothing is analysed
    spin = second.incidence_spin
    assert spin.property("override") is True and second.incidence_reset_action.isEnabled()
    assert "0.25" in spin.toolTip()
    second.add_paths([str(_frames(tmp_path, 1)[0])])
    _done(second)
    assert notices == []  # αi 0.25° is the set-up's, not "from your last session"

    record = settings_record(replace(second.view_model.current_settings(), incidence_deg=None))
    settings.write_text(json.dumps(record), encoding="utf-8")
    second.clear_files()
    assert second.load_settings(settings)
    assert spin.property("override") is False and not second.incidence_reset_action.isEnabled()
    second.dispose()
    second.close()


def test_frames_listed_by_folder_watching_mark_the_earlier_series_map(tmp_path: Path) -> None:
    paths = _frames(tmp_path / "first", 2)
    page = _page(_context(_profiles()))
    page.add_paths([str(path) for path in paths])
    _done(page)
    page._series_map = SimpleNamespace(rows=2)  # a map built from the two frames
    page.series_info_label.setText("Series map: 2 frames")
    watched = _frames(tmp_path / "watched", 1, stem="insitu")
    page.start_watch(tmp_path / "watched")
    assert "Map of the earlier list (2 frames)" in page.series_info_label.text()

    page.series_info_label.setText("Series map: 2 frames")
    late = save_tiff(tmp_path / "watched" / "insitu_9.tif", _ring_frame(200.0, seed=9))

    def poll():
        page.view_model.state.files.append(late)
        return [late]

    page.view_model.poll_watch = poll
    page.poll_watch()
    _done(page)
    assert "Map of the earlier list (2 frames)" in page.series_info_label.text()
    assert page.file_list.count() == len(paths) + len(watched) + 1
    page.stop_watch()
    page.dispose()
    page.close()


def test_the_region_editor_is_as_tall_as_the_page_it_shows(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import GIWAXS

    page = _page(_context(_profiles()))
    page.resize(1600, 1000)
    page.show()
    page.set_mode_choice(GIWAXS)
    page.add_paths([str(_frames(tmp_path, 1)[0])])
    _done(page)
    page.show_step("cuts")
    _app().processEvents()
    editor = page.region_editor
    page.region_list.setCurrentRow(0)  # the full ring: one note
    _app().processEvents()
    note = editor.currentWidget()
    assert editor.height() <= note.sizeHint().height() + 4
    tallest = max(editor.widget(index).sizeHint().height() for index in range(editor.count()))
    assert editor.height() < tallest
    page.region_list.setCurrentRow(page.region_list.count() - 1)  # the ring of I(χ): q and a button
    _app().processEvents()
    assert editor.currentWidget() is not note and editor.height() >= editor.currentWidget().sizeHint().height() - 4
    page.dispose()
    page.close()
