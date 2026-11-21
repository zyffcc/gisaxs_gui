"""The online mode of a watched folder and the timeline of the Series map, on a small in-situ series made of the public
GALAXI frame (``tests/insitu_frames.py``): frames written into a folder over time are reduced and appended to the map
without Build Map or an export, recorded in the results database, a file that cannot be read is logged, Pause stops
the looks, a settle time holds a new file back, and the timeline finds the jump and opens the frame clicked."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtTest import QTest

from src.gimap.app.presentation.i18n import ZH, apply_language
from src.gimap.features.analyze.presentation.folder_watch import FolderWatch, WatchScan
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.series_store import SeriesStore, store_path
from tests.insitu_frames import GALAXI, crop_profile, write_frame
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_live_following import _close
from tests.test_series_cut import _build

pytestmark = pytest.mark.skipif(not GALAXI.is_file(), reason="public test frames (tests/data/external) are not available")


@pytest.fixture(autouse=True)
def _english():
    apply_language("en")
    yield
    apply_language("en")


def _watching(tmp_path: Path, frames: int = 1):
    folder = tmp_path / "run"
    for index in range(frames):
        write_frame(folder, index, jump_at=6)
    page = _page(_context(InMemoryInstrumentProfileRepository([crop_profile()])))
    page.set_mode_choice("gisaxs")
    page.add_paths([str(folder)])
    _done(page)
    page.auto_export_check.setChecked(False)
    page.start_watch(folder)
    page._watch_timer.stop()  # the test chooses when the watch looks
    return page, folder


def _look(page, times: int = 3) -> None:
    for _ in range(times):
        page.poll_watch()
        _done(page)
    QTest.qWait(1200)  # the map is redrawn once a second
    _done(page)


def test_new_frames_grow_the_map_without_build_or_export_and_go_into_the_database(tmp_path):
    page, folder = _watching(tmp_path)
    try:
        assert page.watch_map_action.isChecked() and page._series_map is None
        page.watch_record_action.setChecked(True)
        for index in range(1, 10):
            write_frame(folder, index, jump_at=6)
        _look(page)
        series = page._series_map
        assert series is not None and series.rows == 9  # the frames written while watching, none twice
        assert not list((folder / "gimap_analysis").glob("*.csv"))  # nothing exported
        with SeriesStore(store_path(folder / "gimap_analysis"), read_only=True) as store:
            (name,) = store.series_names()
            curves = store.curves(name, "horizontal")
        assert len(curves) == 9 and curves[0].file.endswith("galaxi_00001.tif")
        np.testing.assert_allclose(curves[-1].intensity, page._series_rows[-1][1])
        # The timeline of the map: noise ≈ 1, the jump where the side peak moves at once (frame 7, row 5).
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("neighbour"))
        timeline = page._series_timeline()
        assert timeline.jumps == (5,) and float(np.nanmedian(timeline.neighbour)) < 2.0
        assert page.series_trace_plot.title_label.text() == "χ² to the frame before"
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("reference"))
        page.series_reference_spin.setValue(9)
        assert page._series_timeline().reference_row == 7  # frame 9 of the list is row 7 of the map (frames 2–10)
    finally:
        page.stop_watch()
        _close(page)


def test_a_click_on_the_timeline_shows_the_frame_and_the_table_names_the_jump(tmp_path):
    folder = tmp_path / "run"
    paths = [write_frame(folder, index, jump_at=6) for index in range(12)]
    page = _page(_context(InMemoryInstrumentProfileRepository([crop_profile()])))
    try:
        page.set_mode_choice("gisaxs")
        page.add_paths([str(path) for path in paths])
        _done(page)
        _build(page, 12)
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("neighbour"))
        page._series_trace_clicked(7.2, 1.0)  # the frame axis: frame 7 is row 6, the jump
        _done(page)
        assert page._series_row == 6
        assert page.view_model.current_path.name == "galaxi_00006.tif"
        written = page.export_series_timeline(tmp_path / "timeline.csv")
        lines = [line.split(",") for line in written.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
        assert lines[0][:5] == ["frame", "file", "chi2_to_previous", "chi2_to_reference", "jump"]
        assert [row[0] for row in lines[1:] if row[4] == "True"] == ["7"]
        saved = page.save_results_database(tmp_path / "db" / "gimap_series.sqlite")
        with SeriesStore(saved, read_only=True) as store:
            assert store.curve_keys(store.series_names()[0])[0][:2] == ("horizontal", 12)
    finally:
        _close(page)


def test_a_file_that_cannot_be_read_is_logged_and_pause_stops_the_looks(tmp_path):
    page, folder = _watching(tmp_path, frames=2)
    try:
        page.watch_record_action.setChecked(True)
        broken = folder / "galaxi_00002.tif"
        broken.write_bytes(b"not a tiff at all" * 100)
        write_frame(folder, 3)
        _look(page)
        skipped = page.skipped_files()
        assert any(Path(file).name == broken.name or file == broken.name for _time, file, _what, _why in skipped)
        dialog = page.show_skipped_files()
        assert broken.name in dialog.text.toPlainText()
        dialog.close()
        page.watch_pause_action.setChecked(True)
        assert page.watch_paused()
        page._watch_tick()
        assert not getattr(page, "_watch_looking", False)  # no look while paused
        page.watch_pause_action.setChecked(False)  # resume: looks again at once
        assert getattr(page, "_watch_looking", False) or page._watch_runner.wait(10)
    finally:
        page.stop_watch()
        _close(page)


def test_a_settle_time_holds_a_new_file_back_until_it_has_rested_that_long(tmp_path):
    path = tmp_path / "frame.tif"
    path.write_bytes(b"x" * 10)
    watch = FolderWatch(lambda _folders: [path])
    watch.start(tmp_path)
    watch._known.clear()  # (as if the file came after the watch began)
    watch.settle_s = 10.0
    state = ((10, None),)

    def look(moment):
        return watch.take(WatchScan(next(watch._numbers), tmp_path, ((path, watch._key(path), state[0]),), (), moment))

    assert look(100.0) == [] and look(102.0) == []  # the same at two looks, but only 2 s
    assert look(111.0) == [path]
    watch.settle_s = 0.0
    other = tmp_path / "other.tif"
    watch2 = FolderWatch(lambda _folders: [other])
    watch2.start(tmp_path)
    watch2._known.clear()
    scan = lambda moment: watch2.take(WatchScan(next(watch2._numbers), tmp_path, ((other, watch2._key(other), (5, None)),), (), moment))  # noqa: E731
    assert scan(0.0) == [] and scan(2.0) == [other]  # without a settle time: two looks


def test_every_new_text_has_its_chinese(tmp_path):
    from src.gimap.app.presentation.i18n.zh_insitu import INSITU_ZH

    page, _folder = _watching(tmp_path)
    try:
        texts = [action.text() for action in page.watch_button.menu().actions() if action.text()]
        texts += [action.text() for action in page.watch_settle_group.actions() if not action.text()[0].isdigit()]
        texts += [action.toolTip() for action in page.watch_button.menu().actions() if action.toolTip()]
        texts += [page.series_export_store_action.text(), page.series_export_store_action.toolTip()]
        assert sorted(text for text in texts if text not in ZH) == []
        for english, chinese in INSITU_ZH.items():
            assert ZH[english].replace("›", "▸") == chinese.replace("›", "▸")  # (› for ▸)
    finally:
        page.stop_watch()
        _close(page)
