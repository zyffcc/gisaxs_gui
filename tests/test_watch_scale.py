"""A folder watch of thousands of frames keeps its looks short: what it works out at every look is kept while what it
depends on stays the same, or worked out for the new frames alone, and gives what working it all out gave — the frame
numbers (also after a row is replaced in place), the frames of the run of a map, the Data row of a frame, the frames
imported, the order of the names; a look is not followed by the next sooner than it took. On the public GALAXI frame
(``tests/insitu_frames.py``)."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.gimap.features.analyze.presentation.frame_listing import FrameListingModel
from src.gimap.features.analyze.presentation.frame_selection import FrameSelection
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from tests.insitu_frames import GALAXI, crop_profile, write_frame
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_insitu_online import _english, _watching  # noqa: F401 (the fixture)
from tests.test_live_following import _close

needs_frames = pytest.mark.skipif(not GALAXI.is_file(), reason="public test frames (tests/data/external) are not available")


def _listed(tmp_path: Path, frames: int):
    folder = tmp_path / "run"
    for index in range(frames):
        write_frame(folder, index)
    page = _page(_context(InMemoryInstrumentProfileRepository([crop_profile()])))
    page.set_mode_choice("gisaxs")
    page.add_paths([str(folder)])
    _done(page)
    return page, folder


@needs_frames
def test_the_frame_numbers_are_kept_until_the_list_or_a_count_changes(tmp_path) -> None:
    page, folder = _listed(tmp_path, 4)
    try:
        model = page.view_model
        numbers = model.frame_numbers()
        assert model.frame_numbers() is numbers  # kept: nothing changed
        files = model.state.files
        files[1], files[2] = files[2], files[1]  # rows replaced in place, the length the same
        swapped = model.frame_numbers()
        assert swapped is not numbers and swapped[(str(files[1]), 0)] == 2 and swapped[(str(files[2]), 0)] == 3
        model.remember_frame_count(files[0], 3)  # a source now holds three frames
        grown = model.frame_numbers()
        assert grown is not swapped and grown[(str(files[1]), 0)] == 4 and len(grown) == 6
    finally:
        _close(page)


@needs_frames
def test_the_frames_of_a_map_of_single_frames_are_found_as_grouping_the_run_finds_them(tmp_path) -> None:
    page, folder = _listed(tmp_path, 8)
    try:
        model = page.view_model
        for first, last in ((1, None), (3, None), (2, 6), (7, 7)):
            built = {"first": first, "last": last or 0, "every": 1}
            in_run = page._watch_series_in_run(built)
            with model.only_frames(first, last):
                groups = model._lazy_groups()
            for path in model.state.files:
                request = model.request_for(path, 0, summed=())
                assert in_run(request) == (groups.index_for(request.path, 0) is not None), (first, last, path.name)
    finally:
        _close(page)


def test_the_data_row_of_a_frame_is_found_as_a_scan_of_the_sources_finds_it() -> None:
    def scanned(spans, path, frame):  # (the scan of every source, before the index)
        before = 0
        for candidate, indices, *_rest in spans:
            if len(indices) == 0:
                continue
            if str(candidate).casefold() == str(path).casefold():
                return before + indices.index(frame) if frame in indices else -1
            before += len(indices)
        return -1

    spans = [(Path(f"C:/Run/f_{k:03d}.edf"), range(1), f"f_{k:03d}.edf", 1, "") for k in range(40)]
    spans[7] = (spans[7][0], range(0), "f_007.edf", 1, "")  # a source with no frame imported: no row
    spans[12] = (Path("C:/Run/stack.h5"), range(2, 9, 3), "stack.h5", 10, "")
    model = FrameListingModel()
    model.reset_sources(spans)
    for path in [Path("c:/run/F_003.EDF"), spans[7][0], spans[12][0], spans[39][0], Path("C:/Run/other.edf")]:
        for frame in range(10):
            assert model.row_for(path, frame) == scanned(spans, path, frame), (path, frame)


@needs_frames
def test_the_frames_imported_are_counted_alike_with_and_without_a_selection(tmp_path) -> None:
    page, _folder = _listed(tmp_path, 5)
    try:
        model = page.view_model
        every = sum(len(model.selected_frame_indices(path)) for path in model.state.files)
        assert model.imported_frame_count() == every == 5
        model.remember_frame_count(model.state.files[2], 3)  # a source of three frames, two of them imported
        model.set_import_selection(model.state.files[2], FrameSelection.range(1, 2))
        assert model.imported_frame_count() == sum(len(model.selected_frame_indices(path)) for path in model.state.files) == 6
    finally:
        _close(page)


def test_names_are_sorted_with_their_numbers_as_numbers() -> None:
    from src.gimap.features.analyze.infrastructure.adapters.frames import _natural_key

    names = ["run_10.edf", "run_9.edf", "Run_100.edf", "run_09b.edf", "a.edf", "run_1.edf"]
    ordered = sorted((Path("C:/data") / name for name in names), key=_natural_key)
    assert [path.name for path in ordered] == ["a.edf", "run_1.edf", "run_9.edf", "run_09b.edf", "run_10.edf",
                                               "Run_100.edf"]


@needs_frames
def test_a_look_is_not_followed_by_the_next_sooner_than_it_took(tmp_path) -> None:
    from src.gimap.features.analyze.presentation.bindings.batch_watch import NOTICED_MS

    page, folder = _watching(tmp_path)
    try:
        page._watch_timer.start()  # (looks on notices only while the timer looks too)
        page._folder_changed()
        page._notice_timer.setInterval(NOTICED_MS)
        remaining = page._notice_timer.remainingTime()
        time.sleep(0.02)
        page._folder_changed()  # another change meanwhile: the look is not put off
        assert page._notice_timer.isActive() and page._notice_timer.remainingTime() < remaining
        page._notice_timer.stop()
        page._watch_look_began = time.monotonic() - 0.4  # a look of 0.4 s
        page._watch_looked(page.view_model.watch.scan())
        assert page._notice_timer.interval() >= 400
        page._watch_look_began = time.monotonic()  # a quick one
        page._watch_looked(page.view_model.watch.scan())
        assert page._notice_timer.interval() == NOTICED_MS
    finally:
        page.stop_watch()
        _close(page)
