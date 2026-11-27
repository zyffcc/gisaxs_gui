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


def test_the_data_rows_are_counted_in_qt_shown_from_their_sources_and_never_edited() -> None:
    from PyQt5.QtCore import QAbstractListModel, QModelIndex, QStringListModel, Qt
    from PyQt5.QtWidgets import QAbstractItemView, QApplication, QListView

    app = QApplication.instance() or QApplication([])
    # The list view lays out every row when rows are added and asks the row count twice per row: answered from Python,
    # 49 ms at 50 000 frames at every look of a watch (2 ms from Qt's string list). Not to be reimplemented here.
    assert issubclass(FrameListingModel, QStringListModel)
    assert not [cls for cls in FrameListingModel.__mro__
                if not cls.__module__.startswith("PyQt5") and "rowCount" in vars(cls)]
    model = FrameListingModel()
    model.reset_sources([(Path("C:/Run/a.edf"), range(1), "a.edf", 1, "tip a"),
                         (Path("C:/Run/stack.h5"), range(0, 9, 4), "stack.h5", 10, "tip s")])
    events = []
    model.modelReset.connect(lambda: events.append("reset"))
    model.rowsInserted.connect(lambda _parent, first, last: events.append((first, last)))
    model.append_sources([(Path("C:/Run/b.edf"), range(0), "b.edf", 1, ""),  # (no frame imported: no row)
                          (Path("C:/Run/c.edf"), range(1), "c.edf", 1, "tip c")])
    assert events == [(4, 4)] and model.rowCount() == model._rows == len(model.stringList()) == 5
    assert [model.data(model.index(row)) for row in range(5)] == ["a.edf", "1 · stack.h5", "5 · stack.h5",
                                                                  "9 · stack.h5", "c.edf"]
    assert model.data(model.index(3), Qt.UserRole) == (Path("C:/Run/stack.h5"), 8)
    assert model.data(model.index(5)) is None and model.rowCount(model.index(0)) == 0
    plain = QAbstractListModel.flags(model, model.index(0))
    assert model.flags(model.index(2)) == plain == Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemNeverHasChildren
    assert model.flags(QModelIndex()) == Qt.NoItemFlags
    view = QListView()
    view.setModel(model)
    view.show()
    view.edit(model.index(2))  # (a double-click: no editor, the row stays as it is)
    app.processEvents()
    assert view.state() != QAbstractItemView.EditingState and model.data(model.index(2)) == "5 · stack.h5"
    model.reset_sources([])
    assert events[-1] == "reset" and model.rowCount() == 0 and model.stringList() == []
    view.close()
    view.deleteLater()


@pytest.mark.parametrize("seed", range(6))
def test_a_look_applied_to_the_sources_it_marked_finds_what_looking_at_every_source_finds(seed, tmp_path) -> None:
    """``FolderWatch.take`` passes by the sources the look found as they were reported (14 ms of the window at 50 000
    frames at every look): what it reports and remembers is what looking at every one gives (``look=None``), with
    sources added, growing, changed back, held by their writer or gone, ``follow``, looks applied out of order or
    begun before another was applied, ``settle_s`` and sources complete by themselves."""
    import dataclasses
    import random

    from src.gimap.features.analyze.presentation.folder_watch import FolderWatch

    rng = random.Random(seed)
    files: dict[str, int] = {}
    held: set[str] = set()

    def expand(_folders):
        return [tmp_path / name for name in sorted(files)]

    def size_of(path):
        if path.name in held:
            raise OSError("held by its writer")
        return files[path.name]

    def watch_pair():
        made = []
        for _ in range(2):
            watch = FolderWatch(expand, size_of, lambda path: files[path.name] % 3,
                                (lambda path: files[path.name] % 5 == 0) if seed % 2 else None)
            watch.settle_s = 0.5 if seed % 3 == 0 else 0.0
            made.append(watch)
        return made

    for index in range(6):
        files[f"f_{index:03d}.edf"] = 10
    marked, every = watch_pair()
    for watch in (marked, every):
        watch.start(tmp_path)
    clock, pending, names, before = 0.0, [], 6, {}
    for _step in range(600):
        for _change in range(rng.randrange(4)):
            pick, name = rng.random(), rng.choice(sorted(files))
            if pick < 0.2 and len(files) < 12:
                files[f"f_{names:03d}.edf"] = rng.choice((0, 5, 10))
                names += 1
            elif pick < 0.45:  # (it grows)
                before.setdefault(name, files[name])
                files[name] += rng.choice((1, 5))
            elif pick < 0.7 and before:  # (written again as it was: as it was reported, or seen changed)
                name = rng.choice(sorted(before))
                if name in files:
                    files[name] = before.pop(name)
            elif pick < 0.8:
                held.symmetric_difference_update({name})
            elif pick < 0.85 and len(files) > 4:
                files.pop(name)
            elif pick < 0.95:
                marked.follow(tmp_path / name)
                every.follow(tmp_path / name)
        clock += rng.choice((0.1, 0.3, 0.6))
        a, b = marked.scan(), every.scan()
        pending.append((dataclasses.replace(a, moment=clock), dataclasses.replace(b, moment=clock, look=None)))
        while pending and (len(pending) > 4 or rng.random() < 0.6):
            first, second = pending.pop(rng.randrange(len(pending)) if rng.random() < 0.4 else 0)
            if rng.random() < 0.2:  # (begun before another look was applied, its number drawn after it)
                first = dataclasses.replace(first, number=next(marked._numbers))
                second = dataclasses.replace(second, number=next(every._numbers))
            assert marked.take(first) == every.take(second)
            for name in ("changed", "waiting", "moving", "_known", "_sizes", "_since"):
                assert getattr(marked, name) == getattr(every, name), name
    assert marked._known and len(marked._noted) < 400  # (the states set before a look applied are forgotten)


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


@needs_frames
def test_the_data_list_adds_the_rows_of_the_new_sources_as_listing_all_again_gives_them(tmp_path) -> None:
    """A look that lists new frames adds their Data rows at the end (``refresh_frame_list(appended=…)``) instead of
    listing every source again: the same rows."""
    page, folder = _watching(tmp_path, 3)
    try:
        model, listing = page.view_model, page.frame_listing_model
        resets = []
        listing.modelReset.connect(lambda: resets.append(1))
        for index in range(3, 7):
            write_frame(folder, index, jump_at=6)
            page.poll_watch()
            _done(page)
        page.poll_watch()  # (the last frame is taken once it rests)
        _done(page)
        assert len(model.state.files) == 7 and not resets  # (rows added, the list not made again)
        added = [(str(span[0]), list(span[1]), span[2], span[3], span[4]) for span in listing.spans]
        whole = FrameListingModel()
        whole.reset_sources([page._frame_list_span(path) for path in model.state.files])
        assert added == [(str(span[0]), list(span[1]), span[2], span[3], span[4]) for span in whole.spans]
        assert (listing.ends, listing._where) == (whole.ends, whole._where)
        page.refresh_frame_list()  # (everything again: the same)
        assert [str(span[0]) for span in listing.spans] == [str(span[0]) for span in whole.spans]
    finally:
        page.stop_watch()
        _close(page)


@needs_frames
def test_frame_numbers_of_a_list_grown_at_its_end_are_those_made_from_the_whole_list(tmp_path) -> None:
    from src.gimap.features.analyze.presentation.frame_sequences import FrameNumbers

    page, folder = _listed(tmp_path, 4)
    try:
        model = page.view_model
        before = model.frame_numbers()
        for index in range(4, 7):
            path = write_frame(folder, index)
            model.state.files.append(Path(path))
            model.remember_frame_count(Path(path), 2 if index == 5 else 1)
        grown = model.frame_numbers()
        whole = FrameNumbers((path, model.known_frame_count(path)) for path in model.state.files)
        assert grown is not before and grown.sources == whole.sources and grown.total == whole.total
        assert list(grown.sources) == list(whole.sources) and grown.agrees_with(before)
        assert model.frame_numbers() is grown
    finally:
        _close(page)


@needs_frames
def test_the_frames_listed_are_counted_again_only_for_new_sources_and_changed_counts(tmp_path) -> None:
    page, folder = _listed(tmp_path, 4)
    try:
        model = page.view_model
        assert model.listed_frames() == 4
        for index in range(4, 7):
            path = Path(write_frame(folder, index))
            model.state.files.append(path)
            model.remember_frame_count(path, 1)
        assert model.listed_frames() == 7
        model.remember_frame_count(model.state.files[0], 5)  # (a source before grew: everything again)
        assert model.listed_frames() == 11 == sum(model.known_frame_count(path) for path in model.state.files)
        model.state.files.pop(1)
        assert model.listed_frames() == 10
    finally:
        _close(page)
