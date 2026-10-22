"""Following a source while a detector writes it: a frame every fraction of a second, for as long as the run lasts.

A second process appends frames to

* one stack file (the default file format, the file kept open and flushed after each frame);
* the module files of one detector, one module after the other;
* a file in the mode in which others may read while it is written (SWMR), each frame announced a moment before
  its pixels are there;
* a file in the latest format without that mode, which nobody can read until it is closed.

The folder watch looks at the folder meanwhile, as its timer does. Whatever it hands out (to the Series map, to
Export New Frames While Watching) is read at once, as an export would read it: every frame once, in order, none
left out, each one whole. Frames are handed out while the writer is still at work, not only after it has ended.
Every file is synthetic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource
from src.gimap.features.analyze.presentation import sources_model
from src.gimap.features.analyze.presentation.folder_watch import FolderWatch
from src.gimap.shared.detector_io import forget_layout
from src.gimap.shared.detector_io import nexus_files
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_review_analyze_run_and_clear import _profiles
from tests.test_series_map import _ring_frame

H, W = 24, 36
LOOK_S = 0.5
"""Seconds between two looks of the tests (the timer of the page looks every two seconds)."""


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    apply_language("en")
    yield
    apply_language("en")
    forget_layout()


_WRITER = """
    import os, sys, time
    import h5py
    import numpy as np
    mode, period, frames, paths = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    H, W = 24, 36
    rings = np.load(paths.pop()) if paths[-1].endswith(".npy") else None   # frames a GIWAXS reduction can work on
    dataset = "/entry/data/data" if rings is not None else "/entry/instrument/detector/data"
    shape = rings.shape[1:] if rings is not None else (H, W)
    latest = "latest" if mode in ("swmr", "latest") else None
    # The default format is followed because a writer's HDF5 lock does not stop a reader: on Linux / macOS it is
    # advisory, and a detector server's lock does not reach a share. On Windows HDF5 locks the bytes themselves
    # (nobody can read the file until it is closed), so there the writer writes without that lock.
    unlocked = {"locking": False} if os.name == "nt" and mode in ("stack", "modules") else {}
    handles = [h5py.File(path, "w", libver=latest, **unlocked) for path in paths]
    stacks = []
    for number, handle in enumerate(handles, start=1):
        stacks.append(handle.create_dataset(dataset, shape=(0,) + shape, maxshape=(None,) + shape,
                                            dtype="f4" if rings is not None else "i4", chunks=(1,) + shape))
        if len(handles) > 1:    # the module files of one detector: where each module sits
            handle.create_dataset("/entry/instrument/detector/translation/distance", data=[0.0, (number - 1) * (H + 4), 0.0])
        handle.flush()
    if mode == "swmr":
        for handle in handles:
            handle.swmr_mode = True
    print("writing", flush=True)
    for index in range(frames):
        for number, (handle, data) in enumerate(zip(handles, stacks), start=1):   # one module after the other
            data.resize(index + 1, axis=0)
            if mode == "swmr":      # the frame is announced before its pixels are there
                data.flush()
                time.sleep(period / 4)
            data[index] = rings[index] if rings is not None else 1000 * number + index + 1
            data.flush() if mode == "swmr" else handle.flush()
            if len(handles) > 1:
                time.sleep(period / (2 * len(handles)))
        time.sleep(period)
    print("written", flush=True)
    if mode in ("swmr", "latest"):
        sys.stdin.readline()    # the run is over; the file stays open until the test says so
    for handle in handles:
        handle.close()
"""


def _writer(mode: str, paths, *, frames: int, period: float = 0.3) -> subprocess.Popen:
    """Another program that writes ``frames`` frames into ``paths``, one every ``period`` seconds (and a little
    more). ``stack`` / ``modules``: the default file format, kept open, flushed after each frame; ``swmr``: readable
    meanwhile; ``latest``: not readable until closed. A last path ending in ``.npy`` holds the frames to write
    (frames a reduction can work on) instead of flat ones. ``swmr`` and ``latest`` keep the file open after the
    last frame until ``_release``."""
    process = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_WRITER), mode, str(period), str(frames), *map(str, paths)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "writing"
    return process


def _release(process: subprocess.Popen) -> None:
    process.communicate("\n", timeout=60)
    assert process.returncode == 0


def _model(profiles=None):
    return create_analyze_view_model(_context(profiles))


def _whole(image, index: int, modules: int) -> bool:
    """The frame read holds what the writer wrote for frame ``index`` in every module, and nothing else: no
    pixel of a frame that was announced and not written yet (0), no module left out."""
    values = set(np.unique(image.data[np.isfinite(image.data)]).astype(int).tolist())
    files = image.metadata.get("module_files") or [image.source_path]
    return values == {1000 * number + index + 1 for number in range(1, modules + 1)} and len(files) == modules


class _Following:
    """Looks at the watched folder as the timer does and reads every frame handed out at once."""

    def __init__(self, model, modules: int = 1):
        self.model, self.modules = model, modules
        self.source = DetectorIoFrameSource()
        self.handed: list[tuple[str, int]] = []
        self.broken: list[tuple[str, int]] = []
        self.grown: list[tuple[str, int]] = []

    def look(self) -> None:
        added = self.model.poll_watch()
        self.grown += [(path.name, count) for path, count in self.model.watch_grown]
        for request in self.model.watch_requests(added):
            frame = (request.path.name, request.frame_index)
            self.handed.append(frame)
            if not _whole(self.source.load(request.path, request.frame_index), request.frame_index, self.modules):
                self.broken.append(frame)

    def while_written(self, writer: subprocess.Popen, done=None) -> int:
        """Look every ``LOOK_S`` until the writer has ended (or ``done()``); frames handed out by then."""
        deadline = time.monotonic() + 120
        while (writer.poll() is None if done is None else not done()) and time.monotonic() < deadline:
            self.look()
            time.sleep(LOOK_S)
        return len(self.handed)

    def rest(self, looks: int = 4) -> None:
        """A few more looks, ``LOOK_S`` apart as the timer's looks are (two looks a moment apart saw a frame the
        SWMR writer had announced but not yet written as resting, on a busy machine)."""
        for _look in range(looks):
            self.look()
            time.sleep(LOOK_S)


# -- the watch itself --------------------------------------------------------------------------------------------------


def test_the_watch_says_which_sources_changed_again_and_what_they_were_at_the_look_before(tmp_path: Path) -> None:
    """``moving``: a source seen changed at two looks in a row, with its state at the look before."""
    sizes = {"a.cbf": 10}
    stamps = {"a.cbf": "one"}
    watch = FolderWatch(lambda _folders: [tmp_path / name for name in sorted(sizes)],
                        lambda path: sizes[path.name], lambda path: stamps[path.name])
    watch.start(tmp_path)
    assert watch.poll() == [] and watch.moving == [] and watch.rests(tmp_path / "a.cbf")
    sizes["a.cbf"], stamps["a.cbf"] = 20, "two"
    assert watch.poll() == [] and watch.moving == []  # changed once: it may rest at the next look
    assert not watch.rests(tmp_path / "a.cbf")
    sizes["a.cbf"], stamps["a.cbf"] = 30, "three"
    assert watch.poll() == [] and watch.changed == []
    assert watch.moving == [(tmp_path / "a.cbf", (20, "two"))]  # changed again: what it was at the look before
    sizes["a.cbf"], stamps["a.cbf"] = 40, "four"
    watch.poll()
    assert watch.moving == [(tmp_path / "a.cbf", (30, "three"))]
    assert watch.poll() == [] and watch.moving == [] and [path.name for path in watch.changed] == ["a.cbf"]  # it rests
    # A new source that never rests is no new frame for the watch (``ready``); the caller may list it all the same.
    sizes["b.cbf"], stamps["b.cbf"] = 5, "b1"
    assert watch.poll() == [] and watch.moving == []
    sizes["b.cbf"], stamps["b.cbf"] = 6, "b2"
    assert watch.poll() == [] and watch.moving == [(tmp_path / "b.cbf", (5, "b1"))]
    watch.follow(tmp_path / "b.cbf")  # … then it is a known source: when it rests it has changed, it is not new
    sizes["b.cbf"], stamps["b.cbf"] = 7, "b3"
    assert watch.poll() == [] and watch.moving == [(tmp_path / "b.cbf", (6, "b2"))]
    assert watch.poll() == [] and [path.name for path in watch.changed] == ["b.cbf"] and watch.moving == []


# -- one source that grows for as long as the run lasts ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["stack", "swmr"])
def test_a_file_that_grows_at_every_look_is_followed_frame_by_frame(tmp_path: Path, mode: str) -> None:
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model)
    path = tmp_path / "live_00001.nxs"
    frames = 16
    writer = _writer(mode, [path], frames=frames)
    try:
        if mode == "swmr":  # (it keeps the file open after its last frame: the run is over when all are there)
            live = following.while_written(writer, done=lambda: model.frame_count(path) >= frames)
        else:
            live = following.while_written(writer)
        following.rest()
        # Every frame once, in order, none left out; every one whole when it was handed out.
        assert following.handed == [("live_00001.nxs", index) for index in range(frames)]
        assert following.broken == []
        assert live >= 4  # handed out while the detector was writing, not after the run
        assert [path.name for path in model.state.files] == ["live_00001.nxs"] and model.listed_frames() == frames
        following.rest(2)
        assert len(following.handed) == frames  # the run is over: nothing more, nothing twice
    finally:
        if mode == "swmr":
            _release(writer)
        writer.wait(60)
    if mode == "swmr":  # the writer closed its file: a change without a new frame
        following.rest(3)
        assert len(following.handed) == frames and model.describe(path).complete


def test_a_listed_file_that_starts_to_grow_hands_out_only_its_new_frames(tmp_path: Path) -> None:
    """The file was there, with frames, when the watch began; then a writer appends to it for a while."""
    path = tmp_path / "insitu_00001.nxs"
    with h5py.File(path, "w") as handle:
        data = np.zeros((4, H, W), dtype=np.int32) + np.arange(4, dtype=np.int32)[:, None, None] + 1001
        handle.create_dataset("/entry/instrument/detector/data", data=data, maxshape=(None, H, W), chunks=(1, H, W))
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model)
    following.look()
    assert following.handed == [] and model.listed_frames() == 4

    def append(count: int) -> None:
        with h5py.File(path, "a") as handle:
            data = handle["/entry/instrument/detector/data"]
            start = data.shape[0]
            data.resize(start + count, axis=0)
            data[start:] = (np.arange(start, start + count, dtype=np.int32) + 1001)[:, None, None]

    append(2)                       # 6 frames
    following.look()                # seen changed for the first time: nothing yet
    assert following.handed == []
    append(3)                       # 9 frames
    following.look()                # changed again: the frames it had at the look before
    assert following.handed == [("insitu_00001.nxs", 4), ("insitu_00001.nxs", 5)]
    assert following.grown == [("insitu_00001.nxs", 2)] and model.followed_frame(path) == 5
    append(1)                       # 10 frames
    following.look()
    assert following.handed[2:] == [("insitu_00001.nxs", index) for index in (6, 7, 8)] and model.followed_frame(path) == 8
    assert model.growing(path)      # its newest frame may still be being written
    # A series run started now (Batch Export, a Series map) takes the frames handed out, not the newest one.
    assert model.frame_count(path) == 10 and model.frames_to_run(path) == 9
    following.look()                # it rests: the rest
    assert following.handed[5:] == [("insitu_00001.nxs", 9)] and following.broken == []
    assert not model.growing(path) and model.frames_to_run(path) == 10
    following.rest(2)
    assert len(following.handed) == 6 and model.followed_frame(path) == 9
    model.stop_watch()
    append(2)                       # nothing is watched: every frame there is
    assert model.frames_to_run(path) == 12


def test_a_file_written_again_in_place_at_every_look_waits_until_it_rests(tmp_path: Path) -> None:
    """A “latest frame” file that a beamline overwrites all the time does not grow: what it held at the look
    before is not what it holds now, so it is listed, and its frame handed out, only once it rests."""
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model)
    path = tmp_path / "latest_frame.h5"

    def write(value: int) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("/entry/instrument/detector/data", data=np.full((1, H, W), value, dtype=np.int32))
        os.utime(path, ns=(time.time_ns(), time.time_ns() + value * 1_000_000_000))  # (written at another time)

    for value in (5, 6, 7, 8):
        write(value)
        following.look()
        assert model.state.files == [] and following.handed == []
    write(1001)
    following.rest(2)
    assert [listed.name for listed in model.state.files] == ["latest_frame.h5"]
    assert following.handed == [("latest_frame.h5", 0)] and following.broken == []


def test_the_modules_of_a_detector_written_one_after_the_other_are_followed_whole(tmp_path: Path) -> None:
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model, modules=3)
    paths = [tmp_path / f"scan_00001_m{number:02d}.nxs" for number in (1, 2, 3)]
    frames = 22
    writer = _writer("modules", paths, frames=frames, period=0.25)
    try:
        live = following.while_written(writer)
        following.rest()
        # No frame twice, none skipped, none with a module missing or not written yet.
        assert following.handed == [("scan_00001_m01.nxs", index) for index in range(frames)]
        assert following.broken == []
        assert live >= 2  # while it was written (a set the watch meets waits a few looks first: more files may follow)
        assert [path.name for path in model.state.files] == ["scan_00001_m01.nxs"] and model.listed_frames() == frames
        following.rest(sources_model.SET_RESTS + 1)
        assert len(following.handed) == frames
    finally:
        writer.wait(60)


def test_a_second_scan_of_the_detector_is_followed_from_its_first_look(tmp_path: Path) -> None:
    """A set with as many module files as the scan before it does not wait: it is whole as far as its files go."""
    def scan(number: int, frames: int) -> None:
        for module in (1, 2):
            with h5py.File(tmp_path / f"scan_{number:05d}_m{module:02d}.nxs", "w") as handle:
                data = np.zeros((frames, H, W), dtype=np.int32) + np.arange(frames, dtype=np.int32)[:, None, None]
                handle.create_dataset("/entry/instrument/detector/data", data=data + 1000 * module + 1,
                                      maxshape=(None, H, W), chunks=(1, H, W))
                handle.create_dataset("/entry/instrument/detector/translation/distance", data=[0.0, (module - 1) * (H + 4), 0.0])

    scan(1, 3)
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model, modules=2)
    paths = [tmp_path / f"scan_00002_m{number:02d}.nxs" for number in (1, 2)]
    writer = _writer("modules", paths, frames=14, period=0.25)
    try:
        looks = 0
        while not following.handed and writer.poll() is None:
            following.look()
            looks += 1
            time.sleep(LOOK_S)
        assert following.handed and looks <= 4  # not after SET_RESTS looks: at the second look that sees its frames
        following.while_written(writer)
        following.rest()
        assert following.handed == [("scan_00002_m01.nxs", index) for index in range(14)] and following.broken == []
    finally:
        writer.wait(60)


def _modules(paths, frames: int) -> None:
    """Module files of one detector as its writer leaves them after ``frames`` frames (frame i of module n holds
    1000 n + i + 1, as ``_whole`` expects); a file that is not there yet is begun."""
    for path in paths:
        number = int(Path(path).stem[-2:])
        with h5py.File(path, "a") as handle:
            if "/entry/instrument/detector/data" not in handle:
                handle.create_dataset("/entry/instrument/detector/data", shape=(0, H, W), maxshape=(None, H, W),
                                      dtype="i4", chunks=(1, H, W))
                handle.create_dataset("/entry/instrument/detector/translation/distance",
                                      data=[0.0, (number - 1) * (H + 4), 0.0])
            data = handle["/entry/instrument/detector/data"]
            before = data.shape[0]
            data.resize(frames, axis=0)
            for index in range(before, frames):
                data[index] = 1000 * number + index + 1


def test_a_set_whose_last_module_file_shows_up_a_look_late_waits_one_look_more(tmp_path: Path) -> None:
    """A second scan of the detector: its m01 and m02 are seen growing at one look, its m03 shows up at the next.
    With as many module files as the scan before it the set is whole at once, but what it held at the look before
    was held by two files: it says nothing of m03, whose newest frame may still be being written. Nothing is handed
    out at that look; the frames follow at the look after."""
    _modules([tmp_path / f"scan_00001_m{module:02d}.nxs" for module in (1, 2, 3)], 3)
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model, modules=3)
    scan = [tmp_path / f"scan_00002_m{module:02d}.nxs" for module in (1, 2, 3)]
    _modules(scan[:2], 2)
    following.look()                # new: seen once
    _modules(scan[:2], 3)
    following.look()                # grows: listed and followed; two module files of three: it waits
    assert [path.name for path in model.state.files] == ["scan_00001_m01.nxs", "scan_00002_m01.nxs"]
    assert following.handed == []
    _modules(scan[:2], 4)
    _modules(scan[2:], 4)
    following.look()                # m03 is there: whole, but the frames of the look before were of two files
    assert len(model.describe(scan[0]).members) == 3 and following.handed == []
    _modules(scan, 5)
    following.look()                # what all three held at the look before
    assert following.handed == [("scan_00002_m01.nxs", index) for index in range(4)]
    following.rest()
    assert following.handed == [("scan_00002_m01.nxs", index) for index in range(5)] and following.broken == []


def test_a_file_nobody_can_read_until_it_is_closed_is_waited_for_and_loses_nothing(tmp_path: Path) -> None:
    model = _model()
    model.start_watch(tmp_path)
    following = _Following(model)
    path = tmp_path / "live_00001.nxs"
    frames = 8
    writer = _writer("latest", [path], frames=frames, period=0.2)
    try:
        said = set()
        for _look in range(5):  # the detector is writing
            following.look()
            said |= {(waiting.name, why) for waiting, why in model.watch_waiting}
            time.sleep(LOOK_S)
        assert writer.stdout.readline().strip() == "written"  # the run is over, the file is still open
        following.rest(2)
        assert following.handed == [] and model.state.files == []
        assert said == {("live_00001.nxs", "live_00001.nxs is being written by another program and cannot be read "
                                           "until it is closed.")}
    finally:
        _release(writer)
    following.rest()
    assert following.handed == [("live_00001.nxs", index) for index in range(frames)] and following.broken == []
    assert model.watch_waiting == [] and model.listed_frames() == frames


# -- the page ------------------------------------------------------------------------------------------------------------


def _rows(page) -> list[str]:
    return [page.file_list.item(row).text() for row in range(page.file_list.count())]


def _close(page) -> None:
    page.tasks.wait(60)
    page.dispose()
    page.close()


def _pump(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _app().processEvents()
        time.sleep(0.01)


def _statuses(page) -> list[tuple[str, str]]:
    """Every sentence the status line says from now on, with its level."""
    said: list[tuple[str, str]] = []
    real = page._status

    def status(text, level="info"):
        said.append((text, level))
        real(text, level)

    page._status = status
    return said


def test_the_page_follows_a_growing_file_and_exports_every_frame_once(tmp_path: Path) -> None:
    frames = 12
    rings = tmp_path / "frames_to_write.npy"
    np.save(rings, np.stack([_ring_frame(200.0 + 10 * index, seed=index) for index in range(frames)]))
    folder = tmp_path / "run"
    folder.mkdir()
    page = _page(_context(_profiles()))
    writer = None
    try:
        page.set_mode_choice("giwaxs")
        page.start_watch(folder)
        page._watch_timer.stop()  # the test looks itself
        page.auto_export_check.setChecked(True)
        said = _statuses(page)
        writer = _writer("stack", [folder / "live_00001.h5", rings], frames=frames, period=0.4)
        shown: list[tuple[int, int]] = []  # (the frame on screen, frames handed out so far) after every look
        live_rows: list[str] = []
        notes: list[tuple[str, int, int]] = []  # (what the Data step says of the growth, the newest whole frame, frames)
        while writer.poll() is None:
            page.poll_watch()
            _pump(LOOK_S)
            if page.file_list.count():
                handed = page.view_model.followed_frame(folder / "live_00001.h5")
                shown.append((page.view_model.state.frame_index, -1 if handed is None else handed))
                live_rows.append(_rows(page)[0])
                analysis = page.view_model.state.analysis
                if analysis is not None and "Still growing" in page.source_label.text():
                    notes.append((page.source_label.text().splitlines()[-1], handed, page.frame_spin.maximum()))
        for _look in range(4):
            page.poll_watch()
            _done(page)
        exported = sorted(found.name for found in (folder / "gimap_analysis").rglob("*_radial.csv"))
        assert exported == [f"live_00001_frame{index:04d}_radial.csv" for index in range(frames)]  # each frame, once
        stamps = {found.name: found.stat().st_mtime_ns for found in (folder / "gimap_analysis").rglob("*_radial.csv")}
        # While it was written: listed, its row counting up, the newest frame handed out on screen (never one
        # that is still being written), and the status line saying what arrived.
        # (How many looks fit into the run depends on how busy the machine is: what each look did must be right.)
        assert len(set(live_rows)) >= 2 and live_rows[-1].startswith("live_00001.h5 · ")
        assert shown and all(frame <= handed for frame, handed in shown) and max(frame for frame, _h in shown) >= 1
        new = [text for text, level in said if text.startswith("live_00001.h5: ") and level == "ok"]
        assert new and all(text.endswith(("new frames", "new frame")) for text in new)
        # Whenever the Data step said which frame was the newest complete one, more were counted already.
        assert all(text.startswith("Still growing: frame ") and text.endswith(" is the newest complete one")
                   for text, _handed, _frames in notes)
        assert all(int(text.split()[3]) <= handed + 1 < frames_then for text, handed, frames_then in notes)
        assert "Still growing" not in page.source_label.text()  # the run is over: every frame counted is complete
        # At the end: every frame, the last one on screen.
        analysis = page.view_model.state.analysis
        assert _rows(page) == [f"live_00001.h5 · {frames} frames"]
        assert (analysis.frame_index, analysis.frame_count) == (frames - 1, frames) and page.frame_spin.value() == frames
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert {found.name: found.stat().st_mtime_ns for found in (folder / "gimap_analysis").rglob("*_radial.csv")} == stamps
    finally:
        if writer is not None:
            writer.wait(60)
        page.stop_watch()
        _close(page)


def test_the_page_says_once_which_file_it_waits_for_and_lists_it_when_it_is_closed(tmp_path: Path) -> None:
    page = _page(_context())
    writer = None
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        said = _statuses(page)
        writer = _writer("latest", [tmp_path / "live_00001.nxs"], frames=6, period=0.2)
        for _look in range(6):
            page.poll_watch()
            _pump(LOOK_S)
        waits = (f"Watching {tmp_path.name}: live_00001.nxs is being written by another program and cannot be read "
                 "until it is closed.")
        assert [text for text, _level in said] == [waits] and page.file_list.count() == 0  # once, however many looks
        _release(writer)
        writer = None
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert _rows(page) == ["live_00001.nxs · 6 frames"]  # nothing is lost
    finally:
        if writer is not None:
            _release(writer)
        page.stop_watch()
        _close(page)


_KILLED = """
    import os, signal, sys, h5py
    handle = h5py.File(sys.argv[1], "w", libver="latest")
    data = handle.create_dataset("/entry/instrument/detector/data", shape=(0, 24, 36), maxshape=(None, 24, 36),
                                 dtype="i4", chunks=(1, 24, 36))
    if len(sys.argv) > 2:   # a module file of a detector: where the module sits
        handle.create_dataset("/entry/instrument/detector/translation/distance", data=[0.0, float(sys.argv[2]), 0.0])
    for index in range(5):
        data.resize(index + 1, axis=0)
        data[index] = index
        handle.flush()
    # the writer dies without closing the file, which keeps its mark "open for writing". Not os._exit: on Windows
    # that still unloads HDF5, which then closes its files cleanly; a kill skips that on every platform.
    os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))
"""


def test_a_file_left_by_a_writer_that_ended_is_not_waited_for_for_ever(tmp_path: Path) -> None:
    path = tmp_path / "crashed_00001.nxs"
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        said = _statuses(page)
        subprocess.run([sys.executable, "-c", textwrap.dedent(_KILLED), str(path)], check=False, timeout=60)
        page.poll_watch()
        page.poll_watch()
        waits = (f"Watching {tmp_path.name}: crashed_00001.nxs is being written by another program and cannot be read "
                 "until it is closed.")
        assert said == [(waits, "info")]  # just written: nothing tells it from a file its writer holds
        stat = path.stat()
        os.utime(path, (stat.st_atime, stat.st_mtime - 10 * nexus_files.UNFINISHED_AFTER_S))  # … a while later
        page.poll_watch()
        page.poll_watch()
        left = (f"Watching {tmp_path.name}: crashed_00001.nxs was left open for writing and has not changed for a "
                "while: the program that wrote it seems to have ended without closing it. It cannot be read as it is; "
                "running h5clear -s on a copy of it makes readable what was stored in it.")
        assert said[1:] == [(left, "warning")]  # said once, as something to act on
        assert page.file_list.count() == 0  # not listed by the watch: it holds nothing that can be read
        # It is said once, so a switch of the language says it again itself, while the line still shows it.
        apply_language("zh", [page])
        page.refresh_language()
        chinese = page.status_text()
        assert chinese.startswith(f"正在监视 {tmp_path.name}：crashed_00001.nxs 仍标记为“正在写入”")
        assert "h5clear -s" in chinese and page.status_level() == "warning"
        page.poll_watch()
        assert len(said) == 3  # … and not once more at the next look
        apply_language("en", [page])
        page.refresh_language()
        assert page.status_text() == left
        page._status("Ready")  # the line says something else by now: a switch leaves it alone
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text() != chinese
        apply_language("en", [page])
        page.refresh_language()
        # Opened by hand: the same sentence, as the reason the frame cannot be read.
        page.add_paths([str(path)])
        _done(page)
        assert page.status_text() == "Could not analyse the frame: " + left.split(": ", 1)[1]
        assert page.status_level() == "error"
    finally:
        page.stop_watch()
        _close(page)


def _killed(path: Path, *position: float) -> Path:
    """A file whose writer ended without closing it, in the latest format (nobody can read it); with a
    ``position``, a module file of a detector."""
    subprocess.run([sys.executable, "-c", textwrap.dedent(_KILLED), str(path), *map(str, position)], check=False,
                   timeout=60)
    return path


def _long_ago(path: Path) -> None:
    """… and nothing has changed it for a while since."""
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime - 10 * nexus_files.UNFINISHED_AFTER_S))


def test_what_the_watch_waits_for_is_said_once_also_while_another_source_grows(tmp_path: Path) -> None:
    """A file nobody can read lies in the watched folder while another source gains frames. Which file is waited
    for is said once, not again after every look that brought frames; again only when the reason changes."""
    from tests.test_analyze_nexus_sources import _grow, _stack

    growing = _stack(tmp_path / "insitu_00001.nxs", 2)
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        _done(page)
        said = _statuses(page)
        held = _killed(tmp_path / "pilatus_00007.nxs")
        frames = [2]

        def looks(count: int) -> None:  # a frame more at every other look: looks with new frames and quiet ones
            for look in range(count):
                if look % 2 == 0:
                    frames[0] += 1
                    _grow(growing, frames[0])
                page.poll_watch()
                _done(page)

        looks(8)

        def about() -> list[tuple[str, str]]:
            return [(text, level) for text, level in said if "pilatus_00007.nxs" in text]

        waits = (f"Watching {tmp_path.name}: pilatus_00007.nxs is being written by another program and cannot be read "
                 "until it is closed.")
        assert about() == [(waits, "info")]
        assert sum(text.startswith("insitu_00001.nxs: ") for text, _level in said) >= 3  # frames were said meanwhile
        held.unlink()               # nothing is waited for any more …
        looks(2)
        _killed(held)               # … until it is there again: said again, once
        looks(6)
        assert about() == [(waits, "info"), (waits, "info")]
        _long_ago(held)             # its writer is gone: another reason, said once
        looks(8)
        left = (f"Watching {tmp_path.name}: pilatus_00007.nxs was left open for writing and has not changed for a while: "
                "the program that wrote it seems to have ended without closing it. It cannot be read as it is; running "
                "h5clear -s on a copy of it makes readable what was stored in it.")
        assert about() == [(waits, "info"), (waits, "info"), (left, "warning")]
    finally:
        page.stop_watch()
        _close(page)


def test_with_a_sum_the_frames_on_screen_end_at_the_newest_complete_one(tmp_path: Path) -> None:
    """Sum 3 while a file grows: the frame shown is the first of the three it sums, so that they end at the newest
    frame handed out, never at one that may still be being written."""
    from tests.test_analyze_nexus_sources import _grow, _stack

    path = _stack(tmp_path / "insitu_00001.nxs", 4)
    page = _page(_context())
    try:
        page.sum_spin.setValue(3)
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        _done(page)
        shown = []  # (the frames summed on screen, the newest frame handed out) after every look that brought frames
        for frames in range(6, 18, 2):
            _grow(path, frames)
            page.poll_watch()
            _done(page)
            if page.view_model.watch_grown:
                analysis = page.view_model.state.analysis
                shown.append(((analysis.frame_index, *(index for _path, index in analysis.summed_frames)),
                              page.view_model.followed_frame(path)))
        assert len(shown) >= 4
        assert all(summed == (newest - 2, newest - 1, newest) for summed, newest in shown)
        assert page.view_model.growing(path) and "sum of 3 frames" in page.file_meta.text()
    finally:
        page.stop_watch()
        _close(page)


def test_a_module_file_left_by_a_writer_that_ended_is_named_with_how_to_repair_it(tmp_path: Path) -> None:
    """Of a set of module files one was left unfinished by a writer that was killed. Once it has not changed for a
    while it is not waited for: the set is listed, never exported, and the Data step names that file and says how
    it can be repaired (not only that it "cannot be read yet")."""
    folder = tmp_path / "run"
    folder.mkdir()
    _modules([folder / "scan_00050_m01.nxs", folder / "scan_00050_m03.nxs"], 5)
    left = _killed(folder / "scan_00050_m02.nxs", H + 4)
    page = _page(_context())
    try:
        page.start_watch(folder)
        page._watch_timer.stop()
        page.auto_export_check.setChecked(True)
        _done(page)
        assert _rows(page) == ["scan_00050 · 5 frames"]  # just written: its writer may still be at work
        assert page.source_warning_label.toolTip() == ("Module file m02 cannot be read yet: that part of the detector "
                                                       "is left out")
        _long_ago(left)
        said = _statuses(page)
        for _look in range(3):
            page.poll_watch()
            _done(page)
        sentence = ("Module file m02 was left open for writing and has not changed for a while: the program that wrote "
                    "it seems to have ended without closing it, and that part of the detector is left out. Running "
                    "h5clear -s on a copy of it makes readable what was stored in it")
        assert page.source_warning_label.toolTip() == sentence and page.step_rail.state("data") == "warn"
        assert any(sentence in text and level == "warning" for text, level in said)
        assert page.view_model.watch_held and not (folder / "gimap_analysis").exists()  # every frame lacks m02
        apply_language("zh", [page])
        page.refresh_language()
        assert page.source_warning_label.toolTip().startswith("模块文件 m02 仍标记为“正在写入”")
        assert "h5clear -s" in page.source_warning_label.toolTip()
    finally:
        page.stop_watch()
        _close(page)


def test_a_damaged_module_file_holds_its_set_back_and_the_watch_says_what_is_wrong(tmp_path: Path) -> None:
    from tests.test_nexus_followups import _damage, _module

    paths = [_module(tmp_path / "src" / f"scan_m{module:02d}.nxs", module, frames=3) for module in (1, 2, 3)]
    if not _damage(paths[1]):
        pytest.skip("this HDF5 library reads, or refuses to open, every damage tried")
    folder = tmp_path / "run"
    folder.mkdir()
    model = _model()
    model.start_watch(folder)
    following = _Following(model, modules=3)
    for path in paths:
        (folder / path.name).write_bytes(path.read_bytes())
    following.rest(3)
    # Just written: it may be a copy that is not complete yet. The watch waits for it, and says what is wrong.
    assert model.state.files == [] and [(path.name, why) for path, why in model.watch_waiting] == [(
        "scan_m01.nxs", "scan_m02.nxs looks damaged or incomplete (the NeXus/HDF5 file could not be opened) — if it "
                        "is still being written, try again later.")]
    stat = (folder / "scan_m02.nxs").stat()
    os.utime(folder / "scan_m02.nxs", (stat.st_atime, stat.st_mtime - 10 * nexus_files.UNFINISHED_AFTER_S))
    following.rest(sources_model.SET_RESTS + 3)
    # Unchanged for a while: nothing to wait for. One row (the set, not three files) that says what it lacks,
    # and no frame of it is handed out: every frame lacks that part of the detector.
    assert [path.name for path in model.state.files] == ["scan_m01.nxs"] and model.watch_waiting == []
    assert following.handed == [] and [path.name for path in model.watch_held] == ["scan_m01.nxs"]
    layout = model.describe(model.state.files[0])
    assert layout.kind == "module_set" and [path.name for path in layout.pending] == ["scan_m02.nxs"]
    assert layout.warning_texts() == ("Module file m02 cannot be read yet: that part of the detector is left out",)


def test_modules_written_one_after_the_other_are_no_warning_while_the_set_grows(tmp_path: Path) -> None:
    """At most looks of a live run some module files hold a frame more than the others. That is how a detector
    writes, not a problem: the Data step says which frame is the newest complete one. A set that RESTS with
    unequal module files is warned of."""
    from tests.test_analyze_nexus_sources import _grow, _module

    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=4) for module in (1, 2)]
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        _done(page)
        said = _statuses(page)
        _grow(paths[0], 5)
        page.poll_watch()  # seen changed for the first time
        _done(page)
        _grow(paths[1], 6)
        _grow(paths[0], 7)
        page.poll_watch()  # changed again: it grows (7 and 6 frames; both held 4 at the look before: nothing new)
        _done(page)
        _grow(paths[1], 8)
        _grow(paths[0], 9)
        page.poll_watch()  # and again: 9 and 8 frames now, and the 6 that both held at the look before are handed out
        _done(page)
        layout = page.view_model.describe(paths[0])
        assert "different numbers of frames" in layout.warning_texts()[0] and page.view_model.growing(paths[0])
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"
        assert not any("different numbers of frames" in text for text, _level in said)
        assert ("scan: 2 new frames", "ok") in said
        assert page.source_label.text().splitlines()[-1] == "Still growing: frame 6 is the newest complete one"
        assert page.view_model.state.analysis.frame_index == 5 and page.frame_total_label.text() == "of 8"
        page.poll_watch()  # it rests, with 9 and 8 frames
        _done(page)
        unequal = "Module files hold different numbers of frames (m02: 8; the others 9): frames after frame 8 are left out"
        assert page.source_warning_label.text() == unequal and page.step_rail.state("data") == "warn"
        assert "Still growing" not in page.source_label.text() and page.view_model.state.analysis.frame_index == 7
        assert any(unequal in text for text, _level in said)  # now it is said: the set rests like that
    finally:
        page.stop_watch()
        _close(page)
