"""NeXus / HDF5 sources in Analyze: every layout becomes the right rows and frames, growth is followed,
and the page says what a source is.

Every file is synthetic (h5py, in ``tmp_path``) and goes through the real ``DetectorIoFrameSource`` and the
view model; the page is driven where the text on screen matters. The layouts: the module files of a detector
(one file per module, each with every frame), files that are only named like modules, a master with its
data files, a stack in an ``.h5`` file, a file of model weights, modules with different numbers of frames, a
module that is missing, a series that grows inside its file(s), a file its writer holds, and a scan folder
whose frames are one level down.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QThread

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application.ports import frame_label, parts_missing, source_name, source_stem
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.features.analyze.infrastructure.adapters import SUPPORTED_SUFFIXES, DetectorIoFrameSource
from src.gimap.features.analyze.infrastructure.adapters.frames import FrameReadError
from src.gimap.features.analyze.presentation import source_texts, sources_model
from src.gimap.features.analyze.presentation.bindings import batch_watch, data_step
from src.gimap.features.analyze.presentation.folder_watch import FolderWatch
from src.gimap.features.analyze.presentation.source_texts import FILE_FILTER, short_name
from src.gimap.shared.detector_io import DETECTOR_SUFFIXES, detect_nxs_frame_count, forget_layout, load_detector_image
from src.gimap.shared.detector_io import nexus_files, nexus_layout
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_assistant_calibration import save_tiff
from tests.test_review_analyze_run_and_clear import _profiles
from tests.test_series_map import _ring_frame

DET = "/entry/instrument/detector"
H, W = 24, 36


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    apply_language("en")
    yield
    apply_language("en")
    forget_layout()


# -- synthetic sources -------------------------------------------------------------------------------------------------


def _module(path: Path, module: int, frames: int = 2, libver=None) -> Path:
    """One detector module file as the Lambda server writes it: every frame, and where the module sits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver=libver) as handle:
        data = np.full((frames, H, W), 100 * module, dtype=np.int32) + np.arange(frames, dtype=np.int32)[:, None, None]
        handle.create_dataset(f"{DET}/data", data=data, maxshape=(None, H, W), chunks=(1, H, W))
        handle.create_dataset(f"{DET}/translation/distance", data=np.asarray([0.0, (module - 1) * (H + 4), 0.0], "f4"))
    return path


def _stack(path: Path, frames: int, dataset: str = f"{DET}/data", libver=None) -> Path:
    """One file that holds the whole series (frame k is filled with k). ``libver="latest"``: the file format in
    which a writer marks the file as open for writing (the Lambda server's)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver=libver) as handle:
        data = np.zeros((frames, H, W), dtype=np.int32) + np.arange(frames, dtype=np.int32)[:, None, None]
        handle.create_dataset(dataset, data=data, maxshape=(None, H, W), chunks=(1, H, W))
    return path


def _grow(path: Path, frames: int) -> None:
    """The writer appended frames (the dataset is longer now)."""
    with h5py.File(path, "a") as handle:
        data = handle[f"{DET}/data"]
        before = data.shape[0]
        data.resize(frames, axis=0)
        data[before:] = np.arange(before, frames, dtype=np.int32)[:, None, None] + int(data[0, 0, 0])


def _ring_stack(path: Path, frames: int) -> Path:
    """A stack of synthetic GIWAXS frames the instrument profile of ``_profiles`` fits (curves can be exported)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        first = _ring_frame(200.0, seed=0)
        data = handle.create_dataset("/entry/data/data", shape=(frames,) + first.shape, maxshape=(None,) + first.shape,
                                     dtype="f4", chunks=(1,) + first.shape)
        for index in range(frames):
            data[index] = _ring_frame(200.0 + 10 * index, seed=index)
    return path


def _grow_rings(path: Path, frames: int) -> None:
    with h5py.File(path, "a") as handle:
        data = handle["/entry/data/data"]
        before = data.shape[0]
        data.resize(frames, axis=0)
        for index in range(before, frames):
            data[index] = _ring_frame(200.0 + 10 * index, seed=index)


def _master(folder: Path, name: str, chunks=(3, 3, 2)) -> Path:
    """Dectris / NXmx: ``<name>_master.h5`` with links to ``<name>_data_00000k.h5``."""
    folder.mkdir(parents=True, exist_ok=True)
    master = folder / f"{name}_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        entry.create_group("instrument/detector/detectorSpecific").create_dataset(
            "pixel_mask", data=np.zeros((H, W), dtype=np.uint32))
        for number, count in enumerate(chunks, start=1):
            data_file = folder / f"{name}_data_{number:06d}.h5"
            with h5py.File(data_file, "w") as chunk:
                chunk.create_dataset("/entry/data/data", data=np.full((count, H, W), number, dtype=np.uint32))
            data[f"data_{number:06d}"] = h5py.ExternalLink(data_file.name, "/entry/data/data")
    return master


def _weights(path: Path) -> Path:
    """``model.save("x.h5")`` of Keras: no detector image in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["keras_version"], handle.attrs["backend"], handle.attrs["model_config"] = "2.11.0", "tensorflow", "{}"
        handle.create_dataset("model_weights/dense/dense/kernel:0", data=np.zeros((256, 128), dtype=np.float32))
    return path


_WRITER = """
    import sys
    import h5py
    path, mode = sys.argv[1], sys.argv[2]
    dataset = "/entry/instrument/detector/data"
    with h5py.File(path, "a" if mode == "append" else "w", libver="latest") as handle:
        if mode in ("new", "module", "swmr"):
            data = handle.create_dataset(dataset, shape=(6, 24, 36), maxshape=(None, 24, 36), dtype="i4", chunks=(1, 24, 36))
            data[:] = 1
            if mode == "module":  # the second module of a detector: where it sits
                handle.create_dataset("/entry/instrument/detector/translation/distance", data=[0.0, 28.0, 0.0])
            if mode == "swmr":  # a detector server: others may read while it writes
                handle.swmr_mode = True
        else:
            data = handle[dataset]
            data.resize(9, axis=0)
            data[6:] = 7
        handle.flush()
        print("holding", flush=True)
        sys.stdin.readline()   # holds the file until the test says so
        if mode != "swmr":
            data.resize(9, axis=0)
"""


def _writer(path: Path, mode: str) -> subprocess.Popen:
    """Another program that writes ``path`` and keeps it open (``new``: creates it with 6 frames; ``module``: the
    same with a module position; ``append``: adds frames to it; ``swmr``: creates it with 6 frames in the mode in
    which others may read meanwhile); ``_release`` lets it close the file, with 9 frames (``swmr``: the 6)."""
    process = subprocess.Popen([sys.executable, "-c", textwrap.dedent(_WRITER), str(path), mode],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "holding"
    return process


def _release(process: subprocess.Popen) -> None:
    process.communicate("\n", timeout=30)
    assert process.returncode == 0


def _names(paths) -> list[str]:
    return [Path(path).name for path in paths]


def _model(profiles=None):
    """The view model over the real frame source."""
    return create_analyze_view_model(_context(profiles))


def _count_opens(monkeypatch) -> list[str]:
    opened: list[str] = []
    real = h5py.File

    def counting(name, *args, **kwargs):
        opened.append(Path(str(name)).name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(nexus_files.h5py, "File", counting)
    return opened


def _rows(page) -> list[str]:
    return [page.file_list.item(row).text() for row in range(page.file_list.count())]


def _close(page) -> None:
    page.tasks.wait(60)
    page.dispose()
    page.close()


def _look_twice(model) -> list[Path]:
    """Two looks at the watched folder: what changed is reported once it has rested for one of them."""
    first = model.poll_watch()
    assert first == [] and model.watch_grown == []  # seen once: it may still be growing
    return model.poll_watch()


def _frames_of(requests) -> list[tuple[str, int]]:
    return [(request.path.name, request.frame_index) for request in requests]


def _rest(model, looks: int) -> list[tuple[str, int]]:
    """``looks`` more looks at the watched folder in which nothing arrives: the frames handed out meanwhile."""
    handed: list[tuple[str, int]] = []
    for _look in range(looks):
        handed += _frames_of(model.watch_requests(model.poll_watch()))
    return handed


def _later(path: Path, seconds: float = 2.0) -> None:
    """The file was written ``seconds`` later than its clock says (whatever the resolution of the file system's
    times, it is seen as written again)."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + int(seconds * 1e9)))


def _as_on_disk(analysis) -> bool:
    """The frame in memory holds the pixels its files hold now."""
    now = load_detector_image(analysis.path, frame_idx=analysis.frame_index)
    held = analysis.raw_data if analysis.raw_data is not None else analysis.data
    return held.shape == now.data.shape and bool(np.array_equal(held, np.asarray(now.data, dtype=held.dtype), equal_nan=True))


def _count_loads(monkeypatch) -> list[str]:
    """Every frame read from disk by the frame source, by file name."""
    loads: list[str] = []
    real = DetectorIoFrameSource.load

    def load(self, path, frame_index=0):
        loads.append(Path(path).name)
        return real(self, path, frame_index)

    monkeypatch.setattr(DetectorIoFrameSource, "load", load)
    return loads


def _damage(path: Path) -> bool:
    """Overwrite 16 bytes inside the first 2 kB of an HDF5 file (its metadata) so that the library opens it and
    then fails with an error that is no ``OSError`` (``RuntimeError: Link iteration failed …``), as a damaged or
    half-written file does. ``False``: this library read, or refused to open, every damage tried."""
    good = path.read_bytes()
    for start in range(96, 2048, 8):
        path.write_bytes(good[:start] + b"\xff" * 16 + good[start + 16:])
        forget_layout()
        try:
            detect_nxs_frame_count(path)
        except RuntimeError:
            forget_layout()
            return True
        except Exception:  # noqa: BLE001 - another kind of failure: try the next place
            continue
    path.write_bytes(good)
    forget_layout()
    return False


# -- listing: every layout becomes the right rows and frames ----------------------------------------------------------


def test_the_suffixes_and_the_open_filter_are_those_of_the_reader() -> None:
    assert SUPPORTED_SUFFIXES == DETECTOR_SUFFIXES and {".h5", ".hdf5", ".hdf", ".nx", ".nxs"} <= set(SUPPORTED_SUFFIXES)
    assert all(f"*{suffix}" in FILE_FILTER for suffix in DETECTOR_SUFFIXES) and FILE_FILTER.endswith(";;All files (*)")


def test_a_module_set_is_one_row_with_the_frames_every_module_has(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_00007_m{module:02d}.nxs", module, frames=5) for module in (1, 2, 3)]
    source = DetectorIoFrameSource()
    assert _names(source.expand([tmp_path])) == ["scan_00007_m01.nxs"]
    assert _names(source.expand([paths[2]])) == ["scan_00007_m01.nxs"]  # any module file named stands for the set
    model = _model()
    added = model.add_paths([tmp_path])
    assert _names(added) == ["scan_00007_m01.nxs"] and model.listed_frames() == 5 and model.frame_count(added[0]) == 5
    assert _frames_of(model.batch_requests()) == [("scan_00007_m01.nxs", index) for index in range(5)]
    layout = model.describe(added[0])
    assert layout.kind == "module_set" and layout.summary_text() == "3 module files · 5 frames · stitched"
    assert source.settled_size(added[0]) == sum(path.stat().st_size for path in paths)
    assert model.add_paths([paths[1]]) == [] and model.already_listed == [0]  # another module of it: the same row


def test_files_only_named_like_modules_are_one_row_each(tmp_path: Path) -> None:
    for index, name in enumerate(("film1.nxs", "film2.nxs", "film3.nxs")):
        _stack(tmp_path / name, 3 + index)
    model = _model()
    added = model.add_paths([tmp_path])
    assert _names(added) == ["film1.nxs", "film2.nxs", "film3.nxs"]
    assert [model.frame_count(path) for path in added] == [3, 4, 5] and model.listed_frames() == 12
    image = DetectorIoFrameSource().load(added[1], 2)  # its own data, not a neighbour's
    assert float(np.nanmax(image.data)) == 2.0 and image.metadata["frame_count"] == 4


def test_a_master_with_its_data_files_is_one_row(tmp_path: Path) -> None:
    master = _master(tmp_path, "run7", chunks=(3, 3, 2))
    source = DetectorIoFrameSource()
    assert _names(source.expand([tmp_path])) == ["run7_master.h5"]
    assert _names(source.expand([tmp_path / "run7_data_000002.h5"])) == ["run7_master.h5"]
    model = _model()
    model.add_paths([tmp_path])
    assert model.listed_frames() == 8 and model.describe(master).summary_text() == "master + 3 data files · 8 frames"
    assert source.settled_size(master) == sum(path.stat().st_size for path in tmp_path.iterdir())
    assert float(np.nanmax(source.load(master, 6).data)) == 3.0  # frame 7 is in the third data file


def test_a_stack_in_an_h5_file_is_listed_and_a_file_of_model_weights_is_not(tmp_path: Path) -> None:
    stack = _stack(tmp_path / "frames_of_a_series.h5", 4, dataset="/entry/data/data")
    weights = _weights(tmp_path / "weights.h5")
    cbf = tmp_path / "frame_00001.cbf"
    cbf.write_bytes(b"x" * 32)
    source = DetectorIoFrameSource()
    assert _names(source.expand([tmp_path])) == ["frame_00001.cbf", "frames_of_a_series.h5"]
    assert source.frame_count(stack) == 4 and source.describe(cbf) is None
    # Named by the person, the weights file is listed; reading it says what it is.
    assert _names(source.expand([weights])) == ["weights.h5"]
    with pytest.raises(FrameReadError, match=r"^weights\.h5 holds no detector image\.$"):
        source.load(weights)
    model = _model()
    model.add_paths([weights])
    assert model.frame_count(weights) == 1 and model.listed_frames() == 1  # no count: one row, never a crash


def test_modules_with_different_numbers_of_frames_give_the_frames_all_of_them_have(tmp_path: Path) -> None:
    for module, frames in ((1, 5), (2, 5), (3, 3)):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=frames)
    model = _model()
    (path,) = model.add_paths([tmp_path])
    layout = model.describe(path)
    assert model.frame_count(path) == 3 and len(model.all_frames()) == 3 and not layout.complete
    assert source_texts.warning_texts(layout, notes=False) == [
        "Module files hold different numbers of frames (m03: 3; the others 5): frames after frame 3 are left out"]
    assert not parts_missing(layout) and model.whole_frames(path) == 3  # the three frames are whole
    with pytest.raises(FrameReadError, match=r"scan_m01\.nxs has fewer frames than asked for \(frame 4 of 3\)\."):
        DetectorIoFrameSource().load(path, 3)


def test_a_missing_module_is_said_and_its_frames_are_not_whole(tmp_path: Path) -> None:
    for module in (1, 2, 4):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=4)
    model = _model()
    (path,) = model.add_paths([tmp_path])
    layout = model.describe(path)
    assert source_texts.warning_texts(layout) == ["Module file m03 is missing: that part of the detector is left out"]
    assert parts_missing(layout) and model.frame_count(path) == 4 and model.whole_frames(path) == 0


def test_a_scan_folder_whose_frames_are_one_level_down_is_not_empty(tmp_path: Path) -> None:
    scan = tmp_path / "scan_00042"
    for module in (1, 2):
        _module(scan / "lmbdp03" / f"scan_00042_m{module:02d}.nxs", module, frames=3)
    (scan / "pilatus").mkdir()
    (scan / "pilatus" / "scan_00042_00001.cbf").write_bytes(b"x" * 32)
    (scan / "gimap_analysis").mkdir()
    _stack(scan / "gimap_analysis" / "exported_frame.h5", 1)  # what GIMaP wrote itself is no frame of the scan
    (scan / ".hidden").mkdir()
    _stack(scan / ".hidden" / "cache.h5", 1)
    source = DetectorIoFrameSource()
    found = source.expand([scan])
    assert [(path.parent.name, path.name) for path in found] == [
        ("lmbdp03", "scan_00042_m01.nxs"), ("pilatus", "scan_00042_00001.cbf")]
    assert source.expand([tmp_path]) == []  # two levels down: not searched
    # A folder with frames of its own is not searched further down.
    _stack(scan / "overview.h5", 2)
    assert _names(source.expand([scan])) == ["overview.h5"]


def test_a_folder_is_listed_once_and_each_file_is_opened_once(tmp_path: Path, monkeypatch) -> None:
    for scan in range(1, 41):
        for module in (1, 2, 3):
            _module(tmp_path / f"scan_{scan:05d}_m{module:02d}.nxs", module, frames=2)
    listings: list[str] = []
    real_listdir = os.listdir
    monkeypatch.setattr(nexus_files.os, "listdir", lambda folder: listings.append(str(folder)) or real_listdir(folder))
    opened = _count_opens(monkeypatch)
    source = DetectorIoFrameSource()
    started = time.perf_counter()
    found = source.expand([tmp_path])
    assert len(found) == 40 and time.perf_counter() - started < 5.0
    assert len(opened) == 120 and len(set(opened)) == 120 and len(listings) <= 2  # no file twice, no listing per file
    del opened[:], listings[:]
    assert len(source.expand([tmp_path])) == 40 and opened == []  # asked again (the watch does, every look): nothing is opened
    assert [source.frame_count(path) for path in found] == [2] * 40 and opened == []


# -- the frame count follows the file ---------------------------------------------------------------------------------


def test_the_frame_count_follows_a_file_that_grows(tmp_path: Path) -> None:
    path = _stack(tmp_path / "insitu_00001.nxs", 6)
    model = _model()
    model.add_paths([path])
    assert model.frame_count(path) == 6 and model.listed_frames() == 6
    _grow(path, 12)
    assert model.frame_count(path) == 12 and model.listed_frames() == 12 and len(model.batch_requests()) == 12


def test_a_read_that_fails_keeps_the_last_count_and_is_never_remembered(tmp_path: Path) -> None:
    listed = _stack(tmp_path / "listed_00001.nxs", 6, libver="latest")
    model = _model()
    model.add_paths([listed])
    assert model.frame_count(listed) == 6
    writer = _writer(listed, "append")
    try:
        with pytest.raises(FrameReadError) as caught:
            DetectorIoFrameSource().load(listed, 0)
        assert str(caught.value) == "listed_00001.nxs is being written by another program and cannot be read until it is closed."
        assert model.frame_count(listed) == 6 and model.listed_frames() == 6  # the last good count, not 1
        assert model.describe(listed).frames == 6  # as it was when it was last read
        assert DetectorIoFrameSource().frame_count(listed) == 6  # the frame source itself keeps it (not only the model)
    finally:
        _release(writer)
    assert model.frame_count(listed) == 9 and model.listed_frames() == 9

    fresh = tmp_path / "fresh_00002.nxs"
    writer = _writer(fresh, "new")  # first met while its writer holds it
    try:
        model.add_paths([fresh])
        assert model.frame_count(fresh) == 1 and model._frame_counts.get(str(fresh).casefold()) is None
        assert model.describe(fresh) is None and model.whole_frames(fresh) == 0
    finally:
        _release(writer)
    assert model.frame_count(fresh) == 9 and model.listed_frames() == 18


# -- the folder watch ----------------------------------------------------------------------------------------------------


def test_the_watch_reports_frames_added_to_a_listed_file(tmp_path: Path) -> None:
    path = _stack(tmp_path / "insitu_00001.nxs", 6)
    model = _model()
    model.start_watch(tmp_path)
    assert _names(model.state.files) == ["insitu_00001.nxs"]
    assert model.poll_watch() == [] and model.watch_grown == [] and model.watch_requests() == []  # what was there is not new
    _grow(path, 12)
    assert _look_twice(model) == []  # no new file …
    assert [(source.name, count) for source, count in model.watch_grown] == [("insitu_00001.nxs", 6)]  # … six new frames
    assert _frames_of(model.watch_requests()) == [("insitu_00001.nxs", index) for index in range(6, 12)]
    assert model.frame_count(path) == 12 and model.listed_frames() == 12
    assert model.poll_watch() == [] and model.watch_grown == [] and model.watch_requests() == []  # each frame once


def test_new_frames_join_the_groups_of_a_sum_like_new_files(tmp_path: Path) -> None:
    path = _stack(tmp_path / "insitu_00001.nxs", 2)
    model = _model()
    model.start_watch(tmp_path)
    model.set_sum_count(4)
    _grow(path, 7)
    _look_twice(model)
    requests = model.watch_requests()  # frames 3–7 are new: one group of four, the fifth waits for three more
    assert [(request.frame_index, [index for _path, index in request.summed_frames]) for request in requests] == [(2, [3, 4, 5])]
    _grow(path, 10)
    _look_twice(model)
    requests = model.watch_requests()
    assert [(request.frame_index, [index for _path, index in request.summed_frames]) for request in requests] == [(6, [7, 8, 9])]


def test_frames_found_while_nothing_is_exported_are_not_exported_later(tmp_path: Path) -> None:
    path = _stack(tmp_path / "insitu_00001.nxs", 2)
    model = _model()
    model.start_watch(tmp_path)
    _grow(path, 4)
    _look_twice(model)
    assert model.watch_requests(wanted=False) == []  # Export New Frames While Watching is off
    _grow(path, 6)
    _look_twice(model)
    assert _frames_of(model.watch_requests()) == [("insitu_00001.nxs", 4), ("insitu_00001.nxs", 5)]


def test_a_module_set_that_grows_in_every_module_is_followed(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=6) for module in (1, 2, 3)]
    model = _model()
    model.start_watch(tmp_path)
    assert _names(model.state.files) == ["scan_m01.nxs"] and model.listed_frames() == 6
    for path in paths:
        _grow(path, 12)
    assert _look_twice(model) == []
    assert [(source.name, count) for source, count in model.watch_grown] == [("scan_m01.nxs", 6)]
    assert _frames_of(model.watch_requests()) == [("scan_m01.nxs", index) for index in range(6, 12)]
    assert model.listed_frames() == 12 and len(model.state.files) == 1


def test_modules_that_grow_one_after_the_other_give_only_frames_every_module_has(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=6) for module in (1, 2, 3)]
    model = _model()
    source = DetectorIoFrameSource()
    model.start_watch(tmp_path)
    (listed,) = model.state.files
    handed: list[tuple[str, int]] = []
    for step, path in enumerate(paths):
        _grow(path, 12)
        assert _look_twice(model) == [] and len(model.state.files) == 1
        requests = model.watch_requests()
        for request in requests:  # whatever is handed out is whole: every module file is in the frame
            assert len(source.load(request.path, request.frame_index).metadata["module_files"]) == 3
        handed += _frames_of(requests)
        if step < 2:  # a module still has six frames: the six are all the set holds
            assert requests == [] and model.frame_count(listed) == 6 and model.watch_grown == []
            assert "different numbers of frames" in source_texts.warning_texts(model.describe(listed))[0]
    assert handed == [("scan_m01.nxs", index) for index in range(6, 12)]  # each new frame once, none before it was whole
    assert model.poll_watch() == [] and model.watch_requests() == []


def test_a_set_that_arrives_module_by_module_is_one_row_and_is_exported_whole(tmp_path: Path) -> None:
    model = _model()
    source = DetectorIoFrameSource()
    model.start_watch(tmp_path)
    assert model.poll_watch() == []
    _module(tmp_path / "scan_m02.nxs", 2, frames=4)
    _module(tmp_path / "scan_m03.nxs", 3, frames=4)
    assert model.poll_watch() == []
    added = model.poll_watch()
    assert _names(added) == ["scan_m02.nxs"] and _names(model.state.files) == ["scan_m02.nxs"]
    assert model.watch_requests(added) == [] and _names(model.watch_held) == ["scan_m02.nxs"]  # m01 is missing: it waits
    assert _rest(model, sources_model.SET_RESTS + 2) == []  # … however long it rests: a part of the detector is missing
    assert _names(model.watch_held) == ["scan_m02.nxs"] and model.whole_frames(model.state.files[0]) == 0
    _module(tmp_path / "scan_m01.nxs", 1, frames=4)
    assert model.poll_watch() == []
    assert model.poll_watch() == []  # not a second row: the row of the set takes the name of its first module
    assert _names(model.state.files) == ["scan_m01.nxs"]
    assert [(row, before.name, now.name) for row, before, now in model.replaced] == [(0, "scan_m02.nxs", "scan_m01.nxs")]
    # No module is missing any more, but more may follow: the set rests before its frames are handed out.
    assert model.watch_requests() == [] and _names(model.watch_held) == ["scan_m01.nxs"]
    handed = _rest(model, sources_model.SET_RESTS)
    assert handed == [("scan_m01.nxs", index) for index in range(4)] and model.watch_held == []
    assert all(len(source.load(tmp_path / name, index).metadata["module_files"]) == 3 for name, index in handed)
    assert model.listed_frames() == 4 and model.poll_watch() == [] and model.watch_requests() == []
    # Opening the folder again, or any of its module files, adds no row either.
    assert model.add_paths([tmp_path]) == [] and model.add_paths([tmp_path / "scan_m03.nxs"]) == []
    assert len(model.state.files) == 1


def test_a_listed_set_whose_first_module_arrives_later_keeps_its_row(tmp_path: Path) -> None:
    _module(tmp_path / "scan_m02.nxs", 2, frames=4)
    _module(tmp_path / "scan_m03.nxs", 3, frames=4)
    model = _model()
    assert _names(model.add_paths([tmp_path])) == ["scan_m02.nxs"]
    _module(tmp_path / "scan_m01.nxs", 1, frames=4)
    assert model.add_paths([tmp_path]) == [] and model.already_listed == [0]
    assert _names(model.state.files) == ["scan_m01.nxs"] and len(model.replaced) == 1
    assert model.describe(model.state.files[0]).complete and model.listed_frames() == 4


def test_a_file_its_writer_holds_waits_and_is_listed_once_it_is_closed(tmp_path: Path) -> None:
    model = _model()
    model.start_watch(tmp_path)
    path = tmp_path / "live_00001.nxs"
    writer = _writer(path, "new")
    try:
        for _look in range(3):  # however long it rests: it cannot be read, so it is not listed
            assert model.poll_watch() == [] and model.state.files == []
        assert [(waiting.name, why) for waiting, why in model.watch_waiting] == [
            ("live_00001.nxs", "live_00001.nxs is being written by another program and cannot be read until it is closed.")]
    finally:
        _release(writer)
    assert model.poll_watch() == []
    added = model.poll_watch()  # the next looks try again
    assert _names(added) == ["live_00001.nxs"] and model.watch_waiting == []
    assert model.frame_count(added[0]) == 9 and len(model.watch_requests(added)) == 9


def test_a_module_still_being_written_holds_the_frames_of_its_set_back(tmp_path: Path) -> None:
    model = _model()
    source = DetectorIoFrameSource()
    model.start_watch(tmp_path)
    _module(tmp_path / "scan_m01.nxs", 1, frames=9)
    writer = _writer(tmp_path / "scan_m02.nxs", "module")
    try:
        assert model.poll_watch() == []
        added = model.poll_watch()
        assert _names(added) == ["scan_m01.nxs"]  # what can be read is listed, and says what it lacks …
        assert source_texts.warning_texts(model.describe(added[0])) == [
            "Module file m02 cannot be read yet: that part of the detector is left out"]
        assert model.watch_requests(added) == [] and _names(model.watch_held) == ["scan_m01.nxs"]  # … nothing is exported
        assert _names(path for path, _why in model.watch_waiting) == ["scan_m02.nxs"]
        assert model.poll_watch() == [] and len(model.state.files) == 1 and model.watch_requests() == []
    finally:
        _release(writer)
    assert _look_twice(model) == [] and len(model.state.files) == 1  # the module joins the row that is there
    handed = _frames_of(model.watch_requests()) + _rest(model, sources_model.SET_RESTS)  # (and the set has rested)
    assert handed == [("scan_m01.nxs", index) for index in range(9)] and model.watch_held == []
    assert all(len(source.load(tmp_path / name, index).metadata["module_files"]) == 2 for name, index in handed)
    assert model.describe(model.state.files[0]).complete and model.poll_watch() == [] and model.watch_requests() == []


def test_a_module_held_again_by_its_writer_does_not_bring_the_frames_handed_out_again(tmp_path: Path) -> None:
    """A set whose frames were handed out; then a writer appends to one of its module files and holds it. The
    set waits for that file, and afterwards only the frames that are new are handed out: the others were
    stitched from these very files."""
    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=6, libver="latest") for module in (1, 2, 3)]
    model = _model()
    model.start_watch(tmp_path)
    _grow(paths[0], 9)
    _grow(paths[2], 9)
    writer = _writer(paths[1], "append")  # (adds its frames 7–9 and keeps the file open)
    try:
        assert _rest(model, 3) == [] and [(waiting.name, why) for waiting, why in model.watch_waiting] == [
            ("scan_m01.nxs", "scan_m02.nxs is being written by another program and cannot be read until it is closed.")]
        assert parts_missing(model.describe(paths[0])) and model.whole_frames(paths[0]) == 0
    finally:
        _release(writer)
    assert _rest(model, 2) == [("scan_m01.nxs", index) for index in (6, 7, 8)]  # at once, and only the new ones
    assert model.watch_held == [] and model.watch_waiting == [] and model.settling(paths[0]) is None


def test_the_watch_looks_at_the_folder_without_changing_anything(tmp_path: Path) -> None:
    """``scan`` only reads (it runs on another thread); ``take`` applies it, and never an older look."""
    source = DetectorIoFrameSource()
    watch = FolderWatch(source.expand, source.settled_size)
    assert watch.scan() is None and watch.take(None) == []
    watch.start(tmp_path)
    (tmp_path / "frame_00001.cbf").write_bytes(b"x" * 10)
    first, second, third = watch.scan(), watch.scan(), watch.scan()
    assert watch.take(second) == [] and watch.take(first) == []  # the older look is dropped
    assert _names(watch.take(third)) == ["frame_00001.cbf"]
    stale = watch.scan()
    watch.start(tmp_path / "elsewhere")
    assert watch.take(stale) == []  # a look at the folder watched before
    watch.start(tmp_path)
    (tmp_path / "frame_00001.cbf").write_bytes(b"x" * 30)  # a known file written again
    assert watch.poll() == [] and watch.changed == []
    assert watch.poll() == [] and _names(watch.changed) == ["frame_00001.cbf"]
    assert watch.poll() == [] and watch.changed == []


# -- naming ------------------------------------------------------------------------------------------------------------


def test_a_module_set_is_named_by_its_scan_and_frames_count_from_one(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_00007_m{module:02d}.nxs", module) for module in (1, 2)]
    stack = _stack(tmp_path / "film_m1.nxs", 2)  # named like a module, read on its own
    source = DetectorIoFrameSource()
    layout = source.describe(paths[0])
    assert source_stem(paths[0], layout) == "scan_00007" and source_name(paths[0], layout) == "scan_00007"
    assert source_stem(paths[0], source.load(paths[0]).metadata) == "scan_00007"  # the frame's metadata says it too
    assert source_stem(stack, source.describe(stack)) == "film_m1" and source_name(stack, source.describe(stack)) == "film_m1.nxs"
    assert source_stem(paths[0]) == "scan_00007_m01" and source_name(Path("frame_00012.cbf")) == "frame_00012.cbf"
    assert frame_label(3, 12) == "frame 4 of 12" and source_texts.frame_text(3, 12) == "frame 4 of 12"
    assert short_name("a" * 20) == "a" * 20 and short_name("jg_gisaxs_12nm_insitu_00001r3_00001_m01.nxs") == (
        "jg_gisaxs_12nm…3_00001_m01.nxs")
    assert len(short_name("x" * 80)) == source_texts.NAME_LIMIT


# -- the page ----------------------------------------------------------------------------------------------------------

ZH_PINS = {
    "summary": "3 个模块文件 · 5 帧 · 已拼接",
    "orientation": "P03 Lambda：转置并上下翻转（即旋转 90°）",
    "unequal": "各模块文件的帧数不同（m03: 3；其余为 5）：第 3 帧之后的帧不使用",
}


def test_the_data_step_says_what_a_module_set_is(tmp_path: Path) -> None:
    for module in (1, 2, 3):
        _module(tmp_path / f"scan_00007_m{module:02d}.nxs", module, frames=5)
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        assert _rows(page) == ["scan_00007 · 5 frames"]  # one row, named by the scan, with its frames
        tip = page.file_list.item(0).toolTip().splitlines()
        assert tip[0].endswith("scan_00007_m01.nxs") and tip[1] == "3 module files · 5 frames · stitched"
        assert tip[2] == "scan_00007_m01.nxs, scan_00007_m02.nxs, scan_00007_m03.nxs"
        assert page.file_chip.full_text() == "scan_00007" and page.file_chip.toolTip().endswith("scan_00007_m01.nxs")
        assert page.source_label.text().splitlines() == [
            "3 module files · 5 frames · stitched", "P03 Lambda: transposed and flipped (rotated by 90°)"]
        assert not page.source_label.isHidden() and page.source_warning_label.isHidden()
        assert page.file_meta.text().endswith("frame 1 of 5") and page.step_rail.state("data") == "ok"
        assert not page.frame_bar.isHidden() and page.frame_spin.maximum() == 5
        assert page.frame_total_label.text() == "of 5" and page.data_batch_button.text() == "Batch Export 5 Frames…"
        assert page.file_position_label.text() == "frame 1 / 5"  # beside ‹ › (which step its frames)

        apply_language("zh", [page])
        page.refresh_language()
        assert page.file_position_label.text() == "第 1 / 5 帧"
        assert _rows(page) == ["scan_00007 · 5 帧"] and page.file_list.item(0).toolTip().splitlines()[1] == ZH_PINS["summary"]
        assert page.source_label.text().splitlines() == [ZH_PINS["summary"], ZH_PINS["orientation"]]
        assert page.frame_total_label.text() == "共 5 帧" and page.file_meta.text().endswith("第 1 帧，共 5 帧")
        apply_language("en", [page])
        page.refresh_language()
        assert _rows(page) == ["scan_00007 · 5 frames"] and page.frame_total_label.text() == "of 5"
    finally:
        _close(page)


def test_the_data_step_warns_of_modules_with_different_numbers_of_frames(tmp_path: Path) -> None:
    for module, frames in ((1, 5), (2, 5), (3, 3)):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=frames)
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        warning = "Module files hold different numbers of frames (m03: 3; the others 5): frames after frame 3 are left out"
        assert _rows(page) == ["scan · 3 frames"] and page.frame_spin.maximum() == 3
        assert page.source_warning_label.text() == warning and not page.source_warning_label.isHidden()
        assert page.source_warning_label.property("gimapRole") == "warning"
        assert page.step_rail.state("data") == "warn" and warning in page.file_list.item(0).toolTip()
        assert warning in page.status_text() and page.status_level() == "warning"  # said with the frame, too

        apply_language("zh", [page])
        page.refresh_language()
        assert page.source_warning_label.text() == ZH_PINS["unequal"] and ZH_PINS["unequal"] in page.status_text()
    finally:
        _close(page)


def test_the_data_step_warns_of_a_missing_module(tmp_path: Path) -> None:
    for module in (1, 2, 4):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=2)
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        warning = "Module file m03 is missing: that part of the detector is left out"
        assert page.source_warning_label.text() == warning and page.step_rail.state("data") == "warn"
        assert page.source_label.text().splitlines()[0] == "3 module files · 2 frames · stitched"
        assert warning in page.banner_label.text()  # no profile fits the smaller frame: the banner says why
        apply_language("zh", [page])
        page.refresh_language()
        assert page.source_warning_label.text() == "缺少模块文件 m03：探测器的这一部分没有数据"
    finally:
        _close(page)


def test_files_read_on_their_own_say_so_without_a_warning_state(tmp_path: Path) -> None:
    """Lambda files that are only named like modules: a note with the source, no warning at every frame."""
    for name in ("film1.nxs", "film2.nxs"):
        _module(tmp_path / name, 1, frames=2)  # the same module position in both: two measurements
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        assert _rows(page) == ["film1.nxs · 2 frames", "film2.nxs · 2 frames"]
        note = page.source_label.text().splitlines()[-1]
        assert "named like modules of one detector" in note and "film1.nxs" in note and "film2.nxs" in note
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"
        assert page.status_level() != "warning" or "named like" not in page.status_text()
        card = page.data_info_label.text().splitlines()
        assert "Files listed: 2" in card and "Frames listed: 4" in card  # the total the batch works on
        assert page.data_batch_button.text() == "Batch Export 4 Frames…"
    finally:
        _close(page)


def test_a_long_file_name_does_not_push_the_buttons_out_of_the_data_step(tmp_path: Path) -> None:
    name = "jg_gisaxs_12nm_insitu_run_0001_sample_A_annealed_at_453K_00017.nxs"
    assert len(name) > 60
    _stack(tmp_path / name, 12)
    page = _page(_context())
    try:
        page.resize(1090, 700)  # the Analyze page of a 1280×800 window
        page.show()
        page.add_paths([str(tmp_path)])
        _done(page)
        _app().processEvents()
        viewport = page.step_pages["data"].viewport()
        content = page.step_pages["data"].widget()
        assert content.width() <= viewport.width()  # nothing is wider than the panel
        for button in (page.watch_button, page.clear_button):
            assert button.visibleRegion().boundingRect().width() == button.width() > 0
            assert button.mapTo(viewport, button.rect().topRight()).x() < viewport.width()
        assert page.frame_spin.visibleRegion().boundingRect().width() == page.frame_spin.width() > 0
        intro = page.step_intro["data"].text()
        assert name not in intro and intro.startswith(short_name(name) + ":")  # shortened in the sentence …
        assert page.file_chip.full_text() == name and name in page.file_list.item(0).toolTip()  # … whole elsewhere
        assert page.file_list.item(0).text() == f"{name} · 12 frames"  # (the list shortens it in the middle itself)
        # Whatever else the sentence holds that cannot break (a detector name from a file header, a larger
        # font): it is cut at the edge of the panel and never widens it.
        page.step_intro["data"].setText(f"{short_name(name)}: {'Lambda_9M_GaAs_' * 5}, 24×36 px.")
        for _turn in range(3):
            _app().processEvents()
        assert page.step_intro["data"].minimumSizeHint().width() > viewport.width()
        assert content.width() <= viewport.width()
        assert page.clear_button.mapTo(viewport, page.clear_button.rect().topRight()).x() < viewport.width()
    finally:
        _close(page)


def test_the_stepper_and_the_keys_step_through_the_frames_of_the_series(tmp_path: Path) -> None:
    first = _stack(tmp_path / "run_1.nxs", 3)
    second = _stack(tmp_path / "run_2.nxs", 2)
    page = _page(_context())
    try:
        page.resize(1600, 900)
        page.show()
        page.add_paths([str(first)])
        _done(page)
        page.command_bar._apply(False)
        page._show_file_position()
        label = page.file_position_label
        assert page.frame_listing_model.rowCount() == 3 and page.file_list.count() == 1
        assert [page.frame_listing_model.ref(row) for row in range(3)] == [(first, index) for index in range(3)]
        # One file of several frames: the stepper is there, steps frames and says which one is shown.
        assert not page.next_file_button.isHidden() and not page.previous_file_button.isHidden()
        assert label.text() == "frame 1 / 3" and not label.isHidden()
        assert label.toolTip().startswith("Frame 1 of 3 of run_1.nxs")
        assert page.next_file_button.toolTip() == "Next frame (Page Down)"
        assert page.next_file_button.isEnabled() and not page.previous_file_button.isEnabled()
        page.next_file_button.click()
        _done(page)
        assert page.view_model.state.frame_index == 1 and page.frame_spin.value() == 2 and label.text() == "frame 2 / 3"
        assert page.file_meta.text().endswith("frame 2 of 3") and page.view_model.state.analysis.frame_index == 1
        assert page.frame_list.currentIndex().row() == 1
        assert page.view_model.state.analysis.frame_count == 3 and _as_on_disk(page.view_model.state.analysis)

        page.add_paths([str(second)], show_listed=False)
        _done(page)
        page.show_frame(0, 2)
        _done(page)
        # The visible list and stepper count every imported frame, while the source list remains deduplicated.
        assert page.file_list.count() == len(page.view_model.state.files) == 2
        assert page.frame_listing_model.rowCount() == page.view_model.listed_frames() == 5
        assert [page.frame_listing_model.ref(row) for row in range(5)] == [
            *((first, index) for index in range(3)), *((second, index) for index in range(2))]
        assert label.text() == "3 / 5" and page.file_meta.text().endswith("frame 3 of 3")
        assert page.frame_list.currentIndex().row() == 2
        assert label.toolTip().startswith("Frame 3 of 3 of run_1.nxs")
        page._step_frame(+1)  # Page Down at the last frame of a file: the next file, at its first frame
        _done(page)
        assert (page.file_list.currentRow(), page.view_model.state.frame_index) == (1, 0)
        assert label.text() == "4 / 5" and page.file_meta.text().endswith("frame 1 of 2")
        assert page.frame_list.currentIndex().row() == 3
        assert page.view_model.state.analysis.frame_count == 2 and _as_on_disk(page.view_model.state.analysis)
        page._step_frame(-1)  # Page Up at the first frame: the file before, at its last frame
        _done(page)
        assert (page.file_list.currentRow(), page.view_model.state.frame_index) == (0, 2)
        assert label.text() == "3 / 5" and page.frame_list.currentIndex().row() == 2
        assert page.view_model.state.analysis.path.name == "run_1.nxs" and page.view_model.state.analysis.frame_index == 2
        page._step_frame(-1)
        page._step_frame(-1)
        page._step_frame(-1)  # nothing before the first frame of the first file
        _done(page)
        assert (page.file_list.currentRow(), page.view_model.state.frame_index) == (0, 0)
        assert label.text() == "1 / 5" and page.frame_list.currentIndex().row() == 0
        assert not page.previous_file_button.isEnabled() and page.next_file_button.isEnabled()
        apply_language("zh", [page])
        page.refresh_language()
        assert label.text() == "1 / 5" and label.toolTip().startswith("run_1.nxs 的第 1 帧，共 3 帧")
        assert page.next_file_button.toolTip() == "下一帧（Page Down）" and page.file_meta.text().endswith("第 1 帧，共 3 帧")
    finally:
        _close(page)


def test_the_frame_bar_stays_visible_on_every_step(tmp_path: Path) -> None:
    """Frames are stepped where the work is done: the bar is above the step pages, not inside the Data step."""
    _stack(tmp_path / "run_1.nxs", 4)
    single = save_tiff(tmp_path / "single.tif", np.ones((16, 16), dtype=np.float32))
    page = _page(_context())
    try:
        page.resize(1090, 700)
        page.show()
        page.add_paths([str(tmp_path / "run_1.nxs")])
        _done(page)
        for step in ("data", "geometry", "mask", "cuts", "results", "export"):
            page.show_step(step)
            _app().processEvents()
            assert page.frame_bar.isVisible() and page.frame_spin.isVisible(), step
            assert not page.step_pages[step].isAncestorOf(page.frame_bar)
        assert page.frame_total_label.text() == "of 4" and page.frame_spin.toolTip().startswith("The frame shown")
        page.frame_spin.setValue(3)  # on the Export step: the frame changes all the same
        _done(page)
        assert page.view_model.state.analysis.frame_index == 2 and page.file_meta.text().endswith("frame 3 of 4")
        page.add_paths([str(single)])
        _done(page)
        assert not page.frame_bar.isVisible()  # one frame: nothing to step
    finally:
        _close(page)


def test_a_frame_shown_again_knows_the_frames_added_since(tmp_path: Path) -> None:
    """F5 on the frame shown keeps the frame in memory; the count beside it is the file's count now."""
    path = _stack(tmp_path / "insitu_00001.nxs", 6)
    page = _page(_context())
    try:
        page.add_paths([str(path)])
        _done(page)
        assert page.frame_spin.maximum() == 6 and _rows(page) == ["insitu_00001.nxs · 6 frames"]
        _grow(path, 9)
        page.refresh_language()  # what is said of the frame is written again (no frame is read): the count is that of now
        assert "Frames in the file: 9 (showing 1)" in page.data_info_label.text() and page.file_meta.text().endswith("frame 1 of 9")
        assert page.view_model.state.analysis.frame_count == 6  # (the frame in memory was read when there were six)
        _grow(path, 12)
        page.run_analysis()  # F5
        _done(page)
        assert page.frame_spin.maximum() == 12 and page.frame_total_label.text() == "of 12"
        assert page.file_meta.text().endswith("frame 1 of 12") and page.data_batch_button.text() == "Batch Export 12 Frames…"
        assert "Frames in the file: 12 (showing 1)" in page.data_info_label.text()
    finally:
        _close(page)


def test_watching_a_folder_that_holds_data_shows_its_first_frame(tmp_path: Path) -> None:
    _stack(tmp_path / "run_1.nxs", 3)
    _stack(tmp_path / "run_2.nxs", 3)
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        _done(page)
        assert _rows(page) == ["run_1.nxs · 3 frames", "run_2.nxs · 3 frames"] and page.file_list.currentRow() == 0
        analysis = page.view_model.state.analysis
        assert analysis is not None and (analysis.path.name, analysis.frame_index) == ("run_1.nxs", 0)
        assert not page.frame_bar.isHidden() and page.data_batch_button.text() == "Batch Export 6 Frames…"
        page.stop_watch()
        page.file_list.setCurrentRow(1)
        _done(page)
        page.start_watch(tmp_path)  # a frame is on screen: it stays
        _done(page)
        assert page.file_list.currentRow() == 1 and page.file_list.count() == 2
    finally:
        _close(page)


def test_watching_follows_frames_added_to_a_file_and_exports_each_once(tmp_path: Path) -> None:
    path = _ring_stack(tmp_path / "insitu_00001.h5", 2)
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.start_watch(tmp_path)
        _done(page)
        page.auto_export_check.setChecked(True)
        assert page.poll_watch() == [] and page.view_model.state.analysis.frame_count == 2
        _grow_rings(path, 5)
        assert page.poll_watch() == []
        assert page.poll_watch() == []  # no new file: three new frames of the listed one
        assert page.status_text() == "insitu_00001.h5: 3 new frames" and page.status_level() == "ok"
        _done(page)
        exported = tmp_path / "gimap_analysis"
        names = sorted(found.name for found in exported.rglob("*_radial.csv"))
        assert names == [f"insitu_00001_frame{index:04d}_radial.csv" for index in (2, 3, 4)]  # the frames from before stay out
        # The newest frame is shown, and everything that counts frames knows the five.
        analysis = page.view_model.state.analysis
        assert (analysis.frame_index, analysis.frame_count) == (4, 5) and page.frame_spin.value() == 5
        assert _rows(page) == ["insitu_00001.h5 · 5 frames"] and page.frame_total_label.text() == "of 5"
        assert page.data_batch_button.text() == "Batch Export 5 Frames…"
        stamps = {found.name: found.stat().st_mtime_ns for found in exported.rglob("*_radial.csv")}
        assert page.poll_watch() == [] and page.poll_watch() == []
        _done(page)
        assert {found.name: found.stat().st_mtime_ns for found in exported.rglob("*_radial.csv")} == stamps  # none twice
        _grow_rings(path, 6)
        page.poll_watch()
        page.poll_watch()
        assert page.status_text() == "insitu_00001.h5: 1 new frame"
        _done(page)
        assert len(list(exported.rglob("*_radial.csv"))) == 4
    finally:
        page.stop_watch()
        _close(page)


def test_a_set_with_a_module_missing_is_shown_but_not_exported_until_it_is_whole(tmp_path: Path, monkeypatch) -> None:
    page = _page(_context())
    exported: list[tuple[str, int, int]] = []
    source = DetectorIoFrameSource()

    def export(request, destination, **_options):
        modules = len(source.load(request.path, request.frame_index).metadata["module_files"])
        exported.append((request.path.name, request.frame_index, modules))
        return []

    monkeypatch.setattr(page.view_model, "analyze_and_export", export)
    try:
        page.start_watch(tmp_path)
        page.auto_export_check.setChecked(True)
        _module(tmp_path / "scan_m02.nxs", 2, frames=2)
        _module(tmp_path / "scan_m03.nxs", 3, frames=2)
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert _rows(page) == ["scan · 2 frames"] and exported == []  # listed and shown, with what is missing …
        missing = "Module file m01 is missing: that part of the detector is left out"
        assert page.source_warning_label.text() == missing and page.step_rail.state("data") == "warn"
        assert page.status_level() == "warning" and missing in page.status_text()  # … and why nothing is exported yet
        _module(tmp_path / "scan_m01.nxs", 1, frames=2)
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert _rows(page) == ["scan · 2 frames"] and page.file_list.count() == 1  # the same row
        assert _names(page.view_model.state.files) == ["scan_m01.nxs"]
        assert exported == []  # nothing is missing now, but more module files may follow: the set rests first
        for _look in range(sources_model.SET_RESTS):
            page.poll_watch()
        _done(page)
        assert sorted(exported) == [("scan_m01.nxs", 0, 3), ("scan_m01.nxs", 1, 3)]  # whole, each frame once
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"
        assert len(page.view_model.state.analysis.metadata["module_files"]) == 3  # read again from all three files
    finally:
        page.stop_watch()
        _close(page)


def test_the_watch_says_what_it_waits_for(tmp_path: Path) -> None:
    page = _page(_context())
    writer = None
    try:
        page.start_watch(tmp_path)
        writer = _writer(tmp_path / "live_00001.nxs", "new")
        assert page.poll_watch() == [] and page.file_list.count() == 0
        assert page.status_text() == (f"Watching {tmp_path.name}: live_00001.nxs is being written by another program "
                                      "and cannot be read until it is closed.")
        page._status("Ready")
        assert page.poll_watch() == [] and page.status_text() == "Ready"  # said once, not at every look
        _release(writer)
        writer = None
        page.poll_watch()
        added = page.poll_watch()
        _done(page)
        assert _names(added) == ["live_00001.nxs"] and _rows(page) == ["live_00001.nxs · 9 frames"]
    finally:
        if writer is not None:
            _release(writer)
        page.stop_watch()
        _close(page)


def test_the_timer_looks_at_the_folder_on_another_thread(tmp_path: Path) -> None:
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()  # the test ticks itself
        _stack(tmp_path / "run_1.nxs", 3)
        threads: list[int] = []
        scan = page.view_model.watch.scan

        def watched_scan():
            threads.append(threading.get_ident())
            return scan()

        page.view_model.watch.scan = watched_scan
        deadline = time.monotonic() + 30
        while page.file_list.count() == 0 and time.monotonic() < deadline:
            page._watch_timer.timeout.emit()  # what the timer does every two seconds
            page._watch_timer.timeout.emit()  # a look is under way: this one is left out
            page._watch_runner.wait(10)
        _done(page)
        assert _rows(page) == ["run_1.nxs · 3 frames"] and len(threads) >= 2
        assert threading.get_ident() not in threads and QThread.currentThread() is page.thread()
        page.stop_watch()
        page._watch_tick()  # not watching: no look
        assert not page._watch_runner.is_busy()
    finally:
        _close(page)


def test_opening_a_scan_folder_says_which_subfolder_its_frames_are_in(tmp_path: Path, monkeypatch) -> None:
    said: list[str] = []
    monkeypatch.setattr(data_step, "show_toast", lambda _parent, text, **_options: said.append(text))
    scan = tmp_path / "scan_00042"
    for module in (1, 2):
        _module(scan / "lmbdp03" / f"scan_00042_m{module:02d}.nxs", module, frames=3)
    page = _page(_context())
    try:
        added = page.add_paths([str(scan)])
        _done(page)
        assert _names(added) == ["scan_00042_m01.nxs"] and _rows(page) == ["scan_00042 · 3 frames"]
        assert said == ["scan_00042 holds no frames itself: its subfolder lmbdp03 was opened"]
        save_tiff(scan / "pilatus" / "scan_00042_00001.tif", np.ones((8, 8), dtype=np.float32))
        page.clear_files()
        del said[:]
        apply_language("zh", [page])
        page.add_paths([str(scan)])
        _done(page)
        assert said == ["scan_00042 本身没有帧：已打开它的这些子文件夹：lmbdp03, pilatus"]
        del said[:]
        page.add_paths([str(scan / "lmbdp03")])  # the folder that holds them: nothing to say
        assert said == []
    finally:
        _close(page)


def test_a_file_without_a_detector_image_says_so_when_it_is_opened_itself(tmp_path: Path) -> None:
    weights = _weights(tmp_path / "weights.h5")
    page = _page(_context())
    try:
        assert page.add_paths([str(tmp_path)]) == []  # in a folder it is no frame
        assert page.status_text() == "No new detector frames (CBF, NeXus/HDF5, TIFF, EDF) found."
        page.add_paths([str(weights)])
        _done(page)
        assert page.status_text() == "Could not analyse the frame: weights.h5 holds no detector image."
        assert page.step_rail.state("data") == "error" and page.source_label.text() == "no detector image"
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text() == "无法分析这一帧：weights.h5 中没有探测器图像。"
        assert page.source_label.text() == "没有探测器图像"
    finally:
        _close(page)


# -- the frame on screen follows its files ------------------------------------------------------------------------------


def test_a_module_that_arrives_is_read_into_the_frame_on_screen(tmp_path: Path) -> None:
    """The row, its path and the frame stay the same; the pixels must not stay those of three modules."""
    for module in (1, 2, 4):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=3)
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()  # the test looks itself
        _done(page)
        page.show_frame(0, 2)  # the newest frame, as the watch shows it
        _done(page)
        assert len(page.view_model.state.analysis.metadata["module_files"]) == 3
        assert page.step_rail.state("data") == "warn" and not page.source_warning_label.isHidden()
        _module(tmp_path / "scan_m03.nxs", 3, frames=3)
        page.poll_watch()
        page.poll_watch()
        _done(page)
        analysis = page.view_model.state.analysis
        assert (analysis.path.name, analysis.frame_index) == ("scan_m01.nxs", 2)  # the same row, the same frame …
        assert len(analysis.metadata["module_files"]) == 4 and _as_on_disk(analysis)  # … read again, from four files
        assert page.source_label.text().splitlines()[0] == "4 module files · 3 frames · stitched"
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"
        assert "missing" not in page.file_list.item(0).toolTip() and "scan_m03.nxs" in page.file_list.item(0).toolTip()
    finally:
        page.stop_watch()
        _close(page)


def test_f5_reads_the_frame_again_when_its_files_changed_and_only_then(tmp_path: Path, monkeypatch) -> None:
    for module in (1, 2, 4):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=3)
    loads = _count_loads(monkeypatch)
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        assert loads == ["scan_m01.nxs"]
        page.run_analysis()  # F5, nothing changed: the frame in memory is used
        _done(page)
        assert loads == ["scan_m01.nxs"]
        _module(tmp_path / "scan_m03.nxs", 3, frames=3)  # the missing module is copied in
        page.run_analysis()
        _done(page)
        analysis = page.view_model.state.analysis
        assert loads == ["scan_m01.nxs"] * 2 and len(analysis.metadata["module_files"]) == 4 and _as_on_disk(analysis)
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"
        tip = page.file_list.item(0).toolTip()  # the row says the same as the step: its tooltip is written again
        assert "4 module files · 3 frames · stitched" in tip and "scan_m03.nxs" in tip and "missing" not in tip
        page.run_analysis()
        _done(page)
        assert loads == ["scan_m01.nxs"] * 2  # as it is on disk now: used again
        with h5py.File(tmp_path / "scan_m02.nxs", "a") as handle:  # a module file written again, the same size
            handle[f"{DET}/data"][0] = 999
        _later(tmp_path / "scan_m02.nxs")
        page.run_analysis()
        _done(page)
        analysis = page.view_model.state.analysis
        assert len(loads) == 3 and float(np.nanmax(analysis.data)) == 999.0 and _as_on_disk(analysis)
    finally:
        _close(page)


def test_f5_reads_a_frame_file_that_was_written_again(tmp_path: Path) -> None:
    """Not only NeXus: a TIFF written again under the same name is read again."""
    path = save_tiff(tmp_path / "frame_00001.tif", np.ones((16, 16), dtype=np.float32))
    page = _page(_context())
    try:
        page.add_paths([str(path)])
        _done(page)
        assert float(np.nanmax(page.view_model.state.analysis.data)) == 1.0
        save_tiff(path, np.full((16, 16), 2.0, dtype=np.float32))
        _later(path)
        page.run_analysis()
        _done(page)
        assert float(np.nanmax(page.view_model.state.analysis.data)) == 2.0
    finally:
        _close(page)


def test_a_watched_file_written_again_in_place_is_read_again(tmp_path: Path) -> None:
    """Its size and its frames stay the same: only the time of its files says that it changed."""
    path = _stack(tmp_path / "run_1.nxs", 3)
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        _done(page)
        assert float(np.nanmax(page.view_model.state.analysis.data)) == 0.0  # frame 1 is filled with 0
        size = path.stat().st_size
        with h5py.File(path, "a") as handle:
            handle[f"{DET}/data"][0] = 7
        _later(path)
        assert path.stat().st_size == size
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert _names(page.view_model.watch_changed) == ["run_1.nxs"] and page.view_model.watch_grown == []
        analysis = page.view_model.state.analysis
        assert analysis.frame_index == 0 and float(np.nanmax(analysis.data)) == 7.0  # the same frame, read again
        page.poll_watch()
        assert page.view_model.watch_changed == []  # said once
    finally:
        page.stop_watch()
        _close(page)


def test_the_row_of_a_source_that_is_not_on_screen_follows_its_files_too(tmp_path: Path) -> None:
    _stack(tmp_path / "a_first.nxs", 2)
    for module in (1, 2, 4):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=3)
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        _done(page)
        assert _rows(page) == ["a_first.nxs · 2 frames", "scan · 3 frames"] and page.file_list.currentRow() == 0
        assert "Module file m03 is missing" in page.file_list.item(1).toolTip()
        _module(tmp_path / "scan_m03.nxs", 3, frames=3)
        page.poll_watch()
        page.poll_watch()
        _done(page)
        tip = page.file_list.item(1).toolTip()
        assert "4 module files · 3 frames · stitched" in tip and "scan_m03.nxs" in tip and "missing" not in tip
        assert page.file_list.currentRow() == 0 and page.view_model.state.analysis.path.name == "a_first.nxs"  # (stays)
    finally:
        page.stop_watch()
        _close(page)


def test_still_being_written_goes_when_the_writer_closes_the_file(tmp_path: Path) -> None:
    """A series written in the mode in which others may read: when the writer closes it no frame is new, but
    the file is finished, and the page must stop saying that it is being written."""
    page = _page(_context())
    path = tmp_path / "swmr_00001.nxs"
    writer = None
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        writer = _writer(path, "swmr")
        page.poll_watch()
        page.poll_watch()
        _done(page)
        still = "swmr_00001.nxs is still being written: it is read again when it changes"
        assert _rows(page) == ["swmr_00001.nxs · 6 frames"] and page.source_warning_label.text() == still
        assert page.step_rail.state("data") == "warn" and still in page.file_list.item(0).toolTip()
        _release(writer)
        writer = None
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert page.view_model.describe(path).complete and page.view_model.watch_grown == []  # no frame is new …
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"  # … the page follows
        assert "still being written" not in page.file_list.item(0).toolTip()
        assert "still being written" not in page.status_text()
    finally:
        if writer is not None:
            _release(writer)
        page.stop_watch()
        _close(page)


# -- a file the library cannot read ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["bytes overwritten", "the library's error"])
def test_a_damaged_file_between_two_good_ones_is_listed_and_says_what_is_wrong(tmp_path: Path, monkeypatch, how) -> None:
    """h5py raises ``RuntimeError`` for inconsistent metadata (a damaged file, one a plain writer is writing).
    One such file must not stop the listing, nor leave the list and the rows out of step."""
    _stack(tmp_path / "a_good_00001.nxs", 3)
    damaged = _stack(tmp_path / "damaged_00001.nxs", 4)
    _stack(tmp_path / "z_good_00002.nxs", 2)
    if how == "bytes overwritten":
        if not _damage(damaged):
            pytest.skip("this HDF5 library gives an OSError for every damage tried")
    else:
        real = h5py.File

        def failing(name, *args, **kwargs):
            if Path(str(name)).name == damaged.name:
                raise RuntimeError("Link iteration failed (bad heap free list)")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(nexus_files.h5py, "File", failing)
    source = DetectorIoFrameSource()
    with pytest.raises(RuntimeError):
        detect_nxs_frame_count(damaged)  # what the reader gives …
    with pytest.raises((OSError, ValueError)) as caught:
        source.frame_count(damaged)  # … and what the frame source makes of it: an error its callers handle
    assert "damaged_00001.nxs looks damaged or incomplete" in str(caught.value)
    assert source.describe(damaged) is None
    page = _page(_context())
    try:
        added = page.add_paths([str(tmp_path)])
        _done(page)
        names = ["a_good_00001.nxs", "damaged_00001.nxs", "z_good_00002.nxs"]
        assert _names(added) == names and _names(page.view_model.state.files) == names
        assert _rows(page) == ["a_good_00001.nxs · 3 frames", "damaged_00001.nxs", "z_good_00002.nxs · 2 frames"]
        assert page.file_list.currentRow() == 0 and page.view_model.state.analysis.path.name == "a_good_00001.nxs"
        assert page.data_batch_button.text() == "Batch Export 6 Frames…"  # 3 + 1 (not counted yet) + 2
        page.file_list.setCurrentRow(1)
        _done(page)
        assert page.status_level() == "error" and page.status_text() == (
            "Could not analyse the frame: damaged_00001.nxs looks damaged or incomplete (the NeXus/HDF5 file could "
            "not be opened) — if it is still being written, try again later.")
        page.start_watch(tmp_path)  # the watch lists the folder too, and looks at it
        page._watch_timer.stop()
        assert page.poll_watch() == [] and page.poll_watch() == [] and page.file_list.count() == 3
    finally:
        page.stop_watch()
        _close(page)


# -- a master and its data files ------------------------------------------------------------------------------------------


def _master_series(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """A master with three data files (3 + 3 + 2 frames) in ``src``, an empty folder ``copy``, and their names in
    the order of a copy by name: the data files first, the master last."""
    _master(tmp_path / "src", "series_7", chunks=(3, 3, 2))
    (tmp_path / "copy").mkdir()
    names = ["series_7_data_000001.h5", "series_7_data_000002.h5", "series_7_data_000003.h5", "series_7_master.h5"]
    return tmp_path / "src", tmp_path / "copy", names


def test_a_master_met_after_its_data_files_takes_all_their_rows(tmp_path: Path) -> None:
    source, folder, names = _master_series(tmp_path)
    for name in names[:2]:
        shutil.copy(source / name, folder)
    page = _page(_context())
    try:
        page.add_paths([str(folder)])  # Open Folder while the copy is under way
        _done(page)
        assert _rows(page) == ["series_7_data_000001.h5 · 3 frames", "series_7_data_000002.h5 · 3 frames"]
        page.show_frame(1, 1)  # the second frame of the second data file is on screen
        _done(page)
        assert float(np.nanmax(page.view_model.state.analysis.data)) == 2.0  # (data file k is filled with k)
        for name in names[2:]:
            shutil.copy(source / name, folder)
        assert page.add_paths([str(folder)]) == []  # Open Folder again: nothing is new, the rows are one source now
        _done(page)
        model = page.view_model
        assert _rows(page) == ["series_7_master.h5 · 8 frames"] and _names(model.state.files) == ["series_7_master.h5"]
        assert page.data_batch_button.text() == "Batch Export 8 Frames…" and model.listed_frames() == 8
        assert _frames_of(model.batch_requests()) == [("series_7_master.h5", index) for index in range(8)]  # none twice
        assert page.file_list.currentRow() == 0 and model.state.current_index == 0
        assert model.state.analysis.path.name == "series_7_master.h5"  # the row on screen is the master's …
        assert page.file_chip.full_text() == "series_7_master.h5"
        # … and the frame on screen is the same frame of the series: the fifth (three in the first data file).
        assert model.state.analysis.frame_index == 4 and page.frame_spin.value() == 5 and page.frame_spin.maximum() == 8
        assert float(np.nanmax(model.state.analysis.data)) == 2.0 and page.file_meta.text().endswith("frame 5 of 8")
    finally:
        _close(page)


def test_a_master_that_arrives_last_while_watched_is_one_row_and_brings_no_frame_twice(tmp_path: Path) -> None:
    source, folder, names = _master_series(tmp_path)
    model = _model()
    model.start_watch(folder)
    handed: list[tuple[str, int]] = []
    for name in names:
        shutil.copy(source / name, folder)
        handed += _rest(model, 3)
    # Each data file was a stack of its own until the master said what they are: one row, eight frames, and the
    # frames handed out under the names of the data files are not handed out again under the master's.
    assert _names(model.state.files) == ["series_7_master.h5"] and model.listed_frames() == 8
    assert handed == [(names[0], 0), (names[0], 1), (names[0], 2), (names[1], 0), (names[1], 1), (names[1], 2),
                      (names[2], 0), (names[2], 1)]
    assert _frames_of(model.batch_requests()) == [("series_7_master.h5", index) for index in range(8)]


def test_a_master_that_arrives_with_more_data_files_hands_out_the_frames_that_are_left(tmp_path: Path) -> None:
    source, folder, names = _master_series(tmp_path)
    model = _model()
    model.start_watch(folder)
    shutil.copy(source / names[0], folder)
    assert _rest(model, 2) == [(names[0], 0), (names[0], 1), (names[0], 2)]
    for name in names[1:]:  # the rest of the series, and the master, arrive together
        shutil.copy(source / name, folder)
    handed = _rest(model, 2)
    assert _names(model.state.files) == ["series_7_master.h5"] and [name for name, _index in handed] == ["series_7_master.h5"] * 5
    assert [index for _name, index in handed] == [3, 4, 5, 6, 7]  # frames 1–3 went out under the first data file's name
    assert _rest(model, 3) == []


def test_a_listed_master_keeps_what_it_handed_out_when_it_takes_the_row_of_a_data_file(tmp_path: Path) -> None:
    """The master and its first data file are watched; the third data file was opened by name (a row of its own:
    the second is missing). When the second arrives the row joins the master, and only the frames that are new
    are handed out."""
    source, folder, names = _master_series(tmp_path)
    for name in (names[3], names[0], names[2]):
        shutil.copy(source / name, folder)
    model = _model()
    model.start_watch(folder)
    assert _names(model.state.files) == ["series_7_master.h5"] and model.frame_count(model.state.files[0]) == 3
    assert _names(model.add_paths([folder / names[2]])) == [names[2]]  # opened by name
    shutil.copy(source / names[1], folder)
    handed = _rest(model, 2)
    assert _names(model.state.files) == ["series_7_master.h5"] and model.listed_frames() == 8
    assert handed == [("series_7_master.h5", index) for index in range(3, 8)]  # frames 1–3 were there before the watch


def test_the_rows_of_data_files_leave_the_page_when_their_master_arrives(tmp_path: Path, monkeypatch) -> None:
    source, folder, names = _master_series(tmp_path)
    page = _page(_context())
    exported: list[tuple[str, int]] = []
    monkeypatch.setattr(page.view_model, "analyze_and_export",
                        lambda request, destination, **_options: exported.append((request.path.name, request.frame_index)) or [])
    try:
        page.start_watch(folder)
        page._watch_timer.stop()
        page.auto_export_check.setChecked(True)
        for name in names:
            shutil.copy(source / name, folder)
            for _look in range(3):
                page.poll_watch()
            _done(page)
        assert _rows(page) == ["series_7_master.h5 · 8 frames"] and page.data_batch_button.text() == "Batch Export 8 Frames…"
        assert len(exported) == 8 and len(set(exported)) == 8  # eight frames, each exported once
        analysis = page.view_model.state.analysis
        assert analysis.path.name == "series_7_master.h5" and page.file_list.currentRow() == 0
    finally:
        page.stop_watch()
        _close(page)


def test_a_master_that_gains_its_data_files_while_watched_hands_out_each_frame_once(tmp_path: Path) -> None:
    source, folder, names = _master_series(tmp_path)
    model = _model()
    model.start_watch(folder)
    handed: list[tuple[str, int]] = []
    for name in [names[3], *names[:3]]:  # the master first (a file writer that fills the data files one by one)
        shutil.copy(source / name, folder)
        handed += _rest(model, 3)
        assert _names(model.state.files) == ["series_7_master.h5"]
    assert handed == [("series_7_master.h5", index) for index in range(8)] and model.listed_frames() == 8


def test_data_files_after_a_missing_one_are_not_rows_of_their_own(tmp_path: Path) -> None:
    master = _master(tmp_path, "x", chunks=(3, 3, 2))
    (tmp_path / "x_data_000002.h5").rename(tmp_path / "kept_elsewhere.bin")
    _stack(tmp_path / "y_data_000001.h5", 2, dataset="/entry/data/data")  # named alike, the file of no master
    source = DetectorIoFrameSource()
    # Frames after the gap cannot be read, so the file that holds them is no source of its own in front of the master.
    assert _names(source.expand([tmp_path])) == ["x_master.h5", "y_data_000001.h5"]
    assert _names(source.parts(master)) == ["x_master.h5", "x_data_000001.h5", "x_data_000003.h5"]
    assert _names(source.expand([tmp_path / "x_data_000003.h5"])) == ["x_data_000003.h5"]  # named by the person: opened
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        missing = "Data file x_data_000002.h5 is missing: frames after frame 3 cannot be read"
        assert _rows(page) == ["x_master.h5 · 3 frames", "y_data_000001.h5 · 2 frames"]
        assert page.source_warning_label.text() == missing and page.step_rail.state("data") == "warn"
        (tmp_path / "kept_elsewhere.bin").rename(tmp_path / "x_data_000002.h5")  # the missing file is copied in
        page.add_paths([str(tmp_path)])
        _done(page)
        assert _rows(page) == ["x_master.h5 · 8 frames", "y_data_000001.h5 · 2 frames"]
        assert page.source_warning_label.isHidden() and page.step_rail.state("data") == "ok"
    finally:
        _close(page)


def test_a_data_file_listed_on_its_own_leaves_the_list_when_its_master_says_it_is_one_of_its_files(tmp_path: Path) -> None:
    """Listed before its master was there, and after a data file that is missing: still a file of the master."""
    source, folder, names = _master_series(tmp_path)
    shutil.copy(source / names[0], folder)
    shutil.copy(source / names[2], folder)
    model = _model()
    assert _names(model.add_paths([folder])) == [names[0], names[2]]
    shutil.copy(source / names[3], folder)
    assert model.add_paths([folder]) == []
    assert _names(model.state.files) == ["series_7_master.h5"] and model.listed_frames() == 3
    assert [(row, path.name) for row, path in model.dropped] == [(1, names[2])]
    assert [(row, before.name, now.name) for row, before, now in model.replaced] == [(0, names[0], names[3])]


# -- a set of module files that arrives file by file ------------------------------------------------------------------


def test_a_set_that_arrives_first_module_first_is_handed_out_once_when_it_rests(tmp_path: Path) -> None:
    """Without a gap in its numbering nothing says that a set is incomplete: the first module alone, then two …
    look whole. Its frames wait until no further module file has arrived for ``SET_RESTS`` looks."""
    rests = sources_model.SET_RESTS
    model = _model()
    source = DetectorIoFrameSource()
    model.start_watch(tmp_path)
    for module in (1, 2, 3):
        _module(tmp_path / f"scan_00007_m{module:02d}.nxs", module, frames=3)
        assert _rest(model, rests - 1) == []  # (each state of the set rests, but not for long enough)
        (listed,) = model.state.files
        assert listed.name == "scan_00007_m01.nxs" and _names(model.watch_held) == ["scan_00007_m01.nxs"]
        assert model.settling(listed) == (module, None)  # module files so far; no other set says how many there are
        assert model.frame_count(listed) == 3  # listed and shown meanwhile
    handed = _rest(model, rests)
    assert handed == [("scan_00007_m01.nxs", index) for index in range(3)]  # once, with every module
    assert all(len(source.load(tmp_path / name, index).metadata["module_files"]) == 3 for name, index in handed)
    assert model.settling(model.state.files[0]) is None and model.watch_held == [] and _rest(model, 3) == []

    # The next scan of the folder: as many module files as the scan before means that it is whole.
    for module in (1, 2, 3):
        _module(tmp_path / f"scan_00008_m{module:02d}.nxs", module, frames=2)
    assert _rest(model, 2) == [("scan_00008_m01.nxs", 0), ("scan_00008_m01.nxs", 1)]
    # One with fewer module files than the others waits longer, and says how many the others have.
    for module in (1, 2):
        _module(tmp_path / f"scan_00009_m{module:02d}.nxs", module, frames=2)
    assert _rest(model, rests + 2) == [] and model.settling(model.state.files[2]) == (2, 3)
    _module(tmp_path / "scan_00009_m03.nxs", 3, frames=2)
    assert _rest(model, 1) == []  # the file was just seen: it may still be being written
    assert _rest(model, 1) == [("scan_00009_m01.nxs", 0), ("scan_00009_m01.nxs", 1)]  # whole: no further wait
    # A scan that never gets them is handed out in the end (some detector modules may be switched off).
    for module in (1, 2):
        _module(tmp_path / f"scan_00010_m{module:02d}.nxs", module, frames=1)
    assert _rest(model, sources_model.SET_RESTS_FEWER + 2) == [("scan_00010_m01.nxs", 0)]


def test_a_watched_file_that_grows_after_clear_comes_back_with_its_new_frames(tmp_path: Path) -> None:
    path = _stack(tmp_path / "insitu_00001.nxs", 4)
    _stack(tmp_path / "other_00002.nxs", 2)
    model = _model()
    model.start_watch(tmp_path)
    _grow(path, 6)
    assert _rest(model, 2) == [("insitu_00001.nxs", 4), ("insitu_00001.nxs", 5)]
    model.clear_files()  # Analyze ▸ Clear while the folder is watched
    assert model.state.files == [] and _rest(model, 2) == [] and model.watch.active
    _grow(path, 9)
    assert model.poll_watch() == []
    added = model.poll_watch()
    assert _names(added) == ["insitu_00001.nxs"] and _names(model.state.files) == ["insitu_00001.nxs"]  # only what grew
    assert _frames_of(model.watch_requests(added)) == [("insitu_00001.nxs", index) for index in (6, 7, 8)]  # only the new
    model.remove_file(0)  # one file taken off the list comes back the same way
    _grow(path, 10)
    assert _rest(model, 2) == [("insitu_00001.nxs", 9)] and _names(model.state.files) == ["insitu_00001.nxs"]


def test_the_page_lists_a_watched_file_again_when_it_grows_after_clear(tmp_path: Path) -> None:
    path = _stack(tmp_path / "insitu_00001.nxs", 3)
    page = _page(_context())
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        _done(page)
        page.clear_files()
        assert page.file_list.count() == 0 and page.watch_button.isChecked()
        _grow(path, 5)
        page.poll_watch()
        page.poll_watch()
        _done(page)
        assert _rows(page) == ["insitu_00001.nxs · 5 frames"] and page.file_list.currentRow() == 0
        assert page.view_model.state.analysis.path.name == "insitu_00001.nxs"
    finally:
        page.stop_watch()
        _close(page)


def test_a_watched_file_started_again_with_fewer_frames_hands_out_its_frames_again(tmp_path: Path) -> None:
    path = _stack(tmp_path / "run.nxs", 6)
    model = _model()
    model.start_watch(tmp_path)
    path.unlink()
    _stack(path, 2)  # the scan was repeated under the same name
    assert _rest(model, 2) == [("run.nxs", 0), ("run.nxs", 1)] and model.frame_count(path) == 2
    _grow(path, 4)
    assert _rest(model, 2) == [("run.nxs", 2), ("run.nxs", 3)]


def test_a_set_that_was_there_when_the_watch_began_is_followed_without_a_wait(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=2) for module in (1, 2, 3)]
    gapped = [_module(tmp_path / "gap" / f"scan_m{module:02d}.nxs", module, frames=2) for module in (1, 2, 4)]
    model = _model()
    model.start_watch(tmp_path)
    for path in paths:
        _grow(path, 4)
    assert _rest(model, 2) == [("scan_m01.nxs", 2), ("scan_m01.nxs", 3)] and model.settling(model.state.files[0]) is None
    # A set that was there with a module missing gains frames: none is handed out, however long it rests.
    model = _model()
    model.start_watch(tmp_path / "gap")
    for path in gapped:
        _grow(path, 4)
    assert _rest(model, sources_model.SET_RESTS + 2) == [] and _names(model.watch_held) == ["scan_m01.nxs"]
    _module(tmp_path / "gap" / "scan_m03.nxs", 3, frames=4)
    handed = _rest(model, sources_model.SET_RESTS + 2)
    assert handed == [("scan_m01.nxs", index) for index in range(4)]  # whole now: every frame, once


def test_the_watch_says_that_a_set_is_still_arriving_and_exports_it_whole(tmp_path: Path, monkeypatch) -> None:
    page = _page(_context())
    exported: list[tuple[str, int, int]] = []
    source = DetectorIoFrameSource()

    def export(request, destination, **_options):
        modules = len(source.load(request.path, request.frame_index).metadata.get("module_files", [request.path]))
        exported.append((request.path.name, request.frame_index, modules))
        return []

    monkeypatch.setattr(page.view_model, "analyze_and_export", export)
    try:
        page.start_watch(tmp_path)
        page._watch_timer.stop()
        page.auto_export_check.setChecked(True)
        for module in (1, 2, 3):
            _module(tmp_path / f"scan_00007_m{module:02d}.nxs", module, frames=2)
            for _look in range(3):
                page.poll_watch()
            _done(page)
            assert exported == [] and page.file_list.count() == 1
            name = "scan_00007" if module > 1 else "scan_00007_m01.nxs"  # (one file is not a set yet)
            arriving = f"{name} is not exported yet: its module files are still arriving ({module} so far)"
            assert page.status_text().startswith(arriving)  # said with the frame shown, for as long as it waits
            shown = page.view_model.state.analysis  # what is there is shown meanwhile, read again at each arrival
            assert len(shown.metadata.get("module_files", [shown.path])) == module and _as_on_disk(shown)
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text().startswith("scan_00007 还没有导出：它的模块文件还在陆续到达（目前 3 个）")
        apply_language("en", [page])
        page.refresh_language()
        for _look in range(sources_model.SET_RESTS):
            page.poll_watch()
        assert page.status_text() == "scan_00007: 2 new frames" and page.status_level() == "ok"
        _done(page)
        assert sorted(exported) == [("scan_00007_m01.nxs", 0, 3), ("scan_00007_m01.nxs", 1, 3)]  # once, whole
        assert _rows(page) == ["scan_00007 · 2 frames"] and "still arriving" not in page.status_text()
        # The next scan with fewer module files than this one: the sentence says how many the others have.
        for module in (1, 2):
            _module(tmp_path / f"scan_00008_m{module:02d}.nxs", module, frames=1)
        for _look in range(3):
            page.poll_watch()
        _done(page)
        fewer = "scan_00008 is not exported yet: 2 of the 3 module files that the other scans in this folder have are here"
        assert page.status_text().startswith(fewer) and len(exported) == 2
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text().startswith("scan_00008 还没有导出：此文件夹中其他扫描有 3 个模块文件，它目前只有 2 个")
        apply_language("en", [page])
    finally:
        page.stop_watch()
        _close(page)


# -- stepping, long names, many subfolders -----------------------------------------------------------------------------


def test_stepping_back_from_a_single_frame_enters_the_file_before_at_its_last_frame(tmp_path: Path) -> None:
    _stack(tmp_path / "a_stack.nxs", 5)
    _stack(tmp_path / "b_single.nxs", 1)
    _stack(tmp_path / "c_stack.nxs", 4)
    page = _page(_context())
    try:
        page.add_paths([str(tmp_path)])
        _done(page)
        place = lambda: (page.file_list.currentRow(), page.view_model.state.frame_index)  # noqa: E731
        page.show_frame(2, 0)
        _done(page)
        visited = []
        for _press in range(3):  # Page Up from the first frame of c_stack
            page._step_frame(-1)
            _done(page)
            visited.append(place())
        assert visited == [(1, 0), (0, 4), (0, 3)]  # b_single, then a_stack from its last frame on
        assert page.view_model.state.analysis.path.name == "a_stack.nxs" and page.frame_spin.value() == 4
        for _press in range(3):  # and forward again: every frame of the series, in order
            page._step_frame(+1)
            _done(page)
            visited.append(place())
        assert visited[3:] == [(0, 4), (1, 0), (2, 0)]
    finally:
        _close(page)


def test_a_long_name_in_what_is_said_of_a_source_is_shortened_and_whole_in_the_tooltip(tmp_path: Path) -> None:
    stem = "jg_gisaxs_12nm_insitu_00001r3_00001"
    for module in (1, 2, 3):
        _module(tmp_path / "set" / f"{stem}_m{module:02d}.nxs", module, frames=3)
    odd = f"{stem}_m04.nxs"  # named like a module of this scan, with another shape
    with h5py.File(tmp_path / "set" / odd, "w") as handle:
        handle.create_dataset(f"{DET}/data", data=np.ones((3, 50, 70), dtype=np.int32))
        handle.create_dataset(f"{DET}/translation/distance", data=np.asarray([0.0, 99.0, 0.0], "f4"))
    groups = ("scan_0001_sample_alignment_before_annealing", "scan_0002_sample_after_annealing_at_453K")
    (tmp_path / "two").mkdir()
    with h5py.File(tmp_path / "two" / "two_scans_00001.nxs", "w") as handle:  # two scans in one file
        for group in groups:
            handle.create_dataset(f"/{group}/instrument/lambda_detector_upstream/data", data=np.ones((5, 64, 96), dtype=np.int32))
    page = _page(_context())
    try:
        page.resize(1090, 700)  # the Analyze page of a 1280×800 window
        page.show()
        page.add_paths([str(tmp_path / "set")])
        _done(page)
        note = page.source_label.text().splitlines()[-1]
        assert "does not fit the set" in note and odd not in note and short_name(odd) in note  # it fits the panel …
        assert odd in page.source_label.toolTip() and "does not fit the set" in page.source_label.toolTip()  # … and is whole here
        page.add_paths([str(tmp_path / "two")])
        _done(page)
        dataset = f"/{groups[0]}/instrument/lambda_detector_upstream/data"
        warning = page.source_warning_label.text()
        assert warning.startswith("2 image datasets in this file: ") and dataset not in warning
        assert short_name(dataset) in warning and all(len(word) <= source_texts.NAME_LIMIT for word in warning.split())
        other = f"/{groups[1]}/instrument/lambda_detector_upstream/data"
        whole = (f"2 image datasets in this file: {dataset} is shown, not {other} "
                 "(Format Converter and Geometry Calibration let you choose the dataset)")
        assert page.source_warning_label.toolTip() == whole  # both are named, and where a person can choose
        assert f"{whole}." in page.status_text()  # the status line has room
        viewport = page.step_pages["data"].viewport()
        assert page.step_pages["data"].widget().width() <= viewport.width()
    finally:
        _close(page)


def test_the_notice_for_frames_one_level_down_names_a_few_subfolders_only(tmp_path: Path, monkeypatch) -> None:
    said: list[str] = []
    monkeypatch.setattr(data_step, "show_toast", lambda _parent, text, **_options: said.append(text))
    raw = tmp_path / "raw"
    for scan in range(1, 41):
        _stack(raw / f"jg_gisaxs_12nm_insitu_{scan:05d}" / f"jg_gisaxs_12nm_insitu_{scan:05d}.nxs", 2)
    page = _page(_context())
    try:
        added = page.add_paths([str(raw)])
        _done(page)
        assert len(added) == 40 and said == [
            "raw holds no frames itself: its subfolders jg_gisaxs_12nm_insitu_00001, jg_gisaxs_12nm_insitu_00002, "
            "jg_gisaxs_12nm_insitu_00003 and 37 more were opened"]
        apply_language("zh", [page])
        assert page._found_below([str(raw)], added) == (
            "raw 本身没有帧：已打开它的这些子文件夹：jg_gisaxs_12nm_insitu_00001, jg_gisaxs_12nm_insitu_00002, "
            "jg_gisaxs_12nm_insitu_00003，另有 37 个")
    finally:
        _close(page)


def test_the_sentence_of_the_empty_data_step_follows_the_interface_language(tmp_path: Path) -> None:
    """While no frame is shown the page composes nothing again after a language switch: the sentence has to be
    one the fixed texts' translation knows (no value filled in)."""
    weights = _weights(tmp_path / "weights.h5")
    page = _page(_context())
    try:
        english = page.step_intro["data"].text()
        chinese = "打开或拖入探测器帧（CBF、NeXus/HDF5、TIFF、EDF）或文件夹。"
        assert english.startswith("Open or drop detector frames (CBF, NeXus/HDF5, TIFF, EDF) or a folder. Files that belong")
        apply_language("zh", [page])
        page.refresh_language()
        assert page.step_intro["data"].text().startswith(chinese)
        apply_language("en", [page])
        page.refresh_language()
        assert page.step_intro["data"].text() == english
        page.add_paths([str(weights)])  # a file that cannot be shown: the step is empty again
        _done(page)
        assert page.step_intro["data"].text() == english
        apply_language("zh", [page])
        page.refresh_language()
        assert page.step_intro["data"].text().startswith(chinese)
        page.clear_files()
        assert page.step_intro["data"].text().startswith(chinese)
        assert page.summary_label.text() == "打开或拖入探测器文件（CBF、NeXus/HDF5、TIFF、EDF）开始"
    finally:
        _close(page)


def test_no_two_parts_of_the_page_define_the_same_method() -> None:
    """The page and the view model are assembled from mixins; a method defined by two of them hides one silently
    (a frame stepper named like the undo step would take its place)."""
    import inspect
    from collections import defaultdict

    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.features.analyze.presentation.view_model import AnalyzeViewModel

    for assembled in (AnalyzePage, AnalyzeViewModel):
        owners = defaultdict(list)
        for part in assembled.__mro__[1:]:
            if part.__module__.startswith("src.gimap.features.analyze"):
                for name, value in vars(part).items():
                    if inspect.isfunction(value) and not name.startswith("__"):
                        owners[name].append(part.__name__)
        assert {name: parts for name, parts in owners.items() if len(parts) > 1} == {}


def test_every_sentence_about_a_source_has_its_chinese() -> None:
    """The reader's templates and Analyze's own sentences about a source are keys of the Chinese table."""
    templates = [*nexus_layout.SUMMARY_TEMPLATES, *nexus_layout.WARNING_TEMPLATES, *nexus_layout.ERROR_TEMPLATES,
                 *source_texts.ORIENTATIONS.values(), source_texts.ROW_FRAMES, source_texts.FRAMES_TOTAL,
                 source_texts.FILES_AND_MORE, source_texts.NEW_FRAMES, source_texts.NEW_FRAME, data_step.OPEN_HINT,
                 data_step.NOTHING_FOUND, data_step.FOUND_BELOW, data_step.FOUND_BELOW_ONE, data_step.GROWING,
                 batch_watch.WATCH_WAITS,
                 batch_watch.EXPORT_WAITS, batch_watch.SET_ARRIVING, batch_watch.SET_FEWER, "Frames listed: {n}",
                 "Open or drop detector files (CBF, NeXus/HDF5, TIFF, EDF) to start"]
    from src.gimap.features.analyze.presentation.bindings import file_list

    templates += [file_list.FRAME_POSITION, file_list.FRAME_POSITION_TIP, *file_list.STEP_TIPS[True],
                  *file_list.STEP_TIPS[False], data_step.SENTENCE]
    import string

    for template in templates:
        chinese = i18n.ZH.get(template)
        assert chinese and chinese != template, template
        fields = lambda text: sorted(name for _text, name, _spec, _conversion in string.Formatter().parse(text) if name)  # noqa: E731
        assert fields(chinese) == fields(template), template  # the same values are filled in
