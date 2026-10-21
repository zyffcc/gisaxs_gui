"""What the first rounds of the NeXus / HDF5 reader left open (``shared/detector_io``).

* one name for a set of module files (``scan_stem``);
* a damaged module file is a part that cannot be read yet, not the end of its set;
* every file a master links to stands for the master, also one behind a data file that is missing;
* a file its writer left open is told from one its writer holds, once it has not changed for a while;
* a set with fewer module files than another scan of its folder says which ones may be missing;
* frames stored along the last axis are said;
* two image datasets are both named, with where a person can choose.

Every file is synthetic (h5py, in ``tmp_path``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.shared.detector_io import (
    BeingWritten,
    LeftUnfinished,
    cached_layout,
    detect_nxs_frame_count,
    forget_layout,
    group_sources,
    inspect_nexus,
    load_detector_image,
    scan_stem,
)
from src.gimap.shared.detector_io import nexus_files, nexus_layout, nexus_modules

DET = "/entry/instrument/detector"
H, W = 20, 30


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _module(path: Path, module: int, frames: int = 2, *, shape=(H, W)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.full((frames,) + shape, 10 * module, dtype=np.int32) + np.arange(frames, dtype=np.int32)[:, None, None]
        handle.create_dataset(f"{DET}/data", data=data, maxshape=(None,) + shape, chunks=(1,) + shape)
        handle.create_dataset(f"{DET}/translation/distance", data=np.asarray([0.0, (module - 1) * (shape[0] + 4), 0.0], "f4"))
    return path


def _names(paths) -> list[str]:
    return [Path(path).name for path in paths]


def _damage(path: Path) -> str:
    """Overwrite 16 bytes of the file's metadata so that the library opens it and then fails with an error that
    is no ``OSError`` (a damaged or half-written file). Returns the name of that error ('' when this library
    read, or refused to open, every damage tried)."""
    good = path.read_bytes()
    for start in range(96, 2048, 8):
        path.write_bytes(good[:start] + b"\\xff" * 16 + good[start + 16:])
        forget_layout()
        try:
            inspect_nexus(path)
        except OSError:
            continue
        except Exception as error:  # noqa: BLE001 - the kind of failure wanted
            forget_layout()
            return type(error).__name__
    path.write_bytes(good)
    forget_layout()
    return ""


_KILLED = """
    import os, sys, h5py
    path, mode = sys.argv[1], sys.argv[2]
    handle = h5py.File(path, "w", libver="latest")
    data = handle.create_dataset("/entry/instrument/detector/data", shape=(0, 20, 30), maxshape=(None, 20, 30),
                                 dtype="i4", chunks=(1, 20, 30))
    if len(sys.argv) > 3:   # a module file of a detector: where the module sits
        handle.create_dataset("/entry/instrument/detector/translation/distance", data=[0.0, float(sys.argv[3]), 0.0])
    if mode == "swmr":
        handle.swmr_mode = True
    for index in range(6):
        data.resize(index + 1, axis=0)
        data[index] = index
        data.flush() if mode == "swmr" else handle.flush()
    os._exit(1)   # the writer dies: the file keeps its mark "open for writing"
"""


def _killed(path: Path, mode: str, module: int = 0) -> Path:
    """A file whose writer ended without closing it (``plain``: the latest format; ``swmr``: readable meanwhile);
    with a ``module`` number, a module file of a detector (placed as ``_module`` places it)."""
    position = [str((module - 1) * (H + 4))] if module else []
    subprocess.run([sys.executable, "-c", textwrap.dedent(_KILLED), str(path), mode, *position], check=False, timeout=60)
    return path


def _long_ago(path: Path, seconds: float = 600.0) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime - seconds))


# -- one name for a set of module files -------------------------------------------------------------------------------


def test_scan_stem_is_the_name_of_a_module_set_without_its_module_number(tmp_path: Path) -> None:
    for module in (1, 2):
        _module(tmp_path / f"scan_00007_m{module:02d}.nxs", module)
    for module in (1, 2):
        _module(tmp_path / f"runm{module}.nxs", module)  # the module number without an underscore
    with h5py.File(tmp_path / "film_m1.nxs", "w") as handle:  # named like a module, a file of its own
        handle.create_dataset(f"{DET}/data", data=np.zeros((3, H, W), dtype=np.int32))
    layout = cached_layout(tmp_path / "scan_00007_m02.nxs")
    assert layout.kind == "module_set" and scan_stem(layout) == "scan_00007"
    assert scan_stem(cached_layout(tmp_path / "runm1.nxs")) == "run"
    assert scan_stem(cached_layout(tmp_path / "film_m1.nxs")) == "" and scan_stem(None) == ""
    # The metadata of a frame says what it was read from: any module file names the set alike.
    metadata = load_detector_image(tmp_path / "scan_00007_m02.nxs").metadata
    assert scan_stem(metadata) == "scan_00007" and scan_stem(metadata, "elsewhere/scan_00007_m02.nxs") == "scan_00007"
    assert scan_stem({"layout": "stack"}, "film_m1.nxs") == "" and scan_stem({}, "scan_00007_m01.nxs") == ""
    # One rule for every tool that names a set.
    from src.gimap.app.headless_sources import GivenSource
    from src.gimap.features.analyze.application.ports import source_stem

    assert source_stem(tmp_path / "scan_00007_m02.nxs", layout) == "scan_00007"
    assert source_stem(tmp_path / "film_m1.nxs", cached_layout(tmp_path / "film_m1.nxs")) == "film_m1"
    assert GivenSource(str(tmp_path / "scan_00007_m01.nxs"), (), layout).stem == "scan_00007"
    assert GivenSource(str(tmp_path / "film_m1.nxs"), (), None).stem == "film_m1"


# -- a damaged module file --------------------------------------------------------------------------------------------


def test_a_damaged_module_file_is_a_part_that_cannot_be_read_not_the_end_of_its_set(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module, frames=3) for module in (1, 2, 3)]
    error = _damage(paths[1])
    if not error:
        pytest.skip("this HDF5 library reads, or refuses to open, every damage tried")
    with pytest.raises(Exception) as caught:  # the file itself: the library's own error, which is no OSError
        inspect_nexus(paths[1])
    assert type(caught.value).__name__ == error and not isinstance(caught.value, OSError)
    layout = cached_layout(paths[0])
    assert layout.kind == "module_set" and _names(layout.members) == ["scan_m01.nxs", "scan_m03.nxs"]
    assert _names(layout.pending) == ["scan_m02.nxs"] and not layout.complete and layout.frames == 3
    assert layout.warning_texts() == ("Module file m02 cannot be read yet: that part of the detector is left out",)
    assert cached_layout(paths[2]) is layout  # every module file that can be read stands for the same set
    # One row, whatever the order the files are listed in: the damaged file is a part of the set.
    assert _names(group_sources(paths)) == ["scan_m01.nxs"]
    assert _names(group_sources(reversed(paths))) == ["scan_m01.nxs"]
    _module(paths[1], 2, frames=3)  # written again, whole
    layout = cached_layout(paths[0])
    assert len(layout.members) == 3 and layout.pending == () and layout.complete and layout.warnings == ()


def test_a_healthy_file_that_does_not_fit_still_ends_the_set(tmp_path: Path) -> None:
    """Only a file the library cannot walk is taken as damaged: one of another kind is read on its own."""
    for module in (1, 2):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module)
    with h5py.File(tmp_path / "scan_m03.nxs", "w") as handle:
        handle.create_dataset("/entry/notes/table", data=np.zeros((4, 4)))
    layout = cached_layout(tmp_path / "scan_m01.nxs")
    assert layout.kind == "stack" and layout.warnings == ((nexus_layout.NOT_A_MODULE, {"name": "scan_m03.nxs"}),)


# -- a master and every file it links to --------------------------------------------------------------------------------


def _master(folder: Path, name: str, data_names: list[str], counts=(3, 3, 2), virtual: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    master = folder / f"{name}.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        entry.create_group("instrument/detector").attrs["NX_class"] = "NXdetector"
        sources = []
        for number, (data_name, count) in enumerate(zip(data_names, counts), start=1):
            with h5py.File(folder / data_name, "w") as chunk:
                chunk.create_dataset("/entry/data/data", data=np.full((count, H, W), number, dtype=np.uint32))
            sources.append((data_name, count))
            if not virtual:
                data[f"data_{number:06d}"] = h5py.ExternalLink(data_name, "/entry/data/data")
        if virtual:
            layout = h5py.VirtualLayout(shape=(sum(counts), H, W), dtype=np.uint32)
            start = 0
            for data_name, count in sources:
                layout[start:start + count] = h5py.VirtualSource(data_name, "/entry/data/data", shape=(count, H, W))
                start += count
            data.create_virtual_dataset("data", layout, fillvalue=0)
    return master


@pytest.mark.parametrize("data_names", [
    ["run8_data_000001.h5", "run8_data_000002.h5", "run8_data_000003.h5"],  # the usual names
    ["a_first_part.h5", "b_second_part.h5", "c_third_part.h5"],            # names that say nothing
])
@pytest.mark.parametrize("virtual", [False, True])
def test_every_file_a_master_links_to_stands_for_it_also_behind_a_missing_one(tmp_path: Path, data_names, virtual) -> None:
    master = _master(tmp_path, "run8_master", data_names, virtual=virtual)
    (tmp_path / data_names[1]).unlink()
    layout = cached_layout(master)
    assert layout.kind == "chunked_master" and layout.frames == 3 and not layout.complete
    assert layout.warning_texts() == (f"Data file {data_names[1]} is missing: frames after frame 3 cannot be read",)
    last = (tmp_path / data_names[2]).resolve()
    assert last in layout.members + layout.behind  # the master links to it: a file of this source
    if not virtual:
        assert layout.members == (master.resolve(), (tmp_path / data_names[0]).resolve()) and layout.behind == (last,)
    # A folder listing, in any order: one row. The files behind the gap are no sources of their own.
    listing = sorted(tmp_path.iterdir())
    assert _names(group_sources(listing)) == ["run8_master.h5"]
    assert _names(group_sources(reversed(listing))) == ["run8_master.h5"]
    # The file comes back: all frames, nothing out of reach.
    with h5py.File(tmp_path / data_names[1], "w") as chunk:
        chunk.create_dataset("/entry/data/data", data=np.full((3, H, W), 2, dtype=np.uint32))
    layout = cached_layout(master)
    assert layout.frames == 8 and layout.complete and layout.behind == () and len(layout.members) == 4


def test_a_data_file_out_of_its_masters_reach_can_still_be_opened_by_itself(tmp_path: Path) -> None:
    names = ["run8_data_000001.h5", "run8_data_000002.h5", "run8_data_000003.h5"]
    master = _master(tmp_path, "run8_master", names)
    (tmp_path / names[1]).unlink()
    # Named without its master, it is the only way to its frames: a file of its own.
    assert _names(group_sources([tmp_path / names[2]], keep_without_image=True)) == [names[2]]
    assert detect_nxs_frame_count(tmp_path / names[2]) == 2
    # A data file the master reaches stands for the master, as before.
    assert _names(group_sources([tmp_path / names[0]])) == ["run8_master.h5"]
    assert _names(group_sources([tmp_path / names[2], master])) == ["run8_master.h5"]


# -- a file its writer left open ------------------------------------------------------------------------------------------


def test_a_file_left_open_by_a_writer_that_ended_says_so_and_how_to_repair_it(tmp_path: Path) -> None:
    path = _killed(tmp_path / "crashed_00001.nxs", "plain")
    with pytest.raises(BeingWritten) as caught:  # just written: nothing tells it from a file its writer holds
        cached_layout(path)
    assert type(caught.value) is BeingWritten
    assert str(caught.value) == "crashed_00001.nxs is being written by another program and cannot be read until it is closed."
    before = (path.read_bytes(), path.stat().st_size)
    _long_ago(path)
    with pytest.raises(LeftUnfinished) as caught:
        cached_layout(path)
    assert isinstance(caught.value, BeingWritten) and isinstance(caught.value, OSError) and caught.value.path == path
    assert str(caught.value) == (
        "crashed_00001.nxs was left open for writing and has not changed for a while: the program that wrote it seems "
        "to have ended without closing it. It cannot be read as it is; running h5clear -s on a copy of it makes "
        "readable what was stored in it.")
    assert str(caught.value) == nexus_files.LEFT_UNFINISHED.format(name=path.name)
    with pytest.raises(LeftUnfinished):
        detect_nxs_frame_count(path)
    assert group_sources([path]) == [path.resolve()]  # listed all the same: reading it says the sentence
    assert (path.read_bytes(), path.stat().st_size) == before  # the reader never clears the mark itself


def test_how_long_a_while_is(tmp_path: Path, monkeypatch) -> None:
    path = _killed(tmp_path / "crashed_00002.nxs", "plain")
    _long_ago(path, nexus_files.UNFINISHED_AFTER_S - 20)
    with pytest.raises(BeingWritten) as caught:
        cached_layout(path)
    assert not isinstance(caught.value, LeftUnfinished)
    _long_ago(path, 40)
    with pytest.raises(LeftUnfinished):
        cached_layout(path)
    assert 30 <= nexus_files.UNFINISHED_AFTER_S <= 600


@pytest.mark.skipif(shutil.which("h5clear") is None, reason="the HDF5 command-line tools are not installed")
def test_h5clear_on_a_copy_makes_readable_what_the_writer_had_stored(tmp_path: Path) -> None:
    """What the sentence promises, with the tool it names."""
    path = _killed(tmp_path / "crashed_00003.nxs", "plain")
    copy = tmp_path / "copy_00003.nxs"
    shutil.copy(path, copy)
    done = subprocess.run(["h5clear", "-s", str(copy)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    layout = cached_layout(copy)
    assert layout.kind == "stack" and layout.frames == 6
    assert int(load_detector_image(copy, frame_idx=5).data[0, 0]) == 5


def test_a_file_written_for_others_to_read_and_never_closed_is_read_and_says_so(tmp_path: Path, monkeypatch) -> None:
    path = _killed(tmp_path / "left_00001.nxs", "swmr")
    layout = cached_layout(path)  # just written: its writer may still be at work
    assert layout.frames == 6 and not layout.complete
    assert layout.warnings == ((nexus_modules.STILL_WRITTEN, {"name": "left_00001.nxs"}),)
    # Unchanged for a while: the same files, said differently, without anybody touching them.
    _long_ago(path)
    assert cached_layout(path).warnings[0][0] == nexus_modules.LEFT_OPEN
    # While it is watched: the layout that says "still being written" is asked again once the while has passed.
    forget_layout()
    monkeypatch.setattr(nexus_files, "UNFINISHED_AFTER_S", 1.0)
    monkeypatch.setattr(nexus_modules, "UNFINISHED_AFTER_S", 1.0)
    os.utime(path)
    first = cached_layout(path)
    assert first.warnings[0][0] == nexus_modules.STILL_WRITTEN and cached_layout(path) is first
    deadline = time.time() + 10
    while cached_layout(path) is first and time.time() < deadline:
        time.sleep(0.1)
    layout = cached_layout(path)
    assert layout.warning_texts() == (
        "left_00001.nxs was not closed by the program that wrote it and has not changed for a while: it is read as "
        "it is, and its last frame may be incomplete",)
    assert layout.frames == 6 and not layout.complete and int(load_detector_image(path, frame_idx=5).data[0, 0]) == 5


def test_a_module_file_left_by_a_writer_that_ended_is_named_with_how_to_repair_it(tmp_path: Path, monkeypatch) -> None:
    """One module file of a set was left unfinished by a writer that was killed (the latest format: nobody can read
    it). While it may still be written it "cannot be read yet"; once it has not changed for a while the set says
    so, with how it can be repaired. The cached layout is asked again when that while has passed, without anybody
    touching the files: a watch that met the set at once must not say "cannot be read yet" for ever."""
    monkeypatch.setattr(nexus_files, "UNFINISHED_AFTER_S", 3.0)
    monkeypatch.setattr(nexus_modules, "UNFINISHED_AFTER_S", 3.0)
    paths = [_module(tmp_path / "scan_00050_m01.nxs", 1, frames=6), _killed(tmp_path / "scan_00050_m02.nxs", "plain", 2),
             _module(tmp_path / "scan_00050_m03.nxs", 3, frames=6)]
    os.utime(paths[1])  # (the writer ended just now)
    first = cached_layout(paths[0])
    assert first.kind == "module_set" and _names(first.pending) == ["scan_00050_m02.nxs"] and first.frames == 6
    assert first.warning_texts() == ("Module file m02 cannot be read yet: that part of the detector is left out",)
    assert cached_layout(paths[0]) is first
    deadline = time.time() + 15
    while cached_layout(paths[0]) is first and time.time() < deadline:
        time.sleep(0.1)
    layout = cached_layout(paths[0])
    assert layout.warning_texts() == (
        "Module file m02 was left open for writing and has not changed for a while: the program that wrote it seems "
        "to have ended without closing it, and that part of the detector is left out. Running h5clear -s on a copy "
        "of it makes readable what was stored in it",)
    assert _names(layout.pending) == ["scan_00050_m02.nxs"] and not layout.complete and layout.frames == 6
    assert _names(layout.members) == ["scan_00050_m01.nxs", "scan_00050_m03.nxs"]
    assert _names(group_sources(paths)) == ["scan_00050_m01.nxs"]  # one row, as before


def test_module_files_left_by_writers_that_ended_are_named_together(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_00051_m{module:02d}.nxs", module) for module in (1, 2)]
    paths += [_killed(tmp_path / f"scan_00051_m{module:02d}.nxs", "plain", module) for module in (3, 4)]
    _module(tmp_path / "scan_00051_m05.nxs", 5)
    for path in paths[2:]:
        _long_ago(path)
    layout = cached_layout(paths[0])
    assert layout.kind == "module_set" and _names(layout.pending) == ["scan_00051_m03.nxs", "scan_00051_m04.nxs"]
    assert layout.warnings == ((nexus_modules.MODULES_LEFT_UNFINISHED, {"names": "m03, m04"}),)
    assert layout.warning_texts()[0].startswith("Module files m03, m04 were left open for writing")
    assert not layout.complete and len(layout.members) == 3


# -- a set with fewer module files than another scan of its folder -------------------------------------------------------


def _scan(folder: Path, name: str, modules, **options) -> list[Path]:
    return [_module(folder / f"{name}_m{module:02d}.nxs", module, **options) for module in modules]


def test_a_set_with_fewer_modules_than_another_scan_says_which_may_be_missing(tmp_path: Path) -> None:
    whole = _scan(tmp_path, "scan_00001", (1, 2, 3, 4))
    short = _scan(tmp_path, "scan_00002", (1, 2))
    layout = cached_layout(short[0])  # asked first: the other scan has not been read yet
    assert layout.kind == "module_set" and len(layout.members) == 2
    assert layout.warnings == ((nexus_modules.MODULES_FEWER, {
        "count": 2, "other": "scan_00001", "others": 4, "names": "m03, m04"}),)
    assert layout.warning_texts() == (
        "2 module files, but scan_00001 in the same folder has 4: m03, m04 may be missing here "
        "(that part of the detector is then left out)",)
    assert layout.complete  # nothing in the files says a part is missing: it is a remark, the frames are read
    assert cached_layout(whole[0]).warnings == ()
    _module(tmp_path / "scan_00002_m03.nxs", 3)
    assert cached_layout(short[0]).warnings[0][1] == {"count": 3, "other": "scan_00001", "others": 4, "names": "m04"}
    _module(tmp_path / "scan_00002_m04.nxs", 4)
    assert cached_layout(short[0]).warnings == () and len(cached_layout(short[0]).members) == 4


def test_the_command_line_tool_asks_about_a_scan_with_fewer_modules(tmp_path: Path) -> None:
    """Whether a module was switched off or its file is missing is for the person to say: an open point of the
    report, like a part that is known to be missing."""
    from src.gimap.app.headless_sources import describe_source, given_sources

    _scan(tmp_path, "scan_00001", (1, 2, 3))
    short = _scan(tmp_path, "scan_00002", (1, 2))
    (source,) = given_sources([str(short[0])])
    report: dict = {}
    describe_source(report, source)
    assert report["source"]["name"] == "scan_00002" and report["source"]["complete"]
    (item,) = report["needs_attention"]
    assert item["item"] == "source" and item["why"] == (
        "2 module files, but scan_00001 in the same folder has 3: m03 may be missing here "
        "(that part of the detector is then left out).")
    left = _killed(tmp_path / "left" / "left_00001.nxs", "swmr") if (tmp_path / "left").mkdir() is None else None
    _long_ago(left)
    (source,) = given_sources([str(left)])
    report = {}
    describe_source(report, source)
    assert [item["why"] for item in report["needs_attention"]] == [
        "left_00001.nxs was not closed by the program that wrote it and has not changed for a while: it is read "
        "as it is, and its last frame may be incomplete."]


def test_a_larger_scan_that_arrives_later_is_noticed(tmp_path: Path) -> None:
    first = _scan(tmp_path, "scan_00001", (1, 2))
    assert cached_layout(first[0]).warnings == ()  # alone in its folder: nothing says how many modules there are
    time.sleep(0.05)
    _scan(tmp_path, "scan_00002", (1, 2, 3))
    assert cached_layout(first[0]).warnings[0][0] == nexus_modules.MODULES_FEWER
    assert cached_layout(first[0]).warnings[0][1]["names"] == "m03"


def test_only_a_module_set_of_the_same_detector_counts_as_evidence(tmp_path: Path) -> None:
    short = _scan(tmp_path, "scan_00002", (1, 2))
    for number in range(1, 6):  # five files only named like modules: no module positions
        with h5py.File(tmp_path / f"film_m{number}.nxs", "w") as handle:
            handle.create_dataset(f"{DET}/data", data=np.zeros((2, H, W), dtype=np.int32))
    _scan(tmp_path, "other_detector", (1, 2, 3), shape=(H + 6, W))  # a set of another module shape
    assert cached_layout(short[0]).warnings == ()


def test_asking_the_other_scan_opens_no_file_twice(tmp_path: Path, monkeypatch) -> None:
    _scan(tmp_path, "scan_00001", (1, 2, 3))
    _scan(tmp_path, "scan_00002", (1, 2))
    opened: list[str] = []
    real = h5py.File

    def counting(name, *args, **kwargs):
        opened.append(Path(str(name)).name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(nexus_files.h5py, "File", counting)
    rows = group_sources(sorted(tmp_path.iterdir(), reverse=True))  # the short scan first
    assert _names(rows) == ["scan_00002_m01.nxs", "scan_00001_m01.nxs"]
    assert sorted(opened) == sorted(path.name for path in tmp_path.iterdir())  # five files, each once
    group_sources(sorted(tmp_path.iterdir()))
    assert len(opened) == 5  # nothing changed: nothing is opened


# -- frames stored along the last axis ------------------------------------------------------------------------------------


def _one(path: Path, shape, **options) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{DET}/data", data=np.zeros(shape, dtype=np.int32), **options)
    return path


def test_frames_stored_last_are_said(tmp_path: Path) -> None:
    layout = inspect_nexus(_one(tmp_path / "matlab_export.h5", (256, 128, 10)))
    assert (layout.kind, layout.frames, layout.frame_shape) == ("stack", 256, (128, 10))  # read as it always was
    assert layout.warnings == ((nexus_layout.FRAMES_LAST, {
        "dataset": f"{DET}/data", "frames": 256, "rows": 128, "columns": 10}),)
    assert layout.warning_texts() == (
        f"{DET}/data is read as 256 frames of 128 × 10 pixels — if these are 10 frames stored last, this file is "
        "not supported yet",)
    assert load_detector_image(tmp_path / "matlab_export.h5", frame_idx=3).data.shape == (128, 10)


@pytest.mark.parametrize("shape, options, last", [
    ((12, 256, 128), {}, False),                                   # a stack of frames
    ((240, 80, 300), {}, False),                                   # many frames of a wide module
    ((300, 200, 100), {}, False),                                  # the last side is not small beside the others
    ((1000, 600, 100), {"chunks": (1, 600, 100)}, False),          # one frame per chunk: the first axis is the frames
    ((1000, 600, 100), {"maxshape": (None, 600, 100), "chunks": True}, False),  # it grows along the first axis
    ((1000, 600, 100), {}, True),                                  # nothing says: the shape does
    ((300, 200, 100), {"chunks": (300, 200, 1)}, True),            # one frame per chunk along the last axis
    ((300, 200, 100), {"maxshape": (300, 200, None), "chunks": True}, True),    # it grows along the last axis
    ((40, 40, 4000), {}, False),                                   # no detector: 40 pixels a side
])
def test_what_says_that_frames_are_stored_last(tmp_path: Path, shape, options, last: bool) -> None:
    layout = inspect_nexus(_one(tmp_path / "frames.h5", shape, **options))
    assert (nexus_layout.FRAMES_LAST in [template for template, _values in layout.warnings]) is last
    assert layout.frames == shape[0] and layout.frame_shape == shape[1:]


# -- two image datasets -----------------------------------------------------------------------------------------------------


def test_two_image_datasets_are_both_named_with_where_to_choose(tmp_path: Path) -> None:
    path = tmp_path / "two.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/saxs/data", data=np.zeros((2, H, W), dtype=np.int32))
        handle.create_dataset("/entry/waxs/data", data=np.zeros((3, H, W), dtype=np.int32))
    layout = inspect_nexus(path)
    assert layout.dataset == "/entry/saxs/data" and layout.alternatives == ("/entry/waxs/data",)
    assert layout.warnings == ((nexus_layout.SEVERAL_DATASETS, {
        "count": 2, "dataset": "/entry/saxs/data", "others": "/entry/waxs/data"}),)
    assert layout.warning_texts() == (
        "2 image datasets in this file: /entry/saxs/data is shown, not /entry/waxs/data "
        "(Format Converter and Geometry Calibration let you choose the dataset)",)
    assert inspect_nexus(path, dataset="/entry/waxs/data").warnings == ()  # chosen: nothing left to say


def test_many_image_datasets_name_the_first_few_of_the_others(tmp_path: Path) -> None:
    path = tmp_path / "many.h5"
    with h5py.File(path, "w") as handle:
        for number in range(1, 8):
            handle.create_dataset(f"/scan_{number}/image/data", data=np.zeros((2, H, W), dtype=np.int32))
    (template, values), = inspect_nexus(path).warnings
    assert template == nexus_layout.SEVERAL_DATASETS and values["count"] == 7
    assert values["others"] == "/scan_2/image/data, /scan_3/image/data, /scan_4/image/data …"
