"""What a NeXus / HDF5 source is, decided from its content (``shared/detector_io/nexus_layout.py``).

Every file is synthetic (h5py, in ``tmp_path``): detector modules with their positions, files that
are only named like modules, numbered single frames, stacks, Dectris-style masters, NeXus default
plots, renamed detector groups, several image datasets, grids of frames, pictures and line data.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.shared.detector_io import (
    AmbiguousDatasetError,
    BeingWritten,
    HDF5_SUFFIXES,
    NexusLayout,
    _dataset_candidates,
    cached_layout,
    detect_nxs_frame_count,
    forget_layout,
    group_sources,
    inspect_nexus,
    last_known_layout,
    nxs_series_paths,
    open_nexus,
    select_nxs_dataset,
)
from src.gimap.shared.detector_io import nexus_files, nexus_layout

DET = "/entry/instrument/detector"
H, W = 20, 30


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _module(path: Path, module: int, frames: int = 2, *, det: str = DET, position=None, value=None) -> Path:
    """One detector module file as the Lambda server writes it: all frames, and where the module sits."""
    with h5py.File(path, "w") as handle:
        data = np.full((frames, H, W), 10 * module if value is None else value, dtype=np.int32)
        data += np.arange(frames, dtype=np.int32)[:, None, None]
        handle.create_dataset(f"{det}/data", data=data, maxshape=(None, H, W), chunks=(1, H, W))
        offset = [0.0, (module - 1) * (H + 4), 0.0] if position is None else position
        handle.create_dataset(f"{det}/translation/distance", data=np.asarray(offset, dtype=np.float32))
    return path


def _frames_file(path: Path, data: np.ndarray, dataset: str = f"{DET}/data") -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=data)
    return path


def _eiger(folder: Path, name: str, suffix: str, chunks=(3, 3, 2), virtual: bool = False) -> Path:
    """Dectris / NXmx layout: ``<name>_master`` with links to ``<name>_data_00000k.h5`` (and a virtual dataset)."""
    master = folder / f"{name}_master{suffix}"
    total = 0
    with h5py.File(master, "w") as handle:
        handle.attrs["default"] = "entry"
        entry = handle.create_group("entry")
        entry.attrs["NX_class"], entry.attrs["default"] = "NXentry", "data"
        data = entry.create_group("data")
        data.attrs["NX_class"], data.attrs["signal"] = "NXdata", ("data" if virtual else "data_000001")
        detector = entry.create_group("instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        specific = detector.create_group("detectorSpecific")
        specific.create_dataset("pixel_mask", data=np.zeros((H, W), dtype=np.uint32))
        specific.create_dataset("flatfield", data=np.ones((H, W), dtype=np.float32))
        sources = []
        for number, count in enumerate(chunks, start=1):
            data_file = folder / f"{name}_data_{number:06d}.h5"
            with h5py.File(data_file, "w") as chunk:
                stack = np.stack([np.full((H, W), 1000 + total + k, dtype=np.uint32) for k in range(count)])
                chunk.create_dataset("/entry/data/data", data=stack, chunks=(1, H, W))
            sources.append((data_file, count))
            data[f"data_{number:06d}"] = h5py.ExternalLink(data_file.name, "/entry/data/data")
            total += count
        if virtual:
            layout = h5py.VirtualLayout(shape=(total, H, W), dtype=np.uint32)
            start = 0
            for data_file, count in sources:
                layout[start:start + count] = h5py.VirtualSource(data_file.name, "/entry/data/data", shape=(count, H, W))
                start += count
            data.create_virtual_dataset("data", layout, fillvalue=0)
    return master


def _names(paths) -> list[str]:
    return [Path(path).name for path in paths]


# -- detector modules: the names propose, the content decides ---------------------------------------------------------


def test_module_files_with_their_own_positions_are_one_stitched_source(tmp_path: Path) -> None:
    paths = [_module(tmp_path / f"scan_00001_m{module:02d}.nxs", module, frames=4) for module in (1, 2, 3)]
    layout = inspect_nexus(paths[1])  # any module file stands for the set
    assert isinstance(layout, NexusLayout) and layout.kind == "module_set"
    assert layout.path == paths[0].resolve() and _names(layout.members) == _names(paths)
    assert (layout.frames, layout.member_frames, layout.frame_shape) == (4, (4, 4, 4), (H, W))
    assert layout.dataset == f"{DET}/data" and layout.detector_group == DET and layout.orientation == "p03_lambda"
    assert layout.complete and layout.warnings == ()
    assert layout.summary() == ("{modules} module files · {frames} frames · stitched", {"modules": 3, "frames": 4})
    assert layout.summary_text() == "3 module files · 4 frames · stitched"
    assert [stamp[0] for stamp in layout.signature] == [str(member) for member in layout.members]
    assert "module position" in layout.decided_by
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == ["scan_00001_m01.nxs"]
    assert _names(nxs_series_paths(paths[2])) == _names(paths) and detect_nxs_frame_count(paths[2]) == 4


def test_unequal_modules_give_the_frames_every_module_has_and_say_which_is_short(tmp_path: Path) -> None:
    for module, frames in ((1, 5), (2, 3), (3, 5)):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module, frames)
    layout = inspect_nexus(tmp_path / "scan_m01.nxs")
    assert layout.kind == "module_set" and layout.frames == 3 and layout.member_frames == (5, 3, 5)
    assert not layout.complete
    assert layout.warnings == ((nexus_layout.MODULES_UNEQUAL, {"detail": "m02: 3", "full": 5, "frames": 3}),)
    assert layout.warning_texts() == (
        "Module files hold different numbers of frames (m02: 3; the others 5): frames after frame 3 are left out",)
    assert detect_nxs_frame_count(tmp_path / "scan_m03.nxs") == 3  # not the 5 of the file asked


def test_a_gap_in_the_module_numbers_is_said(tmp_path: Path) -> None:
    for module in (1, 2, 4, 6):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module)
    layout = inspect_nexus(tmp_path / "scan_m02.nxs")
    assert layout.kind == "module_set" and len(layout.members) == 4 and not layout.complete
    assert layout.warnings == ((nexus_layout.MODULES_MISSING, {"names": "m03, m05"}),)
    (tmp_path / "scan_m04.nxs").unlink()
    (tmp_path / "scan_m06.nxs").unlink()
    _module(tmp_path / "scan_m03.nxs", 3)
    first_gone = inspect_nexus(tmp_path / "scan_m03.nxs")
    assert first_gone.warnings == () and first_gone.complete
    (tmp_path / "scan_m01.nxs").unlink()
    layout = inspect_nexus(tmp_path / "scan_m03.nxs")
    assert layout.path.name == "scan_m02.nxs"
    assert layout.warning_texts() == ("Module file m01 is missing: that part of the detector is left out",)


def test_two_scans_in_one_folder_are_two_sources(tmp_path: Path) -> None:
    for scan, frames in ((1, 2), (2, 3)):
        for module in (1, 2, 3):
            _module(tmp_path / f"scan_{scan:05d}_m{module:02d}.nxs", module, frames)
    rows = group_sources(sorted(tmp_path.iterdir()))
    assert _names(rows) == ["scan_00001_m01.nxs", "scan_00002_m01.nxs"]
    assert [cached_layout(row).frames for row in rows] == [2, 3]
    assert all(cached_layout(row).kind == "module_set" and len(cached_layout(row).members) == 3 for row in rows)


@pytest.mark.parametrize("names", [
    ("film1.nxs", "film2.nxs", "film3.nxs"),
    ("beam3.nxs", "beam4.nxs"),
    ("run_m1.nxs", "run_m2.nxs", "run_m3.nxs"),
    ("sample_10nm1.nxs", "sample_10nm2.nxs"),
    ("PS_PMMA_m1.nxs", "PS_PMMA_m2.nxs", "PS_PMMA_m10.nxs"),
])
def test_files_only_named_like_modules_stay_separate_frames(tmp_path: Path, names) -> None:
    for index, name in enumerate(names):
        _frames_file(tmp_path / name, np.full((1, H, W), 300 + index, dtype=np.int32))
    rows = group_sources(sorted(tmp_path.iterdir()))
    assert sorted(_names(rows)) == sorted(names)
    for name in names:
        layout = inspect_nexus(tmp_path / name)
        assert layout.kind == "single_frame" and _names(layout.members) == [name] and layout.warnings == ()
        assert layout.orientation == "as_stored"
        assert nxs_series_paths(tmp_path / name) == [tmp_path / name]


def test_files_with_the_same_position_are_not_a_module_set(tmp_path: Path) -> None:
    """Two scans of one single-module detector, named scan1 / scan2: same position, so two frames."""
    for index in (1, 2):
        _module(tmp_path / f"beam{index}.nxs", index, position=[5.0, 7.0, 0.0])
    layouts = [inspect_nexus(tmp_path / f"beam{index}.nxs") for index in (1, 2)]
    assert [layout.kind for layout in layouts] == ["stack", "stack"]
    said = ((nexus_layout.SAME_POSITION, {"first": "beam1.nxs", "second": "beam2.nxs"}),)
    assert layouts[0].warnings == said and layouts[1].warnings == said  # a Lambda file: said, not guessed
    assert layouts[0].warning_texts() == (
        "beam1.nxs and beam2.nxs are named like modules of one detector but hold the same module position: "
        "each file is read on its own",)
    assert layouts[0].orientation == "p03_lambda"
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == ["beam1.nxs", "beam2.nxs"]
    # The verdict stands on those two files: when one of them is written again with its own position, they are a set.
    _module(tmp_path / "beam2.nxs", 2, position=[5.0, 40.0, 0.0])
    assert cached_layout(tmp_path / "beam1.nxs").kind == "module_set"
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == ["beam1.nxs"]


def test_a_module_of_another_shape_or_without_the_dataset_breaks_the_set(tmp_path: Path) -> None:
    _module(tmp_path / "scan_m01.nxs", 1)
    _module(tmp_path / "scan_m02.nxs", 2)
    with h5py.File(tmp_path / "scan_m03.nxs", "w") as handle:
        handle.create_dataset(f"{DET}/data", data=np.zeros((2, H + 2, W), dtype=np.int32))
        handle.create_dataset(f"{DET}/translation/distance", data=[0.0, 99.0, 0.0])
    layout = inspect_nexus(tmp_path / "scan_m01.nxs")
    assert layout.kind == "stack" and layout.warnings == ((nexus_layout.NOT_A_MODULE, {"name": "scan_m03.nxs"}),)
    assert layout.warning_texts() == (
        "scan_m03.nxs is named like a detector module of this scan but does not fit the set "
        "(dataset, module shape or module position): each file is read on its own",)
    assert len(group_sources(sorted(tmp_path.iterdir()))) == 3
    _frames_file(tmp_path / "scan_m03.nxs", np.zeros((2, H, W), dtype=np.int32), "/entry/other/image")  # no such dataset
    assert cached_layout(tmp_path / "scan_m02.nxs").warnings == layout.warnings
    (tmp_path / "scan_m03.nxs").unlink()  # the file that did not fit is gone: the other two are a set
    assert cached_layout(tmp_path / "scan_m02.nxs").kind == "module_set"


def test_a_module_that_cannot_be_read_yet_is_left_out_and_said(tmp_path: Path) -> None:
    for module in (1, 2):
        _module(tmp_path / f"scan_m{module:02d}.nxs", module)
    copying = tmp_path / "scan_m03.nxs"
    copying.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)  # the copy of the third module has only begun
    layout = cached_layout(tmp_path / "scan_m01.nxs")
    assert layout.kind == "module_set" and _names(layout.members) == ["scan_m01.nxs", "scan_m02.nxs"]
    assert layout.pending == (copying.resolve(),) and not layout.complete
    assert layout.warning_texts() == ("Module file m03 cannot be read yet: that part of the detector is left out",)
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == ["scan_m01.nxs"]  # not a row of its own
    _module(copying, 3)
    layout = cached_layout(tmp_path / "scan_m01.nxs")  # read again although no member changed
    assert len(layout.members) == 3 and layout.pending == () and layout.complete and layout.warnings == ()


def test_links_that_lead_back_do_not_trap_the_search(tmp_path: Path) -> None:
    path = tmp_path / "loops.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{DET}/data", data=np.zeros((2, H, W), dtype=np.int32))
        handle["/entry/loop"] = h5py.SoftLink("/entry")
        handle[f"{DET}/up"] = handle["/entry"]
        handle["/entry/again"] = h5py.ExternalLink("loops.nxs", "/entry")
        handle["/entry/data/data"] = handle[f"{DET}/data"]  # NXdata-style second name of the same dataset
        handle["/entry/nowhere"] = h5py.SoftLink("/entry/not/there")
    layout = inspect_nexus(path)
    assert (layout.kind, layout.frames, layout.dataset, layout.detector_group) == ("stack", 2, f"{DET}/data", DET)
    assert layout.alternatives == () and layout.warnings == ()


@pytest.mark.parametrize("stem", ["sample[A]", "run (2) {final}", "a[!b]c", "x[0-9]y"])  # names Windows allows too
def test_glob_characters_in_names_do_not_break_the_sibling_search(tmp_path: Path, stem: str) -> None:
    paths = [_module(tmp_path / f"{stem}_m{module:02d}.nxs", module) for module in (1, 2)]
    layout = inspect_nexus(paths[1])
    assert layout.kind == "module_set" and _names(layout.members) == _names(paths)
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == [paths[0].name]


def test_module_names_without_an_underscore_and_another_detector_group(tmp_path: Path) -> None:
    det = "/scan/instrument/lambda"
    paths = [_module(tmp_path / f"run00007m{module:02d}.nxs", module, det=det) for module in (1, 2, 3)]
    layout = inspect_nexus(paths[0])
    assert layout.kind == "module_set" and len(layout.members) == 3
    assert (layout.dataset, layout.detector_group, layout.entry) == (f"{det}/data", det, "/scan")


# -- one frame per file, one stack per file ------------------------------------------------------------------------------


def test_numbered_single_frames_are_one_source_each(tmp_path: Path) -> None:
    for index in range(1, 5):  # (1, H, W) and plain (H, W) frames
        frame = np.full((H, W), index, dtype=np.int32)
        _frames_file(tmp_path / f"scan_{index:05d}.nxs", frame[None] if index % 2 else frame)
    rows = group_sources(sorted(tmp_path.iterdir()))
    assert _names(rows) == [f"scan_{index:05d}.nxs" for index in range(1, 5)]
    for row in rows:
        layout = cached_layout(row)
        assert (layout.kind, layout.frames, layout.frame_shape) == ("single_frame", 1, (H, W))
        assert layout.summary() == ("1 file · 1 frame", {})


def test_a_single_file_stack_holds_the_whole_series(tmp_path: Path) -> None:
    path = _frames_file(tmp_path / "insitu.nxs", np.zeros((6, H, W), dtype=np.int32))
    layout = inspect_nexus(path)
    assert (layout.kind, layout.frames, layout.member_frames, layout.lead_shape) == ("stack", 6, (6,), (6,))
    assert layout.members == (path.resolve(),) and layout.orientation == "as_stored" and layout.dtype == "int32"
    assert layout.summary() == ("1 file · {frames} frames", {"frames": 6}) and layout.complete
    assert "detector group" in layout.decided_by


@pytest.mark.parametrize("suffix", HDF5_SUFFIXES)
def test_every_hdf5_suffix_is_inspected(tmp_path: Path, suffix: str) -> None:
    path = _frames_file(tmp_path / f"stack{suffix}", np.zeros((3, H, W), dtype=np.int32))
    assert inspect_nexus(path).frames == 3 and detect_nxs_frame_count(path) == 3
    assert group_sources([path]) == [path.resolve()]


def test_other_formats_pass_through_group_sources_untouched(tmp_path: Path) -> None:
    cbf, tif = tmp_path / "a_m01.cbf", tmp_path / "b.tif"  # never opened: they need not exist
    stack = _frames_file(tmp_path / "c.h5", np.zeros((2, H, W), dtype=np.int32))
    assert group_sources([cbf, stack, tif, cbf]) == [cbf, stack.resolve(), tif]
    assert detect_nxs_frame_count(cbf) == 1 and nxs_series_paths(cbf) == [cbf]


# -- masters whose frames are in data files ---------------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".h5", ".nxs"])
@pytest.mark.parametrize("virtual", [False, True])
def test_an_eiger_master_is_one_source_with_all_its_frames(tmp_path: Path, suffix: str, virtual: bool) -> None:
    master = _eiger(tmp_path, "run7", suffix, virtual=virtual)
    layout = inspect_nexus(master)
    assert layout.kind == "chunked_master" and layout.frames == 8 and layout.frame_shape == (H, W)
    assert _names(layout.members) == [master.name, "run7_data_000001.h5", "run7_data_000002.h5", "run7_data_000003.h5"]
    assert layout.detector_group == DET and layout.orientation == "as_stored" and layout.complete
    assert layout.summary() == ("master + {files} data files · {frames} frames", {"files": 3, "frames": 8})
    if virtual:
        assert layout.dataset == "/entry/data/data" and layout.chunks == () and "virtual dataset" in layout.decided_by
    else:  # the pixel mask and the flat field of the master are never the image
        assert layout.dataset == "/entry/data/data_000001" and layout.member_frames == (3, 3, 2)
        assert layout.chunks == (("/entry/data/data_000001", 3), ("/entry/data/data_000002", 3), ("/entry/data/data_000003", 2))
        assert layout.alternatives == ()
    # The data files are not rows of their own, whatever the order they are listed in.
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == [master.name]
    assert _names(group_sources([tmp_path / "run7_data_000002.h5"])) == [master.name]
    assert len(layout.signature) == 4  # a data file that grows changes the signature


def test_a_missing_data_file_stops_the_frames_before_it(tmp_path: Path) -> None:
    master = _eiger(tmp_path, "run8", ".h5")
    (tmp_path / "run8_data_000002.h5").unlink()
    layout = cached_layout(master)
    assert layout.kind == "chunked_master" and layout.frames == 3 and not layout.complete
    assert layout.warning_texts() == ("Data file run8_data_000002.h5 is missing: frames after frame 3 cannot be read",)
    virtual = _eiger(tmp_path / "v", "run9", ".h5", virtual=True) if (tmp_path / "v").mkdir() is None else None
    (tmp_path / "v" / "run9_data_000003.h5").unlink()
    layout = inspect_nexus(virtual)
    assert layout.frames == 6 and layout.warnings[0][0] == nexus_layout.DATA_FILE_MISSING
    # The file arrives: the layout is read again although the master itself did not change.
    with h5py.File(tmp_path / "run8_data_000002.h5", "w") as chunk:
        chunk.create_dataset("/entry/data/data", data=np.zeros((3, H, W), dtype=np.uint32))
    assert cached_layout(master).frames == 8 and cached_layout(master).complete


def test_a_master_whose_data_files_have_not_arrived_is_listed_with_no_frames(tmp_path: Path) -> None:
    master = _eiger(tmp_path, "run5", ".h5")
    for data_file in tmp_path.glob("run5_data_*.h5"):
        data_file.unlink()
    layout = cached_layout(master)
    assert (layout.kind, layout.frames, layout.complete) == ("chunked_master", 0, False)
    assert layout.summary_text() == "master + 0 data files · 0 frames"
    assert layout.warning_texts() == (
        "Data files run5_data_000001.h5, run5_data_000002.h5, run5_data_000003.h5 are missing: "
        "frames after frame 0 cannot be read",)
    assert _names(group_sources([master])) == [master.name]
    with h5py.File(tmp_path / "run5_data_000001.h5", "w") as chunk:
        chunk.create_dataset("/entry/data/data", data=np.zeros((3, H, W), dtype=np.uint32))
    layout = cached_layout(master)
    assert layout.frames == 3 and layout.summary_text() == "master + 1 data file · 3 frames"


def test_one_external_link_to_a_detector_file_is_followed(tmp_path: Path) -> None:
    """MAX IV style: ``scan-0001.h5`` links to ``scan-0001_eiger.h5``."""
    _frames_file(tmp_path / "scan-0001_eiger.h5", np.zeros((5, H, W), dtype=np.uint32), "/entry/instrument/eiger/data")
    master = tmp_path / "scan-0001.h5"
    with h5py.File(master, "w") as handle:
        handle["/entry/measurement/eiger"] = h5py.ExternalLink("scan-0001_eiger.h5", "/entry/instrument/eiger/data")
    layout = inspect_nexus(master)
    assert layout.kind == "chunked_master" and layout.frames == 5
    assert layout.summary() == ("master + 1 data file · {frames} frames", {"frames": 5})
    assert _names(group_sources([tmp_path / "scan-0001_eiger.h5", master])) == [master.name]


# -- which dataset is the image -------------------------------------------------------------------------------------------------


def test_the_default_plot_of_the_file_decides_between_two_detectors(tmp_path: Path) -> None:
    path = tmp_path / "saxs_waxs.nxs"
    with h5py.File(path, "w") as handle:
        handle.attrs["default"] = "entry"
        entry = handle.create_group("entry")
        entry.attrs["NX_class"], entry.attrs["default"] = "NXentry", "waxs"
        for name, frames in (("saxs", 4), ("waxs", 7)):
            group = entry.create_group(name)
            group.attrs["NX_class"], group.attrs["signal"] = "NXdata", "data"
            group.create_dataset("data", data=np.zeros((frames, H, W), dtype=np.int32))
    layout = inspect_nexus(path)
    assert layout.dataset == "/entry/waxs/data" and layout.frames == 7 and layout.alternatives == ("/entry/saxs/data",)
    assert layout.warnings == () and "default plot" in layout.decided_by
    with open_nexus(path) as handle:
        assert select_nxs_dataset(handle) == "/entry/waxs/data"
    assert inspect_nexus(path, dataset="/entry/saxs/data").frames == 4  # the caller's choice wins


def test_the_signal_of_an_nxdata_group_is_the_image_not_its_normalisation(tmp_path: Path) -> None:
    path = tmp_path / "signal.nxs"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/result")
        group.attrs["NX_class"], group.attrs["signal"] = "NXdata", "pil300k_image"
        group.create_dataset("pil300k_image", data=np.zeros((4, H, W), dtype=np.int32))
        group.create_dataset("normalisation", data=np.ones((4, H, W), dtype=np.float32))
        group.create_dataset("pil300k_image_errors", data=np.ones((4, H, W), dtype=np.float32))
        group.create_dataset("flat_field", data=np.ones((H, W), dtype=np.float32))
        group.create_dataset("dark/data", data=np.ones((H, W), dtype=np.float32))
        group.create_dataset("pixel_mask", data=np.ones((H, W), dtype=np.uint8))
    layout = inspect_nexus(path)
    assert layout.dataset == "/entry/result/pil300k_image" and layout.alternatives == () and layout.warnings == ()
    assert "signal" in layout.decided_by
    # A caller may still ask for one of them.
    assert inspect_nexus(path, dataset="/entry/result/flat_field").dataset == "/entry/result/flat_field"


def test_a_detector_group_under_any_name_brings_its_own_mask_and_position(tmp_path: Path) -> None:
    path = tmp_path / "renamed.h5"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("scan_0001")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/pilatus")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("data", data=np.zeros((5, H, W), dtype=np.int32))
        plot = entry.create_group("plot")  # the NXdata link to the same dataset is the same image
        plot.attrs["NX_class"], plot.attrs["signal"] = "NXdata", "data"
        plot["data"] = h5py.SoftLink("/scan_0001/instrument/pilatus/data")
    layout = inspect_nexus(path)
    assert (layout.dataset, layout.detector_group, layout.entry) == (
        "/scan_0001/instrument/pilatus/data", "/scan_0001/instrument/pilatus", "/scan_0001")
    assert layout.alternatives == () and layout.frames == 5


def test_several_image_datasets_use_the_best_and_name_the_others(tmp_path: Path) -> None:
    path = tmp_path / "bliss.h5"
    with h5py.File(path, "w") as handle:
        for scan, frames in (("2.1", 3), ("1.1", 2), ("10.1", 4)):
            detector = handle.create_group(f"{scan}/instrument/eiger")
            handle[scan].attrs["NX_class"], detector.attrs["NX_class"] = "NXentry", "NXdetector"
            detector.create_dataset("data", data=np.zeros((frames, H, W), dtype=np.int32))
    layout = inspect_nexus(path)
    assert layout.dataset == "/1.1/instrument/eiger/data" and layout.frames == 2  # natural order: 1.1, 2.1, 10.1
    assert layout.alternatives == ("/2.1/instrument/eiger/data", "/10.1/instrument/eiger/data")
    # The file does not say which is the image: the one shown, the others, and where a person can choose.
    assert layout.warnings == ((nexus_layout.SEVERAL_DATASETS, {
        "count": 3, "dataset": "/1.1/instrument/eiger/data",
        "others": "/2.1/instrument/eiger/data, /10.1/instrument/eiger/data"}),)
    assert layout.warning_texts() == (
        "3 image datasets in this file: /1.1/instrument/eiger/data is shown, not /2.1/instrument/eiger/data, "
        "/10.1/instrument/eiger/data (Format Converter and Geometry Calibration let you choose the dataset)",)
    with open_nexus(path) as handle:
        with pytest.raises(AmbiguousDatasetError) as caught:
            select_nxs_dataset(handle)
        assert caught.value.paths == [layout.dataset, *layout.alternatives]
        assert select_nxs_dataset(handle, "/10.1/instrument/eiger/data") == "/10.1/instrument/eiger/data"
        assert [path for _score, path in _dataset_candidates(handle)] == [layout.dataset, *layout.alternatives]
    chosen = inspect_nexus(path, dataset="/10.1/instrument/eiger/data")
    assert chosen.frames == 4 and chosen.warnings == () and chosen.decided_by == "the dataset asked for"
    wrong = inspect_nexus(path, dataset="/nowhere/data")
    assert wrong.dataset == layout.dataset and wrong.warnings[0] == (
        nexus_layout.DATASET_NOT_FOUND, {"wanted": "/nowhere/data", "dataset": layout.dataset})


def test_an_image_is_preferred_to_a_table_of_counters(tmp_path: Path) -> None:
    path = tmp_path / "tables.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/measurement/mca", data=np.ones((400, 4096), dtype=np.int32))
        handle.create_dataset("/entry/measurement/pilatus", data=np.ones((1, 64, 96), dtype=np.int32))
        handle.create_dataset("/entry/measurement/positions", data=np.ones((40, 8)))  # too small to be an image
    layout = inspect_nexus(path)
    assert layout.dataset == "/entry/measurement/pilatus" and layout.alternatives == ("/entry/measurement/mca",)


def test_a_grid_of_frames_is_one_flat_series(tmp_path: Path) -> None:
    path = _frames_file(tmp_path / "mesh.nxs", np.zeros((3, 4, H, W), dtype=np.int32))
    layout = inspect_nexus(path)
    assert (layout.kind, layout.frames, layout.lead_shape, layout.frame_shape) == ("stack", 12, (3, 4), (H, W))
    assert layout.warning_texts() == ("A 3×4 grid of frames is read in storage order as 12 frames",)


@pytest.mark.parametrize("shape, dtype", [
    ((96, 128, 3), np.uint8),      # an RGB picture at the detector's place
    ((96, 128, 4), np.uint8),      # RGBA
    ((7, 1, 128), np.int32),       # a line detector
    ((500,), np.float64),          # a curve
    ((5, 2, 3, 40, 60), np.int32),  # five axes
])
def test_pictures_and_line_data_are_not_frames(tmp_path: Path, shape, dtype) -> None:
    path = _frames_file(tmp_path / "other.nxs", np.ones(shape, dtype=dtype))
    layout = inspect_nexus(path)
    assert layout.kind == "no_image" and layout.frames == 0 and layout.dataset == ""
    assert layout.summary() == ("no detector image", {})
    assert group_sources([path]) == [] and group_sources([path], keep_without_image=True) == [path.resolve()]
    with pytest.raises(ValueError, match="other.nxs holds no detector image"):
        detect_nxs_frame_count(path)
    with open_nexus(path) as handle:
        with pytest.raises(ValueError, match="No usable detector image dataset"):
            select_nxs_dataset(handle)


def test_model_weights_and_results_are_not_listed_from_a_folder(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "model.h5", "w") as handle:
        handle.create_dataset("model_weights/dense/kernel", data=np.zeros((256, 128), dtype=np.float32))
        handle.create_dataset("model_weights/dense/bias", data=np.zeros(128, dtype=np.float32))
    with h5py.File(tmp_path / "results.h5", "w") as handle:
        handle.create_dataset("fit/parameters", data=np.zeros((240, 6)))
    _frames_file(tmp_path / "frame.h5", np.zeros((H, W), dtype=np.float32), "/entry/data/data")
    (tmp_path / "broken.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    rows = group_sources(sorted(tmp_path.iterdir()))
    # The weights are image-shaped numbers, but the file says it is a model; the broken file stays, reading it says why.
    assert _names(rows) == ["broken.h5", "frame.h5"]
    assert inspect_nexus(tmp_path / "model.h5").kind == "no_image" and inspect_nexus(tmp_path / "results.h5").kind == "no_image"
    with pytest.raises(OSError):
        inspect_nexus(tmp_path / "broken.h5")
    assert nxs_series_paths(tmp_path / "broken.h5") == [tmp_path / "broken.h5"]


# -- the cache: valid while the files are unchanged -----------------------------------------------------------------------------


def _count_opens(monkeypatch) -> list[str]:
    opened: list[str] = []
    real = h5py.File

    def counting(name, *args, **kwargs):
        opened.append(Path(str(name)).name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(nexus_files.h5py, "File", counting)
    return opened


def test_a_folder_is_listed_once_and_each_file_opened_once(tmp_path: Path, monkeypatch) -> None:
    for scan in (1, 2, 3):
        for module in range(1, 6):
            _module(tmp_path / f"scan_{scan:05d}_m{module:02d}.nxs", module)
    _eiger(tmp_path, "run7", ".h5")
    for index in (1, 2, 3):
        _frames_file(tmp_path / f"film{index}.nxs", np.zeros((1, H, W), dtype=np.int32))
    files = sorted(tmp_path.iterdir())
    opened = _count_opens(monkeypatch)
    listings: list[str] = []
    real_listdir = os.listdir
    monkeypatch.setattr(nexus_files.os, "listdir", lambda folder: listings.append(str(folder)) or real_listdir(folder))

    rows = group_sources(files)

    assert _names(rows) == ["film1.nxs", "film2.nxs", "film3.nxs", "run7_master.h5",
                            "scan_00001_m01.nxs", "scan_00002_m01.nxs", "scan_00003_m01.nxs"]
    assert len(listings) == 1
    # 15 module files + 3 single frames + the master: once each (HDF5 opens the linked data files itself).
    assert sorted(opened) == sorted(path.name for path in files if "_data_" not in path.name)
    opened.clear()
    # The folder changed a moment ago: it is listed once more (a file may have arrived), nothing is opened.
    assert group_sources(files) == rows and opened == [] and len(listings) == 2
    # A folder that has been quiet: asking again costs os.stat calls only.
    quiet = time.time_ns() - 60_000_000_000
    os.utime(tmp_path, ns=(quiet, quiet))
    assert group_sources(files) == rows and len(listings) == 3
    assert group_sources(files) == rows and [cached_layout(row).kind for row in rows].count("module_set") == 3
    assert opened == [] and len(listings) == 3


def test_files_named_alike_that_are_no_module_set_are_opened_once_each(tmp_path: Path, monkeypatch) -> None:
    """A film series of one single-module Lambda: ``film1.nxs`` … all at the same position (never N × N opens)."""
    count = 40
    files = [_module(tmp_path / f"film{index}.nxs", 1, frames=1, position=[0.0, 0.0, 0.0]) for index in range(1, count + 1)]
    opened = _count_opens(monkeypatch)
    stats: list[str] = []
    real_stat = os.stat
    monkeypatch.setattr(nexus_files.os, "stat", lambda path, *args, **kwargs: stats.append(str(path)) or real_stat(path, *args, **kwargs))

    rows = group_sources(files)

    assert _names(rows) == _names(files) and all(cached_layout(row).kind == "single_frame" for row in rows)
    assert cached_layout(rows[7]).warnings == ((nexus_layout.SAME_POSITION, {"first": "film1.nxs", "second": "film2.nxs"}),)
    assert len(opened) <= count + 1, len(opened)  # each file once (film2 once more, as the witness of film1)
    opened.clear()
    stats.clear()
    assert group_sources(files) == rows and opened == []
    assert len(stats) <= 4 * count, len(stats)  # asking again: a few os.stat calls per file, not one per pair of files
    # One more file arrives (a watched folder): the files already known are not opened again.
    files.append(_module(tmp_path / f"film{count + 1}.nxs", 1, frames=1, position=[0.0, 0.0, 0.0]))
    opened.clear()  # (writing the file was counted too)
    stats.clear()
    rows = group_sources(files)
    assert len(rows) == count + 1 and _names(opened) == [f"film{count + 1}.nxs"]
    assert len(stats) <= 4 * count + 8, len(stats)
    assert cached_layout(rows[-1]).warnings == cached_layout(rows[0]).warnings


def test_the_modules_of_a_set_are_opened_once_when_asked_one_after_the_other(tmp_path: Path, monkeypatch) -> None:
    paths = [_module(tmp_path / f"scan_m{module:02d}.nxs", module) for module in range(1, 7)]
    opened = _count_opens(monkeypatch)
    layouts = [cached_layout(path) for path in reversed(paths)]  # the last module first, as a watcher may meet them
    assert all(layout is layouts[0] and layout.kind == "module_set" for layout in layouts)
    assert sorted(opened) == _names(paths)


def test_a_growing_file_invalidates_the_cached_layout(tmp_path: Path, monkeypatch) -> None:
    paths = [_module(tmp_path / f"insitu_m{module:02d}.nxs", module, frames=3) for module in (1, 2)]
    first = cached_layout(paths[0])
    assert first.frames == 3 and cached_layout(paths[1]) is first  # every module: the same remembered set
    for count, path in ((8, paths[0]), (5, paths[1])):  # the modules are not written at the same moment
        with h5py.File(path, "a") as handle:
            handle[f"{DET}/data"].resize(count, axis=0)
    grown = cached_layout(paths[0])
    assert grown is not first and grown.frames == 5 and grown.member_frames == (8, 5) and not grown.complete
    assert grown.signature != first.signature and detect_nxs_frame_count(paths[1]) == 5
    opened = _count_opens(monkeypatch)
    assert cached_layout(paths[0]) is grown and opened == []


def test_a_module_that_arrives_later_joins_the_set(tmp_path: Path) -> None:
    _module(tmp_path / "scan_m02.nxs", 2)
    assert cached_layout(tmp_path / "scan_m02.nxs").kind == "stack"
    _module(tmp_path / "scan_m03.nxs", 3)
    layout = cached_layout(tmp_path / "scan_m02.nxs")
    assert layout.kind == "module_set" and _names(layout.members) == ["scan_m02.nxs", "scan_m03.nxs"]
    _module(tmp_path / "scan_m01.nxs", 1)
    layout = cached_layout(tmp_path / "scan_m02.nxs")
    assert layout.path.name == "scan_m01.nxs" and len(layout.members) == 3 and layout.complete
    assert _names(group_sources(sorted(tmp_path.iterdir()))) == ["scan_m01.nxs"]
    (tmp_path / "scan_m03.nxs").unlink()
    assert len(cached_layout(tmp_path / "scan_m01.nxs").members) == 2


def test_a_failed_inspection_is_not_remembered(tmp_path: Path) -> None:
    path = tmp_path / "copying.nxs"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)  # the copy has only begun
    with pytest.raises(OSError):
        cached_layout(path)
    assert last_known_layout(path) is None
    _frames_file(path, np.zeros((4, H, W), dtype=np.int32))
    assert cached_layout(path).frames == 4 and last_known_layout(path).frames == 4
    forget_layout(path)
    assert last_known_layout(path) is None


def test_the_memory_of_layouts_is_bounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nexus_layout, "_REMEMBERED", 8)
    paths = [_frames_file(tmp_path / f"frame_{index:03d}.h5", np.zeros((H, W), dtype=np.int32)) for index in range(12)]
    for path in paths:
        cached_layout(path)
    assert len(nexus_layout._CACHE) <= 8 and last_known_layout(paths[0]) is None
    assert last_known_layout(paths[-1]).frames == 1 and cached_layout(paths[0]).frames == 1


# -- files a writer still holds ----------------------------------------------------------------------------------------------------

_WRITER = """
    import sys, time
    import h5py
    path, mode = sys.argv[1], sys.argv[2]
    with h5py.File(path, "w", libver="latest") as handle:
        data = handle.create_dataset("/entry/instrument/detector/data", shape=(0, 16, 24), maxshape=(None, 16, 24),
                                     dtype="i4", chunks=(1, 16, 24))
        if mode == "swmr":
            handle.swmr_mode = True
        for index in range(6):
            data.resize(index + 1, axis=0)
            data[index] = index
            data.flush() if mode == "swmr" else handle.flush()
        print("six frames written", flush=True)
        sys.stdin.readline()   # holds the file until the test says so
        data.resize(9, axis=0)
        data[6:] = 7
"""


def _writer(path: Path, mode: str) -> subprocess.Popen:
    process = subprocess.Popen([sys.executable, "-c", textwrap.dedent(_WRITER), str(path), mode],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "six frames written"
    return process


def _release(process: subprocess.Popen) -> None:
    process.communicate("\n", timeout=30)  # lets the writer finish, and closes the pipes
    assert process.returncode == 0


def test_a_file_its_writer_holds_is_being_written_and_never_remembered_as_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "live_00001.nxs"
    writer = _writer(path, "plain")  # the latest file format without SWMR: nobody else can read it
    try:
        with pytest.raises(BeingWritten) as caught:
            cached_layout(path)
        assert caught.value.path == path and isinstance(caught.value, OSError)
        assert str(caught.value) == "live_00001.nxs is being written by another program and cannot be read until it is closed."
        with pytest.raises(BeingWritten):
            detect_nxs_frame_count(path)
        assert last_known_layout(path) is None
        assert group_sources([path]) == [path.resolve()]  # listed all the same
    finally:
        _release(writer)
    layout = cached_layout(path)
    assert layout.frames == 9 and layout.complete and layout.warnings == ()


def test_a_file_written_in_swmr_mode_is_read_while_it_grows(tmp_path: Path) -> None:
    path = tmp_path / "swmr_00001.nxs"
    writer = _writer(path, "swmr")
    try:
        layout = cached_layout(path)
        assert layout.kind == "stack" and layout.frames == 6 and not layout.complete
        assert layout.warnings == ((nexus_layout.STILL_WRITTEN, {"name": "swmr_00001.nxs"}),)
    finally:
        _release(writer)
    deadline = time.time() + 5
    while cached_layout(path).frames != 9 and time.time() < deadline:  # the size on disk says it changed
        time.sleep(0.05)
    layout = cached_layout(path)
    assert layout.frames == 9 and layout.complete and layout.warnings == ()
