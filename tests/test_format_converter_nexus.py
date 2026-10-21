"""The Format Converter on NeXus / HDF5 sources of every layout.

One source is listed once and each of its frames is written once: a set of module files however many of
its files are given, a master with its data files, a stack in one file, the converter's own multi-frame
container. Every HDF5 suffix is an input. Synthetic files only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

from src.gimap.app import AppContext
from src.gimap.features.format_converter.bootstrap import create_format_converter_view_model
from src.gimap.features.format_converter.domain.models import (
    HDF5_TYPES,
    SUPPORTED_SUFFIXES,
    ConversionOptions,
    ConversionRequest,
)
from src.gimap.features.format_converter.domain.rules import (
    is_supported_input_path,
    select_source_frame_indices,
    visible_output_formats,
)
from src.gimap.features.format_converter.infrastructure.adapters import (
    LocalConversionExecutor,
    LocalSourceRepository,
)
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from src.gimap.shared.detector_io import HDF5_SUFFIXES, forget_layout, load_detector_image

DET = "/entry/instrument/detector"
H, W = 20, 30


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def _module(path: Path, module: int, frames: int, position=None) -> Path:
    """One detector module file as the Lambda server writes it; frame k of module m holds 100·m + k."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.full((frames, H, W), 100 * module, dtype=np.int32) + np.arange(frames, dtype=np.int32)[:, None, None]
        handle.create_dataset(f"{DET}/data", data=data, chunks=(1, H, W))
        offset = [0.0, (module - 1) * (H + 4), 0.0] if position is None else position
        handle.create_dataset(f"{DET}/translation/distance", data=np.asarray(offset, dtype=np.float32))
    return path


def _module_set(folder: Path, frames: int = 4, modules=(1, 2, 3), stem: str = "scan_00007") -> list[Path]:
    return [_module(folder / f"{stem}_m{module:02d}.nxs", module, frames) for module in modules]


def _stack(path: Path, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.arange(frames, dtype=np.uint16)[:, None, None] + np.zeros((frames, H, W), dtype=np.uint16) + 7
        handle.create_dataset("/entry/data/data", data=data)
    return path


def _eiger(folder: Path, name: str = "run7", chunks=(3, 3, 2)) -> Path:
    """A Dectris-style master with links to its data files; frame k of the series holds 1000 + k."""
    folder.mkdir(parents=True, exist_ok=True)
    master = folder / f"{name}_master.h5"
    total = 0
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        detector = entry.create_group("instrument/detector")
        detector.attrs["NX_class"] = "NXdetector"
        specific = detector.create_group("detectorSpecific")
        specific.create_dataset("pixel_mask", data=np.zeros((H, W), dtype=np.uint32))
        specific.create_dataset("flatfield", data=np.ones((H, W), dtype=np.float32))
        for number, count in enumerate(chunks, start=1):
            data_file = folder / f"{name}_data_{number:06d}.h5"
            with h5py.File(data_file, "w") as chunk:
                stack = np.stack([np.full((H, W), 1000 + total + k, dtype=np.uint32) for k in range(count)])
                chunk.create_dataset("/entry/data/data", data=stack, chunks=(1, H, W))
            data[f"data_{number:06d}"] = h5py.ExternalLink(data_file.name, "/entry/data/data")
            total += count
    return master


def _convert(sources, destination: Path, **options):
    repository = LocalSourceRepository()
    request = ConversionRequest(
        sources=tuple(sources),
        options=ConversionOptions(output_format=options.pop("output_format", "NumPy"), destination=str(destination), **options),
    )
    return LocalConversionExecutor(repository).execute(request)


# -- what is an input ---------------------------------------------------------------------------------


def test_every_hdf5_suffix_is_an_input_of_the_nexus_kind() -> None:
    assert set(HDF5_SUFFIXES) <= set(SUPPORTED_SUFFIXES)
    assert {SUPPORTED_SUFFIXES[suffix] for suffix in HDF5_SUFFIXES} == set(HDF5_TYPES) == {"NXS", "HDF5"}
    assert (SUPPORTED_SUFFIXES[".nxs"], SUPPORTED_SUFFIXES[".h5"], SUPPORTED_SUFFIXES[".hdf5"]) == ("NXS", "HDF5", "HDF5")
    for suffix in HDF5_SUFFIXES:
        assert is_supported_input_path(f"/data/scan{suffix.upper()}")
    # Frames can be chosen for both kinds, and HDF5 is only hidden as an output when nothing else is the input.
    assert select_source_frame_indices("HDF5", 6, "Every Nth frame", nth_frame=2) == [0, 2, 4]
    assert select_source_frame_indices("NXS", 6, "Every Nth frame", nth_frame=2) == [0, 2, 4]
    assert select_source_frame_indices("CBF", 6, "All") == [0]
    assert visible_output_formats(["HDF5"])["HDF5"] is False and visible_output_formats(["NXS"])["HDF5"] is False
    assert visible_output_formats(["HDF5", "NXS"])["HDF5"] is True


def test_a_stack_in_one_file_is_read_under_every_suffix(tmp_path: Path) -> None:
    repository = LocalSourceRepository()
    for suffix in HDF5_SUFFIXES:
        source = repository.inspect_source(_stack(tmp_path / f"stack{suffix}", 5))
        assert (source.frame_count, source.dataset_shape, source.layout) == (5, (5, H, W), "stack")
        assert source.dataset_path == "/entry/data/data" and source.selected_frames == [0, 1, 2, 3, 4]
        assert source.summary == ("1 file · {frames} frames", {"frames": 5}) and source.warnings == []
        data, metadata = repository.load_frame(source, 3)
        assert float(data[0, 0]) == 10.0 and metadata["frame_index"] == 3
    assert repository.scan_folder(tmp_path) == sorted(str(tmp_path / f"stack{suffix}") for suffix in HDF5_SUFFIXES)
    assert repository.scan_folder(tmp_path, include_nxs=False) == []


# -- a set of module files ----------------------------------------------------------------------------


def test_a_module_set_is_listed_once_and_each_stitched_frame_is_written_once(tmp_path: Path) -> None:
    modules = _module_set(tmp_path / "data", frames=4)
    repository = LocalSourceRepository()
    assert repository.scan_folder(tmp_path / "data") == [str(modules[0])]

    source = repository.inspect_source(modules[1])  # any module file stands for the set
    assert source.path == str(modules[0]) and source.file_type == "NXS"
    assert (source.frame_count, source.layout, source.members) == (4, "module_set", [str(path) for path in modules])
    assert source.summary == ("{modules} module files · {frames} frames · stitched", {"modules": 3, "frames": 4})
    assert (source.name, source.stem) == ("scan_00007", "scan_00007")  # the scan, without the module number

    view_model = create_format_converter_view_model(_context())
    added = view_model.add_paths([str(path) for path in reversed(modules)])  # all three chosen in a file dialog
    assert (added.added, added.errors) == (1, ()) and len(view_model.sources) == 1
    assert view_model.add_paths([str(modules[2])]).added == 0  # already there
    assert view_model.estimate_output(ConversionOptions("NumPy", str(tmp_path / "out")))[0] == 4

    report = view_model.convert(ConversionOptions(output_format="NumPy", destination=str(tmp_path / "out")))
    written = sorted(path.name for path in (tmp_path / "out").glob("*.npy"))
    assert written == [f"scan_00007_{frame:06d}.npy" for frame in (1, 2, 3, 4)]  # N outputs, not 3 N
    assert len(report.succeeded) == 4 and not report.failed
    for frame in range(4):
        expected = np.asarray(load_detector_image(modules[0], frame_idx=frame).data)
        np.testing.assert_array_equal(np.load(tmp_path / "out" / f"scan_00007_{frame + 1:06d}.npy"), expected)
    # Every module is in each frame, and the record names every module file.
    frame = np.load(tmp_path / "out" / "scan_00007_000003.npy")
    assert {100 + 2, 200 + 2, 300 + 2} <= set(np.unique(frame[np.isfinite(frame)]).astype(int))
    records = json.loads((tmp_path / "out" / "conversion_metadata.json").read_text(encoding="utf-8"))
    assert [record["frame_number"] for record in records] == [1, 2, 3, 4]
    assert all(record["module_files"] == [str(path) for path in modules] for record in records)
    assert all(record["layout"] == "module_set" for record in records)


def test_module_files_with_different_frame_counts_convert_the_frames_every_module_has(tmp_path: Path) -> None:
    from src.gimap.shared.detector_io.nexus_modules import MODULES_UNEQUAL

    modules = [_module(tmp_path / f"scan_00007_m{module:02d}.nxs", module, frames) for module, frames in ((1, 5), (2, 5), (3, 3))]
    source = LocalSourceRepository().inspect_source(modules[0])
    assert source.frame_count == 3 and source.selected_frames == [0, 1, 2]
    assert [template for template, _values in source.warnings] == [MODULES_UNEQUAL]
    assert source.warnings[0][1]["frames"] == 3
    report = _convert([source], tmp_path / "out")
    assert len(report.succeeded) == 3 and not report.failed


def test_files_only_named_like_modules_stay_separate_inputs(tmp_path: Path) -> None:
    for name, value in (("film1.nxs", 1), ("film2.nxs", 2)):
        with h5py.File(tmp_path / name, "w") as handle:
            handle.create_dataset(f"{DET}/data", data=np.full((2, H, W), value, dtype=np.int32))
    view_model = create_format_converter_view_model(_context())
    assert view_model.add_paths(view_model.scan_folder(str(tmp_path))).added == 2
    assert [source.name for source in view_model.sources] == ["film1.nxs", "film2.nxs"]
    values = [float(np.nanmax(view_model._load_preview(source, 0)[0])) for source in view_model.sources]
    assert values == [1.0, 2.0]


# -- a master with data files -------------------------------------------------------------------------


def test_an_eiger_style_master_is_one_input_with_all_its_frames(tmp_path: Path) -> None:
    master = _eiger(tmp_path / "eiger")
    repository = LocalSourceRepository()
    assert repository.scan_folder(tmp_path / "eiger") == [str(master)]  # the data files are not inputs

    source = repository.inspect_source(master)
    assert (source.frame_count, source.layout, source.file_type) == (8, "chunked_master", "HDF5")
    assert source.dataset_path == "/entry/data/data_000001"  # the image, never the flat field or the mask
    assert not any("flatfield" in path or "pixel_mask" in path for path in source.dataset_paths)
    assert source.summary == ("master + {files} data files · {frames} frames", {"files": 3, "frames": 8})
    assert source.dataset_shape == (8, H, W) and len(source.members) == 4

    view_model = create_format_converter_view_model(_context())
    everything = sorted(str(path) for path in (tmp_path / "eiger").iterdir())  # master and data files chosen together
    assert view_model.add_paths(everything).added == 1 and view_model.sources[0].path == str(master)

    source.selected_frames = [0, 3, 7]  # one frame of each data file
    report = _convert([source], tmp_path / "out")
    assert not report.failed
    values = [float(np.load(tmp_path / "out" / f"run7_master_{frame:06d}.npy")[0, 0]) for frame in (1, 4, 8)]
    assert values == [1000.0, 1003.0, 1007.0]


def test_an_hdf5_file_without_a_detector_image_is_not_listed_and_says_why_when_chosen(tmp_path: Path) -> None:
    table = tmp_path / "results.h5"
    with h5py.File(table, "w") as handle:
        handle.create_dataset("results/chi2", data=np.arange(12.0))
    _stack(tmp_path / "frames.h5", 2)
    repository = LocalSourceRepository()
    assert repository.scan_folder(tmp_path) == [str(tmp_path / "frames.h5")]
    with pytest.raises(ValueError, match="results.h5 holds no detector image"):
        repository.inspect_source(table)
    view_model = create_format_converter_view_model(_context())
    added = view_model.add_paths([str(table)])
    assert added.added == 0 and added.errors == ("results.h5: results.h5 holds no detector image.",)


# -- datasets -----------------------------------------------------------------------------------------


def test_another_dataset_of_the_file_can_be_chosen(tmp_path: Path) -> None:
    path = tmp_path / "bliss.h5"
    with h5py.File(path, "w") as handle:
        for scan, frames in (("1.1", 2), ("2.1", 5)):
            detector = handle.create_group(f"{scan}/instrument/eiger")
            handle[scan].attrs["NX_class"], detector.attrs["NX_class"] = "NXentry", "NXdetector"
            detector.create_dataset("data", data=np.full((frames, H, W), frames, dtype=np.int32))
    repository = LocalSourceRepository()
    source = repository.inspect_source(path)
    assert source.dataset_paths == ["/1.1/instrument/eiger/data", "/2.1/instrument/eiger/data"]
    assert (source.dataset_path, source.frame_count) == ("/1.1/instrument/eiger/data", 2)
    assert source.warnings and source.warnings[0][1]["count"] == 2  # "2 image datasets in this file: … is shown"
    repository.select_dataset(source, "/2.1/instrument/eiger/data")
    assert (source.dataset_path, source.frame_count, source.selected_frames) == ("/2.1/instrument/eiger/data", 5, [0, 1, 2, 3, 4])
    assert source.dataset_shape == (5, H, W) and source.dataset_paths[0] == "/1.1/instrument/eiger/data"
    assert source.warnings == []  # the person chose: nothing left to say
    assert float(repository.load_frame(source, 4)[0][0, 0]) == 5.0
    with pytest.raises(ValueError, match="Dataset does not exist: /3.1/data"):
        repository.select_dataset(source, "/3.1/data")
    assert source.dataset_path == "/2.1/instrument/eiger/data"  # unchanged by the failed choice


# -- the converter's own files ------------------------------------------------------------------------


def test_the_multi_frame_container_the_converter_writes_is_read_back_frame_by_frame(tmp_path: Path) -> None:
    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "stack.nxs", 4))
    report = _convert([source], tmp_path / "out", output_format="HDF5", container=True)
    container = tmp_path / "out" / "converted_images.h5"
    assert not report.failed and container.is_file()

    again = LocalSourceRepository().inspect_source(container)
    assert (again.frame_count, again.file_type) == (4, "HDF5") and again.dataset_shape == (4, H, W)
    second = _convert([again], tmp_path / "back")
    assert len(second.succeeded) == 4 and not second.failed
    for frame in range(4):
        original = np.asarray(load_detector_image(tmp_path / "stack.nxs", frame_idx=frame).data)
        np.testing.assert_array_equal(np.load(tmp_path / "back" / f"converted_images_{frame + 1:06d}.npy"), original)


def test_a_single_frame_hdf5_output_is_read_back_unchanged(tmp_path: Path) -> None:
    source = LocalSourceRepository().inspect_source(_stack(tmp_path / "stack.nxs", 2))
    _convert([source], tmp_path / "out", output_format="HDF5")
    written = tmp_path / "out" / "stack_000002.h5"
    again = LocalSourceRepository().inspect_source(written)
    assert (again.frame_count, again.layout) == (1, "single_frame")
    np.testing.assert_array_equal(
        LocalSourceRepository().load_frame(again, 0)[0], np.asarray(load_detector_image(tmp_path / "stack.nxs", frame_idx=1).data)
    )


def test_the_estimate_counts_the_frames_and_the_stored_bytes_without_opening_a_file(tmp_path: Path, monkeypatch) -> None:
    repository = LocalSourceRepository()
    stack = repository.inspect_source(_stack(tmp_path / "stack.nxs", 4))  # uint16
    modules = repository.inspect_source(_module_set(tmp_path / "set", frames=2)[0])  # int32, three modules
    monkeypatch.setattr(h5py, "File", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("opened a file")))
    count, size = repository.estimate_output([stack, modules], None)
    assert count == 6
    assert size == 4 * H * W * 2 + 2 * (3 * H * W) * 4


# -- the window: what its lists and counts say -----------------------------------------------------------


def _dispose(*windows) -> None:
    """Close the windows and delete them now, by the event loop: a parentless window left to the garbage
    collector is deleted inside a collection, which can take the interpreter down."""
    from PyQt5.QtCore import QCoreApplication, QEvent

    for window in windows:
        for name in ("_preview_thread", "_conversion_thread", "_load_thread", "_cal_thread", "_inspect_thread", "_extract_thread"):
            thread = getattr(window, name, None)
            if thread is not None:
                thread.wait(10000)
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _dialog():
    from PyQt5.QtWidgets import QApplication

    from src.gimap.features.format_converter.presentation.dialog import FormatConverterDialog

    QApplication.instance() or QApplication([])
    dialog = FormatConverterDialog(app_context=_context())
    dialog._start_preview = lambda _source: None  # no reader thread in these tests
    return dialog


def _rows(dialog) -> list[tuple]:
    tree = dialog.input_tree
    rows = []
    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        rows.append((item.text(0), item.text(1), item.text(2), [item.child(k).text(0) for k in range(item.childCount())]))
    return rows


def _sources(tmp_path: Path) -> list[str]:
    modules = _module_set(tmp_path / "set", frames=4)
    master = _eiger(tmp_path / "eiger")
    stack = _stack(tmp_path / "stack.nxs", 5)
    return [*(str(path) for path in modules), str(master), str(tmp_path / "eiger" / "run7_data_000002.h5"), str(stack)]


def test_the_input_list_names_each_source_once_and_says_what_it_consists_of(tmp_path: Path) -> None:
    dialog = _dialog()
    try:
        dialog.add_paths(_sources(tmp_path))
        assert _rows(dialog) == [
            ("scan_00007", "NXS", "4", [
                "Dataset: /entry/instrument/detector/data", "Shape: 4 × 20 × 30", "Frames: 1–4",
                "3 module files · 4 frames · stitched",
            ]),
            ("run7_master.h5", "HDF5", "8", [
                "Dataset: /entry/data/data_000001", "Shape: 8 × 20 × 30", "Frames: 1–8",
                "master + 3 data files · 8 frames",
            ]),
            ("stack.nxs", "NXS", "5", ["Dataset: /entry/data/data", "Shape: 5 × 20 × 30", "Frames: 1–5"]),
        ]
        # 3 inputs made of 3 + 4 + 1 files, 4 + 8 + 5 frames.
        assert dialog.input_note.text() == "3 inputs (8 files), 17 available image(s) / frame(s)."
        tip = dialog.input_tree.topLevelItem(0).toolTip(0)
        assert all(f"scan_00007_m0{module}.nxs" in tip for module in (1, 2, 3))
        table = dialog.selection_table
        assert [(table.item(row, 1).text(), table.item(row, 2).text(), table.item(row, 3).text()) for row in range(3)] == [
            ("scan_00007", "NXS", "4"), ("run7_master.h5", "HDF5", "8"), ("stack.nxs", "NXS", "5"),
        ]
        review = dialog.view_model.conversion_review(ConversionOptions("NumPy", str(tmp_path / "out")))
        assert review.input_counts == (("HDF5", 4), ("NXS", 4)) and review.image_count == 17
    finally:
        _dispose(dialog)


def test_one_file_per_input_reads_as_before(tmp_path: Path) -> None:
    dialog = _dialog()
    try:
        dialog.add_paths([str(_stack(tmp_path / "a.nxs", 2)), str(_stack(tmp_path / "b.h5", 3))])
        assert dialog.input_note.text() == "2 source file(s), 5 available image(s) / frame(s)."
    finally:
        _dispose(dialog)


def test_a_warning_of_the_reader_is_shown_under_its_source(tmp_path: Path) -> None:
    for module, frames in ((1, 5), (2, 5), (3, 3)):
        _module(tmp_path / f"scan_00007_m{module:02d}.nxs", module, frames)
    dialog = _dialog()
    try:
        dialog.add_paths([str(tmp_path / "scan_00007_m02.nxs")])
        (name, _kind, frames, children), = _rows(dialog)
        assert (name, frames) == ("scan_00007", "3")
        assert children[3] == "3 module files · 3 frames · stitched"
        assert children[4].startswith("⚠ Module files hold different numbers of frames (") and "frame 3" in children[4]
    finally:
        _dispose(dialog)


def test_the_list_reads_in_chinese_with_its_numbers_and_names_kept(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    # The reader's sentences are translated by their templates (their Chinese is in the NeXus table).
    for english, chinese in {
        "{modules} module files · {frames} frames · stitched": "{modules} 个模块文件 · {frames} 帧 · 已拼接",
        "master + {files} data files · {frames} frames": "主文件 + {files} 个数据文件 · {frames} 帧",
    }.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
    dialog = _dialog()
    try:
        i18n.apply_language("zh", [dialog])
        dialog.add_paths(_sources(tmp_path))
        rows = _rows(dialog)
        assert [(name, kind, frames) for name, kind, frames, _children in rows] == [
            ("scan_00007", "NXS", "4"), ("run7_master.h5", "HDF5", "8"), ("stack.nxs", "NXS", "5"),
        ]
        assert rows[0][3] == [
            "数据集：/entry/instrument/detector/data", "形状：4 × 20 × 30", "帧：1–4", "3 个模块文件 · 4 帧 · 已拼接",
        ]
        assert rows[1][3][3] == "主文件 + 3 个数据文件 · 8 帧"
        assert dialog.input_note.text() == "3 个输入（共 8 个文件），共 17 张可用图像 / 帧。"
    finally:
        i18n.apply_language("en", [dialog])
        _dispose(dialog)


def test_the_file_dialog_and_the_folder_dialog_offer_every_hdf5_suffix(tmp_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    from src.gimap.features.format_converter.presentation.display_formatting import INPUT_FILTER
    from src.gimap.features.format_converter.presentation.folder_import_dialog import FolderImportDialog

    QApplication.instance() or QApplication([])
    first, second = INPUT_FILTER.split(";;")[:2]
    for suffix in HDF5_SUFFIXES:
        assert f"*{suffix}" in first and f"*{suffix}" in second
    _stack(tmp_path / "a.h5", 2)
    _module_set(tmp_path, frames=2)
    folder = FolderImportDialog(view_model=create_format_converter_view_model(_context()))
    try:
        assert folder.nxs.text() == "NXS / HDF5"
        folder.path_edit.setText(str(tmp_path))
        folder._accept_if_valid()
        assert [Path(path).name for path in folder.paths] == ["a.h5", "scan_00007_m01.nxs"]
    finally:
        _dispose(folder)
