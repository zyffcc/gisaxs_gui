"""XRR series from NeXus / HDF5 files of every layout.

One file holding the series, a set of detector-module files, a series stored as one frame per file (one
file or one module set per angle), a folder of files. Each case gives the frames once, in order, and the
inspection says what was found. Synthetic files only.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

from src.gimap.features.xrr.application import (
    ExtractXrrSeries,
    InspectXrrSeries,
    SpecularGeometry,
    XrrExtractionRequest,
    XrrExtractionSettings,
    XrrFrameRef,
    XrrSeriesSpec,
)
from src.gimap.features.xrr.infrastructure.adapters import LocalXrrSeriesRepository
from src.gimap.shared.detector_io import forget_layout

DET = "/entry/instrument/detector"
OMEGA = "/entry/sample/transformations/omega"
H, W = 20, 30


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _file(path: Path, frames, *, position=None, omega=None) -> Path:
    """A NeXus file whose frame k holds ``frames[k]`` in every pixel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.asarray(frames, dtype=np.int32)[:, None, None] + np.zeros((len(frames), H, W), dtype=np.int32)
        handle.create_dataset(f"{DET}/data", data=data, chunks=(1, H, W))
        if position is not None:
            handle.create_dataset(f"{DET}/translation/distance", data=np.asarray(position, dtype=np.float32))
        if omega is not None:
            handle.create_dataset(OMEGA, data=np.asarray(omega, dtype=float)).attrs["units"] = "deg"
    return path


def _module_set(folder: Path, stem: str, frames, modules=(1, 2), omega=None) -> list[Path]:
    return [
        _file(folder / f"{stem}_m{module:02d}.nxs", [100 * module + value for value in frames],
              position=[0.0, (module - 1) * (H + 4), 0.0], omega=omega)
        for module in modules
    ]


def _one_per_file(folder: Path, count: int = 12, stem: str = "lmbd_scan", suffix: str = ".nxs", omega: bool = False) -> list[Path]:
    return [
        _file(folder / f"{stem}_{number:05d}{suffix}", [number], omega=[0.1 * number] if omega else None)
        for number in range(1, count + 1)
    ]


def _refs(spec: XrrSeriesSpec) -> tuple[XrrFrameRef, ...]:
    return LocalXrrSeriesRepository().discover(spec)


def _values(refs) -> list[float]:
    repository = LocalXrrSeriesRepository()
    return [float(np.nanmax(repository.load_frame(ref).data)) for ref in refs]


# -- one frame per file -----------------------------------------------------------------------------------


def test_one_file_of_a_series_stored_as_one_frame_per_file_gives_the_whole_series(tmp_path: Path) -> None:
    files = _one_per_file(tmp_path)
    _file(tmp_path / "dark_00001.nxs", [999])  # numbered too, but another name
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    for chosen in (files[4], files[0], files[11]):
        refs = _refs(XrrSeriesSpec(chosen))  # kind "auto", as the window starts
        assert [ref.path.name for ref in refs] == [f"lmbd_scan_{number:05d}.nxs" for number in range(1, 13)]
        assert [(ref.frame_index, ref.sequence_index) for ref in refs] == [(0, index) for index in range(12)]
    assert _values(refs) == [float(number) for number in range(1, 13)]
    assert refs[0].label == "lmbd_scan_00001.nxs" and refs[11].label == "lmbd_scan_00012.nxs"
    layout = LocalXrrSeriesRepository().describe(XrrSeriesSpec(files[4], source_kind="nxs"))
    assert layout.summary == ("{files} files · 1 frame each", {"files": 12})
    assert (layout.files, layout.frames, layout.warnings) == (12, 12, ())


def test_numbered_files_are_taken_in_the_order_of_their_numbers(tmp_path: Path) -> None:
    for number in (10, 9, 2, 1):
        _file(tmp_path / f"refl{number}.h5", [number])
    refs = _refs(XrrSeriesSpec(tmp_path / "refl2.h5"))
    assert [ref.path.name for ref in refs] == ["refl1.h5", "refl2.h5", "refl9.h5", "refl10.h5"]
    # A file whose name holds no number is a series of its own.
    alone = _file(tmp_path / "single.nxs", [5])
    assert [ref.path.name for ref in _refs(XrrSeriesSpec(alone))] == ["single.nxs"]


def test_a_folder_of_one_frame_files_is_the_series(tmp_path: Path) -> None:
    _one_per_file(tmp_path / "scan", count=4)
    for kind in ("auto", "nxs"):
        refs = _refs(XrrSeriesSpec(tmp_path / "scan", source_kind=kind))
        assert [ref.path.name for ref in refs] == [f"lmbd_scan_{number:05d}.nxs" for number in (1, 2, 3, 4)]
    with pytest.raises(ValueError, match="No NeXus / HDF5 detector files"):
        (tmp_path / "empty").mkdir()
        _refs(XrrSeriesSpec(tmp_path / "empty", source_kind="nxs"))


def test_the_angle_of_each_file_comes_from_that_file(tmp_path: Path) -> None:
    files = _one_per_file(tmp_path, count=5, omega=True)
    refs = _refs(XrrSeriesSpec(files[2], angle_mode="nxs_dataset", angle_dataset_path=OMEGA))
    assert [ref.theta_deg for ref in refs] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


def test_one_module_set_per_angle_is_a_series_of_stitched_frames(tmp_path: Path) -> None:
    for number in (1, 2, 3):
        _module_set(tmp_path, f"xrr_{number:05d}", [number])
    refs = _refs(XrrSeriesSpec(tmp_path / "xrr_00002_m02.nxs"))
    assert [ref.path.name for ref in refs] == [f"xrr_{number:05d}_m01.nxs" for number in (1, 2, 3)]
    assert [ref.label for ref in refs] == ["xrr_00001", "xrr_00002", "xrr_00003"]  # the scans, not a module file
    assert _values(refs) == [201.0, 202.0, 203.0]  # stitched: the second module is in every frame
    layout = LocalXrrSeriesRepository().describe(XrrSeriesSpec(tmp_path / "xrr_00002_m02.nxs"))
    assert layout.summary == ("{sets} sets of {modules} module files · 1 frame each", {"sets": 3, "modules": 2})


# -- the series in one source -----------------------------------------------------------------------------


def test_a_stack_in_one_file_is_the_series_under_every_suffix(tmp_path: Path) -> None:
    for suffix in (".nxs", ".h5", ".hdf5"):
        path = _file(tmp_path / f"stack{suffix}", [3, 4, 5, 6])
        refs = _refs(XrrSeriesSpec(path))
        assert [(ref.path.name, ref.frame_index) for ref in refs] == [(path.name, index) for index in range(4)]
        assert [ref.label for ref in refs] == [f"{path.name} · frame {index}" for index in (1, 2, 3, 4)]
        assert LocalXrrSeriesRepository().describe(XrrSeriesSpec(path)).summary == ("1 file · {frames} frames", {"frames": 4})
    # Next to it in the folder: files with more than one frame are never joined to a series by their names.
    assert len(_refs(XrrSeriesSpec(tmp_path / "stack.nxs"))) == 4


def test_a_module_set_with_angles_is_one_series(tmp_path: Path) -> None:
    modules = _module_set(tmp_path, "scan_00001", [1, 2, 3], modules=(1, 2, 3), omega=[0.1, 0.2, 0.3])
    refs = _refs(XrrSeriesSpec(modules[2], source_kind="nxs", angle_mode="nxs_dataset", angle_dataset_path=OMEGA))
    assert len(refs) == 3 and all(ref.path == modules[0] for ref in refs)
    assert [ref.theta_deg for ref in refs] == pytest.approx([0.1, 0.2, 0.3])
    assert [ref.label for ref in refs] == ["scan_00001 · frame 1", "scan_00001 · frame 2", "scan_00001 · frame 3"]
    layout = LocalXrrSeriesRepository().describe(XrrSeriesSpec(modules[2]))
    assert layout.summary == ("{modules} module files · {frames} frames · stitched", {"modules": 3, "frames": 3})


def test_module_files_with_different_frame_counts_give_the_frames_every_module_has_and_say_so(tmp_path: Path) -> None:
    from src.gimap.shared.detector_io.nexus_modules import MODULES_UNEQUAL

    first = _file(tmp_path / "scan_00001_m01.nxs", [1, 2, 3, 4, 5], position=[0, 0, 0], omega=[0.1, 0.2, 0.3, 0.4, 0.5])
    _file(tmp_path / "scan_00001_m02.nxs", [1, 2, 3], position=[0, H + 4, 0], omega=[0.1, 0.2, 0.3])
    spec = XrrSeriesSpec(first, angle_mode="nxs_dataset", angle_dataset_path=OMEGA)
    refs = _refs(spec)
    assert [ref.frame_index for ref in refs] == [0, 1, 2]
    assert [ref.theta_deg for ref in refs] == pytest.approx([0.1, 0.2, 0.3])  # frame k has angle k
    layout = LocalXrrSeriesRepository().describe(spec)
    assert [template for template, _values in layout.warnings] == [MODULES_UNEQUAL]
    assert layout.frames == 3


def test_an_angle_dataset_that_does_not_fit_a_whole_series_is_refused(tmp_path: Path) -> None:
    path = _file(tmp_path / "stack.nxs", [1, 2, 3], omega=[0.1, 0.2])
    with pytest.raises(ValueError, match="contains 2 values but the series has 3 frames"):
        _refs(XrrSeriesSpec(path, angle_mode="nxs_dataset", angle_dataset_path=OMEGA))
    with pytest.raises(ValueError, match="NXS angle dataset was not found: /entry/nowhere"):
        _refs(XrrSeriesSpec(path, angle_mode="nxs_dataset", angle_dataset_path="/entry/nowhere"))


def test_a_folder_of_several_multi_frame_files_is_read_file_after_file(tmp_path: Path) -> None:
    _file(tmp_path / "part_1.nxs", [1, 2, 3])
    _file(tmp_path / "part_2.nxs", [4, 5])
    refs = _refs(XrrSeriesSpec(tmp_path))
    assert [(ref.path.name, ref.frame_index, ref.sequence_index) for ref in refs] == [
        ("part_1.nxs", 0, 0), ("part_1.nxs", 1, 1), ("part_1.nxs", 2, 2), ("part_2.nxs", 0, 3), ("part_2.nxs", 1, 4),
    ]
    assert _values(refs) == [1.0, 2.0, 3.0, 4.0, 5.0]
    layout = LocalXrrSeriesRepository().describe(XrrSeriesSpec(tmp_path))
    assert layout.summary == ("{files} files · {frames} frames in all", {"files": 2, "frames": 5})


def test_a_folder_with_cbf_files_is_still_a_cbf_series(tmp_path: Path) -> None:
    for name in ("scan_2.cbf", "scan_1.cbf"):
        (tmp_path / name).touch()
    _file(tmp_path / "scan_1.nxs", [1])
    refs = _refs(XrrSeriesSpec(tmp_path))
    assert [ref.path.name for ref in refs] == ["scan_1.cbf", "scan_2.cbf"]
    layout = LocalXrrSeriesRepository().describe(XrrSeriesSpec(tmp_path))
    assert layout.summary == ("{files} files · 1 frame each", {"files": 2})
    assert [ref.path.name for ref in _refs(XrrSeriesSpec(tmp_path, source_kind="nxs"))] == ["scan_1.nxs"]


def test_a_file_without_a_detector_image_says_so(tmp_path: Path) -> None:
    table = tmp_path / "results.h5"
    with h5py.File(table, "w") as handle:
        handle.create_dataset("results/chi2", data=np.arange(12.0))
    with pytest.raises(ValueError, match="results.h5 holds no detector image"):
        _refs(XrrSeriesSpec(table))


# -- inspection and extraction ----------------------------------------------------------------------------


def _request(spec: XrrSeriesSpec) -> XrrExtractionRequest:
    return XrrExtractionRequest(
        series=spec,
        geometry=SpecularGeometry(
            distance_m=1.0, energy_kev=12.0, pixel_size_x_m=100e-6, pixel_size_y_m=100e-6,
            beam_center_x_px=10.0, beam_center_y_px=10.0,
        ),
        extraction=XrrExtractionSettings(radius_px=0, aggregation="sum"),
    )


def test_the_inspection_says_what_the_series_consists_of_and_the_extraction_reads_every_file(tmp_path: Path) -> None:
    files = _one_per_file(tmp_path, count=6)
    spec = XrrSeriesSpec(files[0], theta_start_deg=0.0, theta_step_deg=0.001)
    inspection = InspectXrrSeries(LocalXrrSeriesRepository()).execute(spec)
    assert inspection.frame_count == 6 and inspection.first_ref.path == files[0]
    assert inspection.layout.summary == ("{files} files · 1 frame each", {"files": 6})
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(_request(spec))
    assert [point.intensity for point in result.points] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert [point.source_name for point in result.points] == [path.name for path in files]
    assert [point.frame_index for point in result.points] == [0] * 6


def test_a_repository_without_a_description_still_inspects() -> None:
    from tests.test_xrr_application import FakeSeriesRepository, _request as fake_request

    inspection = InspectXrrSeries(FakeSeriesRepository()).execute(fake_request().series)
    assert inspection.frame_count == 3 and inspection.layout is None


def test_the_label_of_a_frame_names_the_frame_whenever_its_file_holds_several() -> None:
    assert XrrFrameRef(Path("a.cbf"), 0, 0).label == "a.cbf"
    assert XrrFrameRef(Path("a.nxs"), 0, 0, frames_in_file=12).label == "a.nxs · frame 1"
    assert XrrFrameRef(Path("a.nxs"), 4, 4, frames_in_file=12).label == "a.nxs · frame 5"
    assert XrrFrameRef(Path("a.nxs"), 4, 4).label == "a.nxs · frame 5"  # as before, for refs made without the count
    assert XrrFrameRef(Path("scan_m01.nxs"), 0, 0, name="scan", frames_in_file=3).label == "scan · frame 1"


# -- the window -------------------------------------------------------------------------------------------


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

    from src.gimap.features.xrr.application import ExportXrrCurve, LoadLastCalibrationGeometry
    from src.gimap.features.xrr.infrastructure import LocalXrrCurveExportAdapter, LocalXrrRecordAdapter
    from src.gimap.features.xrr.presentation.dialog import XrrSeriesDialog
    from src.gimap.features.xrr.presentation.view_model import XrrViewModel

    class NoCalibration:
        def last_calibration(self):
            return None

    QApplication.instance() or QApplication([])
    view_model = XrrViewModel(
        inspect_series=InspectXrrSeries(LocalXrrSeriesRepository()),
        run_extraction=None,
        export_curve=ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()),
        last_calibration=LoadLastCalibrationGeometry(NoCalibration()),
    )
    return XrrSeriesDialog(view_model=view_model, app_context=object())


def _inspect(dialog, path: Path) -> list[str]:
    """Load the first frame as the window does, without its thread; the lines of the summary."""
    dialog.source_picker.set_path(str(path))
    dialog._on_inspected(dialog.view_model.inspect(dialog._series_spec()))
    return dialog.series_summary.text().split("\n")


def test_the_summary_says_which_kind_of_series_was_found(tmp_path: Path) -> None:
    dialog = _dialog()
    try:
        lines = _inspect(dialog, _one_per_file(tmp_path / "files", count=12)[3])
        assert lines[:2] == ["12 frame(s) · first: lmbd_scan_00001.nxs · shape 30 × 20", "12 files · 1 frame each"]
        assert dialog.live_frame_label.text() == "lmbd_scan_00001.nxs"

        lines = _inspect(dialog, _file(tmp_path / "stack.h5", [1, 2, 3, 4, 5]))
        assert lines[:2] == ["5 frame(s) · first: stack.h5 · frame 1 · shape 30 × 20", "1 file · 5 frames"]

        modules = _module_set(tmp_path / "set", "scan_00001", [1, 2, 3], modules=(1, 2, 3))
        lines = _inspect(dialog, modules[1])
        assert lines[0].startswith("3 frame(s) · first: scan_00001 · frame 1 · shape ")
        assert lines[1] == "3 module files · 3 frames · stitched"
        assert not any(line.startswith("⚠") for line in lines)
    finally:
        _dispose(dialog)


def test_a_warning_of_the_reader_is_part_of_the_summary(tmp_path: Path) -> None:
    _file(tmp_path / "scan_00001_m01.nxs", [1, 2, 3, 4, 5], position=[0, 0, 0])
    _file(tmp_path / "scan_00001_m02.nxs", [1, 2, 3], position=[0, H + 4, 0])
    dialog = _dialog()
    try:
        lines = _inspect(dialog, tmp_path / "scan_00001_m01.nxs")
        assert lines[1] == "2 module files · 3 frames · stitched"
        assert lines[2].startswith("⚠ Module files hold different numbers of frames (") and "frame 3" in lines[2]
    finally:
        _dispose(dialog)


def test_the_summary_reads_in_chinese_and_comes_back(tmp_path: Path) -> None:
    from src.gimap.app.presentation.i18n import apply_language

    dialog = _dialog()
    try:
        english = _inspect(dialog, _one_per_file(tmp_path, count=12)[0])
        apply_language("zh", [dialog])
        lines = dialog.series_summary.text().split("\n")
        assert lines[:2] == ["12 帧 · 第一帧：lmbd_scan_00001.nxs · 尺寸 30 × 20", "12 个文件 · 每个 1 帧"]
        apply_language("en", [dialog])
        assert dialog.series_summary.text().split("\n") == english
        # The frame of a multi-frame source is named in the interface language, in the summary and over the preview.
        _inspect(dialog, _file(tmp_path / "stack.h5", [1, 2, 3]))
        apply_language("zh", [dialog])
        assert dialog.series_summary.text().split("\n")[0] == "3 帧 · 第一帧：stack.h5 · 第 1 帧 · 尺寸 30 × 20"
        assert dialog.live_frame_label.text() == "stack.h5 · 第 1 帧"
        apply_language("en", [dialog])
        assert dialog.live_frame_label.text() == "stack.h5 · frame 1"
    finally:
        apply_language("en")
        _dispose(dialog)


def test_hdf5_files_and_folders_of_them_can_be_dropped(tmp_path: Path) -> None:
    from PyQt5.QtCore import QMimeData, QUrl

    from src.gimap.features.xrr.presentation.bindings.files import (
        SERIES_FILTER, SOURCE_KIND_AUTO, SOURCE_KIND_CBF, SOURCE_KIND_NXS, dropped_series,
    )
    from src.gimap.shared.detector_io import HDF5_SUFFIXES

    for suffix in HDF5_SUFFIXES:
        assert f"*{suffix}" in SERIES_FILTER.split(";;")[0]
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(tmp_path / f"scan{suffix}"))])
        assert dropped_series(mime) == str(tmp_path / f"scan{suffix}")

    files = _one_per_file(tmp_path / "nexus", count=3, suffix=".h5")
    (tmp_path / "cbf").mkdir()
    (tmp_path / "cbf" / "scan_1.cbf").touch()
    dialog = _dialog()
    try:
        inspected = []
        dialog._inspect_series = lambda: inspected.append(dialog.source_picker.path())
        dialog.source_kind_combo.setCurrentIndex(SOURCE_KIND_NXS)
        dialog.use_series(str(tmp_path / "nexus"))  # a folder of NeXus / HDF5 files stays an NXS series
        assert dialog.source_kind_combo.currentIndex() == SOURCE_KIND_NXS
        dialog.use_series(str(tmp_path / "cbf"))  # a folder of CBF frames cannot be one
        assert dialog.source_kind_combo.currentIndex() == SOURCE_KIND_CBF
        dialog.use_series(str(files[0]))  # an .h5 file is not a CBF series
        assert dialog.source_kind_combo.currentIndex() == SOURCE_KIND_NXS
        dialog.source_kind_combo.setCurrentIndex(SOURCE_KIND_AUTO)
        dialog.use_series(str(files[1]))
        assert dialog.source_kind_combo.currentIndex() == SOURCE_KIND_AUTO and len(inspected) == 4
    finally:
        _dispose(dialog)
