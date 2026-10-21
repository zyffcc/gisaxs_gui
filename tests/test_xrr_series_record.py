"""The XRR window on NeXus series: what an export records about its files, choosing a folder, the title.

- The record next to an exported curve says what the series was stored in, names every file that was
  read and repeats what the reader had to say (module files with unequal frame counts: the frames left
  out) — also for an extraction that was stopped, and through the worker process.
- Browse can choose a folder of NeXus / HDF5 files, and a source type that cannot read the choice is
  changed to one that can.
- The title over the detector preview is drawn with glyphs in every interface language.

Synthetic files only.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

from src.gimap.features.xrr.application import (
    ExportXrrCurve,
    ExportXrrCurveRequest,
    ExtractXrrSeries,
    InspectXrrSeries,
    SpecularGeometry,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrExtractionSettings,
    XrrSeriesLayout,
    XrrSeriesSpec,
    xrr_export_record,
)
from src.gimap.features.xrr.infrastructure.adapters import LocalXrrSeriesRepository
from src.gimap.shared.detector_io import forget_layout

DET = "/entry/instrument/detector"
OMEGA = "/entry/sample/transformations/omega"
H, W = 20, 30
UNEQUAL = "Module files hold different numbers of frames (m02: 3; the others 5): frames after frame 3 are left out"


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


def _unequal_set(folder: Path) -> list[Path]:
    """Two module files, the second three frames short of the first: five angles, three usable frames."""
    angles = [0.1, 0.2, 0.3, 0.4, 0.5]
    return [
        _file(folder / "scan_00001_m01.nxs", [1, 2, 3, 4, 5], position=[0, 0, 0], omega=angles),
        _file(folder / "scan_00001_m02.nxs", [1, 2, 3], position=[0, H + 4, 0], omega=angles[:3]),
    ]


def _request(spec: XrrSeriesSpec) -> XrrExtractionRequest:
    return XrrExtractionRequest(
        series=spec,
        geometry=SpecularGeometry(
            distance_m=1.0, energy_kev=12.0, pixel_size_x_m=100e-6, pixel_size_y_m=100e-6,
            beam_center_x_px=10.0, beam_center_y_px=10.0,
        ),
        extraction=XrrExtractionSettings(radius_px=0, aggregation="sum"),
    )


# -- the record of the series ---------------------------------------------------------------------------------


def test_the_extraction_returns_what_its_series_was_stored_in(tmp_path: Path) -> None:
    modules = _unequal_set(tmp_path)
    spec = XrrSeriesSpec(modules[0], angle_mode="nxs_dataset", angle_dataset_path=OMEGA)
    seen = []
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(_request(spec), on_progress=seen.append)
    assert [point.theta_deg for point in result.points] == pytest.approx([0.1, 0.2, 0.3])
    layout = result.layout
    assert (layout.files, layout.frames) == (1, 3)
    assert layout.members == tuple(str(path) for path in modules)  # every file that was read
    assert layout.summary_text() == "2 module files · 3 frames · stitched"
    assert layout.warning_texts() == (UNEQUAL,)
    assert all(progress.layout is layout for progress in seen)  # a stopped run knows it too


def test_the_record_of_an_export_says_which_frames_were_left_out_and_names_every_file(tmp_path: Path) -> None:
    from src.gimap.features.xrr.infrastructure import LocalXrrCurveExportAdapter, LocalXrrRecordAdapter

    modules = _unequal_set(tmp_path / "data")
    spec = XrrSeriesSpec(modules[0], angle_mode="nxs_dataset", angle_dataset_path=OMEGA)
    request = _request(spec)
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(request)
    exported = ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()).execute(
        ExportXrrCurveRequest(tmp_path / "out" / "scan_xrr.csv", result, settings=request)
    )
    record = json.loads(exported.record_path.read_text(encoding="utf-8"))
    assert record["points"] == 3 and record["theta_deg"]["last"] == pytest.approx(0.3)
    assert record["source_layout"] == {
        "summary": "2 module files · 3 frames · stitched",
        "sources": 1,
        "frames": 3,
        "files": [str(path) for path in modules],
        "warnings": [UNEQUAL],
    }
    assert record["series"]["source_path"] == str(modules[0])  # the file that was chosen, as before


def test_a_complete_series_and_a_series_of_files_are_recorded_too(tmp_path: Path) -> None:
    files = [_file(tmp_path / "per_file" / f"lmbd_scan_{number:05d}.nxs", [number]) for number in (1, 2, 3, 4)]
    request = _request(XrrSeriesSpec(files[1]))
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(request)
    record = xrr_export_record(ExportXrrCurveRequest(tmp_path / "a.csv", result, settings=request), tmp_path / "a.csv")
    assert record["source_layout"] == {
        "summary": "4 files · 1 frame each", "sources": 4, "frames": 4,
        "files": [str(path) for path in files], "warnings": [],
    }
    # CBF frames: one file each, named.
    for name in ("scan_2.cbf", "scan_1.cbf"):
        (tmp_path / "cbf" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "cbf" / name).touch()
    layout = LocalXrrSeriesRepository().describe(XrrSeriesSpec(tmp_path / "cbf"))
    assert [Path(member).name for member in layout.members] == ["scan_1.cbf", "scan_2.cbf"]
    # A result that does not say what it was read from (older callers, fakes) gives a record without the key.
    plain = xrr_export_record(
        ExportXrrCurveRequest(tmp_path / "a.csv", XrrExtractionResult(result.points), settings=request), tmp_path / "a.csv"
    )
    assert "source_layout" not in plain


def test_the_layout_crosses_the_worker_process_with_the_result_and_with_the_first_progress(tmp_path: Path) -> None:
    from src.gimap.features.xrr.infrastructure.serialization import (
        layout_from_payload,
        layout_to_payload,
        request_to_payload,
        result_from_payload,
    )
    from src.gimap.features.xrr.infrastructure.workers import process_xrr_series_job

    modules = _unequal_set(tmp_path)
    spec = XrrSeriesSpec(modules[0], angle_mode="nxs_dataset", angle_dataset_path=OMEGA)
    reports = []
    payload = process_xrr_series_job(
        request_to_payload(_request(spec)), lambda done, total, text, details: reports.append(details), lambda: False
    )
    json.dumps(payload), json.dumps(reports)  # what goes through the pipe is plain data
    layout = result_from_payload(payload).layout
    assert layout.members == tuple(str(path) for path in modules) and layout.warning_texts() == (UNEQUAL,)
    assert layout_from_payload(reports[0]["layout"]).warning_texts() == (UNEQUAL,)
    assert all("layout" not in details for details in reports[1:])  # said once, not with every frame
    assert layout_from_payload(None) is None and layout_to_payload(None) is None
    assert result_from_payload({"points": []}).layout is None  # a payload made before this key


def test_a_stopped_extraction_still_knows_its_series(tmp_path: Path) -> None:
    """The runner returns nothing for a cancelled job: the view model keeps what the first progress said."""
    from src.gimap.features.xrr.presentation.view_model import XrrViewModel

    modules = _unequal_set(tmp_path)
    spec = XrrSeriesSpec(modules[0], angle_mode="nxs_dataset", angle_dataset_path=OMEGA)
    real = ExtractXrrSeries(LocalXrrSeriesRepository())

    class StoppedAfterTwo:
        def execute(self, request, *, on_progress=None):
            real.execute(request, on_progress=on_progress, is_cancelled=lambda: len(seen) >= 2)
            return XrrExtractionResult(())

        def cancel(self):
            return True

    seen = []
    view_model = XrrViewModel(inspect_series=None, run_extraction=StoppedAfterTwo(), export_curve=None)
    result = view_model.extract(_request(spec), on_progress=seen.append)
    assert len(result.points) == 2 and result.layout.warning_texts() == (UNEQUAL,)
    assert view_model.state.result.layout.frames == 3


# -- the window -----------------------------------------------------------------------------------------------


def _dispose(*windows) -> None:
    """Close the windows and delete them now, by the event loop: a parentless window left to the garbage
    collector is deleted inside a collection, which can take the interpreter down."""
    from PyQt5.QtCore import QCoreApplication, QEvent

    for window in windows:
        for name in ("_inspect_thread", "_extract_thread"):
            thread = getattr(window, name, None)
            if thread is not None:
                thread.wait(10000)
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _dialog():
    from PyQt5.QtWidgets import QApplication

    from src.gimap.features.xrr.application import LoadLastCalibrationGeometry
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


def _inspect(dialog) -> list[str]:
    """Load the first frame as the window does, without its thread; the lines of the summary."""
    dialog._on_inspected(dialog.view_model.inspect(dialog._series_spec()))
    return dialog.series_summary.text().split("\n")


def test_the_status_after_an_extraction_repeats_what_the_reader_said(tmp_path: Path) -> None:
    modules = _unequal_set(tmp_path)
    spec = XrrSeriesSpec(modules[0], angle_mode="nxs_dataset", angle_dataset_path=OMEGA)
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(_request(spec))
    dialog = _dialog()
    try:
        dialog._last_progress_total = 3
        dialog._on_extraction_finished(result)
        text = dialog.job_status.message_label.text()
        assert text.startswith("Extracted 3 XRR point(s).") and f"⚠ {UNEQUAL}" in text
        # Without a remark the line is as it was.
        dialog._on_extraction_finished(XrrExtractionResult(result.points, XrrSeriesLayout(1, 3, ("1 file · {frames} frames", {"frames": 3}))))
        assert dialog.job_status.message_label.text() == "Extracted 3 XRR point(s)."
    finally:
        _dispose(dialog)


def _numbered_before_the_end(folder: Path, count: int = 4) -> list[Path]:
    """A series whose number is not the last number of its names: one file alone is one frame."""
    return [_file(folder / f"scan_{number:05d}_det2.nxs", [number]) for number in range(1, count + 1)]


def test_browse_can_choose_a_folder_of_nexus_files_whatever_the_source_type(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    from src.gimap.features.xrr.presentation.bindings.files import SOURCE_KIND_AUTO, SOURCE_KIND_CBF, SOURCE_KIND_NXS

    files = _numbered_before_the_end(tmp_path / "nexus")
    asked: list[str] = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: asked.append("folder") or str(tmp_path / "nexus"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (asked.append("file") or str(files[0]), ""))
    dialog = _dialog()
    try:
        # One file of such a series is one frame; the folder is the series.
        dialog.source_kind_combo.setCurrentIndex(SOURCE_KIND_AUTO)
        dialog.source_picker.browse_button.click()
        assert asked == ["file"] and _inspect(dialog)[0].startswith("1 frame(s)")
        for kind in (SOURCE_KIND_AUTO, SOURCE_KIND_NXS):
            asked.clear()
            dialog.source_kind_combo.setCurrentIndex(kind)
            dialog.source_folder_button.click()
            assert asked == ["folder"] and dialog.source_picker.path() == str(tmp_path / "nexus")
            assert dialog.source_kind_combo.currentIndex() == kind  # a type that can read it is left alone
            assert _inspect(dialog)[:2] == ["4 frame(s) · first: scan_00001_det2.nxs · shape 30 × 20", "4 files · 1 frame each"]
        # "CBF series" asked for a folder before, and then failed on a folder without CBF frames.
        asked.clear()
        dialog.source_kind_combo.setCurrentIndex(SOURCE_KIND_CBF)
        dialog.source_picker.browse_button.click()
        assert asked == ["folder"] and dialog.source_kind_combo.currentIndex() == SOURCE_KIND_AUTO
        assert _inspect(dialog)[0].startswith("4 frame(s)")
    finally:
        _dispose(dialog)


def test_the_source_type_follows_what_was_chosen_or_dropped(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    from src.gimap.features.xrr.presentation.bindings.files import SOURCE_KIND_AUTO, SOURCE_KIND_CBF, SOURCE_KIND_NXS

    nexus = _numbered_before_the_end(tmp_path / "nexus")[0].parent
    cbf = tmp_path / "cbf"
    cbf.mkdir()
    (cbf / "scan_1.cbf").touch()
    both = tmp_path / "both"
    _file(both / "scan_1.nxs", [1])
    (both / "scan_1.cbf").touch()
    chosen: list[Path] = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(chosen[-1]))
    dialog = _dialog()
    try:
        dialog._inspect_series = lambda: None  # a drop loads the first frame: not here
        for start, folder, expected in (
            (SOURCE_KIND_CBF, nexus, SOURCE_KIND_AUTO),  # no CBF frame in it: its NeXus files are the series
            (SOURCE_KIND_CBF, cbf, SOURCE_KIND_CBF),
            (SOURCE_KIND_CBF, both, SOURCE_KIND_CBF),  # the person's choice stands while it can be read
            (SOURCE_KIND_NXS, cbf, SOURCE_KIND_CBF),
            (SOURCE_KIND_NXS, both, SOURCE_KIND_NXS),
            (SOURCE_KIND_AUTO, nexus, SOURCE_KIND_AUTO),
        ):
            for choose in ("button", "drop"):
                dialog.source_kind_combo.setCurrentIndex(start)
                chosen.append(folder)
                dialog.source_folder_button.click() if choose == "button" else dialog.use_series(str(folder))
                assert dialog.source_kind_combo.currentIndex() == expected, (start, folder.name, choose)
                assert dialog.source_picker.path() == str(folder)
        # Cancelling the dialog changes nothing.
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: "")
        dialog.source_kind_combo.setCurrentIndex(SOURCE_KIND_CBF)
        dialog.source_folder_button.click()
        assert dialog.source_kind_combo.currentIndex() == SOURCE_KIND_CBF and dialog.source_picker.path() == str(nexus)
    finally:
        _dispose(dialog)


def test_the_folder_button_reads_in_chinese_and_sits_in_the_tab_order(tmp_path: Path) -> None:
    from src.gimap.app.presentation.i18n import apply_language

    dialog = _dialog()
    try:
        assert dialog.source_folder_button.text() == "Folder…" and dialog.source_folder_button.toolTip()
        apply_language("zh", [dialog])
        assert dialog.source_folder_button.text() == "文件夹…"
        assert not any(character.isascii() and character.isalpha() for character in dialog.source_folder_button.toolTip().replace("NXS", "").replace("HDF5", "").replace("CBF", ""))
        assert dialog.source_picker.browse_button.nextInFocusChain() is dialog.source_folder_button
    finally:
        apply_language("en")
        _dispose(dialog)


def _undrawable(text) -> list[str]:
    """Characters of a Matplotlib text that none of the fonts it is drawn with has a glyph for."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    fonts = [
        FT2Font(font_manager.findfont(font_manager.FontProperties(family=[family])))
        for family in text.get_fontfamily()
    ]
    return [
        character for character in text.get_text()
        if not character.isspace() and not any(font.get_char_index(ord(character)) for font in fonts)
    ]


def _chinese_font() -> bool:
    from matplotlib import font_manager

    from src.gimap.shared.figures import CJK_FONTS

    names = {font.name for font in font_manager.fontManager.ttflist}
    return any(name in names for name in CJK_FONTS)


def test_the_title_over_the_preview_has_a_glyph_for_every_character(tmp_path: Path) -> None:
    from src.gimap.app.presentation.i18n import apply_language

    if not _chinese_font():
        pytest.skip("no Chinese font on this machine")
    dialog = _dialog()
    try:
        dialog.source_picker.set_path(str(_file(tmp_path / "stack.h5", [1, 2, 3])))
        _inspect(dialog)
        title = dialog.plotter.live_axis.title
        assert title.get_text() == "stack.h5 · frame 1" and _undrawable(title) == []
        apply_language("zh", [dialog])
        assert dialog.live_frame_label.text() == "stack.h5 · 第 1 帧"
        title = dialog.plotter.live_axis.title
        assert title.get_text() == "stack.h5 · 第 1 帧"  # the title follows the caption, also on a language switch
        assert _undrawable(title) == []
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # “Glyph … missing from font” would be a UserWarning
            dialog.plotter.live_canvas.draw()
        apply_language("en", [dialog])
        assert dialog.plotter.live_axis.title.get_text() == "stack.h5 · frame 1"
        # A file named in Chinese, with the interface in English.
        dialog.source_picker.set_path(str(_file(tmp_path / "样品 一.h5", [1, 2])))
        _inspect(dialog)
        title = dialog.plotter.live_axis.title
        assert title.get_text() == "样品 一.h5 · frame 1" and _undrawable(title) == []
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            dialog.plotter.live_canvas.draw()
    finally:
        apply_language("en")
        _dispose(dialog)


# -- files numbered like the series that are not part of it -------------------------------------------------------


def test_files_numbered_like_the_series_that_do_not_fit_are_named_not_dropped_silently(tmp_path: Path) -> None:
    """With angles from start + step, every point after a file that was left out gets the angle of its
    position: the reader of the summary, the status and the record must know a file was left out."""
    from src.gimap.features.xrr.application.models import FILES_LEFT_OUT

    files = [_file(tmp_path / f"lmbd_scan_{number:05d}.nxs", [number]) for number in (1, 2, 4, 5)]
    _file(tmp_path / "lmbd_scan_00003.nxs", [3, 3])  # two frames: not a frame of a one-frame-per-file series
    with h5py.File(tmp_path / "lmbd_scan_00006.nxs", "w") as handle:  # another frame size
        handle.create_dataset(f"{DET}/data", data=np.zeros((1, H + 2, W), dtype=np.int32))
    _file(tmp_path / "other_00001.nxs", [9])  # not numbered like the series: nothing to say
    spec = XrrSeriesSpec(files[0], theta_start_deg=0.1, theta_step_deg=0.1)
    layout = LocalXrrSeriesRepository().describe(spec)
    assert (layout.files, layout.frames) == (4, 4)
    assert layout.warnings == ((FILES_LEFT_OUT, {"count": 2, "names": "lmbd_scan_00003.nxs, lmbd_scan_00006.nxs"}),)
    assert layout.warning_texts() == (
        "2 files numbered like the series are left out (another frame size, or not one frame): "
        "lmbd_scan_00003.nxs, lmbd_scan_00006.nxs. With angles from start and step, the frames after a "
        "file that is left out get the angle of their position",
    )
    request = _request(spec)
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(request)
    assert [point.source_name for point in result.points] == [path.name for path in files]
    record = xrr_export_record(ExportXrrCurveRequest(tmp_path / "a.csv", result, settings=request), tmp_path / "a.csv")
    assert record["source_layout"]["warnings"] == list(layout.warning_texts())
    # A complete series says nothing.
    whole = [_file(tmp_path / "whole" / f"lmbd_scan_{number:05d}.nxs", [number]) for number in (1, 2, 3)]
    assert LocalXrrSeriesRepository().describe(XrrSeriesSpec(whole[0])).warnings == ()
    # Many files: the first few are named and the rest counted.
    many = [_file(tmp_path / "many" / f"lmbd_scan_{number:05d}.nxs", [number]) for number in (1, 2)]
    for number in range(3, 12):
        _file(tmp_path / "many" / f"lmbd_scan_{number:05d}.nxs", [number, number])
    (_template, values), = LocalXrrSeriesRepository().describe(XrrSeriesSpec(many[0])).warnings
    assert values == {"count": 9, "names": "lmbd_scan_00003.nxs, lmbd_scan_00004.nxs, lmbd_scan_00005.nxs …"}


def test_the_frames_and_what_they_are_stored_in_come_from_one_look_at_the_files(tmp_path: Path, monkeypatch) -> None:
    """The record describes the files the points were read from: both are found in the same pass."""
    files = [_file(tmp_path / f"lmbd_scan_{number:05d}.nxs", [number]) for number in (1, 2, 3)]
    spec = XrrSeriesSpec(files[0])
    looks = []
    real = LocalXrrSeriesRepository._series
    monkeypatch.setattr(LocalXrrSeriesRepository, "_series", lambda self, asked: looks.append(asked) or real(self, asked))
    result = ExtractXrrSeries(LocalXrrSeriesRepository()).execute(_request(spec))
    assert len(looks) == 1 and len(result.points) == 3 and result.layout.frames == 3
    inspection = InspectXrrSeries(LocalXrrSeriesRepository()).execute(spec)
    assert len(looks) == 2 and inspection.frame_count == 3 and inspection.layout.frames == 3
