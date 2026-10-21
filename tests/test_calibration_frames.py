"""Geometry Calibration on NeXus / HDF5 files that hold more than one image.

A file with several image datasets (two scans) asks which one is the standard; a file with several
frames says which frame is used and lets the person take another one or the sum of all of them.
Synthetic files only.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.features.calibration.application import (
    AmbiguousImageDatasetError,
    CalibrationCancelledError,
    ImportCalibration,
    LoadCalibrationImage,
)
from src.gimap.features.calibration.infrastructure.adapters import LocalCalibrationImageAdapter
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

H, W = 40, 64


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def _frames(count: int) -> np.ndarray:
    """Frame k holds k + 1 everywhere, and 100 · (k + 1) in one pixel that tells the frames apart."""
    stack = np.empty((count, H, W), dtype=np.uint32)
    for index in range(count):
        stack[index] = index + 1
        stack[index, 3, 5] = 100 * (index + 1)
    return stack


def _stack_file(path: Path, count: int, mask=None) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/instrument/detector/data", data=_frames(count))
        if mask is not None:
            handle.create_dataset("/entry/instrument/detector/pixel_mask", data=mask)
    return path


def _two_scans(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        for scan, value in (("1.1", 7), ("2.1", 9)):
            detector = handle.create_group(f"{scan}/instrument/eiger")
            handle[scan].attrs["NX_class"], detector.attrs["NX_class"] = "NXentry", "NXdetector"
            detector.create_dataset("data", data=np.full((2, H, W), value, dtype=np.int32))
    return path


def _wait(dialog, seconds: float = 30.0) -> None:
    """Until the dialog's reader thread is gone (its results are delivered through the event loop)."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: dialog._load_thread is None and loop.quit())
    timer.start()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec_()
    timer.stop()
    QCoreApplication.processEvents()
    assert dialog._load_thread is None


# -- the adapter ---------------------------------------------------------------------------------------


def test_a_file_with_two_scans_asks_which_dataset_is_the_standard(tmp_path: Path) -> None:
    path = _two_scans(tmp_path / "bliss.h5")
    images = LocalCalibrationImageAdapter()
    with pytest.raises(AmbiguousImageDatasetError) as asked:
        images.load(path)
    assert asked.value.paths == ["/1.1/instrument/eiger/data", "/2.1/instrument/eiger/data"]
    chosen = images.load(path, "/2.1/instrument/eiger/data")
    assert float(np.nanmax(chosen.data)) == 9.0 and chosen.metadata["dataset_path"] == "/2.1/instrument/eiger/data"


def test_without_a_person_to_ask_the_best_dataset_is_read_and_the_file_says_so(tmp_path: Path) -> None:
    """The assistant's calibration has no dialog: it reads what every other reader shows (D1)."""
    path = _two_scans(tmp_path / "bliss.h5")
    image = LocalCalibrationImageAdapter(ask_dataset=False).load(path)
    assert float(np.nanmax(image.data)) == 7.0
    assert any("2 image datasets" in text for text in image.metadata["layout_warnings"])
    from src.gimap.features.calibration.bootstrap import create_headless_calibration

    assert create_headless_calibration().load_image(path).metadata["dataset_path"] == "/1.1/instrument/eiger/data"


def test_one_frame_is_read_by_default_and_the_image_says_which(tmp_path: Path) -> None:
    path = _stack_file(tmp_path / "agbh_00001.nxs", 5)
    images = LocalCalibrationImageAdapter()
    first = images.load(path)
    assert first.data[3, 5] == 100.0 and first.metadata["frame_count"] == 5
    assert first.metadata["frame_index"] == 0 and first.metadata["summed_frames"] == 1
    fourth = images.load(path, frame=3)
    assert fourth.data[3, 5] == 400.0 and fourth.metadata["frame_index"] == 3
    with pytest.raises(IndexError):
        images.load(path, frame=5)


def test_the_sum_adds_the_counts_and_keeps_every_invalid_pixel_invalid(tmp_path: Path) -> None:
    mask = np.zeros((1, H, W), dtype=np.uint32)
    mask[0, 10, 10] = 1
    path = _stack_file(tmp_path / "agbh_00001.nxs", 5, mask=mask)
    seen = []
    image = LocalCalibrationImageAdapter().load(path, frame=0, sum_frames=5, progress=lambda done, total: seen.append((done, total)))
    assert image.data.dtype == np.float32
    assert image.data[0, 0] == 1 + 2 + 3 + 4 + 5 and image.data[3, 5] == 1500.0
    assert np.isnan(image.data[10, 10]) and bool(image.mask[10, 10]) and int(image.mask.sum()) == 1
    assert image.metadata["summed_frames"] == 5 and image.metadata["frame_index"] == 0
    assert image.metadata["summed_frame_indices"] == [0, 4] and image.metadata["frame_count"] == 5
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    # A block that would run past the end stops at the last frame and says what it summed.
    tail = LocalCalibrationImageAdapter().load(path, frame=3, sum_frames=5)
    assert tail.data[0, 0] == 4 + 5 and tail.metadata["summed_frames"] == 2
    assert tail.metadata["summed_frame_indices"] == [3, 4]


def test_a_long_sum_can_be_cancelled_between_two_frames(tmp_path: Path) -> None:
    path = _stack_file(tmp_path / "agbh_00001.nxs", 6)
    read = []
    with pytest.raises(CalibrationCancelledError):
        LocalCalibrationImageAdapter().load(
            path, sum_frames=6, progress=lambda done, _total: read.append(done), cancelled=lambda: len(read) >= 2
        )
    assert read == [1, 2]


def test_other_formats_hold_one_frame(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "agbh.tif"
    Image.fromarray(np.arange(H * W, dtype=np.uint16).reshape(H, W)).save(path)
    image = LocalCalibrationImageAdapter().load(path, sum_frames=4)
    assert image.data.shape == (H, W) and image.metadata["summed_frames"] == 1
    assert image.metadata.get("frame_count", 1) == 1


def test_the_use_case_passes_the_frames_only_when_they_are_chosen(tmp_path: Path) -> None:
    calls = []

    class Old:  # a port from before the frame choice: called as before
        def load(self, path, dataset_path=None):
            calls.append((str(path), dataset_path))
            return "image"

    assert LoadCalibrationImage(Old())("a.cbf", "/entry/data") == "image"
    assert calls == [("a.cbf", "/entry/data")]

    class New:
        def load(self, path, dataset_path=None, **frames):
            calls.append(frames)
            return "image"

    LoadCalibrationImage(New())("a.nxs", None, frame=2, sum_frames=3)
    assert calls[-1] == {"frame": 2, "sum_frames": 3, "progress": None, "cancelled": None}


def test_an_imported_calibration_shows_the_frame_it_was_made_on(tmp_path: Path) -> None:
    from tests.test_calibration_feature import StoragePortStub, _result

    path = _stack_file(tmp_path / "agbh_00001.nxs", 5)
    result = _result(path)
    result.metadata["loader"] = {"frame_index": 3, "frame_count": 5}
    imported = ImportCalibration(StoragePortStub(result), LocalCalibrationImageAdapter())(tmp_path / "result.json")
    assert imported.image.metadata["frame_index"] == 3 and imported.image.data[3, 5] == 400.0
    # A file that no longer has that frame: its first frame, never an error.
    short = _stack_file(tmp_path / "agbh_00001.nxs", 2)
    from src.gimap.shared.detector_io import forget_layout

    forget_layout(short)
    again = ImportCalibration(StoragePortStub(result), LocalCalibrationImageAdapter())(tmp_path / "result.json")
    assert again.image.metadata["frame_index"] == 0


# -- the dialog ----------------------------------------------------------------------------------------


@pytest.fixture()
def dialog(monkeypatch):
    _app()
    boxes = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **_kwargs: boxes.append(args[2]) or QMessageBox.Ok)
    window = GeometryCalibrationDialog(app_context=_context())
    window.boxes = boxes
    window.show()
    yield window
    _wait(window)
    # Deleted now, by the event loop: a parentless window left to the garbage collector is deleted inside a
    # collection, which can take the interpreter down.
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    apply_language(DEFAULT_LANGUAGE)


def test_a_single_frame_shows_no_frame_choice(dialog, tmp_path: Path) -> None:
    assert dialog.frames_row.isHidden()
    dialog.load_image(str(_stack_file(tmp_path / "one.nxs", 1)))
    _wait(dialog)
    assert dialog.image is not None and dialog.frames_row.isHidden() and not dialog.boxes
    assert "frame" not in dialog.stage_label.text() and "frame" not in dialog.preview_info_label.text()


def test_a_multi_frame_standard_says_which_frame_and_offers_another_or_the_sum(dialog, tmp_path: Path) -> None:
    path = _stack_file(tmp_path / "agbh_00001.nxs", 5)
    dialog.load_image(str(path))
    _wait(dialog)
    assert not dialog.frames_row.isHidden() and not dialog.boxes
    assert (dialog.frame_spin.minimum(), dialog.frame_spin.maximum(), dialog.frame_spin.value()) == (1, 5, 1)
    assert dialog.frame_spin.isEnabled() and dialog.frames_total_label.text() == "of 5"
    assert [dialog.frame_mode_combo.itemText(index) for index in range(2)] == ["A single frame", "Sum of all frames"]
    assert "frame 1 of 5" in dialog.stage_label.text() and "frame 1 of 5" in dialog.preview_info_label.text()
    assert dialog.image.data[3, 5] == 100.0

    dialog.frame_spin.setValue(4)  # another frame: read again, the result of the old image is gone
    _wait(dialog)
    assert dialog.image.data[3, 5] == 400.0 and "frame 4 of 5" in dialog.stage_label.text()
    assert dialog.frame_spin.value() == 4 and dialog.result is None

    dialog.frame_mode_combo.setCurrentIndex(1)  # the sum of all frames
    _wait(dialog)
    assert dialog.image.data[3, 5] == 1500.0 and dialog.image.metadata["summed_frames"] == 5
    assert "5 frames summed" in dialog.stage_label.text() and "5 frames summed" in dialog.preview_info_label.text()
    assert not dialog.frame_spin.isEnabled() and dialog.frame_mode_combo.currentIndex() == 1

    # Another file starts with one frame again.
    dialog.load_image(str(_stack_file(tmp_path / "lab6_00002.nxs", 3)))
    _wait(dialog)
    assert dialog.frame_mode_combo.currentIndex() == 0 and dialog.frame_spin.maximum() == 3
    assert dialog.image.data[3, 5] == 100.0 and "frame 1 of 3" in dialog.stage_label.text()


def test_the_frame_choice_reads_in_chinese(dialog, tmp_path: Path) -> None:
    apply_language("zh", [dialog])
    dialog.load_image(str(_stack_file(tmp_path / "agbh_00001.nxs", 5)))
    _wait(dialog)
    assert dialog.framesLabel.text() == "帧："
    assert [dialog.frame_mode_combo.itemText(index) for index in range(2)] == ["单帧", "全部帧求和"]
    assert dialog.frames_total_label.text() == "/ 共 5 帧"
    assert "第 1 帧（共 5 帧）" in dialog.stage_label.text()
    dialog.frame_mode_combo.setCurrentIndex(1)
    _wait(dialog)
    assert "5 帧求和" in dialog.stage_label.text() and "5 帧求和" in dialog.preview_info_label.text()
    apply_language(DEFAULT_LANGUAGE, [dialog])  # and back: the row is English again, exactly
    assert dialog.framesLabel.text() == "Frames:" and dialog.frames_total_label.text() == "of 5"
    assert [dialog.frame_mode_combo.itemText(index) for index in range(2)] == ["A single frame", "Sum of all frames"]


def test_the_dataset_question_is_asked_again_and_the_answer_is_loaded(dialog, tmp_path: Path, monkeypatch) -> None:
    asked = []

    def choose(_parent, _title, _label, items, *_args):
        asked.append(list(items))
        return items[1], True

    monkeypatch.setattr(QInputDialog, "getItem", choose)
    dialog.load_image(str(_two_scans(tmp_path / "bliss.h5")))
    _wait(dialog)
    loop = QEventLoop()  # the answer is loaded a moment later
    QTimer.singleShot(400, loop.quit)
    loop.exec_()
    _wait(dialog)
    assert asked == [["/1.1/instrument/eiger/data", "/2.1/instrument/eiger/data"]]
    assert float(np.nanmax(dialog.image.data)) == 9.0 and not dialog.boxes
    # The frame choice keeps the dataset that was chosen.
    dialog.frame_spin.setValue(2)
    _wait(dialog)
    assert len(asked) == 1 and dialog.image.metadata["dataset_path"] == "/2.1/instrument/eiger/data"
    assert dialog.image.metadata["frame_index"] == 1


def test_hdf5_files_can_be_opened_and_dropped(tmp_path: Path) -> None:
    from PyQt5.QtCore import QMimeData, QUrl

    from src.gimap.features.calibration.presentation.bindings.image_loading import CALIBRATION_IMAGE_FILTER
    from src.gimap.features.calibration.presentation.bindings.input_files import dropped_calibration_file
    from src.gimap.shared.detector_io import HDF5_SUFFIXES

    for suffix in HDF5_SUFFIXES:
        assert f"*{suffix}" in CALIBRATION_IMAGE_FILTER.split(";;")[0]
        path = _stack_file(tmp_path / f"standard{suffix}", 1)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        assert dropped_calibration_file(mime) == str(path)
