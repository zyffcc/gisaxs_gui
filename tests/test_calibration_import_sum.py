"""Importing a calibration that was made on summed frames.

The record says which frames were summed. The dialog shows the first of them at once (reading one frame
is quick), reads the sum on its reader thread and works on that: the frame row says "Sum of all frames",
and Auto Calibration or Manual refine use the image the calibration was made on, not frame 1 alone.
A read that is stopped leaves frame 1 and says so. Synthetic files only.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.features.calibration.application import ImportCalibration, frames_made_on
from src.gimap.features.calibration.infrastructure.adapters import (
    JsonCalibrationStorageAdapter,
    LocalCalibrationImageAdapter,
)
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from tests.test_calibration_feature import StoragePortStub, _result

H, W = 40, 64


def _stack_file(path: Path, count: int) -> Path:
    """Frame k holds k + 1 everywhere, and 100 · (k + 1) in one pixel that tells the frames apart."""
    stack = np.empty((count, H, W), dtype=np.uint32)
    for index in range(count):
        stack[index] = index + 1
        stack[index, 3, 5] = 100 * (index + 1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/instrument/detector/data", data=stack)
    return path


def _made_on(path: Path, *, frame: int = 0, summed: int = 1, count: int = 6):
    result = _result(path)
    result.metadata["loader"] = {"frame_index": frame, "summed_frames": summed, "frame_count": count}
    if summed > 1:
        result.metadata["loader"]["summed_frame_indices"] = [frame, frame + summed - 1]
    return result


# -- the use case ------------------------------------------------------------------------------------------


def test_the_record_says_which_frames_a_calibration_was_made_on(tmp_path: Path) -> None:
    path = tmp_path / "agbh.nxs"
    assert frames_made_on(_made_on(path, summed=6)) == (0, 6)
    assert frames_made_on(_made_on(path, frame=2, summed=3)) == (2, 3)
    assert frames_made_on(_made_on(path, frame=4)) == (4, 1)
    assert frames_made_on(_result(path)) == (0, 1)  # a record from before frames were written down
    odd = _result(path)
    odd.metadata["loader"] = {"frame_index": "x", "summed_frames": None}
    assert frames_made_on(odd) == (0, 1)


def test_an_import_reads_one_frame_at_once_and_the_sum_when_asked(tmp_path: Path) -> None:
    path = _stack_file(tmp_path / "agbh_00001.nxs", 6)
    use_case = ImportCalibration(StoragePortStub(_made_on(path, summed=6)), LocalCalibrationImageAdapter())
    quick = use_case(tmp_path / "result.json")
    assert quick.image.data[3, 5] == 100.0 and quick.image.metadata["summed_frames"] == 1  # frame 1: no long read

    seen = []
    whole = use_case.source_image(quick.result, summed=True, progress=lambda done, total: seen.append((done, total)))
    assert whole.data[3, 5] == 100.0 * (1 + 2 + 3 + 4 + 5 + 6) and whole.data[0, 0] == 21.0
    assert whole.metadata["summed_frames"] == 6 and whole.metadata["summed_frame_indices"] == [0, 5]
    assert seen[-1] == (6, 6)
    # Without a person waiting (the assistant) the import gives the image the calibration was made on.
    assert use_case(tmp_path / "result.json", summed=True).image.data[3, 5] == 2100.0

    part = ImportCalibration(StoragePortStub(_made_on(path, frame=2, summed=3)), LocalCalibrationImageAdapter())
    assert part(tmp_path / "result.json").image.data[3, 5] == 300.0
    assert part(tmp_path / "result.json", summed=True).image.data[3, 5] == 300.0 + 400.0 + 500.0
    # One frame: the same call as before, also with summed=True.
    one = ImportCalibration(StoragePortStub(_made_on(path, frame=3)), LocalCalibrationImageAdapter())
    assert one(tmp_path / "result.json", summed=True).image.data[3, 5] == 400.0


def test_the_assistant_checks_a_saved_calibration_on_the_image_it_was_made_on(tmp_path: Path) -> None:
    from src.gimap.features.calibration.application import HeadlessCalibration

    path = _stack_file(tmp_path / "agbh_00001.nxs", 6)
    asked = []

    class Importing:
        def __call__(self, path, *, summed=False):
            asked.append(summed)
            return ImportCalibration(StoragePortStub(_made_on(Path(tmp_path / "agbh_00001.nxs"), summed=6)), LocalCalibrationImageAdapter())(path, summed=summed)

    summary = HeadlessCalibration(load_image=None, run_calibration=None, import_calibration=Importing()).read_result(str(tmp_path / "r.json"))
    assert asked == [True] and summary["source_image"] == str(path)


# -- the dialog --------------------------------------------------------------------------------------------


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


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


@pytest.fixture()
def dialog(monkeypatch):
    QApplication.instance() or QApplication([])
    boxes = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **_kwargs: boxes.append(args[2]) or QMessageBox.Ok)
    window = GeometryCalibrationDialog(app_context=_context())
    window.boxes = boxes
    window.show()
    yield window
    _wait(window)
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    apply_language(DEFAULT_LANGUAGE)


def _saved(tmp_path: Path, result) -> str:
    path = tmp_path / "agbh.gimap-calibration.json"
    JsonCalibrationStorageAdapter().save(result, path)
    return str(path)


def test_an_imported_calibration_made_on_a_sum_shows_and_uses_the_sum(dialog, tmp_path: Path) -> None:
    source = _stack_file(tmp_path / "agbh_00001.nxs", 6)
    dialog.import_result_from(_saved(tmp_path, _made_on(source, summed=6)))
    # At once: the result, and the first of the frames (the UI thread never reads six frames).
    assert dialog.result is not None and dialog.image.data[3, 5] == 100.0
    assert dialog._load_thread is not None and not dialog.calibrate_button.isEnabled()
    _wait(dialog)
    assert not dialog.boxes
    assert dialog.image.data[3, 5] == 2100.0 and dialog.image.metadata["summed_frames"] == 6
    assert dialog.result is not None and dialog.candidate_table.rowCount() == 1  # the imported result stays
    assert not dialog.frames_row.isHidden() and dialog.frame_mode_combo.currentIndex() == 1
    assert dialog.frame_mode_combo.currentText() == "Sum of all frames" and dialog.frames_total_label.text() == "of 6"
    assert dialog.stage_label.text() == "Imported calibration from agbh.gimap-calibration.json | 6 frames summed"
    assert dialog.calibrate_button.isEnabled()
    # Auto Calibration would now be given the sum, and fit the standard the calibration was made with
    # (left on "Auto Detect", a new run could settle on another standard than the imported one).
    assert dialog.view_model.image is dialog.image and float(dialog.view_model.image.data[0, 0]) == 21.0
    assert dialog.standard_combo.currentData() == "agbh"


def test_an_imported_calibration_made_on_one_frame_reads_nothing_more(dialog, tmp_path: Path) -> None:
    source = _stack_file(tmp_path / "agbh_00001.nxs", 6)
    dialog.import_result_from(_saved(tmp_path, _made_on(source, frame=3)))
    assert dialog._load_thread is None  # as before: nothing runs after the import
    assert dialog.image.data[3, 5] == 400.0 and dialog.frame_mode_combo.currentIndex() == 0
    assert dialog.frame_spin.value() == 4
    assert dialog.stage_label.text() == "Imported calibration from agbh.gimap-calibration.json | frame 4 of 6"
    # A file with one frame says nothing about frames.
    single = _stack_file(tmp_path / "lab6.nxs", 1)
    dialog.import_result_from(_saved(tmp_path, _made_on(single, count=1)))
    assert dialog._load_thread is None and dialog.frames_row.isHidden()
    assert dialog.stage_label.text() == "Imported calibration from agbh.gimap-calibration.json"


def test_a_sum_that_is_not_read_to_its_end_leaves_frame_1_and_says_so(dialog, tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.calibration.application import CalibrationCancelledError

    source = _stack_file(tmp_path / "agbh_00001.nxs", 6)

    def stopped(*_args, **_kwargs):
        raise CalibrationCancelledError("Reading cancelled.")

    monkeypatch.setattr(dialog.view_model, "read_imported_sum", stopped)
    dialog.import_result_from(_saved(tmp_path, _made_on(source, summed=6)))
    _wait(dialog)
    assert not dialog.boxes and dialog.result is not None
    assert dialog.image.data[3, 5] == 100.0 and dialog.frame_mode_combo.currentIndex() == 0  # what is shown
    assert dialog.stage_label.text() == (
        "Imported calibration from agbh.gimap-calibration.json. The calibration was made on 6 frames summed "
        "from frame 1 on: frame 1 alone is shown."
    )
    assert dialog.job_status.state_label.text() == "CANCELLED" and dialog.calibrate_button.isEnabled()


def test_the_lines_about_the_imported_frames_read_in_chinese(dialog, tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.calibration.application import CalibrationCancelledError

    apply_language("zh", [dialog])
    source = _stack_file(tmp_path / "agbh_00001.nxs", 6)
    dialog.import_result_from(_saved(tmp_path, _made_on(source, summed=6)))
    _wait(dialog)
    assert dialog.stage_label.text().endswith("| 6 帧求和")

    def stopped(*_args, **_kwargs):
        raise CalibrationCancelledError("Reading cancelled.")

    monkeypatch.setattr(dialog.view_model, "read_imported_sum", stopped)
    dialog.import_result_from(_saved(tmp_path, _made_on(source, summed=6)))
    _wait(dialog)
    text = dialog.stage_label.text()
    assert "6" in text and "1" in text and not any(word in text for word in ("calibration was made", "alone is shown"))
