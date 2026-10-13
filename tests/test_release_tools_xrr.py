"""XRR Series Extractor, release round: the job messages and the input warning in the interface
language (R43), and a window that fits a 1280x720 logical screen (R72)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.xrr.application import XrrExtractionProgress, XrrExtractionResult, XrrPoint
from src.gimap.features.xrr.presentation.dialog import XrrSeriesDialog
from tests.test_release_tools_i18n import ZH, switch_to_chinese

NO_SOURCE = "Select an NXS module file or a CBF series first."
NO_DATASET = "Enter the NXS motor dataset path containing theta values."


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dispose(dialog) -> None:
    for name in ("_inspect_thread", "_extract_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class FakeViewModel:
    """The window without the job runner (as in ``test_xrr_presentation``)."""

    def __init__(self):
        self.state = SimpleNamespace(inspection=None, result=None, error_message="", running=False)

    def cancel(self):
        return True


def _point(index: int) -> XrrPoint:
    return XrrPoint(
        sequence_index=index, source_name=f"frame_{index}.cbf", frame_index=0, theta_deg=0.1 * index,
        qz_inv_angstrom=0.01 * index, intensity=100.0 + index, roi_center_x_px=12.0, roi_center_y_px=18.0,
        valid_pixels=13,
    )


@pytest.fixture
def chinese(monkeypatch):
    switch_to_chinese(monkeypatch)
    yield
    apply_language("en")


@pytest.fixture
def boxes(monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda _parent, title, text, *_a, **_k: shown.append((title, text)) or QMessageBox.Ok)
    return shown


def _job_line(dialog) -> str:
    return dialog.job_status.message_label.text()


def _run(dialog, tmp_path: Path) -> dict[str, str]:
    """The texts of a run that is refused, progresses, ends and is stopped (no worker thread)."""
    said = {}
    dialog._run_extraction()  # nothing chosen
    said["no source"] = _job_line(dialog)
    dialog.source_picker.set_path(str(tmp_path / "scan_a.nxs"))
    dialog.angle_mode_combo.setCurrentIndex(1)  # NXS motor dataset
    dialog.angle_dataset_edit.setText("")
    dialog._inspect_series()
    said["no dataset"] = _job_line(dialog)

    points = tuple(_point(index) for index in range(3))
    dialog.view_model.state.result = XrrExtractionResult(points[:1])
    dialog._on_extraction_progress(XrrExtractionProgress(
        completed=1, total=3, point=points[0], preview=np.ones((20, 30), dtype=np.float32), preview_shape=(200, 300),
    ))
    said["progress"] = _job_line(dialog)
    dialog.view_model.state.result = XrrExtractionResult(points)
    dialog._on_extraction_finished(dialog.view_model.state.result)
    said["finished"] = _job_line(dialog)
    dialog._last_progress_total = 5  # two frames were never reached
    dialog._on_extraction_finished(dialog.view_model.state.result)
    said["stopped"] = _job_line(dialog)
    dialog._extract_worker = SimpleNamespace(cancel=lambda: None)
    dialog._cancel_extraction()
    said["stopping"] = _job_line(dialog)
    dialog._extract_worker = None
    dialog._on_extraction_failed("No detector frames were found for the selected XRR series.")
    said["failed"] = _job_line(dialog)
    return said


def test_the_english_job_messages_are_unchanged(tmp_path, boxes) -> None:
    _app()
    dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
    try:
        assert _run(dialog, tmp_path) == {
            "no source": NO_SOURCE,
            "no dataset": NO_DATASET,
            "progress": "Extracted frame 1 of 3",
            "finished": "Extracted 3 XRR point(s).",
            "stopped": "Stopped after 3 point(s).",
            "stopping": "Stopping extraction…",
            "failed": "No detector frames were found for the selected XRR series.",
        }
        assert boxes == [("XRR Series Extractor", NO_SOURCE), ("XRR Series Extractor", NO_DATASET)]
        assert dialog._extract_thread is None and dialog._inspect_thread is None  # refused before a thread
    finally:
        _dispose(dialog)


def test_the_job_messages_and_the_input_warning_are_chinese(tmp_path, chinese, boxes) -> None:
    _app()
    dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
    try:
        assert _run(dialog, tmp_path) == {
            "no source": "请先选择一个 NXS 模块文件或一个 CBF 序列。",
            "no dataset": "请填写存有 theta 值的 NXS 电机数据集路径。",
            "progress": "已提取第 1 帧，共 3 帧",
            "finished": "已提取 3 个 XRR 数据点。",
            "stopped": "已停止，共提取 3 个点。",
            "stopping": "正在停止提取…",
            "failed": "所选的 XRR 序列中没有找到探测器帧。",
        }
        assert boxes == [("XRR 序列提取", ZH[NO_SOURCE]), ("XRR 序列提取", ZH[NO_DATASET])]
        # The frame line holds names and numbers only.
        assert dialog.live_frame_label.text().startswith("1/3 · frame_0.cbf · θ ")
    finally:
        _dispose(dialog)


def test_the_window_fits_a_1280_by_720_logical_screen() -> None:
    app = _app()
    dialog = XrrSeriesDialog(view_model=FakeViewModel(), app_context=object())
    try:
        assert dialog.minimumHeight() <= 604  # 90 % of the 672 px such a screen leaves for a window
        dialog.resize(dialog.minimumWidth(), dialog.minimumHeight())
        dialog.show()
        end = time.monotonic() + 0.2
        while time.monotonic() < end:
            app.processEvents()
        assert (dialog.width(), dialog.height()) == (960, 600)
        for control in (dialog.run_button, dialog.export_button, dialog.job_status.message_label):
            assert control.isVisible()
            assert control.mapTo(dialog, QPoint(0, control.height())).y() <= dialog.height()
        assert dialog.preview_tabs.isVisible()
    finally:
        _dispose(dialog)
