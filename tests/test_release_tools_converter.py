"""Format Converter, release round: the run in the interface language, "Convert Current File…"
with a frame the converter cannot read, and a window that fits a 1280x720 logical screen.

The review box, the step boxes, the Add Folder dialog and the progress window compose their texts
with ``tr``/``trf`` (R31, R73). Tools ▸ Convert Current File… with an EDF frame shown in Analyze
opens the converter with a sentence that says why the frame is not an input (R12). The minimum
height is 600 px (R72).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.format_converter.presentation.bindings.workflow import review_text
from src.gimap.features.format_converter.presentation.dialog import (
    ConversionProgressDialog,
    FolderImportDialog,
    FormatConverterDialog,
)
from src.gimap.features.format_converter.presentation.state import ConversionReviewState
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from tests.test_release_tools_i18n import switch_to_chinese

UNSUPPORTED = "{name} is not an NXS, CBF or TIFF file: the converter cannot read it."


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dispose(*windows) -> None:
    """Close the windows and delete them now, by the event loop (not inside a garbage collection)."""
    for window in windows:
        for name in ("_preview_thread", "_conversion_thread"):
            thread = getattr(window, name, None)
            if thread is not None:
                thread.wait(10000)
        if hasattr(window, "running"):
            window.running = False  # a progress window refuses to close while its job runs
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _settle(app, seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def _write_nxs(path: Path, frames: int = 2) -> Path:
    """A synthetic series: ``frames`` images of 8 × 6 pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = np.arange(frames * 8 * 6, dtype=np.uint16).reshape(frames, 8, 6)
        handle.create_dataset("/entry/instrument/detector/data", data=data)
    return path


def _dialog(**kwargs) -> FormatConverterDialog:
    dialog = FormatConverterDialog(app_context=_context(), **kwargs)
    dialog._start_preview = lambda _source: None  # no preview thread in these tests
    return dialog


@pytest.fixture
def chinese(monkeypatch):
    switch_to_chinese(monkeypatch)
    yield
    apply_language("en")


@pytest.fixture
def boxes(monkeypatch):
    """The message boxes a test would have shown: (kind, title, text)."""
    shown = []

    def record(kind):
        def box(_parent, title, text, *_args, **_kwargs):
            shown.append((kind, title, text))
            return QMessageBox.No

        return box

    for kind in ("warning", "information", "question", "critical"):
        monkeypatch.setattr(QMessageBox, kind, record(kind))
    return shown


@pytest.fixture
def review(monkeypatch):
    """The "Confirm conversion" box as it would be shown; nobody presses a button."""
    seen = {}

    def exec_(box):
        seen.update(title=box.windowTitle(), text=box.text(), buttons=sorted(b.text() for b in box.buttons()))
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", exec_)
    return seen


# -- the review ----------------------------------------------------------------------------------------

REVIEW = ConversionReviewState(
    input_summary="2 NXS file(s), 1 TIFF file(s)",
    image_count=12345,
    output_files=12345,
    estimated_bytes=3 * 1024**3,
    destination="D:/beamtime/converted",
    naming="{source}_{frame:06d}.tif",
    is_large_output=True,
    input_counts=(("NXS", 2), ("TIFF", 1)),
)


def test_the_review_reads_in_english_as_it_always_did() -> None:
    assert review_text(REVIEW, "TIFF") == (
        "Input:\n2 NXS file(s), 1 TIFF file(s)\n12,345 selected image(s) / frame(s)\n\n"
        "Output:\nTIFF\nDestination: D:/beamtime/converted\nNaming: {source}_{frame:06d}.tif\n\n"
        "Estimated output:\n12,345 file(s), approximately 3.0 GB"
        "\n\n⚠ Large output: conversion can take considerable time and disk space."
    )


def test_the_review_is_chinese_with_the_numbers_and_names_kept(chinese) -> None:
    assert review_text(REVIEW, "TIFF") == (
        "输入：\n2 个 NXS 文件, 1 个 TIFF 文件\n已选 12,345 张图像 / 帧\n\n"
        "输出：\nTIFF\n输出位置：D:/beamtime/converted\n命名：{source}_{frame:06d}.tif\n\n"
        "预计输出：\n12,345 个文件，约 3.0 GB"
        "\n\n⚠ 输出量很大：转换可能需要很长时间和很多磁盘空间。"
    )


# -- the three steps -------------------------------------------------------------------------------


def test_every_box_of_the_run_is_chinese(tmp_path, chinese, boxes, review) -> None:
    _app()
    dialog = _dialog()
    try:
        dialog._next()
        assert boxes == [("information", "格式转换", "请至少添加一个输入文件。")]
        assert dialog.stack.currentIndex() == 0

        dialog.add_paths([str(_write_nxs(tmp_path / "scan_a.nxs"))])
        assert dialog.input_note.text() == "1 个源文件，共 2 张可用图像 / 帧。"
        row = dialog.input_tree.topLevelItem(0)
        assert row.text(4) == "就绪"  # the model keeps "Ready"
        assert dialog.sources[0].status == "Ready"
        assert [row.child(index).text(0) for index in range(row.childCount())] == [
            "数据集：/entry/instrument/detector/data", "形状：2 × 8 × 6", "帧：1–2",
        ]

        dialog._next()
        dialog.view_model.set_all_included(False)
        dialog._next()
        assert boxes[-1] == ("information", "格式转换", "请至少选择一张要转换的图像或一帧。")
        dialog.view_model.set_all_included(True)
        dialog.selection_table.clearSelection()
        dialog._apply_frame_selection()
        assert boxes[-1] == ("information", "帧选择", "请先选中一行或多行输入。")

        dialog._next()
        assert dialog.stack.currentIndex() == 2
        dialog.destination_edit.setText("")
        dialog._next()
        assert boxes[-1] == ("warning", "输出设置", "请选择输出位置。")
        dialog.destination_edit.setText(str(tmp_path / "converted"))
        dialog.naming_combo.setEditText("{nothing}")
        dialog._next()
        assert boxes[-1][:2] == ("warning", "输出设置") and boxes[-1][2].startswith("输出设置无效：\n")
        assert dialog.naming_example.text() == "示例：命名模板无效"

        dialog.naming_combo.setEditText("{source}_{frame:06d}")
        dialog._next()
        assert review["title"] == "确认转换"
        assert review["buttons"] == sorted(["返回", "转换", "取消"])
        assert review["text"].startswith("输入：\n1 个 NXS 文件\n已选 2 张图像 / 帧\n\n输出：\n")
        assert "\n预计输出：\n2 个文件，约 " in review["text"]
        assert dialog._conversion_thread is None  # nothing was confirmed
    finally:
        _dispose(dialog)


def test_the_file_dialogs_of_the_converter_have_chinese_captions(tmp_path, chinese, boxes, monkeypatch) -> None:
    _app()
    dialog = _dialog()
    folder = FolderImportDialog(dialog, dialog.view_model)
    try:
        captions = []
        monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                            lambda _p, title, *_a: captions.append(title) or ([], ""))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            lambda _p, title, *_a: captions.append(title) or "")
        dialog._choose_files()
        dialog._choose_destination()
        folder._browse()
        assert captions == ["添加探测器图像", "选择输出文件夹", "选择输入文件夹"]

        for check in (folder.cbf, folder.tiff, folder.nxs):
            check.setChecked(False)
        folder._accept_if_valid()
        folder.cbf.setChecked(True)
        folder.path_edit.setText(str(tmp_path / "not_a_folder"))
        folder._accept_if_valid()
        assert boxes == [
            ("warning", "从文件夹添加", "请至少选择一种输入格式。"),
            ("warning", "从文件夹添加", "请选择一个有效的文件夹。"),
        ]
    finally:
        _dispose(folder, dialog)


def test_the_static_texts_of_the_run_are_in_the_table(chinese) -> None:
    """Labels, headers and buttons are translated by the walker when a window is shown."""
    _app()
    dialog = _dialog()
    folder = FolderImportDialog(dialog, dialog.view_model)
    progress = ConversionProgressDialog("D:/beamtime/converted")
    try:
        apply_language("zh", [dialog, folder, progress])
        assert dialog.selection_table.horizontalHeaderItem(0).text() == "选用"
        assert dialog.rangeToLabel.text() == "至"
        assert [caption.text() for caption in dialog.preview_captions] == ["首帧", "中间帧", "末帧"]
        assert folder.windowTitle() == "从文件夹添加"
        assert (folder.folder_label.text(), folder.include_label.text()) == ("文件夹：", "包含：")
        assert folder.recursive.text() == "同时扫描子文件夹"
        assert folder.subfolder_note.text() == "除非勾选此项，否则不会扫描子文件夹。"
        assert progress.title.text() == "正在准备转换…"
        assert progress.time_label.text() == "已用时：00:00:00"
        assert (progress.open_button.text(), progress.report_button.text()) == ("打开输出文件夹", "查看报告")
        apply_language("en", [dialog, folder, progress])
        assert dialog.selection_table.horizontalHeaderItem(0).text() == "Use"
        assert folder.windowTitle() == "Add Folder" and progress.title.text() == "Preparing conversion…"
    finally:
        _dispose(progress, folder, dialog)


# -- the progress window -----------------------------------------------------------------------------

PROGRESS = {
    "en": {
        "title": "Converting scan_a.nxs",
        "detail": "Frame 1 · 1 / 4",
        "job": "scan_a.nxs · frame 1 · 1 / 4",
        "time": "Elapsed: 00:00:00    Remaining: approximately 00:00:00",
        "estimating": "Elapsed: 00:00:00    Remaining: approximately calculating…",
        "paused": ("Conversion paused", "Resume"),
        "cancelling": "Cancelling after the current image…",
        "done": ("Conversion completed", "2 succeeded\n1 failed"),
        "cancelled": "Conversion cancelled",
        "failed": "Conversion could not be completed",
    },
    "zh": {
        "title": "正在转换 scan_a.nxs",
        "detail": "第 1 帧 · 1 / 4",
        "job": "scan_a.nxs · 第 1 帧 · 1 / 4",
        "time": "已用时：00:00:00    剩余：约 00:00:00",
        "estimating": "已用时：00:00:00    剩余：约 正在估算…",
        "paused": ("转换已暂停", "继续"),
        "cancelling": "当前图像完成后取消…",
        "done": ("转换完成", "成功 2 个\n失败 1 个"),
        "cancelled": "转换已取消",
        "failed": "转换未能完成",
    },
}


@pytest.mark.parametrize("language", ("en", "zh"))
def test_the_progress_window_follows_the_language(language, chinese) -> None:
    _app()
    apply_language(language)
    expected = PROGRESS[language]
    dialog = _dialog()
    progress = ConversionProgressDialog("D:/beamtime/converted")
    try:
        dialog._progress_dialog = progress
        dialog._conversion_worker = SimpleNamespace(set_paused=lambda _paused: None, cancel=lambda: None)
        dialog._conversion_started_at = time.time()
        dialog._conversion_progress(0, 4, "scan_a.nxs", 0)
        assert progress.time_label.text() == expected["estimating"]
        dialog._conversion_progress(1, 4, "scan_a.nxs", 0)
        assert progress.title.text() == expected["title"]
        assert progress.detail.text() == expected["detail"]
        assert progress.job_status.message_label.text() == expected["job"]
        assert progress.time_label.text() == expected["time"]
        dialog._toggle_pause()
        assert (progress.title.text(), progress.pause_button.text()) == expected["paused"]
        assert progress.job_status.message_label.text() == expected["paused"][0]
        dialog._toggle_pause()
        dialog._cancel_conversion()
        assert progress.title.text() == expected["cancelling"]
        assert progress.job_status.message_label.text() == expected["cancelling"]

        report = SimpleNamespace(report_path="", cancelled=False, succeeded=[1, 2], failed=[3])
        progress.complete(report)
        assert (progress.title.text(), progress.result.text()) == expected["done"]
        report.cancelled = True
        progress.complete(report)
        assert progress.title.text() == expected["cancelled"]
        progress.fail("disk full")
        assert progress.title.text() == expected["failed"] and progress.result.text() == "disk full"
    finally:
        dialog._conversion_worker = None
        dialog._progress_dialog = None
        _dispose(progress, dialog)


# -- Convert Current File… with a frame the converter cannot read (R12) ----------------------------


def test_a_current_frame_the_converter_cannot_read_is_refused_in_words(tmp_path, boxes) -> None:
    _app()
    edf = tmp_path / "agbh_frame_0001.edf"
    edf.write_bytes(b"")  # only the kind of file matters
    reason = UNSUPPORTED.format(name="agbh_frame_0001.edf")
    dialog = _dialog(current_file=str(edf))
    try:
        assert dialog.sources == [] and boxes == []
        assert dialog.input_note.text() == reason  # not "No input files yet."
        assert dialog.input_note.property("gimapRole") == "warning"
        assert not dialog.current_button.isEnabled()
        assert dialog.current_button.toolTip() == reason
        assert dialog.current_file_problem() == reason

        # The Tools menu reuses the open window: it sets the file, enables the button and adds it.
        nxs = _write_nxs(tmp_path / "scan_a.nxs")
        dialog.current_file = str(nxs)
        dialog.current_button.setEnabled(True)
        dialog.add_paths([str(nxs)])
        assert [Path(source.path).name for source in dialog.sources] == ["scan_a.nxs"]
        assert dialog.current_button.isEnabled() and dialog.current_button.toolTip() == ""
        assert dialog.input_note.text() == "1 source file(s), 2 available image(s) / frame(s)."
        assert dialog.input_note.property("gimapRole") == "muted"

        dialog.current_file = str(edf)
        dialog.current_button.setEnabled(True)
        dialog.add_paths([str(edf)])
        assert len(dialog.sources) == 1 and boxes == []  # said in the window, not in an error box
        assert dialog.input_note.text() == reason
        assert not dialog.current_button.isEnabled() and dialog.current_button.toolTip() == reason

        # The same in one call, which returns the reason for the caller.
        assert dialog.use_current_file(str(nxs)) == ""
        assert dialog.current_button.isEnabled()
        assert dialog.use_current_file(str(edf)) == reason
        assert dialog.use_current_file("") == ""
        assert not dialog.current_button.isEnabled() and dialog.current_button.toolTip() == ""

        # A file somebody chose is still reported by the box, as before.
        dialog.use_current_file(str(edf))
        dialog.add_paths([str(tmp_path / "other_frame.edf")])
        assert len(boxes) == 1 and boxes[0][1] == "Some inputs could not be added"
        assert boxes[0][2].startswith("other_frame.edf: ")
    finally:
        _dispose(dialog)


def test_the_refusal_is_chinese_in_chinese(tmp_path, chinese, boxes) -> None:
    _app()
    dialog = _dialog(current_file=str(tmp_path / "agbh_frame_0001.edf"))
    try:
        assert dialog.input_note.text() == "agbh_frame_0001.edf 不是 NXS、CBF 或 TIFF 文件：转换器无法读取它。"
        assert dialog.current_button.toolTip() == dialog.input_note.text() and boxes == []
    finally:
        _dispose(dialog)


def test_a_current_frame_the_converter_reads_becomes_an_input(tmp_path, boxes) -> None:
    _app()
    nxs = _write_nxs(tmp_path / "scan_a.nxs")
    dialog = _dialog(current_file=str(nxs))
    try:
        assert [Path(source.path).name for source in dialog.sources] == ["scan_a.nxs"]
        assert dialog.current_button.isEnabled() and boxes == []
    finally:
        _dispose(dialog)
    without = _dialog()
    try:
        assert not without.current_button.isEnabled() and without.input_note.text() == "No input files yet."
    finally:
        _dispose(without)


# -- the window fits a 1280x720 logical screen (R72) -------------------------------------------------


def test_every_step_is_whole_at_the_minimum_size(tmp_path) -> None:
    app = _app()
    dialog = _dialog()
    try:
        assert (dialog.minimumWidth(), dialog.minimumHeight()) == (920, 600)  # 604 px are left for a window
        dialog.add_paths([str(_write_nxs(tmp_path / "scan_a.nxs"))])
        dialog.resize(920, 600)
        dialog.show()
        for step in range(3):
            _settle(app, 0.2)
            assert dialog.stack.currentIndex() == step
            assert (dialog.width(), dialog.height()) == (920, 600)
            for button in (dialog.back_button, dialog.cancel_button, dialog.next_button):
                assert button.isVisible()
                bottom = button.mapTo(dialog, QPoint(0, button.height())).y()
                assert bottom <= dialog.height(), (step, button.text())
            if step < 2:
                dialog._next()
    finally:
        _dispose(dialog)
