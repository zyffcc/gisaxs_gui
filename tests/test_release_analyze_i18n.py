"""Release fixes of Analyze's texts in the interface language (R18, R32, R36, R41, and the Analyze parts of R8).

``ZH`` holds the Chinese of this round's new texts (the integrator merges them into the tables; the tests
inject them, as ``tests/test_wave2b_analyze_i18n.py`` does). The keys are the exact English of the code, so a
test that sees the Chinese on screen also pins that the key still matches its source.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QLabel, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language, apply_to
from src.gimap.features.analyze.application import BatchChoices
from src.gimap.features.analyze.infrastructure.adapters.frames import readable_read_error
from src.gimap.features.analyze.presentation import texts
from src.gimap.features.analyze.presentation.texts import MASK_SHAPE, READ_ERRORS, message_text
from src.gimap.features.analyze.presentation.views.command_bar import ElidedLabel
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_review_analyze_run_and_clear import _frames, _profiles
from tests.test_series_map import _wait

ZH = {
    # Mask & Corrections and GIWAXS options: tooltips and a label (R36)
    "Frame subtracted from every analysed frame": "从每一帧中扣除的帧",
    "Stop subtracting the background": "不再扣除背景",
    "frame − scale × background (e.g. the exposure-time ratio)": "帧 − 系数 × 背景（例如曝光时间之比）",
    "Frame of a NeXus background file": "NeXus 背景文件中的第几帧",
    "Also ignore pixels this close to detector gaps and bad pixels, which often read wrong and show up as spikes "
    "in cuts. Remembered between sessions.":
        "同时忽略离探测器缝隙和坏像素这么近的像素：它们的读数常常不准，会在切线上形成尖峰。下次启动时仍然保留。",
    "Half width of the in-plane sectors around χ = ±90°": "χ = ±90° 附近面内扇区的半宽",
    "Half width of the out-of-plane sector around χ = 0°": "χ = 0° 附近面外扇区的半宽",
    "I(q) axis": "I(q) 的横轴",
    "I(q) over a χ range, and I(χ) over its q range": "一段 χ 范围内的 I(q)，以及它的 q 范围内的 I(χ)",
    "I(qz) and I(q∥) inside a q∥–qz rectangle; drag it on the q map to move it":
        "q∥–qz 矩形内的 I(qz) 和 I(q∥)；在 q 图上拖动可以移动它",
    # The card of the Series tab while several frames are listed (R14, R19)
    "No map yet": "还没有热图",
    "The map shows the chosen curve of every frame as intensity against frame and q. A Batch Export fills it as "
    "it goes.": "热图把每一帧所选的曲线显示为强度随帧和 q 的变化。批量导出会边导出边填充它。",
    # Masks (R9)
    "Mask not loaded: {error}": "未载入掩膜：{error}",
    "The mask {name} is {mask_shape}; the frame is {shape}.": "掩膜 {name} 的大小是 {mask_shape}；这一帧是 {shape}。",
    "{name} is not a GIMaP masks file (it is not valid JSON).": "{name} 不是 GIMaP 掩膜文件（不是有效的 JSON）。",
    "{name} is not a GIMaP masks file (it has no masks list).": "{name} 不是 GIMaP 掩膜文件（里面没有掩膜列表）。",
    # A frame that cannot be read (R32)
    "{name} is not there any more (moved or deleted).": "{name} 已经不在那里了（被移动或删除）。",
    "{name} cannot be opened: {error}.": "无法打开 {name}：{error}。",
    "{name} is empty (0 bytes) — it may still be being written, or the copy failed.":
        "{name} 是空的（0 字节）——可能还在写入，或者复制失败了。",
    "{name} cannot be read: no permission (is it open in another program?).":
        "无法读取 {name}：没有权限（它是否正被别的程序打开？）。",
    "{name} has fewer frames than asked for ({error}).": "{name} 的帧数比要读的少（{error}）。",
    "{name} is not a detector image GIMaP can read (CBF, TIFF, EDF or NeXus).":
        "{name} 不是 GIMaP 能读取的探测器图像（CBF、TIFF、EDF 或 NeXus）。",
    "{name} looks damaged or incomplete (the NeXus/HDF5 file could not be opened) — if it is still being written, "
    "try again later.": "{name} 看起来已损坏或不完整（NeXus/HDF5 文件无法打开）——如果它还在写入，请稍后再试。",
    "{name} could not be read: the file looks damaged, incomplete or of another kind ({error}).":
        "无法读取 {name}：文件看起来已损坏、不完整，或者是另一种文件（{error}）。",
    "{name} could not be read ({error}).": "无法读取 {name}（{error}）。",
    # What the curve plots say with nothing to draw (R26, R33)
    "No curves: this frame could not be analysed": "没有曲线：这一帧无法分析",
}


@pytest.fixture
def chinese(monkeypatch):
    """Chinese with this round's entries (both directions); English again afterwards."""
    for english, chinese in ZH.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    apply_language("zh")
    yield
    apply_language("en")


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _listed(tmp_path: Path, count: int):
    paths = _frames(tmp_path, count)
    page = _page(_context(_profiles()))
    page.add_paths([str(path) for path in paths])
    _done(page)
    return page, paths


# -- a frame that cannot be read (R32) ----------------------------------------------------------------------------


def _read_errors(tmp_path: Path, monkeypatch) -> list[tuple[str, dict]]:
    """``(sentence, the values in it)`` of every branch of ``readable_read_error``: the file's name, the reader's
    own words (``error``), and for the NeXus / HDF5 reader's own sentences (``nexus_layout.ERROR_TEMPLATES``)
    what they name. The errors are raised by the reader itself wherever a small file can make it do so."""
    import h5py
    import numpy as np

    from src.gimap.shared.detector_io import BeingWritten, LeftUnfinished, load_detector_image

    frame = tmp_path / "frame_00012.cbf"
    frame.write_bytes(b"x" * 64)
    nexus = tmp_path / "scan_00007_m01.nxs"
    with h5py.File(nexus, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data", data=np.zeros((12, 16, 24), dtype=np.int32))
        handle.create_dataset("entry/instrument/detector/description", data=np.bytes_("Lambda"))
        handle.create_group("entry/notes")
    growing = tmp_path / "growing_00008.nxs"
    with h5py.File(growing, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data", shape=(0, 16, 24), maxshape=(None, 16, 24), dtype="i4")
    weights = tmp_path / "weights.h5"
    with h5py.File(weights, "w") as handle:
        handle.attrs["keras_version"] = "2.11.0"
        handle.create_dataset("model_weights/dense/kernel:0", data=np.zeros((64, 32), dtype=np.float32))
    damaged = tmp_path / "damaged_00009.nxs"
    damaged.write_bytes(b"x" * 64)
    empty = tmp_path / "empty_frame.tif"
    empty.write_bytes(b"")

    def raised(path: Path, **options) -> BaseException:
        """What the reader itself raises for ``path``."""
        try:
            load_detector_image(path, **options)
        except Exception as exc:  # noqa: BLE001 - the error is what is wanted
            return exc
        raise AssertionError(f"{path.name} was read")

    description = "/entry/instrument/detector/description"
    found = [
        (readable_read_error(tmp_path / "moved_away.tif", FileNotFoundError("gone")), {"name": "moved_away.tif"}),
        (readable_read_error(empty, ValueError("no data")), {"name": "empty_frame.tif"}),
        (readable_read_error(nexus, BeingWritten(nexus)), {"name": nexus.name}),
        (readable_read_error(nexus, LeftUnfinished(nexus)), {"name": nexus.name}),
        (readable_read_error(weights, raised(weights)), {"name": "weights.h5"}),
        (readable_read_error(nexus, raised(nexus, dataset_path=description)),
         {"dataset": description, "shape": "()", "dtype": "|S6"}),
        (readable_read_error(nexus, raised(nexus, dataset_path="/entry/notes")), {"dataset": "/entry/notes"}),
        (readable_read_error(frame, PermissionError(13, "Permission denied")), {"name": frame.name}),
        (readable_read_error(growing, raised(growing)), {"name": growing.name}),
        (readable_read_error(nexus, raised(nexus, frame_idx=39)), {"name": nexus.name, "error": "frame 40 of 12"}),
        (readable_read_error(frame, OSError("cannot identify image file: could not identify the format")),
         {"name": frame.name}),
        (readable_read_error(nexus, OSError("Can't synchronously read data (required filter 'bitshuffle' is not "
                                            "registered)")), {"name": nexus.name}),
        (readable_read_error(damaged, raised(damaged)), {"name": damaged.name}),
        (readable_read_error(frame, AttributeError("'NoneType' object has no attribute 'find'")),
         {"name": frame.name, "error": "AttributeError: 'NoneType' object has no attribute 'find'"}),
        (readable_read_error(frame, RuntimeError("the reader gave up")),
         {"name": frame.name, "error": "RuntimeError: the reader gave up"}),
    ]
    locked = tmp_path / "locked_frame.tif"
    locked.write_bytes(b"x" * 64)
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.name == "locked_frame.tif":
            raise OSError(5, "Access is denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    found.append((readable_read_error(locked, ValueError("x")), {"name": locked.name, "error": "Access is denied"}))
    monkeypatch.setattr(Path, "stat", real_stat)
    return found


def _template_of(sentence: str):
    return next((template for template, pattern, _convert in texts._PATTERNS if pattern.fullmatch(sentence)), None)


def test_every_reason_a_frame_cannot_be_read_has_its_template(tmp_path: Path, monkeypatch) -> None:
    """The sentences are made in the infrastructure and recognised in the presentation: the two lists agree."""
    from src.gimap.shared.detector_io.nexus_layout import ERROR_TEMPLATES, FRAME_BEYOND

    sentences = _read_errors(tmp_path, monkeypatch)
    matched = [_template_of(sentence) for sentence, _values in sentences]
    assert None not in matched, [sentence for (sentence, _v), found in zip(sentences, matched) if found is None]
    assert sorted(matched) == sorted(READ_ERRORS)  # each sentence its own template, every template used
    for (sentence, values), template in zip(sentences, matched):
        assert message_text(sentence) == sentence == template.format(**values)  # English: as it came
    # The NeXus / HDF5 reader's own sentences reach the page as they are (the frame it lacks: inside “fewer frames”).
    assert set(ERROR_TEMPLATES) - {FRAME_BEYOND} <= set(READ_ERRORS)
    assert texts.FEWER_FRAMES.format(name="x.nxs", error=FRAME_BEYOND.format(frame=40, frames=12)) in [
        sentence.replace("scan_00007_m01.nxs", "x.nxs") for sentence, _values in sentences]


def test_the_reason_is_said_in_the_interface_language_with_its_names_kept(tmp_path: Path, monkeypatch, chinese) -> None:
    for sentence, values in _read_errors(tmp_path, monkeypatch):
        template = _template_of(sentence)
        shown = message_text(sentence)
        said = dict(values)
        if said.get("error") == "frame 40 of 12":  # the reader's own words for the frame, in Chinese too
            said["error"] = "第 40 帧，共 12 帧"
        assert shown != sentence and shown == i18n.ZH[template].format(**said), sentence
        assert all(value in shown for value in said.values())  # the file and the reader's own words stay
    assert message_text(MASK_SHAPE.format(name="mask_a.edf", mask_shape="619×487", shape="1043×981")) == (
        "掩膜 mask_a.edf 的大小是 619×487；这一帧是 1043×981。")


def test_the_page_says_a_frame_it_cannot_read_in_chinese_and_again_after_a_switch(tmp_path: Path, chinese) -> None:
    page, _paths = _listed(tmp_path, 1)
    failed = []
    page.analysisFailed.connect(failed.append)
    try:
        broken = tmp_path / "broken_frame.cbf"
        broken.write_bytes(b"not a detector")
        page.add_paths([str(broken)])
        _done(page)
        english = failed[0]
        assert english.startswith("broken_frame.cbf could not be read")  # the signal stays English
        said = message_text(english)
        assert said.startswith("无法读取 broken_frame.cbf") and said != english
        assert page.status_text() == f"无法分析这一帧：{said}" and page.status_level() == "error"
        assert page.step_rail.detail("data") == said and page.data_info_label.text() == said
        assert page.file_meta.text() == "无法读取"
        assert page.top_plot.empty_overlay.label.text() == "没有曲线：这一帧无法分析"

        apply_language("en", [page])
        page.refresh_language()
        assert page.status_text() == f"Could not analyse the frame: {english}"
        assert page.step_rail.detail("data") == english and page.data_info_label.text() == english
        assert page.file_meta.text() == "could not be read"
        assert page.top_plot.empty_overlay.label.text() == "No curves: this frame could not be analysed"
        page._status("Ready")  # the status line says something else by now: it is left alone
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text() == "就绪" and page.step_rail.detail("data") == said  # (the walker's own text)
    finally:
        _close(page)


# -- masks (R9) -----------------------------------------------------------------------------------------------------


def test_a_refused_mask_says_why_in_chinese(tmp_path: Path, chinese) -> None:
    import numpy as np

    from tests.test_assistant_calibration import SHAPE, save_tiff

    page, _paths = _listed(tmp_path, 1)
    try:
        other = save_tiff(tmp_path / "mask_other_detector.tif", np.ones((120, 100), dtype=np.float32))
        assert page.load_masks(other) is False
        assert page.status_text() == (
            f"未载入掩膜：掩膜 mask_other_detector.tif 的大小是 120×100；这一帧是 {SHAPE[0]}×{SHAPE[1]}。")
        (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
        assert page.load_masks(tmp_path / "notes.json") is False
        assert page.status_text() == "notes.json 不是 GIMaP 掩膜文件（里面没有掩膜列表）。"
    finally:
        _close(page)


# -- the command bar's sentence when narrow (R18) ------------------------------------------------------------------


def test_a_shortened_label_gives_its_whole_text_so_the_language_follows(chinese) -> None:
    _app()
    apply_language("en")
    host = QWidget()
    sentence = "No data yet — open or drop detector frames"
    label = ElidedLabel(sentence, host)
    host.resize(150, 40)
    label.setGeometry(0, 0, 150, 30)
    host.show()
    _app().processEvents()
    drawn = QLabel.text(label)
    assert drawn != sentence and "…" in drawn  # too narrow: shortened on screen
    assert label.text() == label.full_text() == sentence  # … but the label's text is the whole sentence
    apply_to(host, "zh")
    assert label.full_text() == "还没有数据——打开或拖入探测器帧" == label.text()
    apply_to(host, "en")
    assert label.full_text() == sentence
    label.setText("film_0001_m01.tif")  # a file name: in no table, left alone
    apply_to(host, "zh")
    assert label.full_text() == "film_0001_m01.tif"
    host.close()
    host.deleteLater()


def test_the_empty_command_bar_is_chinese_at_a_laptop_width(chinese) -> None:
    page = _page(_context())
    try:
        page.resize(1070, 700)  # the page of a 1280 px window: the sentence does not fit and is shortened
        page.show()
        _app().processEvents()
        apply_language("zh", [page])
        page.refresh_language()
        assert page.file_chip.full_text() == "还没有数据——打开或拖入探测器帧"
        page.clear_files()
        assert page.file_chip.full_text() == "还没有数据——打开或拖入探测器帧"
        apply_language("en", [page])
        assert page.file_chip.full_text() == "No data yet — open or drop detector frames"
    finally:
        _close(page)


def test_the_short_button_texts_come_back_exactly_after_a_switch_from_chinese(chinese) -> None:
    """A narrow bar started in Chinese sets its short texts itself (“拟合”); the walker turned that back into
    “fit” (another key with the same Chinese). The bar says the texts of its state again."""
    page = _page(_context())
    try:
        apply_language("zh", [page])
        page.resize(1070, 700)  # the page of a 1280 px window: the short texts
        page.show()
        for _ in range(4):
            _app().processEvents()
        bar = page.command_bar
        assert bar.short and page.fit_button.text() == "拟合" and page.run_pipeline_button.text() == "运行分析"
        apply_language("en", [page])
        page.refresh_language()
        assert (page.run_pipeline_button.text(), page.fit_button.text()) == ("Run Analysis", "Fitting")
        for _ in range(4):
            _app().processEvents()
        spin = page.incidence_spin  # “αi from profile” is wider than “αi 来自配置”: the field follows
        assert spin.text() == "αi from profile" and spin.width() >= spin.sizeHint().width()
        apply_language("zh", [page])
        page.refresh_language()
        assert page.fit_button.text() == "拟合"
        page.resize(1500, 700)  # room for the full texts: in the language of now
        for _ in range(4):
            _app().processEvents()
        assert not bar.short and page.fit_button.text() == "发送到拟合"
        apply_language("en", [page])
        page.refresh_language()
        assert page.fit_button.text() == "Send to Fitting" and page.run_pipeline_button.text() == "Run Automatic Analysis"
    finally:
        _close(page)


# -- tooltips and a label of the options (R36) ----------------------------------------------------------------------


def test_the_options_tooltips_and_the_axis_label_have_chinese(chinese) -> None:
    page = _page(_context())
    try:
        tips = {
            page.background_button: "Frame subtracted from every analysed frame",
            page.background_clear_button: "Stop subtracting the background",
            page.background_scale_spin: "frame − scale × background (e.g. the exposure-time ratio)",
            page.background_frame_spin: "Frame of a NeXus background file",
            page.gap_guard_spin: "Also ignore pixels this close to detector gaps and bad pixels, which often read wrong "
                                 "and show up as spikes in cuts. Remembered between sessions.",
            page.in_plane_spin: "Half width of the in-plane sectors around χ = ±90°",
            page.out_of_plane_spin: "Half width of the out-of-plane sector around χ = 0°",
            page.sector_check: "I(q) over a χ range, and I(χ) over its q range",
            page.box_check: "I(qz) and I(q∥) inside a q∥–qz rectangle; drag it on the q map to move it",
        }
        for widget, english in tips.items():
            assert widget.toolTip() == english  # the key is the exact text of the code
        assert any(label.text() == "I(q) axis" for label in page.findChildren(QLabel))
        apply_language("zh", [page])
        for widget, english in tips.items():
            assert widget.toolTip() == ZH[english], english
        labels = [label.text() for label in page.findChildren(QLabel)]
        assert "I(q) 的横轴" in labels and "I(q) axis" not in labels
        apply_language("en", [page])
        for widget, english in tips.items():
            assert widget.toolTip() == english
    finally:
        _close(page)


# -- texts composed once that follow a language switch (R41, R8) -----------------------------------------------------


def test_the_batch_entry_and_the_undo_tooltips_follow_the_language(tmp_path: Path, chinese) -> None:
    apply_language("en")
    page, _paths = _listed(tmp_path, 3)
    try:
        assert page.data_batch_button.text() == "Batch Export 3 Frames…"
        page.choose_mode("giwaxs")  # a step to undo
        _done(page)
        assert page.undo_button.toolTip() == "Undo: mode (Ctrl+Z)"
        assert page.redo_button.toolTip() == "Nothing to redo (Ctrl+Shift+Z)"
        toasts_before = getattr(page, "_batch_announced", False)

        apply_language("zh", [page])
        page.refresh_language()
        assert page.data_batch_button.text() == "批量导出 3 帧…"
        assert page.undo_button.toolTip() == "撤销：模式（Ctrl+Z）"
        assert page.redo_button.toolTip() == "没有可重做的修改（Ctrl+Shift+Z）"
        assert getattr(page, "_batch_announced", False) == toasts_before  # the first-time note is not said again

        apply_language("en", [page])
        page.refresh_language()
        assert page.data_batch_button.text() == "Batch Export 3 Frames…"
        assert page.undo_button.toolTip() == "Undo: mode (Ctrl+Z)"
        page.clear_files()  # no frame: the tooltips still follow (nothing else is composed)
        apply_language("zh", [page])
        page.refresh_language()
        assert page.undo_button.toolTip().startswith("撤销：") or page.undo_button.toolTip() == "没有可撤销的修改（Ctrl+Z）"
    finally:
        _close(page)


def test_the_panel_of_a_batch_that_ended_follows_the_language(tmp_path: Path, chinese) -> None:
    apply_language("en")
    page, _paths = _listed(tmp_path / "frames", 3)
    target = tmp_path / "exported"
    try:
        page.run_batch(target, BatchChoices(tables=True, per_frame=False, speed="gentle"), stem="film")
        _wait(page, lambda: not page.batch_running())
        _done(page)
        panel = page.batch_panel
        assert not panel.isHidden() and panel.title_label.text() == "Batch Export done"
        english = panel.now_label.text()
        # (The folder as the batch panel names it: whole, or by its last names with the whole path in the tooltip.)
        from src.gimap.features.analyze.presentation import batch_texts

        folder = getattr(batch_texts, "short_folder", str)(target)
        assert english.startswith("3 frames in 0:") and english.endswith(f"→ {folder}")

        apply_language("zh", [page])
        page.refresh_language()
        assert panel.title_label.text() == "批量导出完成"
        assert panel.now_label.text().startswith("3 帧，用时 0:") and panel.now_label.text().endswith(f"→ {folder}")
        assert not panel.open_button.isHidden() and panel.pause_button.isHidden()

        apply_language("en", [page])
        page.refresh_language()
        assert panel.title_label.text() == "Batch Export done" and panel.now_label.text().startswith("3 frames in 0:")
    finally:
        _close(page)
