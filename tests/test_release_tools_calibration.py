"""Geometry Calibration, release round: the run in the interface language, an unreadable image
said in words, and Analyze told about an applied calibration.

The engine's stage names, confidence, warnings and reasons stay English (the exported record and
the assistant read them) and are translated where the window shows them (R30, R73). A calibration
image that cannot be read is reported as a sentence, not as the reader's own error (R48). After
Apply the window calls ``geometry_calibrated()`` of the Analyze page when it has one.
"""

from __future__ import annotations

import ast
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.calibration.application import CalibrationCancelledError
from src.gimap.features.calibration.domain import (
    MANUAL_REFINEMENT_WARNING,
    CalibrationCandidate,
    CalibrationResult,
)
from src.gimap.features.calibration.presentation.bindings.image_loading import read_error_text
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from tests.test_release_tools_i18n import ZH, switch_to_chinese

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGBH = PROJECT_ROOT / "tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf"
DOMAIN = PROJECT_ROOT / "src/gimap/features/calibration/domain"
COVERAGE_WARNING = "Matched rings have very limited azimuthal coverage; verify the overlay before applying."
ONE_RING_WARNING = "Only one ring is matched; center and distance remain ambiguous."
DAMAGED = "{name} could not be read: the file looks damaged, incomplete or of another kind ({error})."


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dispose(dialog) -> None:
    """Close the window and delete it now, by the event loop (not inside a garbage collection)."""
    for name in ("_load_thread", "_cal_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _settle(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait(app, done, seconds: float = 60.0) -> None:
    end = time.monotonic() + seconds
    while not done() and time.monotonic() < end:
        _settle(app)
    assert done()


def _context() -> AppContext:
    context = AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    context.instrument_profiles = InMemoryInstrumentProfileRepository()
    return context


def _result(source: Path, *, confidence: str = "Medium", warnings=()) -> CalibrationResult:
    """A synthetic result (no detector data): the texts are what is looked at."""
    candidate = CalibrationCandidate(
        "agbh", 123.5, 234.5, 1456.7, matched_ring_count=3, rms_residual_px=1.25,
        confidence=confidence, warnings=list(warnings),
    )
    return CalibrationResult(
        str(source), 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M", 172e-6, 172e-6,
        candidate, [candidate], datetime.now(timezone.utc).isoformat(),
    )


def _show(dialog, result) -> None:
    dialog.result = result
    dialog._remember_fitted_geometry()
    dialog._populate_candidates()
    dialog._show_candidate(result.selected_candidate)


def _legend(dialog) -> str:
    return " ".join(re.sub(r"<[^>]+>", "", dialog.overlay_legend.text()).replace("&nbsp;", " ").split())


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


# -- the English is what it was ---------------------------------------------------------------------


def test_the_english_result_texts_are_unchanged(tmp_path) -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        _show(dialog, _result(tmp_path / "scan_agbh.cbf"))
        assert dialog.result_confidence.text() == "Medium"
        assert dialog.result_warning.text() == "None"
        assert dialog.preview_info_label.text() == (
            "  ·  Silver behenate (AgBH)  ·  1456.70 mm  ·  3 matched rings  ·  Medium confidence"
        )  # no image is open: the name is empty
        # The centre of the fit (index 123.5, 234.5) in the pixels of Analyze: + 0.5.
        assert [dialog.candidate_table.item(0, column).text() for column in range(6)] == [
            "Silver behenate (AgBH)", "1456.70 mm", "124.0, 235.0", "3", "1.25 px", "Medium",
        ]
        assert _legend(dialog) == "━━ Center ┄┄┄ Detected ━━ Matched ╌╌╌ Other theoretical"
        _show(dialog, _result(tmp_path / "scan_agbh.cbf", warnings=[ONE_RING_WARNING, COVERAGE_WARNING]))
        assert dialog.result_warning.text() == f"{ONE_RING_WARNING} {COVERAGE_WARNING}"
    finally:
        _dispose(dialog)


@pytest.mark.skipif(not AGBH.is_file(), reason="pyFAI AgBh test frame not available")
@pytest.mark.parametrize("language", ("en", "zh"))
def test_the_line_of_a_loaded_image_follows_the_language(language, chinese) -> None:
    app = _app()
    apply_language(language)
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        dialog.load_image(str(AGBH))
        assert dialog.stage_label.text() == {"en": "Reading image...", "zh": "正在读取图像…"}[language]
        _wait(app, lambda: dialog._load_thread is None and dialog.image is not None)
        assert dialog.stage_label.text() == {
            "en": "Loaded Pilatus1M.edf — 981 × 1043 pixels. Click Auto Calibration.",
            "zh": "已载入 Pilatus1M.edf — 981 × 1043 像素。请点击“自动标定”。",
        }[language]
        assert dialog.pixel_label.text() == {
            "en": "172.000 × 172.000 µm (metadata)",
            "zh": "172.000 × 172.000 µm（来自元数据）",
        }[language]
    finally:
        _dispose(dialog)
        _settle(app, 0.1)


# -- Chinese ------------------------------------------------------------------------------------------


def test_a_result_is_shown_in_chinese_and_kept_in_english(tmp_path, chinese) -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        result = _result(tmp_path / "scan_agbh.cbf", warnings=[ONE_RING_WARNING, COVERAGE_WARNING])
        _show(dialog, result)
        assert dialog.result_confidence.text() == "中"
        assert dialog.result_warning.text() == f"{ZH[ONE_RING_WARNING]} {ZH[COVERAGE_WARNING]}"
        assert dialog.preview_info_label.text() == (
            "  ·  山嵛酸银（AgBH）  ·  1456.70 mm  ·  匹配 3 个环  ·  置信度：中"
        )
        assert dialog.candidate_table.item(0, 0).text() == "山嵛酸银（AgBH）"
        assert dialog.candidate_table.item(0, 5).text() == "中"
        assert dialog.candidate_table.item(0, 1).text() == "1456.70 mm"  # values and units stay
        assert _legend(dialog) == "━━ 中心 ┄┄┄ 检测到的环 ━━ 匹配的环 ╌╌╌ 其他理论环"
        # What is exported and what the assistant reads stays the engine's English.
        assert result.selected_candidate.confidence == "Medium"
        assert result.selected_candidate.warnings == [ONE_RING_WARNING, COVERAGE_WARNING]
        target = tmp_path / "agbh.gimap-calibration.json"
        dialog.view_model.export_result(target)
        record = target.read_text(encoding="utf-8")
        assert '"confidence": "Medium"' in record and ONE_RING_WARNING in record
        assert "置信度" not in record and ZH[ONE_RING_WARNING] not in record

        _show(dialog, _result(tmp_path / "scan_agbh.cbf", confidence="High"))
        assert dialog.result_confidence.text() == "高" and dialog.result_warning.text() == "无"
    finally:
        _dispose(dialog)


def test_the_legend_is_written_again_after_a_language_switch(chinese) -> None:
    app = _app()
    apply_language("en")
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        assert _legend(dialog).endswith("Other theoretical")
        apply_language("zh", [dialog])
        _settle(app)
        assert _legend(dialog) == "━━ 中心 ┄┄┄ 检测到的环 ━━ 匹配的环 ╌╌╌ 其他理论环"
        apply_language("en", [dialog])
        assert _legend(dialog) == "━━ Center ┄┄┄ Detected ━━ Matched ╌╌╌ Other theoretical"
    finally:
        _dispose(dialog)


def test_the_job_line_is_chinese_for_every_state_of_a_run(chinese, boxes) -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        dialog._calibration_progress(35, "Detecting rings")
        assert dialog.stage_label.text() == "正在检测衍射环"
        dialog._calibration_failed(CalibrationCancelledError("Calibration cancelled by user."))
        assert dialog.stage_label.text() == "标定已取消。" and boxes == []
        dialog._calibration_failed(RuntimeError("A stable beam-center proposal could not be found."))
        assert dialog.stage_label.text() == "标定失败。请调整输入后重试。"
        assert boxes == [("warning", "几何标定", "找不到稳定的光束中心估计。")]
        # A reason with values in it (not a key of the table) is shown as it is.
        dialog._calibration_failed(RuntimeError("unexpected: 3 != 4"))
        assert boxes[-1] == ("warning", "几何标定", "unexpected: 3 != 4")

        dialog._calibration_finished(_result(Path("scan_agbh.cbf")))
        assert dialog.stage_label.text() == "标定完成。请检查选中的候选解，然后点击“应用”。"
    finally:
        _dispose(dialog)


def test_a_wrong_custom_range_and_the_file_dialogs_are_chinese(tmp_path, chinese, boxes, monkeypatch) -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        dialog.image = SimpleNamespace()  # only "an image is open" matters here
        dialog.range_combo.setCurrentIndex(3)  # Custom
        dialog.custom_min_spin.setValue(900.0)
        dialog.custom_max_spin.setValue(300.0)
        dialog.start_calibration()
        assert boxes == [("warning", "标定输入", "自定义距离的下限必须小于上限。")]
        assert dialog._cal_thread is None
        dialog.image = None

        asked = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            lambda _p, title, _start, pattern: asked.append((title, pattern)) or ("", ""))
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            lambda _p, title, _start, pattern: asked.append((title, pattern)) or ("", ""))
        dialog.result = _result(tmp_path / "scan_agbh.cbf")
        dialog.export_result()
        dialog.import_result()
        assert asked == [("导出标定", "JSON 文件 (*.json)"), ("导入标定", "JSON 文件 (*.json)")]

        not_a_calibration = tmp_path / "other.json"
        not_a_calibration.write_text('{"format": "something else"}', encoding="utf-8")
        dialog.import_result_from(str(not_a_calibration))
        assert boxes[-1] == ("warning", "导入标定", "这个文件不是 GIMaP 的几何标定文件。")
    finally:
        _dispose(dialog)


# -- every sentence of the engine has its Chinese ----------------------------------------------------


def _engine_sentences() -> dict[str, set[str]]:
    """Stage names, warnings and reasons as the calibration domain writes them."""
    found = {"stage": set(), "warning": set(), "reason": set()}
    for path in sorted(DOMAIN.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            last = node.args[-1]
            if not (isinstance(last, ast.Constant) and isinstance(last.value, str)):
                continue
            if name == "_stage":
                found["stage"].add(last.value)
            elif name == "_add_warning" or (name == "append" and "warnings" in ast.unparse(node.func)):
                found["warning"].add(last.value)
            elif name in ("CalibrationError", "ValueError") and "{" not in last.value:
                found["reason"].add(last.value)
    found["warning"].add(MANUAL_REFINEMENT_WARNING)
    return found


def test_every_stage_warning_and_reason_of_the_engine_has_chinese() -> None:
    sentences = _engine_sentences()
    assert len(sentences["stage"]) == 8 and len(sentences["warning"]) >= 7 and len(sentences["reason"]) >= 6
    missing = sorted(text for group in sentences.values() for text in group if text not in ZH)
    assert missing == []
    assert {"High", "Medium", "Low"} <= set(ZH)


# -- an image that cannot be read (R48) -------------------------------------------------------------


def test_an_unreadable_image_is_said_in_words() -> None:
    cut_off = AttributeError("'NoneType' object has no attribute 'find'")
    assert read_error_text("broken.cbf", cut_off) == DAMAGED.format(
        name="broken.cbf", error="AttributeError: 'NoneType' object has no attribute 'find'"
    )
    # A reader that only asserts has no text of its own: the sentence still ends properly.
    assert read_error_text("broken.cbf", AssertionError()) == DAMAGED.format(name="broken.cbf", error="AssertionError")
    assert read_error_text("gone.cbf", FileNotFoundError("Calibration image was not found: D:/x/gone.cbf")) == (
        "gone.cbf was not found: it may have been moved or deleted."
    )
    assert read_error_text("locked.cbf", PermissionError(13, "Permission denied")).startswith(
        "Could not read locked.cbf: PermissionError"
    )


def test_a_cut_off_file_shows_the_sentence_and_keeps_the_window_usable(tmp_path, boxes, chinese) -> None:
    app = _app()
    broken = tmp_path / "broken_frame.cbf"
    broken.write_bytes(b"###CBF: VERSION 1.5\r\n\r\ndata_broken\r\n")  # a header that stops
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        dialog.load_image(str(broken))
        _wait(app, lambda: dialog._load_thread is None)
        assert dialog.image is None
        assert dialog.stage_label.text() == "图像读取失败。"
        (kind, title, text), = boxes
        assert (kind, title) == ("warning", "读取标定图像")
        assert text.startswith("无法读取 broken_frame.cbf：文件看起来已损坏、不完整，或不是这种格式（")
        assert dialog.open_button.isEnabled() and not dialog.calibrate_button.isEnabled()
    finally:
        _dispose(dialog)
        _settle(app, 0.1)


# -- Analyze is told about an applied calibration ---------------------------------------------------


class _AnalyzePage:
    def __init__(self, dialog, calls):
        self._dialog, self._calls = dialog, calls

    def geometry_calibrated(self) -> None:
        # The instrument profile is recorded before Analyze is told.
        self._calls.append(("geometry_calibrated", self._dialog.view_model.last_profile is not None))

    def _refresh_profiles(self) -> None:
        self._calls.append(("_refresh_profiles", True))


class _OlderAnalyzePage:
    def __init__(self, calls):
        self._calls = calls

    def _refresh_profiles(self) -> None:
        self._calls.append(("_refresh_profiles", True))


def test_apply_tells_analyze_that_the_geometry_was_calibrated(tmp_path, boxes) -> None:
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        calls = []
        dialog.result = _result(tmp_path / "scan_agbh.cbf")
        dialog.main_window = SimpleNamespace(components=SimpleNamespace(analyze_page=_AnalyzePage(dialog, calls)))
        dialog.apply_result()
        assert calls == [("geometry_calibrated", True)]  # not the profile refresh as well

        calls.clear()
        dialog.main_window = SimpleNamespace(components=SimpleNamespace(analyze_page=_OlderAnalyzePage(calls)))
        dialog._sync_main_window_geometry()
        assert calls == [("_refresh_profiles", True)]

        dialog.main_window = SimpleNamespace(components=SimpleNamespace(analyze_page=None))
        dialog._sync_main_window_geometry()  # a window without Analyze: nothing to tell
        dialog.main_window = None
        dialog._sync_main_window_geometry()
        assert calls == [("_refresh_profiles", True)]
        _settle(app)
    finally:
        _dispose(dialog)
