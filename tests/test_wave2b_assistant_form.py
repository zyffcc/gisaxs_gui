"""Wave 2b, the automatic analysis's answer form (assistant-14) and its texts in Chinese (assistant-3, first pass).

The calibration field has a … button that opens a file dialog (images of a standard and .poni files) in the
frame's folder; the standard is a choice of the four GIMaP can fit, read by its key; Enter in any field runs
again; answers stay for the session only. The form, the line under Run and the progress panel are composed
in the interface language and again after a switch.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QComboBox, QFileDialog, QLineEdit, QToolButton

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.features.assistant.presentation import GuidedAnalysis
from src.gimap.features.assistant.presentation.guided_questions import STANDARD_LABELS, answer_text, calibration_filters

ZH_TEST = {
    "Only you can answer these — then run again:": "只有你能回答这些问题 —— 然后再运行：",
    "Incidence angle αi (°)": "入射角 αi (°)",
    "e.g. 0.2": "例如 0.2",
    "Choose…": "选择…",
    "Standard in that image": "该图像中的标样",
    "No detector image is open.": "没有打开探测器图像。",
    "The analysis stopped: {message}": "分析已停止：{message}",
    "Automatic analysis — stopped by an error": "自动分析 —— 因错误停止",
}
QUESTIONS = [
    {"item": "incidence angle αi", "option": "incidence_deg", "why": "No detector image is open.", "hint": ""},
    {"item": "calibration", "option": "calibration", "why": "No calibration found.", "hint": "Name it."},
    {"item": "calibration standard", "option": "standard", "why": "AgBH and LaB6 fit equally well.", "hint": ""},
]


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait(condition, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        QApplication.processEvents()
        time.sleep(0.005)
        assert time.monotonic() < deadline, "timed out"
    QApplication.processEvents()


class _Status:
    """Analyze as the form sees it: the frame shown."""

    def __init__(self, path: str):
        self.path = path

    def status(self) -> dict:
        return {"path": self.path, "mode": "giwaxs"}


@pytest.fixture()
def zh(monkeypatch):
    for english, chinese in ZH_TEST.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    yield
    apply_language(DEFAULT_LANGUAGE)


def test_the_standard_is_a_choice_of_the_four_and_the_run_reads_its_key() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    guided._show_questions(QUESTIONS)
    choice = guided.question_fields["standard"]
    assert isinstance(choice, QComboBox) and choice.objectName() == "guidedAnswer_standard"
    assert [choice.itemData(index) for index in range(choice.count())] == ["", "agbh", "lab6", "ceo2", "lab6_ceo2"]
    assert choice.itemText(1) == STANDARD_LABELS["agbh"] == "AgBH"
    assert choice.currentIndex() == 0 and guided.options().standard is None  # nothing chosen: nothing made up
    choice.setCurrentIndex(choice.findData("lab6_ceo2"))
    assert answer_text(choice) == "lab6_ceo2" and guided.options().standard == "lab6_ceo2"
    # The next round keeps the choice, and says it in words when the question is not asked again.
    guided._show_questions(QUESTIONS)
    assert guided.question_fields["standard"].currentData() == "lab6_ceo2"
    guided._show_questions(QUESTIONS[:1])
    assert "LaB6 + CeO2" in guided.answers_label.text() and guided.options().standard == "lab6_ceo2"
    guided.files_cleared()  # Clear or a project: the answers are forgotten (this session only)
    assert guided.options().standard is None and not guided.question_fields
    guided.results.deleteLater()


def test_the_calibration_file_is_chosen_in_a_dialog_that_starts_in_the_frames_folder(tmp_path: Path, monkeypatch) -> None:
    _app()
    frame = tmp_path / "beamtime" / "film_00012.tif"
    frame.parent.mkdir()
    chosen = tmp_path / "calib" / "AgBH_00001.tif"
    chosen.parent.mkdir()
    asked = []

    def ask(parent, title, folder, filters):
        asked.append((title, folder, filters))
        return str(chosen).replace("\\", "/"), filters.split(";;")[0]

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(ask))
    guided = GuidedAnalysis(lambda: _Status(str(frame)))
    guided._show_questions(QUESTIONS)
    field = guided.question_fields["calibration"]
    browse = guided.questions.findChild(QToolButton, "guidedBrowseCalibration")
    assert isinstance(field, QLineEdit) and browse is not None and browse.text() == "…"
    assert field.parentWidget() is browse.parentWidget()  # the … sits right beside the field
    browse.click()
    title, folder, filters = asked[-1]
    assert title == "Calibration File" and Path(folder) == frame.parent
    first = filters.split(";;")[0]
    assert "*.poni" in first and "*.tif" in first and "*.cbf" in first and filters.endswith("(*)")
    assert filters == calibration_filters()
    assert Path(field.text()) == chosen and guided.options().calibration == field.text()
    browse.click()  # a path already given: the dialog starts where it is
    assert Path(asked[-1][1]) == chosen.parent
    guided.results.deleteLater()


def test_enter_in_any_answer_field_runs_again() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)  # no Analyze: a run ends at once ("no image is open")
    guided._show_questions(QUESTIONS)
    requested, started = [], []
    guided.questions.runRequested.connect(lambda: requested.append(True))
    guided.started.connect(started.append)
    field = guided.question_fields["incidence_deg"]
    field.setText("0.3")
    QTest.keyClick(field, Qt.Key_Return)
    assert requested == [True] and len(started) == 1  # the run itself, not only the signal
    _wait(lambda: not guided._busy())
    assert guided._answers.get("incidence_deg") == "0.3"
    guided._show_questions(QUESTIONS)
    QTest.keyClick(guided.question_fields["standard"], Qt.Key_Enter)  # a closed drop-down too
    assert len(requested) == 2
    _wait(lambda: not guided._busy())
    guided._show_questions(QUESTIONS)  # the run without a frame asked nothing: the questions again
    QTest.keyClick(guided.question_fields["calibration"], Qt.Key_Return)
    assert len(requested) == 3
    _wait(lambda: not guided._busy())
    guided.results.deleteLater()


def test_the_form_and_the_status_follow_a_switch_of_the_language(zh) -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    guided._show_questions(QUESTIONS)
    guided.question_fields["incidence_deg"].setText("0.25")
    guided._failed("no frame")
    assert guided.status_label.text() == "The analysis stopped: no frame"
    apply_language("zh")
    guided.refresh_language()  # what the main window calls after the switch
    assert guided.status_label.text() == "分析已停止：no frame"
    field = guided.question_fields["incidence_deg"]
    assert field.text() == "0.25" and field.placeholderText() == "例如 0.2"  # what was typed stays
    assert "没有打开探测器图像。" in field.toolTip()  # the run's fixed sentence, translated
    assert guided.questions.title_label.text() == "只有你能回答这些问题 —— 然后再运行："
    assert guided.question_fields["standard"].itemText(0) == "选择…"
    assert guided.progress_panel.title_label.text() == "自动分析 —— 因错误停止"
    apply_language(DEFAULT_LANGUAGE)
    guided.refresh_language()
    assert guided.status_label.text() == "The analysis stopped: no frame"
    assert guided.question_fields["incidence_deg"].text() == "0.25"
    guided.results.deleteLater()


def test_the_notes_are_not_kept_beyond_the_session() -> None:
    _app()
    first = GuidedAnalysis(lambda: None)
    first.notes_edit.setPlainText("alpha_i = 0.4 deg, 11.8 keV")
    second = GuidedAnalysis(lambda: None)  # a new session: the notes of the last beamtime are not carried over
    assert second.notes_edit.toPlainText() == "" and second.options().incidence_deg is None
    for guided in (first, second):
        guided.results.deleteLater()
