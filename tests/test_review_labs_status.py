"""Review fixes, Labs: the status messages made at run time can be translated (trf), and the shared Labs
status bar shows the message of the tool that is open, never the other tool's."""

from __future__ import annotations

import ast
import string
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QMessageBox

from src.gimap.app.presentation.i18n import ZH, apply_language
from tests.test_labs_trainset import _settle, _trainset, _wait

ROOT = Path(__file__).resolve().parents[1]
LABS = [ROOT / "src" / "gimap" / "features" / name / "presentation" for name in ("trainset", "prediction")]
# Calls that put a message in front of the user: the status bar, the activity log, a message box, a toast,
# a job status line. A composed (f-string, .format, %) English text there can never match the zh table.
MESSAGE_CALLS = {
    "emit", "_append_status_message", "information", "warning", "critical", "question", "show_toast",
    "set_local_job_status", "set_preview_busy", "set_what_if_busy",
}


def _sources():
    for folder in LABS:
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path, ast.parse(path.read_text(encoding="utf-8"))


def _composed(node) -> bool:
    """An f-string with a value, a ``.format`` or a ``%`` on a literal (a ``"\\n".join`` of data is allowed)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        return False
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("tr", "trf"):
        return False
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(value, ast.FormattedValue) for value in node.values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        template = node.func.value  # tr(template).format(...) is trf: the table holds the template
        return not (isinstance(template, ast.Call) and getattr(template.func, "id", "") == "tr")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant):
            return True
        return _composed(node.left) or _composed(node.right)
    if isinstance(node, ast.IfExp):
        return _composed(node.body) or _composed(node.orelse)
    return False


def test_no_labs_message_is_composed_where_the_zh_table_cannot_match_it():
    found = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in MESSAGE_CALLS and any(_composed(argument) for argument in node.args):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert found == [], "compose these with trf(template, **values) and give the template its Chinese"


def test_every_labs_template_has_its_chinese_with_the_same_fields():
    """Each literal tr()/trf() text of Trainset and 2D Prediction is in the zh table (exact English key),
    and a template keeps its {fields} in Chinese (trf fills them in after the lookup)."""
    missing, fields = [], []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("tr", "trf") and node.args):
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            english = first.value
            if english not in ZH:
                missing.append(english)
                continue
            names = {field for _, field, _, _ in string.Formatter().parse(english) if field}
            if names != {field for _, field, _, _ in string.Formatter().parse(ZH[english]) if field}:
                fields.append(english)
    assert missing == [] and fields == []


def _message(window) -> str:
    return window.statusbar.currentMessage()


@pytest.fixture(autouse=True)
def _no_modal_boxes(monkeypatch):
    """A message box would wait for a click offscreen: record it instead."""
    shown = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *args, **kwargs: shown.append(args)))
    return shown


def test_the_formatted_status_messages_reach_the_bar_in_chinese(tmp_path, _no_modal_boxes):
    window, trainset, page = _trainset(tmp_path)
    prediction = window.runtime.prediction
    _wait(lambda: prediction._initialized)
    try:
        # Each tool speaks on its own page (the shell may show only the open tool's messages).
        window.runtime.navigate("trainset")
        # English is unchanged: the templates give the very text the f-strings gave.
        trainset._region_created("beam_center", {"x": 12.0, "y": 34.5})
        assert _message(window) == "Beam center selected at x=12.0, y=34.5 px"
        apply_language("zh")
        trainset._load_reference(str(tmp_path / "reference.npy"))
        key = "Loaded reference scattering file: {name}"
        assert _message(window) == ZH[key].format(name="reference.npy") != key.format(name="reference.npy")
        trainset._region_created("beam_center", {"x": 12.0, "y": 34.5})
        key = "Beam center selected at x={x:.1f}, y={y:.1f} px"
        assert _message(window) == ZH[key].format(x=12.0, y=34.5)
        trainset._begin_mask("rectangle")
        assert _no_modal_boxes == []
        # One fixed sentence per shape (wave 2b, labs-4): neither language says "a ellipse".
        assert _message(window) == ZH["Draw a rectangular fixed mask in ROI coordinates"]
        trainset._begin_mask("ellipse")
        assert _message(window) == ZH["Draw an elliptical fixed mask in ROI coordinates"]

        window.runtime.navigate("predict")
        prediction.current_parameters["input_file"] = str(tmp_path / "frame_00007.cbf")
        prediction._predict_single_file()
        assert _message(window) == ZH["Processing file: {name}"].format(name="frame_00007.cbf")
        prediction._latest_display_request = 41
        prediction._on_loader_progress(41, 40, "frame_00007.cbf")
        key = "Image loading {progress}%: {message}"
        assert _message(window) == ZH[key].format(progress=40, message="frame_00007.cbf")
    finally:
        apply_language("en")
        window.close()


def test_switching_between_the_labs_tools_shows_the_open_tools_own_message(tmp_path):
    window, trainset, _page = _trainset(tmp_path)
    prediction = window.runtime.prediction
    _wait(lambda: prediction._initialized)
    runtime = window.runtime
    window.show()
    _settle(0.3)
    try:
        runtime.navigate("trainset")
        trainset.status_updated.emit("Draw the rectangular ROI on the detector image")
        runtime.navigate("predict")
        prediction._append_status_message("Batch results are ready in the workspace")
        assert _message(window) == "Batch results are ready in the workspace"

        runtime.navigate("trainset")  # the Prediction message does not stay over Trainset
        assert _message(window) == "Draw the rectangular ROI on the detector image"
        runtime.navigate("predict")
        assert _message(window) == "Batch results are ready in the workspace"

        # A message said while the other tool is open is that tool's again when it is shown.
        trainset.generation_started.emit()  # the runtime says it for Trainset
        runtime.navigate("analyze")
        runtime.navigate("trainset")
        assert _message(window) == "Trainset generation started..."
        prediction.status_updated.emit("GISAXS prediction finished!")
        runtime.navigate("predict")
        assert _message(window) == "GISAXS prediction finished!"

        # A tool that has said nothing yet clears the bar rather than keeping the other tool's line.
        prediction.page_status.text = ""
        runtime.navigate("trainset")
        runtime.navigate("predict")
        assert _message(window) == ""

        apply_language("zh")
        trainset.status_updated.emit("Trainset settings restored")
        runtime.navigate("predict")
        runtime.navigate("trainset")
        assert _message(window) == ZH["Trainset settings restored"]
    finally:
        apply_language("en")
        window.close()


def test_a_failed_preview_is_marked_failed_in_either_language(tmp_path):
    window, trainset, page = _trainset(tmp_path)
    try:
        apply_language("zh")
        trainset._preview_busy = True
        trainset._preview_failed("BornAgain is not installed")
        assert page.preview_job_status._state == "failed"
        assert ZH["Preview failed: {error}"].format(error="BornAgain is not installed") in (
            page.preview_job_status.message_label.text()
        )
    finally:
        apply_language("en")
        window.close()


def test_the_batch_summary_and_counts_keep_their_english_and_follow_chinese():
    from PyQt5.QtWidgets import QLabel

    from src.gimap.features.prediction.presentation.workflow_components import PredictionInputModePanel

    holder = type("Holder", (), {})()
    holder.summary = QLabel()
    PredictionInputModePanel.set_batch_summary(holder, files=7, jobs=3, skipped=1)
    assert holder.summary.text() == "7 files selected · 3 prediction jobs · 1 trailing file skipped"
    PredictionInputModePanel.set_batch_summary(holder, files=2, jobs=1)
    assert holder.summary.text() == "2 files selected · 1 prediction job"
    try:
        apply_language("zh")
        PredictionInputModePanel.set_batch_summary(holder, files=7, jobs=3, skipped=2)
        summary = ZH["{files} files selected · {jobs} prediction jobs"].format(files=7, jobs=3)
        assert holder.summary.text() == ZH["{summary} · {skipped} trailing files skipped"].format(
            summary=summary, skipped=2
        )
    finally:
        apply_language("en")
