"""Review check, Labs: the fixed status texts of Trainset and 2D Prediction have their Chinese, and the batch
hint and the empty batch plan follow the interface language (they are set while the page is open)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QRadioButton, QStackedWidget, QWidget

from src.gimap.app.presentation.i18n import ZH, apply_language

ROOT = Path(__file__).resolve().parents[1]
LABS = [ROOT / "src" / "gimap" / "features" / name / "presentation" for name in ("trainset", "prediction")]
# The argument that holds the message, per call: the shared status bar (runtime translates it with tr()),
# the activity log, a job status line (JobStatus translates it), a loader error shown in a message box.
MESSAGE_ARGUMENT = {
    "_append_status_message": 0,
    "set_preview_busy": 2,
    "set_local_job_status": 1,
    "set_what_if_busy": 1,
}
SIGNALS = {"status_updated", "error_occurred"}


def _literals(node):
    """The fixed texts an argument can be: a literal, either branch of ``a if c else b``, the fallback of
    ``error or "..."``, or inside tr()."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "tr" and node.args:
        return _literals(node.args[0])
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _literals(node.body) + _literals(node.orelse)
    if isinstance(node, ast.BoolOp):
        return [text for value in node.values for text in _literals(value)]
    return []


def _message_arguments(call: ast.Call):
    func = call.func
    if not isinstance(func, ast.Attribute):
        return []
    if func.attr == "emit" and isinstance(func.value, ast.Attribute) and func.value.attr in SIGNALS:
        return call.args[:1]
    index = MESSAGE_ARGUMENT.get(func.attr)
    return call.args[index:index + 1] if index is not None else []


def test_every_fixed_labs_status_text_has_its_chinese():
    missing = []
    for folder in LABS:
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                for argument in _message_arguments(node):
                    for text in _literals(argument):
                        if any(character.isalpha() for character in text) and text.strip() not in ZH:
                            missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {text!r}")
    assert missing == []


class _Signal:
    def __init__(self):
        self.sent = []

    def emit(self, value):
        self.sent.append(value)


def _panel():
    from src.gimap.features.prediction.presentation.workflow_components import PredictionInputModePanel

    QApplication.instance() or QApplication([])
    holder = type("Holder", (), {})()
    holder.ui = type("Ui", (), {})()
    holder.ui.gisaxsPredictMultiFilesRadioButton = QRadioButton()
    holder.ui.gisaxsPredictMultiFilesRadioButton.setAutoExclusive(False)  # alone here: let it uncheck
    holder.ui.gisaxsPredictShowMultiFileResultsButton = QPushButton()
    holder.pages = QStackedWidget()
    for _ in range(2):
        holder.pages.addWidget(QWidget())
    holder.hint = QLabel()
    holder.summary = QLabel()
    holder.mode_changed = _Signal()
    return PredictionInputModePanel, holder


def test_the_batch_hint_and_the_empty_plan_follow_the_language():
    panel, holder = _panel()
    batch_hint = "Use an inclusive file-number range and choose how many files form one prediction."
    single_hint = "Stack controls how many consecutive detector files contribute to this prediction."
    empty = "No detector files selected by the current folder and range."
    holder.ui.gisaxsPredictMultiFilesRadioButton.setChecked(True)
    panel.sync_mode(holder)
    panel.set_batch_summary(holder, files=0, jobs=0)
    assert (holder.hint.text(), holder.summary.text()) == (batch_hint, empty)  # English unchanged
    try:
        apply_language("zh")
        panel.sync_mode(holder)
        assert holder.hint.text() == ZH[batch_hint]
        panel.set_batch_summary(holder, files=0, jobs=0)
        assert holder.summary.text() == ZH[empty]
        holder.ui.gisaxsPredictMultiFilesRadioButton.setChecked(False)
        panel.sync_mode(holder)
        assert holder.hint.text() == ZH[single_hint]
        assert holder.mode_changed.sent[-2:] == ["multi_files", "single_file"]
    finally:
        apply_language("en")
