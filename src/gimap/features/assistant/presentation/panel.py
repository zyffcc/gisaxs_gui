"""The AI panel: what the assistant is doing now, every step it took, the changes to apply or undo, and its report."""

from __future__ import annotations

import json
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..application import BILLING_SUBSCRIPTION, GOALS, RUN_CANCELLED, RUN_COMPLETED, RUN_FAILED, RunOutcome, StepRecord
from .operation_cards import OperationList
from .report_view import report_html

LIVE_CHARS = 280
STATE_TEXT = {
    "idle": "Ready",
    "running": "Working…",
    "stopping": "Stopping…",
    RUN_COMPLETED: "Finished",
    RUN_CANCELLED: "Stopped",
    RUN_FAILED: "Failed",
}
FAILED_COLOR = QColor("#c62828")
NOTE_COLOR = QColor("#607d8b")
NOTICE_COLOR = QColor("#1e88e5")


def _tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


class AssistantPanel(QWidget):
    stopRequested = pyqtSignal()
    runAgainRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    operationRequested = pyqtSignal(str, str)
    """(operation id, "apply" | "undo" | "dismiss")."""
    undoAllRequested = pyqtSignal()
    followUpRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("assistantPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.state_label = QLabel(STATE_TEXT["idle"], self)
        self.state_label.setObjectName("assistantState")
        self.state_label.setProperty("gimapRole", "heading")
        header.addWidget(self.state_label)
        header.addStretch(1)
        self.model_label = QLabel("", self)
        self.model_label.setProperty("gimapRole", "muted")
        header.addWidget(self.model_label)
        layout.addLayout(header)
        self.task_label = QLabel("", self)
        self.task_label.setWordWrap(True)
        self.task_label.setProperty("gimapRole", "muted")
        layout.addWidget(self.task_label)
        self.live_label = QLabel("", self)
        self.live_label.setObjectName("assistantLive")
        self.live_label.setWordWrap(True)
        self.live_label.setTextFormat(Qt.PlainText)
        self.live_label.setProperty("gimapRole", "muted")
        self.live_label.setMinimumHeight(self.fontMetrics().lineSpacing() * 2)
        self.live_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.live_label)

        splitter = QSplitter(Qt.Vertical, self)
        self.splitter = splitter
        self.step_list = QListWidget(splitter)
        self.step_list.setObjectName("assistantStepList")
        self.step_list.setWordWrap(True)
        self.step_list.setAlternatingRowColors(True)
        self.operations = OperationList(splitter)
        self.report_view = QTextBrowser(splitter)
        self.report_view.setObjectName("assistantReport")
        self.report_view.setOpenExternalLinks(False)
        self.report_view.setPlaceholderText("The report appears here when the AI finishes.")
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        self.operations.actionRequested.connect(self.operationRequested)
        self.operations.undoAllRequested.connect(self.undoAllRequested)
        layout.addWidget(splitter, 1)

        self.usage_label = QLabel("", self)
        self.usage_label.setObjectName("assistantUsage")
        self.usage_label.setProperty("gimapRole", "muted")
        layout.addWidget(self.usage_label)
        follow = QHBoxLayout()
        self.follow_edit = QLineEdit(self)
        self.follow_edit.setObjectName("assistantFollowUp")
        self.follow_edit.setPlaceholderText("Ask a follow-up about this frame… (the answer can bring new cards)")
        self.follow_button = QPushButton("Ask", self)
        self.follow_button.setObjectName("assistantFollowUpButton")
        follow.addWidget(self.follow_edit, 1)
        follow.addWidget(self.follow_button)
        layout.addLayout(follow)
        buttons = QHBoxLayout()
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setObjectName("assistantStopButton")
        self.again_button = QPushButton("Run Again…", self)
        self.again_button.setObjectName("assistantAgainButton")
        self.save_button = QPushButton("Save Report…", self)
        self.save_button.setObjectName("assistantSaveButton")
        self.copy_button = QPushButton("Copy", self)
        for button in (self.stop_button, self.again_button, self.save_button, self.copy_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.stop_button.clicked.connect(self.stopRequested)
        self.again_button.clicked.connect(self.runAgainRequested)
        self.save_button.clicked.connect(self.saveRequested)
        self.copy_button.clicked.connect(self._copy)
        self.follow_button.clicked.connect(self._follow_up)
        self.follow_edit.returnPressed.connect(self._follow_up)
        self._rows: dict[int, QListWidgetItem] = {}
        self.speaker = "AI"
        self._live_kind = ""
        self._live_text = ""
        self._set_buttons(running=False, finished=False)

    # -- run lifecycle -----------------------------------------------------------------

    def begin(self, frame: str, goals, model: str, permission: str) -> None:
        self.step_list.clear()
        self._rows.clear()
        self.report_view.clear()
        self.usage_label.clear()
        self.operations.show_operations(())
        self.speaker = "Claude" if "claude" in model.lower() else "AI"
        self._set_live("", "Starting…")
        self.state_label.setText(STATE_TEXT["running"])
        self.model_label.setText(model)
        wanted = ", ".join(GOALS[goal].split(":")[0].split(" (")[0] for goal in goals.goals)
        mode = {"confirm": "asks before writing", "preview": "preview first"}.get(permission, "fully automatic")
        self.task_label.setText(f"{frame} · {wanted} · {mode}")
        self._set_buttons(running=True, finished=False)

    def stopping(self) -> None:
        self.state_label.setText(STATE_TEXT["stopping"])
        self.stop_button.setEnabled(False)

    def finish(self, outcome: RunOutcome, *, cost: Optional[float], elapsed: float, language: str = "English") -> None:
        self._set_live("", "")
        self.state_label.setText(STATE_TEXT.get(outcome.state, outcome.state.title()))
        self.report_view.setHtml(report_html(outcome, cost=cost, elapsed=elapsed, language=language))
        self.show_operations(outcome.results.operations)
        self.usage(outcome.usage, cost, elapsed, outcome.billing)
        self._set_buttons(running=False, finished=True)
        if outcome.state == RUN_FAILED:
            self._add_note(outcome.message, color=FAILED_COLOR)

    def _follow_up(self) -> None:
        text = self.follow_edit.text().strip()
        if text and self.follow_button.isEnabled():
            self.follow_edit.clear()
            self.followUpRequested.emit(text)

    def _set_buttons(self, *, running: bool, finished: bool) -> None:
        self.follow_button.setEnabled(finished and not running)
        self.follow_edit.setEnabled(not running)
        self.stop_button.setVisible(running)
        self.stop_button.setEnabled(running)
        self.again_button.setVisible(not running)
        self.save_button.setEnabled(finished)
        self.copy_button.setEnabled(finished)

    # -- progress ----------------------------------------------------------------------

    def step_started(self, step: StepRecord) -> None:
        item = QListWidgetItem(f"{step.index}. {step.tool} …")
        item.setToolTip(self._tooltip(step))
        self.step_list.addItem(item)
        self.step_list.scrollToItem(item)
        self._rows[step.index] = item
        self._set_live("tool", f"Running {step.tool}…")

    def step_finished(self, step: StepRecord) -> None:
        item = self._rows.get(step.index)
        if item is None:
            return
        mark = "" if step.ok else " ×"
        item.setText(f"{step.index}. {step.tool}{mark} — {step.summary}" if step.summary else f"{step.index}. {step.tool}{mark}")
        item.setToolTip(self._tooltip(step))
        if not step.ok:
            item.setForeground(QBrush(FAILED_COLOR))

    def show_operations(self, operations) -> None:
        self.operations.show_operations(list(operations))
        if self.operations.cards:
            total = max(1, sum(self.splitter.sizes()))
            self.splitter.setSizes([int(total * 0.2), int(total * 0.45), int(total * 0.35)])

    def model_text(self, text: str) -> None:
        self._add_note(f"{self.speaker}: {text}")

    def model_progress(self, kind: str, text: str) -> None:
        if kind == "thinking":
            self._set_live("thinking", (self._live_text if self._live_kind == "thinking" else "") + text)
        elif kind == "text":
            self._set_live("text", (self._live_text if self._live_kind == "text" else "") + text)
        elif kind == "tool":
            self._set_live("tool", f"Preparing {text}…")
        elif kind == "retry":
            self._set_live("retry", text)

    def usage(self, total, cost: Optional[float], elapsed: float, billing: str = "") -> None:
        read = total.input_tokens + total.cache_read_input_tokens + total.cache_creation_input_tokens
        text = f"{_tokens(read)} tokens in ({_tokens(total.cache_read_input_tokens)} cached) · {_tokens(total.output_tokens)} out"
        if billing == BILLING_SUBSCRIPTION:
            text += " · your Claude plan" + (f" (≈ ${cost:.2f} at API prices)" if cost is not None else "")
        elif cost is not None:
            text += f" · ≈ ${cost:.2f}"
        self.usage_label.setText(f"{text} · {elapsed:.0f} s")

    def notice(self, text: str) -> None:
        self._add_note(text, color=NOTICE_COLOR)

    # -- helpers -----------------------------------------------------------------------

    def _set_live(self, kind: str, text: str) -> None:
        self._live_kind = kind
        self._live_text = text
        shown = " ".join(text.split())
        if len(shown) > LIVE_CHARS:
            shown = "…" + shown[-LIVE_CHARS:]
        prefix = {"thinking": "Thinking: ", "text": f"{self.speaker}: "}.get(kind, "")
        self.live_label.setText(prefix + shown if shown else "")

    def _add_note(self, text: str, *, color: QColor = NOTE_COLOR) -> None:
        item = QListWidgetItem(text)
        font = item.font()
        font.setItalic(True)
        item.setFont(font)
        item.setForeground(QBrush(color))
        item.setToolTip(text)
        self.step_list.addItem(item)
        self.step_list.scrollToItem(item)

    @staticmethod
    def _tooltip(step: StepRecord) -> str:
        arguments = json.dumps(step.arguments, ensure_ascii=False)
        result = step.result if isinstance(step.result, str) else json.dumps(step.result, ensure_ascii=False)
        if result and len(result) > 1500:
            result = result[:1500] + " …"
        lines = [f"{step.tool}({arguments})", f"{step.seconds:.2f} s"]
        if result:
            lines.append(result)
        return "\n".join(lines)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.report_view.toPlainText())


__all__ = ["AssistantPanel"]
