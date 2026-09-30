"""Operation cards: each change the AI made or suggests, with a picture, to apply, dismiss or undo."""

from __future__ import annotations

from typing import Optional, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..application import APPLIED, DISMISSED, FAILED, FROM_PROPOSAL, FROM_RUN, PROPOSED, SUPERSEDED, UNDONE, Operation

THUMBNAIL_WIDTH = 150
STATE_TEXT = {
    PROPOSED: "Suggested",
    APPLIED: "Applied",
    DISMISSED: "Dismissed",
    UNDONE: "Undone",
    FAILED: "Could not be applied",
}
STATE_COLOR = {PROPOSED: "#1e88e5", APPLIED: "#2e7d32", DISMISSED: "#78909c", UNDONE: "#78909c", FAILED: "#c62828"}


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    label.setTextFormat(Qt.PlainText)
    return label


class OperationCard(QFrame):
    actionRequested = pyqtSignal(str, str)
    """(operation id, "apply" | "undo" | "dismiss")."""

    def __init__(self, operation: Operation, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.operation_id = operation.id
        self.setObjectName("assistantOperationCard")
        self.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        self.picture = QLabel(self)
        self.picture.setObjectName("assistantOperationPicture")
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setFixedWidth(THUMBNAIL_WIDTH)
        if operation.preview_png:
            pixmap = QPixmap()
            if pixmap.loadFromData(operation.preview_png):
                self.picture.setPixmap(pixmap.scaledToWidth(THUMBNAIL_WIDTH, Qt.SmoothTransformation))
        if self.picture.pixmap() is None or self.picture.pixmap().isNull():
            self.picture.setText("no picture")
            self.picture.setProperty("gimapRole", "muted")
        layout.addWidget(self.picture)

        text = QVBoxLayout()
        text.setSpacing(2)
        head = QHBoxLayout()
        self.title_label = QLabel(operation.title, self)
        self.title_label.setObjectName("assistantOperationTitle")
        self.title_label.setWordWrap(True)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        head.addWidget(self.title_label, 1)
        self.state_label = QLabel(STATE_TEXT.get(operation.state, operation.state), self)
        self.state_label.setObjectName("assistantOperationState")
        self.state_label.setStyleSheet(f"color: {STATE_COLOR.get(operation.state, '#607d8b')};")
        head.addWidget(self.state_label)
        text.addLayout(head)
        if operation.source == FROM_PROPOSAL:
            origin = "Suggested by the AI"
        elif operation.state == APPLIED:
            origin = "Made by the AI during its analysis"
        else:
            origin = "The AI used this during its analysis; Analyze was restored — apply to keep it"
        text.addWidget(_muted(f"{origin} · {operation.tool}", self))
        if operation.why:
            self.why_label = _muted(f"Why: {operation.why}", self)
            text.addWidget(self.why_label)
        if operation.effect:
            text.addWidget(_muted(f"Effect: {operation.effect}", self))
        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply", self)
        self.apply_button.setObjectName("assistantOperationApply")
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setObjectName("assistantOperationUndo")
        self.dismiss_button = QPushButton("Dismiss", self)
        self.dismiss_button.setObjectName("assistantOperationDismiss")
        for button in (self.apply_button, self.undo_button, self.dismiss_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        text.addLayout(buttons)
        layout.addLayout(text, 1)
        can_apply = operation.state in (PROPOSED, DISMISSED, UNDONE)
        self.apply_button.setVisible(can_apply)
        self.undo_button.setVisible(operation.state == APPLIED)
        self.undo_button.setEnabled(operation.inverse is not None)
        self.dismiss_button.setVisible(operation.state == PROPOSED)
        self.apply_button.clicked.connect(lambda: self.actionRequested.emit(self.operation_id, "apply"))
        self.undo_button.clicked.connect(lambda: self.actionRequested.emit(self.operation_id, "undo"))
        self.dismiss_button.clicked.connect(lambda: self.actionRequested.emit(self.operation_id, "dismiss"))


class OperationList(QWidget):
    """The cards of one run, newest last, with 'Undo all'."""

    actionRequested = pyqtSignal(str, str)
    undoAllRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("assistantOperations")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        head = QHBoxLayout()
        self.heading = QLabel("Changes", self)
        self.heading.setProperty("gimapRole", "heading")
        head.addWidget(self.heading)
        head.addStretch(1)
        self.undo_all_button = QPushButton("Undo All", self)
        self.undo_all_button.setObjectName("assistantOperationsUndoAll")
        head.addWidget(self.undo_all_button)
        layout.addLayout(head)
        self.hint = _muted("", self)
        layout.addWidget(self.hint)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.body = QWidget(self.scroll)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        self.body_layout.addStretch(1)
        self.scroll.setWidget(self.body)
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.scroll, 1)
        self.cards: list[OperationCard] = []
        self.undo_all_button.clicked.connect(self.undoAllRequested)
        self.show_operations(())

    def show_operations(self, operations: Sequence[Operation]) -> None:
        for card in self.cards:
            card.setParent(None)
            card.deleteLater()
        self.cards = []
        # The model's own suggestions (with reasons) first, then the changes of its analysis.
        operations = sorted(
            (operation for operation in operations if operation.state != SUPERSEDED and not operation.no_effect),
            key=lambda operation: operation.source != FROM_PROPOSAL,
        )
        for operation in operations:
            card = OperationCard(operation, self.body)
            card.actionRequested.connect(self.actionRequested)
            self.body_layout.insertWidget(self.body_layout.count() - 1, card)
            self.cards.append(card)
        waiting = sum(1 for operation in operations if operation.state == PROPOSED)
        applied = sum(1 for operation in operations if operation.state == APPLIED)
        self.heading.setText(f"Changes ({len(operations)})")
        self.hint.setText(
            f"{waiting} suggested change(s) wait for you: the picture shows the result; Apply changes Analyze, "
            "Undo takes it back." if waiting else
            ("Every change can be undone." if applied else "")
        )
        self.undo_all_button.setEnabled(applied > 0)
        self.setVisible(bool(operations))

    def card(self, identifier: str) -> Optional[OperationCard]:
        return next((card for card in self.cards if card.operation_id == identifier), None)


__all__ = ["OperationCard", "OperationList"]
