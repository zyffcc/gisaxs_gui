"""A vertical list of process steps with their state, for workspaces that guide a flow.

Each step shows a number (or a mark when done), a title and one line of what was found
("PILATUS 2M · 1 frame", "No geometry yet"). The state colours the mark: ``pending``,
``ok``, ``warn``, ``error`` or ``busy``. Clicking a step emits ``stepChosen(key)``; the
current step is highlighted.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

STATES = ("pending", "ok", "warn", "error", "busy")
MARKS = {"ok": "✓", "warn": "!", "error": "×", "busy": "…"}


class _Step(QWidget):
    clicked = pyqtSignal()

    def __init__(self, number: int, title: str, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("gimapStep", True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.number = number
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(10)
        self.mark = QLabel(str(number), self)
        self.mark.setProperty("gimapStepMark", True)
        self.mark.setAlignment(Qt.AlignCenter)
        self.mark.setFixedSize(22, 22)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        self.title = QLabel(title, self)
        self.title.setProperty("gimapStepTitle", True)
        self.detail = QLabel("", self)
        self.detail.setProperty("gimapStepDetail", True)
        self.detail.setWordWrap(True)
        self.detail.hide()
        text.addWidget(self.title)
        text.addWidget(self.detail)
        layout.addWidget(self.mark, 0, Qt.AlignTop)
        layout.addLayout(text, 1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)

    def restyle(self) -> None:
        for widget in (self, self.mark, self.title, self.detail):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class StepRail(QWidget):
    stepChosen = pyqtSignal(str)

    def __init__(self, steps, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("gimapStepRail")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._steps: dict[str, _Step] = {}
        self._current: Optional[str] = None
        for number, (key, title) in enumerate(steps, start=1):
            step = _Step(number, title, self)
            step.setObjectName(f"gimapStep_{key}")
            step.clicked.connect(lambda key=key: self._choose(key))
            layout.addWidget(step)
            self._steps[key] = step
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def keys(self) -> list[str]:
        return list(self._steps)

    def current(self) -> Optional[str]:
        return self._current

    def _choose(self, key: str) -> None:
        self.set_current(key)
        self.stepChosen.emit(key)

    def set_current(self, key: str) -> None:
        if key not in self._steps:
            return
        self._current = key
        for name, step in self._steps.items():
            step.setProperty("current", name == key)
            step.restyle()

    def set_state(self, key: str, state: str, detail: str = "") -> None:
        step = self._steps.get(key)
        if step is None:
            return
        state = state if state in STATES else "pending"
        step.setProperty("state", state)
        step.mark.setProperty("state", state)
        step.mark.setText(MARKS.get(state, str(step.number)))
        step.detail.setText(detail)
        step.detail.setVisible(bool(detail))
        step.setToolTip(detail)
        step.restyle()

    def state(self, key: str) -> str:
        step = self._steps.get(key)
        return str(step.property("state") or "pending") if step is not None else "pending"

    def detail(self, key: str) -> str:
        step = self._steps.get(key)
        return step.detail.text() if step is not None else ""


__all__ = ["StepRail"]
