"""A vertical list of process steps with their state, for workspaces that guide a flow.

Each step shows a number (or a mark when done), a title and one line of what was found
("PILATUS 2M · 1 frame", "No geometry yet"); a longer line is cut with "…" and is whole in the step's
tooltip, so a step never pushes the next ones down. The state colours the mark: ``pending``,
``ok``, ``warn``, ``error`` or ``busy``. Clicking a step emits ``stepChosen(key)``; the
current step is highlighted.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
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
        self.setFocusPolicy(Qt.TabFocus)  # the keyboard reaches it; a click does not leave a focus frame
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
        self.detail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # a long line never widens the rail
        self.detail.setMinimumWidth(0)
        self.detail.hide()
        self.detail.installEventFilter(self)
        self.full_detail = ""
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

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.fit_detail()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        """The detail's own width can change after this row's resize (the layout runs later)."""
        if watched is self.detail and event.type() == QEvent.Resize and event.size().width() != event.oldSize().width():
            self.fit_detail()
        return False

    def fit_detail(self) -> None:
        """The detail on one line at the width it has: cut with "…" when longer (whole in the tooltip)."""
        detail = self.detail
        text = self.full_detail
        if detail.isHidden() or not text:
            return
        margins = detail.contentsMargins()
        room = max(1, detail.width() - margins.left() - margins.right() - 2 * detail.margin())
        shown = detail.fontMetrics().elidedText(text, Qt.ElideRight, room) if detail.width() > 1 else text
        if shown != detail.text():
            detail.setText(shown)

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
        step.full_detail = detail
        step.detail.setText(detail)
        step.detail.setVisible(bool(detail))
        step.setToolTip(detail)
        step.restyle()
        step.fit_detail()

    def state(self, key: str) -> str:
        step = self._steps.get(key)
        return str(step.property("state") or "pending") if step is not None else "pending"

    def detail(self, key: str) -> str:
        step = self._steps.get(key)
        return step.full_detail if step is not None else ""  # the whole line, also when it is cut on screen


__all__ = ["StepRail"]
