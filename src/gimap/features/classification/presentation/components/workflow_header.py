"""State-driven workflow navigation for the Classification workbench."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ClassificationWorkflowStep(QFrame):
    """One navigable step whose completion state is supplied by the binding."""

    requested = pyqtSignal(str)

    def __init__(
        self,
        number: int,
        key: str,
        title: str,
        button_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName(f"classificationWorkflowStep{number}")
        self.setProperty("classificationWorkflowStep", True)
        self.setProperty("workflowState", "blocked")
        self.setProperty("workflowSelected", False)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(7)
        self.badge = QLabel(str(number), self)
        self.badge.setProperty("workflowBadge", True)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedSize(22, 22)
        layout.addWidget(self.badge)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)
        self.button = QToolButton(self)
        self.button.setObjectName(button_name)
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setAutoRaise(True)
        self.button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.button.setProperty("workflowLabel", True)
        self.button.clicked.connect(lambda _checked=False: self.requested.emit(self.key))
        self.message_label = QLabel("", self)
        self.message_label.setProperty("workflowMessage", True)
        self.message_label.setWordWrap(True)
        text_layout.addWidget(self.button)
        text_layout.addWidget(self.message_label)
        layout.addLayout(text_layout, 1)

    def set_state(self, state: str, message: str = "") -> None:
        self.setProperty("workflowState", state)
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        self.setToolTip(message or self.button.text())
        self._refresh_style()

    def set_selected(self, selected: bool) -> None:
        self.setProperty("workflowSelected", bool(selected))
        self.button.setChecked(bool(selected))
        self._refresh_style()

    def set_guidance_visible(self, visible: bool) -> None:
        self.message_label.setVisible(bool(visible) and bool(self.message_label.text()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.requested.emit(self.key)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.requested.emit(self.key)
            return
        super().keyPressEvent(event)

    def _refresh_style(self) -> None:
        for widget in (self, self.badge, self.button, self.message_label):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class ClassificationWorkflowHeader(QFrame):
    """Keep navigation position independent from verified workflow progress."""

    step_requested = pyqtSignal(str)
    guided_changed = pyqtSignal(bool)
    STEPS = (
        ("Data", "Data", "datasetStepButton"),
        ("Prepare", "Prepare", "preprocessingStepButton"),
        ("Explore", "Explore & label", "algorithmsStepButton"),
        ("Train", "Train & review", "resultsStepButton"),
        ("Apply", "Apply & export", "applyStepButton"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("classificationWorkflowHeader")
        self.setProperty("classificationWorkflowHeader", True)
        self._guided = True
        self._compact = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)
        title_row = QHBoxLayout()
        self.title_label = QLabel("Classification workbench", self)
        self.title_label.setProperty("workflowTitle", True)
        self.mode_button = QToolButton(self)
        self.mode_button.setObjectName("classificationWorkflowGuidedButton")
        self.mode_button.setText("Guided")
        self.mode_button.setCheckable(True)
        self.mode_button.setChecked(True)
        self.mode_button.setToolTip("Show or hide guidance while keeping workflow shortcuts")
        self.mode_button.toggled.connect(self._on_guided_changed)
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        title_row.addWidget(self.mode_button)
        layout.addLayout(title_row)

        self.subtitle_label = QLabel(
            "Progress reflects completed data, reduction, labels, and model results—not click history.",
            self,
        )
        self.subtitle_label.setProperty("workflowSubtitle", True)
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)

        self.steps = [
            ClassificationWorkflowStep(number, key, title, button_name, self)
            for number, (key, title, button_name) in enumerate(self.STEPS, start=1)
        ]
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for step in self.steps:
            step.requested.connect(self.step_requested.emit)
            row.addWidget(step, 1)
        layout.addLayout(row)

    @property
    def buttons(self) -> dict[str, QToolButton]:
        return {step.key: step.button for step in self.steps}

    def set_step_state(self, key: str, state: str, message: str = "") -> None:
        step = self._step(key)
        if step is not None:
            step.set_state(state, message)
            step.set_guidance_visible(self._guided and not self._compact)

    def set_selected_step(self, key: str) -> None:
        for step in self.steps:
            step.set_selected(step.key == key)

    def set_compact(self, compact: bool) -> None:
        self._compact = bool(compact)
        self._apply_guidance_visibility()

    def _on_guided_changed(self, guided: bool) -> None:
        self._guided = bool(guided)
        self.mode_button.setText("Guided" if guided else "Compact")
        self._apply_guidance_visibility()
        self.guided_changed.emit(bool(guided))

    def _apply_guidance_visibility(self) -> None:
        visible = self._guided and not self._compact
        self.subtitle_label.setVisible(visible)
        for step in self.steps:
            step.set_guidance_visible(visible)

    def _step(self, key: str) -> ClassificationWorkflowStep | None:
        return next((step for step in self.steps if step.key == key), None)


__all__ = ["ClassificationWorkflowHeader", "ClassificationWorkflowStep"]
