"""Sticky, state-driven workflow navigation for the Fitting workspace."""

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

from .workflow_state import (
    FittingWorkflowState,
    initial_workflow_state,
)


class WorkflowStep(QFrame):
    """Clickable navigation step whose state is supplied by the ViewModel."""

    requested = pyqtSignal(str)

    def __init__(self, number: int, key: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName(f"fittingWorkflowStep{number}")
        self.setProperty("fittingWorkflowStep", True)
        self.setProperty("workflowState", "blocked")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.badge = QLabel(str(number), self)
        self.badge.setObjectName(f"fittingWorkflowStep{number}Badge")
        self.badge.setProperty("workflowBadge", True)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedSize(22, 22)
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"fittingWorkflowStep{number}Label")
        self.title_label.setProperty("workflowLabel", True)
        self.message_label = QLabel("", self)
        self.message_label.setObjectName(f"fittingWorkflowStep{number}Message")
        self.message_label.setProperty("workflowMessage", True)
        self.message_label.setWordWrap(True)
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.message_label)
        layout.addWidget(self.badge)
        layout.addLayout(text_layout, 1)

    def set_state(self, state: str, message: str = "") -> None:
        self.setProperty("workflowState", state)
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message) and self.property("guided") is not False)
        # A workflow step is navigation, not an execution gate.  Users may
        # inspect any step while the command inside it remains readiness-gated.
        self.setEnabled(True)
        self.setToolTip(message or self.title_label.text())
        self._refresh_style()

    def set_selected(self, selected: bool) -> None:
        self.setProperty("workflowSelected", bool(selected))
        self._refresh_style()

    def set_guided(self, guided: bool) -> None:
        self.setProperty("guided", bool(guided))
        self.message_label.setVisible(bool(guided) and bool(self.message_label.text()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.isEnabled() and event.button() == Qt.LeftButton:
            self.requested.emit(self.key)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.isEnabled() and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.requested.emit(self.key)
            return
        super().keyPressEvent(event)

    def _refresh_style(self) -> None:
        for widget in (self, *self.findChildren(QLabel)):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class FittingWorkflowHeader(QFrame):
    """Render verified workflow state and provide shortcuts to each section."""

    step_requested = pyqtSignal(str)
    guided_changed = pyqtSignal(bool)
    DISPLAY_STEPS = (
        ("import", "Import data"),
        ("setup", "Experiment setup"),
        ("center_cut", "Yoneda & cut"),
        ("fit", "Fit"),
    )
    STEP_TITLES = tuple(title for _key, title in DISPLAY_STEPS)
    SHORT_TITLES = {
        "import": "Import",
        "setup": "Setup",
        "center_cut": "Yoneda & Cut",
        "fit": "Fit",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fittingWorkflowHeader")
        self.setProperty("fittingWorkflowHeader", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)
        title_row = QHBoxLayout()
        title = QLabel("Fitting workbench", self)
        title.setObjectName("fittingWorkflowTitle")
        title.setProperty("workflowTitle", True)
        self.mode_button = QToolButton(self)
        self.mode_button.setObjectName("fittingWorkflowGuidedButton")
        self.mode_button.setText("Guided")
        self.mode_button.setCheckable(True)
        self.mode_button.setChecked(True)
        self.mode_button.setToolTip("Show or hide guidance while keeping workflow shortcuts")
        self.mode_button.toggled.connect(self._on_guided_changed)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.mode_button)
        layout.addLayout(title_row)
        self.subtitle = QLabel(
            "Verified progress: a step completes only after its operation succeeds.", self
        )
        self.subtitle.setObjectName("fittingWorkflowSubtitle")
        self.subtitle.setProperty("workflowSubtitle", True)
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        self.steps = [
            WorkflowStep(number, key, self.SHORT_TITLES[key], self)
            for number, (key, _title) in enumerate(self.DISPLAY_STEPS, start=1)
        ]
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for step in self.steps:
            step.requested.connect(self.step_requested.emit)
            row.addWidget(step, 1)
        layout.addLayout(row)
        self.render(initial_workflow_state())

    def render(self, workflow: FittingWorkflowState) -> None:
        by_key = {step.key: step for step in workflow.steps}
        for widget in self.steps:
            if widget.key == "center_cut":
                status, message = self._combined_center_cut_state(
                    by_key["center"], by_key["cut"]
                )
                widget.set_state(status, message)
            else:
                state = by_key[widget.key]
                widget.set_state(state.status, state.message)

    def set_selected_step(self, key: str) -> None:
        selected_key = "center_cut" if key in {"center", "cut", "center_cut"} else key
        for widget in self.steps:
            widget.set_selected(widget.key == selected_key)

    @staticmethod
    def _combined_center_cut_state(center, cut) -> tuple[str, str]:
        """Present two verified scientific states as one navigation shortcut."""
        if "error" in {center.status, cut.status}:
            status = "error"
        elif "running" in {center.status, cut.status}:
            status = "running"
        elif cut.status == "complete":
            status = "complete"
        elif "stale" in {center.status, cut.status}:
            status = "stale"
        elif "available" in {center.status, cut.status} or center.status == "complete":
            status = "available"
        else:
            status = "blocked"
        return status, cut.message or center.message

    def set_guided(self, guided: bool) -> None:
        self.mode_button.blockSignals(True)
        self.mode_button.setChecked(bool(guided))
        self.mode_button.blockSignals(False)
        self._apply_guided(bool(guided))

    def _on_guided_changed(self, guided: bool) -> None:
        self._apply_guided(guided)
        self.guided_changed.emit(guided)

    def _apply_guided(self, guided: bool) -> None:
        self.mode_button.setText("Guided" if guided else "Compact")
        self.subtitle.setVisible(guided)
        for step in self.steps:
            step.set_guided(guided)


__all__ = ["FittingWorkflowHeader", "WorkflowStep"]
