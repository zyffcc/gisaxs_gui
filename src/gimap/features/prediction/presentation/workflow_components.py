"""Small, feature-owned presentation components for Prediction."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import CARD_SPACING

from .workflow_state import PredictionWorkflowSnapshot


class PredictionDisclosure(QWidget):
    """Progressive disclosure for secondary Prediction controls."""

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.toggle = QToolButton(self)
        self.toggle.setObjectName(f"{object_name}Toggle")
        self.toggle.setProperty("predictionDisclosure", True)
        self.toggle.setCheckable(True)
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.content = QWidget(self)
        self.content.setObjectName(f"{object_name}Content")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 2, 4, 4)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(bool(expanded))
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(bool(expanded))

    def add_widget(self, widget: QWidget) -> None:
        widget.setParent(self.content)
        self.content_layout.addWidget(widget)


class PredictionWorkflowStep(QFrame):
    """Keyboard-focusable navigation target for one workflow step."""

    requested = pyqtSignal(int)

    def __init__(self, number: int, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.number = number
        self.setObjectName(f"predictionWorkflowStep{number}")
        self.setProperty("predictionWorkflowStep", True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        badge = QLabel(str(number), self)
        badge.setProperty("predictionWorkflowBadge", True)
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(22, 22)
        label = QLabel(text, self)
        label.setProperty("predictionWorkflowLabel", True)
        label.setWordWrap(True)
        layout.addWidget(badge)
        layout.addWidget(label, 1)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.requested.emit(self.number)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.requested.emit(self.number)
            event.accept()
            return
        super().keyPressEvent(event)


class PredictionWorkflowHeader(QFrame):
    """Three-step summary of the basic Prediction workflow."""

    step_requested = pyqtSignal(int)

    STEP_TITLES = ("Import data", "Import model", "Predict")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("predictionWorkflowHeader")
        self.setProperty("predictionWorkflowHeader", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title = QLabel("Prediction workbench", self)
        title.setProperty("predictionWorkflowTitle", True)
        subtitle = QLabel(
            "Import detector data, load a compatible model, then run prediction.", self
        )
        subtitle.setProperty("predictionWorkflowSubtitle", True)
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.steps = []
        for number, text in enumerate(self.STEP_TITLES, start=1):
            step = PredictionWorkflowStep(number, text, self)
            step.requested.connect(self.step_requested)
            row.addWidget(step, 1)
            self.steps.append(step)
        layout.addLayout(row)
        self.render(PredictionWorkflowSnapshot())

    def bind(self, ui) -> None:
        """Expose the header for binding-driven state rendering."""
        ui.predictionWorkflowHeader = self

    def set_active_step(self, active_step: int) -> None:
        """Compatibility helper for tests and non-runtime previews."""
        active_step = max(1, min(len(self.steps), int(active_step)))
        self._apply_states(
            tuple(
                "complete" if number < active_step else "active" if number == active_step else "upcoming"
                for number in range(1, len(self.steps) + 1)
            )
        )

    def render(self, snapshot: PredictionWorkflowSnapshot) -> None:
        self._apply_states(snapshot.step_states())

    def _apply_states(self, states) -> None:
        for step, state in zip(self.steps, states):
            step.setProperty("workflowState", state)
            step.setAccessibleDescription(f"Workflow step: {state}")
            step.style().unpolish(step)
            step.style().polish(step)


class PredictionInputModePanel(QWidget):
    """Show only the controls relevant to single-file or folder prediction."""

    mode_changed = pyqtSignal(str)

    def __init__(self, ui, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = ui
        self.setObjectName("predictionInputModePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_SPACING)

        selector = QFrame(self)
        selector.setObjectName("predictionModeSelector")
        selector_layout = QHBoxLayout(selector)
        selector_layout.setContentsMargins(4, 4, 4, 4)
        selector_layout.setSpacing(4)
        ui.gisaxsPredictSingleFileRadioButton.setText("Single file")
        ui.gisaxsPredictMultiFilesRadioButton.setText("Folder batch")
        selector_layout.addWidget(ui.gisaxsPredictSingleFileRadioButton, 1)
        selector_layout.addWidget(ui.gisaxsPredictMultiFilesRadioButton, 1)
        layout.addWidget(selector)

        self.pages = QStackedWidget(self)
        self.pages.setObjectName("predictionInputModePages")
        single_page = self._make_picker_page(
            "Choose detector file",
            ui.gisaxsPredictChooseGisaxsFileButton,
            ui.gisaxsPredictChooseGisaxsFileValue,
        )
        batch_page = self._make_picker_page(
            "Choose data folder",
            ui.gisaxsPredictChooseFolderButton,
            ui.gisaxsPredictChooseFolderValue,
        )
        self.pages.addWidget(single_page)
        self.pages.addWidget(batch_page)
        layout.addWidget(self.pages)

        range_panel = QFrame(self)
        range_panel.setObjectName("gisaxsPredictRangePanel")
        range_layout = QGridLayout(range_panel)
        range_layout.setContentsMargins(10, 8, 10, 8)
        range_layout.setHorizontalSpacing(8)
        range_layout.setVerticalSpacing(6)
        range_layout.addWidget(ui.gisaxsPredictStackLabel, 0, 0)
        range_layout.addWidget(ui.gisaxsPredictStackValue, 0, 1)
        range_layout.addWidget(ui.gisaxsPredictEveryLabel, 1, 0)
        range_layout.addWidget(ui.gisaxsPredictEveryValue, 1, 1)
        range_layout.setColumnStretch(1, 1)
        layout.addWidget(range_panel)

        self.hint = QLabel(self)
        self.hint.setObjectName("gisaxsPredictRangeHintLabel")
        self.hint.setProperty("cardMeta", True)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.summary = QLabel("Choose a folder to calculate the batch plan.", self)
        self.summary.setObjectName("predictionBatchPlanSummary")
        self.summary.setProperty("predictionBatchSummary", True)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(ui.gisaxsPredictShowMultiFileResultsButton)
        ui.predictionBatchPlanSummary = self.summary
        ui.gisaxsPredictSingleFileRadioButton.toggled.connect(self.sync_mode)
        ui.gisaxsPredictMultiFilesRadioButton.toggled.connect(self.sync_mode)
        self.sync_mode()

    def _make_picker_page(self, title: str, button, value) -> QWidget:
        page = QWidget(self.pages)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(6)
        label = QLabel(title, page)
        label.setProperty("predictionFieldLabel", True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(button)
        row.addWidget(value, 1)
        page_layout.addWidget(label)
        page_layout.addLayout(row)
        return page

    def sync_mode(self) -> None:
        is_batch = self.ui.gisaxsPredictMultiFilesRadioButton.isChecked()
        self.pages.setCurrentIndex(1 if is_batch else 0)
        self.hint.setText(
            "Use an inclusive file-number range and choose how many files form one prediction."
            if is_batch
            else "Stack controls how many consecutive detector files contribute to this prediction."
        )
        self.summary.setVisible(is_batch)
        self.ui.gisaxsPredictShowMultiFileResultsButton.setVisible(is_batch)
        self.mode_changed.emit("multi_files" if is_batch else "single_file")

    def set_batch_summary(self, *, files: int, jobs: int, skipped: int = 0) -> None:
        if files <= 0:
            self.summary.setText("No detector files selected by the current folder and range.")
            return
        message = f"{files} files selected · {jobs} prediction job{'s' if jobs != 1 else ''}"
        if skipped:
            message += f" · {skipped} trailing file{'s' if skipped != 1 else ''} skipped"
        self.summary.setText(message)


class PredictionCanvasEmptyState(QLabel):
    """Overlay a clear next action on an empty Prediction graphics view."""

    def __init__(self, view: QGraphicsView, text: str) -> None:
        super().__init__(text, view.viewport())
        self.view = view
        self.setObjectName(f"{view.objectName()}EmptyState")
        self.setProperty("predictionEmptyState", True)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        view.viewport().installEventFilter(self)
        scene = view.scene()
        if scene is not None:
            scene.changed.connect(lambda _regions: self.refresh())
        self.refresh()

    def eventFilter(self, watched, event):
        if watched is self.view.viewport() and event.type() in (
            QEvent.Resize,
            QEvent.Show,
            QEvent.Paint,
        ):
            self.refresh()
        return False

    def refresh(self) -> None:
        scene = self.view.scene()
        self.setVisible(scene is None or not scene.items())
        self.setGeometry(self.view.viewport().rect().adjusted(24, 24, -24, -24))
        self.raise_()


__all__ = [
    "PredictionCanvasEmptyState",
    "PredictionDisclosure",
    "PredictionInputModePanel",
    "PredictionWorkflowHeader",
    "PredictionWorkflowStep",
]
