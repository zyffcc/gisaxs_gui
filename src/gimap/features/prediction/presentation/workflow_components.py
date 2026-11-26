"""Small, feature-owned presentation components for Prediction."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
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

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.layout_primitives import CARD_SPACING

from .workflow_state import PredictionWorkflowSnapshot

CBF_FILE_TIP = (
    "A detector frame (CBF, EDF or TIFF), or the first of a numbered series, to predict from: "
    "use frames like those the module was trained on."
)
CBF_FOLDER_TIP = (
    "A folder of numbered detector frames (CBF, EDF or TIFF; the format most of them have): those in the range "
    "below are used."
)
STACK_TIP = "How many consecutive detector files contribute to this prediction."
RANGE_TIP = "Use an inclusive file-number range and choose how many files form one prediction."
WATCH_TEXT = "Watch for new frames"
EMPTY_WATCHED = "No frames in the folder yet: with Watch for new frames, the batch waits for them."
WATCHING_SUMMARY = "Watching for new frames · {predicted} predicted"
WATCHING_SUMMARY_LEFT_OUT = "Watching for new frames · {predicted} predicted · {left_out} left out to keep up"
KEEP_UP_TEXT = "Keep up with the detector"
WATCH_TIP = ("After the frames of the folder, go on with every frame written into it while the batch runs (a frame is "
             "taken once it is completely written). Stop ends the batch; unticking ends it once the frames taken are "
             "predicted.")
KEEP_UP_TIP = ("When frames arrive faster than the model predicts them, at most two frames (or stacks) wait for the "
               "model and older ones are left out, so the prediction stays a few frames behind the newest at most. Left-out "
               "frames are counted, stay in the folder and can be predicted later with a folder batch. Unticked, every "
               "frame is predicted in turn, however far behind.")


def range_field_tip(is_batch: bool, indices=()) -> str:
    """Tooltip of Stack (single file) or Range (folder batch): what the field means, then the file numbers found."""
    if indices:
        found = trf("Available index range: {first} - {last}", first=indices[0], last=indices[-1])
    else:
        found = tr("No valid indices detected yet")
    return tr(RANGE_TIP if is_batch else STACK_TIP) + "\n" + found


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
        self._title = title
        self.toggle.setText(title)
        self.refresh_language()
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

    def refresh_language(self) -> None:
        """The title (an exact key) and the tooltip composed from it, in the interface language."""
        title = tr(self._title)
        self.toggle.setText(title)
        self.toggle.setToolTip(trf("Show or hide: {title}", title=title))

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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        title = QLabel("Prediction workbench", self)
        title.setProperty("predictionWorkflowTitle", True)
        # The three steps below say the same; the sentence stays as the tooltip so the rail
        # keeps room for step 2 above the sticky Predict card.
        title.setToolTip("Import detector data, load a compatible model, then run prediction.")
        layout.addWidget(title)

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
        # Prediction reads CBF, EDF and TIFF frames: the tooltip says so (a dropped file of another kind is
        # refused with a message).
        ui.gisaxsPredictChooseGisaxsFileButton.setText("Choose frame…")
        ui.gisaxsPredictChooseGisaxsFileButton.setToolTip(CBF_FILE_TIP)
        ui.gisaxsPredictChooseFolderButton.setToolTip(CBF_FOLDER_TIP)
        single_page = self._make_picker_page(
            ui.gisaxsPredictChooseGisaxsFileButton,
            ui.gisaxsPredictChooseGisaxsFileValue,
        )
        batch_page = self._make_picker_page(
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
        for label in (ui.gisaxsPredictStackLabel, ui.gisaxsPredictEveryLabel):
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # a form: label, then its field
        range_layout.addWidget(ui.gisaxsPredictStackLabel, 0, 0)
        range_layout.addWidget(ui.gisaxsPredictStackValue, 0, 1, Qt.AlignLeft)
        range_layout.addWidget(ui.gisaxsPredictEveryLabel, 1, 0)
        range_layout.addWidget(ui.gisaxsPredictEveryValue, 1, 1, Qt.AlignLeft)
        # A folder batch can go on with the frames written into its folder (bindings/frame_watch.py).
        ui.gisaxsPredictWatchCheck = QCheckBox(WATCH_TEXT, range_panel)
        ui.gisaxsPredictWatchCheck.setObjectName("gisaxsPredictWatchCheck")
        ui.gisaxsPredictWatchCheck.setToolTip(WATCH_TIP)
        ui.gisaxsPredictKeepUpCheck = QCheckBox(KEEP_UP_TEXT, range_panel)
        ui.gisaxsPredictKeepUpCheck.setObjectName("gisaxsPredictKeepUpCheck")
        ui.gisaxsPredictKeepUpCheck.setToolTip(KEEP_UP_TIP)
        ui.gisaxsPredictKeepUpCheck.setChecked(True)
        ui.gisaxsPredictKeepUpCheck.setEnabled(False)  # with Watch only
        watch_row = QHBoxLayout()
        watch_row.setContentsMargins(0, 2, 0, 0)
        watch_row.setSpacing(16)
        watch_row.addWidget(ui.gisaxsPredictWatchCheck)
        watch_row.addWidget(ui.gisaxsPredictKeepUpCheck)
        watch_row.addStretch(1)
        range_layout.addLayout(watch_row, 2, 0, 1, 3)
        # The free width goes to an empty third column: each field stays right beside its label.
        range_layout.setColumnStretch(2, 1)
        layout.addWidget(range_panel)
        # What Stack / Range means is the tooltip of the field (``range_field_tip``, set by the binding).
        self._summary_values = None  # the last batch plan shown (set_batch_summary)
        self.summary = QLabel("Choose a folder to calculate the batch plan.", self)
        self.summary.setObjectName("predictionBatchPlanSummary")
        self.summary.setProperty("predictionBatchSummary", True)
        self.summary.setWordWrap(True)
        # The plan and the button to its results share one row (keeps step 2 in view).
        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(8)
        summary_row.addWidget(self.summary, 1)
        summary_row.addWidget(ui.gisaxsPredictShowMultiFileResultsButton, 0, Qt.AlignVCenter)
        layout.addLayout(summary_row)
        ui.predictionBatchPlanSummary = self.summary
        ui.gisaxsPredictSingleFileRadioButton.toggled.connect(self.sync_mode)
        ui.gisaxsPredictMultiFilesRadioButton.toggled.connect(self.sync_mode)
        self.sync_mode()

    def _make_picker_page(self, button, value) -> QWidget:
        """The chooser button and its field (the button names what to choose)."""
        page = QWidget(self.pages)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(button)
        row.addWidget(value, 1)
        page_layout.addLayout(row)
        return page

    def sync_mode(self) -> None:
        is_batch = self.ui.gisaxsPredictMultiFilesRadioButton.isChecked()
        self.pages.setCurrentIndex(1 if is_batch else 0)
        self.summary.setVisible(is_batch)
        self.ui.gisaxsPredictShowMultiFileResultsButton.setVisible(is_batch)
        for name in ("gisaxsPredictWatchCheck", "gisaxsPredictKeepUpCheck"):  # a folder batch only
            box = getattr(self.ui, name, None)
            if box is not None:
                box.setVisible(is_batch)
        self.mode_changed.emit("multi_files" if is_batch else "single_file")

    def refresh_language(self) -> None:
        """The batch plan in the interface language (after a switch)."""
        if getattr(self, "_watch_values", None) is not None:
            self.set_watch_summary(**self._watch_values)
        elif getattr(self, "_summary_values", None) is not None:
            self.set_batch_summary(**self._summary_values)

    def set_batch_summary(self, *, files: int, jobs: int, skipped: int = 0, watch: bool = False) -> None:
        self._summary_values = dict(files=files, jobs=jobs, skipped=skipped, watch=watch)
        self._watch_values = None
        if files <= 0:
            self.summary.setText(tr(EMPTY_WATCHED if watch else "No detector files selected by the current folder and range."))
            return
        if jobs == 1:
            message = trf("{files} files selected · {jobs} prediction job", files=files, jobs=jobs)
        else:
            message = trf("{files} files selected · {jobs} prediction jobs", files=files, jobs=jobs)
        if skipped == 1:
            message = trf("{summary} · {skipped} trailing file skipped", summary=message, skipped=skipped)
        elif skipped:
            message = trf("{summary} · {skipped} trailing files skipped", summary=message, skipped=skipped)
        self.summary.setText(message)

    def set_watch_summary(self, *, predicted: int, left_out: int = 0) -> None:
        """While a batch watches its folder: how many new predictions, and how many frames were left out."""
        self._watch_values = dict(predicted=predicted, left_out=left_out)
        if left_out:
            self.summary.setText(trf(WATCHING_SUMMARY_LEFT_OUT, predicted=predicted, left_out=left_out))
        else:
            self.summary.setText(trf(WATCHING_SUMMARY, predicted=predicted))


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
    "CBF_FILE_TIP",
    "CBF_FOLDER_TIP",
    "RANGE_TIP",
    "STACK_TIP",
    "range_field_tip",
    "PredictionCanvasEmptyState",
    "PredictionDisclosure",
    "PredictionInputModePanel",
    "PredictionWorkflowHeader",
    "PredictionWorkflowStep",
]
