"""Static layout of the live view of the main screen: what a running Analyze is doing and has found so far (behaviour
in ``presentation/live_panel.py``).

At the top, one line across the view says what the analysis does now (``live_status.py``) and, under it when the
analysis package has stopped working, a note with **Stop Analysis** (hidden otherwise). Left, the four steps of the
analysis — each with its state and one line of what it found — and the time since the start with the processes at
work. Right, the step shown: its name, one paragraph on what it does, a note when reading
the frames from disk is the slow part (hidden otherwise), a picture with a line under it, and a plot.
Reading the frames: the detector summed (the pixels the mask leaves out marked) beside the frames read last on the
grid of q_y and exit angle; below, the count rate of every frame read so far. Finding the solutions: the measured
image of the window solved last beside the model image of its solution; below, the sizes found so far. Writing the
report and drawing the images show the solutions too. A click on a step that has started shows what it found.

A newer package reports its steps as a tree: the tree (``sputter_steps_view.py``) then takes the place of the four
steps, and the right half has a second page for what a step of the tree found. *Show the Preliminary Result* (top
right, hidden otherwise) goes back to the preliminary result when the user came here with **Steps**.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import ErrorBanner, StepRail

from .sputter_main_view import ElidedLabel
from .sputter_steps_view import SputterStepsView

LIVE_STEP_TITLES = (("intake", "Read the frames"), ("predict", "Find the solutions"), ("report", "Write the report"),
                    ("present", "Draw the images"))
"""(step of the package, its name) in their order."""
LIVE_TITLE = "Analysis at work"
LIVE_HINT = "Click a step that has started to see what it found."
PRELIMINARY_BUTTON = "Show the Preliminary Result"
STOP_BUTTON = "Stop Analysis"
STEPS_WIDTH = 360


def _stacked(parent: QWidget, count: int) -> tuple:
    """A stack of ``count`` pages, each with a layout without margins: (stack, layouts)."""
    stack = QStackedWidget(parent)
    layouts = []
    for _ in range(count):
        page = QWidget(stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        stack.addWidget(page)
        layouts.append(layout)
    return stack, layouts


class SputterLiveView(SputterStepsView):
    """The live view: a page of the main screen's stack (``_main_view`` adds it)."""

    def _live_view(self, parent: QWidget) -> QWidget:
        view = QWidget(parent)
        view.setObjectName("sputterLiveView")
        self.live_view = view
        outer = QVBoxLayout(view)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.live_status = ElidedLabel(view)  # what the analysis does now: the page sets it
        self.live_status.setObjectName("sputterLiveStatus")
        self.live_status.setProperty("gimapRole", "info")
        # while the preliminary result exists: back to it (the user came here with Steps)
        self.live_preliminary_button = QPushButton(PRELIMINARY_BUTTON, view)
        self.live_preliminary_button.setObjectName("sputterLivePreliminary")
        self.live_preliminary_button.setProperty("gimapRole", "link")
        self.live_preliminary_button.setCursor(Qt.PointingHandCursor)
        self.live_preliminary_button.setVisible(False)
        top.addWidget(self.live_status, 1)
        top.addWidget(self.live_preliminary_button)
        outer.addLayout(top)
        # shown only when the package has stopped working (the page sets its words; its button stops the analysis)
        self.live_stall_note = ErrorBanner("", "", view, level="warning", dismissible=False, show_details=True)
        self.live_stall_note.setObjectName("sputterLiveStallNote")
        self.live_stall_note.title_label.setVisible(False)
        self.live_stall_note.details_button.setText(STOP_BUTTON)
        self.live_stall_note.setVisible(False)
        outer.addWidget(self.live_stall_note)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        outer.addLayout(layout, 1)

        side = QWidget(view)
        side.setFixedWidth(STEPS_WIDTH)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(8)
        heading = QLabel(LIVE_TITLE, side)
        heading.setProperty("gimapRole", "heading")
        self.live_steps = StepRail(LIVE_STEP_TITLES, side)
        self.live_steps.setObjectName("sputterLiveSteps")
        self.live_steps.set_wrap(True)  # a line says how far a long step has come: it is shown whole
        self.live_clock = QLabel("", side)
        self.live_clock.setObjectName("sputterLiveClock")
        self.live_clock.setProperty("gimapRole", "muted")
        hint = QLabel(LIVE_HINT, side)
        hint.setWordWrap(True)
        hint.setProperty("gimapMeta", True)
        column.addWidget(heading)
        column.addWidget(self.live_steps)
        column.addWidget(self._live_tree_area(side), 1)  # a newer package's steps as a tree (sputter_steps_view.py)
        column.addWidget(self.live_clock)
        self.live_steps_stretch = QWidget(side)  # takes the room the tree takes when there is one
        column.addWidget(self.live_steps_stretch, 1)
        column.addWidget(hint)
        self.live_hint = hint
        layout.addWidget(side)

        self.live_right = QStackedWidget(view)  # 0: the step at work, 1: what a step of the tree found
        work = QWidget(self.live_right)
        right = QVBoxLayout(work)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        self.live_title = QLabel("", view)
        self.live_title.setObjectName("sputterLiveTitle")
        self.live_title.setProperty("gimapRole", "heading")
        self.live_explain = QLabel("", view)
        self.live_explain.setObjectName("sputterLiveExplain")
        self.live_explain.setWordWrap(True)
        self.live_explain.setProperty("gimapMeta", True)
        # shown only when reading the frames from disk is the slow part (the page sets its words)
        self.live_disk_note = ErrorBanner("", "", view, level="warning", dismissible=False)
        self.live_disk_note.setObjectName("sputterLiveDiskNote")
        self.live_disk_note.title_label.setVisible(False)
        self.live_disk_note.setVisible(False)
        self.live_picture_host, (self.live_frames_layout, self.live_patterns_layout) = _stacked(view, 2)
        self.live_caption = ElidedLabel(view)
        self.live_caption.setObjectName("sputterLiveCaption")
        self.live_caption.setProperty("gimapRole", "strong")
        self.live_plot_host, (self.live_trace_layout, self.live_evolution_layout) = _stacked(view, 2)
        right.addWidget(self.live_title)
        right.addWidget(self.live_explain)
        right.addWidget(self.live_disk_note)
        right.addWidget(self.live_picture_host, 5)
        right.addWidget(self.live_caption)
        right.addWidget(self.live_plot_host, 4)
        self.live_right.addWidget(work)
        self.live_right.addWidget(self._live_detail_page(self.live_right))
        layout.addWidget(self.live_right, 1)
        return view


__all__ = ["LIVE_HINT", "LIVE_STEP_TITLES", "LIVE_TITLE", "PRELIMINARY_BUTTON", "STOP_BUTTON", "SputterLiveView"]
