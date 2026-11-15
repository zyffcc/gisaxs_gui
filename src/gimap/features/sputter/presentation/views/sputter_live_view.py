"""Static layout of the live view of the main screen: what a running Analyze is doing and has found so far (behaviour
in ``presentation/live_panel.py``).

Left, the four steps of the analysis — each with its state and one line of what it found — and the time since the
start. Right, the step shown: its name, one paragraph on what it does, a picture with a line under it, and a plot.
Reading the frames: the detector summed (the pixels the mask leaves out marked) beside the frames read last on the
grid of q_y and exit angle; below, the count rate of every frame read so far. Finding the solutions: the measured
image of the window solved last beside the model image of its solution; below, the sizes found so far. Writing the
report and drawing the images show the solutions too. A click on a step that has started shows what it found.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import StepRail

from .sputter_main_view import ElidedLabel

LIVE_STEP_TITLES = (("intake", "Read the frames"), ("predict", "Find the solutions"), ("report", "Write the report"),
                    ("present", "Draw the images"))
"""(step of the package, its name) in their order."""
LIVE_TITLE = "Analysis at work"
LIVE_HINT = "Click a step that has started to see what it found."
STEPS_WIDTH = 300


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


class SputterLiveView:
    """The live view: a page of the main screen's stack (``_main_view`` adds it)."""

    def _live_view(self, parent: QWidget) -> QWidget:
        view = QWidget(parent)
        view.setObjectName("sputterLiveView")
        self.live_view = view
        layout = QHBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        side = QWidget(view)
        side.setFixedWidth(STEPS_WIDTH)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(8)
        heading = QLabel(LIVE_TITLE, side)
        heading.setProperty("gimapRole", "heading")
        self.live_steps = StepRail(LIVE_STEP_TITLES, side)
        self.live_steps.setObjectName("sputterLiveSteps")
        self.live_clock = QLabel("", side)
        self.live_clock.setObjectName("sputterLiveClock")
        self.live_clock.setProperty("gimapRole", "muted")
        hint = QLabel(LIVE_HINT, side)
        hint.setWordWrap(True)
        hint.setProperty("gimapMeta", True)
        column.addWidget(heading)
        column.addWidget(self.live_steps)
        column.addWidget(self.live_clock)
        column.addStretch(1)
        column.addWidget(hint)
        layout.addWidget(side)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.live_title = QLabel("", view)
        self.live_title.setObjectName("sputterLiveTitle")
        self.live_title.setProperty("gimapRole", "heading")
        self.live_explain = QLabel("", view)
        self.live_explain.setObjectName("sputterLiveExplain")
        self.live_explain.setWordWrap(True)
        self.live_explain.setProperty("gimapMeta", True)
        self.live_picture_host, (self.live_frames_layout, self.live_patterns_layout) = _stacked(view, 2)
        self.live_caption = ElidedLabel(view)
        self.live_caption.setObjectName("sputterLiveCaption")
        self.live_caption.setProperty("gimapRole", "strong")
        self.live_plot_host, (self.live_trace_layout, self.live_evolution_layout) = _stacked(view, 2)
        right.addWidget(self.live_title)
        right.addWidget(self.live_explain)
        right.addWidget(self.live_picture_host, 5)
        right.addWidget(self.live_caption)
        right.addWidget(self.live_plot_host, 4)
        layout.addLayout(right, 1)
        return view


__all__ = ["LIVE_HINT", "LIVE_STEP_TITLES", "LIVE_TITLE", "SputterLiveView"]
