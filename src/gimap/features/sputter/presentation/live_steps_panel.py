"""The step tree of a running Analyze and of a result (a mixin of ``SputterPage``, through ``LivePanelMixin``).

A newer analysis package reports its steps as a tree (``application/live_steps.py``). While it runs, the tree takes
the place of the four step lines at the left of the live view: ○ pending, ● running — with the line of what happens
now under the step at work (``live_status.py``; under the joint fit: its iteration, the windows of the batch at work,
the time so far and left) —, ✓ done with what it found at the right, ✗ failed, – skipped; a parent folds its children
away once it is done. A click on a step marked › opens what it found in the right half of the live view
(``step_pictures.py``: the beam centre and the mask on the detector, the frames read, the start of the deposition, the
time windows, the networks compared, the joint fit's iterations and solution — the joint fit's picture follows every
iteration while it is open); *Back to the Analysis* closes it. The details are read from the result's ``trace``
folder (during the import from the import's own, ``ClientStarted.import_trace``) when the step is opened and dropped
when it is closed.

While the main screen shows the preliminary result, **Steps** beside its note line opens the live view again, and
*Show the Preliminary Result* goes back. When the run ends with a result, the tree is kept with it
(``trace/gimap_steps.json``); **Details ▸ Steps** shows the tree of the result shown, and what a step found at its
right — a result without a ``trace`` folder (an older package) says so in one line. An older package that reports no
steps keeps the four step lines.
"""

from __future__ import annotations

from typing import Optional

from src.gimap.app.presentation.i18n import tr

from ..application import StepTree
from ..application.live_steps import TRACE_FOLDER
from .step_pictures import StepDetailPicture
from .step_texts import BACK_TO_RUN, CLICK_STEP, NO_DETAILS, NO_STEPS, node_summary, step_title
from .texts import message_text
from .views.sputter_live_view import LIVE_HINT, PRELIMINARY_BUTTON
from .views.sputter_main_view import STEPS_BUTTON

LIVE_PAGE = 2


class LiveStepsMixin:
    """Needs ``_live``, ``_live_folder``, the widgets of ``views/sputter_steps_view.py``, ``result_stack``,
    ``service`` and ``client_result``."""

    def _init_live_steps(self) -> None:
        self._live_trace: tuple = ()
        self._live_detail = ""
        self._result_steps: Optional[StepTree] = None
        self.live_detail_picture = StepDetailPicture(self.live_detail_layout)
        self.steps_picture = StepDetailPicture(self.steps_detail_layout)
        self.live_tree.stepChosen.connect(self._live_tree_chosen)
        self.live_detail_back.clicked.connect(self._live_close_detail)
        self.steps_tree.stepChosen.connect(self._result_step_chosen)
        self.steps_button.clicked.connect(self._show_live_steps)
        self.live_preliminary_button.clicked.connect(self._back_to_preliminary)

    # -- the run ----------------------------------------------------------------------------------------------
    def _live_steps_begin(self, trace_folders=()) -> None:
        self._live_trace = tuple(trace_folders)
        self.live_tree.clear()
        self._live_close_detail()
        self._live_show_tree()

    def _live_steps_end(self, folder=None) -> None:
        """The run ended; with a result (``folder``) the tree it was told is kept with it."""
        run = self._live
        if folder is not None and run is not None and run.steps.reported:
            try:
                self.service.client.save_steps(folder, run.steps)
            except OSError:  # a result that cannot be written to: its tree is rebuilt from its files
                pass
        self._live_close_detail()
        self.live_tree.clear()
        self._live_trace = ()
        self.steps_button.setVisible(False)
        self.live_preliminary_button.setVisible(False)

    def _live_show_tree(self) -> None:
        """The tree in place of the step lines once the package reports its steps, with what happens now under the
        step at work."""
        run = self._live
        reported = run is not None and run.steps.reported
        self.live_tree_area.setVisible(reported)
        self.live_steps.setVisible(not reported)
        self.live_steps_stretch.setVisible(not reported)
        self.live_hint.setText(tr(CLICK_STEP if reported else LIVE_HINT))
        if reported:
            running = run.steps.running()
            self.live_tree.show_steps(run.steps.ordered(), title=step_title, summary=node_summary,
                                      hint=self._live_status_words() if running else "", hint_at=running)

    def _live_steps_changed(self, kind: str) -> None:
        """An event of the run: the tree follows its steps; an open joint fit follows its iterations."""
        if kind in ("step", "models"):
            self._live_show_tree()
        if kind == "joint" and self._live_detail:
            node = self._live.steps.get(self._live_detail) if self._live is not None else None
            if node is not None and node.detail == "joint.json":
                self._live_open(self._live_detail)

    def _live_tree_chosen(self, step_id: str) -> None:
        node = self._live.steps.get(step_id) if self._live is not None else None
        if node is None or not node.detail:  # nothing saved to show: follow the analysis
            self._live_close_detail()
            return
        self._live_open(step_id)

    def _live_open(self, step_id: str) -> None:
        node = self._live.steps.get(step_id)
        self._live_detail = step_id
        self.live_tree.set_current(step_id)
        self.live_detail_title.setText(step_title(node))
        self.live_detail_back.setText(tr(BACK_TO_RUN))
        try:
            self.live_detail_picture.show(self.service.client.step_detail(self._live_trace, node.detail))
        except (OSError, ValueError) as exc:
            self.live_detail_picture.show(None, message_text(exc))
        self.live_right.setCurrentIndex(1)

    def _live_close_detail(self) -> None:
        """Back to the analysis at work: the step's details are dropped."""
        self._live_detail = ""
        self.live_tree.set_current("")
        self.live_detail_picture.clear()
        self.live_right.setCurrentIndex(0)

    # -- between the preliminary result and the live view -----------------------------------------------------
    def _show_steps_button(self, shown: bool) -> None:
        self.steps_button.setText(tr(STEPS_BUTTON))
        self.steps_button.setVisible(shown)
        on_live = self._live is not None and self.result_stack.currentIndex() == LIVE_PAGE
        self.live_preliminary_button.setText(tr(PRELIMINARY_BUTTON))
        self.live_preliminary_button.setVisible(on_live and getattr(self, "_live_preliminary", None) is not None)

    def _show_live_steps(self) -> None:
        """Steps (beside the note line of the preliminary result): the live view of the run."""
        if self._live is None:
            return
        self.result_stack.setCurrentIndex(LIVE_PAGE)
        self._live_show_status()
        self._live_changed("all")

    def _back_to_preliminary(self) -> None:
        if self._live is not None and getattr(self, "_live_preliminary", None) is not None:
            self.result_stack.setCurrentIndex(1)
            self._live_show_status()
            self._live_show_joint()

    # -- Details ▸ Steps ----------------------------------------------------------------------------------------
    def show_result_steps(self) -> None:
        """The tree of the result shown (read again from its folder); what a step found is dropped."""
        result = self.client_result
        if getattr(self, "steps_picture", None) is None:  # the page is still being built
            return
        self.steps_picture.clear()
        self.steps_detail_title.setText("")
        tree = None
        if result is not None and self.service.client is not None:
            try:
                tree = self.service.client.steps(result.folder)
            except (OSError, ValueError):
                tree = None
        self._result_steps = tree
        self.tab_steps.setCurrentIndex(0 if result is None else 1)
        if tree is None:
            self.steps_tree.clear()
            self.steps_note.setText(tr(NO_STEPS) if result is not None else "")
            return
        self.steps_note.setText(tr(CLICK_STEP) if tree.has_details() else "")
        self.steps_tree.show_steps(tree.ordered(), title=step_title, summary=node_summary)

    def _result_step_chosen(self, step_id: str) -> None:
        tree, result = self._result_steps, self.client_result
        node = tree.get(step_id) if tree is not None else None
        if node is None or result is None:
            return
        self.steps_tree.set_current(step_id)
        self.steps_detail_title.setText(step_title(node))
        if not node.detail:
            self.steps_picture.show(None, tr(NO_DETAILS))
            return
        try:
            self.steps_picture.show(self.service.client.step_detail([result.folder / TRACE_FOLDER], node.detail))
        except (OSError, ValueError) as exc:
            self.steps_picture.show(None, message_text(exc))

    def redraw_step_pictures(self) -> None:
        self.live_detail_picture.redraw()
        self.steps_picture.redraw()

    def refresh_steps_language(self) -> None:
        self._live_show_tree()
        if self._live_detail and self._live is not None:
            self._live_open(self._live_detail)
        chosen = self.steps_tree.current()
        self.show_result_steps()
        if chosen:
            self._result_step_chosen(chosen)


__all__ = ["LiveStepsMixin"]
