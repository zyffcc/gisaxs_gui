"""The Trainset workflow steps in the shared StepRail, with the row API the page has always used.

Each step shows its number (or a mark), its title and its state line ("Reference loaded"); the state
colours the mark like the other workspaces (pending, ok, busy, error). ``setCurrentRow``,
``currentRow`` and ``currentRowChanged`` keep the callers that move between steps by index.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.components import StepRail

STEP_KEYS = ("dataset", "preview", "model", "run", "monitor")
STEP_TITLES = ("Dataset Design", "Local Preview", "Model Design", "Local Run", "Monitor & Results")
NOT_STARTED = "Not started"

_ERROR_WORDS = (
    "FAIL", "CANCEL", "TIMEOUT", "TIMED OUT", "OUT_OF_MEMORY", "OUT OF MEMORY", "NODE_FAIL", "PREEMPTED", "STOPPED",
)
_BUSY_WORDS = ("RUNNING", "PENDING", "SUBMITTED", "JOB ", "CONFIGURING", "COMPLETING", "REQUEUED")


def rail_state(state: str) -> str:
    """The colour of a step from its (English) state line: not started is pending, a running or
    submitted job is busy, a failed or cancelled one an error, anything reached is ok."""
    text = str(state or "").strip()
    if not text or text == NOT_STARTED:
        return "pending"
    upper = text.upper()
    if upper.startswith(_ERROR_WORDS):
        return "error"
    if upper.startswith(_BUSY_WORDS):
        return "busy"
    return "ok"


class TrainsetStepRail(StepRail):
    """StepRail for the five Trainset steps, addressed by row like the list it replaces."""

    currentRowChanged = pyqtSignal(int)

    def __init__(self, titles=STEP_TITLES, parent: Optional[QWidget] = None):
        super().__init__(list(zip(STEP_KEYS, titles)), parent)

    def count(self) -> int:
        return len(self.keys())

    def currentRow(self) -> int:  # noqa: N802 - the QListWidget name its callers use
        keys = self.keys()
        current = self.current()
        return keys.index(current) if current in keys else -1

    def setCurrentRow(self, row: int) -> None:  # noqa: N802 - the QListWidget name its callers use
        keys = self.keys()
        if not 0 <= int(row) < len(keys):
            return
        changed = self.current() != keys[row]
        self.set_current(keys[row])
        if changed:
            self.currentRowChanged.emit(int(row))

    def _choose(self, key: str) -> None:
        """A click or Enter on a step: it becomes the current row (``currentRowChanged``), then ``stepChosen``."""
        if key in self.keys():
            self.setCurrentRow(self.keys().index(key))
            self.stepChosen.emit(key)

    def set_row_state(self, row: int, state: str, detail: str) -> None:
        """The state line of step ``row``: ``state`` (English) picks the colour, ``detail`` is what it shows."""
        keys = self.keys()
        if 0 <= row < len(keys):
            self.set_state(keys[row], rail_state(state), detail)


__all__ = ["NOT_STARTED", "STEP_KEYS", "STEP_TITLES", "TrainsetStepRail", "rail_state"]
