"""The progress lines of the client analysis package, and the share of a run that is done.

The package's ``analyze`` command runs its steps one after the other and writes on stdout (the contract with the
package):

* ``[n/N] <step> ...`` when step n of N starts (``intake``, ``predict``, ``report``, ``present``; a step that has
  nothing to do says so on the same line: ``[1/4] intake ... reused the imported scan``);
* ``done in <s> s`` or ``FAILED (see <log>) in <s> s`` when it ends (indented);
* ``status: ok; …`` or ``status: failed at <step>; …`` as the last line.

Its ``static`` command (one sample measured after its deposition) names its steps without a number —
``[intake] ...``, ``[predict] ...``, ``[report] ...`` — and ends each with ``done in <s> s`` or
``FAILED: <reason> in <s> s``: ``ClientProgress.static()`` follows it.

A step says nothing while it works, so the bar shows the steps that are finished.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STEP_TEXT = {
    "intake": "Reading the frames",
    "predict": "Finding the solutions of every time window",
    "report": "Writing the report",
    "present": "Drawing the measured and the model images",
}
"""What the page says for each step (the package's own lines go to the log)."""
STEP_WEIGHTS = {"intake": 0.35, "predict": 0.45, "report": 0.05, "present": 0.15}
"""The share of a run each step takes, roughly (reading the frames and the solutions take most of it)."""
STATIC_STEP_TEXT = {
    "intake": "Reading the frames",
    "predict": "Finding the solution of the sample",
    "report": "Writing the report and the images of the series",
}
STATIC_STEP_WEIGHTS = {"intake": 0.3, "predict": 0.45, "report": 0.25}
STEP_OF = "Step {index} of {total}: {stage}…"
STEP_REUSED = "Step {index} of {total}: {stage} — already done for this scan"
_STEP = re.compile(r"^\[(\d+)/(\d+)\]\s+(\w+)[^.]*\.\.\.(.*)$")
_NAMED = re.compile(r"^\[(\w+)\]\s*\.\.\.(.*)$")
_END = re.compile(r"^\s+(done|FAILED)\b")
_STATUS = re.compile(r"^status:\s*([^;]+)")


@dataclass
class ClientProgress:
    """Follows one run of the package: the step, whether it is finished, the steps that failed and the status."""

    index: int = 0
    total: int = 0
    step: str = ""
    finished: bool = False
    reused: bool = False
    done: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    status: str = ""
    texts: dict = field(default_factory=lambda: STEP_TEXT)
    weights: dict = field(default_factory=lambda: STEP_WEIGHTS)

    @classmethod
    def static(cls) -> "ClientProgress":
        """The progress of the ``static`` command."""
        return cls(texts=STATIC_STEP_TEXT, weights=STATIC_STEP_WEIGHTS)

    def feed(self, line: str) -> bool:
        """Take one line of output; True when it was a progress line (the rest is the package's log)."""
        line = line.rstrip("\r\n")
        found = _STEP.match(line)
        if found is not None:
            self._started(found.group(3), int(found.group(1)), int(found.group(2)), found.group(4))
            return True
        found = _NAMED.match(line)
        if found is not None and found.group(1) in self.weights:
            names = list(self.weights)
            self._started(found.group(1), names.index(found.group(1)) + 1, len(names), found.group(2))
            return True
        found = _END.match(line)
        if found is not None and self.step:
            self.finished = True
            (self.done if found.group(1) == "done" else self.failed).append(self.step)
            return True
        found = _STATUS.match(line)
        if found is not None:
            self.status = found.group(1).strip()
            return True
        return False

    def _started(self, step: str, index: int, total: int, rest: str) -> None:
        self.index, self.total, self.step = index, total, step
        self.reused = bool(rest.strip())
        self.finished = self.reused  # a step with nothing to do ends on its own line
        if self.reused:
            self.done.append(step)

    @property
    def fraction(self) -> float:
        """The share of the run that is finished (0–1): the steps before this one, and this one once it ended."""
        if not self.step or self.total <= 0:
            return 0.0
        if self.status:
            return 1.0
        names = list(self.weights)
        if self.step in names and self.total == len(names):
            before = sum(self.weights[name] for name in names[:names.index(self.step)])
            return min(1.0, before + (self.weights[self.step] if self.finished else 0.0))
        return min(1.0, (self.index - 1 + (1 if self.finished else 0)) / self.total)

    def message(self) -> tuple:
        """The template and values of what is happening now (the page translates the template and the stage)."""
        if not self.step:
            return "", {}
        template = STEP_REUSED if self.reused else STEP_OF
        return template, {"index": self.index, "total": self.total, "stage": self.texts.get(self.step, self.step)}


__all__ = ["ClientProgress", "STATIC_STEP_TEXT", "STATIC_STEP_WEIGHTS", "STEP_OF", "STEP_REUSED", "STEP_TEXT",
           "STEP_WEIGHTS"]
