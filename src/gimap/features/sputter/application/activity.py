"""Is the analysis package working? Its process tree, sampled while it runs.

The package's own lines and live events say what it has found; between them it can be silent for a while (a network
starting its worker processes, the joint fit inside an iteration). Every few seconds the infrastructure samples the
package's process tree — its processes, how many of them used the processor or read from disk since the sample
before — and, while none did, the last line of the step's log (``ClientActivity``). ``ActivityWatch`` keeps the
latest sample and when the package last worked or said anything:

* while it works, the page says how many of its processes are at work (the clock line);
* when none of its processes has worked and nothing came from it for ``STALL_S``, the package has most likely stopped
  on an error that never reached its main process — a worker process that ended while its pool waits for it waits for
  ever — and the page says so, with the last line of its log, and offers to stop it. It is not stopped for the user:
  a disk that does not answer looks the same for a while.

Without psutil there are no samples and nothing is said.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STALL_S = 60.0
"""Seconds without work or news after which the package is said to have stopped working."""


@dataclass(frozen=True)
class ClientActivity:
    """One sample of the package's process tree."""

    processes: int
    """Processes in the tree (the package's main process, the step's process and their workers)."""
    busy: int
    """Those that used the processor or read from disk since the sample before."""
    log_line: str = ""
    """The last line of the newest step log of the run ("" while a process works, or when there is none)."""


@dataclass
class ActivityWatch:
    """The latest sample and when the package last worked or said anything (monotonic seconds)."""

    started: float = 0.0
    latest: Optional[ClientActivity] = None
    worked_at: Optional[float] = None
    news_at: Optional[float] = None

    def feed(self, sample: ClientActivity, now: float) -> None:
        self.latest = sample
        if sample.busy > 0:
            self.worked_at = now

    def news(self, now: float) -> None:
        """A line or an event of the package arrived."""
        self.news_at = now

    def busy(self) -> Optional[int]:
        """Processes at work in the latest sample (None: no sample yet)."""
        return self.latest.busy if self.latest is not None else None

    def quiet(self, now: float) -> Optional[float]:
        """Seconds since the package last worked or said anything (None without a sample)."""
        if self.latest is None:
            return None
        since = max(value for value in (self.started, self.worked_at, self.news_at) if value is not None)
        return max(0.0, now - since)

    def stalled(self, now: float) -> bool:
        """None of the package's processes works and nothing has come from it for ``STALL_S``."""
        quiet = self.quiet(now)
        return quiet is not None and self.latest.busy == 0 and quiet >= STALL_S

    def log_line(self) -> str:
        return self.latest.log_line if self.latest is not None else ""


__all__ = ["ActivityWatch", "ClientActivity", "STALL_S"]
