"""The refinement after the preliminary result, the joint fit of the whole scan and the pace of a running Analyze's
counts (a part of ``client_live.LiveRun``).

After the network's preliminary result the package refines every window and fits the whole scan at once: the events
``preliminary``, ``refine``, ``joint_stage`` and ``joint`` (see ``client_live``). Each count of the run — the frames
read, a phase of a network, the windows drawn — keeps the time and the count of its first and of its latest event, so
the page can say how long it still needs at the pace it has gone; the joint fit says its own seconds.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from ..domain.client import number

PACE_MIN_S = 1.0
"""A count says how fast it goes once its first and its latest event are at least this far apart."""


class LiveProgressMixin:
    """Needs the fields of ``LiveRun``: ``progress``, ``phases``, ``paces``, ``preliminary``, ``refining``,
    ``joint_stage(_at)`` and ``joint(_at)``."""

    def _refinement(self, event, now: float) -> str:
        """The events of the refinement after the network's preliminary result, and of the joint fit."""
        kind = event.kind
        if kind == "preliminary":
            self.preliminary = True
        elif kind == "refine":
            self.refining = event
        elif kind == "joint_stage":
            self.joint_stage, self.joint_stage_at = event, now
            if event.get("stage") == "fit":
                self.joint = None
        elif event.number("iteration") is not None:  # joint
            self.joint, self.joint_at = event, now
        else:
            return ""
        return kind

    def _restart(self, family: str) -> None:
        """A network starts its windows again: its counts start from nothing (its solutions so far stay)."""
        for key in [key for key in self.progress if key[0] == family]:
            self.progress.pop(key)
        for key in [key for key in self.paces if isinstance(key, tuple) and key[0] == family]:
            self.paces.pop(key)
        self.phases.pop(family, None)
        self.joint_stage, self.joint = None, None

    # -- how fast it goes ------------------------------------------------------------------------------------------
    def _pace(self, key, count: float, now: float) -> None:
        first = self.paces.get(key)
        if first is None or count < first[1]:
            self.paces[key] = (now, count, now, count)
        else:
            self.paces[key] = (first[0], first[1], now, count)

    def left(self, key, total: float, now: Optional[float] = None) -> Optional[float]:
        """The seconds a count still needs to reach ``total`` at the pace it has gone so far (None until it has gone
        for ``PACE_MIN_S``, and once it is there)."""
        found = self.paces.get(key)
        if found is None or not total:
            return None
        start, first, latest, count = found
        if latest - start < PACE_MIN_S or count <= first or count >= total:
            return None
        now = time.monotonic() if now is None else now
        return max(0.0, (total - count) * (latest - start) / (count - first) - max(0.0, now - latest))

    def joint_elapsed(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds since the joint fit started its iterations (None before)."""
        now = time.monotonic() if now is None else now
        if self.joint is not None and self.joint.number("seconds") is not None:
            return self.joint.number("seconds") + max(0.0, now - self.joint_at)
        if self.joint_stage is not None and self.joint_stage.get("stage") == "fit":
            return max(0.0, now - self.joint_stage_at)
        return None

    def joint_left(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds the joint fit still needs if it takes all its iterations: seconds per iteration so far times the
        iterations left (None before the first iteration or after the last)."""
        event = self.joint
        if event is None:
            return None
        iteration, total, seconds = (event.number(key) for key in ("iteration", "total", "seconds"))
        if not iteration or not total or seconds is None or iteration >= total:
            return None
        now = time.monotonic() if now is None else now
        return max(0.0, seconds / iteration * (total - iteration) - max(0.0, now - self.joint_at))

    def joint_points(self) -> tuple:
        """(times, {quantity: values}) of the current solution of the joint fit, in the order of time (empty without
        one)."""
        found = self.joint.get("windows") if self.joint is not None else None
        rows = sorted((row for row in found or () if isinstance(row, dict) and number(row.get("t")) is not None),
                      key=lambda row: number(row.get("t")))
        if not rows:
            return [], {}
        keys = {key for row in rows for key in row if key not in ("t", "fit_deviation")}
        values = {key: [math.nan if number(row.get(key)) is None else number(row.get(key)) for row in rows]
                  for key in sorted(keys)}
        return [number(row.get("t")) for row in rows], values


__all__ = ["LiveProgressMixin", "PACE_MIN_S"]
