"""The refinement after the preliminary result, the joint fit of the whole scan and the pace of a running Analyze's
counts (a part of ``client_live.LiveRun``).

After the network's preliminary result the package refines every window and fits the whole scan at once: the events
``preliminary``, ``refine``, ``joint_stage`` and ``joint`` (see ``client_live``); a newer package also says, several
times a minute, how many windows of the batch at work are done (``joint_progress``: the iteration being computed, its
phase — ``derivatives``, the long part of an iteration, ``step``, a trial step, ``pool``, preparing —, ``done`` of
``windows`` and the seconds since the fit started). Each count of the run — the frames read, a phase of a network, the
windows drawn — keeps the time and the count of its first and of its latest event, so the page can say how long it
still needs at the pace it has gone; the joint fit says its own seconds, and its time left is seconds per window ×
windows left in the batch + seconds per iteration × iterations left (``joint_batch_left``).
"""

from __future__ import annotations

import math
import time
from typing import Optional

from ..domain.client import number

PACE_MIN_S = 1.0
"""A count says how fast it goes once its first and its latest event are at least this far apart."""
DERIVATIVES_SHARE = 0.9
"""The share of an iteration of the joint fit its derivatives take (the long part; the trial steps are the rest): only
for the bar of the job status."""


class LiveProgressMixin:
    """Needs the fields of ``LiveRun``: ``progress``, ``phases``, ``paces``, ``preliminary``, ``refining``,
    ``joint_stage(_at)``, ``joint(_at)``, ``joint_progress(_at)`` and ``joint_batch``."""

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
                self.joint_progress, self.joint_batch = None, None
        elif kind == "joint_progress":
            return self._joint_progress(event, now)
        elif event.number("iteration") is not None:  # joint
            self.joint, self.joint_at = event, now
        else:
            return ""
        return kind

    def _joint_progress(self, event, now: float) -> str:
        """Windows done in the current batch of the joint fit (``joint_progress``): the batch — an iteration's phase —
        keeps the seconds and the count of its first event, for the pace of its windows."""
        done, windows, seconds = (event.number(key) for key in ("done", "windows", "seconds"))
        if done is None or not windows or seconds is None:
            return ""
        key = (event.number("iteration"), str(event.get("phase") or ""))
        batch = self.joint_batch
        if batch is None or batch[0] != key or done < batch[2]:
            self.joint_batch = (key, seconds, done)
        self.joint_progress, self.joint_progress_at = event, now
        return "joint_progress"

    def _restart(self, family: str) -> None:
        """A network starts its windows again: its counts start from nothing (its solutions so far stay)."""
        for key in [key for key in self.progress if key[0] == family]:
            self.progress.pop(key)
        for key in [key for key in self.paces if isinstance(key, tuple) and key[0] == family]:
            self.paces.pop(key)
        self.phases.pop(family, None)
        self.joint_stage, self.joint = None, None
        self.joint_progress, self.joint_batch = None, None

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
        said = [(event.number("seconds") + max(0.0, now - at)) for event, at in
                ((self.joint, self.joint_at), (self.joint_progress, self.joint_progress_at))
                if event is not None and event.number("seconds") is not None]
        if said:
            return max(said)
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

    def joint_seconds_per_window(self) -> Optional[float]:
        """Seconds one window of the current batch takes: from the batch's first event to its latest, else — the first
        event of an iteration's derivatives — from the end of the iteration before (None while it is not known)."""
        event, batch = self.joint_progress, self.joint_batch
        if event is None or batch is None:
            return None
        done, seconds = event.number("done"), event.number("seconds")
        _key, first_seconds, first_done = batch
        if done > first_done and seconds > first_seconds:
            return (seconds - first_seconds) / (done - first_done)
        before = self.joint.number("seconds") if self.joint is not None else None
        if event.get("phase") == "derivatives" and done > 0 and before is not None and seconds > before:
            return (seconds - before) / done
        return None

    def joint_seconds_per_iteration(self) -> Optional[float]:
        """Seconds an iteration of the joint fit takes: as the iterations so far took them, else (before the first is
        done) its derivatives, every window at the pace of the current batch."""
        event = self.joint
        if event is not None:
            iteration, seconds = event.number("iteration"), event.number("seconds")
            if iteration and seconds is not None:
                return seconds / iteration
        per_window = self.joint_seconds_per_window()
        windows = self.joint_progress.number("windows") if self.joint_progress is not None else None
        return per_window * windows if per_window is not None and windows else None

    def joint_batch_left(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds the joint fit still needs, from the batch at work: seconds per window × the windows left in this
        batch + seconds per iteration × the iterations after this one (None while the pace is not known)."""
        event = self.joint_progress
        per_window = self.joint_seconds_per_window()
        if event is None or per_window is None:
            return None
        done, windows = event.number("done") or 0.0, event.number("windows") or 0.0
        iteration, total = event.number("iteration") or 0.0, event.number("total") or 0.0
        per_iteration = self.joint_seconds_per_iteration() or 0.0
        now = time.monotonic() if now is None else now
        # the iteration of a batch is the one being computed (0 while the pool is prepared): the ones after it are left
        left = per_window * max(0.0, windows - done) + per_iteration * max(0.0, total - iteration)
        return max(0.0, left - max(0.0, now - self.joint_progress_at))

    def joint_batch_share(self) -> Optional[float]:
        """How much of the joint fit is done (0–1) by the batch at work: the iterations before it and the share of its
        windows — the derivatives taken as ``DERIVATIVES_SHARE`` of an iteration, its trial steps as the rest; 0 while
        the pool is prepared (None without a total)."""
        event = self.joint_progress
        if event is None or not event.number("total") or not event.number("windows"):
            return None
        phase, total = event.get("phase"), event.number("total")
        if phase == "pool":
            return 0.0
        before = max(0.0, (event.number("iteration") or 0.0) - 1.0)
        part = min(1.0, (event.number("done") or 0.0) / event.number("windows"))
        within = DERIVATIVES_SHARE * part if phase == "derivatives" else DERIVATIVES_SHARE + (1 - DERIVATIVES_SHARE) * part
        return min(1.0, (before + within) / total)

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


__all__ = ["DERIVATIVES_SHARE", "LiveProgressMixin", "PACE_MIN_S"]
