"""Which new frames of a watched folder a folder batch of 2D Prediction takes, and which it leaves out to keep up.

Watch for new frames: once the frames of the folder are predicted, the batch goes on with every frame written into the
folder after it started. A new frame is taken once it is completely written:

* an EDF whose header announces its size is taken at the first look that sees it whole (``complete``);
* any other file once it has the same size and modification time at two looks in a row (a frame still being
  written is not read on its first rows);
* an EDF that says it is incomplete once it has not changed for ``INCOMPLETE_REST_S`` (a damaged file: its
  prediction then says what is wrong with it, rather than the watch waiting for it for ever).

Frames are taken in the order of their numbers; with Every N they are summed in stacks of N, and frames that do not
yet fill a stack wait for the next ones. A frame whose number is outside the batch's Range is not predicted (counted).

Keep up with the detector (``KeepUp``): besides the one the model predicts, at most two stacks wait — one in the model's
queue and ``WAITING`` here; when more arrive, the oldest waiting ones are left out (counted, and still in the folder for a later folder batch). A model
as fast as the detector leaves nothing out; a slower one stays at most a few predictions behind the newest frame.
Without it every frame is predicted in turn, however far behind the detector that falls.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

INCOMPLETE_REST_S = 5.0
"""How long an EDF that says it is incomplete must stay unchanged before it is taken all the same."""

WAITING = 1
"""Stacks that may wait here with Keep up with the detector (besides one in the model's queue and the one predicted)."""


@dataclass(frozen=True)
class FolderEntry:
    """One frame file seen at a look at the folder."""

    path: str
    stamp: tuple
    """``(bytes, modification time in ns)``."""
    complete: Optional[bool] = None
    """What the file says itself: ``True`` whole, ``False`` still being written, ``None`` not known (two looks)."""


class FrameWatch:
    """The new frames of a folder, as stacks of ``every`` once they are completely written (``take``)."""

    def __init__(self, known: Iterable[str], every: int = 1, *, waiting: Sequence[str] = (),
                 indices: Optional[Iterable[int]] = None, index_of: Callable[[str], Optional[int]] = lambda _p: None):
        self.every = max(1, int(every))
        self._known = set(known) | set(waiting)
        self._last: dict[str, tuple[tuple, float]] = {}
        """Each new frame not taken yet: its stamp at the look before, and since when it has had it."""
        self._ready: list[str] = list(waiting)
        """Taken frames that do not fill a stack yet (with Every N), in their order."""
        self._indices = None if indices is None else frozenset(indices)
        self._index_of = index_of
        self.outside = 0
        """New frames left out because their number is outside the batch's Range."""

    @property
    def known(self) -> frozenset:
        """Every frame taken (or there when the watch began): a look need not report them again."""
        return frozenset(self._known)

    @property
    def waiting(self) -> tuple[str, ...]:
        """Frames taken that wait for the rest of their stack."""
        return tuple(self._ready)

    @property
    def unsettled(self) -> int:
        """New frames seen that are still being written."""
        return len(self._last)

    def _order(self, path: str):
        index = self._index_of(path)
        return (0 if index is not None else 1, index if index is not None else 0, path.casefold())

    def take(self, entries: Iterable[FolderEntry], now: Optional[float] = None) -> list[list[str]]:
        """The stacks completed by the frames of this look that are completely written, oldest first."""
        now = time.monotonic() if now is None else float(now)
        seen = set()
        for entry in entries:
            seen.add(entry.path)
            if entry.path in self._known:
                continue
            before = self._last.get(entry.path)
            same = before is not None and before[0] == entry.stamp
            since = before[1] if same else now
            if entry.complete is True:
                whole = True
            elif entry.complete is None:
                whole = same and entry.stamp[0] > 0
            else:
                whole = same and now - since >= INCOMPLETE_REST_S
            if not whole:
                self._last[entry.path] = (entry.stamp, since)
                continue
            self._known.add(entry.path)
            self._last.pop(entry.path, None)
            if self._indices is not None and self._index_of(entry.path) not in self._indices:
                self.outside += 1
                continue
            self._ready.append(entry.path)
        for gone in [path for path in self._last if path not in seen]:  # deleted or renamed before it was whole
            del self._last[gone]
        self._ready.sort(key=self._order)
        full = len(self._ready) // self.every * self.every
        stacks = [self._ready[start:start + self.every] for start in range(0, full, self.every)]
        self._ready = self._ready[full:]
        return stacks


class KeepUp:
    """With ``on``: at most ``depth`` stacks wait here for the model; when more come, the oldest are left out.

    The model gets one stack at a time (``release``) whenever nothing waits in its own queue (``free``), so what it
    predicts next is never more than ``depth`` + 1 stacks older than the newest."""

    def __init__(self, on: bool, depth: int = WAITING):
        self.on = bool(on)
        self.depth = max(1, int(depth))
        self.pending: deque = deque()
        self.left_out = 0
        """Frames left out to keep up with the detector."""

    def offer(self, stacks: Sequence[Sequence[str]], free: bool) -> list[list[str]]:
        """The stacks to give the model now, of the ones just taken (all of them without Keep up)."""
        stacks = [list(stack) for stack in stacks]
        if not self.on:
            return stacks
        self.pending.extend(stacks)
        while len(self.pending) > self.depth:
            self.left_out += len(self.pending.popleft())
        return self.release(free)

    def release(self, free: bool) -> list[list[str]]:
        """The oldest stack waiting, when the model's queue is ``free``."""
        if not free or not self.pending:
            return []
        return [self.pending.popleft()]

    def release_all(self) -> list[list[str]]:
        """Every stack waiting (Keep up unticked: they are predicted in turn)."""
        stacks, self.pending = list(self.pending), deque()
        return stacks


__all__ = ["INCOMPLETE_REST_S", "WAITING", "FolderEntry", "FrameWatch", "KeepUp"]
