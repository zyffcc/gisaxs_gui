"""Report new detector frames in a folder once they are completely written, and listed ones that changed.

A source that changes at every look (a detector writing a frame a second into one file for minutes) never
rests. It is reported all the same, as ``moving``, with what it was at the look before: whatever it held
then has been in its files for a whole look and can be read, while its newest part may still be written.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class WatchScan:
    """One look at the watched folder (``FolderWatch.scan``): what is in it, and how large, at that moment."""

    number: int
    folder: Path
    sources: tuple[tuple[Path, str, tuple], ...]
    """``(path, key, state)`` of every source that could be measured; the state is ``(bytes, stamp)``."""
    waiting: tuple[tuple[Path, str], ...]
    """``(path, why)`` of the sources that could not be measured (their writer still holds them)."""


class FolderWatch:
    """Poll a folder; a frame counts as new when its size is unchanged between polls.

    Detector servers write files incrementally, so a file is only reported
    after two consecutive polls saw the same non-zero size.  ``expand`` maps the
    folder to its frames (for example collapsing NXS module series).

    A source that was reported, or was there when the watch began, is looked at again at every poll:
    once its files changed and then rested for one poll it is in ``changed`` (a series that grows
    inside its file, a module file that arrived).  ``stamp_of`` adds what the size does not show
    (modification times, the number of frames).  A source ``size_of`` cannot measure now
    (``OSError``: its writer holds it) is in ``waiting`` and is asked again at the next poll.  A source,
    known or new, that was seen changed at the look before and has changed again is in ``moving`` with
    its state at that look; a new one stays new (unreported) until it rests, unless the caller takes it
    up meanwhile (``follow``).

    ``poll`` is ``take(scan())``.  ``scan`` only reads the folder and changes nothing here, so it may
    run on another thread while the window stays free; ``take`` applies it on the caller's thread.
    """

    def __init__(
        self,
        expand: Callable[[Sequence[Path]], list[Path]],
        size_of: Optional[Callable[[Path], int]] = None,
        stamp_of: Optional[Callable[[Path], object]] = None,
    ):
        self._expand = expand
        self._size_of = size_of or (lambda path: Path(path).stat().st_size)
        self._stamp_of = stamp_of
        self.folder: Optional[Path] = None
        self._known: dict[str, Optional[tuple]] = {}
        """Source → the state it was reported with (``None``: listed, never measured)."""
        self._sizes: dict[str, tuple] = {}
        """Source → its state at the poll before, until it has rested."""
        self._numbers = itertools.count(1)
        self._taken = 0
        self.changed: list[Path] = []
        """After a poll: the known sources whose files changed and have rested since."""
        self.waiting: list[tuple[Path, str]] = []
        """After a poll: ``(path, why)`` of the sources that cannot be measured yet."""
        self.moving: list[tuple[Path, tuple]] = []
        """After a poll: ``(path, its state at the look before)`` of the sources that changed at that look and
        again since: they do not rest."""

    @property
    def active(self) -> bool:
        return self.folder is not None

    def start(self, folder: Path, *, already_listed: Sequence[Path] = ()) -> None:
        """Watch ``folder``; frames already present or listed are not reported."""
        self.folder = Path(folder)
        self._sizes.clear()
        self.changed, self.waiting, self.moving = [], [], []
        self._known = {self._key(path): None for path in already_listed}
        for path in self._expand([self.folder]):
            self._known[self._key(path)] = self._measure(path)[0]
        self._taken = next(self._numbers)  # a look begun before this is of the watch before

    def stop(self) -> None:
        self.folder = None
        self._sizes.clear()
        self.changed, self.waiting, self.moving = [], [], []

    def _measure(self, path: Path) -> tuple[Optional[tuple], str]:
        """``((bytes, stamp), "")`` of a source, or ``(None, why)`` when it cannot be measured now."""
        try:
            size = int(self._size_of(path))
            stamp = self._stamp_of(path) if self._stamp_of is not None else None
        except OSError as error:
            return None, str(error)
        return (size, stamp), ""

    def scan(self) -> Optional[WatchScan]:
        """Look at the folder: its sources and their state now (``None``: nothing is watched)."""
        folder, number = self.folder, next(self._numbers)
        if folder is None or not folder.is_dir():
            return None
        sources, waiting = [], []
        for path in self._expand([folder]):
            state, why = self._measure(path)
            if state is None:
                waiting.append((path, why))
            else:
                sources.append((path, self._key(path), state))
        return WatchScan(number, folder, tuple(sources), tuple(waiting))

    def take(self, scan: Optional[WatchScan]) -> list[Path]:
        """Apply a look at the folder: the new, completely written frames since the last one (oldest name
        first); ``changed``, ``moving`` and ``waiting`` say what else it found. A look at another folder, or one
        older than a look applied already, is dropped."""
        self.changed, self.waiting, self.moving = [], [], []
        if scan is None or scan.folder != self.folder or scan.number <= self._taken:
            return []
        self._taken = scan.number
        self.waiting = list(scan.waiting)
        ready: list[Path] = []
        for path, key, state in scan.sources:
            known = key in self._known
            if known and self._known[key] == state:
                self._sizes.pop(key, None)
                continue
            before = self._sizes.get(key)
            if (known or state[0] > 0) and before == state:  # the same at two looks: it rests
                self._sizes.pop(key, None)
                self._known[key] = state
                (self.changed if known else ready).append(path)
            else:
                if before is not None:  # changed at the look before, and again: what it was then
                    self.moving.append((path, before))
                self._sizes[key] = state
        return ready

    def poll(self) -> list[Path]:
        """New, completely written frames since the last poll (oldest name first)."""
        return self.take(self.scan())

    def follow(self, path: Path) -> None:
        """The caller lists a new source although it has not rested (it is ``moving``): from now on it is a
        known source, which has ``changed`` when it rests at last."""
        self._known.setdefault(self._key(path), None)

    def rests(self, path: Path) -> bool:
        """At the last look applied the source was as at the look before (``False``: it was seen changed, or
        new, and has not rested yet; one of its files may still be being written)."""
        return self._key(path) not in self._sizes

    @staticmethod
    def _key(path: Path) -> str:
        return str(Path(path).resolve()).casefold()


__all__ = ["FolderWatch", "WatchScan"]
