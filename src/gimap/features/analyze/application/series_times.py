"""The times of the rows of a Series map (``series_times``), also of rows that arrive one after another (``RowTimes``).

Read from all its rows at every drawing, the times of a live map of 50 000 rows took 0.08 s of a busy computer, and
0.3 s once the time stamps read before no longer fitted in the cache of ``_text_seconds`` (a file time of its own for
every frame: about 60 000 rows, 1.7 hours at 10 frames a second), longer with every row more. ``RowTimes`` reads the
time stamps of a row once, when it arrives, and keeps what each way of reading the times found of the rows so far;
which way is used still depends on all the rows. ``series_times`` is ``RowTimes`` of all the rows: a map made whole
and a map grown have the same times, value for value.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from ..domain.series_map import RowFacts

FILE_TIME_AGREEMENT_S = 1.5
"""How far the files' modification times may stray from their header time stamps (whole seconds, plus writing)."""

_STAMP = re.compile(
    r"(?P<moment>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)(?:[.,](?P<fraction>\d+))?\s*"
    r"(?P<zone>Z|[+-]\d{2}(?::?\d{2})?)?")
_NOT_SEEN = object()


def _seconds(stamp: str) -> Optional[float]:
    """Seconds of an ISO time stamp, also as detectors write it: “+0200” for “+02:00” (the P03 Lambda), “Z”, a
    space for the “T”, a fraction of any length (Python 3.10 reads only 3 or 6 digits and “+HH:MM”)."""
    return _text_seconds(str(stamp or "").strip())


@lru_cache(maxsize=1 << 16)
def _text_seconds(text: str) -> Optional[float]:
    """``_seconds`` of a stripped text, kept (a map made whole again reads the time stamps of all its rows)."""
    found = _STAMP.fullmatch(text)
    if found is not None:
        zone = found["zone"] or ""
        zone = "+00:00" if zone == "Z" else (zone if len(zone) != 5 else f"{zone[:3]}:{zone[3:]}")
        zone = f"{zone}:00" if len(zone) == 3 else zone
        fraction = f".{found['fraction'][:6].ljust(6, '0')}" if found["fraction"] else ""
        text = found["moment"].replace(" ", "T") + fraction + zone
    try:
        return datetime.fromisoformat(text).timestamp() if text else None
    except ValueError:
        return None


class _PerFrame:
    """Whether values are times of the frames: every row has one, they differ between the rows of one source (a
    file's start time, the same for each of its frames, is no time of a frame) and never run backwards. Read on
    from the row it stopped at, when asked; once not, never again (rows only come after)."""

    def __init__(self) -> None:
        self.holds, self._read, self._last = True, 0, {}

    def through(self, facts: Sequence[RowFacts], values: Sequence) -> bool:
        if not self.holds:
            return False
        last = self._last
        for row in range(self._read, len(values)):
            value = values[row]
            if value is None:
                break
            key = facts[row].source.casefold()
            before = last.get(key, _NOT_SEEN)
            if before is not _NOT_SEEN and not value > before:
                break
            if row and not float(value) - float(values[row - 1]) >= 0:
                break
            last[key] = value
        else:
            self._read = len(values)
            return True
        self.holds, self._last = False, {}
        return False


class _FileTimes:
    """Seconds since the first row from the files' modification times, when every row is a file of its own with a
    header time stamp of whole seconds that two rows share somewhere (frames faster than the stamps), and the files'
    times agree with those stamps (one offset, within ``FILE_TIME_AGREEMENT_S``) and never run backwards. Files
    copied without their times disagree, and are not used."""

    def __init__(self) -> None:
        self.holds, self._tie, self._written = True, False, []
        self._low, self._high, self._stamp = float("inf"), float("-inf"), None

    def extend(self, facts: Sequence[RowFacts], stamps: Sequence[Optional[float]]) -> None:
        if not self.holds or not facts:
            return
        if any(fact.frame_time_s is not None or fact.frame_count > 1 for fact in facts):
            self.holds, self._written = False, []
            return
        files = [_seconds(fact.file_time) for fact in facts]
        if any(value is None for value in [*files, *stamps]):
            self.holds, self._written = False, []
            return
        written, stamped = np.asarray(files, dtype=float), np.asarray(stamps, dtype=float)
        if self._written:  # (the rows before: their last file time and stamp)
            written, stamped = np.r_[self._written[-1], written], np.r_[self._stamp, stamped]
        if np.any(np.diff(written) < 0):
            self.holds, self._written = False, []
            return
        self._tie = self._tie or bool(np.any(np.diff(stamped) == 0))
        offsets = np.asarray(files, dtype=float) - np.asarray(stamps, dtype=float)
        self._low, self._high = min(self._low, float(offsets.min())), max(self._high, float(offsets.max()))
        self._written.extend(files)
        self._stamp = stamps[-1]

    def times(self) -> Optional[np.ndarray]:
        if not (self.holds and self._tie) or float(self._high - self._low) > FILE_TIME_AGREEMENT_S:
            return None
        written = np.asarray(self._written, dtype=float)
        times = written - written[0]
        return times if times[-1] > 0 else None


class RowTimes:
    """``series_times`` of rows whose facts arrive one after another (``extend``): see the module docstring."""

    def __init__(self, facts: Sequence[RowFacts] = ()) -> None:
        self._facts: list[RowFacts] = []
        self._starts: list[Optional[float]] = []
        self._frame_times: list = []
        self._indices: list[int] = []
        self._files = _FileTimes()
        self._per_frame = (_PerFrame(), _PerFrame())
        self._read = 0
        """Rows read for the frame numbers × exposure (only when the times are not of the frames: ``_read_on``)."""
        self._exposures: set = set()
        self._several_frames = False
        self._sources: set = set()
        self._starts_missing = False
        self.extend(facts)

    def __len__(self) -> int:
        return len(self._facts)

    def extend(self, facts: Sequence[RowFacts]) -> None:
        facts = list(facts)
        starts = [_seconds(fact.timestamp) for fact in facts]
        self._facts.extend(facts)
        self._starts.extend(starts)
        self._frame_times.extend([fact.frame_time_s for fact in facts])
        self._indices.extend([fact.frame_index for fact in facts])
        self._files.extend(facts, starts)

    def _read_on(self) -> None:
        """The exposures, multi-frame sources, sources and missing start times of the rows not read yet."""
        facts, starts = self._facts[self._read:], self._starts[self._read:]
        self._read = len(self._facts)
        if len(self._exposures) < 2:  # (two: never one again)
            self._exposures.update(fact.exposure_s for fact in facts)
        self._several_frames = self._several_frames or any(fact.frame_count > 1 for fact in facts)
        if len(self._sources) < 2:
            self._sources.update(fact.source.casefold() for fact in facts)
        self._starts_missing = self._starts_missing or any(start is None for start in starts)

    def times(self) -> tuple[Optional[np.ndarray], str]:
        """``(seconds since the first row, where they come from)``, or ``(None, "")`` when the frames carry no
        time. In this order: a time of each frame (a per-frame dataset, or time stamps that differ between the
        rows of a source and never run backwards: ``timestamps``), else — for multi-frame sources with one
        exposure time — frame number × exposure (``exposure``, one source) or the start time of each file + frame
        number × exposure (``start_exposure``, several); both a lower bound: the read-out between frames is not
        in the file."""
        if len(self._facts) < 2:
            return None, ""
        files = self._files.times()
        if files is not None:
            return files, "file_times"
        for per_frame, values in zip(self._per_frame, (self._frame_times, self._starts)):
            if per_frame.through(self._facts, values):
                times = np.asarray(values, dtype=float) - float(values[0])
                if times[-1] > 0:
                    return times, "timestamps"
        self._read_on()
        exposures = self._exposures
        if len(exposures) != 1 or exposures == {None} or float(next(iter(exposures))) <= 0:
            return None, ""
        if not self._several_frames:
            return None, ""
        exposure = float(next(iter(exposures)))
        indices = np.asarray(self._indices)
        if len(self._sources) == 1:
            return np.asarray((indices - self._indices[0]) * exposure, dtype=float), "exposure"
        if self._starts_missing:
            return None, ""
        times = np.asarray(np.asarray(self._starts, dtype=float) + indices * exposure, dtype=float)
        times -= times[0]
        if np.all(np.diff(times) >= 0) and times[-1] > 0:
            return times, "start_exposure"
        return None, ""


def series_times(facts: Sequence[RowFacts]) -> tuple[Optional[np.ndarray], str]:
    """The times of the rows of a map (``RowTimes.times``)."""
    return RowTimes(facts).times()


__all__ = ["FILE_TIME_AGREEMENT_S", "RowTimes", "series_times"]
