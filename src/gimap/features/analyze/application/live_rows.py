"""The rows of a live Series map as frames arrive: each frame once, in the order of the frame list.

A watch puts the rows of its map in the order of the listed frames before every redraw (``ListedRows``): every frame
once (of two rows of one frame, the one that sums more frames), only the frames of the list, by their number in it.
Done from all the rows each time, it took the window 0.1 s at 50 000 rows, longer with every row more. ``ListedRows``
keeps the order made before and adds the rows that arrived while they are new frames numbered after the others and
the frames before keep their numbers (``numbers_agree``); the order is the order of all the rows, row for row
(``tests/test_series_angles.py``). ``series_angles.AngleSplit`` does the same for the split by incidence angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import is_
from types import MappingProxyType
from typing import Mapping, Optional, Sequence


def row_identity(row) -> tuple[str, int]:
    """The frame of a row ``(x, intensity, x_label, label, ref)``: its source whatever the case, and frame."""
    return str(row[4][0]).casefold(), int(row[4][1])


def _summed(row) -> int:
    return getattr(row[3], "summed", 1)


def unique_rows(rows: Sequence) -> list:
    """Every frame once, in the order they came: of two rows of one frame, the one that sums more frames (a short
    last sum completed)."""
    kept: dict = {}
    for row in rows:
        key = row_identity(row)
        old = kept.get(key)
        if old is None or _summed(row) > _summed(old):
            kept[key] = row
    return list(kept.values())


def numbers_agree(before: Optional[Mapping], now: Optional[Mapping]) -> bool:
    """Every frame ``before`` numbers has the same number in ``now`` (``False``: not known; look frame by frame). A
    mapping can say so of another itself (``now.agrees_with(before)``: the frame numbers of the frame list)."""
    if before is None or now is None:
        return before is None and now is None
    agrees = getattr(now, "agrees_with", None)
    if callable(agrees):
        return bool(agrees(before))
    return isinstance(before, dict) and isinstance(now, dict) and before.items() <= now.items()


def kept_numbers(numbers: Optional[Mapping]) -> Optional[Mapping]:
    """``numbers`` as they are now, to compare with later: a dictionary is copied (it may be changed in place), other
    mappings (the frame numbers of the frame list cannot change) are kept; empty: ``None``."""
    if not numbers:
        return None
    return dict(numbers) if isinstance(numbers, dict) else numbers


def increasing_after(last, keys: Sequence) -> bool:
    """``keys`` each greater than the one before, the first greater than ``last``, none infinite (a frame without a
    number): appended, they leave rows sorted by them sorted."""
    return all(a < b for a, b in zip([last, *keys], keys)) and (not keys or keys[-1] < float("inf"))


@dataclass
class _Listed:
    rows: list
    keys: list
    frames: set
    numbers: Mapping
    places: dict
    strict: bool
    handed: list


class ListedRows:
    """The rows of a live map in the order of the frame list, made again as rows arrive (see the module docstring)."""

    def __init__(self) -> None:
        self._last: Optional[_Listed] = None

    def forget(self) -> None:
        self._last = None

    def order(self, rows: Sequence, numbers: Mapping) -> tuple[list, Mapping]:
        """``(rows, places)``: every frame of ``rows`` once that ``numbers`` (``row_identity`` → its number in the
        frame list) numbers, sorted by its number; ``places``: ``row_identity`` → its row."""
        last, self._last = self._last, None  # (kept again once made: a failure leaves nothing half done)
        state = self._grown(last, rows, numbers) if last is not None else None
        state = state or self._whole(rows, numbers)
        state.handed = list(state.rows)
        self._last = state if state.numbers is not None else None
        return state.handed, MappingProxyType(state.places)

    @staticmethod
    def _whole(rows: Sequence, numbers: Mapping) -> _Listed:
        unique: dict = {}
        for row in rows:
            identity = row_identity(row)
            if identity in numbers:
                old = unique.get(identity)
                if old is None or _summed(row) > _summed(old):
                    unique[identity] = row
        identities = sorted(unique, key=numbers.get)
        keys = [numbers.get(identity) for identity in identities]
        ordered = [unique[identity] for identity in identities]
        strict = increasing_after(float("-inf"), keys)
        return _Listed(ordered, keys, set(identities), kept_numbers(numbers),
                       {identity: place for place, identity in enumerate(identities)}, strict, [])

    @staticmethod
    def _grown(last: _Listed, rows: Sequence, numbers: Mapping) -> Optional[_Listed]:
        """The order before with the rows that arrived; ``None``: made from all the rows."""
        before = len(last.rows)
        if (not last.strict or not last.rows or not numbers or len(rows) < before
                or not all(map(is_, rows, last.rows)) or not numbers_agree(last.numbers, numbers)):
            return None
        new = list(rows[before:])
        frames = [row_identity(row) for row in new]
        if len(set(frames)) < len(frames) or any(frame in last.frames for frame in frames):
            return None  # (a frame again: the row that sums more is kept, wherever it is)
        listed = [(row, frame) for row, frame in zip(new, frames) if frame in numbers]
        keys = [numbers.get(frame) for _row, frame in listed]
        if not increasing_after(last.keys[-1], keys):
            return None  # (a frame listed before others: sorted again)
        last.places.update((frame, place) for place, (_row, frame) in enumerate(listed, before))
        last.rows.extend(row for row, _frame in listed)
        last.keys.extend(keys)
        last.frames.update(frame for _row, frame in listed)
        last.numbers = kept_numbers(numbers)
        return last


__all__ = ["ListedRows", "increasing_after", "kept_numbers", "numbers_agree", "row_identity", "unique_rows"]
