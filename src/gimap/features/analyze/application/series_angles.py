"""Rows of a series measured at different incidence angles: one map per angle.

A map stacks one cut of every frame at the same detector rows. When the frames of a run switch between incidence
angles (a reference angle and short looks at others between spray pulses), those rows are another exit angle — and
another q_z — at every αi, so rows of different angles do not belong in one picture. The angle of a row is its
nominal αi (``incidence_record.nominal_incidence``: typed, from the file header or side file, from the profile).
Angles closer than ``TOLERANCE_DEG`` are one angle (a motor reads 0.399998 for 0.4).

By default a map holds the angle most of its rows were measured at, when that angle has at least two rows and the
rows have more than one angle; a scan through αi (every angle once) stays one map of all its rows.

A live map sorts its rows by angle again at every redraw (``AngleSplit``). Done from all the rows each time, it took
the window 0.35 s at 50 000 rows — every frame once, in the order of the frame numbers, the groups of angles twice,
the rows of the angle and the others — longer with every row more. ``AngleSplit`` adds the rows that arrived to the
split before while they are new frames that come after the others (by frame number) and the rows before are as they
were; the split is the split of all the rows, row for row (``tests/test_series_angles.py``). The rows come in
the order of the frame list (``live_rows.ListedRows``).
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import is_
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence

from .live_rows import increasing_after, kept_numbers, numbers_agree, row_identity, unique_rows

TOLERANCE_DEG = 0.005
ALL = "all"
"""The choice of a map of every row, whatever its angle."""
INCIDENCE_LINE = "frames at alpha_i = {angle:g} deg only; {n} frames at other angles left out ({others})"
"""What a series export says of a map of one angle."""


def row_incidence(row) -> Optional[float]:
    """The nominal αi of a row ``(x, intensity, x_label, label, ref)`` (``RowLabel.incidence_deg``)."""
    value = getattr(row[3], "incidence_deg", None)
    return None if value is None else float(value)


def same_incidence(first: Optional[float], second: Optional[float]) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return abs(float(first) - float(second)) <= TOLERANCE_DEG


def _count_angles(groups: list, rows: Sequence) -> list:
    """``groups`` (``[angle, rows]`` in the order the angles came) with ``rows`` counted in: a row joins the first
    group within ``TOLERANCE_DEG`` of its angle, else begins one."""
    for row in rows:
        angle = row_incidence(row)
        for group in groups:
            if same_incidence(group[0], angle):
                group[1] += 1
                break
        else:
            groups.append([angle, 1])
    return groups


def _by_angle(groups: list) -> list[tuple[Optional[float], int]]:
    return sorted(((angle, count) for angle, count in groups), key=lambda group: (group[0] is None, group[0] or 0.0))


def incidence_groups(rows: Sequence) -> list[tuple[Optional[float], int]]:
    """``(angle, rows)`` of every angle among ``rows``, by angle (rows without one last)."""
    return _by_angle(_count_angles([], rows))


def _default(groups: Sequence[tuple[Optional[float], int]], first: Optional[float]):
    """``default_incidence`` of rows with these ``incidence_groups``, ``first`` the angle of the first row."""
    if len(groups) < 2:
        return ALL
    most = max(count for _angle, count in groups)
    if most < 2:
        return ALL
    tied = [angle for angle, count in groups if count == most]
    return next((angle for angle in tied if same_incidence(angle, first)), tied[0])


def default_incidence(rows: Sequence):
    """The angle a map of ``rows`` holds unless one is chosen: the angle of most rows when the rows have several
    angles and that one has two rows or more (ties: the first row's), else ``ALL``."""
    return _default(incidence_groups(rows), row_incidence(rows[0]) if len(rows) else None)


def split_rows(rows: Sequence, angle) -> tuple[list, list]:
    """``(kept, aside)``: the rows at ``angle`` (``ALL``: every row), and the others."""
    if angle == ALL:
        return list(rows), []
    kept, aside = [], []
    for row in rows:
        (kept if same_incidence(row_incidence(row), angle) else aside).append(row)
    return kept, aside


def incidence_line(angle, aside: Sequence, groups: Optional[Sequence] = None) -> str:
    """The export line of a map of one angle (empty for a map of every row); ``groups``: those of ``aside``, when
    known (``incidence_groups``)."""
    if angle == ALL or angle is None or not aside:
        return ""
    others = ", ".join(f"{value:g} deg: {count}" if value is not None else f"no angle: {count}"
                       for value, count in (incidence_groups(aside) if groups is None else groups))
    return INCIDENCE_LINE.format(angle=float(angle), n=len(aside), others=others)


# -- the rows of a live map, by angle -----------------------------------------------------------------------------

class Split(NamedTuple):
    """The rows of a map, by angle (``AngleSplit.split``)."""

    kept: list
    """The rows of the map: at ``angle``, in frame order."""
    aside: list
    angle: object
    """``ALL``, or the angle of the map."""
    groups: list
    """``incidence_groups`` of all the rows, either side."""
    rows: int
    """All the rows, either side (every frame once)."""
    places: Mapping
    """``row_identity`` → its row in ``kept``."""
    line: str
    """``incidence_line`` of the map."""


@dataclass
class _Sorted:
    everything: list
    keys: Optional[list]
    """The frame number of every row of ``everything`` (sorted by them), or ``None``: not sorted (no numbers)."""
    frames: set
    groups: list
    aside_groups: list
    kept: list
    aside: list
    angle: object
    choice: object
    numbers: Optional[Mapping]
    found: bool
    """A row is at the angle chosen (else the map shows the angle of most rows)."""
    places: dict
    alone: bool
    """No row at the angle: the map shows every row."""
    strict: bool
    """Every row has a frame number, each greater than the one before (then the order cannot be another)."""
    handed: tuple = ((), ())
    """The lists of the last split handed out: the rows and those aside, as they are asked for again."""


class AngleSplit:
    """The rows of a live map, and those kept aside, split by angle again as rows arrive (see the module docstring).
    ``split`` is what was done from all the rows: ``unique_rows``, sorted by frame number, ``default_incidence`` or
    the angle chosen, ``split_rows``."""

    def __init__(self) -> None:
        self._last: Optional[_Sorted] = None

    def forget(self) -> None:
        self._last = None

    def split(self, rows: Sequence, aside: Sequence, numbers: Optional[Mapping], choice) -> Split:
        """``rows``: those of the map before and those that arrived; ``aside``: those set aside before; ``numbers``:
        ``row_identity`` → frame number (``None`` or empty: the rows as they came); ``choice``: ``None`` (the angle of
        most rows), ``ALL`` or an angle."""
        last, self._last = self._last, None  # (kept again once made: a failure leaves nothing half done)
        numbers = numbers or None
        state = self._grown(last, rows, aside, numbers, choice) if last is not None else None
        state = state or self._whole(rows, aside, numbers, choice)
        groups = _by_angle(state.groups)
        line = incidence_line(state.angle, state.aside, _by_angle(state.aside_groups))
        kept, set_aside = list(state.kept), list(state.aside)
        state.handed = (kept, set_aside)
        self._last = state
        return Split(kept, set_aside, state.angle, groups, len(state.everything), MappingProxyType(state.places), line)

    @staticmethod
    def _whole(rows: Sequence, aside: Sequence, numbers: Optional[Mapping], choice) -> _Sorted:
        everything = unique_rows([*rows, *aside])
        keys = None
        if numbers:
            unsorted = [numbers.get(row_identity(row), float("inf")) for row in everything]
            order = sorted(range(len(everything)), key=unsorted.__getitem__)
            everything, keys = [everything[i] for i in order], [unsorted[i] for i in order]
        groups = _count_angles([], everything)
        first = row_incidence(everything[0]) if everything else None
        angle = _default(_by_angle(groups), first) if choice is None else choice
        found = True
        if angle != ALL and not any(same_incidence(row_incidence(row), angle) for row in everything):
            found, angle = False, _default(_by_angle(groups), first)
        kept, set_aside = split_rows(everything, angle)
        alone = not kept
        if alone:
            kept, set_aside, angle = list(everything), [], ALL
        strict = keys is not None and increasing_after(float("-inf"), keys)
        return _Sorted(everything, keys, {row_identity(row) for row in everything}, groups,
                       _count_angles([], set_aside), kept, set_aside, angle, choice, kept_numbers(numbers), found,
                       {row_identity(row): place for place, row in enumerate(kept)}, alone, strict)

    @staticmethod
    def _grown(last: _Sorted, rows: Sequence, aside: Sequence, numbers: Optional[Mapping], choice) -> Optional[_Sorted]:
        """The split before with the rows that arrived; ``None``: made from all the rows."""
        handed_rows, handed_aside = last.handed
        before = len(last.kept)
        if (last.alone or not last.strict or not last.everything or choice != last.choice or numbers is None
                or aside is not handed_aside or len(aside) != len(last.aside) or not all(map(is_, aside, last.aside))
                or len(rows) < before or not all(map(is_, rows, last.kept))
                or not numbers_agree(last.numbers, numbers)):
            return None
        new = list(rows[before:])
        frames = [row_identity(row) for row in new]
        if len(set(frames)) < len(frames) or any(frame in last.frames for frame in frames):
            return None  # (a frame again: the row that sums more is kept, wherever it is)
        keys = [numbers.get(frame, float("inf")) for frame in frames]
        if not increasing_after(last.keys[-1], keys):
            return None  # (a frame before others, or one without a number: sorted again)
        last.everything.extend(new)
        last.keys.extend(keys)
        last.frames.update(frames)
        _count_angles(last.groups, new)
        first = row_incidence(last.everything[0])
        angle = _default(_by_angle(last.groups), first) if choice is None else choice
        if choice is not None and angle != ALL:
            last.found = last.found or any(same_incidence(row_incidence(row), angle) for row in new)
            if not last.found:
                angle = _default(_by_angle(last.groups), first)
        if angle == last.angle:
            at, other = split_rows(new, angle)
            last.places.update((row_identity(row), place) for place, row in enumerate(at, len(last.kept)))
            last.kept.extend(at)
            last.aside.extend(other)
            _count_angles(last.aside_groups, other)
        else:  # (another angle has most rows now)
            last.kept, last.aside = split_rows(last.everything, angle)
            if not last.kept:
                last.kept, last.aside, angle, last.alone = list(last.everything), [], ALL, True
            last.aside_groups = _count_angles([], last.aside)
            last.places = {row_identity(row): place for place, row in enumerate(last.kept)}
        last.angle, last.numbers = angle, kept_numbers(numbers)
        return last


__all__ = ["ALL", "TOLERANCE_DEG", "AngleSplit", "Split", "default_incidence", "incidence_groups", "incidence_line",
           "row_incidence", "same_incidence", "split_rows"]
