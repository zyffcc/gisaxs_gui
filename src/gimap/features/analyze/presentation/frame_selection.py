"""Compact source-scoped frame selections, independent of widgets and pixel reading.

An absent selection means every frame, including frames appended later. Explicit ranges have a
closed last frame. Indices are physical zero-based indices; selecting a subset never renumbers data.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class IndexRanges(Sequence):
    """A small number of disjoint increasing ranges without expanding their values."""

    def __init__(self, ranges):
        self.ranges = tuple(item for item in ranges if len(item))
        self.ends = []
        total = 0
        for item in self.ranges:
            total += len(item)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            from .frame_sequences import SequenceSlice
            return SequenceSlice(self, index)
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        return self.ranges[part][index - (self.ends[part - 1] if part else 0)]

    def __iter__(self):
        for item in self.ranges:
            yield from item

    def __contains__(self, index):
        return any(index in item for item in self.ranges)

    def index(self, value, start=0, stop=None):
        offset = 0
        for item in self.ranges:
            if value in item:
                position = offset + item.index(value)
                if position >= start and (stop is None or position < stop):
                    return position
            offset += len(item)
        raise ValueError(value)


def _normal_ranges(ranges):
    """Union ranges; adjacent chunks and the common contiguous case stay compact."""
    result = []

    def covers(outer, inner):
        return (inner.start in outer and inner[-1] in outer
                and (len(inner) == 1 or inner.step % outer.step == 0))

    for item in sorted((item for item in ranges if len(item)), key=lambda item: item.start):
        if any(covers(previous, item) for previous in result):
            continue
        split = len(result)
        while split and result[split - 1][-1] >= item.start:
            split -= 1
        if split < len(result):
            cluster = [*result[split:], item]
            first, last = min(part.start for part in cluster), max(part[-1] for part in cluster)
            grid = all(part.step == item.step and (part.start - first) % item.step == 0 for part in cluster)
            connected, edge = True, first - item.step
            for part in sorted(cluster, key=lambda part: part.start):
                connected = connected and part.start <= edge + item.step
                edge = max(edge, part[-1])
            if grid and connected:
                result[split:] = [range(first, last + 1, item.step)]
                continue
            if all(covers(item, part) for part in cluster):
                result[split:] = [item]
                continue
            # Only genuinely irregular overlapping grids need an explicit union.
            values = sorted(set().union(*cluster))
            result[split:] = []
            for value in values:
                if result and result[-1].step == 1 and value == result[-1].stop:
                    result[-1] = range(result[-1].start, value + 1)
                else:
                    result.append(range(value, value + 1))
            continue
        if result and result[-1].step == item.step and item.start == result[-1][-1] + item.step:
            result[-1] = range(result[-1].start, item[-1] + 1, item.step)
            continue
        result.append(item)
    return tuple(result)


def indices_between(indices, first: int, stop: int):
    """Clip physical indices without enumerating a range."""
    ranges = (indices,) if isinstance(indices, range) else indices.ranges
    clipped = []
    for item in ranges:
        start = item.start + max(0, (int(first) - item.start + item.step - 1) // item.step) * item.step
        candidate = range(start, min(item.stop, int(stop)), item.step)
        if len(candidate):
            clipped.append(candidate)
    return clipped[0] if len(clipped) == 1 else IndexRanges(clipped)


@dataclass(frozen=True)
class FrameSelection:
    """A finite selection; ``None`` is used separately for All."""

    ranges: tuple[range, ...]

    def __post_init__(self):
        if any(item.start < 0 or item.step < 1 for item in self.ranges):
            raise ValueError("Frame selection must increase from a non-negative index.")
        object.__setattr__(self, "ranges", _normal_ranges(self.ranges))

    @classmethod
    def range(cls, first: int, last: int, every: int = 1):
        first, last, every = int(first), int(last), int(every)
        if first < 0 or last < first or every < 1:
            raise ValueError("Frame range needs 0 <= first <= last and a positive step.")
        return cls((range(first, last + 1, every),))

    @classmethod
    def explicit(cls, indices):
        values = sorted(set(int(index) for index in indices))
        if any(value < 0 for value in values):
            raise ValueError("Frame indices must be non-negative.")
        return cls(_normal_ranges(range(value, value + 1) for value in values))

    @classmethod
    def from_record(cls, record):
        if record is None or record.get("mode") == "all":
            return None
        if "indices" in record:
            return cls.explicit(record["indices"])
        if "ranges" in record:
            ranges = [cls.range(*item).ranges[0] for item in record["ranges"]]
            return cls(_normal_ranges(ranges))
        return cls.range(record["first"], record["last"], record.get("every", 1))

    def to_record(self):
        if len(self.ranges) == 1:
            item = self.ranges[0]
            return {"first": item.start, "last": item[-1], "every": item.step}
        return {"ranges": [[item.start, item[-1], item.step] for item in self.ranges]}

    def indices(self, count: int):
        ranges = [range(item.start, min(item.stop, max(0, int(count))), item.step) for item in self.ranges]
        return ranges[0] if len(ranges) == 1 else IndexRanges(ranges)

    def shifted(self, offset: int):
        return FrameSelection(tuple(range(item.start + offset, item.stop + offset, item.step) for item in self.ranges))

    def union(self, other):
        return FrameSelection(_normal_ranges((*self.ranges, *other.ranges)))


class ImportSelectionMixin:
    """Needs source counts, state.files, and the source regrouping helpers."""

    def import_selection(self, path):
        return self._import_selections.get(str(path).casefold())

    def set_import_selection(self, path, selection) -> None:
        if isinstance(selection, dict):
            selection = FrameSelection.from_record(selection)
        elif isinstance(selection, range):
            if selection.step < 1 or selection.start < 0:
                raise ValueError("Frame selection must increase from a non-negative index.")
            selection = FrameSelection((selection,))
        elif selection is not None and not isinstance(selection, FrameSelection):
            selection = FrameSelection.explicit(selection)
        count = self.known_frame_count(Path(path))
        if selection is not None and any(item[-1] >= count for item in selection.ranges if len(item)):
            raise ValueError(f"Frame selection exceeds the source's {count} frames.")
        key = str(path).casefold()
        if selection is None:
            self._import_selections.pop(key, None)
        else:
            self._import_selections[key] = selection
        self._watch_pending = [frame for frame in self._watch_pending if self.frame_is_imported(*frame)]

    def selected_frame_indices(self, path, *, usable: bool = False):
        count = self._usable_frames(Path(path)) if usable else self.known_frame_count(Path(path))
        selection = self.import_selection(path)
        return range(count) if selection is None else selection.indices(count)

    def frame_is_imported(self, path, index) -> bool:
        return int(index) in self.selected_frame_indices(path)

    def imported_frame_count(self) -> int:
        return sum(len(self.selected_frame_indices(path)) for path in self.state.files)

    def total_frame_count(self) -> int:
        """All physical frames, including those outside an import selection; no disk access."""
        return sum(self._frame_counts.get(str(path).casefold(), 1) for path in self.state.files)

    def iter_imported_frames(self):
        for path in self.state.files:
            for index in self.selected_frame_indices(path):
                yield Path(path), index

    def export_import_selections(self) -> dict:
        return {str(path): selected.to_record() for path in self.state.files
                if (selected := self.import_selection(path)) is not None}

    def apply_import_selections(self, record: dict) -> list[str]:
        """Restore selections of present listed sources; old projects without this field mean All."""
        self._import_selections.clear()
        paths = {str(path).casefold(): path for path in self.state.files}
        notes = []
        for name, selected in dict(record or {}).items():
            path = paths.get(str(name).casefold())
            if path is None:
                continue
            try:
                self.set_import_selection(path, selected)
            except (ValueError, KeyError, TypeError) as exc:
                notes.append(f"Frame selection for {path.name} could not be restored: {exc}")
        return notes

    def _migrate_import_selection(self, before, now, *, merge: bool = False, restrict: bool = False) -> None:
        """A chunk becomes part of a master, or the canonical module path changes."""
        old_key, new_key = str(before).casefold(), str(now).casefold()
        selection = self._import_selections.pop(old_key, None)
        layout = self.describe(now)
        is_master = layout is not None and layout.kind == "chunked_master"
        if not is_master:
            if merge:
                existing = self._import_selections.get(new_key)
                if selection is None or existing is None:
                    self._import_selections.pop(new_key, None)
                else:
                    self._import_selections[new_key] = existing.union(selection)
            elif selection is not None:
                self._import_selections[new_key] = selection
            return
        if selection is None and not merge and not restrict:
            return  # sources imported as All remain All when their complete master arrives
        count = self._frame_counts.get(old_key, 1)
        if selection is None:
            selection = FrameSelection((range(count),))
        offset = self._frame_within(Path(now), Path(before), 0)
        selection = selection.shifted(offset)
        existing = self._import_selections.get(new_key)
        if merge and existing is None:
            return  # an explicitly listed master already includes every frame
        self._import_selections[new_key] = selection if existing is None else existing.union(selection)


__all__ = ["FrameSelection", "ImportSelectionMixin", "IndexRanges", "indices_between"]
