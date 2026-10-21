"""Lazy frame references, consecutive sum groups, and original frame numbering.

These sequences retain source metadata only. A million-frame stack requires one range, and an
AnalysisRequest is made only when the caller actually asks for that processing item.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .frame_selection import indices_between


def _position(index, length):
    index = int(index)
    if index < 0:
        index += length
    if not 0 <= index < length:
        raise IndexError(index)
    return index


class SequenceSlice(Sequence):
    """A view of any sequence; repeated slicing composes indices instead of copying items."""

    def __init__(self, values, selection):
        if isinstance(values, SequenceSlice):
            self.values, self.indices = values.values, values.indices[selection]
        else:
            self.values, self.indices = values, range(len(values))[selection]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SequenceSlice(self, index)
        return self.values[self.indices[_position(index, len(self))]]


class FrameSequence(Sequence):
    """Physical indices of each source, compressed as ranges, in canonical source order."""

    def __init__(self, spans):
        self.spans = tuple((Path(path), indices) for path, indices in spans if len(indices))
        self.ends = []
        total = 0
        for _path, indices in self.spans:
            total += len(indices)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SequenceSlice(self, index)
        index = _position(index, len(self))
        part = bisect_right(self.ends, index)
        path, indices = self.spans[part]
        return path, int(indices[index - (self.ends[part - 1] if part else 0)])

    def __iter__(self):
        for path, indices in self.spans:
            for index in indices:
                yield path, int(index)

    def within(self, numbers, first: int, last):
        spans = []
        for path, indices in self.spans:
            offset, count = numbers.sources[str(path).casefold()]
            low, high = max(0, first - 1 - offset), min(count, (last - offset) if last else count)
            spans.append((path, indices_between(indices, low, high)))
        return FrameSequence(spans)


class FrameNumbers(Mapping):
    """Original physical sequence numbers, without one dictionary entry per frame."""

    def __init__(self, paths_counts):
        self.sources = {}
        self.total = 0
        for path, count in paths_counts:
            count = max(0, int(count))
            self.sources[str(path).casefold()] = self.total, count
            self.total += count

    def __len__(self):
        return self.total

    def __getitem__(self, key):
        path, index = key
        offset, count = self.sources[str(path).casefold()]
        if not 0 <= int(index) < count:
            raise KeyError(key)
        return offset + int(index) + 1

    def __iter__(self):
        for path, (_offset, count) in self.sources.items():
            for index in range(count):
                yield path, index


class SumGroups(Sequence):
    """Lazy groups of adjacent selected frames; gaps close a group instead of summing across them."""

    def __init__(self, frames: FrameSequence, size: int, counts):
        self.segments, self.ends, self.size = [], [], size
        total = 0
        single_spans, single_key, previous_number = [], None, None
        counts = tuple(counts)
        numbers = FrameNumbers(counts)
        counts = {str(path).casefold(): count for path, count in counts}

        def add(spans, group_size):
            nonlocal total
            values = FrameSequence(spans)
            if len(values):
                groups = (len(values) + group_size - 1) // group_size
                total += groups
                self.segments.append((values, group_size))
                self.ends.append(total)

        for path, indices in frames.spans:
            single = counts.get(str(path).casefold(), 1) <= 1
            if single:
                number, key = numbers[(str(path).casefold(), 0)], path.suffix.lower()
                if single_spans and (single_key != key or number != previous_number + 1):
                    add(single_spans, size)
                    single_spans = []
                single_spans.append((path, indices))
                single_key, previous_number = key, number
                continue
            add(single_spans, size)
            single_spans = []
            ranges = (indices,) if isinstance(indices, range) else indices.ranges
            pending, pending_size = [], size
            for item in ranges:
                if pending and pending[-1][1][-1] + 1 == item.start:
                    if item.step == 1:
                        pending.append((path, item))
                        pending_size = size
                        continue
                    # A sparse range may start directly after the previous range's last point.
                    pending.append((path, item[:1]))
                    add(pending, size)
                    pending = []
                    item = item[1:]
                else:
                    add(pending, pending_size)
                    pending = []
                if not len(item):
                    continue
                if item.step == 1:
                    pending, pending_size = [(path, item)], size
                else:
                    # Sparse points are isolated except for a possible connection at the next boundary.
                    add([(path, item[:-1])], 1)
                    pending, pending_size = [(path, item[-1:])], 1
            add(pending, pending_size)
        add(single_spans, size)

    def summary(self):
        """Counts and short-group sizes, calculated from segments only."""
        frames, short, short_count, short_last = 0, [], 0, None
        for values, size in self.segments:
            frames += len(values)
            tail = len(values) % size
            if tail:
                short.append(tail)
                short_count += 1
                short_last = tail
            elif size == 1 and self.size > 1:
                # A sparse segment has physically isolated one-frame groups.
                short.append(1)
                short_count += len(values)
                short_last = 1
        return {"frames": frames, "groups": len(self), "short": short,
                "short_count": short_count, "short_last": short_last}

    def tail(self, requested_size):
        if not self.segments:
            return ()
        values, size = self.segments[-1]
        remaining = len(values) % size
        if size != requested_size or not remaining:
            return ()
        return tuple(values[-remaining:])

    def index_for(self, path, frame_index):
        """The group containing one physical frame, using metadata rather than request enumeration."""
        key = str(path).casefold()
        previous = 0
        for segment, (values, size) in enumerate(self.segments):
            offset = 0
            for source, indices in values.spans:
                if str(source).casefold() == key and int(frame_index) in indices:
                    return previous + (offset + indices.index(int(frame_index))) // size
                offset += len(indices)
            previous = self.ends[segment]
        return None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SequenceSlice(self, index)
        index = _position(index, len(self))
        part = bisect_right(self.ends, index)
        values, size = self.segments[part]
        local = index - (self.ends[part - 1] if part else 0)
        return values[local * size:(local + 1) * size]


class BatchRequests(Sequence):
    """Requests generated from a frozen setup; later editing cannot change a queued frame's settings."""

    def __init__(self, groups, prototype, cut, apply_cut):
        self.groups, self.prototype, self.cut, self.apply_cut = groups, prototype, cut, apply_cut

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SequenceSlice(self, index)
        group = self.groups[index]
        path, frame_index = group[0]
        return self.apply_cut(replace(self.prototype, path=path, frame_index=frame_index,
                                      summed_frames=tuple(group[1:])), self.cut)


class BatchGroups(Sequence):
    def __init__(self, groups, size, factory):
        self.groups, self.size, self.factory = groups, size, factory

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SequenceSlice(self, index)
        group = self.groups[index]
        return self.factory(group[0], len(group), self.size)


__all__ = ["BatchGroups", "BatchRequests", "FrameNumbers", "FrameSequence", "SequenceSlice", "SumGroups"]
