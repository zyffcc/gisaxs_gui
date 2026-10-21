"""A sparse filter over lazy batch requests: excluding a few odd rows must not create every request."""

from bisect import bisect_right
from collections.abc import Sequence


class ExcludedFrames(Sequence):
    def __init__(self, requests, excluded):
        self.requests = requests
        self.parts, self.ends = [], []
        start, total = 0, 0
        for stop in [*sorted({int(index) for index in excluded if 0 <= index < len(requests)}), len(requests)]:
            if stop > start:
                self.parts.append(range(start, stop))
                total += stop - start
                self.ends.append(total)
            start = stop + 1

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        before = self.ends[part - 1] if part else 0
        return self.requests[self.parts[part][index - before]]


__all__ = ["ExcludedFrames"]
