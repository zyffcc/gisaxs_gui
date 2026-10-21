"""A batch queue that constructs a frame request only when a worker needs it."""
from collections import deque
from collections.abc import Mapping, Sequence


class PendingFrames:
    def __init__(self, requests):
        self.requests = requests
        self._next = 0
        self._retries = deque()

    def __len__(self):
        return len(self.requests) - self._next + len(self._retries)

    def __iter__(self):
        yield from self._retries
        for index in range(self._next, len(self.requests)):
            yield index, self.requests[index]

    def pop(self, index=0):
        if index != 0:
            raise IndexError("A pending batch takes its next frame from the front.")
        if self._retries and (self._next >= len(self.requests) or self._retries[0][0] < self._next):
            return self._retries.popleft()
        if self._next >= len(self.requests):
            raise IndexError("No pending frame.")
        current = self._next
        self._next += 1
        return current, self.requests[current]

    def append(self, entry):
        self._retries.append(entry)

    def sort(self, *, key=None):
        self._retries = deque(sorted(self._retries, key=key))

    def names(self, naming):
        return PendingNames(self.requests, self._next, tuple(self._retries), naming)


class PendingNames(Sequence):
    def __init__(self, requests, first, retries, naming):
        self.requests, self.first, self.retries, self.naming = requests, first, retries, naming
        self._extras = []

    def __len__(self):
        return len(self.retries) + len(self.requests) - self.first + len(self._extras)

    def extend(self, names):
        self._extras.extend(names)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ordinary = len(self.retries) + len(self.requests) - self.first
        if index >= ordinary:
            return self._extras[index - ordinary]
        request = (self.retries[index][1] if index < len(self.retries)
                   else self.requests[self.first + index - len(self.retries)])
        return self.naming(request)


class LazyBatchNumbers(Mapping):
    def __init__(self, requests, numbers):
        self.requests, self.numbers = requests, numbers

    def __len__(self):
        return len(self.requests)

    def __iter__(self):
        return iter(range(len(self)))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise KeyError(index)
        request = self.requests[index]
        return self.numbers.get((str(request.path).casefold(), int(request.frame_index)))
