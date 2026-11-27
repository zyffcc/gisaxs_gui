"""An array that grows by rows in place: a live Series map and its picture add the rows that arrive.

Copying all rows for every few that arrive (``np.vstack``) costs as much as the rows already there: for a map of
20 000 rows of 1 000 points, 80 MB at every drawing. Here the rows go after the ones before, into room kept free
(``GROW``); a view of the first rows is the array so far. A view handed out before keeps showing the rows it had:
rows are only ever written after them, and when the room runs out the rows move to a larger array, the old one
staying with the views of it.
"""

from __future__ import annotations

import numpy as np

GROW = 1.5
"""When the room is full, the array grows to this many times its rows (and at least to the rows needed)."""


class GrowingRows:
    """The rows of ``first`` and the rows added after (``extend``); ``view()``: all of them as one array."""

    def __init__(self, first: np.ndarray):
        self._array = first
        self.rows = int(first.shape[0])

    def view(self) -> np.ndarray:
        return self._array[: self.rows]

    def extend(self, more: np.ndarray) -> np.ndarray:
        """Add ``more`` (rows of the same columns) after the rows so far; returns the view of all rows."""
        more = np.asarray(more)
        need = self.rows + int(more.shape[0])
        if need > self._array.shape[0]:
            room = max(need, int(GROW * self._array.shape[0]) + 1)
            bigger = np.empty((room, *self._array.shape[1:]), dtype=self._array.dtype)
            bigger[: self.rows] = self._array[: self.rows]
            self._array = bigger
        self._array[self.rows: need] = more
        self.rows = need
        return self.view()


__all__ = ["GROW", "GrowingRows"]
