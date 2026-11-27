"""Virtual Data rows: one lightweight reference per frame, with no image or per-row Qt item.

The model is a string list only for its row count, which then stays in C++: Qt's list view lays out all its rows
whenever rows are added, and asks the model for the index of each row twice, each asking the row count. Answered from
Python, that was 100 000 calls into the interpreter at 50 000 frames, 49 ms of the window at every look of a folder
watch (2 ms in C++). The list holds one empty string per row; what a row shows comes from ``data``, and no row is
edited, dragged or dropped as a string list's rows can be (``flags``).
"""
from bisect import bisect_right
from PyQt5.QtCore import QAbstractListModel, QStringListModel, Qt
from src.gimap.app.presentation.i18n import trf


FRAME_ROW = "{frame} · {name}"
FRAME_DETAIL = "Frame {frame} of {total}"


class FrameListingModel(QStringListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.spans, self.ends, self._where = [], [], {}
        self._rows = 0
        """Rows in all (as many as the strings of the list; read from Python without asking Qt)."""

    def reset_sources(self, spans):
        self.spans, self.ends, self._where = [], [], {}
        self._rows = 0
        self._add(spans)
        self.setStringList([""] * self._rows)  # (a reset of the model, its rows as they are now)

    def append_sources(self, spans):
        """``reset_sources`` of the sources before and ``spans`` after them, as rows added at the end (a folder
        watch lists a few sources at every look)."""
        kept = [span for span in spans if len(span[1])]
        if not kept:
            return
        first = self._rows
        self._add(kept)
        self.insertRows(first, self._rows - first)  # (rows inserted at the end, their spans already listed)

    def flags(self, index):  # noqa: N802
        """A list's rows: selectable, never edited (a double-click shows the frame), dragged or dropped."""
        return QAbstractListModel.flags(self, index)

    def _add(self, spans):
        total = self.ends[-1] if self.ends else 0
        for path, indices, name, count, tip in spans:
            if len(indices):
                self._where.setdefault(str(path).casefold(), len(self.spans))
                total += len(indices)
                self.spans.append((path, indices, name, count, tip))
                self.ends.append(total)
        self._rows = total

    def location(self, row):
        if not 0 <= row < self._rows:
            raise IndexError(row)
        span = bisect_right(self.ends, row)
        offset = row - (self.ends[span - 1] if span else 0)
        return span, offset

    def ref(self, row):
        span, offset = self.location(row)
        path, indices, _name, _count, _tip = self.spans[span]
        return path, indices[offset]

    def row_for(self, path, frame):
        span = self._where.get(str(path).casefold())
        if span is None:
            return -1
        before = self.ends[span - 1] if span else 0
        indices = self.spans[span][1]
        if isinstance(indices, range):
            return before + (frame - indices.start) // indices.step if frame in indices else -1
        try:
            return before + indices.index(frame)
        except ValueError:
            return -1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= self._rows:
            return None
        span, offset = self.location(index.row())
        path, indices, name, count, tip = self.spans[span]
        frame = indices[offset]
        if role == Qt.DisplayRole:
            return trf(FRAME_ROW, name=name, frame=frame + 1, total=count) if count > 1 else name
        if role == Qt.ToolTipRole:
            return f"{tip}\n" + trf(FRAME_DETAIL, frame=frame + 1, total=count)
        if role == Qt.UserRole:
            return path, frame
        return None

    def language_changed(self):
        if self._rows:
            self.dataChanged.emit(self.index(0), self.index(self._rows - 1))
