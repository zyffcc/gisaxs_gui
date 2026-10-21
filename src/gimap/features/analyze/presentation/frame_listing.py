"""Virtual Data rows: one lightweight reference per frame, with no image or per-row Qt item."""
from bisect import bisect_right
from PyQt5.QtCore import QAbstractListModel, QModelIndex, Qt
from src.gimap.app.presentation.i18n import trf


FRAME_ROW = "{frame} · {name}"
FRAME_DETAIL = "Frame {frame} of {total}"


class FrameListingModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.spans, self.ends = [], []

    def reset_sources(self, spans):
        self.beginResetModel()
        self.spans, self.ends = [], []
        total = 0
        for path, indices, name, count, tip in spans:
            if len(indices):
                total += len(indices)
                self.spans.append((path, indices, name, count, tip))
                self.ends.append(total)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else (self.ends[-1] if self.ends else 0)

    def location(self, row):
        if not 0 <= row < self.rowCount():
            raise IndexError(row)
        span = bisect_right(self.ends, row)
        offset = row - (self.ends[span - 1] if span else 0)
        return span, offset

    def ref(self, row):
        span, offset = self.location(row)
        path, indices, _name, _count, _tip = self.spans[span]
        return path, indices[offset]

    def row_for(self, path, frame):
        before = 0
        for candidate, indices, _name, _count, _tip in self.spans:
            if str(candidate).casefold() == str(path).casefold():
                if isinstance(indices, range):
                    return before + (frame - indices.start) // indices.step if frame in indices else -1
                try:
                    return before + indices.index(frame)
                except ValueError:
                    return -1
            before += len(indices)
        return -1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= self.rowCount():
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
        if self.rowCount():
            self.dataChanged.emit(self.index(0), self.index(self.rowCount() - 1))
