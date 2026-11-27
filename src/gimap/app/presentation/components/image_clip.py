"""A large picture given to the image item a part at a time: the part in view and as much again around it.

pyqtgraph's image item colours the whole picture it holds whenever it draws it anew — a new picture, or another
down-sampling factor after a zoom — however little of it is in view. A Series map of 50 000 rows of 1 000 points
took the window 0.4 s at each of them once zoomed in to a few thousand rows (whole in view, down-sampled: 17 ms):
every redraw of a growing map, every zoom step. A picture of more than ``CLIP_CELLS`` cells is therefore given to
the item cut to the view and as much again on each side (``ImageClip``), at multiples of the item's down-sampling
factors: its blocks are those of the whole picture, so what is drawn is pixel for pixel what the whole picture gives
(``tests/test_image_clip.py``; a picture with NaN is coloured as one, even where the part holds none:
``detector_display.NanSafeImageItem``). The part follows the view (zoom, pan, a resize) and the whole picture stays
for what needs all of it: the limits, the histogram of the colour bar, the figure export, the view of everything.

Display only: the picture is the displayed copy (``display_image``); no value is changed or left out of a record.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QTimer

CLIP_CELLS = 4_000_000
"""Pictures of more cells are given to the image item a part at a time (a detector frame of 4 megapixels or less is
given whole, as ever)."""
FOLLOW_MS = 30
"""After the view moved: how soon the part is cut again (a zoom or pan of many steps cuts it once)."""


def _span(low: float, high: float, factor: int, size: int) -> tuple[int, int]:
    """``[low, high)`` of an axis of ``size`` cells, widened to multiples of ``factor`` (down-sampled blocks of
    the whole axis) and kept within it. An axis of no more than one block is given whole (as the whole picture
    gives it); a view beside the picture gets the block nearest to it."""
    if size <= factor:
        return 0, size
    first = min(max(0, math.floor(low / factor) * factor), (size - 1) // factor * factor)
    last = max(first + factor, min(size, math.ceil(high / factor) * factor))
    return first, min(size, last)


class ImageClip:
    """The whole displayed picture of a view, given to ``item`` (in ``box``) as the part around the view."""

    def __init__(self, item, box) -> None:
        self.item, self.box = item, box
        self.whole: Optional[np.ndarray] = None
        self.rect: Optional[QRectF] = None
        self.part: Optional[tuple] = None
        """``(row0, row1, column0, column1, (fx, fy))`` given to the item; ``None``: the whole picture."""
        self._timer = QTimer(box)
        self._timer.setSingleShot(True)
        self._timer.setInterval(FOLLOW_MS)
        self._timer.timeout.connect(self.follow)
        box.sigRangeChanged.connect(lambda *_args: self._timer.start())
        box.sigResized.connect(lambda *_args: self._timer.start())

    @property
    def cut(self) -> bool:
        return self.part is not None

    def show(self, whole: np.ndarray, rect: QRectF, levels) -> None:
        """Give the item ``whole`` (placed at ``rect``) — the part around the view when it is large."""
        self.whole, self.rect = whole, QRectF(rect)
        self.item.full_image = whole if whole.size > CLIP_CELLS else None
        # pyqtgraph colours a picture with NaN another way than one without: the part as the whole would be.
        self.item.full_image_nans = whole.size > CLIP_CELLS and whole.dtype.kind == "f" and bool(np.isnan(whole.min()))
        if whole.size <= CLIP_CELLS:
            self.part = None
            self.item.setImage(whole, autoLevels=False, levels=levels)
            self.item.setRect(self.rect)
            return
        self._give(self._wanted(), levels)

    def clear(self) -> None:
        self.whole = self.rect = self.part = None
        self.item.full_image, self.item.full_image_nans = None, False
        self._timer.stop()

    def follow(self) -> None:
        """The view moved: a part that no longer covers it (with half its span on each side), or of other
        down-sampling factors, is cut again."""
        if self.whole is None or self.whole.size <= CLIP_CELLS:
            return
        wanted = self._wanted()
        if self.part is not None and self.part[4] == wanted[4] and self._covers(self.part, 0.5):
            return
        self._give(wanted, self.item.getLevels())

    # -- the part ------------------------------------------------------------------------------------------------

    def _view(self) -> tuple[float, float, float, float]:
        """The view in rows and columns of the whole picture: ``(row0, row1, column0, column1)``."""
        (x0, x1), (y0, y1) = self.box.viewRange()
        rows, columns = self.whole.shape[:2]
        rect = self.rect
        if rect.height() == 0 or rect.width() == 0:
            return 0.0, float(rows), 0.0, float(columns)
        r0, r1 = sorted(((y0 - rect.y()) / rect.height() * rows, (y1 - rect.y()) / rect.height() * rows))
        c0, c1 = sorted(((x0 - rect.x()) / rect.width() * columns, (x1 - rect.x()) / rect.width() * columns))
        return r0, r1, c0, c1  # (a picture placed upside down: its rows the other way)

    def _factors(self) -> tuple[int, int]:
        """The item's down-sampling factors at this view (pyqtgraph's ``_computeDownsampleFactors``: screen pixels
        per picture pixel along each axis)."""
        rows, columns = self.whole.shape[:2]
        rect = self.rect
        origin = self.box.mapViewToDevice(QPointF(rect.x(), rect.y()))
        across = self.box.mapViewToDevice(QPointF(rect.x() + rect.width() / columns, rect.y()))
        down = self.box.mapViewToDevice(QPointF(rect.x(), rect.y() + rect.height() / rows))
        if origin is None or across is None or down is None:
            return 1, 1
        width = math.hypot(across.x() - origin.x(), across.y() - origin.y())
        height = math.hypot(down.x() - origin.x(), down.y() - origin.y())
        if width == 0 or height == 0:
            return 1, 1
        return max(1, int(1.0 / width)), max(1, int(1.0 / height))

    def _wanted(self, factors: Optional[tuple[int, int]] = None) -> tuple:
        """The view and its span again on each side, cut at multiples of the factors, within the picture."""
        fx, fy = self._factors() if factors is None else factors
        r0, r1, c0, c1 = self._view()
        rows, columns = self.whole.shape[:2]
        span_r, span_c = r1 - r0, c1 - c0
        low_r, high_r = _span(r0 - span_r, r1 + span_r, fy, rows)
        low_c, high_c = _span(c0 - span_c, c1 + span_c, fx, columns)
        return low_r, high_r, low_c, high_c, (fx, fy)

    def _covers(self, part: tuple, margin: float) -> bool:
        r0, r1, c0, c1 = self._view()
        rows, columns = self.whole.shape[:2]
        span_r, span_c = (r1 - r0) * margin, (c1 - c0) * margin
        return (part[0] <= max(0.0, r0 - span_r) and part[1] >= min(rows, r1 + span_r)
                and part[2] <= max(0.0, c0 - span_c) and part[3] >= min(columns, c1 + span_c))

    def _give(self, part: tuple, levels, again: bool = True) -> None:
        low_r, high_r, low_c, high_c, factors = part
        rows, columns = self.whole.shape[:2]
        rect = self.rect
        self.part = part
        self.item.setImage(self.whole[low_r:high_r, low_c:high_c], autoLevels=False, levels=levels)
        self.item.setRect(QRectF(rect.x() + low_c * rect.width() / columns, rect.y() + low_r * rect.height() / rows,
                                 (high_c - low_c) * rect.width() / columns, (high_r - low_r) * rect.height() / rows))
        drawn = self.item._computeDownsampleFactors()
        if again and drawn[0] is not None and tuple(drawn) != factors:  # (a rounding apart: the item's own factors)
            self._give(self._wanted(tuple(drawn)), levels, again=False)


__all__ = ["CLIP_CELLS", "FOLLOW_MS", "ImageClip"]
