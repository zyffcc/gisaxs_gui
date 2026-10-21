"""What stands in the band along the bottom edge of a window, where its notices lie (``toast.py``).

A page keeps its status line there: the text at the left, and while a job runs a progress bar and Cancel at
the right end. ``band_content`` looks into the widgets of the page shown for what lies in the rows of the
band — only what is drawn there: a widget scrolled out of its scroll area is not — and tells

* where the status text ends: the leftmost label in the band with text (or a status bar's message);
* where the controls at the right end begin, and how high they reach: the progress bars and buttons that
  stand one beside the other at the right end of the band (a gap wider than ``GAP`` ends them; a button
  elsewhere in the band is no part of them). The notices of another widget of the same window that lie
  there (``obstacles``: a page shows some of its notices itself) count as such controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtWidgets import QAbstractButton, QLabel, QProgressBar, QStatusBar, QWidget

GAP = 32
"""Pixels between two controls of the right end at most (the page's margin, the spacing of a row)."""


@dataclass(frozen=True)
class BandContent:
    text_end: Optional[int] = None
    """x (in the window) where the status text ends; ``None``: no text in the band."""
    text: str = ""
    """The status text itself."""
    controls_left: Optional[int] = None
    """x where the progress bar / buttons at the right end of the band begin; ``None``: none there."""
    controls_top: Optional[int] = None
    """y of the top of those controls."""


def _drawn_rect(widget: QWidget, parent: QWidget) -> QRect:
    """``widget`` in ``parent``'s coordinates, cut to what its ancestors show of it (a scroll area's view)."""
    rect = QRect(widget.mapTo(parent, QPoint(0, 0)), widget.size())
    ancestor = widget.parentWidget()
    while ancestor is not None and ancestor is not parent and not rect.isEmpty():
        rect &= QRect(ancestor.mapTo(parent, QPoint(0, 0)), ancestor.size())
        ancestor = ancestor.parentWidget()
    return rect


def _text_end(label: QLabel, rect: QRect) -> int:
    """Where the text of ``label`` ends: a left-aligned text as wide as its longest line, else the label."""
    if int(label.alignment()) & int(Qt.AlignRight | Qt.AlignHCenter | Qt.AlignJustify):
        return rect.right()
    widest = max((label.fontMetrics().horizontalAdvance(line) for line in label.text().split("\n")), default=0)
    margins = label.contentsMargins()
    return min(rect.right(), rect.left() + margins.left() + 2 * label.margin() + max(0, label.indent()) + widest)


def band_content(roots: Iterable[Optional[QWidget]], parent: QWidget, top: int, bottom: int,
                 obstacles: Iterable[QRect] = ()) -> BandContent:
    """What the widgets of ``roots`` (the page shown, a status bar) show in the rows ``top`` … ``bottom`` of
    ``parent``; ``obstacles``: other rectangles there (in ``parent``'s coordinates) the row keeps off as it
    keeps off the controls. A notice's own label and buttons are no content of the band."""
    texts: list[tuple[int, int, str]] = []
    controls: list[QRect] = [rect for rect in obstacles if not rect.isEmpty() and rect.bottom() >= top and rect.top() <= bottom]
    for root in roots:
        if root is None or not root.isVisible() or not parent.isAncestorOf(root):
            continue
        if isinstance(root, QStatusBar) and root.currentMessage():
            rect = _drawn_rect(root, parent)
            if rect.bottom() >= top and rect.top() <= bottom:
                end = rect.left() + 8 + root.fontMetrics().horizontalAdvance(root.currentMessage())
                texts.append((rect.left(), min(rect.right(), end), root.currentMessage()))
        for widget in root.findChildren((QLabel, QProgressBar, QAbstractButton)):
            holder = widget.parentWidget()
            if not widget.isVisible() or (holder is not None and holder.property("gimapToast")):
                continue
            rect = _drawn_rect(widget, parent)
            if rect.isEmpty() or rect.bottom() < top or rect.top() > bottom:
                continue
            if isinstance(widget, QLabel):
                if widget.text():
                    texts.append((rect.left(), _text_end(widget, rect), widget.text()))
            else:
                controls.append(rect)
    status = min(texts) if texts else (0, None, "")
    left = top_of = None
    boundary = parent.width()
    for rect in sorted(controls, key=lambda found: found.right(), reverse=True):
        if rect.right() < boundary - GAP:
            break
        boundary = min(boundary, rect.left())
        left, top_of = boundary, rect.top() if top_of is None else min(top_of, rect.top())
    return BandContent(status[1], status[2], left, top_of)


__all__ = ["BandContent", "GAP", "band_content"]
