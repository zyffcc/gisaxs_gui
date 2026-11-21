"""The page stack in a scroll area: on a small screen a page scrolls instead of being cut off.

Below the size a page needs (its minimum size hint) Qt squeezed the page anyway: at 1024 × 640 Analyze's
detector image shrank to a strip and the Results text and table were cut off at the right edge. In a scroll
area the page keeps the size it needs and scroll bars appear; when the window is large enough the page fills
it exactly as before.

The stack sits in a holder that asks only the page shown what it needs at least (a stack would answer for its
largest page, shown or not: Start would scroll because Analyze needs more) and never answers with a height for
a width (a page of wrapped texts would then scroll at sizes where it fits by squeezing them, as it always did).
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QFrame, QScrollArea, QSizePolicy, QStackedWidget, QWidget


class PageHolder(QWidget):
    """Holds the stack at its own size; its minimum is the minimum of the page shown."""

    def __init__(self, stack: QStackedWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("pageHolder")
        self.stack = stack
        stack.setParent(self)
        stack.show()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        stack.currentChanged.connect(lambda _index: self.updateGeometry())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        page = self.stack.currentWidget()
        if page is None:
            return QSize(0, 0)
        return page.minimumSizeHint().expandedTo(page.minimumSize()).expandedTo(QSize(0, 0))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return self.minimumSizeHint()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.stack.setGeometry(self.rect())
        super().resizeEvent(event)


def scroll_pages(stack: QStackedWidget) -> QScrollArea:
    """Put ``stack`` into a scroll area where it stood in its parent's layout; returns the scroll area."""
    parent = stack.parentWidget()
    layout = parent.layout() if parent is not None else None
    index = layout.indexOf(stack) if layout is not None else -1
    if layout is not None:
        layout.removeWidget(stack)
    scroll = QScrollArea(parent)
    scroll.setObjectName("pageScrollArea")
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setFocusPolicy(Qt.NoFocus)
    scroll.setWidgetResizable(True)
    scroll.setSizePolicy(stack.sizePolicy())
    scroll.setWidget(PageHolder(stack))
    if layout is not None:
        layout.insertWidget(max(0, index), scroll)
    return scroll


__all__ = ["PageHolder", "scroll_pages"]
