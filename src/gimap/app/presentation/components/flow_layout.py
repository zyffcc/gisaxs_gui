"""A layout that places widgets in a row and wraps to the next row when the row is full.

Toolbars and button rows use it so a narrow panel shows every control whole, on two lines,
instead of cutting texts. ``addStretch`` is accepted (and ignored) so it can replace a
``QHBoxLayout`` in code written for one.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget, QWidgetItem


class FlowLayout(QLayout):
    def __init__(self, parent: Optional[QWidget] = None, *, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # -- QHBoxLayout-like API ----------------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt API
        self._items.append(item)

    def insertWidget(self, index: int, widget: QWidget) -> None:  # noqa: N802 - Qt API
        self.addChildWidget(widget)
        self._items.insert(max(0, min(int(index), len(self._items))), QWidgetItem(widget))
        self.invalidate()

    def addStretch(self, _stretch: int = 0) -> None:  # noqa: N802 - accepted for QHBoxLayout callers
        return None

    def addSpacing(self, _size: int) -> None:  # noqa: N802
        return None

    # -- QLayout -------------------------------------------------------------------------

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802 - Qt API
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt API
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        width = sum(item.sizeHint().width() for item in self._visible()) + self.spacing() * max(0, len(self._visible()) - 1)
        height = max((item.sizeHint().height() for item in self._visible()), default=0)
        margins = self.contentsMargins()
        return QSize(width + margins.left() + margins.right(), height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _visible(self) -> list[QLayoutItem]:
        return [item for item in self._items if not (item.widget() is not None and item.widget().isHidden())]

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line = area.x(), area.y(), 0
        spacing = self.spacing()
        for item in self._visible():
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line > 0:
                x, y, line = area.x(), y + line + spacing, 0
            if apply:
                height = hint.height()
                policy = item.widget().sizePolicy() if item.widget() is not None else None
                if policy is not None and policy.verticalPolicy() == QSizePolicy.Fixed:
                    height = hint.height()
                item.setGeometry(QRect(QPoint(x, y), QSize(hint.width(), height)))
            x += hint.width() + spacing
            line = max(line, hint.height())
        return y + line - rect.y() + margins.bottom()


__all__ = ["FlowLayout"]
