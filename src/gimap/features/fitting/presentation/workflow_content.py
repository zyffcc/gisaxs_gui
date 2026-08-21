"""Current-step sizing for the fitting workflow's left content rail."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


class WorkflowContentStack(QWidget):
    """Show one task and exclude hidden tasks from layout size calculation.

    ``QStackedWidget`` deliberately reports the largest page's minimum size.
    That behaviour is useful for dialog stacks, but it created a 1300 px-tall
    workflow rail when the much smaller Import page was visible.  A regular
    layout excludes hidden widgets, so the scroll canvas follows the current
    task without fixed-height workarounds.
    """

    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages: list[QWidget] = []
        self._current_index = -1
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        # Fixed here means "use the live sizeHint", not a hard-coded pixel
        # height.  The hint changes with the selected task and its visible data.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def addWidget(self, page: QWidget) -> int:
        page.setParent(self)
        index = len(self._pages)
        self._pages.append(page)
        self._layout.addWidget(page)
        if self._current_index == -1:
            self._current_index = 0
            page.show()
        else:
            page.hide()
        self.updateGeometry()
        return index

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int) -> QWidget | None:
        return self._pages[index] if 0 <= index < len(self._pages) else None

    def currentIndex(self) -> int:
        return self._current_index

    def currentWidget(self) -> QWidget | None:
        return self.widget(self._current_index)

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._pages) or index == self._current_index:
            return
        current = self.currentWidget()
        if current is not None:
            current.hide()
        self._current_index = index
        self._pages[index].show()
        self._layout.invalidate()
        self._layout.activate()
        self.updateGeometry()
        self.currentChanged.emit(index)

    def sync_height(self) -> None:
        current = self.currentWidget()
        if current is None:
            return
        if current.layout() is not None:
            current.layout().activate()
        self._layout.invalidate()
        self._layout.activate()
        self.updateGeometry()

    def sizeHint(self):
        current = self.currentWidget()
        if current is None:
            return super().sizeHint()
        hint = current.sizeHint()
        if current.objectName() == "fittingFitStepPage":
            hint.setHeight(current.minimumSizeHint().height())
        return hint

    def minimumSizeHint(self):
        current = self.currentWidget()
        return current.minimumSizeHint() if current is not None else super().minimumSizeHint()


__all__ = ["WorkflowContentStack"]
