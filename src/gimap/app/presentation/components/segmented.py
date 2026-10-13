"""Segmented control: a row of exclusive buttons for a few mutually exclusive choices.

Its API mirrors the part of ``QComboBox`` pages use (``addItem``,
``findData``, ``currentData``, ``setCurrentIndex``, ``activated``), so it
replaces a small combo box without changing the page logic.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QButtonGroup, QHBoxLayout, QSizePolicy, QToolButton, QWidget


class SegmentedControl(QWidget):
    activated = pyqtSignal(int)
    """Emitted when the user picks a segment (not for ``setCurrentIndex``)."""
    currentIndexChanged = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("segmented", True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonClicked.connect(self._clicked)
        self._buttons: list[QToolButton] = []
        self._data: list[Any] = []

    def addItem(self, text: str, data: Any = None) -> None:
        button = QToolButton(self)
        button.setText(text)
        button.setCheckable(True)
        button.setAutoRaise(False)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._group.addButton(button, len(self._buttons))
        self._buttons.append(button)
        self._data.append(data if data is not None else text)
        self._layout.addWidget(button)
        self._update_positions()
        if len(self._buttons) == 1:
            button.setChecked(True)

    def addItems(self, texts) -> None:
        for text in texts:
            self.addItem(text)

    def count(self) -> int:
        return len(self._buttons)

    def button(self, index: int) -> QToolButton:
        return self._buttons[index]

    def itemText(self, index: int) -> str:
        return self._buttons[index].text()

    def itemData(self, index: int) -> Any:
        return self._data[index]

    def findData(self, data: Any) -> int:
        try:
            return self._data.index(data)
        except ValueError:
            return -1

    def findText(self, text: str) -> int:
        for index, button in enumerate(self._buttons):
            if button.text() == text:
                return index
        return -1

    def currentIndex(self) -> int:
        return self._group.checkedId()

    def currentData(self) -> Any:
        index = self.currentIndex()
        return self._data[index] if 0 <= index < len(self._data) else None

    def currentText(self) -> str:
        index = self.currentIndex()
        return self._buttons[index].text() if 0 <= index < len(self._buttons) else ""

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._buttons) or index == self.currentIndex():
            return
        self._buttons[index].setChecked(True)
        if not self.signalsBlocked():
            self.currentIndexChanged.emit(index)

    def setItemToolTip(self, index: int, text: str) -> None:
        self._buttons[index].setToolTip(text)

    def _clicked(self, button: QToolButton) -> None:
        index = self._buttons.index(button)
        self.currentIndexChanged.emit(index)
        self.activated.emit(index)

    def _update_positions(self) -> None:
        last = len(self._buttons) - 1
        for index, button in enumerate(self._buttons):
            position = "only" if last == 0 else "first" if index == 0 else "last" if index == last else "middle"
            button.setProperty("segment", position)


__all__ = ["SegmentedControl"]
