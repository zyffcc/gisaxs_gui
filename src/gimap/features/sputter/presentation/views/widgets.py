"""Small widgets of the Sputter page's views: a combo box that says when it is shown, and a one-line label that never
widens its window."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget

from src.gimap.app.presentation.components import SafeWheelComboBox


class ShownComboBox(SafeWheelComboBox):
    """A combo box that says when it is shown or opened (``aboutToShow``): its choices can be worked out again."""

    aboutToShow = pyqtSignal()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.aboutToShow.emit()

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        self.aboutToShow.emit()
        super().showPopup()


class ElidedLabel(QLabel):
    """One line that never widens its window: text that does not fit ends in “…” and is whole in the tooltip.
    ``full_text`` is what was set."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)

    def set_full_text(self, text: str) -> None:
        self._full = str(text)
        self.setToolTip(self._full)
        self._fit()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(20, self.width() - 2))
        if shown != self.text():
            self.setText(shown)


__all__ = ["ElidedLabel", "ShownComboBox"]
