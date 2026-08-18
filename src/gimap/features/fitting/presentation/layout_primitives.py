"""Small layout primitives owned by the Fitting presentation。"""

from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAbstractButton,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import (
    CARD_MARGIN,
    CARD_SPACING,
    SECTION_MIN_WIDTH,
    normalize_button,
)


class CardFrame(QFrame):
    """Non-collapsible card wrapper for generated Fitting controls。"""

    def __init__(self, title: str, object_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("card", True)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(CARD_MARGIN, 12, CARD_MARGIN, CARD_MARGIN)
        self.body_layout.setSpacing(CARD_SPACING)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("cardTitle", True)
        self.body_layout.addWidget(self.title_label)

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        widget.setParent(self)
        self.body_layout.addWidget(widget, stretch)

    def lock_to_natural_height(self) -> None:
        if self.layout() is not None:
            self.layout().activate()
        self.adjustSize()
        natural_height = max(self.minimumSizeHint().height(), self.sizeHint().height())
        if natural_height > 0:
            self.setMinimumHeight(natural_height)
        self.setMaximumHeight(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.updateGeometry()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that ignores wheel events。"""

    def wheelEvent(self, event) -> None:
        if event is not None:
            event.ignore()


class CurrentPageHeightStackedWidget(QStackedWidget):
    """Keep the stack height synchronized with its visible parameter page."""

    def __init__(self, parent=None, *, fitting_view_model=None):
        super().__init__(parent)
        if fitting_view_model is None:
            raise ValueError("CurrentPageHeightStackedWidget requires FittingViewModel")
        self.fitting_view_model = fitting_view_model
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.currentChanged.connect(
            lambda _index: QTimer.singleShot(0, self.sync_current_height)
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_current_height()

    def sync_current_height(self) -> None:
        current_widget = self.currentWidget()
        hint = current_widget.sizeHint() if current_widget is not None else super().sizeHint()
        height = max(1, hint.height())
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def sizeHint(self):
        current_widget = self.currentWidget()
        return current_widget.sizeHint() if current_widget is not None else super().sizeHint()

    def minimumSizeHint(self):
        current_widget = self.currentWidget()
        if current_widget is not None:
            return current_widget.minimumSizeHint()
        return super().minimumSizeHint()


def take_widget(layout, widget: QWidget) -> None:
    index = layout.indexOf(widget)
    if index != -1:
        layout.takeAt(index)
    widget.setParent(None)


def detach_from_parent_layout(widget: QWidget) -> None:
    parent = widget.parentWidget()
    if parent is not None and parent.layout() is not None:
        take_widget(parent.layout(), widget)
    else:
        widget.setParent(None)


def configure_button(
    button: QAbstractButton,
    minimum_width: int,
    maximum_width: int,
    horizontal=QSizePolicy.Preferred,
) -> None:
    normalize_button(button, wide=horizontal == QSizePolicy.MinimumExpanding)
    button.setMinimumWidth(minimum_width)
    button.setMaximumWidth(maximum_width)
    button.setSizePolicy(horizontal, QSizePolicy.Fixed)


__all__ = ["CardFrame", "CurrentPageHeightStackedWidget", "NoWheelDoubleSpinBox"]
