"""Small layout primitives owned by the Fitting presentation。"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractButton,
    QFrame,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import (
    CARD_MARGIN,
    CARD_SPACING,
    SECTION_MIN_WIDTH,
    normalize_button,
)
from src.gimap.app.presentation import SafeWheelDoubleSpinBox


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
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.body_layout.addWidget(self.title_label)

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        widget.setParent(self)
        self.body_layout.addWidget(widget, stretch)

    def lock_to_natural_height(self) -> None:
        """Let the parent layout use this card's current natural height.

        Older code copied ``adjustSize()`` into ``minimumHeight``.  When a card
        was assembled inside a large bootstrap canvas, that one-time geometry
        became a permanent minimum and later compact pages could never shrink.
        """
        if self.layout() is not None:
            self.layout().activate()
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.updateGeometry()


class NoWheelDoubleSpinBox(SafeWheelDoubleSpinBox):
    """Legacy name for the shared safe-wheel double spin box."""


class DisclosurePanel(QWidget):
    """Small presentation-only disclosure for secondary controls."""

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.toggle = QToolButton(self)
        self.toggle.setObjectName(f"{object_name}Toggle")
        self.toggle.setProperty("fittingDisclosure", True)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setText(title)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.content = QWidget(self)
        self.content.setObjectName(f"{object_name}Content")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 2, 4, 4)
        self.content_layout.setSpacing(8)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)

    def add_widget(self, widget: QWidget) -> None:
        widget.setParent(self.content)
        self.content_layout.addWidget(widget)


class CurrentPageHeightStackedWidget(QStackedWidget):
    """Keep the stack height synchronized with its visible parameter page."""

    def __init__(self, parent=None, *, fitting_view_model=None):
        super().__init__(parent)
        if fitting_view_model is None:
            raise ValueError("CurrentPageHeightStackedWidget requires FittingViewModel")
        self.fitting_view_model = fitting_view_model
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
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
        self.setMaximumHeight(16777215)
        self.updateGeometry()

    def sizeHint(self):
        current_widget = self.currentWidget()
        return current_widget.sizeHint() if current_widget is not None else super().sizeHint()

    def minimumSizeHint(self):
        current_widget = self.currentWidget()
        if current_widget is not None:
            return current_widget.minimumSizeHint()
        return super().minimumSizeHint()


class CurrentPageHeightTabWidget(QTabWidget):
    """Report the current tab's height instead of the tallest hidden tab."""

    def _current_page_height(self, *, minimum: bool) -> int:
        page = self.currentWidget()
        if page is None:
            return 0
        page_hint = page.minimumSizeHint() if minimum else page.sizeHint()
        return self.tabBar().sizeHint().height() + max(1, page_hint.height()) + 12

    def sizeHint(self):
        hint = super().sizeHint()
        height = self._current_page_height(minimum=False)
        if height > 0:
            hint.setHeight(height)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        height = self._current_page_height(minimum=True)
        if height > 0:
            hint.setHeight(height)
        return hint


class CurrentPageSizeTabWidget(CurrentPageHeightTabWidget):
    """Isolate a workspace tab from the size hints of hidden sibling pages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(
            lambda _index: QTimer.singleShot(0, self.refresh_current_page_geometry)
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_current_page_geometry()

    def refresh_current_page_geometry(self) -> None:
        current = self.currentIndex()
        for index in range(self.count()):
            page = self.widget(index)
            if page is None:
                continue
            policy = page.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Expanding)
            policy.setVerticalPolicy(
                QSizePolicy.Preferred if index == current else QSizePolicy.Ignored
            )
            page.setSizePolicy(policy)
            if index != current:
                page.setMinimumSize(0, 0)
        if self.layout() is not None:
            self.layout().invalidate()
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().invalidate()
            parent.updateGeometry()

    def _current_page_width(self, *, minimum: bool) -> int:
        page = self.currentWidget()
        if page is None:
            return 0
        page_hint = page.minimumSizeHint() if minimum else page.sizeHint()
        return max(self.tabBar().sizeHint().width(), page_hint.width()) + 8

    def sizeHint(self):
        hint = super().sizeHint()
        width = self._current_page_width(minimum=False)
        if width > 0:
            hint.setWidth(width)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        width = self._current_page_width(minimum=True)
        if width > 0:
            hint.setWidth(width)
        return hint


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


__all__ = [
    "CardFrame",
    "CurrentPageHeightTabWidget",
    "CurrentPageSizeTabWidget",
    "CurrentPageHeightStackedWidget",
    "DisclosurePanel",
    "NoWheelDoubleSpinBox",
]
