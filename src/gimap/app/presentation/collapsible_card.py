"""Persistent collapsible card used by workspace presentation layers。"""

from __future__ import annotations

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import CARD_MARGIN, CARD_SPACING, SECTION_MIN_WIDTH


class CardContentResizeHandle(QFrame):
    """Small drag handle that changes only its card's expanded content height。"""

    def __init__(self, card: "CollapsibleCardFrame") -> None:
        super().__init__(card)
        self.card = card
        self._press_y = 0
        self._start_height = 0
        self.setFixedHeight(9)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to resize this card's content area.")
        self.setStyleSheet(
            "QFrame { border: 0; border-top: 2px solid #d6dee9; margin: 3px 35%; }"
            "QFrame:hover { border-top-color: #7c93ad; }"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press_y = event.globalY()
            self._start_height = self.card.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self.card._set_user_expanded_height(
                self._start_height + event.globalY() - self._press_y
            )
            event.accept()
            return
        super().mouseMoveEvent(event)


class CollapsibleCardFrame(QFrame):
    """Card wrapper with a persistent collapse/expand header。"""

    SETTINGS_PREFIX = "cut_fitting/right_cards"

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        *,
        default_expanded: bool = True,
        settings_prefix: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        prefix = settings_prefix or self.SETTINGS_PREFIX
        self._settings_key = f"{prefix}/{object_name}/expanded"
        self.setObjectName(object_name)
        self.setProperty("card", True)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(CARD_MARGIN, 8, CARD_MARGIN, CARD_MARGIN)
        self.body_layout.setSpacing(CARD_SPACING)

        self.header_widget = QWidget(self)
        self.header_widget.setObjectName(f"{object_name}Header")
        self.header_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QLabel(title, self.header_widget)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("cardTitle", True)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.header_button = QToolButton(self.header_widget)
        self.header_button.setObjectName(f"{object_name}ToggleButton")
        self.header_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.header_button.setCheckable(True)
        self.header_button.setAutoRaise(True)
        self.header_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.header_button.toggled.connect(self.set_expanded)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.header_button, 0, Qt.AlignRight)
        self.body_layout.addWidget(self.header_widget)

        self.content_widget = QWidget(self)
        self.content_widget.setObjectName(f"{object_name}Content")
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(CARD_SPACING)
        self.body_layout.addWidget(self.content_widget, 1)

        expanded = QSettings().value(self._settings_key, default_expanded, type=bool)
        self.header_button.blockSignals(True)
        self.header_button.setChecked(bool(expanded))
        self.header_button.blockSignals(False)
        self.set_expanded(bool(expanded))

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        widget.setParent(self.content_widget)
        self.content_layout.addWidget(widget, stretch)

    def enable_content_resize(self, base_height: int, maximum_height: int | None = None) -> None:
        """Add an internal resize handle while keeping the title bar fixed。"""

        self._content_resize_handle = CardContentResizeHandle(self)
        self.body_layout.addWidget(self._content_resize_handle)
        self.body_layout.activate()
        margins = self.body_layout.contentsMargins()
        visible_body_height = (
            self.header_widget.sizeHint().height()
            + self.content_widget.minimumSizeHint().height()
            + self._content_resize_handle.height()
            + margins.top()
            + margins.bottom()
            + self.body_layout.spacing() * 2
            + self.frameWidth() * 2
            + 4
        )
        natural_height = max(
            1,
            int(base_height),
            visible_body_height,
            self.minimumSizeHint().height(),
            self.sizeHint().height(),
        )
        self._resize_base_height = natural_height
        requested_max = int(maximum_height) if maximum_height is not None else natural_height * 2
        self._resize_max_height = max(natural_height, requested_max, natural_height * 2)
        self._user_expanded_height = natural_height
        self.set_expanded(self.is_expanded())

    def _set_user_expanded_height(self, height: int) -> None:
        if not self.is_expanded():
            return
        height = max(self._resize_base_height, min(int(height), self._resize_max_height))
        self._user_expanded_height = height
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.updateGeometry()

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self.header_button.setChecked(expanded)
        self.header_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content_widget.setVisible(expanded)
        QSettings().setValue(self._settings_key, expanded)
        if expanded:
            if hasattr(self, "_user_expanded_height"):
                self._set_user_expanded_height(self._user_expanded_height)
            else:
                self.setMaximumHeight(16777215)
                margins = self.body_layout.contentsMargins()
                header_height = self.header_widget.sizeHint().height()
                self.setMinimumHeight(
                    max(
                        header_height + margins.top() + margins.bottom(),
                        self.sizeHint().height(),
                    )
                )
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        else:
            header_height = self.header_widget.sizeHint().height()
            margins = self.body_layout.contentsMargins()
            collapsed_height = header_height + margins.top() + margins.bottom()
            self.setMinimumHeight(collapsed_height)
            self.setMaximumHeight(collapsed_height)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if hasattr(self, "_content_resize_handle"):
            self._content_resize_handle.setVisible(expanded)
        self.updateGeometry()

    def is_expanded(self) -> bool:
        return self.header_button.isChecked()


__all__ = ["CardContentResizeHandle", "CollapsibleCardFrame"]
