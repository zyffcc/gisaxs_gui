"""Section containers for workspace information hierarchy."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def describe_title(title_label: QLabel, description: str) -> None:
    """``description`` as the tooltip of a section title, with the pointer that says there is one."""
    title_label.setToolTip(description)
    if description:
        title_label.setCursor(Qt.WhatsThisCursor)
    else:
        title_label.unsetCursor()


class ParameterSection(QFrame):
    """Titled container for one cohesive parameter group.

    The description is what the section is for: it is needed once, so it is the title's tooltip and not a
    line on screen. ``show_description`` keeps it as a line for the rare text needed every time (where a
    value comes from when that changes how far it can be trusted)."""

    def __init__(
        self,
        title: str,
        description: str = "",
        parent: QWidget | None = None,
        *,
        show_description: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setProperty("gimapSection", True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(title, self)
        self.title_label.setProperty("gimapSectionTitle", True)
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.header_actions = QHBoxLayout()
        self.header_actions.setContentsMargins(0, 0, 0, 0)
        self.header_actions.setSpacing(6)
        header.addLayout(self.header_actions)
        root.addLayout(header)

        self._show_description = bool(show_description)
        self.description_label = QLabel("", self)
        self.description_label.setProperty("gimapSectionDescription", True)
        self.description_label.setWordWrap(True)
        root.addWidget(self.description_label)
        self.set_description(description)

        self.content = QWidget(self)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        root.addWidget(self.content)

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_description(self, description: str) -> None:
        self.description_label.setText(description)
        self.description_label.setVisible(bool(description) and self._show_description)
        describe_title(self.title_label, "" if self._show_description else description)

    def add_header_action(self, widget: QWidget) -> None:
        self.header_actions.addWidget(widget)

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self.content_layout.addWidget(widget, stretch)

    def add_layout(self, layout, stretch: int = 0) -> None:
        self.content_layout.addLayout(layout, stretch)


class AdvancedSection(QFrame):
    """Collapsible low-frequency parameter container preserving child state."""

    expandedChanged = pyqtSignal(bool)

    def __init__(
        self,
        title: str = "Advanced",
        description: str = "",
        parent: QWidget | None = None,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setProperty("gimapSection", True)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setProperty("gimapAdvancedToggle", True)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(bool(expanded))
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self.toggle_button.setText(title)
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # the whole row opens it
        from ..i18n import tr

        # A fixed sentence (the title is the button's own text): the language walker knows it, so it follows a
        # switch of the interface language in both directions, as the title and a description do.
        self.toggle_button.setToolTip(description or tr("Show or hide this section"))
        root.addWidget(self.toggle_button)

        # The description is the button's tooltip only: a line under the title would repeat it word for word.
        self.description_label = QLabel(description, self)
        self.description_label.setProperty("gimapSectionDescription", True)
        self.description_label.setWordWrap(True)
        self.description_label.hide()
        root.addWidget(self.description_label)

        self.content = QWidget(self)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 2, 4, 4)
        self.content_layout.setSpacing(8)
        self.content.setVisible(bool(expanded))
        root.addWidget(self.content)

        self.toggle_button.toggled.connect(self.set_expanded)

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self.toggle_button.blockSignals(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.blockSignals(False)
        self.toggle_button.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self.content.setVisible(expanded)
        self.expandedChanged.emit(expanded)

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self.content_layout.addWidget(widget, stretch)

    def add_layout(self, layout, stretch: int = 0) -> None:
        self.content_layout.addLayout(layout, stretch)
