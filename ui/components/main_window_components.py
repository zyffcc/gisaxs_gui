"""Component layer for the generated main window.

The generated ``Ui_MainWindow`` still creates the individual controls.  These
classes only reorganize those controls into named, testable pieces without
changing object names that controllers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.user_settings import user_settings
from ui.style_loader import apply_main_window_styles


@dataclass(frozen=True)
class PageDefinition:
    index: int
    name: str
    widget_name: str


class NavigationSidebar(QWidget):
    """Owns the left navigation buttons while preserving generated buttons."""

    def __init__(self, buttons: Sequence[QAbstractButton], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("navigationSidebar")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        for button in buttons:
            button.setParent(self)
            button.setProperty("navigationButton", True)
            button.setMinimumHeight(38)
            button.setMaximumHeight(44)
            layout.addWidget(button)

        layout.addStretch(1)


class ContentStack:
    """Small facade around the generated central QStackedWidget."""

    PAGES = (
        PageDefinition(0, "Trainset Build", "trainsetBuildPage"),
        PageDefinition(1, "GISAXS Predict", "gisaxsPredictPage"),
        PageDefinition(2, "Cut Fitting", "gisaxsFittingPage"),
        PageDefinition(3, "Classification", "classificationPage"),
    )

    def __init__(self, stack: QStackedWidget):
        self.stack = stack
        self.stack.setObjectName("mainWindowWidget")
        self._setup_adaptive_behavior()

    def page_name(self, index: int) -> str:
        for page in self.PAGES:
            if page.index == index:
                return page.name
        return f"Page {index}"

    def _setup_adaptive_behavior(self) -> None:
        try:
            from utils.layout_utils import LayoutUtils

            LayoutUtils.setup_adaptive_stacked_widget(self.stack)
        except Exception as exc:
            print(f"Stacked widget adaptive setup skipped: {exc}")


class MainShell(QSplitter):
    """Top-level resizable shell containing sidebar and main content."""

    SETTINGS_KEY = "main_splitter_sizes"

    def __init__(
        self,
        central_widget: QWidget,
        source_layout,
        sidebar_area: QScrollArea,
        content_widget: QWidget,
        parent: QWidget | None = None,
    ):
        super().__init__(Qt.Horizontal, parent or central_widget)
        self.setObjectName("mainShell")
        self.setHandleWidth(6)

        self._remove_from_layout(source_layout, sidebar_area)
        self._remove_from_layout(source_layout, content_widget)

        sidebar_area.setMinimumWidth(140)
        sidebar_area.setMaximumWidth(260)
        sidebar_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.addWidget(sidebar_area)
        self.addWidget(content_widget)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)

        source_layout.addWidget(self)
        self.restore_sizes()

    @staticmethod
    def _remove_from_layout(layout, widget: QWidget) -> None:
        index = layout.indexOf(widget)
        if index != -1:
            layout.takeAt(index)

    def restore_sizes(self) -> None:
        sizes = user_settings.get(self.SETTINGS_KEY, None)
        if isinstance(sizes, (list, tuple)) and len(sizes) == 2:
            self.setSizes([max(120, int(sizes[0])), max(400, int(sizes[1]))])
            return

        self.setSizes([160, 900])

    def save_sizes(self) -> None:
        user_settings.set(self.SETTINGS_KEY, self.sizes())
        user_settings.save_settings()


class MainWindowComponents:
    """Builds and owns the maintainable component hierarchy."""

    def __init__(self, ui):
        self.ui = ui
        self._clear_generated_inline_styles(ui.centralwidget)
        self.sidebar = self._create_sidebar()
        self.content = ContentStack(ui.mainWindowWidget)
        self.shell = MainShell(
            ui.centralwidget,
            ui.horizontalLayout,
            ui.sideBarScrollArea,
            ui.mainContentWidget,
        )
        apply_main_window_styles(ui)

    def _create_sidebar(self) -> NavigationSidebar:
        buttons = [
            self.ui.cutAndFittingButton,
            self.ui.gisaxsPredictButton,
            self.ui.trainsetBuildButton,
            self.ui.ClassficationButton,
            self.ui.WAXSButton,
        ]
        sidebar = NavigationSidebar(buttons)
        self.ui.sideBarScrollArea.setWidget(sidebar)
        self.ui.sideBarScrollArea.setWidgetResizable(True)
        return sidebar

    def save_state(self) -> None:
        self.shell.save_sizes()

    @staticmethod
    def _clear_generated_inline_styles(root: QWidget) -> None:
        for widget in _walk_widgets(root):
            if widget.styleSheet():
                widget.setStyleSheet("")


def _walk_widgets(root: QWidget) -> Iterable[QWidget]:
    yield root
    yield from root.findChildren(QWidget)
