"""Feature-free application shell layout and content-stack primitives."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QWidget

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation.adaptive_stack import configure_adaptive_stack
from src.gimap.app.presentation.responsive_layout import apply_window_profile, current_profile

from .navigation import NavigationSidebar


@dataclass(frozen=True)
class PageDefinition:
    index: int
    name: str
    widget_name: str


class ContentStack:
    """Small facade around the generated central QStackedWidget."""

    PAGES = (
        PageDefinition(0, "Trainset Build", "trainsetBuildPage"),
        PageDefinition(1, "GIMaP Predict", "gisaxsPredictPage"),
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
            configure_adaptive_stack(self.stack)
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
        profile=None,
        navigation_sidebar: NavigationSidebar | None = None,
        *,
        preferences: UserPreferencesRepository,
    ):
        super().__init__(Qt.Horizontal, parent or central_widget)
        self.preferences = preferences
        self.profile = profile or current_profile(
            central_widget,
            preferences=preferences,
        )
        self.navigation_sidebar = navigation_sidebar
        self.sidebar_area = sidebar_area
        self.content_widget = content_widget
        self.setObjectName("mainShell")
        self.setHandleWidth(6)
        self.setChildrenCollapsible(False)
        self.setOpaqueResize(True)
        self._enforce_window_minimum_width(central_widget)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(0)
        central_widget.setContentsMargins(0, 0, 0, 0)

        self._remove_from_layout(source_layout, sidebar_area)
        self._remove_from_layout(source_layout, content_widget)

        sidebar_min = self._sidebar_min_width()
        sidebar_max = self._sidebar_max_width()
        sidebar_area.setMinimumWidth(sidebar_min)
        sidebar_area.setMaximumWidth(sidebar_max)
        sidebar_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        content_widget.setMinimumWidth(self.profile.content_min)
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.addWidget(sidebar_area)
        self.addWidget(content_widget)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 5)
        self.setCollapsible(0, False)
        self.setCollapsible(1, False)

        source_layout.addWidget(self)
        if self.navigation_sidebar is not None:
            self.navigation_sidebar.collapsedChanged.connect(self._on_sidebar_collapsed_changed)
        self.restore_sizes()

    @staticmethod
    def _remove_from_layout(layout, widget: QWidget) -> None:
        index = layout.indexOf(widget)
        if index != -1:
            layout.takeAt(index)

    def restore_sizes(self) -> None:
        if self._sidebar_collapsed():
            self._apply_sidebar_width(NavigationSidebar.COLLAPSED_WIDTH)
            self.setSizes([NavigationSidebar.COLLAPSED_WIDTH, self.profile.content_min])
            return

        sizes = self.preferences.get(self.SETTINGS_KEY, None)
        if isinstance(sizes, (list, tuple)) and len(sizes) == 2:
            sidebar_width = min(
                self._sidebar_max_width(), max(self._sidebar_min_width(), int(sizes[0]))
            )
            self.setSizes(
                [
                    sidebar_width,
                    max(self.profile.content_min, int(sizes[1])),
                ]
            )
            return

        self.setSizes([self._sidebar_default_width(), self.profile.content_min])

    def save_sizes(self) -> None:
        if not self._sidebar_collapsed():
            self.preferences.set(self.SETTINGS_KEY, self.sizes())
        self.preferences.set(NavigationSidebar.SETTINGS_KEY, self._sidebar_collapsed())
        self.preferences.save()

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        sidebar = self.widget(0)
        content = self.widget(1)
        self._apply_sidebar_width(
            NavigationSidebar.COLLAPSED_WIDTH
            if self._sidebar_collapsed()
            else self._sidebar_default_width()
        )
        content.setMinimumWidth(profile.content_min)
        self.setSizes([sidebar.width(), profile.content_min])

    def _on_sidebar_collapsed_changed(self, collapsed: bool) -> None:
        if collapsed:
            self._apply_sidebar_width(NavigationSidebar.COLLAPSED_WIDTH)
            self.setSizes(
                [NavigationSidebar.COLLAPSED_WIDTH, max(self.profile.content_min, self.width())]
            )
            return

        width = self._sidebar_default_width()
        self._apply_sidebar_width(width)
        self.setSizes([width, max(self.profile.content_min, self.width() - width)])

    def _sidebar_collapsed(self) -> bool:
        return bool(self.navigation_sidebar is not None and self.navigation_sidebar.is_collapsed())

    def _sidebar_min_width(self) -> int:
        if self._sidebar_collapsed():
            return NavigationSidebar.COLLAPSED_WIDTH
        return NavigationSidebar.EXPANDED_WIDTH

    def _sidebar_max_width(self) -> int:
        if self._sidebar_collapsed():
            return NavigationSidebar.COLLAPSED_WIDTH
        return NavigationSidebar.EXPANDED_WIDTH

    def _sidebar_default_width(self) -> int:
        return NavigationSidebar.EXPANDED_WIDTH

    def _apply_sidebar_width(self, width: int) -> None:
        width = (
            NavigationSidebar.COLLAPSED_WIDTH
            if self._sidebar_collapsed()
            else NavigationSidebar.EXPANDED_WIDTH
        )
        sidebar = self.widget(0)
        if self.navigation_sidebar is not None:
            self.navigation_sidebar.apply_layout_state(self._sidebar_collapsed())
        sidebar.setMinimumWidth(width)
        sidebar.setMaximumWidth(width)
        if self.navigation_sidebar is not None:
            self.navigation_sidebar.setFixedWidth(width)
        sidebar.updateGeometry()

    def apply_initial_sidebar_state(self) -> None:
        collapsed = self._sidebar_collapsed()
        width = NavigationSidebar.COLLAPSED_WIDTH if collapsed else NavigationSidebar.EXPANDED_WIDTH
        if self.navigation_sidebar is not None:
            self.navigation_sidebar.apply_layout_state(collapsed)
        self.sidebar_area.setMinimumWidth(width)
        self.sidebar_area.setMaximumWidth(width)
        self.sidebar_area.setFixedWidth(width)
        self.sidebar_area.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sidebar_area.updateGeometry()
        self.sidebar_area.update()
        self.setSizes([width, max(self.profile.content_min, self.width() - width)])
        self.updateGeometry()
        self.update()

    def _enforce_window_minimum_width(self, central_widget: QWidget) -> None:
        window = central_widget.window()
        profile = current_profile(window, preferences=self.preferences)
        apply_window_profile(window, profile, preferences=self.preferences)

        def apply_after_layout() -> None:
            try:
                apply_window_profile(
                    window,
                    profile,
                    preferences=self.preferences,
                )
            except RuntimeError:
                # The main window may be deleted before this deferred callback runs.
                return

        QTimer.singleShot(
            0,
            apply_after_layout,
        )


__all__ = ["ContentStack", "MainShell", "PageDefinition"]
