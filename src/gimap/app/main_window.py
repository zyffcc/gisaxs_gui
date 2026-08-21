"""Application-owned composition layer for the main window.

The Python ``ApplicationWindowView`` creates the shell controls. These
classes only reorganize those controls into named, testable pieces without
changing object names that feature presentation bindings depend on.
"""

from __future__ import annotations

from typing import Iterable

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .presentation.collapsible_card import (
    CardContentResizeHandle,
    CollapsibleCardFrame,
)
from .presentation.navigation import NavigationSidebar
from .presentation.shell import ContentStack, MainShell, PageDefinition
from src.gimap.features.classification.presentation.page import ClassificationPage
from src.gimap.features.prediction.presentation import (
    GisaxsPredictWorkspace,
    PredictCard,
    PredictModelLibraryCard,
)
from src.gimap.features.trainset.presentation import TrainsetBuildPage
from src.gimap.features.waxs.presentation.page import InSituProcessingWidget
from src.gimap.features.fitting.presentation import (
    CardFrame,
    CutLineCard,
    DetectorPreviewCard,
    FittingControlsCard,
    FittingPlotControlsCard,
    FittingRegionControl,
    GisaxsFittingWorkspace,
    GisaxsInputCard,
    ModelParameterCard,
    NoWheelDoubleSpinBox,
    ParticleOptionsLayout,
    PlotCanvasArea,
    PlotOptionsControl,
    PlotPreviewCard,
    PlotSamplingControl,
    SectionCard,
    StatusCard,
)
from src.gimap.app.presentation.responsive_layout import (
    apply_density_profile,
    apply_window_profile,
    current_profile,
    install_adaptive_window_profile,
)
from src.gimap.app.presentation.style_loader import apply_main_window_styles


class MainWindowComponents:
    """Builds and owns the maintainable component hierarchy."""

    def __init__(self, ui):
        self.ui = ui
        self.preferences = ui.app_context.preferences
        self.responsive_profile = current_profile(
            ui.centralwidget,
            preferences=self.preferences,
        )
        self._clear_view_inline_styles(ui.centralwidget)
        self.trainset_page = self._create_trainset_page()
        self.classification_page = self._create_classification_page()
        self.waxs_page = self._create_waxs_page()
        self.sidebar = self._create_sidebar()
        self.content = ContentStack(ui.mainWindowWidget)
        from src.gimap.features.fitting.bootstrap import create_fitting_view_model

        self.fitting_view_model = create_fitting_view_model(ui.app_context)
        self.fitting_workspace = GisaxsFittingWorkspace(
            ui,
            self.responsive_profile,
            preferences=ui.app_context.preferences,
            view_model=self.fitting_view_model,
        )
        self.predict_workspace = GisaxsPredictWorkspace(ui, self.responsive_profile)
        self.shell = MainShell(
            ui.centralwidget,
            ui.horizontalLayout,
            ui.sideBarScrollArea,
            ui.mainContentWidget,
            profile=self.responsive_profile,
            navigation_sidebar=self.sidebar,
            preferences=ui.app_context.preferences,
        )
        apply_main_window_styles(ui)
        apply_density_profile(
            ui.centralwidget,
            self.responsive_profile,
            preferences=self.preferences,
        )
        apply_window_profile(
            ui.centralwidget.window(),
            self.responsive_profile,
            resize_window=True,
            preferences=self.preferences,
        )
        install_adaptive_window_profile(
            ui.centralwidget.window(),
            self._on_screen_profile_changed,
            preferences=self.preferences,
        )
        QTimer.singleShot(0, self.shell.apply_initial_sidebar_state)

    def _create_trainset_page(self) -> TrainsetBuildPage:
        host = self.ui.trainsetBuildPage
        layout = host.layout()
        if layout is None:
            layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        page = TrainsetBuildPage(host)
        layout.addWidget(page)
        self.ui.trainsetWorkspace = page
        return page

    def _create_classification_page(self) -> ClassificationPage:
        host = self.ui.classificationPage
        layout = host.layout()
        if layout is None:
            layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        page = ClassificationPage(host)
        layout.addWidget(page)
        self.ui.classificationWorkspace = page
        return page

    def _create_waxs_page(self) -> InSituProcessingWidget:
        from src.gimap.features.waxs.bootstrap import create_waxs_view_model

        page = InSituProcessingWidget(
            view_model=create_waxs_view_model(self.ui.app_context),
        )
        stack = self.ui.mainWindowWidget
        host = getattr(self.ui, "waxsPageHost", None)
        host_index = stack.indexOf(host) if host is not None else -1
        if host_index >= 0:
            stack.insertWidget(host_index, page)
            stack.removeWidget(host)
            host.setParent(None)
            host.deleteLater()
            delattr(self.ui, "waxsPageHost")
            page_index = host_index
        else:
            page_index = stack.addWidget(page)
        self.ui.waxsPage = page
        self.ui.waxsPageIndex = page_index
        return page

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
        self.ui.sideBarScrollArea.setFrameShape(QFrame.NoFrame)
        self.ui.sideBarScrollArea.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.ui.sideBarScrollArea.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.ui.sideBarScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.sideBarScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ui.mainWindowWidget.currentChanged.connect(self._sync_sidebar_to_content_page)
        return sidebar

    def _sync_sidebar_to_content_page(self, page_index: int) -> None:
        rail_index_by_page = {
            2: 0,  # Cut & Fitting
            1: 1,  # 2D Prediction
            0: 2,  # Trainset Build
            3: 3,  # Classification
            getattr(self.ui, "waxsPageIndex", -1): 4,  # WAXS
        }
        rail_index = rail_index_by_page.get(page_index)
        if rail_index is not None:
            self.sidebar.set_active_index(rail_index)

    def save_state(self) -> None:
        self.fitting_workspace.save_state()
        self.shell.save_sizes()

    def apply_responsive_profile(self, profile) -> None:
        self.responsive_profile = profile
        apply_density_profile(
            self.ui.centralwidget,
            profile,
            preferences=self.preferences,
        )
        apply_window_profile(
            self.ui.centralwidget.window(),
            profile,
            resize_window=False,
            preferences=self.preferences,
        )
        self.fitting_workspace.apply_responsive_profile(profile)
        self.shell.apply_responsive_profile(profile)
        if hasattr(self, "predict_workspace"):
            self.predict_workspace.apply_responsive_profile(profile)
        apply_density_profile(
            self.ui.centralwidget,
            profile,
            preferences=self.preferences,
        )

    def _on_screen_profile_changed(self, profile, screen) -> None:
        if profile.key == self.responsive_profile.key:
            apply_density_profile(
                self.ui.centralwidget,
                profile,
                preferences=self.preferences,
            )
            return
        self.apply_responsive_profile(profile)
        apply_main_window_styles(self.ui)

    @staticmethod
    def _clear_view_inline_styles(root: QWidget) -> None:
        for widget in _walk_widgets(root):
            if widget.styleSheet():
                widget.setStyleSheet("")


def _walk_widgets(root: QWidget) -> Iterable[QWidget]:
    yield root
    yield from root.findChildren(QWidget)


__all__ = [
    "CardContentResizeHandle",
    "CardFrame",
    "CollapsibleCardFrame",
    "ContentStack",
    "CutLineCard",
    "DetectorPreviewCard",
    "FittingControlsCard",
    "FittingPlotControlsCard",
    "FittingRegionControl",
    "GisaxsFittingWorkspace",
    "GisaxsInputCard",
    "GisaxsPredictWorkspace",
    "MainShell",
    "MainWindowComponents",
    "ModelParameterCard",
    "NavigationSidebar",
    "NoWheelDoubleSpinBox",
    "PageDefinition",
    "ParticleOptionsLayout",
    "PlotCanvasArea",
    "PlotOptionsControl",
    "PlotPreviewCard",
    "PlotSamplingControl",
    "PredictCard",
    "PredictModelLibraryCard",
    "SectionCard",
    "StatusCard",
]
