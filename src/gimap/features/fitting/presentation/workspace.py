"""Feature-owned workspace layout for Fitting controls。"""

from __future__ import annotations

from typing import Sequence

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation import apply_design_system
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)
from src.gimap.app.presentation.layout_primitives import (
    CARD_SPACING,
    INPUT_WIDGET_TYPES,
    normalize_checkbox,
    normalize_input,
)
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .views import FittingWorkspaceView
from .cut_card import CutLineCard
from .input_card import GisaxsInputCard
from .layout_primitives import configure_button as _configure_button
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget
from .model_card import ModelParameterCard
from .preview_cards import DetectorPreviewCard, FittingPlotControlsCard, PlotPreviewCard, StatusCard
from .run_card import FittingControlsCard


class GisaxsFittingWorkspace:
    """Three-region layout for the cut/fitting page."""

    SETTINGS_KEY = "gisaxs_fitting_splitter_sizes"
    DEFAULT_WORK_SIZES = [760, 680]

    def __init__(
        self,
        ui,
        profile=None,
        *,
        preferences: UserPreferencesRepository,
    ):
        self.ui = ui
        self.preferences = preferences
        self.profile = profile or current_profile(ui.centralwidget)
        self.DEFAULT_WORK_SIZES = list(self.profile.work_sizes)
        self._legacy_fitting_scroll_area = ui.gisaxsFittingPageScrollArea
        self.page_splitter = QSplitter(Qt.Horizontal, ui.gisaxsFittingPage)
        self._workspace_ui = FittingWorkspaceView()
        self._workspace_ui.setupUi(self.page_splitter)
        self.fixed_controls_stack = self._workspace_ui.gisaxsFixedControlsStack
        self.work_area_contents = self._workspace_ui.gisaxsWorkAreaContents
        self.right_panel = self._workspace_ui.gisaxsRightCollapsiblePanel
        self.preview_scroll_area = self._workspace_ui.gisaxsPreviewScrollArea

        self.work_splitter = QSplitter(Qt.Vertical, ui.gisaxsFittingPage)
        self.work_splitter.setObjectName("gisaxsMainWorkSplitter")
        self.work_splitter.setHandleWidth(8)
        self.work_splitter.setChildrenCollapsible(False)
        self.work_splitter.setOpaqueResize(True)
        self.work_splitter.setMinimumWidth(self.profile.workspace_min)
        self.work_splitter.setMinimumHeight(
            sum(self.DEFAULT_WORK_SIZES) + self.work_splitter.handleWidth()
        )
        self.work_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._detach_preview_widgets()
        self._relax_fixed_sizes()
        self._install_page_splitter()
        self._build_left_work_area()
        self._build_preview_area()
        self._legacy_fitting_scroll_area.deleteLater()
        self._configure_button_responsiveness()
        self._apply_page_overflow_policy()
        self.restore_sizes()

    def _detach_preview_widgets(self) -> None:
        _take_widget(self.ui.gridLayout_23, self.ui.gisaxsInputGraphicsView)
        _take_widget(self.ui.gridLayout_24, self.ui.curvePlotControlWidget)
        _take_widget(self.ui.verticalLayout_19, self.ui.FittingTextBrowser)

    def _relax_fixed_sizes(self) -> None:
        self.ui.fitBox.setMinimumWidth(0)
        self.ui.fitBox.setMaximumWidth(16777215)
        self.ui.gisaxsInputBox.setMinimumWidth(0)
        self.ui.gisaxsInputBox.setMaximumWidth(16777215)
        self.ui.curvePlotControlWidget.setMinimumWidth(0)
        self.ui.curvePlotControlWidget.setMaximumWidth(16777215)
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(self.profile.workspace_min)
        self.ui.gisaxsFittingPageScrollArea.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.ui.gisaxsFittingPageScrollArea.setWidgetResizable(True)
        self._configure_expanding_inputs()

    def _build_left_work_area(self) -> None:
        workspace_ui = self._workspace_ui
        self.fitting_input_section = workspace_ui.fittingInputSection
        self.fitting_configure_section = workspace_ui.fittingConfigureSection
        self.fitting_advanced_section = workspace_ui.fittingAdvancedSection
        self.fitting_run_section = workspace_ui.fittingRunSection
        self.fitting_export_section = workspace_ui.fittingExportSection

        for section, title, description, content, layout in (
            (
                self.fitting_input_section,
                workspace_ui.fittingInputTitle,
                workspace_ui.fittingInputDescription,
                workspace_ui.fittingInputContent,
                workspace_ui.fittingInputContentLayout,
            ),
            (
                self.fitting_configure_section,
                workspace_ui.fittingConfigureTitle,
                workspace_ui.fittingConfigureDescription,
                workspace_ui.fittingConfigureContent,
                workspace_ui.fittingConfigureContentLayout,
            ),
            (
                self.fitting_run_section,
                workspace_ui.fittingRunTitle,
                workspace_ui.fittingRunDescription,
                workspace_ui.fittingRunContent,
                workspace_ui.fittingRunContentLayout,
            ),
            (
                self.fitting_export_section,
                workspace_ui.fittingExportTitle,
                workspace_ui.fittingExportDescription,
                workspace_ui.fittingExportContent,
                workspace_ui.fittingExportContentLayout,
            ),
        ):
            bind_parameter_section(section, title, description, content, layout)
        bind_advanced_section(
            self.fitting_advanced_section,
            workspace_ui.fittingAdvancedToggle,
            workspace_ui.fittingAdvancedDescription,
            workspace_ui.fittingAdvancedContent,
            workspace_ui.fittingAdvancedContentLayout,
        )

        gisaxs_card = GisaxsInputCard(self.ui, self.profile)
        cut_line_card = CutLineCard(self.ui, self.profile)
        fitting_controls_card = FittingControlsCard(self.ui, self.profile)
        model_parameters_card = ModelParameterCard(self.ui, self.profile)
        workspace_ui.fittingInputContentLayout.addWidget(gisaxs_card)
        workspace_ui.fittingConfigureContentLayout.addWidget(cut_line_card)
        workspace_ui.fittingAdvancedContentLayout.addWidget(model_parameters_card)
        workspace_ui.fittingRunContentLayout.addWidget(fitting_controls_card)

        _detach_from_parent_layout(self.ui.FittingExportButton)
        _detach_from_parent_layout(fitting_controls_card.fitExportPlotButton)
        self.ui.fitExportPlotButton = fitting_controls_card.fitExportPlotButton
        workspace_ui.fittingExportContentLayout.addWidget(
            self.ui.FittingExportButton
        )
        workspace_ui.fittingExportContentLayout.addWidget(
            fitting_controls_card.fitExportPlotButton
        )
        workspace_ui.fittingExportContentLayout.addStretch(1)

        fixed_stack_min_height = self._fixed_stack_min_height()
        self.fixed_controls_stack.setMinimumHeight(fixed_stack_min_height)
        self.fixed_controls_stack.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )
        self.work_splitter.hide()
        self.work_splitter.setParent(None)

        layout = self.work_area_contents.layout()
        self.work_area_contents.setMinimumHeight(
            fixed_stack_min_height
            + layout.contentsMargins().top()
            + layout.contentsMargins().bottom()
        )
        self.work_area_contents.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )

    def _build_preview_area(self) -> None:
        workspace_ui = self._workspace_ui
        self.fitting_preview_panel = workspace_ui.fittingPreviewPanel
        self.fitting_results_panel = workspace_ui.fittingResultsPanel
        self.fitting_plot_advanced_section = (
            workspace_ui.fittingPlotAdvancedSection
        )
        self.fitting_log_section = workspace_ui.fittingLogSection
        for section, title, description, content, layout in (
            (
                self.fitting_preview_panel,
                workspace_ui.fittingPreviewTitle,
                workspace_ui.fittingPreviewDescription,
                workspace_ui.fittingPreviewContent,
                workspace_ui.fittingPreviewContentLayout,
            ),
            (
                self.fitting_results_panel,
                workspace_ui.fittingResultsTitle,
                workspace_ui.fittingResultsDescription,
                workspace_ui.fittingResultsContent,
                workspace_ui.fittingResultsContentLayout,
            ),
        ):
            bind_parameter_section(section, title, description, content, layout)
        bind_advanced_section(
            self.fitting_plot_advanced_section,
            workspace_ui.fittingPlotAdvancedToggle,
            workspace_ui.fittingPlotAdvancedDescription,
            workspace_ui.fittingPlotAdvancedContent,
            workspace_ui.fittingPlotAdvancedContentLayout,
        )
        bind_advanced_section(
            self.fitting_log_section,
            workspace_ui.fittingLogToggle,
            workspace_ui.fittingLogDescription,
            workspace_ui.fittingLogContent,
            workspace_ui.fittingLogContentLayout,
        )

        self.detector_preview_card = DetectorPreviewCard(
            self.ui.gisaxsInputGraphicsView,
            self.profile,
        )
        self.fitting_plot_card = PlotPreviewCard(
            self.ui,
            self.ui.curvePlotControlWidget,
            self.ui.fitGraphicsView,
            self.profile,
        )
        self.fitting_controls_card = FittingPlotControlsCard(
            self.ui,
            self.ui.curvePlotControlWidget,
            self.profile,
        )
        self.run_log_card = StatusCard(self.ui.FittingTextBrowser, self.profile)
        self.ui.detectorPreviewCard = self.detector_preview_card
        self.ui.fittingPlotCard = self.fitting_plot_card
        self.ui.fittingPlotControlsCard = self.fitting_controls_card
        self.ui.runLogCard = self.run_log_card
        workspace_ui.fittingPreviewContentLayout.addWidget(
            self.detector_preview_card
        )
        workspace_ui.fittingResultsContentLayout.addWidget(
            self.fitting_plot_card
        )
        workspace_ui.fittingPlotAdvancedContentLayout.addWidget(
            self.fitting_controls_card
        )
        workspace_ui.fittingLogContentLayout.addWidget(self.run_log_card)

        self.right_panel.setMinimumWidth(self._preview_min_width())
        self.right_panel.setMaximumWidth(self._preview_max_width())
        self.right_panel.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.preview_scroll_area.setMinimumWidth(self._preview_min_width())
        self.preview_scroll_area.setMaximumWidth(self._preview_max_width())
        self.page_splitter.setStretchFactor(0, 3)
        self.page_splitter.setStretchFactor(1, 2)
        self.page_splitter.setCollapsible(0, False)
        self.page_splitter.setCollapsible(1, False)
        apply_design_system(self.page_splitter)

    def _configure_button_responsiveness(self) -> None:
        expanding_actions = [
            "gisaxsInputImportButton",
            "gisaxsInputCenterAutoFindingButton",
            "gisaxsInputDetectorParaButton",
            "fitImport1dFileButton",
            "FittingManualFittingButton",
            "FittingAutoFittingButton",
            "FittingClearFittingButton_2",
            "FittingAutoKButton",
            "FittingExportButton",
        ]
        preferred_actions = [
            "gisaxsInputCutButton",
            "gisaxsInputShowButton",
        ]

        for name in expanding_actions:
            button = getattr(self.ui, name, None)
            if button is not None:
                _configure_button(
                    button,
                    minimum_width=scale_value(108, self.profile, 88),
                    maximum_width=scale_value(220, self.profile, 180),
                    horizontal=QSizePolicy.MinimumExpanding,
                )

        for name in preferred_actions:
            button = getattr(self.ui, name, None)
            if button is not None:
                _configure_button(
                    button,
                    minimum_width=scale_value(78, self.profile, 64),
                    maximum_width=scale_value(140, self.profile, 116),
                    horizontal=QSizePolicy.Preferred,
                )

        plus_button = getattr(self.ui, "pushButton", None)
        if plus_button is not None:
            plus_button.setMinimumWidth(scale_value(220, self.profile, 190))
            plus_button.setMaximumWidth(scale_value(320, self.profile, 280))
            plus_button.setMinimumHeight(scale_value(36, self.profile, 32))
            plus_button.setMaximumHeight(scale_value(40, self.profile, 36))
            plus_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

    def _configure_expanding_inputs(self) -> None:
        for widget in self.ui.gisaxsFittingPage.findChildren(INPUT_WIDGET_TYPES):
            normalize_input(widget)
        for checkbox in self.ui.gisaxsFittingPage.findChildren(QCheckBox):
            normalize_checkbox(checkbox)

    def _install_page_splitter(self) -> None:
        layout = self.ui.verticalLayout_19
        _take_widget(layout, self._legacy_fitting_scroll_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.page_splitter)
        self.ui.gisaxsFittingPageScrollArea = (
            self._workspace_ui.gisaxsFittingPageScrollArea
        )
        self.ui.gisaxsFittingPageScrollAreaWidgetContents = (
            self.work_area_contents
        )

    def _page_min_width(self) -> int:
        return (
            self.profile.workspace_min
            + self._preview_min_width()
            + self.page_splitter.handleWidth()
        )

    def _preview_min_width(self) -> int:
        return max(self.profile.preview_min, scale_value(420, self.profile, 340))

    def _preview_max_width(self) -> int:
        """Allow the preview column to grow to twice its profile-default width."""
        return max(self._preview_min_width(), int(self.profile.page_sizes[1]) * 2)

    def _apply_page_overflow_policy(self) -> None:
        min_width = self._page_min_width()
        self.page_splitter.setMinimumWidth(min_width)
        self.page_splitter.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        QTimer.singleShot(0, self._set_page_sizes)

    def _available_page_width(self) -> int:
        width = self.page_splitter.width()
        if width > 0:
            return width
        width = self.ui.gisaxsFittingPage.width()
        return width if width > 0 else self._page_min_width()

    def _set_page_sizes(self, sizes: Sequence[int] | None = None) -> None:
        try:
            available = max(
                1, self._available_page_width() - self.page_splitter.handleWidth()
            )
        except RuntimeError:
            # A zero-delay layout callback can outlive its Qt splitter at shutdown.
            return
        left_min = self.profile.workspace_min
        right_min = self._preview_min_width()

        if available < left_min + right_min:
            self.page_splitter.setSizes([left_min, right_min])
            return

        if sizes and len(sizes) == 2:
            left = max(left_min, int(sizes[0]))
            right = max(right_min, int(sizes[1]))
        else:
            left, right = self.profile.page_sizes
            left = max(left_min, int(left))
            right = max(right_min, int(right))

        overflow = left + right - available
        if overflow > 0:
            reducible_left = max(0, left - left_min)
            reduce_left = min(reducible_left, overflow)
            left -= reduce_left
            overflow -= reduce_left
        if overflow > 0:
            reducible_right = max(0, right - right_min)
            reduce_right = min(reducible_right, overflow)
            right -= reduce_right

        self.page_splitter.setSizes([left, right])

    def restore_sizes(self) -> None:
        sizes = self.preferences.get(self.SETTINGS_KEY, None)
        if isinstance(sizes, dict):
            if sizes.get("profile") != self.profile.key:
                self._set_page_sizes(self.profile.page_sizes)
                self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
                return
            page_sizes = sizes.get("page")
            work_sizes = sizes.get("work")
            if isinstance(page_sizes, (list, tuple)) and len(page_sizes) == 2:
                self._set_page_sizes(page_sizes)
            else:
                self._set_page_sizes(self.profile.page_sizes)
            if (
                self.work_splitter.count() >= 2
                and isinstance(work_sizes, (list, tuple))
                and len(work_sizes) == 2
            ):
                self.work_splitter.setSizes(
                    [
                        max(self.DEFAULT_WORK_SIZES[0], int(work_sizes[0])),
                        max(self.DEFAULT_WORK_SIZES[1], int(work_sizes[1])),
                    ]
                )
            elif self.work_splitter.count() >= 2:
                self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
            return

        self._set_page_sizes(self.profile.page_sizes)
        if self.work_splitter.count() >= 2:
            self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)

    def save_state(self) -> None:
        self.preferences.set(
            self.SETTINGS_KEY,
            {
                "page": self.page_splitter.sizes(),
                "work": self.work_splitter.sizes() if self.work_splitter.count() >= 2 else [],
                "profile": self.profile.key,
            },
        )

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        self.DEFAULT_WORK_SIZES = list(profile.work_sizes)
        self._configure_button_responsiveness()
        fitting_card = self.fixed_controls_stack.findChild(
            FittingControlsCard, "FittingControlsCard"
        )
        if fitting_card is not None:
            fitting_card.apply_responsive_profile(profile)
        self.right_panel.setMinimumWidth(self._preview_min_width())
        self.right_panel.setMaximumWidth(self._preview_max_width())
        self.preview_scroll_area.setMinimumWidth(self._preview_min_width())
        self.preview_scroll_area.setMaximumWidth(self._preview_max_width())
        self.work_splitter.setMinimumWidth(profile.workspace_min)
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(profile.workspace_min)
        self._apply_page_overflow_policy()

        fixed_min = self._fixed_stack_min_height()
        self.fixed_controls_stack.setMinimumHeight(fixed_min)
        if self.work_area_contents.layout() is not None:
            margins = self.work_area_contents.layout().contentsMargins()
            self.work_area_contents.setMinimumHeight(fixed_min + margins.top() + margins.bottom())
            self.work_area_contents.layout().invalidate()
        self.fixed_controls_stack.layout().invalidate()
        self.fixed_controls_stack.adjustSize()
        self.work_area_contents.adjustSize()
        self._set_page_sizes(profile.page_sizes)
        if self.work_splitter.count() >= 2:
            self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)

    def _fixed_stack_min_height(self) -> int:
        sections = [
            getattr(self, name, None)
            for name in (
                "fitting_input_section",
                "fitting_configure_section",
                "fitting_advanced_section",
                "fitting_run_section",
                "fitting_export_section",
            )
        ]
        sections = [section for section in sections if section is not None]
        if sections:
            heights = [
                max(section.minimumSizeHint().height(), section.sizeHint().height())
                for section in sections
            ]
            return sum(heights) + (len(heights) - 1) * CARD_SPACING
        card_names = ("GisaxsInputCard", "CutLineCard", "FittingControlsCard", "ModelParameterCard")
        card_heights = [
            max(
                widget.minimumHeight(),
                widget.minimumSizeHint().height(),
                widget.sizeHint().height(),
            )
            for name in card_names
            if (widget := self.fixed_controls_stack.findChild(QWidget, name)) is not None
        ]
        if not card_heights:
            return self.fixed_controls_stack.minimumHeight()
        return sum(card_heights) + (len(card_heights) - 1) * CARD_SPACING


__all__ = ["GisaxsFittingWorkspace"]
