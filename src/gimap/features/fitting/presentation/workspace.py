"""Feature-owned workspace layout for Fitting controls。"""

from __future__ import annotations

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QLabel,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation import apply_design_system, install_safe_wheel_behavior
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
from .fitting_theme import fitting_stylesheet
from .input_card import GisaxsInputCard
from .layout_primitives import configure_button as _configure_button
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget
from .layout_primitives import CurrentPageSizeTabWidget
from .model_card import ModelParameterCard
from .preview_cards import (
    DetectorPreviewCard,
    FittingPlotControlsCard,
    PlotPreviewCard,
    StatusCard,
)
from .run_card import FittingControlsCard
from .workflow_header import FittingWorkflowHeader
from .workflow_content import WorkflowContentStack
from .workflow_navigation import navigate_workflow_step, save_guided_mode
from .workspace_responsiveness import FittingWorkspaceResponsivenessMixin
from .workspace_context import FittingContextContainer


class GisaxsFittingWorkspace(FittingWorkspaceResponsivenessMixin):
    """Three-region layout for the cut/fitting page."""

    SETTINGS_KEY = "gisaxs_fitting_splitter_sizes_v2"
    DEFAULT_WORK_SIZES = [760, 680]

    def __init__(
        self,
        ui,
        profile=None,
        *,
        preferences: UserPreferencesRepository,
        view_model,
    ):
        self.ui = ui
        self.preferences = preferences
        self.view_model = view_model
        self.profile = profile or current_profile(ui.centralwidget)
        self.DEFAULT_WORK_SIZES = list(self.profile.work_sizes)
        self._legacy_fitting_scroll_area = ui.gisaxsFittingPageScrollArea
        self.page_splitter = QSplitter(Qt.Horizontal, ui.gisaxsFittingPage)
        self._workspace_ui = FittingWorkspaceView()
        self._workspace_ui.setupUi(self.page_splitter)
        self.fixed_controls_stack = self._workspace_ui.gisaxsFixedControlsStack
        self.left_shell = self._workspace_ui.gisaxsFittingLeftShell
        self.work_area_contents = self._workspace_ui.gisaxsWorkAreaContents
        self.right_panel = self._workspace_ui.gisaxsRightCollapsiblePanel
        self.preview_scroll_area = self._workspace_ui.gisaxsPreviewScrollArea

        self.work_splitter = QSplitter(Qt.Vertical, ui.gisaxsFittingPage)
        self.work_splitter.setObjectName("gisaxsMainWorkSplitter")
        self.work_splitter.setHandleWidth(8)
        self.work_splitter.setChildrenCollapsible(False)
        self.work_splitter.setOpaqueResize(True)
        self.work_splitter.setMinimumWidth(self._control_min_width())
        self.work_splitter.setMinimumHeight(
            sum(self.DEFAULT_WORK_SIZES) + self.work_splitter.handleWidth()
        )
        self.work_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._detach_preview_widgets()
        self._relax_fixed_sizes()
        self._install_page_splitter()
        self._build_left_work_area()
        self._build_preview_area()
        self._build_workflow_content_stack()
        self._legacy_fitting_scroll_area.deleteLater()
        self._configure_button_responsiveness()
        self._apply_page_overflow_policy()
        self.restore_sizes()
        install_safe_wheel_behavior(self.page_splitter)
        QTimer.singleShot(0, self._refresh_fixed_stack_geometry)

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
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(self._control_min_width())
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
        self.workflow_header = FittingWorkflowHeader(workspace_ui.fittingWorkflowHost)
        workspace_ui.fittingWorkflowHostLayout.addWidget(self.workflow_header)
        self.ui.fittingWorkflowHeader = self.workflow_header

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
        model_parameters_card = ModelParameterCard(self.ui, self.profile)
        cut_line_card = CutLineCard(
            self.ui,
            self.profile,
            view_model=self.view_model,
            preferences=self.preferences,
        )
        fitting_controls_card = FittingControlsCard(
            self.ui,
            self.profile,
            model_parameters_card=model_parameters_card,
            preferences=self.preferences,
        )
        self.model_parameters_card = model_parameters_card
        fitting_controls_card.mode_tabs.currentChanged.connect(
            lambda _index: QTimer.singleShot(
                0,
                lambda: QTimer.singleShot(0, self._refresh_fixed_stack_geometry),
            )
        )
        fitting_controls_card.mode_tabs.currentChanged.connect(
            lambda _index: QTimer.singleShot(40, self._refresh_fixed_stack_geometry)
        )
        workspace_ui.fittingInputContentLayout.addWidget(gisaxs_card)
        workspace_ui.fittingConfigureContentLayout.addWidget(cut_line_card)
        workspace_ui.fittingRunContentLayout.addWidget(fitting_controls_card)
        guided = bool(self.preferences.get("fitting.guided_workflow", True))
        self.workflow_header.set_guided(guided)
        self.workflow_header.guided_changed.connect(
            lambda value: save_guided_mode(self.preferences, value)
        )
        self.workflow_header.step_requested.connect(
            lambda key: navigate_workflow_step(self, key)
        )
        self.cut_line_card = cut_line_card

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

        self.fixed_controls_stack.setMinimumHeight(0)
        self.fixed_controls_stack.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.work_splitter.hide()
        self.work_splitter.setParent(None)

        layout = self.work_area_contents.layout()
        self.work_area_contents.setMinimumHeight(0)
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
            self.ui,
            self.ui.gisaxsInputGraphicsView,
            self.profile,
        )
        self.fitting_plot_card = PlotPreviewCard(
            self.ui,
            self.ui.curvePlotControlWidget,
            self.ui.fitGraphicsView,
            self.profile,
        )
        self.curve_plot_card = self.fitting_plot_card
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

        self.preview_tabs = CurrentPageSizeTabWidget(self.right_panel)
        self.preview_tabs.setObjectName("fittingPreviewTabs")
        self.preview_tabs.setDocumentMode(True)
        self.preview_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        workspace_ui.fittingPreviewAreaLayout.removeWidget(self.fitting_preview_panel)
        workspace_ui.fittingPreviewAreaLayout.removeWidget(self.fitting_results_panel)
        self.preview_tabs.addTab(self.fitting_preview_panel, "Detector")
        self.preview_tabs.addTab(self.fitting_results_panel, "Curve")
        self.inline_feedback = QLabel("", self.right_panel)
        self.inline_feedback.setObjectName("fittingInlineFeedback")
        self.inline_feedback.setProperty("feedbackKind", "error")
        self.inline_feedback.setWordWrap(True)
        self.inline_feedback.setVisible(False)
        workspace_ui.fittingPreviewAreaLayout.insertWidget(0, self.preview_tabs, 1)
        self.preview_tabs.currentChanged.connect(self._sync_preview_page_chrome)
        workspace_ui.fittingPlotAdvancedToggle.toggled.connect(
            lambda _expanded: QTimer.singleShot(
                0, self.preview_tabs.refresh_current_page_geometry
            )
        )
        self._sync_preview_page_chrome(self.preview_tabs.currentIndex())
        self.ui.fittingPreviewTabs = self.preview_tabs
        self.ui.fittingInlineFeedback = self.inline_feedback
        workspace_ui.fittingPreviewTitle.hide()
        workspace_ui.fittingPreviewDescription.hide()
        workspace_ui.fittingResultsTitle.hide()
        workspace_ui.fittingResultsDescription.hide()

        workspace_ui.fittingFixedControlsLayout.removeWidget(self.fitting_export_section)
        self.fitting_export_section.setParent(self.fitting_results_panel)
        workspace_ui.fittingResultsPanelLayout.addWidget(self.fitting_export_section)
        workspace_ui.fittingPreviewAreaLayout.removeWidget(
            self.fitting_plot_advanced_section
        )
        self.fitting_plot_advanced_section.setParent(self.fitting_results_panel)
        workspace_ui.fittingResultsPanelLayout.addWidget(
            self.fitting_plot_advanced_section
        )
        self.fitting_plot_advanced_section.set_expanded(False)
        self.fitting_log_section.set_expanded(False)

        self.right_panel.setMinimumWidth(self._preview_min_width())
        self.right_panel.setMaximumWidth(16777215)
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.preview_scroll_area.setMinimumWidth(self._preview_min_width())
        self.preview_scroll_area.setMaximumWidth(16777215)
        self.ui.gisaxsFittingPageScrollArea.setMaximumWidth(
            self._control_target_width() + scale_value(80, self.profile, 60)
        )
        self.left_shell.setMinimumWidth(self._control_min_width())
        self.left_shell.setMaximumWidth(
            self._control_target_width() + scale_value(80, self.profile, 60)
        )
        self.left_shell.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.page_splitter.setStretchFactor(0, 0)
        self.page_splitter.setStretchFactor(1, 1)
        self.page_splitter.setCollapsible(0, False)
        self.page_splitter.setCollapsible(1, False)
        apply_design_system(self.page_splitter)
        self.page_splitter.setStyleSheet(
            self.page_splitter.styleSheet() + "\n" + fitting_stylesheet()
        )

    def _build_workflow_content_stack(self) -> None:
        """Show one left-side task at a time while preserving every legacy widget."""
        layout = self.fixed_controls_stack.layout()
        for section in (
            self.fitting_input_section,
            self.fitting_configure_section,
            self.fitting_advanced_section,
            self.fitting_run_section,
        ):
            layout.removeWidget(section)

        self.fitting_fit_step_page = QWidget(self.fixed_controls_stack)
        self.fitting_fit_step_page.setObjectName("fittingFitStepPage")
        fit_layout = QVBoxLayout(self.fitting_fit_step_page)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setSpacing(CARD_SPACING)
        if self.fitting_advanced_section.content.layout().count() == 0:
            self.fitting_advanced_section.hide()
        else:
            fit_layout.addWidget(self.fitting_advanced_section)
        fit_layout.addWidget(self.fitting_run_section)
        fit_layout.addStretch(1)

        self.workflow_content_stack = WorkflowContentStack(self.fixed_controls_stack)
        self.workflow_content_stack.setObjectName("fittingWorkflowContentStack")
        self.workflow_content_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.workflow_content_stack.addWidget(self.fitting_input_section)
        self.workflow_content_stack.addWidget(self.fitting_configure_section)
        self.workflow_content_stack.addWidget(self.fitting_fit_step_page)
        layout.insertWidget(0, self.workflow_content_stack)
        layout.setAlignment(Qt.AlignTop)
        self.ui.fittingWorkflowContentStack = self.workflow_content_stack
        self.show_workflow_step("import")
        QTimer.singleShot(0, self.workflow_content_stack.sync_height)

    def _sync_preview_page_chrome(self, tab_index: int) -> None:
        """Keep persistent tab navigation above page-specific controls."""
        index = int(tab_index)
        if index == 0:
            target_layout = self._workspace_ui.fittingPreviewPanelLayout
        else:
            target_layout = self._workspace_ui.fittingResultsPanelLayout

        _detach_from_parent_layout(self.inline_feedback)
        target_layout.insertWidget(0, self.inline_feedback)

        toolbar = self.fitting_plot_card.toolbar
        if index == 1:
            _detach_from_parent_layout(toolbar)
            target_layout.insertWidget(1, toolbar)
            toolbar.show()
        else:
            toolbar.hide()

    def show_workflow_step(self, key: str) -> None:
        """Select navigation independently from verified workflow completion."""
        configure_keys = {"setup", "center", "cut", "center_cut"}
        page_index = 2 if key == "fit" else (1 if key in configure_keys else 0)
        self.workflow_content_stack.setCurrentIndex(page_index)
        if key in configure_keys:
            self.cut_line_card.show_step(key)
        self.workflow_header.set_selected_step(key)
        self.ui.gisaxsFittingPageScrollArea.verticalScrollBar().setValue(0)
        self.workflow_content_stack.sync_height()
        QTimer.singleShot(0, self._refresh_fixed_stack_geometry)

    def _configure_button_responsiveness(self) -> None:
        expanding_actions = [
            "gisaxsInputImportButton",
            "gisaxsInputCenterAutoFindingButton",
            "gisaxsInputDetectorParaButton",
            "gisaxsInputCutButton",
            "fitImport1dFileButton",
            "FittingManualFittingButton",
            "FittingAutoFittingButton",
            "FittingClearFittingButton_2",
            "FittingAutoKButton",
            "FittingExportButton",
        ]
        preferred_actions = [
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
        self.context_container = FittingContextContainer(
            self.page_splitter,
            self.view_model.insitu,
            self.ui.gisaxsFittingPage,
        )
        self.context_bar = self.context_container.context_bar
        self.context_stack = self.context_container.stack
        self.context_button_group = self.context_container.button_group
        self.single_context_button = self.context_container.single_button
        self.insitu_context_button = self.context_container.insitu_button
        self.insitu_series_page = self.context_container.insitu_page
        layout.addWidget(self.context_container, 1)
        self.ui.fittingWorkspace = self
        self.ui.fittingContextStack = self.context_stack
        self.ui.fittingInsituSeriesPage = self.insitu_series_page
        self.ui.fittingSingleContextButton = self.single_context_button
        self.ui.fittingInsituContextButton = self.insitu_context_button
        self.show_context("single")
        self.ui.gisaxsFittingPageScrollArea = (
            self._workspace_ui.gisaxsFittingPageScrollArea
        )
        self.ui.gisaxsFittingPageScrollAreaWidgetContents = (
            self.work_area_contents
        )

    def show_context(self, context: str) -> None:
        """Switch task context without resetting either page's local navigation."""
        self.context_container.show_context(context)

__all__ = ["GisaxsFittingWorkspace"]
