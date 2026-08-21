"""Feature-owned layout wrapper for the Prediction page controls。"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)
from src.gimap.app.presentation.layout_primitives import (
    CARD_SPACING,
    FORM_ROW_SPACING,
    normalize_button,
)
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .cards import PredictCard, PredictModelLibraryCard
from .control_style import apply_prediction_control_style
from .views import PredictionWorkspaceView
from .preview_layout import PredictionPreviewLayout
from .workbench_layout import PredictionWorkbenchLayout
from .workflow_components import (
    PredictionDisclosure,
    PredictionInputModePanel,
)


def _take_widget(layout, widget: QWidget) -> None:
    index = layout.indexOf(widget)
    if index != -1:
        layout.takeAt(index)
    widget.setParent(None)


def _detach_from_parent_layout(widget: QWidget) -> None:
    parent = widget.parentWidget()
    if parent is not None and parent.layout() is not None:
        _take_widget(parent.layout(), widget)
    else:
        widget.setParent(None)


class GisaxsPredictWorkspace:
    """Reorganize existing Prediction controls without owning prediction workflow。"""

    def __init__(self, ui, profile=None) -> None:
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        self._preview_layout: PredictionPreviewLayout | None = None
        self._build()

    def _build(self) -> None:
        page = self.ui.gisaxsPredictPage
        source_layout = getattr(self.ui, "verticalLayout_16", None)
        if (
            source_layout is None
            or page.findChild(QWidget, "gisaxsPredictWorkspaceSplitter") is not None
        ):
            return

        for widget in (self.ui.widget_2, self.ui.gisaxsPredictImageShowWidget):
            _take_widget(source_layout, widget)
        apply_prediction_control_style(self.ui, self.profile)

        contents = QWidget(page)
        workspace_ui = PredictionWorkspaceView()
        workspace_ui.setupUi(contents)
        self._workspace_ui = workspace_ui
        self.prediction_input_section = workspace_ui.predictionInputSection
        self.prediction_configure_section = workspace_ui.predictionConfigureSection
        self.prediction_advanced_section = workspace_ui.predictionAdvancedSection
        self.prediction_preview_panel = workspace_ui.predictionPreviewPanel
        self.prediction_run_section = workspace_ui.predictionRunSection
        self.prediction_results_section = workspace_ui.predictionResultsSection
        self.prediction_export_section = workspace_ui.predictionExportSection

        for section, title, description, content, layout in (
            (
                self.prediction_input_section,
                workspace_ui.predictionInputTitle,
                workspace_ui.predictionInputDescription,
                workspace_ui.predictionInputContent,
                workspace_ui.predictionInputContentLayout,
            ),
            (
                self.prediction_configure_section,
                workspace_ui.predictionConfigureTitle,
                workspace_ui.predictionConfigureDescription,
                workspace_ui.predictionConfigureContent,
                workspace_ui.predictionConfigureContentLayout,
            ),
            (
                self.prediction_preview_panel,
                workspace_ui.predictionPreviewTitle,
                workspace_ui.predictionPreviewDescription,
                workspace_ui.predictionPreviewContent,
                workspace_ui.predictionPreviewContentLayout,
            ),
            (
                self.prediction_run_section,
                workspace_ui.predictionRunTitle,
                workspace_ui.predictionRunDescription,
                workspace_ui.predictionRunContent,
                workspace_ui.predictionRunContentLayout,
            ),
            (
                self.prediction_results_section,
                workspace_ui.predictionResultsTitle,
                workspace_ui.predictionResultsDescription,
                workspace_ui.predictionResultsContent,
                workspace_ui.predictionResultsContentLayout,
            ),
            (
                self.prediction_export_section,
                workspace_ui.predictionExportTitle,
                workspace_ui.predictionExportDescription,
                workspace_ui.predictionExportContent,
                workspace_ui.predictionExportContentLayout,
            ),
        ):
            bind_parameter_section(section, title, description, content, layout)
        bind_advanced_section(
            self.prediction_advanced_section,
            workspace_ui.predictionAdvancedToggle,
            workspace_ui.predictionAdvancedDescription,
            workspace_ui.predictionAdvancedContent,
            workspace_ui.predictionAdvancedContentLayout,
        )
        self.prediction_results_section.header_actions = (
            workspace_ui.predictionResultsHeaderActionsLayout
        )

        self.input_card = self._build_input_card(contents)
        self.model_card = self._build_model_card(contents)
        self.run_card = self._build_run_card(contents)
        self.results_card = self._build_results_card(contents)
        self.model_library_card = PredictModelLibraryCard(contents, self.profile)
        self.model_library_card.set_expanded(True)
        workspace_ui.predictionInputContentLayout.addWidget(self.input_card)
        workspace_ui.predictionConfigureContentLayout.addWidget(self.model_card)
        workspace_ui.predictionAdvancedContentLayout.addWidget(self.model_library_card)
        workspace_ui.predictionPreviewContentLayout.addWidget(self.results_card)
        workspace_ui.predictionRunContentLayout.addWidget(self.run_card)

        _detach_from_parent_layout(self.ui.gisaxsPredictRunLogTitle)
        _detach_from_parent_layout(self.ui.predictStatusTextBrowser)
        workspace_ui.predictionResultsContentLayout.addWidget(
            self.ui.gisaxsPredictRunLogTitle
        )
        workspace_ui.predictionResultsContentLayout.addWidget(
            self.ui.predictStatusTextBrowser
        )

        for output_section in (
            self.gisaxs_preview_output_section,
            self.predict2d_preview_output_section,
        ):
            output_section.setVisible(False)
        _detach_from_parent_layout(self.ui.gisaxsImageExportButton)
        _detach_from_parent_layout(self.ui.predict2dExportButton)
        workspace_ui.predictionExportContentLayout.addWidget(
            self.ui.gisaxsImageExportButton
        )
        workspace_ui.predictionExportContentLayout.addWidget(
            self.ui.predict2dExportButton
        )
        workspace_ui.predictionExportContentLayout.addStretch(1)
        self.workbench_layout = PredictionWorkbenchLayout(
            self.ui,
            self.profile,
            contents,
            workspace_ui,
            input_mode_panel=self.input_mode_panel,
            input_section=self.prediction_input_section,
            configure_section=self.prediction_configure_section,
            advanced_section=self.prediction_advanced_section,
            run_section=self.prediction_run_section,
            results_section=self.prediction_results_section,
            export_section=self.prediction_export_section,
            input_card=self.input_card,
            model_card=self.model_card,
            run_card=self.run_card,
            results_card=self.results_card,
        )
        self._expose_workbench_layout()
        source_layout.addWidget(contents)

        self.ui.gisaxsPredictInputCard = self.input_card
        self.ui.gisaxsPredictModelCard = self.model_card
        self.ui.gisaxsPredictRunCard = self.run_card
        self.ui.gisaxsPredictResultsCard = self.results_card
        self.ui.predictModelLibraryCard = self.model_library_card
    def _build_input_card(self, parent: QWidget) -> PredictCard:
        card = PredictCard("Import data", "GisaxsPredictInputCard", parent)

        for widget in (
            self.ui.gisaxsPredictSingleFileRadioButton,
            self.ui.gisaxsPredictMultiFilesRadioButton,
            self.ui.gisaxsPredictChooseGisaxsFileButton,
            self.ui.gisaxsPredictChooseGisaxsFileValue,
            self.ui.gisaxsPredictChooseFolderButton,
            self.ui.gisaxsPredictChooseFolderValue,
            self.ui.widget_5,
            self.ui.gisaxsPredictStackLabel,
            self.ui.gisaxsPredictStackValue,
            self.ui.gisaxsPredictEveryLabel,
            self.ui.gisaxsPredictEveryValue,
        ):
            _detach_from_parent_layout(widget)
        self.ui.widget_5.setVisible(False)

        self.ui.gisaxsPredictShowMultiFileResultsButton = QPushButton(
            "Open batch results", card
        )
        self.ui.gisaxsPredictShowMultiFileResultsButton.setObjectName(
            "gisaxsPredictShowMultiFileResultsButton"
        )
        normalize_button(self.ui.gisaxsPredictShowMultiFileResultsButton, wide=True)
        self.input_mode_panel = PredictionInputModePanel(self.ui, card.content_widget)
        self.ui.predictionInputModePanel = self.input_mode_panel
        card.add_content(self.input_mode_panel)
        return card

    def _build_model_card(self, parent: QWidget) -> PredictCard:
        card = PredictCard("Import model", "GisaxsPredictModelCard", parent)
        form = QWidget(card.content_widget)
        grid = QGridLayout(form)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(CARD_SPACING)
        grid.setVerticalSpacing(FORM_ROW_SPACING)

        for widget in (
            self.ui.gisaxsPredictModuleSelectLabel,
            self.ui.gisaxsPredictModuleSelectCombox,
            self.ui.gisaxsPredictFrameworkLabel,
            self.ui.gisaxsPredictFrameworkCombox,
            self.ui.widget_4,
        ):
            _detach_from_parent_layout(widget)
        for widget in (
            self.ui.gisaxsPredictEditButton,
            self.ui.gisaxsPredictModelImportButton,
        ):
            _detach_from_parent_layout(widget)
        self.ui.widget_4.hide()

        self.ui.gisaxsPredictModelStatusTextLabel = QLabel("Not loaded", form)
        self.ui.gisaxsPredictModelStatusTextLabel.setObjectName("gisaxsPredictModelStatusTextLabel")
        self.ui.gisaxsPredictModelStatusTextLabel.setProperty("predictionModelStatus", True)
        self.ui.gisaxsPredictModelStatusTextLabel.setProperty("modelState", "idle")
        self.ui.gisaxsPredictFrameworkStatusLabel = QLabel("Framework: checking...", form)
        self.ui.gisaxsPredictFrameworkStatusLabel.setObjectName("gisaxsPredictFrameworkStatusLabel")
        self.ui.gisaxsPredictFrameworkStatusLabel.setProperty("cardMeta", True)
        if not hasattr(self.ui, "gisaxsPredictReloadConfigButton"):
            self.ui.gisaxsPredictReloadConfigButton = QPushButton("Reload Config", form)
            self.ui.gisaxsPredictReloadConfigButton.setObjectName("gisaxsPredictReloadConfigButton")
            normalize_button(self.ui.gisaxsPredictReloadConfigButton)
        else:
            self.ui.gisaxsPredictReloadConfigButton.setParent(form)
        self.ui.gisaxsPredictModuleSelectLabel.setText("Prediction setup")
        self.ui.gisaxsPredictModelImportButton.setText("Import model...")
        self.ui.gisaxsPredictModelImportButton.setProperty("gimapPrimaryAction", True)
        grid.addWidget(self.ui.gisaxsPredictModuleSelectLabel, 0, 0)
        grid.addWidget(self.ui.gisaxsPredictModuleSelectCombox, 0, 1)
        grid.addWidget(self.ui.gisaxsPredictModelImportButton, 1, 0, 1, 2)
        grid.addWidget(self.ui.gisaxsPredictModelStatusTextLabel, 2, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        card.add_content(form)

        technical = PredictionDisclosure(
            "Advanced model configuration",
            "predictionTechnicalModelDisclosure",
            card.content_widget,
        )
        technical_form = QWidget(technical.content)
        technical_grid = QGridLayout(technical_form)
        technical_grid.setContentsMargins(0, 0, 0, 0)
        technical_grid.setHorizontalSpacing(CARD_SPACING)
        technical_grid.setVerticalSpacing(FORM_ROW_SPACING)
        technical_grid.addWidget(self.ui.gisaxsPredictFrameworkLabel, 0, 0)
        technical_grid.addWidget(self.ui.gisaxsPredictFrameworkCombox, 0, 1)
        technical_grid.addWidget(self.ui.gisaxsPredictFrameworkStatusLabel, 1, 0, 1, 2)
        actions = QHBoxLayout()
        actions.addWidget(self.ui.gisaxsPredictReloadConfigButton)
        actions.addWidget(self.ui.gisaxsPredictEditButton)
        actions.addStretch(1)
        technical_grid.addLayout(actions, 2, 0, 1, 2)
        technical_grid.setColumnStretch(1, 1)
        technical.add_widget(technical_form)
        card.add_content(technical)
        self.ui.predictionTechnicalModelDisclosure = technical
        return card

    def _build_run_card(self, parent: QWidget) -> PredictCard:
        card = PredictCard("Run / Prediction", "GisaxsPredictRunCard", parent)
        run = QWidget(card.content_widget)
        layout = QVBoxLayout(run)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_SPACING)

        _detach_from_parent_layout(self.ui.gisaxsPredictPredictButton)
        _detach_from_parent_layout(self.ui.predictStatusTextBrowser)
        self.ui.gisaxsPredictStopButton = QPushButton("Stop", run)
        self.ui.gisaxsPredictStopButton.setObjectName("gisaxsPredictStopButton")
        self.ui.gisaxsPredictStopButton.setEnabled(False)
        self.ui.gisaxsPredictStopButton.setVisible(False)
        normalize_button(self.ui.gisaxsPredictStopButton)

        status_grid = QGridLayout()
        status_grid.setContentsMargins(0, 0, 0, 0)
        status_grid.setHorizontalSpacing(CARD_SPACING)
        status_grid.setVerticalSpacing(4)
        self.ui.gisaxsPredictInputReadyLabel = QLabel("Input: Missing", run)
        self.ui.gisaxsPredictModelReadyLabel = QLabel("Model: Not loaded", run)
        self.ui.gisaxsPredictFrameworkReadyLabel = QLabel("Framework: Checking", run)
        self.ui.gisaxsPredictModeLabel = QLabel("Mode: Single File", run)
        for label in (
            self.ui.gisaxsPredictInputReadyLabel,
            self.ui.gisaxsPredictModelReadyLabel,
            self.ui.gisaxsPredictFrameworkReadyLabel,
            self.ui.gisaxsPredictModeLabel,
        ):
            label.setProperty("predictionReadiness", True)
        status_grid.addWidget(self.ui.gisaxsPredictInputReadyLabel, 0, 0)
        status_grid.addWidget(self.ui.gisaxsPredictModelReadyLabel, 0, 1)
        status_grid.addWidget(self.ui.gisaxsPredictFrameworkReadyLabel, 1, 0)
        status_grid.addWidget(self.ui.gisaxsPredictModeLabel, 1, 1)

        self.ui.gisaxsPredictPredictButton.setProperty("gimapPrimaryAction", True)
        self.ui.gisaxsPredictStopButton.setProperty("gimapDangerAction", True)
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addWidget(self.ui.gisaxsPredictPredictButton, 1)
        button_row.addWidget(self.ui.gisaxsPredictStopButton)

        log_title = QLabel("Run Log", run)
        log_title.setObjectName("gisaxsPredictRunLogTitle")
        log_title.setProperty("sectionTitle", True)
        self.ui.gisaxsPredictRunLogTitle = log_title

        layout.addLayout(status_grid)
        readiness_hint = QLabel(
            "Prediction becomes available when input, model and framework are ready.", run
        )
        readiness_hint.setProperty("cardMeta", True)
        readiness_hint.setWordWrap(True)
        layout.addWidget(readiness_hint)
        layout.addLayout(button_row)
        layout.addWidget(log_title)
        layout.addWidget(self.ui.predictStatusTextBrowser)

        card.add_content(run)
        return card

    def _build_results_card(self, parent: QWidget) -> PredictCard:
        card = PredictCard("Results / Preview", "GisaxsPredictResultsCard", parent)
        _detach_from_parent_layout(self.ui.gisaxsPredictImageShowWidget)
        self._preview_layout = PredictionPreviewLayout(self.ui, self.profile)
        self._preview_layout.rebuild()
        self.gisaxs_preview_output_section = self._preview_layout.gisaxs_output_section
        self.predict2d_preview_output_section = self._preview_layout.predict2d_output_section
        if (
            self.gisaxs_preview_output_section is None
            or self.predict2d_preview_output_section is None
        ):
            raise RuntimeError("Prediction preview output sections were not created")
        self.ui.gisaxsPredictImageShowWidget.setParent(card.content_widget)
        card.setMinimumHeight(scale_value(520, self.profile, 420))
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.add_content(self.ui.gisaxsPredictImageShowWidget, 1)
        return card

    def _expose_workbench_layout(self) -> None:
        self.page_splitter = self.workbench_layout.splitter
        self.left_rail = self.workbench_layout.left_rail
        self.left_scroll_area = self.workbench_layout.left_scroll_area
        self.right_scroll_area = self.workbench_layout.right_scroll_area
        self.workflow_header = self.workbench_layout.workflow_header
        self.activity_disclosure = self.workbench_layout.activity_disclosure
        self.input_empty_state = self.workbench_layout.input_empty_state
        self.result_empty_state = self.workbench_layout.result_empty_state

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        apply_prediction_control_style(self.ui, self.profile)
        if hasattr(self, "workbench_layout"):
            self.workbench_layout.apply_responsive_profile(profile)
__all__ = ["GisaxsPredictWorkspace"]
