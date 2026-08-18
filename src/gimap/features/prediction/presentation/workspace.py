"""Feature-owned layout wrapper for the Prediction page controls。"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation import apply_design_system
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)
from src.gimap.app.presentation.layout_primitives import (
    CARD_SPACING,
    FORM_ROW_SPACING,
    SECTION_MIN_WIDTH,
    make_scroll_area,
    normalize_button,
)
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .cards import PredictCard, PredictModelLibraryCard
from .control_style import apply_prediction_control_style
from .views import PredictionWorkspaceView
from .preview_layout import PredictionPreviewLayout


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
            or page.findChild(QWidget, "gisaxsPredictOuterScrollArea") is not None
        ):
            return

        for widget in (self.ui.widget_2, self.ui.gisaxsPredictImageShowWidget):
            _take_widget(source_layout, widget)
        self._relax_predict_sizes()

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
        _detach_from_parent_layout(self.ui.gisaxsPredictShowMultiFileResultsButton)
        workspace_ui.predictionResultsHeaderActionsLayout.addWidget(
            self.ui.gisaxsPredictShowMultiFileResultsButton
        )
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
        apply_design_system(contents)

        scroll_area = make_scroll_area(contents, horizontal=True)
        scroll_area.setObjectName("gisaxsPredictOuterScrollArea")
        scroll_area.setMinimumWidth(SECTION_MIN_WIDTH)
        source_layout.addWidget(scroll_area)

        self.ui.gisaxsPredictOuterScrollArea = scroll_area
        self.ui.gisaxsPredictInputCard = self.input_card
        self.ui.gisaxsPredictModelCard = self.model_card
        self.ui.gisaxsPredictRunCard = self.run_card
        self.ui.gisaxsPredictResultsCard = self.results_card
        self.ui.predictModelLibraryCard = self.model_library_card
    def _relax_predict_sizes(self) -> None:
        apply_prediction_control_style(self.ui, self.profile)

    def _build_input_card(self, parent: QWidget) -> PredictCard:
        card = PredictCard("Input", "GisaxsPredictInputCard", parent)
        form = QWidget(card.content_widget)
        grid = QGridLayout(form)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(CARD_SPACING)
        grid.setVerticalSpacing(FORM_ROW_SPACING)

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
            "Show Multi-File Results", form
        )
        self.ui.gisaxsPredictShowMultiFileResultsButton.setObjectName(
            "gisaxsPredictShowMultiFileResultsButton"
        )
        normalize_button(self.ui.gisaxsPredictShowMultiFileResultsButton, wide=True)

        mode_row = QWidget(form)
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(CARD_SPACING)
        mode_layout.addWidget(self.ui.gisaxsPredictSingleFileRadioButton)
        mode_layout.addWidget(self.ui.gisaxsPredictMultiFilesRadioButton)
        mode_layout.addStretch(1)

        range_panel = QFrame(form)
        range_panel.setObjectName("gisaxsPredictRangePanel")
        range_panel.setStyleSheet(
            """
            QFrame#gisaxsPredictRangePanel {
                background: #f8fafc;
                border: 1px solid #dbe3ec;
                border-radius: 8px;
                padding: 4px;
            }
            QLabel { color: #334155; font-weight: 600; }
            """
        )
        range_layout = QHBoxLayout(range_panel)
        range_layout.setContentsMargins(8, 6, 8, 6)
        range_layout.setSpacing(8)
        self.ui.gisaxsPredictStackLabel.setMinimumWidth(scale_value(48, self.profile, 42))
        self.ui.gisaxsPredictEveryLabel.setMinimumWidth(scale_value(44, self.profile, 38))
        self.ui.gisaxsPredictStackValue.setMinimumWidth(scale_value(180, self.profile, 150))
        self.ui.gisaxsPredictStackValue.setMaximumWidth(16777215)
        self.ui.gisaxsPredictStackValue.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ui.gisaxsPredictEveryValue.setMinimumWidth(scale_value(72, self.profile, 64))
        self.ui.gisaxsPredictEveryValue.setMaximumWidth(scale_value(96, self.profile, 84))
        range_layout.addWidget(self.ui.gisaxsPredictStackLabel)
        range_layout.addWidget(self.ui.gisaxsPredictStackValue, 1)
        range_layout.addWidget(self.ui.gisaxsPredictEveryLabel)
        range_layout.addWidget(self.ui.gisaxsPredictEveryValue)

        hint = QLabel("Inclusive range. Every = files stacked per prediction.", form)
        hint.setObjectName("gisaxsPredictRangeHintLabel")
        hint.setProperty("cardMeta", True)
        hint.setWordWrap(True)

        grid.addWidget(QLabel("Mode:", form), 0, 0)
        grid.addWidget(mode_row, 0, 1, 1, 2)
        grid.addWidget(self.ui.gisaxsPredictChooseGisaxsFileButton, 1, 0)
        grid.addWidget(self.ui.gisaxsPredictChooseGisaxsFileValue, 1, 1, 1, 2)
        grid.addWidget(self.ui.gisaxsPredictChooseFolderButton, 2, 0)
        grid.addWidget(self.ui.gisaxsPredictChooseFolderValue, 2, 1, 1, 2)
        grid.addWidget(range_panel, 3, 0, 1, 2)
        grid.addWidget(self.ui.gisaxsPredictShowMultiFileResultsButton, 3, 2)
        grid.addWidget(hint, 4, 0, 1, 3)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        card.add_content(form)
        return card

    def _build_model_card(self, parent: QWidget) -> PredictCard:
        card = PredictCard("Model", "GisaxsPredictModelCard", parent)
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

        self.ui.gisaxsPredictModelStatusTextLabel = QLabel("Not loaded", form)
        self.ui.gisaxsPredictModelStatusTextLabel.setObjectName("gisaxsPredictModelStatusTextLabel")
        self.ui.gisaxsPredictModelStatusTextLabel.setProperty("cardMeta", True)
        self.ui.gisaxsPredictFrameworkStatusLabel = QLabel("Framework: checking...", form)
        self.ui.gisaxsPredictFrameworkStatusLabel.setObjectName("gisaxsPredictFrameworkStatusLabel")
        self.ui.gisaxsPredictFrameworkStatusLabel.setProperty("cardMeta", True)
        if not hasattr(self.ui, "gisaxsPredictReloadConfigButton"):
            self.ui.gisaxsPredictReloadConfigButton = QPushButton("Reload Config", form)
            self.ui.gisaxsPredictReloadConfigButton.setObjectName("gisaxsPredictReloadConfigButton")
            normalize_button(self.ui.gisaxsPredictReloadConfigButton)
        else:
            self.ui.gisaxsPredictReloadConfigButton.setParent(form)

        grid.addWidget(self.ui.gisaxsPredictModuleSelectLabel, 0, 0)
        grid.addWidget(self.ui.gisaxsPredictModuleSelectCombox, 0, 1)
        grid.addWidget(self.ui.gisaxsPredictReloadConfigButton, 0, 2)
        grid.addWidget(self.ui.gisaxsPredictFrameworkLabel, 1, 0)
        grid.addWidget(self.ui.gisaxsPredictFrameworkCombox, 1, 1)
        grid.addWidget(self.ui.gisaxsPredictFrameworkStatusLabel, 1, 2)
        grid.addWidget(QLabel("Model:", form), 2, 0)
        grid.addWidget(self.ui.gisaxsPredictModelStatusTextLabel, 2, 1)
        grid.addWidget(self.ui.widget_4, 2, 2)
        grid.setColumnStretch(1, 1)
        card.add_content(form)
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
            label.setProperty("cardMeta", True)
        status_grid.addWidget(self.ui.gisaxsPredictInputReadyLabel, 0, 0)
        status_grid.addWidget(self.ui.gisaxsPredictModelReadyLabel, 0, 1)
        status_grid.addWidget(self.ui.gisaxsPredictFrameworkReadyLabel, 1, 0)
        status_grid.addWidget(self.ui.gisaxsPredictModeLabel, 1, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.ui.gisaxsPredictPredictButton)
        button_row.addWidget(self.ui.gisaxsPredictStopButton)

        log_title = QLabel("Run Log", run)
        log_title.setObjectName("gisaxsPredictRunLogTitle")
        log_title.setProperty("sectionTitle", True)
        self.ui.gisaxsPredictRunLogTitle = log_title

        layout.addLayout(status_grid)
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

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        self._relax_predict_sizes()


__all__ = ["GisaxsPredictWorkspace"]
