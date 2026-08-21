"""Assemble the modern two-pane Prediction workbench."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QScrollArea, QSplitter, QVBoxLayout, QWidget

from src.gimap.app.presentation import apply_design_system
from src.gimap.app.presentation.components import ParameterSection, PlotPanel
from src.gimap.app.presentation.responsive_layout import scale_value

from .prediction_theme import prediction_stylesheet
from .workflow_components import (
    PredictionCanvasEmptyState,
    PredictionDisclosure,
    PredictionWorkflowHeader,
)


def _detach_from_parent_layout(widget: QWidget) -> None:
    parent = widget.parentWidget()
    if parent is not None and parent.layout() is not None:
        index = parent.layout().indexOf(widget)
        if index != -1:
            parent.layout().takeAt(index)
    widget.setParent(None)


class PredictionWorkbenchLayout:
    """Own layout composition only; prediction commands stay in the ViewModel/bindings."""

    def __init__(
        self,
        ui,
        profile,
        contents: QWidget,
        workspace_ui,
        *,
        input_mode_panel,
        input_section,
        configure_section,
        advanced_section,
        run_section,
        results_section,
        export_section,
        input_card,
        model_card,
        run_card,
        results_card,
    ) -> None:
        self.ui = ui
        self.profile = profile
        self.contents = contents
        self.workspace_ui = workspace_ui
        self.input_mode_panel = input_mode_panel
        self.input_section = input_section
        self.configure_section = configure_section
        self.advanced_section = advanced_section
        self.run_section = run_section
        self.results_section = results_section
        self.export_section = export_section
        self.input_card = input_card
        self.model_card = model_card
        self.run_card = run_card
        self.results_card = results_card
        self._build()

    def _build(self) -> None:
        self._clear_workspace_host()
        self.splitter = QSplitter(Qt.Horizontal, self.contents)
        self.splitter.setObjectName("gisaxsPredictWorkspaceSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)

        self.left_rail = self._build_control_rail()
        self.right_scroll_area = self._build_canvas()
        self.splitter.addWidget(self.left_rail)
        self.splitter.addWidget(self.right_scroll_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self._apply_splitter_sizes()
        self.workspace_ui.predictionWorkspaceLayout.addWidget(self.splitter)

        self.ui.gisaxsPredictOuterScrollArea = self.left_scroll_area
        self.ui.gisaxsPredictCanvasScrollArea = self.right_scroll_area
        self.ui.gisaxsPredictWorkspaceSplitter = self.splitter
        self.ui.predictionWorkbenchLayout = self
        self.input_mode_panel.mode_changed.connect(self._sync_presented_mode)
        self._sync_presented_mode(
            "multi_files"
            if self.ui.gisaxsPredictMultiFilesRadioButton.isChecked()
            else "single_file"
        )
        self.workflow_header.bind(self.ui)
        self.workflow_header.step_requested.connect(self._navigate_to_step)
        apply_design_system(self.contents)
        self.contents.setStyleSheet(
            self.contents.styleSheet() + "\n" + prediction_stylesheet()
        )

    def _clear_workspace_host(self) -> None:
        layout = self.workspace_ui.predictionWorkspaceLayout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def _build_control_rail(self) -> QWidget:
        rail = QFrame(self.splitter)
        rail.setObjectName("predictionControlRail")
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(8)

        left_contents = QWidget(rail)
        left_contents.setObjectName("predictionLeftContents")
        layout = QVBoxLayout(left_contents)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.workflow_header = PredictionWorkflowHeader(left_contents)
        layout.addWidget(self.workflow_header)

        self.workspace_ui.predictionInputTitle.setText("1. Import data")
        self.workspace_ui.predictionInputDescription.setText(
            "Choose one detector file or switch to Folder batch for a numbered sequence."
        )
        self.workspace_ui.predictionConfigureTitle.setText("2. Import model")
        self.workspace_ui.predictionConfigureDescription.setText(
            "Select a prediction setup, then import its compatible trained model."
        )
        self.workspace_ui.predictionRunTitle.setText("3. Predict")
        self.workspace_ui.predictionRunDescription.setText(
            "Check readiness and start the single-file or folder prediction job."
        )
        for section in (
            self.input_section,
            self.configure_section,
            self.advanced_section,
        ):
            section.setParent(left_contents)
            layout.addWidget(section)
        layout.addStretch(1)
        for card in (self.input_card, self.model_card, self.run_card):
            card.header_widget.hide()

        scroll_area = QScrollArea(rail)
        scroll_area.setObjectName("gisaxsPredictOuterScrollArea")
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(left_contents)
        rail_layout.addWidget(scroll_area, 1)

        self.run_section.setParent(rail)
        self.run_section.setProperty("predictionStickyRun", True)
        rail_layout.addWidget(self.run_section, 0)
        rail.setMinimumWidth(self._control_min_width())
        rail.setMaximumWidth(self._control_target_width() + 80)
        self.left_scroll_area = scroll_area
        self.ui.gisaxsPredictControlRail = rail
        return rail

    def _build_canvas(self) -> QScrollArea:
        right_contents = QWidget(self.splitter)
        right_contents.setObjectName("predictionRightContents")
        layout = QVBoxLayout(right_contents)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        _detach_from_parent_layout(self.results_card)
        self.plot_panel = PlotPanel(
            "Prediction canvas",
            "Inspect the detector input, generated 2D output and prediction curves.",
            right_contents,
            empty_title="Import detector data",
            empty_message="Choose a detector file or folder batch to begin.",
        )
        self.plot_panel.setObjectName("predictionPlotPanel")
        self.results_card.setParent(self.plot_panel.plot_host)
        self.results_card.header_widget.hide()
        self.results_card.setMinimumHeight(scale_value(500, self.profile, 390))
        self.ui.gisaxsPredictImageShowTabWidget.setTabText(0, "Input preview")
        self.ui.gisaxsPredictImageShowTabWidget.setTabText(1, "Prediction result")
        self.plot_panel.set_plot_widget(self.results_card)
        self.plot_panel.empty_state.set_content(
            "Import detector data",
            "Choose a detector file or folder batch to begin.",
            "Go to Import data",
        )
        self.plot_panel.empty_state.actionRequested.connect(
            lambda: self._navigate_to_step(1)
        )
        self.canvas_status_label = QLabel("No input loaded", self.plot_panel.toolbar_widget)
        self.canvas_status_label.setObjectName("predictionCanvasStatusLabel")
        self.canvas_status_label.setProperty("cardMeta", True)
        self.plot_panel.add_toolbar_widget(self.canvas_status_label)
        self.plot_panel.add_toolbar_stretch()

        _detach_from_parent_layout(self.ui.gisaxsImageExportButton)
        _detach_from_parent_layout(self.ui.predict2dExportButton)
        self.ui.gisaxsImageExportButton.setText("Export input...")
        self.ui.predict2dExportButton.setText("Export result...")
        self.plot_panel.add_toolbar_widget(self.ui.gisaxsImageExportButton)
        self.plot_panel.add_toolbar_widget(self.ui.predict2dExportButton)
        self.ui.gisaxsPredictImageShowTabWidget.currentChanged.connect(
            self._sync_canvas_actions
        )
        layout.addWidget(self.plot_panel, 1)
        self.input_empty_state = PredictionCanvasEmptyState(
            self.ui.gisaxsImageGraphicsView,
            "Import a detector file or folder batch\nto preview the GISAXS input.",
        )
        self.result_empty_state = PredictionCanvasEmptyState(
            self.ui.predict2dGraphicsView,
            "Run prediction to display the generated\n2D result and parameter curve.",
        )

        self.batch_results_section = ParameterSection(
            "Batch results",
            "Track every prediction job; select a completed row to inspect it in the canvas.",
            right_contents,
        )
        self.batch_results_section.setObjectName("predictionBatchResultsSection")
        self.batch_results_section.setMinimumHeight(scale_value(360, self.profile, 300))
        self.batch_current_file_label = QLabel("No batch result selected", right_contents)
        self.batch_current_file_label.setObjectName("predictionBatchCurrentFileLabel")
        self.batch_current_file_label.setProperty("cardMeta", True)
        self.batch_current_file_label.setWordWrap(True)
        self.batch_results_section.add_widget(self.batch_current_file_label)
        self.batch_results_host = QWidget(self.batch_results_section.content)
        self.batch_results_host.setObjectName("predictionBatchResultsHost")
        self.batch_results_host_layout = QVBoxLayout(self.batch_results_host)
        self.batch_results_host_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_results_host_layout.setSpacing(0)
        self.batch_results_section.add_widget(self.batch_results_host, 1)
        layout.addWidget(self.batch_results_section)

        self.export_section.hide()

        self.workspace_ui.predictionResultsTitle.hide()
        self.workspace_ui.predictionResultsDescription.hide()
        self.ui.gisaxsPredictRunLogTitle.hide()
        self.results_section.setProperty("predictionEmbeddedSection", True)
        self.activity_disclosure = PredictionDisclosure(
            "Activity log", "predictionActivityDisclosure", right_contents
        )
        self.activity_disclosure.add_widget(self.results_section)
        layout.addWidget(self.activity_disclosure)
        self.ui.predictionActivityDisclosure = self.activity_disclosure
        self.ui.predictionPlotPanel = self.plot_panel
        self.ui.predictionCanvasStatusLabel = self.canvas_status_label
        self.ui.predictionBatchResultsSection = self.batch_results_section
        self.ui.predictionBatchResultsHost = self.batch_results_host
        self.ui.predictionBatchResultsHostLayout = self.batch_results_host_layout
        self.ui.predictionBatchCurrentFileLabel = self.batch_current_file_label
        self._sync_canvas_actions(self.ui.gisaxsPredictImageShowTabWidget.currentIndex())
        self.sync_canvas_state(input_ready=False, result_ready=False)

        scroll_area = QScrollArea(self.splitter)
        scroll_area.setObjectName("gisaxsPredictCanvasScrollArea")
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(right_contents)
        scroll_area.setMinimumWidth(self._preview_min_width())
        return scroll_area

    def _sync_canvas_actions(self, index: int) -> None:
        input_tab = self.ui.gisaxsPredictImageShowTabWidget.indexOf(self.ui.gisaxsImageTab)
        is_input = index == input_tab
        self.ui.gisaxsImageExportButton.setVisible(
            is_input and getattr(self, "_canvas_input_ready", False)
        )
        self.ui.predict2dExportButton.setVisible(
            not is_input and getattr(self, "_canvas_result_ready", False)
        )

    def sync_canvas_state(self, *, input_ready: bool, result_ready: bool) -> None:
        self._canvas_input_ready = bool(input_ready)
        self._canvas_result_ready = bool(result_ready)
        if not input_ready:
            self.plot_panel.show_empty(
                "Import detector data",
                "Choose a detector file or folder batch to begin.",
            )
            self.canvas_status_label.setText("No input loaded")
        else:
            self.plot_panel.show_plot()
            self.canvas_status_label.setText(
                "Prediction ready" if result_ready else "Input ready · model/prediction pending"
            )
        self.ui.gisaxsImageParametersWidget.setVisible(input_ready)
        self.ui.predict2dParameterWidget.setVisible(result_ready)
        self.ui.gisaxsImageExportButton.setEnabled(input_ready)
        self.ui.predict2dExportButton.setEnabled(result_ready)
        self._sync_canvas_actions(self.ui.gisaxsPredictImageShowTabWidget.currentIndex())

    def _sync_presented_mode(self, mode: str) -> None:
        is_batch = mode == "multi_files"
        self.ui.gisaxsPredictShowMultiFileResultsButton.setVisible(is_batch)
        self.batch_results_section.setVisible(is_batch)
        self.activity_disclosure.toggle.setText("Activity log")

    def focus_batch_results(self) -> None:
        self.batch_results_section.show()
        self.right_scroll_area.ensureWidgetVisible(self.batch_results_section, 0, 12)

    def _navigate_to_step(self, step: int) -> None:
        target = {
            1: self.input_section,
            2: self.configure_section,
            3: self.run_section,
        }.get(int(step))
        if target is not None:
            if step != 3:
                self.left_scroll_area.ensureWidgetVisible(target, 0, 12)
            target.setFocus(Qt.OtherFocusReason)

    def _control_min_width(self) -> int:
        return {"compact": 420, "normal": 460, "wide": 500}.get(
            self.profile.key, 460
        )

    def _control_target_width(self) -> int:
        return {"compact": 500, "normal": 540, "wide": 580}.get(
            self.profile.key, 540
        )

    def _preview_min_width(self) -> int:
        return max(self.profile.preview_min, scale_value(480, self.profile, 400))

    def _apply_splitter_sizes(self) -> None:
        self.splitter.setSizes(
            [self._control_target_width(), max(self._preview_min_width(), 900)]
        )

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        self.left_rail.setMinimumWidth(self._control_min_width())
        self.left_rail.setMaximumWidth(self._control_target_width() + 80)
        self.right_scroll_area.setMinimumWidth(self._preview_min_width())
        self._apply_splitter_sizes()


__all__ = ["PredictionWorkbenchLayout"]
