"""Component layer for the generated main window.

The generated ``Ui_MainWindow`` still creates the individual controls.  These
classes only reorganize those controls into named, testable pieces without
changing object names that controllers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.user_settings import user_settings
from ui.layout_utils import (
    BUTTON_HEIGHT,
    CARD_MARGIN,
    CARD_SPACING,
    FORM_ROW_SPACING,
    INPUT_WIDGET_TYPES,
    SECTION_MIN_WIDTH,
    make_scroll_area,
    normalize_button,
    normalize_checkbox,
    normalize_input,
    set_expanding_x,
)
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
            normalize_button(button, wide=True)
            button.setMinimumHeight(BUTTON_HEIGHT + 6)
            button.setMaximumHeight(BUTTON_HEIGHT + 12)
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


class CardFrame(QFrame):
    """Modern card wrapper for existing generated widgets."""

    def __init__(self, title: str, object_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("card", True)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(CARD_MARGIN, 12, CARD_MARGIN, CARD_MARGIN)
        self.body_layout.setSpacing(CARD_SPACING)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("cardTitle", True)
        self.body_layout.addWidget(self.title_label)

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        widget.setParent(self)
        self.body_layout.addWidget(widget, stretch)


class GisaxsInputCard(CardFrame):
    def __init__(self, content: QWidget):
        super().__init__("GISAXS Input", "GisaxsInputCard")
        self.setMinimumHeight(160)
        content.setTitle("")
        self.add_content(content)


class CutLineCard(CardFrame):
    def __init__(self, ui):
        super().__init__("Cut Line and Detector", "CutLineCard")
        self.setMinimumHeight(160)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(CARD_SPACING)
        grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.body_layout.addLayout(grid)

        widgets = [
            ui.gisaxsInputCutLineLabel,
            ui.gisaxsInputCutLineVerticalLabel,
            ui.gisaxsInputCutLineVerticalValue,
            ui.gisaxsInputCutLineParallelLabel,
            ui.gisaxsInputCutLineParallelValue,
            ui.gisaxsInputCenterLabel,
            ui.gisaxsInputCutLineCenterWidget,
            ui.gisaxsInputCenterAutoFindingButton,
            ui.gisaxsInputDetectorParaButton,
            ui.gisaxsInputCutButton,
        ]
        for widget in widgets:
            _take_widget(ui.gridLayout_23, widget)

        grid.addWidget(ui.gisaxsInputCutLineLabel, 0, 0)
        grid.addWidget(ui.gisaxsInputCutLineVerticalLabel, 1, 0)
        grid.addWidget(ui.gisaxsInputCutLineVerticalValue, 1, 1)
        grid.addWidget(ui.gisaxsInputCutLineParallelLabel, 1, 2)
        grid.addWidget(ui.gisaxsInputCutLineParallelValue, 1, 3)
        grid.addWidget(ui.gisaxsInputCenterLabel, 2, 0)
        grid.addWidget(ui.gisaxsInputCutLineCenterWidget, 2, 1, 1, 2)
        grid.addWidget(ui.gisaxsInputCenterAutoFindingButton, 2, 3)
        grid.addWidget(ui.gisaxsInputDetectorParaButton, 3, 0)
        grid.addWidget(ui.gisaxsInputCutButton, 3, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)


class FittingControlsCard(CardFrame):
    def __init__(self, ui):
        super().__init__("Fitting Controls", "FittingControlsCard")
        self.setMinimumHeight(220)
        widgets = [
            ui.fitCurrentDataCheckBox,
            ui.widget,
            ui.fitImport1dFileButton,
            ui.fitImport1dFileValue,
            ui.fitMethodWidget,
            ui.fitMethodWidget_2,
            ui.widget_8,
        ]
        for widget in widgets:
            _take_widget(ui.gridLayout_24, widget)
            widget.setMaximumWidth(16777215)
            set_expanding_x(widget)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(CARD_SPACING)
        grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.body_layout.addLayout(grid)
        grid.addWidget(ui.fitCurrentDataCheckBox, 0, 0)
        grid.addWidget(ui.widget, 0, 1, 1, 2)
        grid.addWidget(ui.fitImport1dFileButton, 1, 0)
        grid.addWidget(ui.fitImport1dFileValue, 1, 1, 1, 2)
        grid.addWidget(ui.fitMethodWidget, 2, 0, 1, 2)
        grid.addWidget(ui.fitMethodWidget_2, 2, 2)
        grid.addWidget(ui.widget_8, 3, 0, 1, 3)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)


class ModelParameterCard(CardFrame):
    def __init__(self, ui):
        super().__init__("Model Parameters", "ModelParameterCard")
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        _take_widget(ui.gridLayout_24, ui.widget_7)
        ui.widget_7.setMaximumWidth(16777215)
        ui.widget_7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.model_scroll_area = make_scroll_area(ui.widget_7, horizontal=True)
        self.model_scroll_area.setObjectName("modelParametersScrollArea")
        self.body_layout.addWidget(self.model_scroll_area, 1)


class DetectorPreviewCard(CardFrame):
    def __init__(self, graphics_view: QGraphicsView):
        super().__init__("Detector Preview", "DetectorPreviewCard")
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(260)
        graphics_view.setMinimumSize(320, 240)
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(graphics_view, 1)


class PlotCanvasArea(QFrame):
    def __init__(self, graphics_view: QGraphicsView, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("plotCanvasContainer")
        self.setProperty("previewSection", True)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        graphics_view.setMinimumSize(320, 260)
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(graphics_view, 1)


class SectionCard(QFrame):
    """Small card-style section with a QLabel title instead of QGroupBox title."""

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        fixed_height: int | None = None,
    ):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("sectionCard", True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed if fixed_height else QSizePolicy.Preferred)
        if fixed_height is not None:
            self.setMinimumHeight(fixed_height)
            self.setMaximumHeight(fixed_height)

        self.section_layout = QVBoxLayout(self)
        self.section_layout.setContentsMargins(12, 10, 12, 12)
        self.section_layout.setSpacing(CARD_SPACING)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("sectionTitle", True)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.section_layout.addWidget(self.title_label)


class FittingRegionControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None):
        super().__init__("Fitting Region", "FittingRegionControl", parent, fixed_height=148)

        for widget in (
            ui.fitFittingRegionLabel,
            ui.fitFittingRegionSlider,
            ui.fitFittingRegionMinValue,
            ui.fitFittingRegionMaxValue,
        ):
            _detach_from_parent_layout(widget)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(CARD_SPACING)
        layout.setVerticalSpacing(FORM_ROW_SPACING)
        self.section_layout.addLayout(layout)

        ui.fitFittingRegionLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        ui.fitFittingRegionSlider.setMinimumHeight(28)
        ui.fitFittingRegionSlider.setMaximumHeight(36)
        ui.fitFittingRegionSlider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        normalize_input(ui.fitFittingRegionMinValue)
        normalize_input(ui.fitFittingRegionMaxValue)

        layout.addWidget(ui.fitFittingRegionLabel, 0, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionSlider, 1, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionMinValue, 2, 0)
        layout.addWidget(ui.fitFittingRegionMaxValue, 2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)


class PlotSamplingControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None):
        super().__init__("Sampling", "PlotSamplingControl", parent, fixed_height=124)

        for widget in (
            ui.fitDataPointsNumLabel,
            ui.fitDataPointsNumValue,
            ui.fitInterpolationMethodLabel,
            ui.fitInterpolationMethodValue,
        ):
            _detach_from_parent_layout(widget)

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(CARD_SPACING)
        layout.setVerticalSpacing(FORM_ROW_SPACING)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.section_layout.addLayout(layout)

        for label in (ui.fitDataPointsNumLabel, ui.fitInterpolationMethodLabel):
            label.setMinimumWidth(130)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        normalize_input(ui.fitDataPointsNumValue)
        normalize_input(ui.fitInterpolationMethodValue)

        layout.addRow(ui.fitDataPointsNumLabel, ui.fitDataPointsNumValue)
        layout.addRow(ui.fitInterpolationMethodLabel, ui.fitInterpolationMethodValue)


class PlotOptionsControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None):
        super().__init__("Display Options", "PlotOptionsControl", parent, fixed_height=156)

        checkboxes = (
            ui.fitBGShowCheckBox,
            ui.fitParticle1ShowCheckBox,
            ui.fitResShowCheckBox,
            ui.fitParticle2ShowCheckBox,
            ui.fitParticle3ShowCheckBox,
        )
        for widget in (ui.fitDisplayOptionsLabel, *checkboxes):
            _detach_from_parent_layout(widget)
        ui.fitDisplayOptionsLabel.hide()

        options_contents = QWidget(self)
        options_contents.setObjectName("fitParticlesNumWidget")
        options_contents.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        options_layout = QVBoxLayout(options_contents)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(FORM_ROW_SPACING)
        ui.fitParticlesNumWidget = options_contents

        for checkbox in checkboxes:
            normalize_checkbox(checkbox)

        self._add_option_row(options_layout, ui.fitBGShowCheckBox, ui.fitParticle1ShowCheckBox)
        self._add_option_row(options_layout, ui.fitResShowCheckBox, ui.fitParticle2ShowCheckBox)
        self._add_option_row(options_layout, None, ui.fitParticle3ShowCheckBox)

        options_scroll_area = make_scroll_area(options_contents)
        options_scroll_area.setObjectName("plotDisplayOptionsScrollArea")
        options_scroll_area.setFrameShape(QFrame.NoFrame)
        options_scroll_area.setMinimumHeight(78)
        options_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.section_layout.addWidget(options_scroll_area, 1)

    @staticmethod
    def _add_option_row(layout: QVBoxLayout, left: QWidget | None, right: QWidget | None) -> None:
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(CARD_SPACING)
        if left is not None:
            row_layout.addWidget(left, 1)
        else:
            row_layout.addStretch(1)
        if right is not None:
            row_layout.addWidget(right, 1)
        else:
            row_layout.addStretch(1)
        layout.addWidget(row)


class PlotPreviewCard(CardFrame):
    def __init__(self, ui, content: QWidget, graphics_view: QGraphicsView):
        super().__init__("Fitting Plot", "PlotPreviewCard")
        self._build_plot_layout(ui, content, graphics_view)

        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(760)
        content.setMinimumSize(300, 380)
        content.setMaximumSize(16777215, 16777215)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(content, 1)

    @staticmethod
    def _build_plot_layout(ui, content: QWidget, graphics_view: QGraphicsView) -> None:
        """Build explicit plot subcomponents to avoid dense-control overlap."""
        root_layout = content.layout()
        if root_layout is None:
            root_layout = QGridLayout(content)

        controls = [
            graphics_view,
            content.findChild(QWidget, "fitFittingRegionwidget"),
            content.findChild(QWidget, "fitDataPointsNumWidget"),
            content.findChild(QWidget, "fitFittingShowWidget"),
        ]
        for widget in controls:
            if widget is not None:
                _take_widget(root_layout, widget)

        controls_container = QWidget(content)
        controls_container.setObjectName("plotControlsContainer")
        controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(CARD_SPACING)
        controls_layout.addWidget(FittingRegionControl(ui, controls_container))
        controls_layout.addWidget(PlotSamplingControl(ui, controls_container))
        controls_layout.addWidget(PlotOptionsControl(ui, controls_container))
        controls_container.setMinimumHeight(452)
        controls_container.setMaximumHeight(452)

        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(CARD_SPACING)
        root_layout.addWidget(PlotCanvasArea(graphics_view, content), 0, 0)
        root_layout.addWidget(controls_container, 1, 0)
        root_layout.setRowStretch(0, 1)
        root_layout.setRowStretch(1, 0)
        root_layout.setColumnStretch(0, 1)


class StatusCard(CardFrame):
    def __init__(self, browser: QWidget):
        super().__init__("Run Log", "FittingStatusCard")
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setMinimumHeight(90)
        browser.setMaximumHeight(16777215)
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(browser, 1)


class GisaxsFittingWorkspace:
    """Three-region layout for the cut/fitting page."""

    SETTINGS_KEY = "gisaxs_fitting_splitter_sizes"
    DEFAULT_WORK_SIZES = [760, 420]
    DEFAULT_PREVIEW_SIZES = [300, 860, 160]

    def __init__(self, ui):
        self.ui = ui
        self.page_splitter = QSplitter(Qt.Horizontal, ui.gisaxsFittingPage)
        self.page_splitter.setObjectName("gisaxsFittingWorkspaceSplitter")
        self.page_splitter.setHandleWidth(8)
        self.page_splitter.setChildrenCollapsible(False)
        self.page_splitter.setOpaqueResize(True)

        self.preview_splitter = QSplitter(Qt.Vertical, self.page_splitter)
        self.preview_splitter.setObjectName("gisaxsPreviewSplitter")
        self.preview_splitter.setHandleWidth(8)
        self.preview_splitter.setChildrenCollapsible(False)
        self.preview_splitter.setOpaqueResize(True)
        self.preview_splitter.setMinimumWidth(420)
        self.preview_splitter.setMinimumHeight(
            sum(self.DEFAULT_PREVIEW_SIZES) + 2 * self.preview_splitter.handleWidth()
        )

        self.work_splitter = QSplitter(Qt.Vertical, ui.gisaxsFittingPage)
        self.work_splitter.setObjectName("gisaxsMainWorkSplitter")
        self.work_splitter.setHandleWidth(8)
        self.work_splitter.setChildrenCollapsible(False)
        self.work_splitter.setOpaqueResize(True)
        self.work_splitter.setMinimumWidth(640)
        self.work_splitter.setMinimumHeight(
            sum(self.DEFAULT_WORK_SIZES) + self.work_splitter.handleWidth()
        )
        self.work_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._detach_preview_widgets()
        self._relax_fixed_sizes()
        self._install_page_splitter()
        self._build_left_work_area()
        self._build_preview_area()
        self._configure_button_responsiveness()
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
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(640)
        self.ui.gisaxsFittingPageScrollArea.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.ui.gisaxsFittingPageScrollArea.setWidgetResizable(True)
        self._configure_expanding_inputs()

    def _build_left_work_area(self) -> None:
        self.fixed_controls_stack = QWidget()
        self.fixed_controls_stack.setObjectName("gisaxsFixedControlsStack")
        fixed_layout = QVBoxLayout(self.fixed_controls_stack)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.setSpacing(CARD_SPACING)
        fixed_layout.addWidget(GisaxsInputCard(self.ui.gisaxsInputBox))
        fixed_layout.addWidget(CutLineCard(self.ui))
        fixed_layout.addWidget(FittingControlsCard(self.ui))
        fixed_layout.addStretch(1)
        self.fixed_controls_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.work_splitter.addWidget(self.fixed_controls_stack)
        self.work_splitter.addWidget(ModelParameterCard(self.ui))
        self.work_splitter.setStretchFactor(0, 0)
        self.work_splitter.setStretchFactor(1, 1)
        for index in range(self.work_splitter.count()):
            self.work_splitter.setCollapsible(index, False)

        self.work_area_contents = QWidget()
        self.work_area_contents.setObjectName("gisaxsWorkAreaContents")
        layout = QVBoxLayout(self.work_area_contents)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)
        layout.addWidget(self.work_splitter)

        self.ui.gisaxsFittingPageScrollArea.setWidget(self.work_area_contents)
        self.page_splitter.addWidget(self.ui.gisaxsFittingPageScrollArea)

    def _build_preview_area(self) -> None:
        self.preview_splitter.addWidget(DetectorPreviewCard(self.ui.gisaxsInputGraphicsView))
        self.preview_splitter.addWidget(
            PlotPreviewCard(self.ui, self.ui.curvePlotControlWidget, self.ui.fitGraphicsView)
        )
        self.preview_splitter.addWidget(StatusCard(self.ui.FittingTextBrowser))
        self.preview_splitter.setStretchFactor(0, 2)
        self.preview_splitter.setStretchFactor(1, 3)
        self.preview_splitter.setStretchFactor(2, 0)
        for index in range(self.preview_splitter.count()):
            self.preview_splitter.setCollapsible(index, False)

        self.preview_scroll_area = make_scroll_area(self.preview_splitter, horizontal=True)
        self.preview_scroll_area.setObjectName("gisaxsPreviewScrollArea")
        self.preview_scroll_area.setMinimumWidth(420)

        self.page_splitter.addWidget(self.preview_scroll_area)
        self.page_splitter.setStretchFactor(0, 3)
        self.page_splitter.setStretchFactor(1, 2)
        self.page_splitter.setCollapsible(0, False)
        self.page_splitter.setCollapsible(1, False)

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
        ]
        preferred_actions = [
            "gisaxsInputCutButton",
            "gisaxsInputShowButton",
            "FittingExportButton",
        ]

        for name in expanding_actions:
            button = getattr(self.ui, name, None)
            if button is not None:
                _configure_button(button, minimum_width=108, maximum_width=220, horizontal=QSizePolicy.MinimumExpanding)

        for name in preferred_actions:
            button = getattr(self.ui, name, None)
            if button is not None:
                _configure_button(button, minimum_width=78, maximum_width=140, horizontal=QSizePolicy.Preferred)

        plus_button = getattr(self.ui, "pushButton", None)
        if plus_button is not None:
            normalize_button(plus_button, compact=True)

    def _configure_expanding_inputs(self) -> None:
        for widget in self.ui.gisaxsFittingPage.findChildren(INPUT_WIDGET_TYPES):
            normalize_input(widget)
        for checkbox in self.ui.gisaxsFittingPage.findChildren(QCheckBox):
            normalize_checkbox(checkbox)

    def _install_page_splitter(self) -> None:
        layout = self.ui.verticalLayout_19
        _take_widget(layout, self.ui.gisaxsFittingPageScrollArea)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.page_splitter)

    def restore_sizes(self) -> None:
        sizes = user_settings.get(self.SETTINGS_KEY, None)
        if isinstance(sizes, dict):
            page_sizes = sizes.get("page")
            work_sizes = sizes.get("work")
            preview_sizes = sizes.get("preview")
            if isinstance(page_sizes, (list, tuple)) and len(page_sizes) == 2:
                self.page_splitter.setSizes([max(640, int(page_sizes[0])), max(420, int(page_sizes[1]))])
            else:
                self.page_splitter.setSizes([760, 500])
            if isinstance(work_sizes, (list, tuple)) and len(work_sizes) == 2:
                self.work_splitter.setSizes(
                    [
                        max(self.DEFAULT_WORK_SIZES[0], int(work_sizes[0])),
                        max(self.DEFAULT_WORK_SIZES[1], int(work_sizes[1])),
                    ]
                )
            else:
                self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
            if isinstance(preview_sizes, (list, tuple)) and len(preview_sizes) == 3:
                self.preview_splitter.setSizes(
                    [
                        max(self.DEFAULT_PREVIEW_SIZES[0], int(preview_sizes[0])),
                        max(self.DEFAULT_PREVIEW_SIZES[1], int(preview_sizes[1])),
                        max(self.DEFAULT_PREVIEW_SIZES[2], int(preview_sizes[2])),
                    ]
                )
            else:
                self.preview_splitter.setSizes(self.DEFAULT_PREVIEW_SIZES)
            return

        self.page_splitter.setSizes([760, 500])
        self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
        self.preview_splitter.setSizes(self.DEFAULT_PREVIEW_SIZES)

    def save_state(self) -> None:
        user_settings.set(
            self.SETTINGS_KEY,
            {
                "page": self.page_splitter.sizes(),
                "work": self.work_splitter.sizes(),
                "preview": self.preview_splitter.sizes(),
            },
        )


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
        self.setChildrenCollapsible(False)
        self.setOpaqueResize(True)
        self._enforce_window_minimum_width(central_widget)

        self._remove_from_layout(source_layout, sidebar_area)
        self._remove_from_layout(source_layout, content_widget)

        sidebar_area.setMinimumWidth(180)
        sidebar_area.setMaximumWidth(220)
        sidebar_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        content_widget.setMinimumWidth(1060)
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.addWidget(sidebar_area)
        self.addWidget(content_widget)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 5)
        self.setCollapsible(0, False)
        self.setCollapsible(1, False)

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
            self.setSizes([max(180, int(sizes[0])), max(1060, int(sizes[1]))])
            return

        self.setSizes([190, 1260])

    def save_sizes(self) -> None:
        user_settings.set(self.SETTINGS_KEY, self.sizes())
        user_settings.save_settings()

    @staticmethod
    def _enforce_window_minimum_width(central_widget: QWidget) -> None:
        window = central_widget.window()
        window.setMinimumSize(1300, 760)
        QTimer.singleShot(0, lambda: window.setMinimumSize(1300, 760))


class MainWindowComponents:
    """Builds and owns the maintainable component hierarchy."""

    def __init__(self, ui):
        self.ui = ui
        self._clear_generated_inline_styles(ui.centralwidget)
        self.sidebar = self._create_sidebar()
        self.content = ContentStack(ui.mainWindowWidget)
        self.fitting_workspace = GisaxsFittingWorkspace(ui)
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
        self.fitting_workspace.save_state()
        self.shell.save_sizes()

    @staticmethod
    def _clear_generated_inline_styles(root: QWidget) -> None:
        for widget in _walk_widgets(root):
            if widget.styleSheet():
                widget.setStyleSheet("")


def _walk_widgets(root: QWidget) -> Iterable[QWidget]:
    yield root
    yield from root.findChildren(QWidget)


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


def _configure_button(
    button: QAbstractButton,
    minimum_width: int,
    maximum_width: int,
    horizontal=QSizePolicy.Preferred,
) -> None:
    normalize_button(button, wide=horizontal == QSizePolicy.MinimumExpanding)
    button.setMinimumWidth(minimum_width)
    button.setMaximumWidth(maximum_width)
    button.setSizePolicy(horizontal, QSizePolicy.Fixed)
