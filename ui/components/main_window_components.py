"""Component layer for the generated main window.

The generated ``Ui_MainWindow`` still creates the individual controls.  These
classes only reorganize those controls into named, testable pieces without
changing object names that controllers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
from ui.responsive_layout import apply_window_profile, current_profile
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
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        content.setTitle("")
        content.setMinimumHeight(200)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_content(content)


class CutLineCard(CardFrame):
    def __init__(self, ui):
        super().__init__("Cut Line and Detector", "CutLineCard")
        self.setMinimumHeight(230)
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
        self.setMinimumHeight(330)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        ui.fitMethodWidget.setMinimumHeight(120)
        ui.fitMethodWidget.setMaximumHeight(120)
        ui.fitMethodWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ui.fitMethodWidget_2.setMinimumHeight(120)
        ui.fitMethodWidget_2.setMaximumHeight(120)
        ui.fitMethodWidget_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ui.widget_8.setMinimumHeight(48)
        ui.widget_8.setMaximumHeight(48)
        ui.widget_8.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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
        grid.setRowMinimumHeight(2, 120)
        grid.setRowMinimumHeight(3, 48)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 0)
        grid.setRowStretch(2, 0)
        grid.setRowStretch(3, 0)
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


class ParticleOptionsLayout(QVBoxLayout):
    """Dynamic particle checkbox column used by the fitting controller."""

    def addWidget(self, widget: QWidget, stretch: int = 0, alignment: Qt.Alignment = Qt.Alignment()) -> None:
        super().addWidget(widget, stretch, alignment)
        self._refresh_after_change(widget)

    def insertWidget(
        self,
        index: int,
        widget: QWidget,
        stretch: int = 0,
        alignment: Qt.Alignment = Qt.Alignment(),
    ) -> None:
        super().insertWidget(index, widget, stretch, alignment)
        self._refresh_after_change(widget)

    def _refresh_after_change(self, widget: QWidget | None = None) -> None:
        if isinstance(widget, QCheckBox):
            normalize_checkbox(widget)
            if not widget.property("plotOptionGeometryHooked"):
                widget.destroyed.connect(lambda _=None: QTimer.singleShot(0, self._refresh_after_change))
                widget.setProperty("plotOptionGeometryHooked", True)

        parent = self.parentWidget()
        while parent is not None:
            parent.updateGeometry()
            parent.adjustSize()
            if parent.objectName() == "PlotPreviewCard":
                parent.setMinimumHeight(max(760, parent.sizeHint().height()))
                parent.updateGeometry()
                break
            parent = parent.parentWidget()


class PlotOptionsControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None):
        super().__init__("Display Options", "PlotOptionsControl", parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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

        for checkbox in checkboxes:
            normalize_checkbox(checkbox)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.section_layout.addLayout(grid)

        static_column = QWidget(self)
        static_column.setObjectName("plotStaticOptionsColumn")
        static_layout = QVBoxLayout(static_column)
        static_layout.setContentsMargins(0, 0, 0, 0)
        static_layout.setSpacing(FORM_ROW_SPACING)
        static_layout.addWidget(ui.fitBGShowCheckBox)
        static_layout.addWidget(ui.fitResShowCheckBox)
        static_layout.addStretch(1)

        particle_column = QWidget(self)
        particle_column.setObjectName("fitParticlesNumWidget")
        particle_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        particle_layout = ParticleOptionsLayout(particle_column)
        particle_layout.setContentsMargins(0, 0, 0, 0)
        particle_layout.setSpacing(FORM_ROW_SPACING)
        ui.fitParticlesNumWidget = particle_column

        particle_layout.addWidget(ui.fitParticle1ShowCheckBox)
        particle_layout.addWidget(ui.fitParticle2ShowCheckBox)
        particle_layout.addWidget(ui.fitParticle3ShowCheckBox)

        grid.addWidget(static_column, 0, 0)
        grid.addWidget(particle_column, 0, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)


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
    DEFAULT_WORK_SIZES = [760, 680]
    DEFAULT_PREVIEW_SIZES = [300, 860, 160]

    def __init__(self, ui, profile=None):
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        self.DEFAULT_WORK_SIZES = list(self.profile.work_sizes)
        self.DEFAULT_PREVIEW_SIZES = list(self.profile.preview_sizes)
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
        self.preview_splitter.setMinimumWidth(self.profile.preview_min)
        self.preview_splitter.setMinimumHeight(
            sum(self.DEFAULT_PREVIEW_SIZES) + 2 * self.preview_splitter.handleWidth()
        )

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
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(self.profile.workspace_min)
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
        gisaxs_card = GisaxsInputCard(self.ui.gisaxsInputBox)
        cut_line_card = CutLineCard(self.ui)
        fitting_controls_card = FittingControlsCard(self.ui)
        model_parameters_card = ModelParameterCard(self.ui)
        fixed_layout.addWidget(gisaxs_card)
        fixed_layout.addWidget(cut_line_card)
        fixed_layout.addWidget(fitting_controls_card)
        fixed_layout.addStretch(1)
        fixed_stack_min_height = (
            gisaxs_card.minimumHeight()
            + cut_line_card.minimumHeight()
            + fitting_controls_card.minimumHeight()
            + 2 * CARD_SPACING
        )
        self.fixed_controls_stack.setMinimumHeight(fixed_stack_min_height)
        self.work_splitter.setMinimumHeight(
            fixed_stack_min_height
            + self.DEFAULT_WORK_SIZES[1]
            + self.work_splitter.handleWidth()
        )
        self.fixed_controls_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.work_splitter.addWidget(self.fixed_controls_stack)
        self.work_splitter.addWidget(model_parameters_card)
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
        self.preview_scroll_area.setMinimumWidth(self.profile.preview_min)

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
            if sizes.get("profile") != self.profile.key:
                self.page_splitter.setSizes(list(self.profile.page_sizes))
                self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
                self.preview_splitter.setSizes(self.DEFAULT_PREVIEW_SIZES)
                return
            page_sizes = sizes.get("page")
            work_sizes = sizes.get("work")
            preview_sizes = sizes.get("preview")
            if isinstance(page_sizes, (list, tuple)) and len(page_sizes) == 2:
                self.page_splitter.setSizes(
                    [
                        max(self.profile.workspace_min, int(page_sizes[0])),
                        max(self.profile.preview_min, int(page_sizes[1])),
                    ]
                )
            else:
                self.page_splitter.setSizes(list(self.profile.page_sizes))
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

        self.page_splitter.setSizes(list(self.profile.page_sizes))
        self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
        self.preview_splitter.setSizes(self.DEFAULT_PREVIEW_SIZES)

    def save_state(self) -> None:
        user_settings.set(
            self.SETTINGS_KEY,
            {
                "page": self.page_splitter.sizes(),
                "work": self.work_splitter.sizes(),
                "preview": self.preview_splitter.sizes(),
                "profile": self.profile.key,
            },
        )

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        self.DEFAULT_WORK_SIZES = list(profile.work_sizes)
        self.DEFAULT_PREVIEW_SIZES = list(profile.preview_sizes)
        self.preview_splitter.setMinimumWidth(profile.preview_min)
        self.preview_scroll_area.setMinimumWidth(profile.preview_min)
        self.work_splitter.setMinimumWidth(profile.workspace_min)
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(profile.workspace_min)

        fixed_min = self.fixed_controls_stack.minimumHeight()
        self.work_splitter.setMinimumHeight(
            fixed_min + self.DEFAULT_WORK_SIZES[1] + self.work_splitter.handleWidth()
        )
        self.preview_splitter.setMinimumHeight(
            sum(self.DEFAULT_PREVIEW_SIZES) + 2 * self.preview_splitter.handleWidth()
        )
        self.page_splitter.setSizes(list(profile.page_sizes))
        self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
        self.preview_splitter.setSizes(self.DEFAULT_PREVIEW_SIZES)


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
    ):
        super().__init__(Qt.Horizontal, parent or central_widget)
        self.profile = profile or current_profile(central_widget)
        self.setObjectName("mainShell")
        self.setHandleWidth(6)
        self.setChildrenCollapsible(False)
        self.setOpaqueResize(True)
        self._enforce_window_minimum_width(central_widget)

        self._remove_from_layout(source_layout, sidebar_area)
        self._remove_from_layout(source_layout, content_widget)

        sidebar_area.setMinimumWidth(self.profile.sidebar_min)
        sidebar_area.setMaximumWidth(self.profile.sidebar_max)
        sidebar_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        content_widget.setMinimumWidth(self.profile.content_min)
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
            self.setSizes(
                [
                    max(self.profile.sidebar_min, int(sizes[0])),
                    max(self.profile.content_min, int(sizes[1])),
                ]
            )
            return

        self.setSizes([self.profile.sidebar_default, self.profile.content_min])

    def save_sizes(self) -> None:
        user_settings.set(self.SETTINGS_KEY, self.sizes())
        user_settings.save_settings()

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        sidebar = self.widget(0)
        content = self.widget(1)
        sidebar.setMinimumWidth(profile.sidebar_min)
        sidebar.setMaximumWidth(profile.sidebar_max)
        content.setMinimumWidth(profile.content_min)
        self.setSizes([profile.sidebar_default, profile.content_min])

    @staticmethod
    def _enforce_window_minimum_width(central_widget: QWidget) -> None:
        window = central_widget.window()
        profile = current_profile(window)
        apply_window_profile(window, profile)
        QTimer.singleShot(0, lambda: apply_window_profile(window, profile))


class PredictModelLibraryCard(QFrame):
    """Small browser entry point for remotely hosted prediction models."""

    MODEL_LIBRARY_URL = "https://syncandshare.desy.de/index.php/s/ZMF7r57KgefPS2W"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("predictModelLibraryCard")
        self.setProperty("card", True)
        self.setMinimumHeight(118)
        self.setMaximumHeight(136)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(CARD_MARGIN, 12, CARD_MARGIN, CARD_MARGIN)
        layout.setSpacing(CARD_SPACING)

        text_column = QWidget(self)
        text_layout = QVBoxLayout(text_column)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)

        title = QLabel("Model Library", text_column)
        title.setObjectName("predictModelLibraryTitle")
        title.setProperty("cardTitle", True)
        text_layout.addWidget(title)

        description = QLabel(
            "Browse the shared DESY model repository, download a model, then use Model Import.",
            text_column,
        )
        description.setObjectName("predictModelLibraryDescription")
        description.setProperty("cardBody", True)
        description.setWordWrap(True)
        text_layout.addWidget(description)

        url_label = QLabel(self.MODEL_LIBRARY_URL, text_column)
        url_label.setObjectName("predictModelLibraryUrl")
        url_label.setProperty("cardMeta", True)
        url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_layout.addWidget(url_label)

        self.open_button = QPushButton("Browse Models", self)
        self.open_button.setObjectName("gisaxsPredictBrowseModelLibraryButton")
        self.open_button.setToolTip(self.MODEL_LIBRARY_URL)
        normalize_button(self.open_button, wide=True)
        self.open_button.clicked.connect(self.open_model_library)

        layout.addWidget(text_column, 1)
        layout.addWidget(self.open_button, 0, Qt.AlignVCenter)

    def open_model_library(self) -> None:
        QDesktopServices.openUrl(QUrl(self.MODEL_LIBRARY_URL))


class MainWindowComponents:
    """Builds and owns the maintainable component hierarchy."""

    def __init__(self, ui):
        self.ui = ui
        self.responsive_profile = current_profile(ui.centralwidget)
        self._clear_generated_inline_styles(ui.centralwidget)
        self.sidebar = self._create_sidebar()
        self.content = ContentStack(ui.mainWindowWidget)
        self.fitting_workspace = GisaxsFittingWorkspace(ui, self.responsive_profile)
        self.predict_model_library = self._install_predict_model_library_card()
        self.shell = MainShell(
            ui.centralwidget,
            ui.horizontalLayout,
            ui.sideBarScrollArea,
            ui.mainContentWidget,
            profile=self.responsive_profile,
        )
        apply_main_window_styles(ui)
        apply_window_profile(ui.centralwidget.window(), self.responsive_profile)

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

    def _install_predict_model_library_card(self) -> PredictModelLibraryCard | None:
        layout = getattr(self.ui, "verticalLayout_16", None)
        anchor = getattr(self.ui, "widget_2", None)
        if layout is None or anchor is None:
            return None
        if self.ui.gisaxsPredictPage.findChild(QWidget, "predictModelLibraryCard") is not None:
            return None

        card = PredictModelLibraryCard(self.ui.gisaxsPredictPage)
        insert_index = layout.indexOf(anchor)
        if insert_index < 0:
            layout.insertWidget(0, card)
        else:
            layout.insertWidget(insert_index + 1, card)
        return card

    def save_state(self) -> None:
        self.fitting_workspace.save_state()
        self.shell.save_sizes()

    def apply_responsive_profile(self, profile) -> None:
        self.responsive_profile = profile
        apply_window_profile(self.ui.centralwidget.window(), profile)
        self.fitting_workspace.apply_responsive_profile(profile)
        self.shell.apply_responsive_profile(profile)

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
