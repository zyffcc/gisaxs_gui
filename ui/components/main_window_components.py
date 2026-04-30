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
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
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


class CardFrame(QFrame):
    """Modern card wrapper for existing generated widgets."""

    def __init__(self, title: str, object_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setProperty("card", True)
        self.setMinimumWidth(360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(14, 12, 14, 14)
        self.body_layout.setSpacing(10)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("cardTitle", True)
        self.body_layout.addWidget(self.title_label)

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        widget.setParent(self)
        self.body_layout.addWidget(widget, stretch)


class GisaxsInputCard(CardFrame):
    def __init__(self, content: QGroupBox):
        super().__init__("GISAXS Input", "GisaxsInputCard")
        content.setTitle("")
        self.add_content(content)


class CutLineCard(CardFrame):
    def __init__(self, ui):
        super().__init__("Cut Line and Detector", "CutLineCard")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
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
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
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
        _take_widget(ui.gridLayout_24, ui.widget_7)
        ui.widget_7.setMaximumWidth(16777215)
        ui.widget_7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.body_layout.addWidget(ui.widget_7, 1)


class DetectorPreviewCard(CardFrame):
    def __init__(self, graphics_view: QGraphicsView):
        super().__init__("Detector Preview", "DetectorPreviewCard")
        self.setMinimumWidth(360)
        graphics_view.setMinimumSize(320, 240)
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(graphics_view, 1)


class PlotPreviewCard(CardFrame):
    def __init__(self, content: QWidget, graphics_view: QGraphicsView):
        super().__init__("Fitting Plot", "PlotPreviewCard")
        self._build_plot_layout(content, graphics_view)

        self.setMinimumWidth(360)
        content.setMinimumSize(300, 320)
        content.setMaximumSize(16777215, 16777215)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(content, 1)

    @staticmethod
    def _build_plot_layout(content: QWidget, graphics_view: QGraphicsView) -> None:
        """Separate the plot canvas from the controls below it."""
        root_layout = content.layout()
        if root_layout is None:
            root_layout = QGridLayout(content)

        fitting_region = content.findChild(QWidget, "fitFittingRegionwidget")
        data_points = content.findChild(QWidget, "fitDataPointsNumWidget")
        show_options = content.findChild(QWidget, "fitFittingShowWidget")
        controls = [
            graphics_view,
            fitting_region,
            data_points,
            show_options,
        ]
        for widget in controls:
            if widget is not None:
                _take_widget(root_layout, widget)

        plot_canvas_container = QFrame(content)
        plot_canvas_container.setObjectName("plotCanvasContainer")
        plot_canvas_container.setProperty("previewSection", True)
        plot_canvas_layout = QVBoxLayout(plot_canvas_container)
        plot_canvas_layout.setContentsMargins(0, 0, 0, 0)
        plot_canvas_layout.setSpacing(0)
        graphics_view.setMinimumSize(320, 200)
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_canvas_layout.addWidget(graphics_view, 1)

        if fitting_region is not None:
            fitting_region.setMinimumHeight(72)
            fitting_region.setMaximumHeight(118)
            fitting_region.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        plot_options_container = QFrame(content)
        plot_options_container.setObjectName("plotOptionsContainer")
        plot_options_container.setProperty("previewSection", True)
        plot_options_layout = QHBoxLayout(plot_options_container)
        plot_options_layout.setContentsMargins(0, 0, 0, 0)
        plot_options_layout.setSpacing(10)

        if data_points is not None:
            data_points.setMaximumWidth(150)
            data_points.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            plot_options_layout.addWidget(data_points, 0)

        if show_options is not None:
            show_options.setMinimumWidth(220)
            show_options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            plot_options_layout.addWidget(show_options, 1)

        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)
        root_layout.addWidget(plot_canvas_container, 0, 0, 1, 1)
        if fitting_region is not None:
            root_layout.addWidget(fitting_region, 1, 0, 1, 1)
        root_layout.addWidget(plot_options_container, 2, 0, 1, 1)
        root_layout.setRowStretch(0, 1)
        root_layout.setRowStretch(1, 0)
        root_layout.setRowStretch(2, 0)
        root_layout.setColumnStretch(0, 1)


class StatusCard(CardFrame):
    def __init__(self, browser: QWidget):
        super().__init__("Run Log", "FittingStatusCard")
        browser.setMinimumHeight(90)
        browser.setMaximumHeight(140)
        self.add_content(browser)


class GisaxsFittingWorkspace:
    """Three-region layout for the cut/fitting page."""

    SETTINGS_KEY = "gisaxs_fitting_splitter_sizes"

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
        self.work_area_contents = QWidget()
        self.work_area_contents.setObjectName("gisaxsWorkAreaContents")
        layout = QVBoxLayout(self.work_area_contents)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(GisaxsInputCard(self.ui.gisaxsInputBox))
        layout.addWidget(CutLineCard(self.ui))
        layout.addWidget(FittingControlsCard(self.ui))
        layout.addWidget(ModelParameterCard(self.ui), 1)

        self.ui.gisaxsFittingPageScrollArea.setWidget(self.work_area_contents)
        self.page_splitter.addWidget(self.ui.gisaxsFittingPageScrollArea)

    def _build_preview_area(self) -> None:
        self.preview_splitter.addWidget(DetectorPreviewCard(self.ui.gisaxsInputGraphicsView))
        self.preview_splitter.addWidget(
            PlotPreviewCard(self.ui.curvePlotControlWidget, self.ui.fitGraphicsView)
        )
        self.preview_splitter.addWidget(StatusCard(self.ui.FittingTextBrowser))
        self.preview_splitter.setStretchFactor(0, 2)
        self.preview_splitter.setStretchFactor(1, 3)
        self.preview_splitter.setStretchFactor(2, 0)
        for index in range(self.preview_splitter.count()):
            self.preview_splitter.setCollapsible(index, False)

        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setObjectName("gisaxsPreviewScrollArea")
        self.preview_scroll_area.setMinimumWidth(420)
        self.preview_scroll_area.setWidgetResizable(True)
        self.preview_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_scroll_area.setWidget(self.preview_splitter)

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
                _configure_button(button, minimum_width=108, maximum_width=190, horizontal=QSizePolicy.Preferred)

        for name in preferred_actions:
            button = getattr(self.ui, name, None)
            if button is not None:
                _configure_button(button, minimum_width=78, maximum_width=140, horizontal=QSizePolicy.Preferred)

        plus_button = getattr(self.ui, "pushButton", None)
        if plus_button is not None:
            plus_button.setMinimumSize(30, 30)
            plus_button.setMaximumSize(34, 34)
            plus_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _configure_expanding_inputs(self) -> None:
        input_types = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)
        for widget in self.ui.gisaxsFittingPage.findChildren(input_types):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            widget.setMinimumWidth(max(widget.minimumWidth(), 72))

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
            preview_sizes = sizes.get("preview")
            if isinstance(page_sizes, (list, tuple)) and len(page_sizes) == 2:
                self.page_splitter.setSizes([max(640, int(page_sizes[0])), max(420, int(page_sizes[1]))])
            if isinstance(preview_sizes, (list, tuple)) and len(preview_sizes) == 3:
                self.preview_splitter.setSizes([max(260, int(preview_sizes[0])), max(320, int(preview_sizes[1])), max(90, int(preview_sizes[2]))])
            return

        self.page_splitter.setSizes([760, 500])
        self.preview_splitter.setSizes([300, 380, 110])

    def save_state(self) -> None:
        user_settings.set(
            self.SETTINGS_KEY,
            {
                "page": self.page_splitter.sizes(),
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
        window.setMinimumWidth(1300)
        QTimer.singleShot(0, lambda: window.setMinimumWidth(1300))


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


def _configure_button(
    button: QAbstractButton,
    minimum_width: int,
    maximum_width: int,
    horizontal=QSizePolicy.Preferred,
) -> None:
    button.setMinimumHeight(30)
    button.setMaximumHeight(36)
    button.setMinimumWidth(minimum_width)
    button.setMaximumWidth(maximum_width)
    button.setSizePolicy(horizontal, QSizePolicy.Preferred)
