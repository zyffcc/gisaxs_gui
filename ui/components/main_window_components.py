"""Component layer for the generated main window.

The generated ``Ui_MainWindow`` still creates the individual controls.  These
classes only reorganize those controls into named, testable pieces without
changing object names that controllers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from PyQt5.QtCore import QEvent, QTimer, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
from ui.responsive_layout import apply_density_profile, apply_window_profile, current_profile, scale_value
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
    def __init__(self, content: QWidget, profile=None):
        super().__init__("GIMaP Input", "GisaxsInputCard")
        profile = profile or current_profile(content)
        self.setMinimumHeight(scale_value(260, profile, 210))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        content.setTitle("")
        content.setMinimumHeight(scale_value(200, profile, 165))
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_content(content)


class CutLineCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Cut Line and Detector", "CutLineCard")
        profile = profile or current_profile(ui.centralwidget)
        self.setMinimumHeight(scale_value(230, profile, 185))
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
    def __init__(self, ui, profile=None):
        super().__init__("Fitting Controls", "FittingControlsCard")
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        group_spacing = scale_value(14, self.profile, 10)
        group_margin = scale_value(14, self.profile, 10)
        group_top = scale_value(24, self.profile, 20)
        self.setMinimumHeight(scale_value(760, self.profile, 660))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._managed_group_layouts = []
        self._managed_buttons = []
        self._managed_inputs = []
        self._managed_spinboxes = []
        self._managed_labels = []
        containers = [
            ui.fitCurrentDataCheckBox,
            ui.widget,
            ui.fitImport1dFileButton,
            ui.fitImport1dFileValue,
            ui.fitMethodWidget,
            ui.fitMethodWidget_2,
            ui.widget_8,
        ]
        for widget in containers:
            _detach_from_parent_layout(widget)
            widget.setMaximumWidth(16777215)
            set_expanding_x(widget)

        controls = [
            ui.fitCurrentDataCheckBox,
            ui.fitLogXCheckBox,
            ui.fitLogYCheckBox,
            ui.fitNormCheckBox,
            ui.fitImport1dFileButton,
            ui.fitImport1dFileValue,
            ui.fitMethodLabel,
            ui.fitMethodValue,
            ui.FittingAutoFittingButton,
            ui.fitKLabel,
            ui.fitKValue,
            ui.FittingAutoKButton,
            ui.fitIntResLabel,
            ui.fitIntResValue,
            ui.fitSigmaResLabel,
            ui.fitSigmaResValue,
            ui.fitNuResLabel,
            ui.fitNuResValue,
            ui.FittingClearFittingButton_2,
            ui.FittingManualFittingButton,
            ui.FittingExportButton,
        ]
        for widget in controls:
            _detach_from_parent_layout(widget)
            widget.setParent(self)
            widget.setMaximumWidth(16777215)

        ui.fitIntResLabel.setText("Intensity (Res.)")
        ui.fitMethodLabel.setText("Method:")
        ui.fitKLabel.setText("k:")
        ui.FittingAutoKButton.setText("Auto-K: OFF")
        ui.fitMethodValue.setToolTip("Method selection is not implemented yet.")
        self._method_notice_combo = ui.fitMethodValue
        self._method_notice_queued = False
        ui.fitMethodValue.installEventFilter(self)
        ui.fitMethodValue.currentIndexChanged.connect(lambda _index: self._queue_method_notice())

        self.fitExportPlotButton = QPushButton("Export Plot", self)
        self.fitExportPlotButton.setObjectName("fitExportPlotButton")
        self.fitExportPlotButton.clicked.connect(ui.FittingExportButton.click)

        for button in (
            ui.FittingClearFittingButton_2,
            ui.FittingManualFittingButton,
            ui.FittingExportButton,
            ui.FittingAutoFittingButton,
            ui.FittingAutoKButton,
            self.fitExportPlotButton,
        ):
            self._managed_buttons.append(button)

        for input_widget in (
            ui.fitImport1dFileValue,
            ui.fitMethodValue,
            ui.fitKValue,
            ui.fitIntResValue,
            ui.fitSigmaResValue,
            ui.fitNuResValue,
        ):
            self._managed_inputs.append(input_widget)

        self._managed_spinboxes = [ui.fitKValue, ui.fitIntResValue, ui.fitSigmaResValue, ui.fitNuResValue]

        data_options_group = self._make_group("Display Options")
        data_layout = QHBoxLayout(data_options_group)
        self._configure_group_layout(data_layout, group_margin, group_top, group_spacing)
        for checkbox in (
            ui.fitCurrentDataCheckBox,
            ui.fitLogXCheckBox,
            ui.fitLogYCheckBox,
            ui.fitNormCheckBox,
        ):
            data_layout.addWidget(checkbox)
        data_layout.addStretch(1)
        data_layout.addWidget(self.fitExportPlotButton)

        external_group = self._make_group("External 1D Data")
        external_layout = QHBoxLayout(external_group)
        self._configure_group_layout(external_layout, group_margin, group_top, group_spacing)
        external_layout.addWidget(ui.fitImport1dFileButton, 0)
        external_layout.addWidget(ui.fitImport1dFileValue, 1)

        method_group = self._make_group("Auto Fitting / Method")
        method_layout = QGridLayout(method_group)
        self._configure_group_layout(method_layout, group_margin, group_top, group_spacing)
        method_layout.addWidget(ui.fitMethodLabel, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        method_layout.addWidget(ui.fitMethodValue, 0, 1)
        method_layout.addWidget(ui.FittingAutoFittingButton, 0, 2)
        self.methodInfoLabel = QLabel("Method selection is not implemented yet.", method_group)
        self.methodInfoLabel.setObjectName("fitMethodInfoLabel")
        self.methodInfoLabel.setStyleSheet("color: #2563eb;")
        method_layout.addWidget(self.methodInfoLabel, 1, 1, 1, 2)
        method_layout.setColumnStretch(1, 1)

        k_group = self._make_group("Scaling Factor k")
        k_layout = QGridLayout(k_group)
        self._configure_group_layout(k_layout, group_margin, group_top, group_spacing)
        self._managed_labels.append(ui.fitKLabel)
        k_layout.addWidget(ui.fitKLabel, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        k_layout.addWidget(ui.fitKValue, 0, 1)
        k_layout.addWidget(ui.FittingAutoKButton, 1, 1)
        self.kInfoLabel = QLabel(
            "Scaling factor <b>k</b> amplifies the calculated fitting intensity.<br>"
            "Simple model: <i>I</i><sub>fit</sub>(q) = "
            "k &middot; <i>I</i><sub>model</sub>(q).",
            k_group,
        )
        self.kInfoLabel.setObjectName("fitKInfoLabel")
        self.kInfoLabel.setTextFormat(Qt.RichText)
        self.kInfoLabel.setWordWrap(True)
        self._style_info_label(self.kInfoLabel)
        self.kInfoLabel.setToolTip(
            "<b>Scaling factor k</b><br>"
            "<i>I</i><sub>fit</sub>(q) = k &middot; <i>I</i><sub>model</sub>(q)<br>"
            "Auto-K minimizes:<br>"
            "&Sigma;<sub>i</sub>[k &middot; <i>I</i><sub>base</sub>(q<sub>i</sub>) "
            "- <i>I</i><sub>exp</sub>(q<sub>i</sub>)]<sup>2</sup><br>"
            "Estimate:<br>"
            "k = (&lt;<i>I</i><sub>base</sub>, <i>I</i><sub>exp</sub>&gt;) / "
            "(&lt;<i>I</i><sub>base</sub>, <i>I</i><sub>base</sub>&gt;)"
        )
        ui.fitKLabel.setToolTip(self.kInfoLabel.toolTip())
        ui.fitKValue.setToolTip(self.kInfoLabel.toolTip())
        ui.FittingAutoKButton.setToolTip(self.kInfoLabel.toolTip())
        k_layout.addWidget(self.kInfoLabel, 2, 0, 1, 2)
        k_layout.setColumnStretch(1, 1)
        k_group.setMinimumHeight(scale_value(220, self.profile, 190))

        resolution_group = self._make_group("Resolution Function")
        resolution_layout = QGridLayout(resolution_group)
        self._configure_group_layout(resolution_layout, group_margin, group_top, group_spacing)
        for row, (label, value) in enumerate(
            (
                (ui.fitIntResLabel, ui.fitIntResValue),
                (ui.fitSigmaResLabel, ui.fitSigmaResValue),
                (ui.fitNuResLabel, ui.fitNuResValue),
            )
        ):
            self._managed_labels.append(label)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            resolution_layout.addWidget(label, row, 0)
            resolution_layout.addWidget(value, row, 1)
        self.resolutionInfoLabel = QLabel(
            "Adjust the instrumental resolution used during fitting.<br>"
            "These parameters control peak broadening and smoothing effects.",
            resolution_group,
        )
        self.resolutionInfoLabel.setObjectName("fitResolutionInfoLabel")
        self.resolutionInfoLabel.setTextFormat(Qt.RichText)
        self.resolutionInfoLabel.setWordWrap(True)
        self._style_info_label(self.resolutionInfoLabel)
        self.resolutionInfoLabel.setToolTip(
            "<b>Resolution component</b><br>"
            "R(q) = <i>I</i><sub>res</sub> / "
            "[1 + (q / &sigma;<sub>res</sub>)<sup>2</sup>]<sup>&nu;</sup><br>"
            "The fitting model receives "
            "&sigma;<sub>res</sub>, &nu;<sub>res</sub>, "
            "<i>I</i><sub>res</sub>, and k as global parameters."
        )
        for widget in (
            ui.fitIntResLabel,
            ui.fitIntResValue,
            ui.fitSigmaResLabel,
            ui.fitSigmaResValue,
            ui.fitNuResLabel,
            ui.fitNuResValue,
        ):
            widget.setToolTip(self.resolutionInfoLabel.toolTip())
        resolution_layout.addWidget(self.resolutionInfoLabel, 3, 0, 1, 2)
        resolution_layout.setColumnStretch(1, 1)
        resolution_group.setMinimumHeight(scale_value(220, self.profile, 195))

        actions_group = self._make_group("Fitting Actions")
        actions_layout = QHBoxLayout(actions_group)
        self._configure_group_layout(actions_layout, group_margin, group_top, group_spacing)
        actions_layout.addWidget(ui.FittingClearFittingButton_2)
        actions_layout.addWidget(ui.FittingManualFittingButton)
        actions_layout.addWidget(ui.FittingExportButton)

        parameter_row = QWidget(self)
        parameter_row.setObjectName("fitParameterCardsRow")
        parameter_layout = QHBoxLayout(parameter_row)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setSpacing(group_spacing)
        parameter_layout.addWidget(k_group, 1)
        parameter_layout.addWidget(resolution_group, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(group_spacing)
        self._main_controls_layout = layout
        self.body_layout.addLayout(layout)
        layout.addWidget(data_options_group)
        layout.addWidget(external_group)
        layout.addWidget(method_group)
        layout.addWidget(parameter_row)
        layout.addWidget(actions_group)
        self.apply_responsive_profile(self.profile)

    def _make_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title, self)
        group.setObjectName(title.replace(" ", "").replace("/", "") + "Group")
        group.setStyleSheet(
            "QGroupBox {"
            "border: 1px solid #d7dee8;"
            "border-radius: 7px;"
            "margin-top: 10px;"
            "padding-top: 12px;"
            "background: #ffffff;"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "left: 8px;"
            "padding: 0 4px;"
            "}"
        )
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return group

    def _style_info_label(self, label: QLabel) -> None:
        label.setStyleSheet(
            "QLabel {"
            "background: #eff6ff;"
            "border: 1px solid #bfdbfe;"
            "border-radius: 6px;"
            "color: #1d4ed8;"
            "padding: 6px 8px;"
            "line-height: 135%;"
            "}"
        )
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _configure_group_layout(self, layout, margin: int, top: int, spacing: int) -> None:
        layout.setContentsMargins(margin, top, margin, margin)
        if hasattr(layout, "setHorizontalSpacing"):
            layout.setHorizontalSpacing(spacing)
            layout.setVerticalSpacing(max(FORM_ROW_SPACING, spacing - 4))
        else:
            layout.setSpacing(spacing)
        self._managed_group_layouts.append(layout)

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        group_spacing = scale_value(14, profile, 10)
        group_margin = scale_value(14, profile, 10)
        group_top = scale_value(24, profile, 20)
        self.setMinimumHeight(scale_value(760, profile, 660))
        self.setMaximumHeight(16777215)

        if hasattr(self, "_main_controls_layout"):
            self._main_controls_layout.setSpacing(group_spacing)
        for layout in self._managed_group_layouts:
            layout.setContentsMargins(group_margin, group_top, group_margin, group_margin)
            if hasattr(layout, "setHorizontalSpacing"):
                layout.setHorizontalSpacing(group_spacing)
                layout.setVerticalSpacing(max(FORM_ROW_SPACING, group_spacing - 4))
            else:
                layout.setSpacing(group_spacing)

        button_width = scale_value(128, profile, 110)
        input_height = BUTTON_HEIGHT + scale_value(4, profile, 4)
        spinbox_width = scale_value(160, profile, 132)
        label_width = scale_value(132, profile, 108)

        for button in self._managed_buttons:
            button.setMinimumHeight(input_height)
            button.setMinimumWidth(button_width)
            button.setMaximumHeight(16777215)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for input_widget in self._managed_inputs:
            input_widget.setMinimumHeight(input_height)
            input_widget.setMaximumHeight(16777215)
            input_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for spinbox in self._managed_spinboxes:
            spinbox.setMinimumWidth(spinbox_width)
        for label in self._managed_labels:
            label.setMinimumWidth(label_width)

        k_group = self.findChild(QGroupBox, "ScalingFactorkGroup")
        resolution_group = self.findChild(QGroupBox, "ResolutionFunctionGroup")
        if k_group is not None:
            k_group.setMinimumHeight(scale_value(220, profile, 190))
        if resolution_group is not None:
            resolution_group.setMinimumHeight(scale_value(240, profile, 210))
        self.updateGeometry()

    def _show_method_not_implemented(self) -> None:
        self._method_notice_queued = False
        QMessageBox.information(
            self,
            "Method",
            "Method selection is not implemented yet.",
        )

    def _queue_method_notice(self) -> None:
        if self._method_notice_queued:
            return
        self._method_notice_queued = True
        QTimer.singleShot(0, self._show_method_not_implemented)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "_method_notice_combo", None) and event.type() == QEvent.MouseButtonPress:
            self._queue_method_notice()
        return super().eventFilter(obj, event)


class ModelParameterCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Model Parameters", "ModelParameterCard")
        profile = profile or current_profile(ui.centralwidget)
        self.setMinimumHeight(scale_value(260, profile, 210))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        _take_widget(ui.gridLayout_24, ui.widget_7)
        ui.widget_7.setMaximumWidth(16777215)
        ui.widget_7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.model_scroll_area = make_scroll_area(ui.widget_7, horizontal=True)
        self.model_scroll_area.setObjectName("modelParametersScrollArea")
        self.body_layout.addWidget(self.model_scroll_area, 1)


class DetectorPreviewCard(CardFrame):
    def __init__(self, graphics_view: QGraphicsView, profile=None):
        super().__init__("Detector Preview", "DetectorPreviewCard")
        profile = profile or current_profile(graphics_view)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(scale_value(260, profile, 210))
        graphics_view.setMinimumSize(scale_value(320, profile, 260), scale_value(240, profile, 190))
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(graphics_view, 1)


class PlotCanvasArea(QFrame):
    def __init__(self, graphics_view: QGraphicsView, parent: QWidget | None = None, profile=None):
        super().__init__(parent)
        profile = profile or current_profile(parent or graphics_view)
        self.setObjectName("plotCanvasContainer")
        self.setProperty("previewSection", True)
        self.setMinimumHeight(scale_value(260, profile, 200))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        graphics_view.setMinimumSize(scale_value(320, profile, 260), scale_value(260, profile, 200))
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
        profile=None,
    ):
        super().__init__(parent)
        profile = profile or current_profile(parent or self)
        self.setObjectName(object_name)
        self.setProperty("sectionCard", True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed if fixed_height else QSizePolicy.Preferred)
        if fixed_height is not None:
            height = scale_value(fixed_height, profile, int(fixed_height * 0.78))
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)

        self.section_layout = QVBoxLayout(self)
        self.section_layout.setContentsMargins(12, 10, 12, 12)
        self.section_layout.setSpacing(CARD_SPACING)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("sectionTitle", True)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.section_layout.addWidget(self.title_label)


class FittingRegionControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        profile = profile or current_profile(parent or ui.centralwidget)
        super().__init__("Fitting Region", "FittingRegionControl", parent, fixed_height=148, profile=profile)

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
        ui.fitFittingRegionSlider.setMinimumHeight(scale_value(28, profile, 24))
        ui.fitFittingRegionSlider.setMaximumHeight(scale_value(36, profile, 30))
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
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        profile = profile or current_profile(parent or ui.centralwidget)
        super().__init__("Sampling", "PlotSamplingControl", parent, fixed_height=124, profile=profile)

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
                base_min_height = getattr(parent, "_base_min_height", 760)
                parent.setMinimumHeight(max(base_min_height, parent.sizeHint().height()))
                parent.updateGeometry()
                break
            parent = parent.parentWidget()


class PlotOptionsControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        super().__init__("Display Options", "PlotOptionsControl", parent, profile=profile)
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
    def __init__(self, ui, content: QWidget, graphics_view: QGraphicsView, profile=None):
        super().__init__("Fitting Plot", "PlotPreviewCard")
        profile = profile or current_profile(content)
        self._base_min_height = scale_value(760, profile, 620)
        self._build_plot_layout(ui, content, graphics_view, profile)

        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(self._base_min_height)
        content.setMinimumSize(scale_value(300, profile, 260), scale_value(380, profile, 300))
        content.setMaximumSize(16777215, 16777215)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(content, 1)

    @staticmethod
    def _build_plot_layout(ui, content: QWidget, graphics_view: QGraphicsView, profile) -> None:
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
        controls_layout.addWidget(FittingRegionControl(ui, controls_container, profile))
        controls_layout.addWidget(PlotSamplingControl(ui, controls_container, profile))
        controls_layout.addWidget(PlotOptionsControl(ui, controls_container, profile))

        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(CARD_SPACING)
        root_layout.addWidget(PlotCanvasArea(graphics_view, content, profile), 0, 0)
        root_layout.addWidget(controls_container, 1, 0)
        root_layout.setRowStretch(0, 1)
        root_layout.setRowStretch(1, 0)
        root_layout.setColumnStretch(0, 1)


class StatusCard(CardFrame):
    def __init__(self, browser: QWidget, profile=None):
        super().__init__("Run Log", "FittingStatusCard")
        profile = profile or current_profile(browser)
        self.setMinimumHeight(scale_value(120, profile, 96))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setMinimumHeight(scale_value(90, profile, 72))
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
        gisaxs_card = GisaxsInputCard(self.ui.gisaxsInputBox, self.profile)
        cut_line_card = CutLineCard(self.ui, self.profile)
        fitting_controls_card = FittingControlsCard(self.ui, self.profile)
        model_parameters_card = ModelParameterCard(self.ui, self.profile)
        fixed_layout.addWidget(gisaxs_card)
        fixed_layout.addWidget(cut_line_card)
        fixed_layout.addWidget(fitting_controls_card)
        fixed_layout.addStretch(1)
        fixed_stack_min_height = self._fixed_stack_min_height()
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
        self.preview_splitter.addWidget(DetectorPreviewCard(self.ui.gisaxsInputGraphicsView, self.profile))
        self.preview_splitter.addWidget(
            PlotPreviewCard(self.ui, self.ui.curvePlotControlWidget, self.ui.fitGraphicsView, self.profile)
        )
        self.preview_splitter.addWidget(StatusCard(self.ui.FittingTextBrowser, self.profile))
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
        self._configure_button_responsiveness()
        fitting_card = self.fixed_controls_stack.findChild(FittingControlsCard, "FittingControlsCard")
        if fitting_card is not None:
            fitting_card.apply_responsive_profile(profile)
        self.preview_splitter.setMinimumWidth(profile.preview_min)
        self.preview_scroll_area.setMinimumWidth(profile.preview_min)
        self.work_splitter.setMinimumWidth(profile.workspace_min)
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(profile.workspace_min)

        fixed_min = self._fixed_stack_min_height()
        self.fixed_controls_stack.setMinimumHeight(fixed_min)
        self.work_splitter.setMinimumHeight(
            fixed_min + self.DEFAULT_WORK_SIZES[1] + self.work_splitter.handleWidth()
        )
        self.preview_splitter.setMinimumHeight(
            sum(self.DEFAULT_PREVIEW_SIZES) + 2 * self.preview_splitter.handleWidth()
        )
        self.page_splitter.setSizes(list(profile.page_sizes))
        self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
        self.preview_splitter.setSizes(self.DEFAULT_PREVIEW_SIZES)

    def _fixed_stack_min_height(self) -> int:
        card_names = ("GisaxsInputCard", "CutLineCard", "FittingControlsCard")
        card_heights = [
            widget.minimumHeight()
            for name in card_names
            if (widget := self.fixed_controls_stack.findChild(QWidget, name)) is not None
        ]
        if not card_heights:
            return self.fixed_controls_stack.minimumHeight()
        return sum(card_heights) + (len(card_heights) - 1) * CARD_SPACING


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

    def __init__(self, parent: QWidget | None = None, profile=None):
        super().__init__(parent)
        profile = profile or current_profile(parent)
        self.setObjectName("predictModelLibraryCard")
        self.setProperty("card", True)
        self.setMinimumHeight(scale_value(118, profile, 96))
        self.setMaximumHeight(scale_value(136, profile, 112))
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
        apply_density_profile(ui.centralwidget, self.responsive_profile)
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

        card = PredictModelLibraryCard(self.ui.gisaxsPredictPage, self.responsive_profile)
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
        apply_density_profile(self.ui.centralwidget, profile)
        apply_window_profile(self.ui.centralwidget.window(), profile)
        self.fitting_workspace.apply_responsive_profile(profile)
        self.shell.apply_responsive_profile(profile)
        apply_density_profile(self.ui.centralwidget, profile)

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
