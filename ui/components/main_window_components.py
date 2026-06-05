"""Component layer for the generated main window.

The generated ``Ui_MainWindow`` still creates the individual controls.  These
classes only reorganize those controls into named, testable pieces without
changing object names that controllers depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from PyQt5.QtCore import QEvent, QSettings, QTimer, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractButton,
    QBoxLayout,
    QCheckBox,
    QDoubleSpinBox,
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
    QToolButton,
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
from ui.responsive_layout import (
    apply_density_profile,
    apply_window_profile,
    current_profile,
    install_adaptive_window_profile,
    scale_value,
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


class CollapsibleCardFrame(QFrame):
    """Card wrapper with a persistent collapse/expand header."""

    SETTINGS_PREFIX = "cut_fitting/right_cards"

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        *,
        default_expanded: bool = True,
    ):
        super().__init__(parent)
        self._title = title
        self._settings_key = f"{self.SETTINGS_PREFIX}/{object_name}/expanded"
        self.setObjectName(object_name)
        self.setProperty("card", True)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(CARD_MARGIN, 8, CARD_MARGIN, CARD_MARGIN)
        self.body_layout.setSpacing(CARD_SPACING)

        self.header_widget = QWidget(self)
        self.header_widget.setObjectName(f"{object_name}Header")
        self.header_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QLabel(title, self.header_widget)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("cardTitle", True)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.header_button = QToolButton(self.header_widget)
        self.header_button.setObjectName(f"{object_name}ToggleButton")
        self.header_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.header_button.setCheckable(True)
        self.header_button.setAutoRaise(True)
        self.header_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.header_button.toggled.connect(self.set_expanded)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.header_button, 0, Qt.AlignRight)
        self.body_layout.addWidget(self.header_widget)

        self.content_widget = QWidget(self)
        self.content_widget.setObjectName(f"{object_name}Content")
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(CARD_SPACING)
        self.body_layout.addWidget(self.content_widget, 1)

        settings = QSettings()
        expanded = settings.value(self._settings_key, default_expanded, type=bool)
        self.header_button.blockSignals(True)
        self.header_button.setChecked(bool(expanded))
        self.header_button.blockSignals(False)
        self.set_expanded(bool(expanded))

    def add_content(self, widget: QWidget, stretch: int = 0) -> None:
        widget.setParent(self.content_widget)
        self.content_layout.addWidget(widget, stretch)

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self.header_button.setChecked(expanded)
        self.header_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content_widget.setVisible(expanded)
        QSettings().setValue(self._settings_key, expanded)
        if expanded:
            self.setMaximumHeight(16777215)
            margins = self.body_layout.contentsMargins()
            header_height = self.header_widget.sizeHint().height()
            self.setMinimumHeight(max(header_height + margins.top() + margins.bottom(), self.sizeHint().height()))
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            header_height = self.header_widget.sizeHint().height()
            margins = self.body_layout.contentsMargins()
            collapsed_height = header_height + margins.top() + margins.bottom()
            self.setMinimumHeight(collapsed_height)
            self.setMaximumHeight(collapsed_height)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.updateGeometry()

    def is_expanded(self) -> bool:
        return self.header_button.isChecked()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e) -> None:
        if e is not None:
            e.ignore()


class GisaxsInputCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("GISAXS Image Input", "GisaxsInputCard")
        self.ui = ui
        content = ui.gisaxsInputBox
        profile = profile or current_profile(ui.centralwidget)
        self.setMinimumHeight(scale_value(332, profile, 272))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        content.setTitle("")
        if hasattr(content, "setFlat"):
            content.setFlat(True)
        content.setMinimumHeight(scale_value(272, profile, 224))
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._rebuild_layout(content, profile)
        self.add_content(content)

    def _rebuild_layout(self, content: QWidget, profile) -> None:
        layout = content.layout()
        if layout is None:
            layout = QGridLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        if hasattr(layout, "setHorizontalSpacing"):
            layout.setHorizontalSpacing(CARD_SPACING)
            layout.setVerticalSpacing(FORM_ROW_SPACING)
        else:
            layout.setSpacing(CARD_SPACING)

        self._detach_input_widgets()
        self._rebuild_stack_widget(profile)

        file_section = self._create_section_widget("File Input", content)
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(CARD_SPACING)
        self._configure_file_controls(profile)
        self.ui.gisaxsInputFileNavigationWidget = QWidget(file_section)
        self.ui.gisaxsInputFileNavigationWidget.setObjectName("gisaxsInputFileNavigationWidget")
        self.ui.gisaxsInputFileNavigationLayout = QHBoxLayout(self.ui.gisaxsInputFileNavigationWidget)
        self.ui.gisaxsInputFileNavigationLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.gisaxsInputFileNavigationLayout.setSpacing(max(6, CARD_SPACING - 2))
        self.ui.gisaxsInputFileNavigationWidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        file_row.addWidget(self.ui.gisaxsInputImportButton, 0)
        file_row.addWidget(self.ui.gisaxsInputImportButtonValue, 1)
        file_row.addWidget(self.ui.gisaxsInputFileNavigationWidget, 0, Qt.AlignRight)
        file_section.layout().addLayout(file_row)

        mode_section = self._create_section_widget("Load Mode", content)
        mode_grid = QGridLayout()
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setHorizontalSpacing(CARD_SPACING)
        mode_grid.setVerticalSpacing(max(4, FORM_ROW_SPACING - 2))
        normalize_input(self.ui.gisaxsInputModelCombox)
        self.ui.gisaxsInputModelCombox.setMinimumWidth(scale_value(110, profile, 92))
        self.ui.gisaxsInputModelCombox.setMaximumWidth(scale_value(150, profile, 132))
        self.ui.gisaxsInputModelCombox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        mode_grid.addWidget(self.ui.gisaxsInputModelCombox, 0, 0, Qt.AlignTop)
        mode_grid.addWidget(self.ui.gisaxsInputStackWidget, 0, 1, Qt.AlignTop)
        mode_grid.setColumnStretch(0, 0)
        mode_grid.setColumnStretch(1, 1)
        mode_section.layout().addLayout(mode_grid)

        show_section = self._create_section_widget("Image Display", content)
        show_row = QHBoxLayout()
        show_row.setContentsMargins(0, 0, 0, 0)
        show_row.setSpacing(CARD_SPACING)
        self.ui.gisaxsInputAutoShowCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        show_row.addWidget(self.ui.gisaxsInputAutoShowCheckBox, 0)
        show_row.addStretch(1)
        show_row.addWidget(self.ui.gisaxsInputShowButton, 0)
        show_section.layout().addLayout(show_row)

        scale_section = self._create_section_widget("Display Range", content)
        scale_grid = QGridLayout()
        scale_grid.setContentsMargins(0, 0, 0, 0)
        scale_grid.setHorizontalSpacing(CARD_SPACING)
        scale_grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.ui.gisaxsInputAutoScaleCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputIntLogCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        normalize_input(self.ui.gisaxsInputVminValue)
        normalize_input(self.ui.gisaxsInputVmaxValue)
        self.ui.gisaxsInputVminLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ui.gisaxsInputVmaxLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        scale_grid.addWidget(self.ui.gisaxsInputAutoScaleCheckBox, 0, 0)
        scale_grid.addWidget(self.ui.gisaxsInputVminLabel, 0, 1)
        scale_grid.addWidget(self.ui.gisaxsInputVminValue, 0, 2)
        scale_grid.addWidget(self.ui.gisaxsInputVmaxLabel, 0, 3)
        scale_grid.addWidget(self.ui.gisaxsInputVmaxValue, 0, 4)
        scale_grid.addWidget(self.ui.gisaxsInputIntLogCheckBox, 0, 5)
        scale_grid.setColumnStretch(6, 1)
        scale_section.layout().addLayout(scale_grid)

        layout.addWidget(file_section, 0, 0, 1, 4)
        layout.addWidget(mode_section, 1, 0, 1, 4)
        layout.addWidget(show_section, 2, 0, 1, 4)
        layout.addWidget(scale_section, 3, 0, 1, 4)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

    def _detach_input_widgets(self) -> None:
        widgets = (
            self.ui.gisaxsInputImportButton,
            self.ui.gisaxsInputImportButtonValue,
            self.ui.gisaxsInputModelCombox,
            self.ui.gisaxsInputStackWidget,
            self.ui.gisaxsInputColorScaleLabel,
            self.ui.gisaxsInputAutoScaleCheckBox,
            self.ui.gisaxsInputIntLogCheckBox,
            self.ui.gisaxsInputAutoShowCheckBox,
            self.ui.gisaxsInputShowButton,
            self.ui.gisaxsInputVminLabel,
            self.ui.gisaxsInputVminValue,
            self.ui.gisaxsInputVmaxLabel,
            self.ui.gisaxsInputVmaxValue,
        )
        for widget in widgets:
            _detach_from_parent_layout(widget)
        self.ui.gisaxsInputColorScaleLabel.hide()

    def _rebuild_stack_widget(self, profile) -> None:
        stack_layout = self.ui.gisaxsInputStackWidget.layout()
        if isinstance(stack_layout, QBoxLayout):
            _take_widget(stack_layout, self.ui.gisaxsInputStackValue)
            _take_widget(stack_layout, self.ui.gisaxsInputStackDisplayLabel)
            stack_layout.setDirection(QBoxLayout.TopToBottom)
            stack_layout.setContentsMargins(0, 0, 0, 0)
            stack_layout.setSpacing(max(4, CARD_SPACING - 2))

        editor_widget = getattr(self.ui, 'gisaxsInputStackEditorWidget', None)
        if editor_widget is None:
            editor_widget = QWidget(self.ui.gisaxsInputStackWidget)
            editor_widget.setObjectName('gisaxsInputStackEditorWidget')
            editor_layout = QHBoxLayout(editor_widget)
            editor_layout.setContentsMargins(0, 0, 0, 0)
            editor_layout.setSpacing(max(4, CARD_SPACING - 2))
            self.ui.gisaxsInputStackEditorWidget = editor_widget
            self.ui.gisaxsInputStackEditorLayout = editor_layout
        else:
            editor_layout = getattr(self.ui, 'gisaxsInputStackEditorLayout', None)
            if editor_layout is None:
                editor_layout = QHBoxLayout(editor_widget)
                editor_layout.setContentsMargins(0, 0, 0, 0)
                editor_layout.setSpacing(max(4, CARD_SPACING - 2))
                self.ui.gisaxsInputStackEditorLayout = editor_layout

        if isinstance(editor_layout, QBoxLayout):
            _take_widget(editor_layout, self.ui.gisaxsInputStackValue)

        normalize_input(self.ui.gisaxsInputStackValue)
        self.ui.gisaxsInputStackValue.setMinimumWidth(scale_value(120, profile, 96))
        self.ui.gisaxsInputStackValue.setMaximumWidth(scale_value(156, profile, 136))
        self.ui.gisaxsInputStackValue.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputStackDisplayLabel.setWordWrap(True)
        self.ui.gisaxsInputStackDisplayLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.ui.gisaxsInputStackDisplayLabel.setMinimumHeight(scale_value(36, profile, 30))
        self.ui.gisaxsInputStackDisplayLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.ui.gisaxsInputStackDisplayLabel.setStyleSheet("color: #64748b;")
        self.ui.gisaxsInputStackWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        if isinstance(stack_layout, QBoxLayout):
            if isinstance(editor_layout, QBoxLayout):
                editor_layout.addWidget(self.ui.gisaxsInputStackValue, 0, Qt.AlignLeft)
            stack_layout.addWidget(editor_widget, 0)
            stack_layout.addWidget(self.ui.gisaxsInputStackDisplayLabel, 0, Qt.AlignTop)
            stack_layout.addStretch(1)

    def _configure_file_controls(self, profile) -> None:
        normalize_input(self.ui.gisaxsInputImportButtonValue)
        self.ui.gisaxsInputImportButtonValue.setMinimumWidth(scale_value(260, profile, 220))
        self.ui.gisaxsInputImportButtonValue.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    @staticmethod
    def _create_section_widget(title: str, parent: QWidget) -> QWidget:
        section = QWidget(parent)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(max(6, CARD_SPACING - 2))
        label = QLabel(title, section)
        label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
        section_layout.addWidget(label)
        return section


class CutLineCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Cut Line and Detector", "CutLineCard")
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        self.setMinimumHeight(scale_value(370, self.profile, 315))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._managed_value_spinboxes = []
        self._managed_step_spinboxes = []
        self._managed_step_reset_buttons = []
        self._managed_labels = []
        self._managed_action_buttons = []

        self._detach_generated_widgets()
        self._rebuild_center_controls()

        cutline_group = self._make_group("Cut Line")
        cutline_layout = QGridLayout(cutline_group)
        self._configure_group_layout(cutline_layout)

        for col, text in enumerate(("Parameter", "Value", "Step", "Reset")):
            header_label = QLabel(text, cutline_group)
            header_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
            cutline_layout.addWidget(header_label, 0, col)

        rows = (
            (ui.gisaxsInputCutLineVerticalLabel, ui.gisaxsInputCutLineVerticalValue, "Vertical (px)", "gisaxsInputCutLineVerticalStep", 1.0),
            (ui.gisaxsInputCutLineParallelLabel, ui.gisaxsInputCutLineParallelValue, "Parallel (px)", "gisaxsInputCutLineParallelStep", 1.0),
            (ui.gisaxsInputCenterVerticalLabel, ui.gisaxsInputCenterVerticalValue, "Center Vertical (px)", "gisaxsInputCenterVerticalStep", 1.0),
            (ui.gisaxsInputCenterParallelLabel, ui.gisaxsInputCenterParallelValue, "Center Parallel (px)", "gisaxsInputCenterParallelStep", 1.0),
        )

        for row_index, (label, value_box, label_text, step_name, default_step) in enumerate(rows, 1):
            label.setText(label_text)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            normalize_input(value_box)
            step_box, reset_button = self._create_step_controls(step_name, value_box, default_step)
            cutline_layout.addWidget(label, row_index, 0)
            cutline_layout.addWidget(value_box, row_index, 1)
            cutline_layout.addWidget(step_box, row_index, 2)
            cutline_layout.addWidget(reset_button, row_index, 3)
            self._managed_labels.append(label)
            self._managed_value_spinboxes.append(value_box)

        unit_hint = QLabel("All cut geometry values use pixel units.", cutline_group)
        unit_hint.setStyleSheet("color: #64748b;")
        unit_hint.setWordWrap(True)
        self._managed_labels.append(unit_hint)
        cutline_layout.addWidget(unit_hint, 5, 0, 1, 3)
        cutline_layout.addWidget(ui.gisaxsInputCenterAutoFindingButton, 5, 3)
        cutline_layout.setColumnStretch(0, 0)
        cutline_layout.setColumnStretch(1, 1)
        cutline_layout.setColumnStretch(2, 0)
        cutline_layout.setColumnStretch(3, 0)

        detector_group = self._make_group("Detector and Cut")
        detector_layout = QGridLayout(detector_group)
        self._configure_group_layout(detector_layout)
        detector_hint = QLabel(
            "Configure detector parameters here before cutting the selected region.",
            detector_group,
        )
        detector_hint.setObjectName("cutLineDetectorHintLabel")
        detector_hint.setWordWrap(True)
        self._style_info_label(detector_hint)
        detector_layout.addWidget(ui.gisaxsInputDetectorParaButton, 0, 0)
        detector_layout.addWidget(detector_hint, 0, 1)
        detector_layout.addWidget(ui.gisaxsInputCutButton, 0, 2)
        detector_layout.setColumnStretch(0, 0)
        detector_layout.setColumnStretch(1, 1)
        detector_layout.setColumnStretch(2, 0)

        self._managed_action_buttons.extend(
            [
                ui.gisaxsInputCenterAutoFindingButton,
                ui.gisaxsInputDetectorParaButton,
                ui.gisaxsInputCutButton,
            ]
        )

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(scale_value(12, self.profile, 8))
        content_layout.addWidget(cutline_group)
        content_layout.addWidget(detector_group)
        self.body_layout.addLayout(content_layout)
        self._apply_responsive_profile()

    def _detach_generated_widgets(self) -> None:
        widgets = [
            self.ui.gisaxsInputCutLineLabel,
            self.ui.gisaxsInputCutLineVerticalLabel,
            self.ui.gisaxsInputCutLineVerticalValue,
            self.ui.gisaxsInputCutLineParallelLabel,
            self.ui.gisaxsInputCutLineParallelValue,
            self.ui.gisaxsInputCenterLabel,
            self.ui.gisaxsInputCenterAutoFindingButton,
            self.ui.gisaxsInputDetectorParaButton,
            self.ui.gisaxsInputCutButton,
            self.ui.gisaxsInputCutLineCenterWidget,
        ]
        for widget in widgets:
            _detach_from_parent_layout(widget)
        self.ui.gisaxsInputCutLineLabel.hide()
        self.ui.gisaxsInputCenterLabel.hide()

    def _rebuild_center_controls(self) -> None:
        center_layout = self.ui.gisaxsInputCutLineCenterWidget.layout()
        if isinstance(center_layout, QBoxLayout):
            for widget in (
                self.ui.gisaxsInputCenterVerticalLabel,
                self.ui.gisaxsInputCenterVerticalValue,
                self.ui.gisaxsInputCenterParallelLabel,
                self.ui.gisaxsInputCenterParallelValue,
            ):
                _take_widget(center_layout, widget)
        self.ui.gisaxsInputCutLineCenterWidget.hide()

    def _create_step_controls(self, object_name: str, value_spinbox: QDoubleSpinBox, default_step: float):
        step_box = NoWheelDoubleSpinBox(self)
        step_box.setObjectName(object_name)
        step_box.setDecimals(4)
        step_box.setRange(1e-4, 1e6)
        step_box.setSingleStep(default_step)
        step_box.setValue(default_step)
        step_box.setProperty("defaultStepValue", default_step)
        step_box.valueChanged.connect(lambda new_step, spin=value_spinbox: spin.setSingleStep(float(new_step)))
        reset_button = QPushButton("Reset", self)
        reset_button.setObjectName(f"{object_name}ResetButton")
        reset_button.clicked.connect(lambda _checked=False, step=step_box: self._reset_step_spinbox(step))
        setattr(self.ui, object_name, step_box)
        setattr(self.ui, reset_button.objectName(), reset_button)
        self._managed_step_spinboxes.append(step_box)
        self._managed_step_reset_buttons.append(reset_button)
        return step_box, reset_button

    @staticmethod
    def _reset_step_spinbox(step_spinbox: QDoubleSpinBox) -> None:
        default_value = step_spinbox.property("defaultStepValue")
        if default_value is None:
            return
        step_spinbox.setValue(float(default_value))

    def _make_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title, self)
        group.setObjectName(title.replace(" ", "") + "Group")
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

    @staticmethod
    def _style_info_label(label: QLabel) -> None:
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

    def _configure_group_layout(self, layout) -> None:
        group_margin = scale_value(10, self.profile, 8)
        group_top = scale_value(18, self.profile, 14)
        group_spacing = scale_value(12, self.profile, 8)
        layout.setContentsMargins(group_margin, group_top, group_margin, group_margin)
        if hasattr(layout, "setHorizontalSpacing"):
            layout.setHorizontalSpacing(group_spacing)
            layout.setVerticalSpacing(max(FORM_ROW_SPACING, group_spacing - 4))
        else:
            layout.setSpacing(group_spacing)

    def _apply_responsive_profile(self) -> None:
        input_height = BUTTON_HEIGHT + scale_value(4, self.profile, 4)
        value_width = scale_value(140, self.profile, 118)
        step_width = scale_value(92, self.profile, 78)
        reset_width = scale_value(88, self.profile, 76)
        action_width = scale_value(132, self.profile, 108)
        label_width = scale_value(156, self.profile, 132)

        for label in self._managed_labels:
            label.setMinimumWidth(label_width)
        for spinbox in self._managed_value_spinboxes:
            spinbox.setMinimumHeight(input_height)
            spinbox.setMinimumWidth(value_width)
            spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for step_box in self._managed_step_spinboxes:
            normalize_input(step_box)
            step_box.setMinimumHeight(input_height)
            step_box.setMinimumWidth(step_width)
            step_box.setMaximumWidth(step_width)
            step_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        for reset_button in self._managed_step_reset_buttons:
            normalize_button(reset_button)
            reset_button.setMinimumHeight(input_height)
            reset_button.setMinimumWidth(reset_width)
            reset_button.setMaximumWidth(reset_width)
            reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        for button in self._managed_action_buttons:
            normalize_button(button)
            button.setMinimumHeight(input_height)
            button.setMinimumWidth(action_width)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class FittingControlsCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Fitting Controls", "FittingControlsCard")
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        group_spacing = scale_value(12, self.profile, 8)
        group_margin = scale_value(10, self.profile, 8)
        group_top = scale_value(18, self.profile, 14)
        self.setMinimumHeight(scale_value(640, self.profile, 560))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._managed_group_layouts = []
        self._managed_buttons = []
        self._managed_inputs = []
        self._managed_spinboxes = []
        self._managed_step_spinboxes = []
        self._managed_step_reset_buttons = []
        self._managed_secondary_action_buttons = []
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
        ui.fitBGLabel = QLabel("BG:", self)
        ui.fitBGLabel.setObjectName("fitBGLabel")
        ui.fitBGValue = QDoubleSpinBox(self)
        ui.fitBGValue.setObjectName("fitBGValue")
        ui.fitBGValue.setDecimals(6)
        ui.fitBGValue.setRange(-1e10, 1e10)
        ui.fitBGValue.setSingleStep(0.1)
        ui.fitBGStep = NoWheelDoubleSpinBox(self)
        ui.fitBGStep.setObjectName("fitBGStep")
        ui.fitBGStep.setDecimals(6)
        ui.fitBGStep.setRange(1e-9, 1e9)
        ui.fitBGStep.setValue(0.1)
        ui.fitBGStep.setProperty("defaultStepValue", 0.1)
        ui.fitBGStep.valueChanged.connect(lambda value: ui.fitBGValue.setSingleStep(float(value)))
        ui.fitKStep = NoWheelDoubleSpinBox(self)
        ui.fitKStep.setObjectName("fitKStep")
        ui.fitKStep.setDecimals(6)
        ui.fitKStep.setRange(1e-9, 1e9)
        ui.fitKStep.setValue(0.1)
        ui.fitKStep.setProperty("defaultStepValue", 0.1)
        ui.fitKStep.valueChanged.connect(lambda value: ui.fitKValue.setSingleStep(float(value)))
        ui.fitIntResStep = NoWheelDoubleSpinBox(self)
        ui.fitIntResStep.setObjectName("fitIntResStep")
        ui.fitIntResStep.setDecimals(6)
        ui.fitIntResStep.setRange(1e-9, 1e9)
        ui.fitIntResStep.setValue(0.01)
        ui.fitIntResStep.setProperty("defaultStepValue", 0.01)
        ui.fitIntResStep.valueChanged.connect(lambda value: ui.fitIntResValue.setSingleStep(float(value)))
        ui.fitSigmaResStep = NoWheelDoubleSpinBox(self)
        ui.fitSigmaResStep.setObjectName("fitSigmaResStep")
        ui.fitSigmaResStep.setDecimals(6)
        ui.fitSigmaResStep.setRange(1e-9, 1e9)
        ui.fitSigmaResStep.setValue(0.1)
        ui.fitSigmaResStep.setProperty("defaultStepValue", 0.1)
        ui.fitSigmaResStep.valueChanged.connect(lambda value: ui.fitSigmaResValue.setSingleStep(float(value)))
        ui.fitNuResStep = NoWheelDoubleSpinBox(self)
        ui.fitNuResStep.setObjectName("fitNuResStep")
        ui.fitNuResStep.setDecimals(6)
        ui.fitNuResStep.setRange(1e-9, 1e9)
        ui.fitNuResStep.setValue(0.1)
        ui.fitNuResStep.setProperty("defaultStepValue", 0.1)
        ui.fitNuResStep.valueChanged.connect(lambda value: ui.fitNuResValue.setSingleStep(float(value)))
        ui.FittingAutoKButton.setText("Auto-K: OFF")
        ui.fitMethodValue.setToolTip("Method selection is not implemented yet.")
        self._method_notice_combo = ui.fitMethodValue
        self._method_notice_queued = False
        ui.fitMethodValue.installEventFilter(self)
        ui.fitMethodValue.currentIndexChanged.connect(lambda _index: self._queue_method_notice())

        self.fitExportPlotButton = QPushButton("Export Plot", self)
        self.fitExportPlotButton.setObjectName("fitExportPlotButton")
        self.fitExportPlotButton.clicked.connect(ui.FittingExportButton.click)

        self._managed_secondary_action_buttons.append(ui.FittingAutoKButton)

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
            ui.fitBGValue,
            ui.fitBGStep,
            ui.fitKValue,
            ui.fitKStep,
            ui.fitIntResValue,
            ui.fitIntResStep,
            ui.fitSigmaResValue,
            ui.fitSigmaResStep,
            ui.fitNuResValue,
            ui.fitNuResStep,
        ):
            self._managed_inputs.append(input_widget)

        self._managed_spinboxes = [ui.fitBGValue, ui.fitKValue, ui.fitIntResValue, ui.fitSigmaResValue, ui.fitNuResValue]
        self._managed_step_spinboxes = [ui.fitBGStep, ui.fitKStep, ui.fitIntResStep, ui.fitSigmaResStep, ui.fitNuResStep]

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

        global_group = self._make_group("Global Parameters")
        global_layout = QGridLayout(global_group)
        self._configure_group_layout(global_layout, group_margin, group_top, group_spacing)
        for col, text in enumerate(("Parameter", "Value", "Step", "Reset")):
            header_label = QLabel(text, global_group)
            header_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
            global_layout.addWidget(header_label, 0, col)

        reset_buttons = {
            ui.fitBGStep.objectName(): self._create_step_reset_button(global_group, ui.fitBGStep),
            ui.fitIntResStep.objectName(): self._create_step_reset_button(global_group, ui.fitIntResStep),
            ui.fitSigmaResStep.objectName(): self._create_step_reset_button(global_group, ui.fitSigmaResStep),
            ui.fitNuResStep.objectName(): self._create_step_reset_button(global_group, ui.fitNuResStep),
        }

        parameter_rows = (
            (ui.fitKLabel, ui.fitKValue, ui.fitKStep, "Scale Factor k"),
            (ui.fitBGLabel, ui.fitBGValue, ui.fitBGStep, "Background"),
            (ui.fitIntResLabel, ui.fitIntResValue, ui.fitIntResStep, "Resolution Intensity"),
            (ui.fitSigmaResLabel, ui.fitSigmaResValue, ui.fitSigmaResStep, "Resolution Sigma"),
            (ui.fitNuResLabel, ui.fitNuResValue, ui.fitNuResStep, "Resolution Nu"),
        )
        for row, (label, value, step, text) in enumerate(parameter_rows, 1):
            label.setText(text)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._managed_labels.append(label)
            global_layout.addWidget(label, row, 0)
            global_layout.addWidget(value, row, 1)
            global_layout.addWidget(step, row, 2)
            reset_button = reset_buttons.get(step.objectName())
            if reset_button is not None:
                global_layout.addWidget(reset_button, row, 3)

        global_layout.addWidget(ui.FittingAutoKButton, 1, 3)

        self.kInfoLabel = QLabel(
            "Model: I_fit(q) = BG + k * (sum(I_component(q)) + I_resolution(q))",
            global_group,
        )
        self.kInfoLabel.setObjectName("fitKInfoLabel")
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
        self.kInfoLabel.setToolTip(self.kInfoLabel.toolTip())
        ui.fitBGLabel.setToolTip(
            "Global Background\n"
            "I_fit(q) = BG + k * sum(I_component(q)) + k * I_resolution(q)\n"
            "BG is stored once and is not part of individual component cards."
        )
        ui.fitBGValue.setToolTip(ui.fitBGLabel.toolTip())
        resolution_tooltip = (
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
            ui.fitIntResStep,
            ui.fitSigmaResLabel,
            ui.fitSigmaResValue,
            ui.fitSigmaResStep,
            ui.fitNuResLabel,
            ui.fitNuResValue,
            ui.fitNuResStep,
        ):
            widget.setToolTip(resolution_tooltip)
        global_layout.addWidget(self.kInfoLabel, 6, 0, 1, 4)
        global_layout.setColumnStretch(1, 1)
        global_layout.setColumnStretch(2, 0)
        global_layout.setColumnStretch(3, 0)
        global_group.setMinimumHeight(scale_value(238, self.profile, 210))

        actions_group = self._make_group("Fitting Actions")
        actions_layout = QHBoxLayout(actions_group)
        self._configure_group_layout(actions_layout, group_margin, group_top, group_spacing)
        actions_layout.addWidget(ui.FittingClearFittingButton_2)
        actions_layout.addWidget(ui.FittingManualFittingButton)
        actions_layout.addWidget(ui.FittingExportButton)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(group_spacing)
        self._main_controls_layout = layout
        self.body_layout.addLayout(layout)
        layout.addWidget(data_options_group)
        layout.addWidget(external_group)
        layout.addWidget(actions_group)
        layout.addWidget(method_group)
        layout.addWidget(global_group)
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

    def _create_step_reset_button(self, parent: QWidget, step_spinbox: QDoubleSpinBox) -> QPushButton:
        button = QPushButton("Reset", parent)
        button.setObjectName(f"{step_spinbox.objectName()}ResetButton")
        button.clicked.connect(lambda _checked=False, spinbox=step_spinbox: self._reset_step_spinbox(spinbox))
        self._managed_step_reset_buttons.append(button)
        return button

    @staticmethod
    def _reset_step_spinbox(step_spinbox: QDoubleSpinBox) -> None:
        default_value = step_spinbox.property("defaultStepValue")
        if default_value is None:
            return
        step_spinbox.setValue(float(default_value))

    def _configure_group_layout(self, layout, margin: int, top: int, spacing: int) -> None:
        layout.setContentsMargins(margin, top, margin, margin)
        if hasattr(layout, "setHorizontalSpacing"):
            layout.setHorizontalSpacing(spacing)
            layout.setVerticalSpacing(max(FORM_ROW_SPACING, spacing - 4))
        else:
            layout.setSpacing(spacing)
        self._managed_group_layouts.append(layout)

    def _apply_secondary_action_button_width(self, minimum_width: int, input_height: int) -> None:
        buttons = [
            *self._managed_step_reset_buttons,
            *self._managed_secondary_action_buttons,
        ]
        if not buttons:
            return

        target_width = max(minimum_width, max(button.sizeHint().width() for button in buttons))
        for button in buttons:
            button.setMinimumHeight(input_height)
            button.setMinimumWidth(target_width)
            button.setMaximumWidth(target_width)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        group_spacing = scale_value(12, profile, 8)
        group_margin = scale_value(10, profile, 8)
        group_top = scale_value(18, profile, 14)
        self.setMinimumHeight(scale_value(640, profile, 560))
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
        spinbox_width = scale_value(138, profile, 118)
        step_width = scale_value(92, profile, 78)
        secondary_action_width = scale_value(88, profile, 76)
        label_width = scale_value(128, profile, 112)

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
            spinbox.setMaximumWidth(16777215)
        for spinbox in self._managed_step_spinboxes:
            spinbox.setMinimumWidth(step_width)
            spinbox.setMaximumWidth(step_width)
        self._apply_secondary_action_button_width(secondary_action_width, input_height)
        for label in self._managed_labels:
            label.setMinimumWidth(label_width)

        global_group = self.findChild(QGroupBox, "GlobalParametersGroup")
        if global_group is not None:
            global_group.setMinimumHeight(scale_value(238, profile, 210))
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
        self.setMinimumHeight(scale_value(220, profile, 180))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        add_button = getattr(ui, "pushButton", None)
        if add_button is not None:
            _detach_from_parent_layout(add_button)
            add_button.setText("+ Add Component")
            add_button.setMinimumWidth(scale_value(220, profile, 190))
            add_button.setMaximumWidth(scale_value(320, profile, 280))
            add_button.setMinimumHeight(scale_value(36, profile, 32))
            add_button.setMaximumHeight(scale_value(40, profile, 36))
            add_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

            self.body_layout.removeWidget(self.title_label)
            header = QWidget(self)
            header.setObjectName("modelParametersHeader")
            header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(CARD_SPACING)
            self.title_label.setParent(header)
            self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            header_layout.addWidget(self.title_label)
            header_layout.addStretch(1)
            header_layout.addWidget(add_button, 0, Qt.AlignRight)
            self.body_layout.insertWidget(0, header)

        _take_widget(ui.gridLayout_24, ui.widget_7)
        ui.widget_7.setMinimumWidth(0)
        ui.widget_7.setMaximumWidth(16777215)
        ui.widget_7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if ui.widget_7.layout() is not None:
            ui.widget_7.layout().setContentsMargins(0, 0, 0, 0)
            ui.widget_7.layout().setSpacing(0)
        inner_scroll_area = getattr(ui, "scrollArea", None)
        if inner_scroll_area is not None and ui.widget_7.layout() is not None:
            _take_widget(ui.widget_7.layout(), inner_scroll_area)
            content_widget = inner_scroll_area.takeWidget()
            if content_widget is not None:
                content_widget.setParent(ui.widget_7)
                content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
                ui.widget_7.layout().addWidget(content_widget)
            inner_scroll_area.deleteLater()
        particle_layout = getattr(ui, "scrollAreaWidgetContents", None)
        particle_layout = particle_layout.layout() if particle_layout is not None else None
        if isinstance(particle_layout, QBoxLayout):
            particle_layout.setDirection(QBoxLayout.TopToBottom)
            particle_layout.setContentsMargins(0, 0, 0, 0)
            particle_layout.setSpacing(scale_value(8, profile, 6))
            particle_layout.setAlignment(Qt.AlignTop)
            for index in range(particle_layout.count()):
                particle_layout.setStretch(index, 0)
        self.body_layout.addWidget(ui.widget_7, 1)


class DetectorPreviewCard(CollapsibleCardFrame):
    def __init__(self, graphics_view: QGraphicsView, profile=None):
        super().__init__("Detector Preview", "DetectorPreviewCard", default_expanded=True)
        profile = profile or current_profile(graphics_view)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(scale_value(260, profile, 210))
        hint = QLabel("Double-click the detector preview to open a larger independent image window.", self)
        hint.setObjectName("DetectorPreviewDoubleClickHint")
        hint.setProperty("cardMeta", True)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b;")
        self.add_content(hint)
        graphics_view.setToolTip("Double-click to open a larger independent image window.")
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
        super().__init__("Fitting Region", "FittingRegionControl", parent, profile=profile)

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

        filter_label = QLabel("Data Filter:", self)
        filter_label.setObjectName("fitRegionDataFilterLabel")
        filter_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        filter_widget = QWidget(self)
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(CARD_SPACING)
        ui.fitRegionPositiveOnlyCheckBox = QCheckBox("Positive Only", filter_widget)
        ui.fitRegionPositiveOnlyCheckBox.setObjectName("fitRegionPositiveOnlyCheckBox")
        ui.fitRegionNegativeOnlyCheckBox = QCheckBox("Negative Only", filter_widget)
        ui.fitRegionNegativeOnlyCheckBox.setObjectName("fitRegionNegativeOnlyCheckBox")
        normalize_checkbox(ui.fitRegionPositiveOnlyCheckBox)
        normalize_checkbox(ui.fitRegionNegativeOnlyCheckBox)
        filter_layout.addWidget(ui.fitRegionPositiveOnlyCheckBox)
        filter_layout.addWidget(ui.fitRegionNegativeOnlyCheckBox)
        filter_layout.addStretch(1)

        hint_label = QLabel(
            "Select Positive Only or Negative Only to edit Fitting Region.",
            self,
        )
        hint_label.setObjectName("fitRegionEditHintLabel")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(
            "QLabel {"
            "background: #eff6ff;"
            "border: 1px solid #bfdbfe;"
            "border-radius: 6px;"
            "color: #1d4ed8;"
            "padding: 6px 8px;"
            "line-height: 135%;"
            "}"
        )
        hint_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hint_label.setMinimumHeight(scale_value(42, profile, 34))
        hint_label.setVisible(False)
        ui.fitRegionEditHintLabel = hint_label

        layout.addWidget(filter_label, 0, 0)
        layout.addWidget(filter_widget, 0, 1)
        layout.addWidget(hint_label, 1, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionLabel, 2, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionSlider, 3, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionMinValue, 4, 0)
        layout.addWidget(ui.fitFittingRegionMaxValue, 4, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.setMinimumHeight(max(self.minimumHeight(), self.sizeHint().height() + scale_value(10, profile, 8)))
        self.setMaximumHeight(16777215)


class PlotSamplingControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        profile = profile or current_profile(parent or ui.centralwidget)
        super().__init__("Sampling", "PlotSamplingControl", parent, profile=profile)

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
            if parent.objectName() in ("PlotPreviewCard", "FittingPlotControlsCard"):
                base_min_height = getattr(parent, "_base_min_height", parent.minimumHeight())
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


class PlotPreviewCard(CollapsibleCardFrame):
    def __init__(self, ui, content: QWidget, graphics_view: QGraphicsView, profile=None):
        super().__init__("Fitting Plot", "PlotPreviewCard", default_expanded=True)
        profile = profile or current_profile(content)
        self._base_min_height = scale_value(360, profile, 280)
        hint = QLabel("Double-click the fitting plot to open a larger independent fit window.", self)
        hint.setObjectName("FittingPlotDoubleClickHint")
        hint.setProperty("cardMeta", True)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b;")
        self.add_content(hint)
        graphics_view.setToolTip("Double-click to open a larger independent fit window.")
        self._build_plot_layout(content, graphics_view, profile)

        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(self._base_min_height)
        self.add_content(content, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @staticmethod
    def _build_plot_layout(content: QWidget, graphics_view: QGraphicsView, profile) -> None:
        """Build only the plot canvas; controls live in FittingPlotControlsCard."""
        root_layout = content.layout()
        if root_layout is None:
            root_layout = QGridLayout(content)

        _take_widget(root_layout, graphics_view)
        plot_area = PlotCanvasArea(graphics_view, content, profile)
        plot_area.setMinimumHeight(scale_value(320, profile, 260))
        content.setMinimumSize(scale_value(300, profile, 260), scale_value(320, profile, 260))
        content.setMaximumSize(16777215, 16777215)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(plot_area, 0, 0)


class FittingPlotControlsCard(CollapsibleCardFrame):
    def __init__(self, ui, content: QWidget, profile=None):
        super().__init__("Fitting Controls", "FittingPlotControlsCard", default_expanded=True)
        profile = profile or current_profile(content)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(SECTION_MIN_WIDTH)

        root_layout = content.layout()
        if root_layout is None:
            root_layout = QGridLayout(content)

        for widget in (
            content.findChild(QWidget, "fitFittingRegionwidget"),
            content.findChild(QWidget, "fitDataPointsNumWidget"),
            content.findChild(QWidget, "fitFittingShowWidget"),
        ):
            if widget is not None:
                _take_widget(root_layout, widget)

        controls_container = QWidget(self.content_widget)
        controls_container.setObjectName("plotControlsContainer")
        controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(CARD_SPACING)
        controls_layout.addWidget(FittingRegionControl(ui, controls_container, profile))
        controls_layout.addWidget(PlotSamplingControl(ui, controls_container, profile))
        controls_layout.addWidget(PlotOptionsControl(ui, controls_container, profile))
        controls_container.setMinimumHeight(controls_container.sizeHint().height())
        controls_container.setMaximumHeight(16777215)
        self.add_content(controls_container)
        self.setMinimumHeight(max(self.minimumHeight(), self.sizeHint().height()))


class StatusCard(CollapsibleCardFrame):
    def __init__(self, browser: QWidget, profile=None):
        super().__init__("Run Log", "FittingStatusCard", default_expanded=True)
        profile = profile or current_profile(browser)
        self.body_layout.setContentsMargins(CARD_MARGIN, 6, CARD_MARGIN, 8)
        self.content_layout.setSpacing(4)
        browser_min_height = scale_value(180, profile, 140)
        self.setMinimumHeight(scale_value(230, profile, 176))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setMinimumHeight(browser_min_height)
        browser.setMaximumHeight(16777215)
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(browser, 1)


class GisaxsFittingWorkspace:
    """Three-region layout for the cut/fitting page."""

    SETTINGS_KEY = "gisaxs_fitting_splitter_sizes"
    DEFAULT_WORK_SIZES = [760, 680]

    def __init__(self, ui, profile=None):
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        self.DEFAULT_WORK_SIZES = list(self.profile.work_sizes)
        self.page_splitter = QSplitter(Qt.Horizontal, ui.gisaxsFittingPage)
        self.page_splitter.setObjectName("gisaxsFittingWorkspaceSplitter")
        self.page_splitter.setHandleWidth(8)
        self.page_splitter.setChildrenCollapsible(False)
        self.page_splitter.setOpaqueResize(True)

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
        self.fixed_controls_stack = QWidget()
        self.fixed_controls_stack.setObjectName("gisaxsFixedControlsStack")
        fixed_layout = QVBoxLayout(self.fixed_controls_stack)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.setSpacing(CARD_SPACING)
        gisaxs_card = GisaxsInputCard(self.ui, self.profile)
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
        self.right_panel = QWidget(self.page_splitter)
        self.right_panel.setObjectName("gisaxsRightCollapsiblePanel")
        self.right_panel.setMinimumWidth(self._preview_min_width())
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(CARD_SPACING)

        self.detector_preview_card = DetectorPreviewCard(self.ui.gisaxsInputGraphicsView, self.profile)
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

        right_layout.addWidget(self.detector_preview_card)
        right_layout.addWidget(self.fitting_plot_card)
        right_layout.addWidget(self.fitting_controls_card)
        right_layout.addWidget(self.run_log_card)
        right_layout.addStretch(1)

        self.preview_scroll_area = make_scroll_area(self.right_panel, horizontal=True)
        self.preview_scroll_area.setObjectName("gisaxsPreviewScrollArea")
        self.preview_scroll_area.setMinimumWidth(self._preview_min_width())

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
        _take_widget(layout, self.ui.gisaxsFittingPageScrollArea)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.page_splitter)

    def _page_min_width(self) -> int:
        return self.profile.workspace_min + self._preview_min_width() + self.page_splitter.handleWidth()

    def _preview_min_width(self) -> int:
        return max(self.profile.preview_min, scale_value(520, self.profile, 460))

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
        available = max(1, self._available_page_width() - self.page_splitter.handleWidth())
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
        sizes = user_settings.get(self.SETTINGS_KEY, None)
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
            if isinstance(work_sizes, (list, tuple)) and len(work_sizes) == 2:
                self.work_splitter.setSizes(
                    [
                        max(self.DEFAULT_WORK_SIZES[0], int(work_sizes[0])),
                        max(self.DEFAULT_WORK_SIZES[1], int(work_sizes[1])),
                    ]
                )
            else:
                self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)
            return

        self._set_page_sizes(self.profile.page_sizes)
        self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)

    def save_state(self) -> None:
        user_settings.set(
            self.SETTINGS_KEY,
            {
                "page": self.page_splitter.sizes(),
                "work": self.work_splitter.sizes(),
                "profile": self.profile.key,
            },
        )

    def apply_responsive_profile(self, profile) -> None:
        self.profile = profile
        self.DEFAULT_WORK_SIZES = list(profile.work_sizes)
        self._configure_button_responsiveness()
        fitting_card = self.fixed_controls_stack.findChild(FittingControlsCard, "FittingControlsCard")
        if fitting_card is not None:
            fitting_card.apply_responsive_profile(profile)
        self.right_panel.setMinimumWidth(self._preview_min_width())
        self.preview_scroll_area.setMinimumWidth(self._preview_min_width())
        self.work_splitter.setMinimumWidth(profile.workspace_min)
        self.ui.gisaxsFittingPageScrollArea.setMinimumWidth(profile.workspace_min)
        self._apply_page_overflow_policy()

        fixed_min = self._fixed_stack_min_height()
        self.fixed_controls_stack.setMinimumHeight(fixed_min)
        self.work_splitter.setMinimumHeight(
            fixed_min + self.DEFAULT_WORK_SIZES[1] + self.work_splitter.handleWidth()
        )
        self._set_page_sizes(profile.page_sizes)
        self.work_splitter.setSizes(self.DEFAULT_WORK_SIZES)

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
        apply_window_profile(ui.centralwidget.window(), self.responsive_profile, resize_window=True)
        install_adaptive_window_profile(ui.centralwidget.window(), self._on_screen_profile_changed)

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
        apply_window_profile(self.ui.centralwidget.window(), profile, resize_window=False)
        self.fitting_workspace.apply_responsive_profile(profile)
        self.shell.apply_responsive_profile(profile)
        apply_density_profile(self.ui.centralwidget, profile)

    def _on_screen_profile_changed(self, profile, screen) -> None:
        if profile.key == self.responsive_profile.key:
            apply_density_profile(self.ui.centralwidget, profile)
            return
        self.apply_responsive_profile(profile)
        apply_main_window_styles(self.ui)

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
