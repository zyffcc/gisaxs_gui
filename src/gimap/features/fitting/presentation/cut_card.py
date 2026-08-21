"""Feature-owned layout for Fitting cut and detector controls。"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QBoxLayout,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation import SafeWheelSpinBox
from src.gimap.app.presentation.layout_primitives import BUTTON_HEIGHT, FORM_ROW_SPACING, normalize_button, normalize_input
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .layout_primitives import CardFrame, DisclosurePanel, NoWheelDoubleSpinBox
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget
from .detector_setup_panel import DetectorSetupPanel


class CutLineCard(CardFrame):
    AUTO_CUT_THICKNESS_KEY = "fitting.yoneda_cut.horizontal_thickness_pixels"
    DEFAULT_AUTO_CUT_THICKNESS = 5

    def __init__(
        self,
        ui,
        profile=None,
        *,
        view_model,
        preferences: UserPreferencesRepository,
    ):
        super().__init__("Experiment Setup & Cut", "CutLineCard")
        self.ui = ui
        self.view_model = view_model
        self.preferences = preferences
        self.profile = profile or current_profile(ui.centralwidget)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._managed_value_spinboxes = []
        self._managed_step_spinboxes = []
        self._managed_step_reset_buttons = []
        self._managed_labels = []
        self._managed_action_buttons = []

        self._detach_view_widgets()
        self._rebuild_center_controls()

        detector_page = QWidget(self)
        detector_page.setObjectName("fittingDetectorSetupPage")
        detector_page_layout = QVBoxLayout(detector_page)
        detector_page_layout.setContentsMargins(0, 0, 0, 0)
        detector_panel = DetectorSetupPanel(
            view_model,
            ui.gisaxsInputDetectorParaButton,
            detector_page,
        )
        ui.fittingDetectorSetupPanel = detector_panel
        detector_page_layout.addWidget(detector_panel)
        detector_page_layout.addStretch(1)

        rows = (
            (
                ui.gisaxsInputCenterVerticalLabel,
                ui.gisaxsInputCenterVerticalValue,
                "Center Vertical (px)",
                "gisaxsInputCenterVerticalStep",
                1.0,
            ),
            (
                ui.gisaxsInputCenterParallelLabel,
                ui.gisaxsInputCenterParallelValue,
                "Center Parallel (px)",
                "gisaxsInputCenterParallelStep",
                1.0,
            ),
            (
                ui.gisaxsInputCutLineVerticalLabel,
                ui.gisaxsInputCutLineVerticalValue,
                "Cut Vertical (px)",
                "gisaxsInputCutLineVerticalStep",
                1.0,
            ),
            (
                ui.gisaxsInputCutLineParallelLabel,
                ui.gisaxsInputCutLineParallelValue,
                "Cut Parallel (px)",
                "gisaxsInputCutLineParallelStep",
                1.0,
            ),
        )

        ui.gisaxsInputCenterAutoFindingButton.setText("Find Yoneda & Set Cut")
        ui.gisaxsInputCenterAutoFindingButton.setProperty("gimapPrimaryAction", True)
        ui.gisaxsInputCutButton.setText("Extract / Update Cut")
        ui.gisaxsInputCutButton.setProperty("gimapPrimaryAction", True)

        center_cut_page = self._build_center_cut_page(rows)

        self._managed_action_buttons.extend(
            [
                ui.gisaxsInputCenterAutoFindingButton,
                ui.gisaxsInputDetectorParaButton,
                ui.gisaxsInputCutButton,
            ]
        )

        self.step_stack = QStackedWidget(self)
        self.step_stack.setObjectName("fittingConfigureStepStack")
        self.step_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        for page in (detector_page, center_cut_page):
            self.step_stack.addWidget(page)
        self.step_stack.currentChanged.connect(
            lambda _index: QTimer.singleShot(0, self._sync_step_height)
        )
        self.body_layout.addWidget(self.step_stack)
        self._step_index = {
            "setup": 0,
            "center": 1,
            "cut": 1,
            "center_cut": 1,
        }
        self.show_step("setup")
        self._apply_responsive_profile()
        self.lock_to_natural_height()

    def _build_center_cut_page(self, rows) -> QWidget:
        page = QWidget(self)
        page.setObjectName("fittingYonedaCutPage")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(scale_value(10, self.profile, 8))
        group = self._make_group("Yoneda & Cut")
        group.setTitle("")
        grid = QGridLayout(group)
        self._configure_group_layout(grid)
        for column, text in enumerate(("Parameter", "Value")):
            header = QLabel(text, group)
            header.setProperty("fittingTableHeader", True)
            grid.addWidget(header, 0, column)

        disclosure = DisclosurePanel("Step sizes", "fittingCutStepDisclosure", group)
        step_grid = QGridLayout()
        step_grid.setContentsMargins(0, 0, 0, 0)
        step_grid.setHorizontalSpacing(scale_value(8, self.profile, 6))
        step_grid.setVerticalSpacing(scale_value(6, self.profile, 5))
        for row_index, (label, value_box, label_text, step_name, default_step) in enumerate(
            rows, 1
        ):
            label.setText(label_text)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            normalize_input(value_box)
            step_box, reset_button = self._create_step_controls(
                step_name, value_box, default_step
            )
            grid.addWidget(label, row_index, 0)
            grid.addWidget(value_box, row_index, 1)
            step_grid.addWidget(QLabel(label_text, disclosure.content), row_index - 1, 0)
            step_grid.addWidget(step_box, row_index - 1, 1)
            step_grid.addWidget(reset_button, row_index - 1, 2)
            self._managed_labels.append(label)
            self._managed_value_spinboxes.append(value_box)

        hint_label = QLabel(
            "Find Yoneda sets the center and a horizontal cut band. Adjust the geometry, "
            "then explicitly extract the 1D curve.",
            group,
        )
        hint_label.setProperty("cardMeta", True)
        hint_label.setWordWrap(True)
        grid.addWidget(hint_label, len(rows) + 1, 0, 1, 2)
        disclosure.content_layout.addLayout(step_grid)
        grid.addWidget(disclosure, len(rows) + 2, 0, 1, 2)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        page_layout.addWidget(group)

        auto_row = QHBoxLayout()
        auto_row.setContentsMargins(0, 0, 0, 0)
        auto_row.setSpacing(scale_value(8, self.profile, 6))
        auto_label = QLabel("Auto horizontal cut thickness", page)
        auto_label.setToolTip(
            "Number of detector rows averaged around the automatically found Yoneda position."
        )
        auto_row.addWidget(auto_label)
        self.auto_cut_thickness_spinbox = SafeWheelSpinBox(page)
        self.auto_cut_thickness_spinbox.setObjectName(
            "gisaxsAutoYonedaCutThicknessSpinBox"
        )
        self.auto_cut_thickness_spinbox.setRange(1, 999)
        self.auto_cut_thickness_spinbox.setSuffix(" px")
        self.auto_cut_thickness_spinbox.setValue(self._load_auto_cut_thickness())
        self.auto_cut_thickness_spinbox.setToolTip(auto_label.toolTip())
        self.auto_cut_thickness_spinbox.editingFinished.connect(
            self._save_auto_cut_thickness
        )
        normalize_input(self.auto_cut_thickness_spinbox)
        self.ui.gisaxsAutoYonedaCutThicknessSpinBox = self.auto_cut_thickness_spinbox
        auto_row.addWidget(self.auto_cut_thickness_spinbox)
        auto_row.addStretch(1)
        auto_row.addWidget(self.ui.gisaxsInputCenterAutoFindingButton)
        page_layout.addLayout(auto_row)
        page_layout.addWidget(self.ui.gisaxsInputCutButton)
        page_layout.addStretch(1)
        # Both compatibility names now refer to the single merged disclosure.
        self.ui.fittingCutStepDisclosure = disclosure
        self.ui.fittingCenterStepDisclosure = disclosure
        return page

    def _load_auto_cut_thickness(self) -> int:
        value = self.preferences.get(
            self.AUTO_CUT_THICKNESS_KEY,
            self.DEFAULT_AUTO_CUT_THICKNESS,
        )
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = self.DEFAULT_AUTO_CUT_THICKNESS
        return min(999, max(1, value))

    def _save_auto_cut_thickness(self) -> None:
        self.preferences.set(
            self.AUTO_CUT_THICKNESS_KEY,
            int(self.auto_cut_thickness_spinbox.value()),
        )
        self.preferences.save()

    def show_step(self, key: str) -> None:
        index = self._step_index.get(key, 0)
        self.step_stack.setCurrentIndex(index)
        self.title_label.setText(
            {
                "setup": "Detector Setup",
                "center": "Yoneda & Cut",
                "cut": "Yoneda & Cut",
                "center_cut": "Yoneda & Cut",
            }.get(key, "Experiment Setup & Cut")
        )
        QTimer.singleShot(0, self._sync_step_height)

    def _sync_step_height(self) -> None:
        page = self.step_stack.currentWidget()
        if page is None:
            return
        height = max(page.minimumSizeHint().height(), page.sizeHint().height())
        self.step_stack.setMinimumHeight(height)
        self.step_stack.setMaximumHeight(16777215)
        self.updateGeometry()

    def _detach_view_widgets(self) -> None:
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

    def _create_step_controls(
        self, object_name: str, value_spinbox: QDoubleSpinBox, default_step: float
    ):
        step_box = NoWheelDoubleSpinBox(self)
        step_box.setObjectName(object_name)
        step_box.setDecimals(4)
        step_box.setRange(1e-4, 1e6)
        step_box.setSingleStep(default_step)
        step_box.setValue(default_step)
        step_box.setProperty("defaultStepValue", default_step)
        step_box.valueChanged.connect(
            lambda new_step, spin=value_spinbox: spin.setSingleStep(float(new_step))
        )
        reset_button = QPushButton("Reset", self)
        reset_button.setObjectName(f"{object_name}ResetButton")
        reset_button.clicked.connect(
            lambda _checked=False, step=step_box: self._reset_step_spinbox(step)
        )
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
            "border: none;"
            "margin-top: 10px;"
            "padding-top: 10px;"
            "background: transparent;"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "left: 0;"
            "padding: 0;"
            "font-weight: 650;"
            "}"
        )
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
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


__all__ = ["CutLineCard"]
