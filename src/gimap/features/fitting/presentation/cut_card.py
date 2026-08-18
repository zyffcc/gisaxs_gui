"""Feature-owned layout for Fitting cut and detector controls。"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QBoxLayout,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from src.gimap.app.presentation.layout_primitives import BUTTON_HEIGHT, FORM_ROW_SPACING, normalize_button, normalize_input
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .layout_primitives import CardFrame, NoWheelDoubleSpinBox
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget


class CutLineCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Cut Line and Detector", "CutLineCard")
        self.ui = ui
        self.profile = profile or current_profile(ui.centralwidget)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._managed_value_spinboxes = []
        self._managed_step_spinboxes = []
        self._managed_step_reset_buttons = []
        self._managed_labels = []
        self._managed_action_buttons = []

        self._detach_view_widgets()
        self._rebuild_center_controls()

        cutline_group = self._make_group("Cut Line")
        cutline_layout = QGridLayout(cutline_group)
        self._configure_group_layout(cutline_layout)

        for col, text in enumerate(("Parameter", "Value", "Step", "Reset")):
            header_label = QLabel(text, cutline_group)
            header_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
            cutline_layout.addWidget(header_label, 0, col)

        rows = (
            (
                ui.gisaxsInputCutLineVerticalLabel,
                ui.gisaxsInputCutLineVerticalValue,
                "Vertical (px)",
                "gisaxsInputCutLineVerticalStep",
                1.0,
            ),
            (
                ui.gisaxsInputCutLineParallelLabel,
                ui.gisaxsInputCutLineParallelValue,
                "Parallel (px)",
                "gisaxsInputCutLineParallelStep",
                1.0,
            ),
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
        )

        for row_index, (label, value_box, label_text, step_name, default_step) in enumerate(
            rows, 1
        ):
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
        self.lock_to_natural_height()

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
