"""Feature-owned layout for manual and AI Fitting run controls。"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QTimer
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.ports import UserPreferencesRepository
from src.gimap.app.presentation.layout_primitives import BUTTON_HEIGHT, FORM_ROW_SPACING, normalize_button, set_expanding_x
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .ai_controls import build_ai_controls
from .global_parameter_controls import build_global_parameter_controls
from .layout_primitives import CardFrame, CurrentPageHeightTabWidget, NoWheelDoubleSpinBox
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .parameter_step_preferences import ParameterStepPreferences


class FittingControlsCard(CardFrame):
    def __init__(
        self,
        ui,
        profile=None,
        *,
        model_parameters_card,
        preferences: UserPreferencesRepository,
    ):
        super().__init__("Fit workspace", "FittingControlsCard")
        self.title_label.hide()
        self.ui = ui
        self.preferences = preferences
        self.parameter_step_preferences = ParameterStepPreferences(preferences)
        self.profile = profile or current_profile(ui.centralwidget)
        group_spacing = scale_value(12, self.profile, 8)
        group_margin = scale_value(10, self.profile, 8)
        group_top = scale_value(18, self.profile, 14)
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
        ui.fitCurrentDataCheckBox.setText("Use current cut")
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
        self.parameter_step_preferences.bind(
            ui.fitBGStep, ui.fitBGValue, "background", 0.1
        )
        ui.fitKStep = NoWheelDoubleSpinBox(self)
        ui.fitKStep.setObjectName("fitKStep")
        self.parameter_step_preferences.bind(ui.fitKStep, ui.fitKValue, "scale", 0.1)
        ui.fitIntResStep = NoWheelDoubleSpinBox(self)
        ui.fitIntResStep.setObjectName("fitIntResStep")
        self.parameter_step_preferences.bind(
            ui.fitIntResStep, ui.fitIntResValue, "resolution_intensity", 0.01
        )
        ui.fitSigmaResStep = NoWheelDoubleSpinBox(self)
        ui.fitSigmaResStep.setObjectName("fitSigmaResStep")
        self.parameter_step_preferences.bind(
            ui.fitSigmaResStep, ui.fitSigmaResValue, "resolution_sigma", 0.0001
        )
        ui.fitNuResStep = NoWheelDoubleSpinBox(self)
        ui.fitNuResStep.setObjectName("fitNuResStep")
        self.parameter_step_preferences.bind(
            ui.fitNuResStep, ui.fitNuResValue, "resolution_nu", 0.1
        )
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

        self._managed_spinboxes = [
            ui.fitBGValue,
            ui.fitKValue,
            ui.fitIntResValue,
            ui.fitSigmaResValue,
            ui.fitNuResValue,
        ]
        self._managed_step_spinboxes = [
            ui.fitBGStep,
            ui.fitKStep,
            ui.fitIntResStep,
            ui.fitSigmaResStep,
            ui.fitNuResStep,
        ]

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

        method_group = build_ai_controls(self, ui, group_margin, group_top, group_spacing)
        global_group = build_global_parameter_controls(
            self, ui, group_margin, group_top, group_spacing
        )

        actions_group = self._make_group("Fitting Actions")
        actions_layout = QHBoxLayout(actions_group)
        self._configure_group_layout(actions_layout, group_margin, group_top, group_spacing)
        ui.FittingAutoRefineButton = QPushButton("Auto Refine", actions_group)
        ui.FittingAutoRefineButton.setObjectName("FittingAutoRefineButton")
        normalize_button(ui.FittingAutoRefineButton)
        ui.FittingAutoRefineButton.setMinimumHeight(scale_value(34, self.profile, 30))
        ui.FittingAutoRefineButton.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._managed_buttons.append(ui.FittingAutoRefineButton)
        actions_layout.addWidget(ui.FittingClearFittingButton_2)
        actions_layout.addWidget(ui.FittingAutoRefineButton)
        actions_layout.addWidget(ui.FittingExportButton)
        ui.FittingManualFittingButton.setText("Plot Current Model")
        ui.FittingManualFittingButton.setProperty("gimapPrimaryAction", True)
        ui.FittingAutoRefineButton.setProperty("gimapPrimaryAction", True)

        manual_page = QWidget(self)
        manual_page.setObjectName("fittingManualModePage")
        manual_layout = QVBoxLayout(manual_page)
        manual_layout.setContentsMargins(8, 10, 8, 10)
        manual_layout.setSpacing(group_spacing)
        manual_layout.addWidget(data_options_group)
        manual_layout.addWidget(external_group)
        manual_layout.addWidget(actions_group)

        ai_page = QWidget(self)
        ai_page.setObjectName("fittingAiModePage")
        ai_layout = QVBoxLayout(ai_page)
        ai_layout.setContentsMargins(8, 10, 8, 10)
        ai_layout.setSpacing(group_spacing)
        ai_layout.addWidget(method_group)

        self.mode_tabs = CurrentPageHeightTabWidget(self)
        self.mode_tabs.setObjectName("fittingModeTabs")
        self.mode_tabs.addTab(model_parameters_card, "Components")
        self.mode_tabs.addTab(global_group, "Global")
        self.mode_tabs.addTab(manual_page, "Data & refine")
        self.mode_tabs.addTab(ai_page, "Auto fit")
        self.mode_tabs.setDocumentMode(True)
        self.mode_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_tabs.currentChanged.connect(
            lambda _index: QTimer.singleShot(0, self._sync_mode_tab_height)
        )
        tuning_disclosure = getattr(ui, "fittingAiTuningDisclosure", None)
        if tuning_disclosure is not None:
            tuning_disclosure.toggle.toggled.connect(
                lambda _expanded: QTimer.singleShot(0, self._sync_mode_tab_height)
            )
        self.ui.fittingModeTabs = self.mode_tabs

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(group_spacing)
        self._main_controls_layout = layout
        self.body_layout.addLayout(layout)
        self.fit_command_bar = QWidget(self)
        self.fit_command_bar.setObjectName("fittingPersistentCommandBar")
        self.fit_command_bar.setProperty("gimapPersistentCommandBar", True)
        command_layout = QHBoxLayout(self.fit_command_bar)
        command_layout.setContentsMargins(8, 6, 8, 6)
        command_layout.setSpacing(group_spacing)
        command_label = QLabel(
            "Preview the current component and global parameter values", self.fit_command_bar
        )
        command_label.setProperty("cardMeta", True)
        command_label.setWordWrap(True)
        command_layout.addWidget(command_label, 1)
        command_layout.addWidget(ui.FittingManualFittingButton, 0)
        layout.addWidget(self.fit_command_bar)
        layout.addWidget(self.mode_tabs)
        self._sync_mode_tab_height()
        self.apply_responsive_profile(self.profile)

    def _make_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title, self)
        group.setObjectName(title.replace(" ", "").replace("/", "") + "Group")
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

    def _create_step_reset_button(
        self, parent: QWidget, step_spinbox: QDoubleSpinBox
    ) -> QPushButton:
        button = QPushButton("Reset", parent)
        button.setObjectName(f"{step_spinbox.objectName()}ResetButton")
        button.clicked.connect(
            lambda _checked=False, spinbox=step_spinbox: self._reset_step_spinbox(spinbox)
        )
        self._managed_step_reset_buttons.append(button)
        return button

    def _reset_step_spinbox(self, step_spinbox: QDoubleSpinBox) -> None:
        self.parameter_step_preferences.reset(step_spinbox)

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
            global_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._sync_mode_tab_height()
        self.setMinimumHeight(0)
        if self.layout() is not None:
            self.layout().activate()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.updateGeometry()

    def _sync_mode_tab_height(self) -> None:
        if not hasattr(self, "mode_tabs"):
            return
        page = self.mode_tabs.currentWidget()
        if page is None:
            return
        if page.layout() is not None:
            page.layout().activate()
        self.mode_tabs.setMinimumHeight(0)
        # A page can inherit a stale, oversized viewport allocation while the
        # outer Single/In-situ context changes. The current page's minimum
        # hint is its allocation-independent natural control height.
        height = self.mode_tabs.minimumSizeHint().height()
        self.mode_tabs.setMinimumHeight(height)
        self.mode_tabs.setMaximumHeight(16777215)
        self.mode_tabs.updateGeometry()
        self._main_controls_layout.invalidate()
        self.body_layout.invalidate()
        self.setMinimumHeight(0)
        if self.layout() is not None:
            self.layout().activate()
        self.updateGeometry()

    def sizeHint(self):
        hint = super().sizeHint()
        if hasattr(self, "mode_tabs"):
            margins = self.body_layout.contentsMargins()
            command_height = self.fit_command_bar.sizeHint().height()
            hint.setHeight(
                margins.top()
                + command_height
                + self._main_controls_layout.spacing()
                + self.mode_tabs.sizeHint().height()
                + margins.bottom()
            )
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        if hasattr(self, "mode_tabs"):
            margins = self.body_layout.contentsMargins()
            command_height = self.fit_command_bar.minimumSizeHint().height()
            hint.setHeight(
                margins.top()
                + command_height
                + self._main_controls_layout.spacing()
                + self.mode_tabs.minimumSizeHint().height()
                + margins.bottom()
            )
        return hint

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
        if (
            obj is getattr(self, "_method_notice_combo", None)
            and event.type() == QEvent.MouseButtonPress
        ):
            self._queue_method_notice()
        return super().eventFilter(obj, event)


__all__ = ["FittingControlsCard"]
