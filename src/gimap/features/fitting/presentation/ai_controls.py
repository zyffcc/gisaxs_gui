"""Construction of the AI controls inside the Fitting run card。"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from src.gimap.app.presentation.layout_primitives import normalize_button
from src.gimap.app.presentation.responsive_layout import scale_value

from .layout_primitives import (
    detach_from_parent_layout as _detach_from_parent_layout,
)


def build_ai_controls(card, ui, group_margin: int, group_top: int, group_spacing: int):
    method_group = card._make_group("AI Auto Fitting")
    method_layout = QVBoxLayout(method_group)
    card._configure_group_layout(
        method_layout, group_margin, group_top, scale_value(10, card.profile, 8)
    )
    _detach_from_parent_layout(ui.fitMethodLabel)
    _detach_from_parent_layout(ui.fitMethodValue)
    _detach_from_parent_layout(ui.FittingAutoFittingButton)
    for legacy_widget in (ui.fitMethodLabel, ui.fitMethodValue, ui.FittingAutoFittingButton):
        legacy_widget.setVisible(False)

    ui.aiFittingModelComboBox = QComboBox(method_group)
    ui.aiFittingRefreshButton = QPushButton("Refresh", method_group)
    ui.aiFittingOpenWorkspaceButton = QPushButton("Open Workspace", method_group)
    ui.aiFittingExportOutputButton = QPushButton("Export Output...", method_group)
    ui.aiFittingExportOutputButton.setEnabled(False)
    ui.aiFittingConstraintComboBox = QComboBox(method_group)
    ui.aiFittingConstraintComboBox.addItems(
        ["Free Prediction", "Fixed K", "Fixed Combination", "Current Manual Model"]
    )
    ui.aiFittingFixedKComboBox = QComboBox(method_group)
    ui.aiFittingFixedKComboBox.setObjectName("aiFittingFixedKComboBox")
    ui.aiFittingFixedKComboBox.addItems(["1", "2", "3", "4"])
    ui.aiFittingFixedKComboBox.setVisible(False)
    ui.aiFittingCombinationButton = QPushButton("Choose Combination...", method_group)
    ui.aiFittingCombinationButton.setObjectName("aiFittingCombinationButton")
    ui.aiFittingCombinationButton.setVisible(False)
    ui.aiFittingAdvancedConstraintsButton = QPushButton("Constraints...", method_group)
    ui.aiFittingFastPredictButton = QPushButton("Fast Predict", method_group)
    ui.aiFittingFullAutoFitButton = QPushButton("Full Auto Fit", method_group)
    ui.aiFittingStopButton = QPushButton("Stop", method_group)
    ui.aiFittingStopButton.setEnabled(False)
    ui.aiFittingSamplesSpinBox = QSpinBox(method_group)
    ui.aiFittingSamplesSpinBox.setObjectName("aiFittingSamplesSpinBox")
    ui.aiFittingSamplesSpinBox.setRange(1, 1_000_000)
    ui.aiFittingSamplesSpinBox.setValue(2000)
    ui.aiFittingRefineTopNSpinBox = QSpinBox(method_group)
    ui.aiFittingRefineTopNSpinBox.setObjectName("aiFittingRefineTopNSpinBox")
    ui.aiFittingRefineTopNSpinBox.setRange(0, 100)
    ui.aiFittingRefineTopNSpinBox.setValue(5)
    ui.aiFittingRefineMaxEvalSpinBox = QSpinBox(method_group)
    ui.aiFittingRefineMaxEvalSpinBox.setObjectName("aiFittingRefineMaxEvalSpinBox")
    ui.aiFittingRefineMaxEvalSpinBox.setRange(1, 100000)
    ui.aiFittingRefineMaxEvalSpinBox.setValue(80)
    ui.aiFittingSamplingStdSpinBox = QDoubleSpinBox(method_group)
    ui.aiFittingSamplingStdSpinBox.setObjectName("aiFittingSamplingStdSpinBox")
    ui.aiFittingSamplingStdSpinBox.setDecimals(5)
    ui.aiFittingSamplingStdSpinBox.setRange(0.00001, 10.0)
    ui.aiFittingSamplingStdSpinBox.setSingleStep(0.001)
    ui.aiFittingSamplingStdSpinBox.setValue(0.005)
    ui.aiFittingTargetLogRmseSpinBox = QDoubleSpinBox(method_group)
    ui.aiFittingTargetLogRmseSpinBox.setObjectName("aiFittingTargetLogRmseSpinBox")
    ui.aiFittingTargetLogRmseSpinBox.setDecimals(8)
    ui.aiFittingTargetLogRmseSpinBox.setRange(0.0, 10.0)
    ui.aiFittingTargetLogRmseSpinBox.setSingleStep(0.00000001)
    ui.aiFittingTargetLogRmseSpinBox.setValue(0.08)
    ui.aiFittingProgressEverySpinBox = QSpinBox(method_group)
    ui.aiFittingProgressEverySpinBox.setObjectName("aiFittingProgressEverySpinBox")
    ui.aiFittingProgressEverySpinBox.setRange(0, 10000)
    ui.aiFittingProgressEverySpinBox.setValue(20)
    ui.aiFittingRefineFtolSpinBox = QDoubleSpinBox(method_group)
    ui.aiFittingRefineFtolSpinBox.setObjectName("aiFittingRefineFtolSpinBox")
    ui.aiFittingRefineFtolSpinBox.setDecimals(10)
    ui.aiFittingRefineFtolSpinBox.setRange(0.0, 1.0)
    ui.aiFittingRefineFtolSpinBox.setSingleStep(0.00000001)
    ui.aiFittingRefineFtolSpinBox.setValue(1e-8)
    ui.aiFittingRefineXtolSpinBox = QDoubleSpinBox(method_group)
    ui.aiFittingRefineXtolSpinBox.setObjectName("aiFittingRefineXtolSpinBox")
    ui.aiFittingRefineXtolSpinBox.setDecimals(10)
    ui.aiFittingRefineXtolSpinBox.setRange(0.0, 1.0)
    ui.aiFittingRefineXtolSpinBox.setSingleStep(0.00000001)
    ui.aiFittingRefineXtolSpinBox.setValue(1e-8)
    ui.aiFittingRefineGtolSpinBox = QDoubleSpinBox(method_group)
    ui.aiFittingRefineGtolSpinBox.setObjectName("aiFittingRefineGtolSpinBox")
    ui.aiFittingRefineGtolSpinBox.setDecimals(10)
    ui.aiFittingRefineGtolSpinBox.setRange(0.0, 1.0)
    ui.aiFittingRefineGtolSpinBox.setSingleStep(0.00000001)
    ui.aiFittingRefineGtolSpinBox.setValue(1e-8)
    for workspace_only_widget in (
        ui.aiFittingSamplingStdSpinBox,
        ui.aiFittingTargetLogRmseSpinBox,
        ui.aiFittingRefineFtolSpinBox,
        ui.aiFittingRefineXtolSpinBox,
        ui.aiFittingRefineGtolSpinBox,
    ):
        workspace_only_widget.setVisible(False)
    card.methodInfoLabel = QLabel("Status: Ready", method_group)
    ui.aiFittingStatusLabel = card.methodInfoLabel
    card.methodInfoLabel.setObjectName("fitMethodInfoLabel")
    card.methodInfoLabel.setWordWrap(True)
    card.methodInfoLabel.setMinimumHeight(scale_value(28, card.profile, 24))
    card.methodInfoLabel.setStyleSheet(
        "QLabel {"
        "color: #1d4ed8;"
        "background: #eff6ff;"
        "border: 1px solid #bfdbfe;"
        "border-radius: 6px;"
        "padding: 5px 8px;"
        "}"
    )

    ui.aiFittingModelComboBox.setMinimumWidth(scale_value(300, card.profile, 240))
    ui.aiFittingModelComboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    ui.aiFittingConstraintComboBox.setMinimumWidth(scale_value(210, card.profile, 180))
    ui.aiFittingConstraintComboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    button_specs = (
        (ui.aiFittingRefreshButton, scale_value(96, card.profile, 88)),
        (ui.aiFittingOpenWorkspaceButton, scale_value(166, card.profile, 146)),
        (ui.aiFittingExportOutputButton, scale_value(146, card.profile, 128)),
        (ui.aiFittingCombinationButton, scale_value(176, card.profile, 150)),
        (ui.aiFittingAdvancedConstraintsButton, scale_value(138, card.profile, 120)),
        (ui.aiFittingFastPredictButton, scale_value(136, card.profile, 120)),
        (ui.aiFittingFullAutoFitButton, scale_value(136, card.profile, 120)),
        (ui.aiFittingStopButton, scale_value(86, card.profile, 76)),
    )
    for button in (
        ui.aiFittingRefreshButton,
        ui.aiFittingOpenWorkspaceButton,
        ui.aiFittingExportOutputButton,
        ui.aiFittingCombinationButton,
        ui.aiFittingAdvancedConstraintsButton,
        ui.aiFittingFastPredictButton,
        ui.aiFittingFullAutoFitButton,
        ui.aiFittingStopButton,
    ):
        normalize_button(button)
        button.setMinimumHeight(scale_value(34, card.profile, 30))
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    for button, width in button_specs:
        button.setMinimumWidth(width)

    def make_ai_label(text: str) -> QLabel:
        label = QLabel(text, method_group)
        label.setMinimumWidth(scale_value(76, card.profile, 66))
        label.setStyleSheet("font-size: 11px; font-weight: 600; color: #475569;")
        return label

    ui.aiFittingModelLabel = make_ai_label("AI Model")
    ui.aiFittingConstraintLabel = make_ai_label("Constraint")

    model_row = QHBoxLayout()
    model_row.setContentsMargins(0, 0, 0, 0)
    model_row.setSpacing(scale_value(8, card.profile, 6))
    model_row.addWidget(ui.aiFittingModelLabel)
    model_row.addWidget(ui.aiFittingModelComboBox, 1)

    model_actions_row = QHBoxLayout()
    model_actions_row.setContentsMargins(scale_value(84, card.profile, 72), 0, 0, 0)
    model_actions_row.setSpacing(scale_value(8, card.profile, 6))
    model_actions_row.addWidget(ui.aiFittingRefreshButton)
    model_actions_row.addWidget(ui.aiFittingOpenWorkspaceButton)
    model_actions_row.addWidget(ui.aiFittingExportOutputButton)
    model_actions_row.addStretch(1)

    control_row = QHBoxLayout()
    control_row.setContentsMargins(0, 0, 0, 0)
    control_row.setSpacing(scale_value(8, card.profile, 6))
    control_row.addWidget(ui.aiFittingConstraintLabel)
    control_row.addWidget(ui.aiFittingConstraintComboBox, 1)
    control_row.addWidget(ui.aiFittingFixedKComboBox)
    control_row.addWidget(ui.aiFittingCombinationButton)
    control_row.addWidget(ui.aiFittingAdvancedConstraintsButton)

    predict_row = QHBoxLayout()
    predict_row.setContentsMargins(scale_value(84, card.profile, 72), 0, 0, 0)
    predict_row.setSpacing(scale_value(8, card.profile, 6))
    predict_row.addWidget(ui.aiFittingFastPredictButton)
    predict_row.addWidget(ui.aiFittingFullAutoFitButton)
    predict_row.addWidget(ui.aiFittingStopButton)
    predict_row.addStretch(1)

    tuning_grid = QGridLayout()
    tuning_grid.setContentsMargins(scale_value(84, card.profile, 72), 0, 0, 0)
    tuning_grid.setHorizontalSpacing(scale_value(8, card.profile, 6))
    tuning_grid.setVerticalSpacing(scale_value(6, card.profile, 5))
    tuning_specs = (
        ("Samples", ui.aiFittingSamplesSpinBox),
        ("Refine top", ui.aiFittingRefineTopNSpinBox),
        ("Max eval", ui.aiFittingRefineMaxEvalSpinBox),
        ("Progress every", ui.aiFittingProgressEverySpinBox),
    )
    for idx, (label_text, editor) in enumerate(tuning_specs):
        label = QLabel(label_text, method_group)
        label.setStyleSheet("font-size: 11px; color: #475569;")
        row, col = divmod(idx, 3)
        tuning_grid.addWidget(label, row, col * 2)
        tuning_grid.addWidget(editor, row, col * 2 + 1)
        editor.setMinimumWidth(scale_value(82, card.profile, 72))
        editor.setMaximumWidth(scale_value(116, card.profile, 104))
        editor.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    method_layout.addLayout(model_row)
    method_layout.addLayout(model_actions_row)
    method_layout.addLayout(control_row)
    method_layout.addLayout(predict_row)
    method_layout.addLayout(tuning_grid)
    method_layout.addWidget(card.methodInfoLabel)

    return method_group


__all__ = ["build_ai_controls"]
