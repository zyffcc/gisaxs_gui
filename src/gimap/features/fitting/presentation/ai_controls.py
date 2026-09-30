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
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import normalize_button
from src.gimap.app.presentation.theme import set_role

from .layout_primitives import (
    DisclosurePanel,
    detach_from_parent_layout as _detach_from_parent_layout,
)


def build_ai_controls(card, ui, group_margin: int, group_top: int, group_spacing: int):
    method_group = card._make_group("1D Predict · current workflow")
    method_layout = QVBoxLayout(method_group)
    card._configure_group_layout(
        method_layout, group_margin, group_top, 10
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
    ui.aiFittingStopButton.setProperty("gimapDangerAction", True)
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
    card.methodInfoLabel.setMinimumHeight(28)
    set_role(card.methodInfoLabel, "hint")

    ui.aiFittingModelComboBox.setMinimumWidth(300)
    ui.aiFittingModelComboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    ui.aiFittingConstraintComboBox.setMinimumWidth(210)
    ui.aiFittingConstraintComboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    button_specs = (
        (ui.aiFittingRefreshButton, 82),
        (ui.aiFittingOpenWorkspaceButton, 128),
        (ui.aiFittingExportOutputButton, 116),
        (ui.aiFittingCombinationButton, 148),
        (ui.aiFittingAdvancedConstraintsButton, 112),
        (ui.aiFittingFastPredictButton, 112),
        (ui.aiFittingFullAutoFitButton, 112),
        (ui.aiFittingStopButton, 72),
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
        button.setMinimumHeight(34)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    for button, width in button_specs:
        button.setMinimumWidth(width)
    ui.aiFittingFastPredictButton.setProperty("gimapPrimaryAction", True)
    ui.aiFittingFullAutoFitButton.setProperty("gimapPrimaryAction", True)

    def make_ai_label(text: str) -> QLabel:
        label = QLabel(text, method_group)
        label.setMinimumWidth(76)
        set_role(label, "caption")
        return label

    ui.aiFittingModelLabel = make_ai_label("AI Model")
    ui.aiFittingConstraintLabel = make_ai_label("Constraint")

    model_row = QHBoxLayout()
    model_row.setContentsMargins(0, 0, 0, 0)
    model_row.setSpacing(8)
    model_row.addWidget(ui.aiFittingModelLabel)
    model_row.addWidget(ui.aiFittingModelComboBox, 1)

    model_actions_row = QHBoxLayout()
    model_actions_row.setContentsMargins(0, 0, 0, 0)
    model_actions_row.setSpacing(8)
    model_actions_row.addWidget(ui.aiFittingRefreshButton)
    model_actions_row.addWidget(ui.aiFittingOpenWorkspaceButton)
    model_actions_row.addWidget(ui.aiFittingExportOutputButton)
    model_actions_row.addStretch(1)

    control_row = QHBoxLayout()
    control_row.setContentsMargins(0, 0, 0, 0)
    control_row.setSpacing(8)
    control_row.addWidget(ui.aiFittingConstraintLabel)
    control_row.addWidget(ui.aiFittingConstraintComboBox, 1)

    constraint_actions_row = QHBoxLayout()
    constraint_actions_row.setContentsMargins(0, 0, 0, 0)
    constraint_actions_row.setSpacing(8)
    constraint_actions_row.addWidget(ui.aiFittingFixedKComboBox)
    constraint_actions_row.addWidget(ui.aiFittingCombinationButton)
    constraint_actions_row.addWidget(ui.aiFittingAdvancedConstraintsButton)
    constraint_actions_row.addStretch(1)

    predict_row = QHBoxLayout()
    predict_row.setContentsMargins(0, 0, 0, 0)
    predict_row.setSpacing(8)
    predict_row.addWidget(ui.aiFittingFastPredictButton)
    predict_row.addWidget(ui.aiFittingFullAutoFitButton)
    predict_row.addWidget(ui.aiFittingStopButton)
    predict_row.addStretch(1)

    tuning_disclosure = DisclosurePanel(
        "Advanced AI tuning",
        "fittingAiTuningDisclosure",
        method_group,
    )
    tuning_content = QWidget(tuning_disclosure.content)
    tuning_grid = QGridLayout(tuning_content)
    tuning_grid.setContentsMargins(0, 0, 0, 0)
    tuning_grid.setHorizontalSpacing(8)
    tuning_grid.setVerticalSpacing(6)
    tuning_specs = (
        ("Samples", ui.aiFittingSamplesSpinBox),
        ("Refine top", ui.aiFittingRefineTopNSpinBox),
        ("Max eval", ui.aiFittingRefineMaxEvalSpinBox),
        ("Progress every", ui.aiFittingProgressEverySpinBox),
    )
    for idx, (label_text, editor) in enumerate(tuning_specs):
        label = QLabel(label_text, method_group)
        set_role(label, "caption")
        row, col = divmod(idx, 2)
        tuning_grid.addWidget(label, row, col * 2)
        tuning_grid.addWidget(editor, row, col * 2 + 1)
        editor.setMinimumWidth(82)
        editor.setMaximumWidth(116)
        editor.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # Retain compatibility handles for settings migration, outside the visible flow.
    legacy = QWidget(method_group)
    legacy.hide()
    for widget in (ui.aiFittingModelLabel, ui.aiFittingModelComboBox,
                   ui.aiFittingConstraintLabel, ui.aiFittingConstraintComboBox,
                   ui.aiFittingFixedKComboBox, ui.aiFittingCombinationButton,
                   ui.aiFittingAdvancedConstraintsButton, ui.aiFittingRefreshButton,
                   ui.aiFittingSamplesSpinBox, ui.aiFittingRefineTopNSpinBox,
                   ui.aiFittingRefineMaxEvalSpinBox, ui.aiFittingProgressEverySpinBox):
        widget.setParent(legacy)
    ui.aiFittingFullAutoFitButton.setText("Fit curve")
    ui.aiFittingFullAutoFitButton.setToolTip(
        "Find the particles behind the curve with the method chosen in Fit Settings (by default: the neural network "
        "proposes models, then a numerical fit refines them against the measured points)"
    )
    ui.aiFittingFastPredictButton.setText("AI guess only")
    ui.aiFittingFastPredictButton.setToolTip(
        "What the neural network (general V5, experimental) proposes, without the numerical refinement: fast, "
        "a starting point rather than a result"
    )
    ui.aiFittingFastPredictButton.setProperty("gimapPrimaryAction", False)
    ui.aiFittingOpenWorkspaceButton.setText("Fit settings && batch…")  # “&&”: a plain ampersand in a button
    ui.aiFittingOpenWorkspaceButton.setToolTip(
        "The method, the particle components and their limits, the fit range; and fitting many curve files at once"
    )
    caption = QLabel(
        "Finds the particles behind the curve — shape (sphere, cylinders), size, size spread and spacing — "
        "and draws the fitted curve on the data. Experimental: always compare the fit with the data.", method_group,
    )
    caption.setToolTip(
        "Method details: the general V5 network proposes composition candidates; the single random-cylinder "
        "specialist applies only to a known single random cylinder; points are the measured (native) ones. A numerical "
        "fall-back does not establish a unique composition."
    )
    caption.setWordWrap(True)
    method_layout.addWidget(caption)
    ui.aiFittingExperimentalButton = QPushButton("Physical fit (no AI)", method_group)
    ui.aiFittingExperimentalButton.setToolTip(
        "Sphere, random cylinder or vertical cylinder with size spread and spacing D, fitted numerically without the "
        "neural network (free amplitudes, wider resolution limits); about half a minute. Auto compares the single "
        "families; give a full composition in Fit Settings to fit a mixture."
    )
    normalize_button(ui.aiFittingExperimentalButton)
    # Two columns fit the control panel at every supported width: the main
    # action and Stop first, the alternatives below, then settings/output.
    actions = QGridLayout()
    actions.setContentsMargins(0, 0, 0, 0)
    actions.setHorizontalSpacing(8)
    actions.setVerticalSpacing(8)
    for index, button in enumerate((
        ui.aiFittingFullAutoFitButton,
        ui.aiFittingStopButton,
        ui.aiFittingExperimentalButton,
        ui.aiFittingFastPredictButton,
        ui.aiFittingOpenWorkspaceButton,
        ui.aiFittingExportOutputButton,
    )):
        button.setMinimumWidth(0)
        button.setMinimumHeight(30)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions.addWidget(button, index // 2, index % 2)
    actions.setColumnStretch(0, 1)
    actions.setColumnStretch(1, 1)
    method_layout.addLayout(actions)
    ui.fittingAiTuningDisclosure = tuning_disclosure
    tuning_disclosure.setParent(legacy)
    card.methodInfoLabel.setMinimumHeight(0)
    card.methodInfoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
    method_layout.addWidget(card.methodInfoLabel)

    return method_group


__all__ = ["build_ai_controls"]
