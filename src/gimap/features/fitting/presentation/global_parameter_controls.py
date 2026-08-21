"""Construction of global scientific parameter editors in the Fitting run card。"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QLabel

from src.gimap.app.presentation.responsive_layout import scale_value


def build_global_parameter_controls(
    card, ui, group_margin: int, group_top: int, group_spacing: int
):
    global_group = card._make_group("Global Parameters")
    global_layout = QGridLayout(global_group)
    global_layout.setAlignment(Qt.AlignTop)
    card._configure_group_layout(global_layout, group_margin, group_top, group_spacing)
    for col, text in enumerate(("Parameter", "Value", "Default step", "Action")):
        header_label = QLabel(text, global_group)
        header_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
        global_layout.addWidget(header_label, 0, col)

    reset_buttons = {
        ui.fitBGStep.objectName(): card._create_step_reset_button(global_group, ui.fitBGStep),
        ui.fitIntResStep.objectName(): card._create_step_reset_button(
            global_group, ui.fitIntResStep
        ),
        ui.fitSigmaResStep.objectName(): card._create_step_reset_button(
            global_group, ui.fitSigmaResStep
        ),
        ui.fitNuResStep.objectName(): card._create_step_reset_button(global_group, ui.fitNuResStep),
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
        card._managed_labels.append(label)
        global_layout.addWidget(label, row, 0)
        global_layout.addWidget(value, row, 1)
        global_layout.addWidget(step, row, 2)
        reset_button = reset_buttons.get(step.objectName())
        if reset_button is not None:
            global_layout.addWidget(reset_button, row, 3)

    global_layout.addWidget(ui.FittingAutoKButton, 1, 3)

    card.kInfoLabel = QLabel(
        "Model: I_fit(q) = BG + k * (sum(I_component(q)) + I_resolution(q))",
        global_group,
    )
    card.kInfoLabel.setObjectName("fitKInfoLabel")
    card.kInfoLabel.setWordWrap(True)
    card._style_info_label(card.kInfoLabel)
    card.kInfoLabel.setToolTip(
        "<b>Scaling factor k</b><br>"
        "<i>I</i><sub>fit</sub>(q) = k &middot; <i>I</i><sub>model</sub>(q)<br>"
        "Auto-K minimizes:<br>"
        "&Sigma;<sub>i</sub>[k &middot; <i>I</i><sub>base</sub>(q<sub>i</sub>) "
        "- <i>I</i><sub>exp</sub>(q<sub>i</sub>)]<sup>2</sup><br>"
        "Estimate:<br>"
        "k = (&lt;<i>I</i><sub>base</sub>, <i>I</i><sub>exp</sub>&gt;) / "
        "(&lt;<i>I</i><sub>base</sub>, <i>I</i><sub>base</sub>&gt;)"
    )
    ui.fitKLabel.setToolTip(card.kInfoLabel.toolTip())
    ui.fitKValue.setToolTip(card.kInfoLabel.toolTip())
    ui.FittingAutoKButton.setToolTip(card.kInfoLabel.toolTip())
    card.kInfoLabel.setToolTip(card.kInfoLabel.toolTip())
    ui.fitBGLabel.setToolTip(
        "Global Background\n"
        "I_fit(q) = BG + k * sum(I_component(q)) + k * I_resolution(q)\n"
        "BG is stored once and is not part of individual component cards."
    )
    ui.fitBGValue.setToolTip(ui.fitBGLabel.toolTip())
    resolution_tooltip = (
        "<b>Resolution component</b><br>"
        "R(q) = <i>I</i><sub>res</sub> / "
        "[1 + (|q| / &sigma;<sub>res</sub>)<sup>&nu;</sup>]<br>"
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
    global_layout.addWidget(card.kInfoLabel, 6, 0, 1, 4)
    step_hint = QLabel(
        "Default step controls the arrow-key and intentional Alt/Option + wheel increment. "
        "Edits are saved for the next session; Reset restores the built-in value.",
        global_group,
    )
    step_hint.setObjectName("fittingParameterStepHint")
    step_hint.setProperty("cardMeta", True)
    step_hint.setWordWrap(True)
    global_layout.addWidget(step_hint, 7, 0, 1, 4)
    global_layout.setColumnStretch(1, 1)
    global_layout.setColumnStretch(2, 0)
    global_layout.setColumnStretch(3, 0)
    global_group.setMinimumHeight(scale_value(238, card.profile, 210))

    return global_group


__all__ = ["build_global_parameter_controls"]
