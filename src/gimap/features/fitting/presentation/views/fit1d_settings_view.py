"""Static controls of the fit1d method on the single-curve page."""

from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QWidget


def setup_fit1d_settings(owner, parent, layout):
    panel = QWidget(parent)
    panel.setObjectName("fit1dSettings")
    form = QFormLayout(panel)
    owner.fit1d_settings_panel = panel
    owner.fit1d_network = QComboBox(panel)
    for text, value in (("General network", "general"), ("Au nanoparticles on Si (adapted)", "au_np_si")):
        owner.fit1d_network.addItem(text, value)
    owner.fit1d_policy = QComboBox(panel)
    for text, value in (("Auto (more starts when needed)", "auto"), ("Fast (one search)", "fast"),
                        ("Robust (always many starts)", "robust")):
        owner.fit1d_policy.addItem(text, value)
    owner.fit1d_mode = QComboBox(panel)
    for text, value in (("GISAXS (Yoneda cut)", "gisaxs"), ("SAXS (dilute, no structure factor)", "saxs")):
        owner.fit1d_mode.addItem(text, value)
    for combo in (owner.fit1d_network, owner.fit1d_policy, owner.fit1d_mode):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
    owner.fit1d_use_qz = QCheckBox("Cut qz (nm⁻¹)", panel)
    owner.fit1d_qz = QDoubleSpinBox(panel)
    owner.fit1d_qz.setRange(0, 10)
    owner.fit1d_qz.setDecimals(6)
    owner.fit1d_qz.setSuffix(" nm⁻¹")
    owner.fit1d_qz.setToolTip("For a Yoneda cut, enter the vacuum qz of the cut; unset means qz = 0 with a warning.")
    owner.fit1d_sigma_floor = QDoubleSpinBox(panel)
    owner.fit1d_sigma_floor.setRange(0, 50)
    owner.fit1d_sigma_floor.setDecimals(2)
    owner.fit1d_sigma_floor.setSuffix(" %")
    form.addRow("fit1d network", owner.fit1d_network)
    form.addRow("fit1d search", owner.fit1d_policy)
    form.addRow("fit1d curve type", owner.fit1d_mode)
    form.addRow(owner.fit1d_use_qz, owner.fit1d_qz)
    form.addRow("fit1d systematic error (%)", owner.fit1d_sigma_floor)
    note = QLabel("Add 1–4 Sphere / Vertical cylinder particles in Model. Their families are given by you; "
                  "fit1d fits all populations together. Its v2 curve, errors and terms are kept separately from "
                  "the manual model. Only the composition is used: Model values, ranges and fixed parameters "
                  "are not passed to fit1d.", panel)
    note.setWordWrap(True)
    form.addRow(note)
    layout.addWidget(panel)
    return panel
