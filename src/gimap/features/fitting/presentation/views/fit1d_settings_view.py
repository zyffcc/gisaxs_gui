"""Static controls of fit1d (Predict parameters) on the single-curve page: the particles to try, shown under the
method, and its rarely changed settings, folded in the Fit step's Options."""

from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QWidget

COMPOSITION_TIP = (
    "Try …: fit1d fits every composition of these shapes and lists them as candidates, best first. "
    "The particles in Model: fit1d fits these, keeping the values you fixed and the ranges you set."
)


def setup_fit1d_choice(owner, parent, layout) -> QWidget:
    """Under the method, for Predict only: which particles fit1d fits."""
    owner.fit1d_row = QWidget(parent)
    row = QHBoxLayout(owner.fit1d_row)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel("Particles", owner.fit1d_row))
    owner.fit1d_composition = QComboBox(owner.fit1d_row)
    owner.fit1d_composition.setObjectName("fit1dComposition")
    owner.fit1d_composition.setToolTip(COMPOSITION_TIP)
    owner.fit1d_composition.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    owner.fit1d_composition.setMinimumContentsLength(12)
    row.addWidget(owner.fit1d_composition, 1)
    layout.addWidget(owner.fit1d_row)
    owner.fit1d_notes = QLabel("", parent)
    owner.fit1d_notes.setObjectName("fit1dPrefitNotes")
    owner.fit1d_notes.setWordWrap(True)
    owner.fit1d_notes.setProperty("gimapRole", "warning")
    owner.fit1d_notes.hide()
    layout.addWidget(owner.fit1d_notes)
    return owner.fit1d_row


def setup_fit1d_options(owner, section) -> QWidget:
    """The settings of Predict, in the Fit step's Options (``AdvancedSection``)."""
    holder = QWidget(section)
    holder.setObjectName("fit1dSettings")
    owner.fit1d_settings_panel = holder
    form = QFormLayout(holder)
    form.setContentsMargins(0, 0, 0, 0)
    owner.fit1d_network = QComboBox(holder)
    for text, value in (("General network", "general"), ("Au nanoparticles on Si (adapted)", "au_np_si")):
        owner.fit1d_network.addItem(text, value)
    owner.fit1d_policy = QComboBox(holder)
    for text, value in (("Auto (more starts when needed)", "auto"), ("Fast (one search)", "fast"),
                        ("Robust (always many starts)", "robust")):
        owner.fit1d_policy.addItem(text, value)
    owner.fit1d_mode = QComboBox(holder)
    for text, value in (("GISAXS (Yoneda cut)", "gisaxs"), ("SAXS (dilute, no structure factor)", "saxs")):
        owner.fit1d_mode.addItem(text, value)
    for combo in (owner.fit1d_network, owner.fit1d_policy, owner.fit1d_mode):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
    owner.fit1d_use_qz = QCheckBox("Cut qz (nm⁻¹)", holder)
    owner.fit1d_qz = QDoubleSpinBox(holder)
    owner.fit1d_qz.setRange(0, 10)
    owner.fit1d_qz.setDecimals(6)
    owner.fit1d_qz.setSuffix(" nm⁻¹")
    owner.fit1d_qz.setToolTip("For a Yoneda cut, enter the vacuum qz of the cut; unset means qz = 0 with a warning.")
    owner.fit1d_sigma_floor = QDoubleSpinBox(holder)
    owner.fit1d_sigma_floor.setRange(0, 50)
    owner.fit1d_sigma_floor.setDecimals(2)
    owner.fit1d_sigma_floor.setSuffix(" %")
    form.addRow("fit1d network", owner.fit1d_network)
    form.addRow("fit1d search", owner.fit1d_policy)
    form.addRow("fit1d curve type", owner.fit1d_mode)
    form.addRow(owner.fit1d_use_qz, owner.fit1d_qz)
    form.addRow("fit1d systematic error (%)", owner.fit1d_sigma_floor)
    section.add_widget(holder)
    return holder
