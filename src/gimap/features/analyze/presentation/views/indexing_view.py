"""Static layout of the Phases window of Analyze: the phases, their reflections and the pole figure of a ring."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components.curve_plot import CurvePlot
from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import DATA_HEADERS

PHASE_COLUMNS = ("Show", "Name", "Cell (Å, °)", "Symmetry", "Orientation")
REFLECTION_COLUMNS = ("Phase", "hkl", "d (Å)", "q (Å⁻¹)", "q∥ (Å⁻¹)", "qz (Å⁻¹)", "χ (°)", "m", "|F|²", "I (%)")
ORIENTATION_ITEMS = (
    ("powder", "Powder (random orientation)"),
    ("plane", "Plane (hkl) parallel to the substrate"),
    ("axis", "Axis [uvw] along the surface normal"),
)
CENTRING_ITEMS = ("P", "A", "B", "C", "I", "F", "R")
MODEL_ITEMS = (("gaussian", "Gaussian + isotropic"), ("maier_saupe", "Maier–Saupe + isotropic"))
DIRECTION_ITEMS = (
    ("normal", "Along the normal (χ0 = 0°)"),
    ("plane", "In the plane (χ0 = 90°)"),
    ("fitted", "Fitted"),
    ("set", "Set χ0"),
)


def make_phases_button(parent: QWidget) -> QToolButton:
    """The button of the image toolbar that opens the Phases window."""
    button = QToolButton(parent)
    button.setObjectName("analyzePhasesButton")
    button.setText("Phases")
    button.setCheckable(True)
    button.setToolTip("Overlay the reflections of known phases (a unit cell or a CIF) on the q map, the detector and "
                      "I(q); assign the rings; fit the pole figure of a ring")
    return button


def _spin(parent: QWidget, low: float, high: float, value: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _index(parent: QWidget, value: int) -> QSpinBox:
    spin = QSpinBox(parent)
    spin.setRange(-30, 30)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    return spin


class IndexingView:
    """``setup_ui(dialog)`` builds the window; the behaviour is in ``indexing_panel.py``."""

    def setup_ui(self, dialog) -> None:
        dialog.setObjectName("analyzeIndexingWindow")
        dialog.setWindowTitle("Phases and Indexing")
        dialog.resize(760, 640)
        layout = QVBoxLayout(dialog)
        self.tabs = QTabWidget(dialog)
        self.tabs.addTab(self._phases_tab(dialog), "Phases")
        self.tabs.addTab(self._reflections_tab(dialog), "Reflections")
        self.tabs.addTab(self._pole_tab(dialog), "Pole Figure")
        layout.addWidget(self.tabs, 1)
        self.status_label = QLabel(dialog)
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("gimapRole", "muted")
        layout.addWidget(self.status_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close, dialog)
        self.buttons.button(QDialogButtonBox.Close).setAutoDefault(False)
        layout.addWidget(self.buttons)

    # -- phases -------------------------------------------------------------------------

    def _phases_tab(self, dialog) -> QWidget:
        tab = QWidget(dialog)
        layout = QVBoxLayout(tab)
        buttons = QHBoxLayout()
        self.add_cif_button = QPushButton("Add from CIF…", tab)
        self.add_cif_button.setToolTip("Read the cell, the symmetry operations and the atom sites of a CIF")
        self.add_cell_button = QPushButton("Add Cell", tab)
        self.add_cell_button.setToolTip("A phase from a unit cell typed in, with its lattice centring")
        self.remove_button = QPushButton("Remove", tab)
        for button in (self.add_cif_button, self.add_cell_button, self.remove_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.phase_table = QTableWidget(0, len(PHASE_COLUMNS), tab)
        self.phase_table.setObjectName("analyzePhaseTable")
        self.phase_table.setHorizontalHeaderLabels(PHASE_COLUMNS)
        self.phase_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.phase_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.phase_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.phase_table.verticalHeader().hide()
        self.phase_table.setWordWrap(False)
        self.phase_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        header = self.phase_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # the long symmetry text is cut (whole in its tooltip)
        self.phase_table.setMinimumHeight(110)
        self.phase_table.setMaximumHeight(170)
        layout.addWidget(self.phase_table)
        layout.addWidget(self._editor(tab))
        layout.addWidget(self._settings(tab))
        layout.addStretch(1)
        return tab

    def _editor(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Selected Phase", parent)
        grid = QGridLayout(box)
        self.name_edit = QLineEdit(box)
        grid.addWidget(QLabel("Name", box), 0, 0)
        grid.addWidget(self.name_edit, 0, 1, 1, 5)
        self.cell_spins = [_spin(box, 0.5, 500.0, 5.0, 4) for _ in range(3)]
        self.angle_spins = [_spin(box, 10.0, 170.0, 90.0, 3) for _ in range(3)]
        for column, (label, spin) in enumerate(zip(("a (Å)", "b (Å)", "c (Å)"), self.cell_spins)):
            grid.addWidget(QLabel(label, box), 1, 2 * column)
            grid.addWidget(spin, 1, 2 * column + 1)
        for column, (label, spin) in enumerate(zip(("α (°)", "β (°)", "γ (°)"), self.angle_spins)):
            grid.addWidget(QLabel(label, box), 2, 2 * column)
            grid.addWidget(spin, 2, 2 * column + 1)
        self.centring_combo = QComboBox(box)
        self.centring_combo.addItems(CENTRING_ITEMS)
        self.centring_combo.setToolTip("Lattice centring of a cell typed in (the reflections it extinguishes)")
        self.symmetry_label = QLabel(box)
        self.symmetry_label.setProperty("gimapRole", "muted")
        self.symmetry_label.setWordWrap(True)
        grid.addWidget(QLabel("Centring", box), 3, 0)
        grid.addWidget(self.centring_combo, 3, 1)
        grid.addWidget(self.symmetry_label, 3, 2, 1, 4)
        self.orientation_combo = QComboBox(box)
        for key, text in ORIENTATION_ITEMS:
            self.orientation_combo.addItem(text, key)
        self.orientation_combo.setToolTip(
            "How the crystallites lie: at random, or at random about the surface normal with a plane parallel to "
            "the substrate or an axis along the normal (a fibre texture)")
        self.index_spins = [_index(box, value) for value in (0, 0, 1)]
        grid.addWidget(QLabel("Orientation", box), 4, 0)
        grid.addWidget(self.orientation_combo, 4, 1, 1, 2)
        indices = QHBoxLayout()
        for spin in self.index_spins:
            indices.addWidget(spin)
        grid.addLayout(indices, 4, 3, 1, 3)
        return box

    def _settings(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Overlay", parent)
        form = QFormLayout(box)
        self.q_max_spin = _spin(box, 0.2, 12.0, 3.0, 2, " Å⁻¹")
        self.q_max_spin.setToolTip("Reflections up to this |q|")
        form.addRow("q max", self.q_max_spin)
        self.refraction_check = QCheckBox("Shift the fibre spots for refraction in the film", box)
        self.refraction_check.setToolTip(
            "Place each spot where refraction puts it (transmitted channel), from αi and the film's αc; with "
            "Analyze's refraction correction on, the q map already shows the true qz of that channel")
        self.critical_spin = _spin(box, 0.0, 2.0, 0.0, 3, " °")
        self.critical_spin.setSpecialValueText("Analyze's αc")
        self.critical_spin.setToolTip("Critical angle of the film; at 0 the one of Analyze's refraction correction")
        self.reflected_check = QCheckBox("Also the reflected channel (DWBA)", box)
        self.reflected_check.setToolTip("The spot of the wave reflected by the substrate first, k_z,f = q_z + k_z,i, "
                                        "drawn hollow")
        self.labels_check = QCheckBox("hkl labels", box)
        self.labels_check.setChecked(True)
        form.addRow(self.refraction_check)
        form.addRow("αc of the film", self.critical_spin)
        form.addRow(self.reflected_check)
        form.addRow(self.labels_check)
        return box

    # -- reflections --------------------------------------------------------------------

    def _reflections_tab(self, dialog) -> QWidget:
        tab = QWidget(dialog)
        layout = QVBoxLayout(tab)
        note = QLabel("Fibre spots (q∥, qz ≥ 0) or powder rings of every phase shown; m: reflections at the "
                      "position; |F|²: their mean; I: m·|F|² relative to the strongest of the phase.", tab)
        note.setWordWrap(True)
        note.setProperty("gimapRole", "muted")
        layout.addWidget(note)
        self.reflection_table = QTableWidget(0, len(REFLECTION_COLUMNS), tab)
        self.reflection_table.setObjectName("analyzeReflectionTable")
        self.reflection_table.setHorizontalHeaderLabels(REFLECTION_COLUMNS)
        self.reflection_table.setProperty(DATA_HEADERS, True)
        self.reflection_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.reflection_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.reflection_table.verticalHeader().hide()
        self.reflection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        enable_table_copy(self.reflection_table)
        layout.addWidget(self.reflection_table, 1)
        row = QHBoxLayout()
        self.use_reflection_button = QPushButton("Pole Figure of This Ring", tab)
        self.use_reflection_button.setToolTip(
            "Move the I(χ) ring onto the selected reflection and fit its pole figure, with χ0 of the reflection")
        row.addWidget(self.use_reflection_button)
        row.addStretch(1)
        layout.addLayout(row)
        return tab

    # -- pole figure --------------------------------------------------------------------

    def _pole_tab(self, dialog) -> QWidget:
        tab = QWidget(dialog)
        layout = QVBoxLayout(tab)
        self.ring_label = QLabel(tab)
        self.ring_label.setWordWrap(True)
        layout.addWidget(self.ring_label)
        form = QGridLayout()
        self.model_combo = QComboBox(tab)
        for key, text in MODEL_ITEMS:
            self.model_combo.addItem(text, key)
        self.direction_combo = QComboBox(tab)
        for key, text in DIRECTION_ITEMS:
            self.direction_combo.addItem(text, key)
        self.direction_spin = _spin(tab, 0.0, 90.0, 0.0, 1, " °")
        self.chi_max_spin = _spin(tab, 20.0, 90.0, 85.0, 1, " °")
        self.chi_max_spin.setToolTip("Bins above this χ are left out of the fit: next to the sample horizon "
                                     "absorption or a shadow lowers the intensity")
        self.background_check = QCheckBox("Automatic", tab)
        self.background_check.setChecked(True)
        self.background_check.setToolTip("The local linear background of a Gaussian fitted to I(q) at the ring")
        self.background_spin = _spin(tab, -1e9, 1e9, 0.0, 4)
        self.background_spin.setEnabled(False)
        form.addWidget(QLabel("Model", tab), 0, 0)
        form.addWidget(self.model_combo, 0, 1)
        form.addWidget(QLabel("Direction", tab), 0, 2)
        form.addWidget(self.direction_combo, 0, 3)
        form.addWidget(self.direction_spin, 0, 4)
        form.addWidget(QLabel("Fit up to χ", tab), 1, 0)
        form.addWidget(self.chi_max_spin, 1, 1)
        form.addWidget(QLabel("Background", tab), 1, 2)
        form.addWidget(self.background_check, 1, 3)
        form.addWidget(self.background_spin, 1, 4)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.fit_button = QPushButton("Fit Pole Figure", tab)
        self.fit_button.setProperty("gimapRole", "primary")
        self.export_button = QPushButton("Export…", tab)
        self.export_button.setToolTip("The pole figure as CSV and the indexing record (.json) beside it")
        buttons.addWidget(self.fit_button)
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.pole_plot = CurvePlot("", tab, log_y=False, log_x=None, sides=False)
        self.pole_plot.setObjectName("analyzePolePlot")
        self.pole_plot.set_labels("χ (°)", "N(χ) ∝ crystallites per degree")
        layout.addWidget(self.pole_plot, 1)
        self.result_label = QLabel(tab)
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.result_label)
        return tab


__all__ = ["DIRECTION_ITEMS", "IndexingView", "MODEL_ITEMS", "ORIENTATION_ITEMS", "REFLECTION_COLUMNS",
           "make_phases_button"]
