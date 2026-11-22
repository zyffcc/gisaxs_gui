"""The prediction settings of 1D Predict (``WorkflowV5Dialog``): the panel and the options it gives.

The options are the values ``application/workflow_v5.validate_options`` checks, in any interface language:
the q unit and the halves are read from the items' data (the text shown is a unit or a translated phrase),
the composition from the items' positions. Saved settings and in-situ recipes so hold exactly the values they held
before the panel was translated. The window has one method, Predict parameters.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel, QPushButton,
)

from src.gimap.app.presentation.i18n import tr, trf

from ..application.workflow_v5 import validate_options
from ..application.substrate_level import Substrate
from .predict_substrate import EDIT_CUT, ON_SUBSTRATE, PredictSubstrateDialog
from .workflow_v5_units import UNIT_LABEL, UNIT_TIP

Q_UNITS = (("Å⁻¹", "A^-1"), ("nm⁻¹", "nm^-1"))
"""(shown, option value) of the unit of q in text files that state none in their header (``workflow_v5_units``)."""
Q_SIDES = (("Each half separately", "both"), ("q > 0 half", "positive"), ("q < 0 half", "negative"))
"""(shown, option value) of the halves of q fitted."""
COMPONENTS = ("Auto / unused", "Sphere", "Random cylinder", "Vertical cylinder")
"""The composition's choices; their position is the option value (0: none)."""
COMPOSITION_TIP = ("Predict parameters: GIMaP's proposal network and its posterior, each solution refined by least "
                   "squares. Complete composition: 1–3 components, each with a paracrystal; Auto: the network chooses.")
COUNTING_TIP = ("Only for Use current cut on a curve Analyze wrote with the pixel counts of its detector columns: "
                "added in quadrature to the counting σ. A curve without σ gets a relative σ estimated from the scatter "
                "of ln I about its trend, as on the Fit page.")
QZ_TIP = ("The out-of-plane q of the horizontal cut: spheres are evaluated at the Born term of that q_z. "
          "Unticked: q_z = 0 (the effective interface).")


class TrimmedSpinBox(QDoubleSpinBox):
    """Shows its value without trailing zeros (“0.01”, not “0.01000000”). Display only: the value keeps all
    its decimals, and the text shown reads back as the same value."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        text = super().textFromValue(value)
        point = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(point) if point in text else text


def _spin(lo, hi, value, decimals) -> QDoubleSpinBox:
    spin = TrimmedSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setValue(value)
    return spin


def _choices(items, chosen) -> QComboBox:
    """A combo of ``(shown, value)`` items with ``chosen`` (a value) selected."""
    combo = QComboBox()
    for text, value in items:
        combo.addItem(text, value)
    combo.setCurrentIndex(max(0, combo.findData(chosen)))
    return combo


class WorkflowV5SettingsMixin:
    """Needs ``self._options`` (validated) and ``self.status`` (for Save Settings) on the dialog."""

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox("Prediction settings — saved for single curves and future batch / in-situ runs")
        outer = QHBoxLayout(box)
        left, right = QFormLayout(), QFormLayout()
        self.component_boxes = []
        row = QHBoxLayout()
        for index in range(3):
            combo = QComboBox()
            combo.setToolTip(COMPOSITION_TIP)
            combo.addItems(list(COMPONENTS))
            combo.setCurrentIndex(
                self._options["components"][index] if index < len(self._options["components"]) else 0)
            row.addWidget(combo)
            self.component_boxes.append(combo)
        left.addRow("Complete composition", row)
        self.unit = _choices(Q_UNITS, self._options["q_unit"])
        self.unit.setToolTip(UNIT_TIP)
        self.side = _choices(Q_SIDES, self._options["side"])
        left.addRow(UNIT_LABEL, self.unit)
        left.addRow("q sides", self.side)
        self.predict_cut = self._options.get("predict_cut", {})
        self.substrate_button = QPushButton(EDIT_CUT)
        self.substrate_button.clicked.connect(self._edit_substrate)
        left.addRow(self.substrate_button)
        self.substrate_note = QLabel()
        self.substrate_note.setWordWrap(True)
        left.addRow(self.substrate_note)
        self.use_qz = QCheckBox("Cut qz (nm⁻¹)")
        self.use_qz.setChecked(self._options["cut_qz"] is not None)
        self.cut_qz = _spin(0, 10, self._options["cut_qz"] or 0.0, 6)
        for widget in (self.use_qz, self.cut_qz):
            widget.setToolTip(QZ_TIP)
        right.addRow(self.use_qz, self.cut_qz)
        self.reflected_beam = QCheckBox("Reflected beam present in the data")
        self.reflected_beam.setChecked(self._options["reflected_beam"])
        right.addRow(self.reflected_beam)
        self.measured_qz = _spin(0, 10, self._options["measured_qz"] or 0., 5)
        self.measured_qz.setSpecialValueText("Unknown")
        right.addRow("Measured q_z for level-1 flags (nm⁻¹)", self.measured_qz)
        self.reflected_beam.toggled.connect(self._refresh_substrate)
        self.use_qz.toggled.connect(self._refresh_substrate)
        self._refresh_substrate()
        self.relative_noise = _spin(0, 10, self._options["relative_noise"], 4)
        self.noise_floor = _spin(0, 1e12, self._options["absolute_noise"], 6)
        self.noise_floor.setSpecialValueText("None")
        for widget in (self.relative_noise, self.noise_floor):
            widget.setToolTip(COUNTING_TIP)
        right.addRow("Relative σ added to counting σ", self.relative_noise)
        right.addRow("Absolute σ added to counting σ", self.noise_floor)
        save = QPushButton("Save Settings")
        save.clicked.connect(self.save_settings)
        right.addRow(save)
        outer.addLayout(left, 1)
        outer.addLayout(right, 1)
        return box

    def _edit_substrate(self):
        dialog = PredictSubstrateDialog(self.predict_cut, self, batch=True)
        if dialog.exec_():
            self.predict_cut = dialog.cut()
            self._refresh_substrate()
        dialog.deleteLater()

    def _refresh_substrate(self):
        substrate = Substrate.from_dict(self.predict_cut.get("substrate"))
        self.substrate_note.setVisible(substrate is not None)
        self._say(self.substrate_note, lambda: trf(ON_SUBSTRATE, substrate=tr(substrate.label)) if substrate else "")
        for field in (self.use_qz, self.cut_qz, self.reflected_beam):
            field.setEnabled(substrate is None)
        self.measured_qz.setEnabled(substrate is None and self.reflected_beam.isChecked() and not self.use_qz.isChecked())

    def options(self) -> dict:
        """The options as set in the panel (the values, never the text shown), validated."""
        return validate_options(
            {
                **self._options,
                "predict_cut": self.predict_cut,
                "method": "posterior",
                "components": [b.currentIndex() for b in self.component_boxes if b.currentIndex()],
                "q_unit": self.unit.currentData(),
                "side": self.side.currentData(),
                "relative_noise": self.relative_noise.value(),
                "absolute_noise": self.noise_floor.value(),
                "reflected_beam": self.reflected_beam.isChecked(),
                "measured_qz": (self.measured_qz.value() or None) if self.measured_qz.isEnabled() else None,
                "cut_qz": self.cut_qz.value() if self.use_qz.isChecked() else None,
            }
        )

    def save_settings(self) -> None:
        try:
            self._options = self.options()
        except ValueError as exc:
            self._say(self.status, str(exc))
            return
        self._say(self.status, lambda: tr("Settings saved. Existing in-situ recipes keep their captured settings."))
        self.settings_changed.emit(self._options)


__all__ = ["COMPONENTS", "COMPOSITION_TIP", "QZ_TIP", "Q_SIDES", "Q_UNITS", "TrimmedSpinBox", "WorkflowV5SettingsMixin"]
