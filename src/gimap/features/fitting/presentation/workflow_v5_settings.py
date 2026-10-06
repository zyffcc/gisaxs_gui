"""The prediction settings of 1D Predict (``WorkflowV5Dialog``): the panel and the options it gives.

The options are the values ``application/workflow_v5.validate_options`` checks, in any interface language:
the q unit and the halves are read from the items' data (the text shown is a unit or a translated phrase),
the composition from the items' positions and the method from its data. Saved settings and in-situ recipes
so hold exactly the values they held before the panel was translated.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
)

from src.gimap.app.presentation.i18n import tr

from ..application.workflow_v5 import validate_options

Q_UNITS = (("nm⁻¹", "nm^-1"), ("Å⁻¹", "A^-1"))
"""(shown, option value) of the unit of q in text files."""
Q_SIDES = (("Each half separately", "both"), ("q > 0 half", "positive"), ("q < 0 half", "negative"))
"""(shown, option value) of the halves of q fitted."""
COMPONENTS = ("Auto / unused", "Sphere", "Random cylinder", "Vertical cylinder")
"""The composition's choices; their position is the option value (0: none)."""
FIT_METHODS = (
    ("General V5 (experimental)", "model"),
    ("Single RC specialist (experimental)", "stable"),
    ("Physical fit (numerical)", "experimental"),
)


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
        for index in range(4):
            combo = QComboBox()
            combo.addItems(list(COMPONENTS))
            combo.setCurrentIndex(
                self._options["components"][index] if index < len(self._options["components"]) else 0)
            row.addWidget(combo)
            self.component_boxes.append(combo)
        left.addRow("Complete composition", row)
        self.unit = _choices(Q_UNITS, self._options["q_unit"])
        self.side = _choices(Q_SIDES, self._options["side"])
        self.numerical = QCheckBox("General V5: improve fit with four numerical steps")
        self.numerical.setChecked(self._options["numerical"])
        self.amplitude_calibration = QCheckBox("Calibrate intensity amplitudes")
        self.amplitude_calibration.setChecked(self._options.get("amplitude_calibration", True))
        self.amplitude_calibration.setToolTip(
            "The single-RC specialist can adjust particle, background and resolution amplitudes while keeping "
            "the neural shape parameters fixed. Broader fitting may still run when curve agreement is poor."
        )
        self.method = _choices(FIT_METHODS, self._options["method"])
        self.method.setToolTip(
            "General V5 proposes multiple compositions but remains experimental. "
            "The specialist requires Complete composition = one Random cylinder and eligible native CBF counts. "
            "Other inputs, fixed resolution and poor curve agreement use numerical fallback; "
            "that fallback does not make the specialist a general model. Scores are not probabilities."
        )
        left.addRow("Fit method", self.method)
        left.addRow("Text-file q unit", self.unit)
        left.addRow("q sides", self.side)
        left.addRow(self.amplitude_calibration)
        left.addRow(self.numerical)
        self.fix_sigma = QCheckBox("Fix σ res (nm⁻¹)")
        self.fix_sigma.setChecked(self._options["sigma_res"] is not None)
        self.sigma_res = _spin(0.001, 0.1, self._options["sigma_res"] or 0.01, 8)
        self.fix_nu = QCheckBox("Fix ν res")
        self.fix_nu.setChecked(self._options["nu_res"] is not None)
        self.nu_res = _spin(1, 20, self._options["nu_res"] or 7, 6)
        self.sigma_res.setToolTip("General V5: 0.007–0.013 nm⁻¹. RC specialist / physical fit: 0.001–0.1 nm⁻¹; "
                                  "fixed resolution uses numerical fallback.")
        self.nu_res.setToolTip("General V5: 5–10. RC specialist / physical fit: 1–20; fixed resolution uses "
                               "numerical fallback.")
        right.addRow(self.fix_sigma, self.sigma_res)
        right.addRow(self.fix_nu, self.nu_res)
        self.relative_noise = _spin(0, 10, self._options["relative_noise"], 4)
        self.noise_floor = _spin(0, 1e12, self._options["absolute_noise"], 6)
        self.noise_floor.setSpecialValueText("Auto: 0.1% peak")
        self.normalizer = _spin(0, 1e20, self._options["normalizer"] or 0, 6)
        self.normalizer.setSpecialValueText("Auto: measured max")
        right.addRow("Relative σ (if missing)", self.relative_noise)
        right.addRow("Absolute σ floor", self.noise_floor)
        right.addRow("Intensity normalizer", self.normalizer)
        self.search = QSpinBox()
        self.search.setRange(1, 34)
        self.search.setValue(self._options["search_combinations"])
        self.conditions = QSpinBox()
        self.conditions.setRange(1, 34)
        self.conditions.setValue(self._options["condition_combinations"])
        left.addRow("Discover combinations", self.search)
        left.addRow("Condition best combinations", self.conditions)
        self.method.currentIndexChanged.connect(self._update_method_controls)
        self._update_method_controls()
        save = QPushButton("Save Settings")
        save.clicked.connect(self.save_settings)
        right.addRow(save)
        outer.addLayout(left, 1)
        outer.addLayout(right, 1)
        return box

    def _update_method_controls(self) -> None:
        legacy = self.method.currentData() == "model"
        self.amplitude_calibration.setEnabled(self.method.currentData() == "stable")
        for widget in (self.numerical, self.normalizer, self.search, self.conditions):
            widget.setEnabled(legacy)

    def options(self) -> dict:
        """The options as set in the panel (the values, never the text shown), validated."""
        return validate_options(
            {
                **self._options,
                "method": self.method.currentData(),
                "components": [b.currentIndex() for b in self.component_boxes if b.currentIndex()],
                "q_unit": self.unit.currentData(),
                "side": self.side.currentData(),
                "sigma_res": self.sigma_res.value() if self.fix_sigma.isChecked() else None,
                "nu_res": self.nu_res.value() if self.fix_nu.isChecked() else None,
                "relative_noise": self.relative_noise.value(),
                "absolute_noise": self.noise_floor.value(),
                "normalizer": self.normalizer.value() or None,
                "numerical": self.numerical.isChecked(),
                "amplitude_calibration": self.amplitude_calibration.isChecked(),
                "search_combinations": self.search.value(),
                "condition_combinations": self.conditions.value(),
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


__all__ = ["COMPONENTS", "FIT_METHODS", "Q_SIDES", "Q_UNITS", "TrimmedSpinBox", "WorkflowV5SettingsMixin"]
