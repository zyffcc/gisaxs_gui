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
from .workflow_v5_units import UNIT_LABEL, UNIT_TIP

Q_UNITS = (("nm⁻¹", "nm^-1"), ("Å⁻¹", "A^-1"))
"""(shown, option value) of the unit of q in text files that state none in their header (``workflow_v5_units``)."""
Q_SIDES = (("Each half separately", "both"), ("q > 0 half", "positive"), ("q < 0 half", "negative"))
"""(shown, option value) of the halves of q fitted."""
COMPONENTS = ("Auto / unused", "Sphere", "Random cylinder", "Vertical cylinder")
"""The composition's choices; their position is the option value (0: none)."""
FIT_METHODS = (
    ("General V5 (experimental)", "model"),
    ("Single RC specialist (experimental)", "stable"),
    ("Physical fit (numerical)", "experimental"),
    ("fit1d (network + physical fit)", "fit1d"),
)
FIT1D_POLICIES = (
    ("Fast (one search)", "fast"),
    ("Auto (more starts when needed)", "auto"),
    ("Robust (always many starts)", "robust"),
)
FIT1D_MODES = (("GISAXS (Yoneda cut)", "gisaxs"), ("SAXS (dilute, no structure factor)", "saxs"))
FIT1D_MODELS = (("General network", "general"), ("Au nanoparticles on Si (adapted)", "au_np_si"))
"""(shown, option value) of the fit1d kit's search, curve type and proposal network."""
FIT1D_TIP = (
    "fit1d is an external kit (folder utils/gisaxs1d_gui_kit, or GISAXS1D_KIT_HOME). It needs the Complete "
    "composition (Sphere / Vertical cylinder; it has no random cylinder): one curve does not identify the "
    "composition. Its physics differs from the manual model: number-weighted sizes, Hosemann paracrystal, "
    "spheres at |q| = sqrt(q∥² + qz²). The curve shown is fit1d's own."
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
        self.unit.setToolTip(UNIT_TIP)
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
            "that fallback does not make the specialist a general model. Scores are not probabilities. "
            "fit1d is an external kit with its own physics and needs the Complete composition "
            "(Sphere / Vertical cylinder)."
        )
        left.addRow("Fit method", self.method)
        left.addRow(UNIT_LABEL, self.unit)
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
        self._build_fit1d(right)
        self.method.currentIndexChanged.connect(self._update_method_controls)
        self._update_method_controls()
        save = QPushButton("Save Settings")
        save.clicked.connect(self.save_settings)
        right.addRow(save)
        outer.addLayout(left, 1)
        outer.addLayout(right, 1)
        return box

    def _build_fit1d(self, form: QFormLayout) -> None:
        """The settings only fit1d uses; enabled when it is the method."""
        self.fit1d_policy = _choices(FIT1D_POLICIES, self._options["fit1d_policy"])
        self.fit1d_mode = _choices(FIT1D_MODES, self._options["fit1d_mode"])
        self.fit1d_model = _choices(FIT1D_MODELS, self._options["fit1d_model"])
        if self.fit1d_model.findData(self._options["fit1d_model"]) < 0:  # a network named in a recipe
            self.fit1d_model.addItem(self._options["fit1d_model"], self._options["fit1d_model"])
            self.fit1d_model.setCurrentIndex(self.fit1d_model.count() - 1)
        self.fit1d_use_qz = QCheckBox("Cut qz (nm⁻¹)")
        self.fit1d_use_qz.setChecked(self._options["fit1d_qz"] is not None)
        self.fit1d_qz = _spin(0, 10, self._options["fit1d_qz"] or 0.0, 6)
        self.fit1d_sigma_floor = _spin(0, 50, 100 * self._options["fit1d_sigma_floor"], 4)
        for widget in (self.fit1d_policy, self.fit1d_mode, self.fit1d_model):
            widget.setToolTip(FIT1D_TIP)
        tip = ("fit1d: the out-of-plane q of the horizontal cut; spheres are evaluated at "
               "|q| = sqrt(q∥² + qz²). Unticked: qz = 0, and the result says so.")
        self.fit1d_use_qz.setToolTip(tip)
        self.fit1d_qz.setToolTip(tip)
        self.fit1d_sigma_floor.setToolTip(
            "fit1d: a relative systematic error added to σ in quadrature. 2–5 % suits bright measured curves.")
        form.addRow("fit1d search", self.fit1d_policy)
        form.addRow("fit1d curve type", self.fit1d_mode)
        form.addRow("fit1d network", self.fit1d_model)
        form.addRow(self.fit1d_use_qz, self.fit1d_qz)
        form.addRow("fit1d systematic error (%)", self.fit1d_sigma_floor)
        self._fit1d_widgets = (self.fit1d_policy, self.fit1d_mode, self.fit1d_model, self.fit1d_use_qz,
                               self.fit1d_qz, self.fit1d_sigma_floor)

    def _update_method_controls(self) -> None:
        legacy = self.method.currentData() == "model"
        fit1d = self.method.currentData() == "fit1d"
        self.amplitude_calibration.setEnabled(self.method.currentData() == "stable")
        for widget in (self.numerical, self.normalizer, self.search, self.conditions):
            widget.setEnabled(legacy)
        for widget in self._fit1d_widgets:
            widget.setEnabled(fit1d)
        for widget in (self.fix_sigma, self.sigma_res):  # fit1d fits the width of its low-q term itself
            widget.setEnabled(not fit1d)

    def options(self) -> dict:
        """The options as set in the panel (the values, never the text shown), validated."""
        return validate_options(
            {
                **self._options,
                "method": self.method.currentData(),
                "components": [b.currentIndex() for b in self.component_boxes if b.currentIndex()],
                "q_unit": self.unit.currentData(),
                "side": self.side.currentData(),
                "sigma_res": self.sigma_res.value()
                if self.fix_sigma.isChecked() and self.method.currentData() != "fit1d" else None,
                "nu_res": self.nu_res.value() if self.fix_nu.isChecked() else None,
                "relative_noise": self.relative_noise.value(),
                "absolute_noise": self.noise_floor.value(),
                "normalizer": self.normalizer.value() or None,
                "numerical": self.numerical.isChecked(),
                "amplitude_calibration": self.amplitude_calibration.isChecked(),
                "search_combinations": self.search.value(),
                "condition_combinations": self.conditions.value(),
                "fit1d_policy": self.fit1d_policy.currentData(),
                "fit1d_mode": self.fit1d_mode.currentData(),
                "fit1d_model": self.fit1d_model.currentData(),
                "fit1d_qz": self.fit1d_qz.value() if self.fit1d_use_qz.isChecked() else None,
                "fit1d_sigma_floor": self.fit1d_sigma_floor.value() / 100,
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
