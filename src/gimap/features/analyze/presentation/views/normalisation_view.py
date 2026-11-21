"""The dark frame, the Normalisation section and the Lorentz factor of Mask & Corrections (static layout).

Behaviour: ``bindings/normalisation.py``. Physics: ``domain/normalisation.py`` and ``domain/intensity.py``."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
)

from src.gimap.app.presentation.components import AdvancedSection

EXPOSURE_ITEMS = (("Off", "off"), ("From the file header", "header"), ("Typed", "value"))
MONITOR_ITEMS = (("Off", "off"), ("Typed", "value"))
"""The monitor combo also lists each monitor value the header of the frame shown holds (``bindings``)."""
MIN_FIELD = 70
"""Least width (px) of a value field of the section: it fits the narrow step panel with its label beside it."""
NORMALISATION_TIP = (
    "Off by default. After the other corrections every frame is multiplied by A / (t · M · T · d): exposure time t, "
    "monitor M, transmission T, thickness d, absolute-scale factor A. σ is multiplied by the same factor; q and the "
    "shape of the curves do not change."
)


def _number_edit(parent, tip: str) -> QLineEdit:
    """A positive number in any notation (a monitor of 1.2e9, an absolute factor of 3.5e-6)."""
    edit = QLineEdit(parent)
    validator = QDoubleValidator(1e-300, 1e300, 12, edit)
    validator.setNotation(QDoubleValidator.ScientificNotation)
    edit.setValidator(validator)
    edit.setToolTip(tip)
    edit.setMinimumWidth(MIN_FIELD)
    return edit


def _spin(parent, low: float, high: float, value: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setMinimumWidth(MIN_FIELD)  # (its hint is as wide as the largest value: the step panel is narrow)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _combo(parent, items) -> QComboBox:
    combo = QComboBox(parent)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)  # a long header name is elided
    combo.setMinimumContentsLength(8)
    for text, key in items:
        combo.addItem(text, key)
    return combo


class NormalisationView:
    """``_dark_row`` (placed in Corrections before the background), ``_normalisation_section`` and
    ``_lorentz_row`` (in the intensity corrections)."""

    def _dark_row(self, section) -> None:
        dark = QHBoxLayout()
        self.dark_label = QLabel("No dark frame", section)
        self.dark_label.setObjectName("analyzeDarkLabel")
        self.dark_label.setProperty("gimapRole", "muted")
        self.dark_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dark_button = QToolButton(section)
        self.dark_button.setObjectName("analyzeDarkButton")
        self.dark_button.setText("Choose…")
        self.dark_button.setToolTip(
            "Dark frame (detector without beam: dark current, offset), subtracted from every frame before the "
            "background. σ then comes from the scatter of the pixels")
        self.dark_clear_button = QToolButton(section)
        self.dark_clear_button.setText("×")
        self.dark_clear_button.setAutoRaise(True)
        self.dark_clear_button.setToolTip("Stop subtracting the dark frame")
        dark.addWidget(self.dark_label, 1)
        dark.addWidget(self.dark_button)
        dark.addWidget(self.dark_clear_button)
        section.add_layout(dark)
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        self.dark_scale_spin = QDoubleSpinBox(section)
        self.dark_scale_spin.setObjectName("analyzeDarkScale")
        self.dark_scale_spin.setDecimals(4)
        self.dark_scale_spin.setRange(-1e6, 1e6)
        self.dark_scale_spin.setSingleStep(0.05)
        self.dark_scale_spin.setValue(1.0)
        self.dark_scale_spin.setKeyboardTracking(False)
        self.dark_scale_spin.setToolTip("frame − scale × dark (e.g. the exposure-time ratio)")
        self.dark_frame_spin = QSpinBox(section)
        self.dark_frame_spin.setRange(0, 100000)
        self.dark_frame_spin.setToolTip("Frame of a NeXus dark file")
        form.addRow("Dark scale", self.dark_scale_spin)
        form.addRow("Dark frame", self.dark_frame_spin)
        section.add_layout(form)

    def _normalisation_section(self, parent) -> AdvancedSection:
        section = AdvancedSection("Normalisation", NORMALISATION_TIP, parent, expanded=False)
        section.setObjectName("analyzeNormalisationSection")
        self.normalisation_section = section
        self.header_values_label = QLabel("", section)
        self.header_values_label.setObjectName("analyzeHeaderValues")
        self.header_values_label.setWordWrap(True)
        self.header_values_label.setProperty("gimapRole", "muted")
        self.header_values_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        section.add_widget(self.header_values_label)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        self.exposure_combo = _combo(section, EXPOSURE_ITEMS)
        self.exposure_combo.setObjectName("analyzeExposureSource")
        self.exposure_combo.setToolTip("Divide by the exposure time: the one the file header states, or a typed one")
        self.exposure_spin = _spin(section, 1e-6, 1e7, 1.0, 6, " s")
        self.exposure_spin.setObjectName("analyzeExposureValue")
        self.monitor_combo = _combo(section, MONITOR_ITEMS)
        self.monitor_combo.setObjectName("analyzeMonitorSource")
        self.monitor_combo.setToolTip(
            "Divide by a monitor (ion chamber, diode, flux or ring current): a value of the file header, or a typed one")
        self.monitor_edit = _number_edit(section, "The monitor value (any positive number, e.g. 1.2e9)")
        self.monitor_edit.setObjectName("analyzeMonitorValue")
        self.monitor_edit.setText("1")
        self.transmission_check = QCheckBox("Transmission T", section)
        self.transmission_spin = _spin(section, 1e-6, 1.0, 1.0, 6)
        self.transmission_spin.setToolTip("Transmission of the sample, above 0 and at most 1")
        self.thickness_check = QCheckBox("Thickness d", section)
        self.thickness_spin = _spin(section, 1e-6, 1e5, 1.0, 6, " mm")
        self.absolute_check = QCheckBox("Absolute scale A", section)
        self.absolute_edit = _number_edit(
            section, "Absolute-scale factor, e.g. from a glassy-carbon or water measurement in the same set-up")
        self.absolute_edit.setText("1")
        rows = (  # a source (header or typed) with its typed value below it; a check with its value beside it
            (QLabel("Exposure t", section), self.exposure_combo), (None, self.exposure_spin),
            (QLabel("Monitor M", section), self.monitor_combo), (None, self.monitor_edit),
            (self.transmission_check, self.transmission_spin),
            (self.thickness_check, self.thickness_spin),
            (self.absolute_check, self.absolute_edit),
        )
        for row, (label, value) in enumerate(rows):
            if label is not None:
                grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
        grid.setColumnStretch(1, 1)
        section.add_layout(grid)
        self.normalisation_status = QLabel("", section)
        self.normalisation_status.setObjectName("analyzeNormalisationStatus")
        self.normalisation_status.setWordWrap(True)
        self.normalisation_status.setProperty("gimapRole", "muted")
        self.normalisation_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        section.add_widget(self.normalisation_status)
        return section

    def _lorentz_row(self, section) -> None:
        self.lorentz_check = QCheckBox("Lorentz factor (fibre texture)", section)
        self.lorentz_check.setObjectName("analyzeLorentzCheck")
        self.lorentz_check.setToolTip(
            "Crystallites oriented at random about the surface normal: divide by L = 1/(cos αi cos αf |sin 2θf|) "
            "(GIXSGUI). I(χ) then counts the crystallites per degree of χ; the Hermans factor of the automatic "
            "analysis is then not weighted by sin χ again. Only for crystalline peaks: it also scales diffuse "
            "scattering and the I(q) of a ring")
        section.add_widget(self.lorentz_check)


__all__ = ["EXPOSURE_ITEMS", "MONITOR_ITEMS", "NORMALISATION_TIP", "NormalisationView"]
