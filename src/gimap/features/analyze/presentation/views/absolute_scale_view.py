"""Controls and plots for calibration against a measured secondary standard."""

from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QComboBox, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from src.gimap.app.presentation.components.curve_plot import CurvePlot


def _number(parent, text="", *, minimum=0.0):
    field = QLineEdit(text, parent)
    validator = QDoubleValidator(minimum, 1e300, 15, field)
    validator.setLocale(QLocale.c())
    field.setValidator(validator)
    return field


def _units(parent):
    combo = QComboBox(parent)
    combo.addItems(["Å⁻¹", "nm⁻¹"])
    return combo


def _file_field(parent):
    widget = QWidget(parent)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(widget)
    button = QPushButton("Browse…", widget)
    button.setAutoDefault(False)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return widget, edit, button


def _form(parent):
    form = QFormLayout(parent)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.DontWrapRows)
    return form


class AbsoluteScaleView:
    def setup_scale_view(self):
        self.setWindowTitle("Absolute Scale from a Standard…")
        self.resize(1024, 702)
        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        layout.addLayout(body, 1)

        scroll = QScrollArea(self)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(342)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.input_panel = QWidget(scroll)
        left = QVBoxLayout(self.input_panel)
        left.setContentsMargins(0, 0, 5, 0)
        left.addWidget(self._standard_group())
        left.addWidget(self._reference_group())
        left.addWidget(self._profile_group())
        left.addStretch(1)
        scroll.setWidget(self.input_panel)
        body.addWidget(scroll, 2)

        right = QVBoxLayout()
        body.addLayout(right, 3)
        self.overlay = CurvePlot("Standard and reference", self, log_y=False, sides=False)
        self.overlay.set_labels("q (Å⁻¹)", "dΣ/dΩ (cm⁻¹)")
        self.overlay.set_empty_text("Fit the standard to compare it with the reference.")
        self.residual = CurvePlot("Residual (scaled standard − reference)", self, log_y=False, sides=False)
        self.residual.log_check.hide()
        self.residual.set_labels("q (Å⁻¹)", "ΔI (cm⁻¹)")
        self.residual.set_empty_text("", bare=True)
        right.addWidget(self.overlay, 3)
        right.addWidget(self.residual, 2)
        self.fit_summary = QLabel(self)
        self.fit_summary.setWordWrap(True)
        self.fit_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self.fit_summary)

        footer = QHBoxLayout()
        self.feedback = QLabel(self)
        self.feedback.setWordWrap(True)
        self.feedback.setTextFormat(Qt.PlainText)
        self.feedback.setTextInteractionFlags(Qt.TextSelectableByMouse)
        footer.addWidget(self.feedback, 1)
        self.buttons = QDialogButtonBox(self)
        self.fit_button = self.buttons.addButton("Fit Scale", QDialogButtonBox.ActionRole)
        self.apply_button = self.buttons.addButton("Apply Scale", QDialogButtonBox.ApplyRole)
        self.close_button = self.buttons.addButton(QDialogButtonBox.Close)
        self.apply_button.setProperty("gimapRole", "primary")
        for button in self.buttons.buttons():
            button.setAutoDefault(False)
        footer.addWidget(self.buttons)
        layout.addLayout(footer)

    def _standard_group(self):
        group = QGroupBox("Standard measurement", self)
        form = _form(group)
        self.source_kind = QComboBox(group)
        self.source_kind.addItem("Detector frame", "frame")
        self.source_kind.addItem("Reduced curve", "curve")
        form.addRow("Input", self.source_kind)
        picker, self.source_path, self.source_browse = _file_field(group)
        form.addRow("Measured file", picker)
        self.source_options = QStackedWidget(group)
        frame_options = QWidget(group)
        frame_form = _form(frame_options)
        frame_form.setContentsMargins(0, 0, 0, 0)
        self.frame_number = QSpinBox(frame_options)
        self.frame_number.setRange(1, 2**31 - 1)
        frame_form.addRow("Frame number", self.frame_number)
        self.source_options.addWidget(frame_options)
        curve_options = QWidget(group)
        curve_form = _form(curve_options)
        curve_form.setContentsMargins(0, 0, 0, 0)
        self.source_unit = _units(curve_options)
        curve_form.addRow("Measured q unit", self.source_unit)
        self.source_options.addWidget(curve_options)
        form.addRow(self.source_options)

        terms = QGridLayout()
        self.exposure = _number(group, "1")
        self.monitor = _number(group, "1")
        self.transmission = _number(group, "1")
        self.std_thickness = _number(group)
        self.std_thickness.setPlaceholderText("Required")
        for index, (label, edit) in enumerate((
            ("Exposure t (s)", self.exposure), ("Monitor M", self.monitor),
            ("Transmission T", self.transmission), ("Standard d (mm)", self.std_thickness),
        )):
            row, col = divmod(index, 2)
            title = QLabel(label, group)
            title.setBuddy(edit)
            terms.addWidget(title, 2 * row, col)
            terms.addWidget(edit, 2 * row + 1, col)
        form.addRow(terms)
        return group

    def _reference_group(self):
        group = QGroupBox("Reference and fit range", self)
        form = _form(group)
        self.reference_kind = QComboBox(group)
        self.reference_kind.addItem("Water", "water")
        self.reference_kind.addItem("Reference curve file", "curve")
        form.addRow("Reference", self.reference_kind)
        self.temperature = QDoubleSpinBox(group)
        self.temperature.setRange(0, 100)
        self.temperature.setDecimals(2)
        self.temperature.setValue(20)
        self.temperature_label = QLabel("Water temperature (°C)", group)
        form.addRow(self.temperature_label, self.temperature)
        self.water_level = QLabel(group)
        self.water_level.setProperty("gimapRole", "muted")
        form.addRow(self.water_level)
        self.ref_picker, self.reference_path, self.reference_browse = _file_field(group)
        self.reference_path.setPlaceholderText("q, dΣ/dΩ, σ")
        self.reference_path.setToolTip("Reference columns are q, dΣ/dΩ (cm⁻¹) and pointwise 1σ (cm⁻¹).")
        self.ref_label = QLabel("Reference file", group)
        form.addRow(self.ref_label, self.ref_picker)
        self.reference_unit = _units(group)
        self.ref_unit_label = QLabel("Reference q unit", group)
        form.addRow(self.ref_unit_label, self.reference_unit)
        self.ref_uncertainty = _number(group, "0")
        self.ref_uncertainty.setToolTip("Relative standard uncertainty shared by all reference points (1σ, %).")
        form.addRow("Shared scale σ (%)", self.ref_uncertainty)
        bounds = QHBoxLayout()
        self.q_min = _number(group)
        self.q_max = _number(group)
        self.q_min.setPlaceholderText("Min")
        self.q_max.setPlaceholderText("Max")
        for edit in (self.q_min, self.q_max):
            edit.setToolTip("Leave both limits blank to fit the full shared q range.")
        bounds.addWidget(self.q_min)
        bounds.addWidget(QLabel("–", group))
        bounds.addWidget(self.q_max)
        form.addRow("Fit q (Å⁻¹)", bounds)
        return group

    def _profile_group(self):
        group = QGroupBox("Apply to sample", self)
        form = _form(group)
        self.profile_label = QLineEdit(group)
        self.profile_label.setReadOnly(True)
        form.addRow("Instrument profile", self.profile_label)
        self.sample_thickness = _number(group)
        self.sample_thickness.setPlaceholderText("Required to apply")
        form.addRow("Sample d (mm)", self.sample_thickness)
        note = QLabel("Set sample exposure, monitor and transmission in Mask & Corrections › Normalisation.", group)
        note.setWordWrap(True)
        note.setProperty("gimapRole", "muted")
        form.addRow(note)
        self.saved_label = QLabel(group)
        self.saved_label.setWordWrap(True)
        self.saved_label.setTextFormat(Qt.PlainText)
        form.addRow(self.saved_label)
        self.saved_button = QPushButton("Use Saved Scale", group)
        self.saved_button.setAutoDefault(False)
        form.addRow(self.saved_button)
        return group
