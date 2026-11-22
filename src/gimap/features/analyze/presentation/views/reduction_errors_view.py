"""Controls for detector response and curve uncertainties in the Mask step."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
    QToolButton,
)

from src.gimap.app.presentation.components import AdvancedSection
from src.gimap.app.presentation.theme import set_role
from .command_bar import ElidedLabel


class ReductionErrorsView:
    def _detector_errors_section(self, parent):
        section = AdvancedSection("Detector and curve errors", parent=parent)
        self.detector_errors_section = section
        self.use_profile_response_check = QCheckBox("Use instrument profile settings", section)
        section.add_widget(self.use_profile_response_check)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.gain_spin = QDoubleSpinBox(section)
        self.gain_spin.setDecimals(6)
        self.gain_spin.setRange(0.000001, 1e9)
        self.gain_spin.setValue(1)
        self.offset_spin = QDoubleSpinBox(section)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setRange(-1e9, 1e9)
        for spin in (self.gain_spin, self.offset_spin):
            spin.setKeyboardTracking(False)
        form.addRow("Counts per ADU (gain)", self.gain_spin)
        form.addRow("Offset (ADU)", self.offset_spin)
        self.error_model_combo = QComboBox(section)
        for label, key in (("Automatic", "auto"), ("Poisson", "poisson"),
                           ("Scatter of the pixels", "scatter"), ("Given σ frame", "sigma")):
            self.error_model_combo.addItem(label, key)
        form.addRow("Error model", self.error_model_combo)
        section.add_layout(form)

        sigma = QHBoxLayout()
        self.sigma_label = ElidedLabel(parent=section, padding=0)
        self.sigma_label.setTextFormat(Qt.PlainText)
        self.sigma_button = QToolButton(section)
        self.sigma_button.setText("Choose…")
        self.sigma_button.setToolTip("Choose the per-pixel standard deviations of the sample frame.")
        sigma.addWidget(self.sigma_label, 1)
        sigma.addWidget(self.sigma_button)
        section.add_layout(sigma)
        self.save_response_button = QPushButton("Save detector settings to profile", section)
        section.add_widget(self.save_response_button)

        self.pool_scatter_check = QCheckBox("Pool scatter below 10 pixels", section)
        section.add_widget(self.pool_scatter_check)
        self.poisson_variance_combo = QComboBox(section)
        self.poisson_variance_combo.addItem("Observed intensity", "observed")
        self.poisson_variance_combo.addItem("Smoothed intensity", "smoothed")
        model = QFormLayout()
        model.setRowWrapPolicy(QFormLayout.WrapLongRows)
        model.addRow("Poisson variance", self.poisson_variance_combo)
        section.add_layout(model)
        self.error_model_status = QLabel(section)
        self.error_model_status.setWordWrap(True)
        set_role(self.error_model_status, "muted")
        section.add_widget(self.error_model_status)
