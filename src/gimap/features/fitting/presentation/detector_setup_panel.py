"""Inline detector and beam setup for the Fitting workspace."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from src.gimap.app.presentation import (
    ParameterCommitCoordinator,
    ParameterUpdatePolicy,
    install_safe_wheel_behavior,
)
from src.gimap.app.presentation.layout_primitives import normalize_button, normalize_input
from src.gimap.features.fitting.application import (
    DetectorSettings,
    energy_to_wavelength,
    wavelength_to_energy,
)


class DetectorSetupPanel(QWidget):
    """Edit detector settings with immediate or debounced persistence."""

    settings_applied = pyqtSignal(dict)
    apply_failed = pyqtSignal(str)

    def __init__(self, view_model, apply_button, parent=None) -> None:
        super().__init__(parent)
        self.view_model = view_model
        self.apply_button = apply_button
        self._syncing_energy_pair = False
        self.setObjectName("fittingDetectorSetupPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._build_ui()
        self._load()
        self._connect()
        install_safe_wheel_behavior(self)

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        self.distance_spinbox = self._spinbox(
            "fittingDetectorDistanceSpinBox", 100.0, 10000.0, 2000.0, 1, 0.1
        )
        self.angle_spinbox = self._spinbox(
            "fittingDetectorAngleSpinBox", 0.01, 10.0, 0.4, 3, 0.01
        )
        self.energy_spinbox = self._spinbox(
            "fittingDetectorEnergySpinBox", 0.1, 200.0, 82.6561, 4, 0.01
        )
        self.wavelength_spinbox = self._spinbox(
            "fittingDetectorWavelengthSpinBox", 0.001, 1.0, 0.015, 4, 0.001
        )
        self.beam_center_x_spinbox = self._spinbox(
            "fittingDetectorBeamCenterXSpinBox", 0.0, 20000.0, 50.0, 2, 0.1
        )
        self.beam_center_y_spinbox = self._spinbox(
            "fittingDetectorBeamCenterYSpinBox", 0.0, 20000.0, 50.0, 2, 0.1
        )
        self.pixel_size_x_spinbox = self._spinbox(
            "fittingDetectorPixelSizeXSpinBox", 1.0, 1000.0, 172.0, 1, 0.1
        )
        self.pixel_size_y_spinbox = self._spinbox(
            "fittingDetectorPixelSizeYSpinBox", 1.0, 1000.0, 172.0, 1, 0.1
        )
        self.show_q_axis_checkbox = QCheckBox("Show detector axes in q", self)
        self.show_q_axis_checkbox.setObjectName("fittingDetectorShowQAxisCheckBox")
        self.horizontal_q_combo = QComboBox(self)
        self.horizontal_q_combo.setObjectName("fittingDetectorHorizontalQComboBox")
        self.horizontal_q_combo.addItem("qy · in-plane component", "qy")
        self.horizontal_q_combo.addItem("qr · signed radial", "qr")
        self.horizontal_q_combo.setToolTip(
            "Choose the horizontal detector coordinate; the vertical axis is always qz."
        )
        normalize_input(self.horizontal_q_combo)

        rows = (
            ("Distance (mm)", self.distance_spinbox, "Grazing angle (°)", self.angle_spinbox),
            ("Energy (keV)", self.energy_spinbox, "Wavelength (nm)", self.wavelength_spinbox),
            (
                "Beam center X (px)",
                self.beam_center_x_spinbox,
                "Beam center Y (px)",
                self.beam_center_y_spinbox,
            ),
            (
                "Pixel size X (µm)",
                self.pixel_size_x_spinbox,
                "Pixel size Y (µm)",
                self.pixel_size_y_spinbox,
            ),
        )
        for row, (left_text, left_editor, right_text, right_editor) in enumerate(rows):
            left_label = QLabel(left_text, self)
            right_label = QLabel(right_text, self)
            left_label.setProperty("fittingFormLabel", True)
            right_label.setProperty("fittingFormLabel", True)
            layout.addWidget(left_label, row, 0)
            layout.addWidget(left_editor, row, 1)
            layout.addWidget(right_label, row, 2)
            layout.addWidget(right_editor, row, 3)

        status_row = len(rows)
        self.status_label = QLabel("", self)
        self.status_label.setObjectName("fittingDetectorSetupStatusLabel")
        self.status_label.setProperty("cardMeta", True)
        self.status_label.setWordWrap(True)
        self.apply_button.setParent(self)
        self.apply_button.setText("Apply detector setup")
        self.apply_button.setProperty("gimapPrimaryAction", True)
        normalize_button(self.apply_button)
        horizontal_q_label = QLabel("Horizontal q", self)
        horizontal_q_label.setProperty("fittingFormLabel", True)
        layout.addWidget(self.show_q_axis_checkbox, status_row, 0, 1, 2)
        layout.addWidget(horizontal_q_label, status_row, 2)
        layout.addWidget(self.horizontal_q_combo, status_row, 3)
        layout.addWidget(self.status_label, status_row + 1, 0, 1, 3)
        layout.addWidget(self.apply_button, status_row + 1, 3, Qt.AlignRight)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

    def _spinbox(
        self,
        object_name: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        step: float,
    ) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox(self)
        spinbox.setObjectName(object_name)
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(step)
        spinbox.setValue(value)
        spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        normalize_input(spinbox)
        return spinbox

    def _connect(self) -> None:
        self.wavelength_spinbox.valueChanged.connect(self._wavelength_changed)
        self.energy_spinbox.valueChanged.connect(self._energy_changed)
        self.parameter_commits = ParameterCommitCoordinator(self)
        self.parameter_commits.register_group(
            "detector_setup",
            commit=self.apply,
            policy=ParameterUpdatePolicy(debounce_ms=220),
        )
        for spinbox in (
            self.distance_spinbox,
            self.angle_spinbox,
            self.energy_spinbox,
            self.wavelength_spinbox,
            self.beam_center_x_spinbox,
            self.beam_center_y_spinbox,
            self.pixel_size_x_spinbox,
            self.pixel_size_y_spinbox,
        ):
            self.parameter_commits.bind_numeric("detector_setup", spinbox)
        self.parameter_commits.bind_toggle(
            "detector_setup", self.show_q_axis_checkbox
        )
        self.horizontal_q_combo.currentIndexChanged.connect(
            lambda _index: self.parameter_commits.flush("detector_setup")
        )
        self.show_q_axis_checkbox.toggled.connect(self.horizontal_q_combo.setEnabled)
        self.apply_button.clicked.connect(
            lambda _checked=False: self.parameter_commits.flush("detector_setup")
        )

    def _load(self) -> None:
        try:
            settings = self.view_model.load_detector_settings()
            self.distance_spinbox.setValue(settings.distance)
            self.angle_spinbox.setValue(settings.grazing_angle)
            self.wavelength_spinbox.setValue(settings.wavelength)
            if settings.wavelength > 0:
                self.energy_spinbox.setValue(wavelength_to_energy(settings.wavelength))
            self.beam_center_x_spinbox.setValue(settings.beam_center_x)
            self.beam_center_y_spinbox.setValue(settings.beam_center_y)
            self.pixel_size_x_spinbox.setValue(settings.pixel_size_x)
            self.pixel_size_y_spinbox.setValue(settings.pixel_size_y)
            self.show_q_axis_checkbox.setChecked(settings.show_q_axis)
            index = self.horizontal_q_combo.findData(settings.horizontal_q_axis)
            self.horizontal_q_combo.setCurrentIndex(index if index >= 0 else 0)
            self.horizontal_q_combo.setEnabled(settings.show_q_axis)
        except Exception as exc:
            self.status_label.setText(f"Could not load detector setup: {exc}")
            self.status_label.setProperty("statusKind", "error")

    def current_settings(self) -> DetectorSettings:
        return DetectorSettings(
            distance=self.distance_spinbox.value(),
            grazing_angle=self.angle_spinbox.value(),
            wavelength=self.wavelength_spinbox.value(),
            beam_center_x=self.beam_center_x_spinbox.value(),
            beam_center_y=self.beam_center_y_spinbox.value(),
            pixel_size_x=self.pixel_size_x_spinbox.value(),
            pixel_size_y=self.pixel_size_y_spinbox.value(),
            show_q_axis=self.show_q_axis_checkbox.isChecked(),
            horizontal_q_axis=str(self.horizontal_q_combo.currentData() or "qy"),
        )

    def apply(self) -> None:
        try:
            settings = self.current_settings()
            self.view_model.save_detector_settings(settings)
            payload = {**settings.__dict__, "energy": self.energy_spinbox.value()}
            self.status_label.setText("Detector setup applied")
            self.status_label.setProperty("statusKind", "complete")
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
            self.settings_applied.emit(payload)
        except Exception as exc:
            message = f"Could not apply detector setup: {exc}"
            self.status_label.setText(message)
            self.status_label.setProperty("statusKind", "error")
            self.apply_failed.emit(message)

    def _wavelength_changed(self, wavelength: float) -> None:
        if self._syncing_energy_pair or wavelength <= 0:
            return
        self._syncing_energy_pair = True
        try:
            self.energy_spinbox.setValue(wavelength_to_energy(wavelength))
        finally:
            self._syncing_energy_pair = False

    def _energy_changed(self, energy: float) -> None:
        if self._syncing_energy_pair or energy <= 0:
            return
        self._syncing_energy_pair = True
        try:
            wavelength = energy_to_wavelength(energy)
            wavelength = min(
                self.wavelength_spinbox.maximum(),
                max(self.wavelength_spinbox.minimum(), wavelength),
            )
            self.wavelength_spinbox.setValue(wavelength)
        finally:
            self._syncing_energy_pair = False


__all__ = ["DetectorSetupPanel"]
