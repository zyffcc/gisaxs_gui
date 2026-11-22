"""Detector geometry form and persistent actions for manual centre adjustment."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..center_controls import make_center_step

PROFILE_NAME_WIDTH = 260
"""Room to read a detector profile name and its frame size."""
PIXEL_NOTE = "Beam centre in pixel-edge coordinates: the first pixel's centre is 0.5."
"""Seen at once: a centre typed in the other convention would be half a pixel off (the whole rule: the fields' tooltip)."""
CENTER_TIP = "Beam centre = direct beam in pixels, pixel j spanning [j, j+1] (the centre of the first pixel is 0.5)."
APPLY_TIP = "Use this centre for this session and redraw the curves"
SAVE_TIP = ("Save all geometry values for this detector: the profile is used for every frame from this detector with "
            "this size")
ENERGY_TIP = "X-ray wavelength and energy: E = hc/λ (12.398 keV·Å); typing one sets the other"
ROTATION_TIP = ("pyFAI's detector rotations (as in a .poni): rot1 about the vertical axis, rot2 about the horizontal "
                "axis, rot3 about the beam; 0 is a detector perpendicular to the beam. The beam centre stays the direct "
                "beam and the distance the distance to it")
LOAD_PONI_TIP = "Read a pyFAI .poni (distance, PONI, rot1–rot3, pixel sizes, wavelength) into these fields"
EXPORT_PONI_TIP = "Write these values as a pyFAI .poni (version 2.1) for pyFAI, GIXSGUI and other programs"


def _spin(parent, low, high, decimals, value, suffix, step=1.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    spin.setValue(value)
    return spin


class GeometryDialogView:
    """Build fields on the dialog; actions are connected by GeometryDialog."""

    def setup_geometry_ui(self, name, defaults, *, allow_delete, center_step_px) -> None:
        self.setWindowTitle("Detector geometry")
        self.setObjectName("analyzeGeometryDialog")
        layout = QVBoxLayout(self)
        note = QLabel(PIXEL_NOTE, self)
        note.setObjectName("analyzeGeometryPixelNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.name_edit = QLineEdit(name, self)
        self.name_edit.setMinimumWidth(PROFILE_NAME_WIDTH)
        self.name_edit.setCursorPosition(0)
        self.distance_spin = _spin(self, 1.0, 1e5, 2, defaults["distance_mm"], " mm", 10.0)
        self.pixel_x_spin = _spin(self, 1.0, 1e4, 3, defaults["pixel_x_um"], " µm")
        self.pixel_y_spin = _spin(self, 1.0, 1e4, 3, defaults["pixel_y_um"], " µm")
        # 0 is “Required”: a wavelength the file does not state is never made up (``GeometryDialog.wavelength_known``).
        self.wavelength_spin = _spin(self, 0.0, 50.0, 5, defaults["wavelength_angstrom"] or 0.0, " Å", 0.01)
        self.wavelength_spin.setSpecialValueText("Required")
        self.energy_spin = _spin(self, 0.0, 500.0, 4, defaults.get("energy_kev") or 0.0, " keV", 0.1)
        self.energy_spin.setSpecialValueText("Required")
        self.energy_spin.setToolTip(ENERGY_TIP)
        self.wavelength_spin.setToolTip(ENERGY_TIP)
        self.wavelength_source_label = QLabel(self)
        self.wavelength_source_label.setObjectName("analyzeGeometryWavelengthSource")
        self.wavelength_source_label.setWordWrap(True)
        self.wavelength_source_label.setMinimumHeight(2 * self.wavelength_source_label.fontMetrics().lineSpacing() + 2)
        self.rotation_spins = []
        for key in ("rot1_deg", "rot2_deg", "rot3_deg"):
            spin = _spin(self, -89.0, 89.0, 4, defaults.get(key, 0.0), " °", 0.01)
            spin.setToolTip(ROTATION_TIP)
            self.rotation_spins.append(spin)
        self.rot1_spin, self.rot2_spin, self.rot3_spin = self.rotation_spins
        self.rot3_spin.setRange(-180.0, 180.0)
        self.center_x_spin = _spin(self, -1e5, 1e5, 4, defaults["center_x"], " px")
        self.center_y_spin = _spin(self, -1e5, 1e5, 4, defaults["center_y"], " px")
        for spin in (self.center_x_spin, self.center_y_spin):
            spin.setToolTip(CENTER_TIP)
        self.center_step_spin = make_center_step(
            self, self.center_x_spin, self.center_y_spin, center_step_px,
        )
        self.incidence_spin = _spin(self, 0.0, 10.0, 3, defaults["incidence_deg"], " °", 0.01)
        self.sample_roll_spin = _spin(self, -180.0, 180.0, 4, defaults.get("sample_roll_deg", 0.0), " °", 0.05)
        form.addRow("Profile name", self.name_edit)
        form.addRow("Sample–detector distance", self.distance_spin)
        form.addRow("Pixel size x", self.pixel_x_spin)
        form.addRow("Pixel size y", self.pixel_y_spin)
        form.addRow("Wavelength", self.wavelength_spin)
        form.addRow("Energy", self.energy_spin)
        form.addRow("", self.wavelength_source_label)
        form.addRow("Detector tilt rot1", self.rot1_spin)
        form.addRow("Detector tilt rot2", self.rot2_spin)
        form.addRow("Rotation about the beam rot3", self.rot3_spin)
        form.addRow("Beam centre x", self.center_x_spin)
        form.addRow("Beam centre y", self.center_y_spin)
        form.addRow("Centre step", self.center_step_spin)
        form.addRow("Incidence angle αi", self.incidence_spin)
        form.addRow("Sample roll ψ", self.sample_roll_spin)
        layout.addLayout(form)
        self.live_preview_check = QCheckBox("Live Preview", self)
        self.live_preview_check.setObjectName("analyzeGeometryLivePreview")
        self.live_preview_check.setToolTip(
            "Update curves automatically as the beam centre changes; the centre step "
            "controls arrow buttons, keys, and the mouse wheel."
        )
        layout.addWidget(self.live_preview_check)
        self.status_label = QLabel(self)
        self.status_label.setObjectName("analyzeGeometryStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(self.status_label.fontMetrics().height())
        self.status_label.setProperty("gimapRole", "info")
        layout.addWidget(self.status_label)
        buttons = QDialogButtonBox(self)
        self.apply_button = buttons.addButton("Apply Centre", QDialogButtonBox.ApplyRole)
        self.apply_button.setAutoDefault(True)
        self.apply_button.setDefault(True)
        self.apply_button.setProperty("gimapRole", "primary")
        self.apply_button.setToolTip(APPLY_TIP)
        self.save_button = buttons.addButton("Save Profile", QDialogButtonBox.ActionRole)
        self.save_button.setAutoDefault(False)
        self.save_button.setToolTip(SAVE_TIP)
        files = QHBoxLayout()
        self.load_poni_button = QPushButton("Load .poni…", self)
        self.load_poni_button.setToolTip(LOAD_PONI_TIP)
        self.export_poni_button = QPushButton("Export .poni…", self)
        self.export_poni_button.setToolTip(EXPORT_PONI_TIP)
        for button in (self.load_poni_button, self.export_poni_button):
            button.setAutoDefault(False)
            files.addWidget(button)
        files.addStretch(1)
        layout.addLayout(files)
        self.delete_button = None
        if allow_delete:
            self.delete_button = buttons.addButton("Delete profile", QDialogButtonBox.DestructiveRole)
            self.delete_button.setAutoDefault(False)
        self.close_button = buttons.addButton(QDialogButtonBox.Close)
        self.close_button.setAutoDefault(False)
        layout.addWidget(buttons)


__all__ = ["APPLY_TIP", "CENTER_TIP", "ENERGY_TIP", "EXPORT_PONI_TIP", "GeometryDialogView", "LOAD_PONI_TIP", "PIXEL_NOTE",
           "PROFILE_NAME_WIDTH", "ROTATION_TIP", "SAVE_TIP"]
