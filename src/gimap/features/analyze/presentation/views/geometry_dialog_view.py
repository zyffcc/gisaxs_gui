"""Detector geometry form and persistent actions for manual centre adjustment."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..center_controls import make_center_step

PROFILE_NAME_WIDTH = 260
"""Room to read a detector profile name and its frame size."""


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
        note = QLabel(
            "Beam centre = direct beam in pixels, pixel j spanning [j, j+1] "
            "(the centre of the first pixel is 0.5). The profile is used for every frame "
            "from this detector with this size.",
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.apply_note = QLabel(
            "Apply Centre updates the curves for this session. "
            "Save Profile saves all geometry values for this detector.",
            self,
        )
        self.apply_note.setWordWrap(True)
        self.apply_note.setProperty("gimapRole", "muted")
        layout.addWidget(self.apply_note)
        form = QFormLayout()
        self.name_edit = QLineEdit(name, self)
        self.name_edit.setMinimumWidth(PROFILE_NAME_WIDTH)
        self.name_edit.setCursorPosition(0)
        self.distance_spin = _spin(self, 1.0, 1e5, 2, defaults["distance_mm"], " mm", 10.0)
        self.pixel_x_spin = _spin(self, 1.0, 1e4, 3, defaults["pixel_x_um"], " µm")
        self.pixel_y_spin = _spin(self, 1.0, 1e4, 3, defaults["pixel_y_um"], " µm")
        self.wavelength_spin = _spin(self, 0.01, 50.0, 5, defaults["wavelength_angstrom"], " Å", 0.01)
        self.center_x_spin = _spin(self, -1e5, 1e5, 4, defaults["center_x"], " px")
        self.center_y_spin = _spin(self, -1e5, 1e5, 4, defaults["center_y"], " px")
        self.center_step_spin = make_center_step(
            self, self.center_x_spin, self.center_y_spin, center_step_px,
        )
        self.incidence_spin = _spin(self, 0.0, 10.0, 3, defaults["incidence_deg"], " °", 0.01)
        form.addRow("Profile name", self.name_edit)
        form.addRow("Sample–detector distance", self.distance_spin)
        form.addRow("Pixel size x", self.pixel_x_spin)
        form.addRow("Pixel size y", self.pixel_y_spin)
        form.addRow("Wavelength", self.wavelength_spin)
        form.addRow("Beam centre x", self.center_x_spin)
        form.addRow("Beam centre y", self.center_y_spin)
        form.addRow("Centre step", self.center_step_spin)
        form.addRow("Incidence angle αi", self.incidence_spin)
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
        self.save_button = buttons.addButton("Save Profile", QDialogButtonBox.ActionRole)
        self.save_button.setAutoDefault(False)
        self.delete_button = None
        if allow_delete:
            self.delete_button = buttons.addButton("Delete profile", QDialogButtonBox.DestructiveRole)
            self.delete_button.setAutoDefault(False)
        self.close_button = buttons.addButton(QDialogButtonBox.Close)
        self.close_button.setAutoDefault(False)
        layout.addWidget(buttons)


__all__ = ["GeometryDialogView", "PROFILE_NAME_WIDTH"]
