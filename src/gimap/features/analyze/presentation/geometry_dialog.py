"""Enter a detector geometry by hand and store it as an instrument profile."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..application import DetectorGeometry

HC_KEV_ANGSTROM = 12.398419843320026


def _spin(parent, low, high, decimals, value, suffix, step=1.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setValue(value)
    return spin


def geometry_defaults(metadata: dict, shape: tuple[int, int]) -> dict[str, float]:
    """Starting values from file metadata; the frame centre when the beam is unknown."""
    rows, columns = shape
    pixel_x = metadata.get("pixel_size_x_m") or 172e-6
    pixel_y = metadata.get("pixel_size_y_m") or pixel_x
    wavelength = metadata.get("wavelength_angstrom") or metadata.get("header_wavelength_angstrom")
    if not wavelength and metadata.get("energy_kev"):
        wavelength = HC_KEV_ANGSTROM / float(metadata["energy_kev"])
    distance = metadata.get("distance_m") or metadata.get("header_distance_m") or 1.0
    return {
        "distance_mm": float(distance) * 1e3,
        "pixel_x_um": float(pixel_x) * 1e6,
        "pixel_y_um": float(pixel_y) * 1e6,
        "wavelength_angstrom": float(wavelength or 1.0),
        "center_x": columns / 2.0,
        "center_y": rows / 2.0,
        "incidence_deg": 0.0,
    }


class GeometryDialog(QDialog):
    DELETED = 2
    """``exec_()`` result when the user deleted the profile being edited."""

    def __init__(
        self,
        name: str,
        defaults: dict[str, float],
        parent: Optional[QWidget] = None,
        *,
        allow_delete: bool = False,
    ):
        super().__init__(parent)
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
        form = QFormLayout()
        self.name_edit = QLineEdit(name, self)
        self.distance_spin = _spin(self, 1.0, 1e5, 2, defaults["distance_mm"], " mm", 10.0)
        self.pixel_x_spin = _spin(self, 1.0, 1e4, 3, defaults["pixel_x_um"], " µm")
        self.pixel_y_spin = _spin(self, 1.0, 1e4, 3, defaults["pixel_y_um"], " µm")
        self.wavelength_spin = _spin(self, 0.01, 50.0, 5, defaults["wavelength_angstrom"], " Å", 0.01)
        self.center_x_spin = _spin(self, -1e5, 1e5, 2, defaults["center_x"], " px")
        self.center_y_spin = _spin(self, -1e5, 1e5, 2, defaults["center_y"], " px")
        self.incidence_spin = _spin(self, 0.0, 10.0, 3, defaults["incidence_deg"], " °", 0.01)
        form.addRow("Profile name", self.name_edit)
        form.addRow("Sample–detector distance", self.distance_spin)
        form.addRow("Pixel size x", self.pixel_x_spin)
        form.addRow("Pixel size y", self.pixel_y_spin)
        form.addRow("Wavelength", self.wavelength_spin)
        form.addRow("Beam centre x", self.center_x_spin)
        form.addRow("Beam centre y", self.center_y_spin)
        form.addRow("Incidence angle αi", self.incidence_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept_if_named)
        buttons.rejected.connect(self.reject)
        self.delete_button = None
        if allow_delete:
            self.delete_button = buttons.addButton("Delete profile", QDialogButtonBox.DestructiveRole)
            self.delete_button.clicked.connect(lambda: self.done(self.DELETED))
        layout.addWidget(buttons)

    def _accept_if_named(self) -> None:
        if self.name_edit.text().strip():
            self.accept()
        else:
            self.name_edit.setFocus()

    def profile_name(self) -> str:
        return self.name_edit.text().strip()

    def geometry(self) -> DetectorGeometry:
        return DetectorGeometry(
            pixel_size_x_m=self.pixel_x_spin.value() * 1e-6,
            pixel_size_y_m=self.pixel_y_spin.value() * 1e-6,
            distance_m=self.distance_spin.value() * 1e-3,
            beam_center_x_px=self.center_x_spin.value(),
            beam_center_y_px=self.center_y_spin.value(),
            wavelength_angstrom=self.wavelength_spin.value(),
            incidence_deg=self.incidence_spin.value(),
        )


__all__ = ["GeometryDialog", "geometry_defaults"]
