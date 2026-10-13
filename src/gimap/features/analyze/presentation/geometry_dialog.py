"""Apply manual beam centres repeatedly and save detector geometry profiles."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget

from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme import set_role

from ..application import DetectorGeometry
from .views.geometry_dialog_view import GeometryDialogView, PROFILE_NAME_WIDTH

HC_KEV_ANGSTROM = 12.398419843320026


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


class GeometryDialog(QDialog, GeometryDialogView):
    applyCenterRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    deleteRequested = pyqtSignal()
    validationFailed = pyqtSignal(str)

    DELETED = 2
    """Legacy result retained for callers that compare the deleted result code."""

    def __init__(
        self,
        name: str,
        defaults: dict[str, float],
        parent: Optional[QWidget] = None,
        *,
        allow_delete: bool = False,
        center_step_px: float = 1.0,
    ):
        super().__init__(parent)
        self.setup_geometry_ui(
            name, defaults, allow_delete=allow_delete, center_step_px=center_step_px,
        )
        self.apply_button.clicked.connect(self.applyCenterRequested)
        self.save_button.clicked.connect(self._save_if_named)
        self.close_button.clicked.connect(self.reject)
        if self.delete_button is not None:
            self.delete_button.clicked.connect(self.deleteRequested)

    def _save_if_named(self) -> None:
        if self.name_edit.text().strip():
            self.saveRequested.emit()
        else:
            self.name_edit.setFocus()
            self.set_status(tr("Enter a profile name before saving."), "warning")
            self.validationFailed.emit("Enter a profile name before saving.")

    def set_status(self, text: str, level: str = "info") -> None:
        self.status_label.setText(text)
        set_role(self.status_label, "success" if level == "ok" else level)

    def center(self) -> tuple[float, float]:
        self.center_x_spin.interpretText()
        self.center_y_spin.interpretText()
        return self.center_x_spin.value(), self.center_y_spin.value()

    def profile_name(self) -> str:
        return self.name_edit.text().strip()

    def geometry(self) -> DetectorGeometry:
        for spin in (self.pixel_x_spin, self.pixel_y_spin, self.distance_spin, self.center_x_spin,
                     self.center_y_spin, self.wavelength_spin, self.incidence_spin):
            spin.interpretText()
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
