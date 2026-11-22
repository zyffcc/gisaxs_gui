"""Apply manual beam centres repeatedly and save detector geometry profiles.

The wavelength and the energy are one value in two units (typing one sets the other); the line under them says
where it came from — the file header, a ``.poni``, the profile in use, or the user — and, when nothing states it,
says so in the warning style: the fields then read “Required” and the geometry cannot be saved or exported until
one is typed (no wavelength is made up). The detector tilt is pyFAI's rot1, rot2, rot3 in degrees.
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt5.QtCore import QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget

from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme import set_role

from ..application import DetectorGeometry
from .views.geometry_dialog_view import GeometryDialogView, PROFILE_NAME_WIDTH  # noqa: F401  (re-exported)

HC_KEV_ANGSTROM = 12.398419843320026
SOURCE_HEADER, SOURCE_PONI, SOURCE_PROFILE, SOURCE_USER, SOURCE_NONE = "header", "poni", "profile", "user", ""
SOURCE_TEXTS = {
    SOURCE_HEADER: "Wavelength from the file header.",
    SOURCE_PONI: "Wavelength from {name}.",
    SOURCE_PROFILE: "Wavelength of profile “{name}”.",
    SOURCE_USER: "Wavelength entered by you.",
    SOURCE_NONE: "No wavelength or energy is known for this frame: enter one (q scales with it).",
}
WAVELENGTH_REQUIRED = "Enter the wavelength or the energy first: the file states none."


def geometry_defaults(metadata: dict, shape: tuple[int, int]) -> dict:
    """Starting values from file metadata; the frame centre when the beam is unknown. The wavelength is ``None``
    when the file states neither it nor the energy (``wavelength_source`` says which)."""
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
        "wavelength_angstrom": float(wavelength) if wavelength else None,
        "energy_kev": HC_KEV_ANGSTROM / float(wavelength) if wavelength else None,
        "wavelength_source": SOURCE_HEADER if wavelength else SOURCE_NONE,
        "center_x": columns / 2.0,
        "center_y": rows / 2.0,
        "incidence_deg": 0.0,
        "rot1_deg": 0.0,
        "rot2_deg": 0.0,
        "rot3_deg": 0.0,
    }


class GeometryDialog(QDialog, GeometryDialogView):
    applyCenterRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    deleteRequested = pyqtSignal()
    validationFailed = pyqtSignal(str)
    loadPoniRequested = pyqtSignal()
    exportPoniRequested = pyqtSignal()

    DELETED = 2
    """Legacy result retained for callers that compare the deleted result code."""

    def __init__(
        self,
        name: str,
        defaults: dict,
        parent: Optional[QWidget] = None,
        *,
        allow_delete: bool = False,
        center_step_px: float = 1.0,
    ):
        super().__init__(parent)
        self._orientation = defaults.get("orientation", 3)
        self.setup_geometry_ui(
            name, defaults, allow_delete=allow_delete, center_step_px=center_step_px,
        )
        self._wavelength_source = (defaults.get("wavelength_source") or SOURCE_NONE,
                                   defaults.get("wavelength_source_name") or "")
        self.apply_button.clicked.connect(self.applyCenterRequested)
        self.save_button.clicked.connect(self._save_if_named)
        self.close_button.clicked.connect(self.reject)
        self.load_poni_button.clicked.connect(self.loadPoniRequested)
        self.export_poni_button.clicked.connect(self._export_if_known)
        if self.delete_button is not None:
            self.delete_button.clicked.connect(self.deleteRequested)
        self.wavelength_spin.valueChanged.connect(self._wavelength_typed)
        self.energy_spin.valueChanged.connect(self._energy_typed)
        self._show_wavelength_source()

    # -- wavelength and energy --------------------------------------------------------------

    def wavelength_known(self) -> bool:
        return self.wavelength_spin.value() > 0

    def _set_pair(self, wavelength: float) -> None:
        with QSignalBlocker(self.wavelength_spin), QSignalBlocker(self.energy_spin):
            self.wavelength_spin.setValue(wavelength if wavelength > 0 else 0.0)
            self.energy_spin.setValue(HC_KEV_ANGSTROM / wavelength if wavelength > 0 else 0.0)

    def _wavelength_typed(self, value: float) -> None:
        self._set_pair(value)
        self._wavelength_source = (SOURCE_USER if value > 0 else SOURCE_NONE, "")
        self._show_wavelength_source()

    def _energy_typed(self, value: float) -> None:
        self._set_pair(HC_KEV_ANGSTROM / value if value > 0 else 0.0)
        self._wavelength_source = (SOURCE_USER if value > 0 else SOURCE_NONE, "")
        self._show_wavelength_source()

    def set_wavelength(self, wavelength: Optional[float], source: str, name: str = "") -> None:
        """Show ``wavelength`` (Å; ``None``: not known) as coming from ``source`` (``SOURCE_*``; ``name``: the file
        or profile it names)."""
        self._set_pair(float(wavelength) if wavelength else 0.0)
        self._wavelength_source = (source if wavelength else SOURCE_NONE, name)
        self._show_wavelength_source()

    def wavelength_source(self) -> tuple[str, str]:
        return self._wavelength_source

    def _show_wavelength_source(self) -> None:
        source, name = self._wavelength_source
        known = self.wavelength_known()
        text = tr(SOURCE_TEXTS[source if known else SOURCE_NONE])
        self.wavelength_source_label.setText(text.format(name=name) if "{name}" in text else text)
        set_role(self.wavelength_source_label, None if known else "warning")
        for spin in (self.wavelength_spin, self.energy_spin):
            set_role(spin, None if known else "error")

    def refresh_language(self) -> None:
        self._show_wavelength_source()

    # -- actions ------------------------------------------------------------------------------

    def _save_if_named(self) -> None:
        if not self.wavelength_known():
            self._wavelength_missing()
        elif self.name_edit.text().strip():
            self.saveRequested.emit()
        else:
            self.name_edit.setFocus()
            self.set_status(tr("Enter a profile name before saving."), "warning")
            self.validationFailed.emit("Enter a profile name before saving.")

    def _export_if_known(self) -> None:
        if self.wavelength_known():
            self.exportPoniRequested.emit()
        else:
            self._wavelength_missing()

    def _wavelength_missing(self) -> None:
        self.wavelength_spin.setFocus()
        self.set_status(tr(WAVELENGTH_REQUIRED), "warning")
        self.validationFailed.emit(WAVELENGTH_REQUIRED)

    def set_status(self, text: str, level: str = "info") -> None:
        self.status_label.setText(text)
        set_role(self.status_label, "success" if level == "ok" else level)

    def center(self) -> tuple[float, float]:
        self.center_x_spin.interpretText()
        self.center_y_spin.interpretText()
        return self.center_x_spin.value(), self.center_y_spin.value()

    def profile_name(self) -> str:
        return self.name_edit.text().strip()

    def set_geometry(self, geometry: DetectorGeometry, *, source: str, name: str = "") -> None:
        """Fill every field from ``geometry`` (a ``.poni`` read: ``source`` ``SOURCE_PONI``, ``name`` its file)."""
        self._orientation = geometry.orientation
        for spin, value in ((self.distance_spin, geometry.distance_m * 1e3),
                            (self.pixel_x_spin, geometry.pixel_size_x_m * 1e6),
                            (self.pixel_y_spin, geometry.pixel_size_y_m * 1e6),
                            (self.center_x_spin, geometry.beam_center_x_px),
                            (self.center_y_spin, geometry.beam_center_y_px),
                            (self.rot1_spin, math.degrees(geometry.rot1_rad)),
                            (self.rot2_spin, math.degrees(geometry.rot2_rad)),
                            (self.rot3_spin, math.degrees(geometry.rot3_rad))):
            spin.setValue(value)
        self.set_wavelength(geometry.wavelength_angstrom, source, name)

    def geometry(self) -> DetectorGeometry:
        """The geometry of the fields (``ValueError`` while the wavelength is not known)."""
        for spin in (self.pixel_x_spin, self.pixel_y_spin, self.distance_spin, self.center_x_spin,
                     self.center_y_spin, self.wavelength_spin, self.incidence_spin, *self.rotation_spins):
            spin.interpretText()
        if not self.wavelength_known():
            raise ValueError(WAVELENGTH_REQUIRED)
        return DetectorGeometry(
            pixel_size_x_m=self.pixel_x_spin.value() * 1e-6,
            pixel_size_y_m=self.pixel_y_spin.value() * 1e-6,
            distance_m=self.distance_spin.value() * 1e-3,
            beam_center_x_px=self.center_x_spin.value(),
            beam_center_y_px=self.center_y_spin.value(),
            wavelength_angstrom=self.wavelength_spin.value(),
            incidence_deg=self.incidence_spin.value(),
            rot1_rad=math.radians(self.rot1_spin.value()),
            rot2_rad=math.radians(self.rot2_spin.value()),
            rot3_rad=math.radians(self.rot3_spin.value()),
            orientation=self._orientation,
        )

    def source_text(self) -> str:
        """The profile's source for a save: entered by hand, and where the wavelength came from."""
        source, name = self._wavelength_source
        origin = {SOURCE_HEADER: "file header", SOURCE_PONI: name or ".poni", SOURCE_PROFILE: f"profile {name}",
                  SOURCE_USER: "entered by hand"}.get(source, "entered by hand")
        return f"entered by hand · wavelength: {origin}"


__all__ = ["GeometryDialog", "SOURCE_HEADER", "SOURCE_NONE", "SOURCE_PONI", "SOURCE_PROFILE", "SOURCE_USER",
           "WAVELENGTH_REQUIRED", "geometry_defaults"]
