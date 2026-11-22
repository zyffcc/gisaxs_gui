"""Where the calibration energy comes from, said next to the field; no energy is ever made up.

The Energy field starts at 0, shown as “Required”. A file that states its energy (its header, or the companion
NXS of a CBF) fills it in and the line under it says so; a file that states none leaves it (or the value the user
entered for the image before) and the line says, in the warning style, that the energy must be entered or checked.
Typing a value makes it “entered by you”. Auto Calibration waits for an energy, and the result records where it
came from (``CalibrationResult.metadata["energy_source"]``, written with the calibration and into the instrument
profile's source).
"""

from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.theme import set_role
from src.gimap.shared.detector_io.radiation import HC_KEV_ANGSTROM

FROM_HEADER, FROM_NXS, FROM_USER, FROM_IMPORT, MISSING, KEPT = (
    "file header", "companion NXS", "entered by user", "imported calibration", "", "kept")
SOURCE_TEXT = {
    FROM_HEADER: "From the file header.",
    FROM_NXS: "From the companion NXS file.",
    FROM_USER: "Entered by you.",
    FROM_IMPORT: "From the imported calibration.",
    MISSING: "Not in the file. Enter the energy in keV or the wavelength in Å.",
    KEPT: "Not in this file: the energy entered for the image before is used — check it.",
}


class EnergySourceMixin:
    """Own the energy source line, its validation and the tilt option of a run."""

    _energy_source = MISSING

    def _bind_energy_source(self) -> None:
        self.wavelength_spin.setMaximum(HC_KEV_ANGSTROM * 10 ** self.energy_spin.decimals())
        self.energy_spin.valueChanged.connect(self._energy_edited)
        self.wavelength_spin.valueChanged.connect(self._wavelength_edited)
        language_changed().connect(self._show_energy_source)
        self._show_energy_source()

    def _energy_edited(self, value: float) -> None:
        self._set_energy(value, FROM_USER if value else MISSING)
        self._update_input_status()

    def _wavelength_edited(self, wavelength: float) -> None:
        energy = HC_KEV_ANGSTROM / wavelength if wavelength else 0.0
        with QSignalBlocker(self.energy_spin):
            self.energy_spin.setValue(energy if energy <= self.energy_spin.maximum() else 0.0)
        self._energy_source = FROM_USER if self.energy_spin.value() else MISSING
        self._show_energy_source()
        self._update_input_status()

    def _update_input_status(self) -> None:
        if self.image is not None and self.result is None and not self._running:
            self._show_loaded_status()

    def _set_energy(self, value, source: str) -> None:
        """Show ``value`` (keV; ``None``/0: none known) as coming from ``source``."""
        if value or source == MISSING:
            with QSignalBlocker(self.energy_spin), QSignalBlocker(self.wavelength_spin):
                self.energy_spin.setValue(float(value or 0.0))
                energy = self.energy_spin.value()
                self.wavelength_spin.setValue(HC_KEV_ANGSTROM / energy if energy else 0.0)
        self._energy_source = source
        self._show_energy_source()

    def _image_energy(self, image) -> None:
        """The energy of an image just read: its own, else the one the user entered before (said), else none."""
        if image.energy_kev:
            source = FROM_NXS if image.metadata.get("energy_source") else FROM_HEADER
            self._set_energy(image.energy_kev, source)
        elif self.energy_spin.value() > 0 and self._energy_source in (FROM_USER, KEPT):
            self._set_energy(None, KEPT)
        else:
            self._set_energy(None, MISSING)

    def _show_energy_source(self) -> None:
        label = getattr(self, "energy_source_label", None)
        if label is None:
            return
        known = self.energy_spin.value() > 0
        source = self._energy_source if known else MISSING
        text = tr(SOURCE_TEXT.get(source, SOURCE_TEXT[MISSING]))
        if not known and self.wavelength_spin.value() > 0:
            text = trf("The energy exceeds {maximum} keV. Check the wavelength in Å.",
                       maximum=f"{self.energy_spin.maximum():g}")
        label.setText(text)
        set_role(label, None if known and source not in (KEPT,) else "warning")
        set_role(self.energy_spin, None if known else "error")
        set_role(self.wavelength_spin, None if known else "error")
        if hasattr(self, "calibrate_button") and not getattr(self, "_running", False):
            self.calibrate_button.setEnabled(self.image is not None and known)

    def energy_ready(self) -> bool:
        """An energy is known; otherwise the field gets the focus and the line says why nothing starts."""
        if self.energy_spin.value() > 0:
            return True
        self._energy_source = MISSING
        self._show_energy_source()
        self.energy_spin.setFocus()
        return False

    def run_energy_source(self) -> str:
        """What the result records as the energy's origin."""
        return {KEPT: FROM_USER, MISSING: FROM_USER}.get(self._energy_source, self._energy_source)

    def run_tilt_mode(self) -> str:
        return "auto" if self.tilt_check.isChecked() else "off"


__all__ = ["EnergySourceMixin", "SOURCE_TEXT"]
