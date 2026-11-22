"""Load .poni… and Export .poni… of the Geometry step, the geometry dialog and the no-geometry banner, and “Use
Anyway” for a profile picked for frames of another size.

* **Load .poni…** (Geometry step, banner): the file is checked against the frame shown (frame size when the file
  states one, header pixel size); a mismatch is said and asked about, never taken silently. A file with a
  wavelength becomes the instrument profile of this detector, picked for the frames (so a ``.poni`` lying beside
  them does not win over the file chosen), and the frame is reduced again. A file without one opens the geometry
  dialog filled with it, the wavelength “Required”. In the open geometry dialog, Load .poni… fills its fields
  (saved with Save Profile).
* **Export .poni…** writes the geometry of the dialog's fields, or the geometry in use (pyFAI PONI 2.1).
* **Use Anyway** (banner): the profile picked by hand is used on frames of this size too (said in the status line
  and remembered until another profile is picked).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from src.gimap.app.presentation.i18n import tr, trf

from ...application import poni_problem, poni_profile, read_poni_file, write_poni_file
from ..geometry_dialog import SOURCE_PONI
from ..texts import message_text

PONI_FILTER = "pyFAI geometry (*.poni)"
LOADED = "Geometry from {name} saved as instrument profile “{profile}” and used for this detector."
EXPORTED = "Geometry written to {name}."


class GeometryFilesMixin:
    """Needs ``view_model``, ``run_analysis``, ``_status``, ``_refresh_profiles``, ``_enter_geometry``."""

    def _poni_start_folder(self) -> str:
        path = self.view_model.current_path
        return str(Path(path).parent) if path is not None else (self._last_folder or "")

    def load_poni(self, path: Optional[str] = None, *, dialog=None) -> bool:
        """Read a ``.poni`` (asks for one when ``path`` is ``None``); ``True`` when its geometry was taken."""
        analysis = self.view_model.state.analysis
        if analysis is None:
            self._status(tr("Open a frame first; the geometry is edited for its detector."), "warning")
            return False
        dialog = dialog if dialog is not None else getattr(self, "_geometry_dialog", None)
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("Load .poni"), self._poni_start_folder(), tr(PONI_FILTER))
            if not path:
                return False
        try:
            poni = read_poni_file(path, frame_shape=analysis.shape)
        except (OSError, ValueError) as exc:
            self._status(trf("Could not read the .poni file: {error}", error=message_text(str(exc))), "error")
            return False
        problem = poni_problem(poni, analysis.shape, (analysis.metadata.get("pixel_size_x_m"),
                                                      analysis.metadata.get("pixel_size_y_m")))
        if problem is not None:
            answer = QMessageBox.question(
                self, tr("Load .poni"),
                trf("{problem}\n\nUse this geometry for the frame shown anyway?", problem=message_text(problem)),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self._status(trf("{name} not used: {problem}", name=Path(path).name, problem=message_text(problem)),
                             "warning")
                return False
        name = Path(path).name
        if dialog is not None or not poni.wavelength_angstrom:
            if dialog is None:  # no wavelength in the file: the dialog asks for it
                self._enter_geometry()
                dialog = getattr(self, "_geometry_dialog", None)
                if dialog is None:
                    return False
            geometry = poni.geometry(wavelength_angstrom=poni.wavelength_angstrom or 1.0,
                                     incidence_deg=analysis.geometry.incidence_deg if analysis.geometry else 0.0)
            dialog.set_geometry(geometry, source=SOURCE_PONI, name=name)
            if not poni.wavelength_angstrom:
                dialog.set_wavelength(None, SOURCE_PONI, name)
            dialog.roll_measurement = poni.sample_roll_measurement
            dialog._poni_tilt = poni.tilt if poni.tilted else None
            self._set_geometry_editor_status(dialog, "Read {name}: check the values, then Save Profile.", "info",
                                             name=name)
            return True
        profile_name = self.view_model.suggested_profile_name()
        profile = poni_profile(poni, path, name=profile_name, detector_name=analysis.detector_name, shape=analysis.shape,
                               incidence_deg=analysis.geometry.incidence_deg if analysis.geometry else 0.0)
        try:
            self.view_model.save_profile(profile.name, profile.geometry, source=profile.source,
                                         dropped_tilt=profile.dropped_tilt, sample_roll_measurement=poni.sample_roll_measurement)
        except (ValueError, OSError) as exc:
            self._status(trf("Could not save the geometry: {error}", error=message_text(str(exc))), "error")
            return False
        if poni.sample_roll_deg is not None:
            self.view_model.set_sample_roll(None)  # use the loaded angle, including an explicit zero
        self.view_model.set_profile(profile.name)  # the file chosen wins over a .poni lying beside the frames
        self._refresh_profiles()
        self._remember()
        self._status(trf(LOADED, name=name, profile=profile.name), "ok")
        self.run_analysis()
        return True

    def export_poni(self, path: Optional[str] = None, *, dialog=None) -> Optional[Path]:
        """Write the geometry (of the open dialog's fields, else the one in use) as a ``.poni``."""
        analysis = self.view_model.state.analysis
        dialog = dialog if dialog is not None else getattr(self, "_geometry_dialog", None)
        try:
            geometry = dialog.geometry() if dialog is not None else (analysis.geometry if analysis else None)
        except ValueError as exc:
            self._status(message_text(str(exc)), "warning")
            return None
        if geometry is None:
            self._status(tr("There is no geometry to export: load a .poni, enter the geometry or calibrate."), "warning")
            return None
        if path is None:
            profile = analysis.resolution.profile if analysis is not None else None
            stem = (profile.name if profile is not None else Path(str(analysis.path)).stem) if analysis else "geometry"
            stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem).strip("_") or "geometry"
            start = Path(self._poni_start_folder()) / f"{stem}.poni"
            path, _ = QFileDialog.getSaveFileName(self, tr("Export .poni"), str(start), tr(PONI_FILTER))
            if not path:
                return None
            if not path.lower().endswith(".poni"):
                path += ".poni"
        profile = analysis.resolution.profile if analysis is not None else None
        try:
            written = write_poni_file(geometry, path, shape=analysis.shape if analysis else None,
                                      source=profile.source if profile is not None and dialog is None else "",
                                      sample_roll_measurement=(dialog.roll_measurement if dialog is not None
                                                               else analysis.sample_roll_measurement))
        except OSError as exc:
            self._status(trf("Could not write the .poni file: {error}", error=str(exc)), "error")
            return None
        self.notify_written(trf(EXPORTED, name=written.name), written.parent)
        return written

    def use_profile_anyway(self) -> None:
        """Use the profile picked by hand on frames of the size shown too (the banner's Use Anyway)."""
        analysis = self.view_model.state.analysis
        resolution = analysis.resolution if analysis is not None else None
        if resolution is None or not resolution.refused_profile:
            return
        shape = tuple(int(value) for value in analysis.shape)
        state = self.view_model.state
        state.forced_shapes = tuple(dict.fromkeys((*state.forced_shapes, shape)))
        self._status(trf("Profile “{name}” is used for {shape} frames too, as you asked.", name=resolution.refused_profile,
                         shape=f"{shape[0]}×{shape[1]}"), "warning")
        self.run_analysis()


__all__ = ["GeometryFilesMixin", "PONI_FILTER"]
