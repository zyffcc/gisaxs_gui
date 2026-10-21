"""Instrument-profile choices, geometry entry and remembered preferences for Analyze.

``FitHandoverMixin`` (Send to Fitting with a folder of its own when the data folder takes no file) is part of
the page through this mixin."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QMessageBox

from src.gimap.app.presentation.i18n import tr, trf

from ..geometry_dialog import GeometryDialog, geometry_defaults
from ..preferences import AnalyzePreferences, load_preferences, save_preferences
from ..views.analyze_page_view import AUTO_PROFILE_TEXT
from .fit_handover import FitHandoverMixin
from .geometry_editor import GeometryEditorMixin


class ProfileActionsMixin(GeometryEditorMixin, FitHandoverMixin):
    """Own the Instrument combo, the no-geometry banner actions and preferences."""

    def _refresh_profiles(self) -> None:
        current = self.view_model.state.profile_name
        with QSignalBlocker(self.profile_combo):
            self.profile_combo.clear()
            self.profile_combo.addItem(AUTO_PROFILE_TEXT, None)
            for name in self.view_model.profile_names():
                self.profile_combo.addItem(name, name)
            index = self.profile_combo.findData(current) if current else 0
            self.profile_combo.setCurrentIndex(max(0, index))

    def _profile_chosen(self, _index: int) -> None:
        self.view_model.set_profile(self.profile_combo.currentData())
        self._remember()
        self.run_analysis()

    def _profile_saved(self, name: str) -> None:
        self._refresh_profiles()
        self._status(trf("Saved instrument profile “{name}”.", name=name), "ok")
        self.run_analysis()

    def _offer_previous_geometry(self) -> None:
        """Show “Use Previous Geometry” only when the former Fitting page stored one."""
        geometry = self.view_model.fitting_geometry()
        self.use_fitting_button.setVisible(geometry is not None)
        if geometry is not None:
            self.use_fitting_button.setToolTip(trf(
                "Save the detector geometry of the former Cut & Fitting page as the instrument profile of this "
                "detector:\n{geometry}", geometry=_describe(geometry)))

    def _use_fitting_geometry(self) -> None:
        geometry = self.view_model.fitting_geometry()
        if geometry is None:
            self._status(tr("No previous geometry is stored; enter the geometry instead."), "error")
            return
        name = self.view_model.suggested_profile_name()
        answer = QMessageBox.question(
            self,
            tr("Use previous geometry"),
            trf("Save profile “{name}” with the geometry of the former Cut & Fitting page?\n\n{geometry}",
                name=name, geometry=_describe(geometry)),
        )
        if answer == QMessageBox.Yes:
            self.view_model.save_profile(name, geometry, source="former Cut & Fitting settings")
            self._profile_saved(name)

    def _enter_geometry(self) -> None:
        """Edit the geometry in use (or enter one); save it as a profile or delete that profile."""
        if self._raise_geometry_editor("_geometry_dialog"):
            return
        analysis = self.view_model.state.analysis
        target = self._geometry_editor_target()
        if target is None:
            self._status(tr("Open a frame first; the geometry is edited for its detector."), "warning")
            return
        current = analysis.geometry
        profile = analysis.resolution.profile
        defaults = geometry_defaults(analysis.metadata, analysis.shape)
        if current is not None:
            defaults.update(
                distance_mm=current.distance_m * 1e3,
                pixel_x_um=current.pixel_size_x_m * 1e6,
                pixel_y_um=current.pixel_size_y_m * 1e6,
                wavelength_angstrom=current.wavelength_angstrom,
                center_x=current.beam_center_x_px,
                center_y=current.beam_center_y_px,
                incidence_deg=current.incidence_deg,
            )
        name = profile.name if profile is not None else self.view_model.suggested_profile_name()
        dialog = GeometryDialog(
            name, defaults, self, allow_delete=profile is not None, center_step_px=self._center_step_px(),
        )
        dialog._edited_profile_name = profile.name if profile is not None else None
        dialog.applyCenterRequested.connect(lambda: self._apply_geometry_editor_center(dialog))
        dialog.saveRequested.connect(lambda: self._save_geometry_editor_profile(dialog))
        dialog.deleteRequested.connect(lambda: self._delete_geometry_editor_profile(dialog))
        dialog.validationFailed.connect(lambda text: self._set_geometry_editor_status(dialog, text, "warning"))
        self._show_geometry_editor("_geometry_dialog", dialog, target)

    def _save_geometry_editor_profile(self, dialog) -> None:
        if not self._geometry_editor_is_current(dialog):
            dialog.close()
            return
        try:
            self.view_model.save_profile(dialog.profile_name(), dialog.geometry(), source="entered by hand")
        except (ValueError, OSError) as exc:
            self._set_geometry_editor_status(dialog, "Could not save the geometry: {error}", "error", error=str(exc))
            return
        self._cancel_geometry_previews()
        dialog._edited_profile_name = dialog.profile_name()
        self._profile_saved(dialog.profile_name())

    def _delete_geometry_editor_profile(self, dialog) -> None:
        if not self._geometry_editor_is_current(dialog):
            dialog.close()
            return
        name = dialog._edited_profile_name
        if name is None:
            return
        answer = QMessageBox.question(self, tr("Delete profile"), trf("Delete instrument profile “{name}”?", name=name))
        if answer == QMessageBox.Yes and self.view_model.delete_profile(name):
            dialog.close()
            self._refresh_profiles()
            self._remember()
            self._status(trf("Deleted instrument profile “{name}”.", name=name), "ok")
            self.run_analysis()

    def _open_calibration(self) -> None:
        if self._calibrate is None:
            return
        self._calibrate(self.view_model.current_path)  # its Apply calls ``geometry_calibrated``
        self._refresh_profiles()
        self.run_analysis()

    def geometry_calibrated(self) -> None:
        """A calibration was applied (the Calibration window calls this after Apply): its geometry is the one
        used from now on. A beam centre set earlier in this session is forgotten (it would replace the
        calibrated centre without a word), the instrument profiles are read again and the frame shown is
        analysed again; without a frame, the next one opened uses the calibration."""
        self.view_model.clear_beam_center()
        self._refresh_profiles()
        self.run_analysis()

    # -- preferences -------------------------------------------------------------------

    def _apply_preferences(self) -> None:
        preferences = load_preferences(self.view_model.settings, self.view_model.profile_names())
        self._last_folder = preferences.last_folder
        self.view_model.set_mode(preferences.mode)
        self.view_model.set_profile(preferences.profile_name)
        self.view_model.set_incidence(preferences.incidence_deg)
        with QSignalBlocker(self.mode_combo):
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(preferences.mode)))
        if preferences.incidence_deg is not None:
            with QSignalBlocker(self.incidence_spin):
                self.incidence_spin.setValue(preferences.incidence_deg)
        # An αi kept from the last session is said once, with the first frame (``bindings/incidence.py``).
        self._incidence_notice_pending = preferences.incidence_deg is not None
        with QSignalBlocker(self.auto_export_check):
            self.auto_export_check.setChecked(preferences.auto_export)
        self._refresh_profiles()

    def _remember(self, *, last_folder: Optional[str] = None) -> None:
        if last_folder is not None:
            self._last_folder = last_folder
        state = self.view_model.state
        try:
            save_preferences(
                self.view_model.settings,
                AnalyzePreferences(
                    mode=getattr(self, "_kept_mode", None) or state.mode,  # not a switch the automatic run made
                    profile_name=state.profile_name,
                    incidence_deg=state.incidence_deg,
                    last_folder=self._last_folder,
                    auto_export=self.auto_export_check.isChecked(),
                ),
            )
        except OSError as exc:
            self._status(trf("Could not save the Analyze preferences: {error}", error=exc), "warning")


def _describe(geometry) -> str:
    return trf(
        "D = {distance} mm, λ = {wavelength} Å, αi = {incidence}°,\nbeam centre ({x}, {y}) px, pixel {px} × {py} µm",
        distance=f"{geometry.distance_m * 1e3:.1f}", wavelength=f"{geometry.wavelength_angstrom:.4f}",
        incidence=f"{geometry.incidence_deg:.3f}", x=f"{geometry.beam_center_x_px:.4f}",
        y=f"{geometry.beam_center_y_px:.4f}", px=f"{geometry.pixel_size_x_m * 1e6:.1f}",
        py=f"{geometry.pixel_size_y_m * 1e6:.1f}",
    )


__all__ = ["ProfileActionsMixin"]
