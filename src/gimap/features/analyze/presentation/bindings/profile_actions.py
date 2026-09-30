"""Instrument-profile choices, geometry entry and remembered preferences for Analyze."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QMessageBox

from ..geometry_dialog import GeometryDialog, geometry_defaults
from ..preferences import AnalyzePreferences, load_preferences, save_preferences
from ..views.analyze_page_view import AUTO_PROFILE_TEXT


class ProfileActionsMixin:
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
        self._status(f"Saved instrument profile “{name}”.", "ok")
        self.run_analysis()

    def _offer_previous_geometry(self) -> None:
        """Show “Use Previous Geometry” only when the former Fitting page stored one."""
        geometry = self.view_model.fitting_geometry()
        self.use_fitting_button.setVisible(geometry is not None)
        if geometry is not None:
            self.use_fitting_button.setToolTip(
                "Save the detector geometry of the former Cut & Fitting page as the "
                f"instrument profile of this detector:\n{_describe(geometry)}"
            )

    def _use_fitting_geometry(self) -> None:
        geometry = self.view_model.fitting_geometry()
        if geometry is None:
            self._status("No previous geometry is stored; enter the geometry instead.", "error")
            return
        name = self.view_model.suggested_profile_name()
        answer = QMessageBox.question(
            self,
            "Use previous geometry",
            f"Save profile “{name}” with the geometry of the former Cut & Fitting page?\n\n"
            f"{_describe(geometry)}",
        )
        if answer == QMessageBox.Yes:
            self.view_model.save_profile(name, geometry, source="former Cut & Fitting settings")
            self._profile_saved(name)

    def _enter_geometry(self) -> None:
        """Edit the geometry in use (or enter one); save it as a profile or delete that profile."""
        analysis = self.view_model.state.analysis
        if analysis is None:
            self._status("Open a frame first; the geometry is edited for its detector.", "warning")
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
        dialog = GeometryDialog(name, defaults, self, allow_delete=profile is not None)
        result = dialog.exec_()
        if result == GeometryDialog.DELETED and profile is not None:
            answer = QMessageBox.question(
                self, "Delete profile", f"Delete instrument profile \u201c{profile.name}\u201d?"
            )
            if answer == QMessageBox.Yes and self.view_model.delete_profile(profile.name):
                self._refresh_profiles()
                self._remember()
                self._status(f"Deleted instrument profile \u201c{profile.name}\u201d.", "ok")
                self.run_analysis()
            return
        if result == GeometryDialog.Accepted:
            try:
                geometry = dialog.geometry()
            except ValueError as exc:
                self._status(str(exc), "error")
                return
            self.view_model.save_profile(dialog.profile_name(), geometry, source="entered by hand")
            self._profile_saved(dialog.profile_name())

    def _open_calibration(self) -> None:
        if self._calibrate is None:
            return
        self._calibrate(self.view_model.current_path)
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
                    mode=state.mode,
                    profile_name=state.profile_name,
                    incidence_deg=state.incidence_deg,
                    last_folder=self._last_folder,
                    auto_export=self.auto_export_check.isChecked(),
                ),
            )
        except OSError as exc:
            self._status(f"Could not save the Analyze preferences: {exc}", "warning")


def _describe(geometry) -> str:
    return (
        f"D = {geometry.distance_m * 1e3:.1f} mm, λ = {geometry.wavelength_angstrom:.4f} Å, "
        f"αi = {geometry.incidence_deg:.3f}°,\n"
        f"beam centre ({geometry.beam_center_x_px:.2f}, {geometry.beam_center_y_px:.2f}) px, "
        f"pixel {geometry.pixel_size_x_m * 1e6:.1f} × {geometry.pixel_size_y_m * 1e6:.1f} µm"
    )


__all__ = ["ProfileActionsMixin"]
