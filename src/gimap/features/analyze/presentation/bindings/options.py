"""Options panel of Analyze: corrections and GIWAXS cuts, applied as they change."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog

from ..views.analyze_page_view import FILE_FILTER

APPLY_DELAY_MS = 250


class OptionsMixin:
    """Own the Options panel; every change re-runs the analysis once it settles."""

    def _connect_options(self) -> None:
        self._options_timer = QTimer(self)
        self._options_timer.setSingleShot(True)
        self._options_timer.setInterval(APPLY_DELAY_MS)
        self._options_timer.timeout.connect(self.run_analysis)
        self.sum_spin.setValue(self.view_model.sum_count)
        self.sum_spin.valueChanged.connect(self._sum_changed)
        self.gap_guard_spin.setValue(self.view_model.state.corrections.gap_guard_px)
        self.gap_guard_spin.valueChanged.connect(self._gap_guard_changed)
        self.bad_pixels_check.setChecked(self.view_model.state.corrections.bad_pixels)
        self.bad_pixels_check.toggled.connect(self._bad_pixels_changed)
        self.background_button.clicked.connect(self._choose_background)
        self.background_clear_button.clicked.connect(self._clear_background)
        for spin in (self.background_scale_spin, self.background_frame_spin):
            spin.valueChanged.connect(self._background_changed)
        for widget in (self.minimum_check, self.maximum_check):
            widget.toggled.connect(self._limits_changed)
        for spin in (self.minimum_spin, self.maximum_spin):
            spin.valueChanged.connect(self._limits_changed)
        for widget in (self.solid_angle_check, self.polarization_check, self.film_check):
            widget.toggled.connect(self._intensity_changed)
        for spin in (self.polarization_spin, self.film_thickness_spin, self.attenuation_spin):
            spin.valueChanged.connect(self._intensity_changed)
        for spin in (self.in_plane_spin, self.out_of_plane_spin):
            spin.valueChanged.connect(self._sector_widths_changed)
        self.bins_spin.valueChanged.connect(self._bins_changed)
        self.x_axis_control.activated.connect(self._x_axis_changed)
        self.sector_check.toggled.connect(self._sector_changed)
        for spin in (self.sector_chi_min, self.sector_chi_max, self.sector_q_min, self.sector_q_max):
            spin.valueChanged.connect(self._sector_changed)
        self.box_check.toggled.connect(self._box_changed)
        for spin in (self.box_par_min, self.box_par_max, self.box_qz_min, self.box_qz_max):
            spin.valueChanged.connect(self._box_changed)
        self.detector_view.boxChanged.connect(self._box_dragged)
        self.sector_grid.setEnabled(False)
        self.box_grid.setEnabled(False)
        for spin in (self.polarization_spin, self.film_thickness_spin, self.attenuation_spin):
            spin.setEnabled(False)

    def _options_changed(self) -> None:
        self._options_timer.start()

    def _sum_changed(self, value: int) -> None:
        self.view_model.set_sum_count(value)
        self._options_changed()

    # -- corrections -------------------------------------------------------------------

    def _gap_guard_changed(self, value: int) -> None:
        self.view_model.set_gap_guard(value)
        self._options_changed()

    def _bad_pixels_changed(self, enabled: bool) -> None:
        self.view_model.set_bad_pixels(enabled)
        self._options_changed()

    def _choose_background(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Background frame", self._last_folder, FILE_FILTER
        )
        if path:
            self._apply_background(path)

    def _apply_background(self, path: str) -> None:
        self.view_model.set_background(
            path,
            frame=self.background_frame_spin.value(),
            scale=self.background_scale_spin.value(),
        )
        self.background_label.setText(Path(path).name)
        self.background_label.setToolTip(path)
        self._options_changed()

    def _clear_background(self) -> None:
        self.view_model.set_background(None)
        self.background_label.setText("No background")
        self.background_label.setToolTip("")
        self._options_changed()

    def _background_changed(self, *_args) -> None:
        path = self.view_model.state.corrections.background_path
        if path:
            self._apply_background(path)

    def _limits_changed(self, *_args) -> None:
        self.view_model.set_valid_range(
            self.minimum_spin.value() if self.minimum_check.isChecked() else None,
            self.maximum_spin.value() if self.maximum_check.isChecked() else None,
        )
        self._options_changed()

    def _intensity_changed(self, *_args) -> None:
        film = self.film_check.isChecked()
        self.polarization_spin.setEnabled(self.polarization_check.isChecked())
        self.film_thickness_spin.setEnabled(film)
        self.attenuation_spin.setEnabled(film)
        self.view_model.set_intensity_corrections(
            solid_angle=self.solid_angle_check.isChecked(),
            polarization=self.polarization_spin.value() if self.polarization_check.isChecked() else None,
            film_thickness_nm=self.film_thickness_spin.value() if film else None,
            attenuation_length_um=self.attenuation_spin.value() if film else None,
        )
        self._options_changed()

    # -- GIWAXS ------------------------------------------------------------------------

    def _sector_widths_changed(self, *_args) -> None:
        self.view_model.set_sector_widths(self.in_plane_spin.value(), self.out_of_plane_spin.value())
        self._options_changed()

    def _bins_changed(self, value: int) -> None:
        self.view_model.set_radial_bins(value or None)
        self._options_changed()

    def _x_axis_changed(self, _index: int) -> None:
        self.view_model.set_x_axis(self.x_axis_control.currentData())
        self._options_changed()

    def _sector_changed(self, *_args) -> None:
        enabled = self.sector_check.isChecked()
        self.sector_grid.setEnabled(enabled)
        if not enabled:
            self.view_model.set_sector(None)
        else:
            q_min = self.sector_q_min.value() or None
            q_max = self.sector_q_max.value() or None
            self.view_model.set_sector(
                (self.sector_chi_min.value(), self.sector_chi_max.value()), (q_min, q_max)
            )
        self._options_changed()

    def _box_changed(self, *_args) -> None:
        enabled = self.box_check.isChecked()
        self.box_grid.setEnabled(enabled)
        if not enabled:
            self.view_model.set_box(None)
        else:
            self.view_model.set_box(
                (self.box_par_min.value(), self.box_par_max.value()),
                (self.box_qz_min.value(), self.box_qz_max.value()),
            )
        self._options_changed()

    def _box_dragged(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """The rectangle moved on the q map: take its corners as the q box."""
        spins = (self.box_par_min, self.box_par_max, self.box_qz_min, self.box_qz_max)
        for spin, value in zip(spins, (x0, x1, y0, y1)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        if not self.box_check.isChecked():
            self.box_check.setChecked(True)  # also applies the box
            return
        self._box_changed()

    def _sync_options(self, kind) -> None:
        """GIWAXS cuts only make sense for a GIWAXS reduction; the GISAXS ones for GISAXS."""
        self.giwaxs_section.setEnabled(kind == "giwaxs")
        self.giwaxs_section.setVisible(kind != "gisaxs")
        self.regions_panel.setVisible(kind == "giwaxs")
        self.gisaxs_cuts.setVisible(kind == "gisaxs")
        self.mirror_fill_check.setEnabled(kind != "gisaxs")
        self.intensity_section.setEnabled(kind != "gisaxs")


__all__ = ["OptionsMixin"]
