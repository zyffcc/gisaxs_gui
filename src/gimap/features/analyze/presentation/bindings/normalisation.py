"""The dark frame, the Normalisation section and the Lorentz factor of Mask & Corrections.

Off by default. The section always says which header values the frame shown offers (exposure time, monitors), and
after an analysis by what the frame was multiplied, each term with its source (``domain/normalisation.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr, trf

from ...application import Normalisation
from ..texts import message_text
from ..views.analyze_page_view import FILE_FILTER

HEADER_ITEM = "header:"
"""Data of a monitor-combo item that takes the header value after the colon."""
SYMBOLS = {"exposure_time_s": ("t", " s"), "monitor": ("M", ""), "transmission": ("T", ""),
           "thickness_mm": ("d", " mm"), "thickness_cm": ("d", " cm"), "absolute_factor": ("A", "")}


def _number(text: str) -> Optional[float]:
    try:
        value = float(str(text).replace(",", "."))
    except ValueError:
        return None
    return value if value > 0 else None


class NormalisationMixin:
    """Needs the widgets of ``NormalisationView``, ``view_model``, ``_options_changed``, ``_status``."""

    def absolute_scale_dialog(self) -> None:
        from ..absolute_scale_dialog import AbsoluteScaleDialog

        vm = self.view_model
        try:
            profile = vm.absolute_scale_profile()
            preview = vm.standard_calibrator()
        except ValueError as exc:
            self._status(message_text(exc), "warning")
            return

        def apply(record, thickness_mm, *, save):
            vm.apply_standard_calibration(profile, record, thickness_mm, save=save)
            self._refresh_profiles()
            self._sync_normalisation_widgets()
            self._options_changed()
            self._status(trf("Applied the absolute scale from profile “{name}”.", name=profile.name), "ok")

        saved = profile.standard_calibration
        dialog = AbsoluteScaleDialog(
            preview, lambda result, thickness: apply(result.record, thickness, save=True), self,
            initial_path=str(vm.current_path or ""), initial_frame=vm.state.frame_index, profile_name=profile.name,
            sample_thickness_mm=vm.state.corrections.normalisation.thickness_mm, saved_record=saved,
            use_saved=lambda thickness: apply(saved, thickness, save=False),
        )
        dialog.exec_()
        dialog.deleteLater()

    def _connect_normalisation(self) -> None:
        self.dark_button.clicked.connect(self._choose_dark)
        self.dark_clear_button.clicked.connect(self._clear_dark)
        for spin in (self.dark_scale_spin, self.dark_frame_spin):
            spin.valueChanged.connect(self._dark_changed)
        for combo in (self.exposure_combo, self.monitor_combo):
            combo.activated.connect(self._normalisation_changed)
        for spin in (self.exposure_spin, self.transmission_spin, self.thickness_spin):
            spin.valueChanged.connect(self._normalisation_changed)
        for check in (self.transmission_check, self.thickness_check, self.absolute_check):
            check.toggled.connect(self._normalisation_changed)
        for edit in (self.monitor_edit, self.absolute_edit):
            edit.editingFinished.connect(self._normalisation_changed)
        self.lorentz_check.toggled.connect(self._lorentz_changed)
        self._sync_normalisation_widgets()

    # -- the dark frame -------------------------------------------------------------------

    def _choose_dark(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Dark frame"), self._last_folder, FILE_FILTER)
        if path:
            self.apply_dark(path)

    def apply_dark(self, path: str | Path) -> None:
        """Subtract this dark frame (with the scale and frame of the fields) from every frame, before the
        background."""
        self.view_model.set_dark(path, frame=self.dark_frame_spin.value(), scale=self.dark_scale_spin.value())
        self._show_dark()
        self._options_changed()

    def _clear_dark(self) -> None:
        self.view_model.set_dark(None)
        self._show_dark()
        self._options_changed()

    def _dark_changed(self, *_args) -> None:
        path = self.view_model.state.corrections.dark_path
        if path:
            self.apply_dark(path)

    def _show_dark(self) -> None:
        path = self.view_model.state.corrections.dark_path
        self.dark_label.setText(Path(path).name if path else tr("No dark frame"))
        self.dark_label.setToolTip(path or "")

    # -- the normalisation ------------------------------------------------------------------

    def _read_normalisation(self) -> Normalisation:
        """The normalisation the fields say (``ValueError`` with the reason when a value is not usable)."""
        exposure = self.exposure_combo.currentData() or "off"
        monitor_data = str(self.monitor_combo.currentData() or "off")
        monitor, key, value = monitor_data, None, None
        if monitor_data.startswith(HEADER_ITEM):
            monitor, key = "header", monitor_data[len(HEADER_ITEM):]
        elif monitor_data == "value":
            value = _number(self.monitor_edit.text())
            if value is None:
                raise ValueError("{what} must be a positive number.".format(what="The monitor value"))
        absolute = None
        if self.absolute_check.isChecked():
            absolute = _number(self.absolute_edit.text())
            if absolute is None:
                raise ValueError("{what} must be a positive number.".format(what="The absolute-scale factor"))
        calibration = self._displayed_calibration
        if calibration is not None and absolute != calibration["factor"]:
            calibration = None
        return Normalisation(
            exposure=exposure, exposure_s=self.exposure_spin.value() if exposure == "value" else None,
            monitor=monitor, monitor_key=key, monitor_value=value,
            transmission=self.transmission_spin.value() if self.transmission_check.isChecked() else None,
            thickness_mm=self.thickness_spin.value() if self.thickness_check.isChecked() else None,
            absolute_factor=absolute, standard_calibration=calibration,
        )

    def _normalisation_changed(self, *_args) -> None:
        self._enable_normalisation_fields()
        try:
            normalisation = self._read_normalisation()
        except ValueError as exc:
            if self.view_model.state.corrections.normalisation.standard_calibration and not self.thickness_check.isChecked():
                self._sync_normalisation_widgets()
            self._status(trf("Normalisation not changed: {error}", error=message_text(exc)), "error")
            return
        if normalisation == self.view_model.state.corrections.normalisation:
            return
        if self.absolute_check.isChecked():
            self._displayed_calibration = normalisation.standard_calibration
        self.view_model.set_normalisation(normalisation)
        self._options_changed()

    def _enable_normalisation_fields(self) -> None:
        self.exposure_spin.setEnabled(self.exposure_combo.currentData() == "value")
        self.monitor_edit.setEnabled(self.monitor_combo.currentData() == "value")
        self.transmission_spin.setEnabled(self.transmission_check.isChecked())
        self.thickness_spin.setEnabled(self.thickness_check.isChecked())
        self.absolute_edit.setEnabled(self.absolute_check.isChecked())

    def _lorentz_changed(self, on: bool) -> None:
        self.view_model.set_lorentz(on)
        self._options_changed()

    def _fill_monitors(self, monitors: dict) -> None:
        """Off, Typed and each monitor of the header shown, plus the header value the state names when the frame
        shown lacks it (said so)."""
        state = self.view_model.state.corrections.normalisation
        with QSignalBlocker(self.monitor_combo):
            self.monitor_combo.clear()
            self.monitor_combo.addItem(tr("Off"), "off")
            self.monitor_combo.addItem(tr("Typed"), "value")
            names = dict(monitors)
            if state.monitor == "header" and state.monitor_key not in names:
                names[state.monitor_key] = None
            for key, value in names.items():
                text = (trf("Header: {key} = {value}", key=key, value=f"{value:.6g}") if value is not None
                        else trf("Header: {key} (not in this frame)", key=key))
                self.monitor_combo.addItem(text, HEADER_ITEM + str(key))
            current = HEADER_ITEM + str(state.monitor_key) if state.monitor == "header" else state.monitor
            self.monitor_combo.setCurrentIndex(max(0, self.monitor_combo.findData(current)))

    def _show_normalisation(self) -> None:
        """What the header of the frame shown offers, and by what the frame was multiplied (after an analysis)."""
        analysis = self.view_model.state.analysis
        header = getattr(analysis, "header_values", None) or {}
        exposure, monitors = header.get("exposure_time_s"), dict(header.get("monitors") or {})
        items = ([trf("exposure {value} s", value=f"{float(exposure):g}")] if exposure else []) + [
            f"{key} = {value:.6g}" for key, value in monitors.items()]
        self.header_values_label.setText(trf("In the file header: {items}", items="; ".join(items)) if items else tr(
            "The file header gives no exposure time and no monitor value.") if analysis is not None else "")
        self.header_values_label.setVisible(analysis is not None)  # (no frame: nothing to say of a header)
        self._fill_monitors(monitors)
        use = getattr(analysis, "normalisation", None)
        if use is None:
            self.normalisation_status.setText("")
            self.normalisation_status.hide()
            return
        terms = []
        for name, value, source in use.terms:
            symbol, unit = SYMBOLS.get(name, (name, ""))
            where = tr("file header") if source.startswith("header") else tr(source)
            terms.append(f"{symbol} = {value:.6g}{unit} ({where})")
        self.normalisation_status.setText(trf("Multiplied by {factor}: {terms}; intensity in {unit}",
                                              factor=f"{use.factor:.6g}", terms=", ".join(terms), unit=use.unit))
        self.normalisation_status.show()

    def _sync_normalisation_widgets(self) -> None:
        """The dark frame, the normalisation and the Lorentz check as the state says (a settings file, Undo)."""
        corrections = self.view_model.state.corrections
        norm = corrections.normalisation
        if getattr(self, "_displayed_calibration", None) is not None and norm.absolute_factor is None:
            self.absolute_edit.setText("1")
        self._displayed_calibration = norm.standard_calibration
        for spin, value in ((self.dark_scale_spin, corrections.dark_scale), (self.dark_frame_spin, corrections.dark_frame),
                            (self.exposure_spin, norm.exposure_s), (self.transmission_spin, norm.transmission),
                            (self.thickness_spin, norm.thickness_mm)):
            if value is not None:
                with QSignalBlocker(spin):
                    spin.setValue(value)
        for check, value in ((self.transmission_check, norm.transmission is not None),
                             (self.thickness_check, norm.thickness_mm is not None),
                             (self.absolute_check, norm.absolute_factor is not None),
                             (self.lorentz_check, corrections.lorentz)):
            with QSignalBlocker(check):
                check.setChecked(bool(value))
        for edit, value in ((self.monitor_edit, norm.monitor_value), (self.absolute_edit, norm.absolute_factor)):
            if value is not None:
                edit.setText(str(float(value)))  # a number as typed back (not a text to translate)
        with QSignalBlocker(self.exposure_combo):
            self.exposure_combo.setCurrentIndex(max(0, self.exposure_combo.findData(norm.exposure)))
        self._show_dark()
        self._show_normalisation()
        self._enable_normalisation_fields()


__all__ = ["HEADER_ITEM", "NormalisationMixin"]
