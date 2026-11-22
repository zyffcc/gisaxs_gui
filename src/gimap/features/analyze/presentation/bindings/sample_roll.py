"""Apply a sample roll to Analyze and measure it from the current observed pixels."""

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

from ...application.incidence_record import frame_incidence_record, incidence_choice, nominal_incidence
from ...application.sample_roll import MATERIALS, critical_angle_deg, measure_frame_angles
from .incidence_text import incidence_text


def _estimate_text(estimate):
    if not estimate:
        return tr("Unavailable")
    return f"{estimate['deg']:.2f} ± {estimate['error_deg']:.2f}°"


def _measurement_text(measured, kind):
    if not measured:
        return ""
    is_rod = measured.get("symmetry_feature", "specular rod" if kind == "gisaxs" else "fibre rings") == "specular rod"
    lines = [trf("Horizon  {estimate}", estimate=_estimate_text(measured.get("horizon"))),
             trf("Specular rod  {estimate}" if is_rod else "Ring symmetry  {estimate}",
                 estimate=_estimate_text(measured.get("symmetry")))]
    method = measured.get("method")
    if measured.get("deg") is not None:
        names = {"combined": "Combined estimate", "horizon": "Horizon estimate", "symmetry": "Symmetry estimate"}
        lines.append(f"{tr(names.get(method, 'Estimate'))}  {_estimate_text(measured)}")
        if measured.get("origin") == "automatic" and not measured.get("significant"):
            lines.append(tr("Below the automatic threshold."))
    elif method == "disagreement":
        lines.append(tr("Estimates disagree. Current roll retained."))
    else:
        lines.append(tr("No roll could be measured."))
    return "\n".join(lines)


class SampleRollMixin:
    _roll_pending = False
    _roll_attempt = None

    def _connect_sample_roll(self):
        self.sample_roll_spin.valueChanged.connect(self._roll_typed)
        self.incidence_lock_check.toggled.connect(self._lock_incidence)
        self.yoneda_angle_spin.valueChanged.connect(self._change_yoneda_angle)
        for key, material in MATERIALS.items():
            self.yoneda_material_combo.addItem("Plate glass" if key == "glass" else material.label, key)
        self.yoneda_material_combo.currentIndexChanged.connect(self._change_yoneda_material)
        # Entering the displayed value (including zero) can intentionally fix it for automatic analysis.
        self._roll_edited = False
        self.sample_roll_spin.lineEdit().textEdited.connect(self._mark_roll_edit)
        self.sample_roll_spin.editingFinished.connect(self._finish_roll_edit)
        self.roll_reset_action.triggered.connect(lambda: self._roll_typed(None))
        self.measure_roll_button.clicked.connect(self._measure_roll)
        self.roll_step_button.clicked.connect(self._show_roll_step)
        self._sync_sample_roll_widgets()

    def _show_roll_step(self):
        self.show_steps_action.setChecked(True)
        self.show_step("geometry")
        self.sample_roll_spin.setFocus()

    def _mark_roll_edit(self, _text):
        self._roll_edited = True

    def _finish_roll_edit(self):
        if self._roll_edited:
            self._roll_typed(self.sample_roll_spin.value())

    def _roll_typed(self, value):
        self._roll_edited = False
        self.view_model.set_sample_roll(value)
        self.view_model.set_incidence_measurement(None)
        self._sync_sample_roll_widgets()
        self.run_analysis()

    def _lock_incidence(self, locked):
        model = self.view_model
        model.state.incidence_locked = locked
        record = model.state.incidence_measurement
        if record:
            reference = record.get("override_deg", record["reference_deg"])
            degrees, decision = incidence_choice(reference, record, locked=locked)
            model.set_incidence_measurement(dict(record, applied_deg=degrees, decision=decision))
        self.run_analysis()

    def _change_yoneda_angle(self, degrees):
        self.view_model.state.yoneda_critical_deg = degrees or None
        self.view_model.set_incidence_measurement(None)
        self.run_analysis()

    def _change_yoneda_material(self):
        self.view_model.state.yoneda_material = self.yoneda_material_combo.currentData()
        self.view_model.set_incidence_measurement(None)
        self._sync_sample_roll_widgets()
        self.run_analysis()

    def _show_yoneda_source(self, analysis):
        state = self.view_model.state
        material = state.yoneda_material
        self.yoneda_angle_row.setVisible(bool(analysis and analysis.kind == "gisaxs"))
        with QSignalBlocker(self.yoneda_material_combo):
            self.yoneda_material_combo.setCurrentIndex(max(0, self.yoneda_material_combo.findData(material)))
        with QSignalBlocker(self.yoneda_angle_spin):
            self.yoneda_angle_spin.setValue(state.yoneda_critical_deg or 0.)
        self.yoneda_angle_spin.setVisible(not material)
        self.yoneda_material_value.setVisible(bool(material))
        if material and analysis and analysis.kind == "gisaxs" and analysis.geometry:
            wavelength = analysis.geometry.wavelength_angstrom
            try:
                angle = critical_angle_deg(material, wavelength / 10.)
                text = trf("{angle}° at {wavelength} Å", angle=f"{angle:.4f}", wavelength=f"{wavelength:.4g}")
            except ValueError:
                text = tr("Outside the optical table")
            self.yoneda_material_value.setText(text)

    def _cancel_roll_measurement(self):
        self._roll_attempt = None
        if self._roll_pending:
            self.tasks.cancel("sample-roll")
            self._roll_pending = False
            self._sync_sample_roll_widgets()

    def _sync_sample_roll_widgets(self):
        model = self.view_model
        analysis = model.state.analysis
        request = model.request() if model.current_path is not None else None
        value = request.sample_roll_deg if request else model.state.sample_roll_deg
        measured = request.sample_roll_measurement if request else model.state.sample_roll_measurement
        profile = analysis.resolution.profile if analysis else None
        self.roll_reset_action.setEnabled(value is not None or bool(measured))
        if value is None:
            value = profile.geometry.sample_roll_deg if profile else 0.0
            if measured is None and profile:
                measured = profile.sample_roll_measurement
        if self._roll_attempt is not None:
            measured = self._roll_attempt
        with QSignalBlocker(self.sample_roll_spin):
            self.sample_roll_spin.setValue(value)
        with QSignalBlocker(self.incidence_lock_check):
            self.incidence_lock_check.setChecked(model.state.incidence_locked)
        self.roll_step_button.setText(f"ψ {value:.4f}".rstrip("0").rstrip(".") + "°")
        kind = analysis.kind if analysis else None
        self._show_yoneda_source(analysis)
        self.measure_roll_button.setText(tr("Cancel measurement" if self._roll_pending else "Measure from frame"))
        self.measure_roll_button.setEnabled(bool(analysis and analysis.geometry and kind in ("gisaxs", "giwaxs")))
        text = tr("Measuring ψ and αi…") if self._roll_pending else _measurement_text(measured, kind)
        self.roll_estimate_label.setText(text)
        self.roll_estimate_label.setVisible(bool(text))
        incidence = incidence_text(analysis.incidence_measurement if analysis and not self._roll_pending else None, kind)
        self.incidence_estimate_label.setText(incidence)
        self.incidence_estimate_label.setVisible(bool(incidence))

    def _measure_roll(self):
        if self._roll_pending:
            self._cancel_roll_measurement()
            return
        model = self.view_model
        analysis, request = model.state.analysis, model.request()
        if not analysis or not analysis.geometry or request != analysis.request:
            return
        self._roll_pending = True
        self._sync_sample_roll_widgets()

        def measured(results):
            self._roll_pending = False
            if model.state.analysis is analysis and model.request() == request:
                result, incidence = results
                record = dict(result.as_dict(), origin="measured", path=str(analysis.path), frame_index=analysis.frame_index,
                              summed_frames=[[str(path), index] for path, index in analysis.summed_frames])
                if result.deg is not None:
                    model.set_sample_roll(result.deg, record)
                reference, source = nominal_incidence(analysis)
                estimates = incidence.as_dict()
                if estimates.get("critical_angle_deg") and not estimates.get("critical_angle_source"):
                    estimates["critical_angle_source"] = ("Yoneda critical angle" if request.yoneda_critical_deg
                                                        else "film refraction setting")
                applied, decision = incidence_choice(reference, estimates, locked=model.state.incidence_locked)
                model.set_incidence_measurement(frame_incidence_record(
                    analysis, estimates, origin="measured", reference_deg=reference, reference_source=source,
                    applied_deg=applied, decision=decision))
                self.run_analysis()
                if result.deg is None:
                    self._roll_attempt = record
            self._sync_sample_roll_widgets()

        def failed(message, _details):
            self._roll_pending = False
            self._sync_sample_roll_widgets()
            self._status(trf("Could not measure ψ and αi. {error}", error=tr(message)), "warning")

        self.tasks.submit("sample-roll", lambda: measure_frame_angles(
            analysis, model.frames, use_resolved_roll=True, critical_angle_deg=request.yoneda_critical_deg,
            yoneda_material=request.yoneda_material),
                          on_done=measured, on_error=failed)
