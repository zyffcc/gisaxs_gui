"""Fit a standard on a worker thread and apply only the fit currently displayed."""

from __future__ import annotations

import math
from threading import Event

from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog

from src.gimap.app.presentation.i18n import apply_to, current_language, language_changed, tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.app.presentation.theme import set_role
from src.gimap.shared.numbers import format_number

from ..application.absolute_scale import StandardRequest, water_intensity
from .texts import message_text
from .views.absolute_scale_view import AbsoluteScaleView
from .views.analyze_page_view import FILE_FILTER

CURVE_FILTER = "Curve files (*.csv *.dat *.txt *.tsv);;All files (*)"
SAMPLE_REQUIRED = "Enter the sample thickness to use a standard calibration."
SAMPLE_PRECISION = "Enter a sample thickness from 0.000001 to 100000 mm with at most six decimal places."


def _value(edit, error, *, zero=False):
    try:
        value = float(edit.text())
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(error)
    return value


class AbsoluteScaleDialog(QDialog, AbsoluteScaleView):
    def __init__(self, preview, apply, parent=None, *, initial_path="", initial_frame=0, profile_name="",
                 sample_thickness_mm=None, saved_record=None, use_saved=None):
        super().__init__(parent)
        self.preview = preview
        self.apply = apply
        self.use_saved = use_saved
        self.saved_record = saved_record
        self.calibration = None
        self._status = ""
        self._stop = None
        # The pool may still be inside a detector read when the user closes the dialog.
        self.tasks = TaskRunner(QApplication.instance(), max_threads=1)
        self.setup_scale_view()
        self.source_path.setText(str(initial_path or ""))
        self.frame_number.setValue(initial_frame + 1)
        self.profile_label.setText(profile_name)
        if sample_thickness_mm is not None:
            self.sample_thickness.setText(format(sample_thickness_mm, ".12g"))
        self.saved_label.setVisible(bool(saved_record and use_saved))
        self.saved_button.setVisible(bool(saved_record and use_saved))
        self._wire()
        self._inputs_changed()
        self.refresh_language()

    def _wire(self):
        self.buttons.rejected.connect(self.reject)
        self.fit_button.clicked.connect(self._fit_or_stop)
        self.apply_button.clicked.connect(lambda: self._apply(False))
        self.saved_button.clicked.connect(lambda: self._apply(True))
        self.source_browse.clicked.connect(lambda: self._browse(False))
        self.reference_browse.clicked.connect(lambda: self._browse(True))
        for combo in (self.source_kind, self.reference_kind, self.source_unit, self.reference_unit):
            combo.currentIndexChanged.connect(self._inputs_changed)
        for edit in (self.source_path, self.reference_path, self.exposure, self.monitor, self.transmission,
                     self.std_thickness, self.ref_uncertainty, self.q_min, self.q_max):
            edit.textChanged.connect(self._inputs_changed)
        self.temperature.valueChanged.connect(self._inputs_changed)
        self.frame_number.valueChanged.connect(self._inputs_changed)
        self.sample_thickness.textChanged.connect(self._sample_edited)
        self.finished.connect(self._finished)
        language_changed().connect(self.refresh_language)

    def _browse(self, reference):
        edit = self.reference_path if reference else self.source_path
        is_curve = reference or self.source_kind.currentData() == "curve"
        title = "Choose the reference curve" if reference else "Choose the standard measurement"
        path, _ = QFileDialog.getOpenFileName(self, tr(title), edit.text(), tr(CURVE_FILTER if is_curve else FILE_FILTER))
        if path:
            edit.setText(path)
            if not reference and not is_curve:
                self.frame_number.setValue(1)

    def _inputs_changed(self):
        self._cancel()
        self.calibration = None
        self._status = ""
        self.source_options.setCurrentIndex(self.source_kind.currentIndex())
        water = self.reference_kind.currentData() == "water"
        self.temperature.setVisible(water)
        self.temperature_label.setVisible(water)
        self.water_level.setVisible(water)
        for widget in (self.ref_label, self.ref_picker, self.ref_unit_label, self.reference_unit):
            widget.setVisible(not water)
        self.overlay.set_curves([])
        self.residual.set_curves([])
        self._render()

    def _sample_edited(self):
        if self._status in (SAMPLE_REQUIRED, SAMPLE_PRECISION):
            self._status = ""
        self._render()

    def _request(self):
        source = self.source_path.text().strip()
        if not source:
            raise ValueError("Choose the standard measurement.")
        ref_path = None
        if self.reference_kind.currentData() == "curve":
            ref_path = self.reference_path.text().strip()
            if not ref_path:
                raise ValueError("Choose a reference curve file.")
        bounds = None
        if self.q_min.text().strip() or self.q_max.text().strip():
            error = "Enter both q limits, or clear both to use the full overlap."
            bounds = (_value(self.q_min, error, zero=True), _value(self.q_max, error))
            if not bounds[0] < bounds[1]:
                raise ValueError("The upper q limit must exceed the lower limit.")
        return StandardRequest(
            source_kind=self.source_kind.currentData(), source_path=source,
            source_q_unit=self.source_unit.currentText(), frame_index=self.frame_number.value() - 1,
            exposure_s=_value(self.exposure, "The exposure time must be positive."),
            monitor=_value(self.monitor, "The monitor value must be positive."),
            transmission=_value(self.transmission, "The transmission must be above 0 and at most 1."),
            thickness_mm=_value(self.std_thickness, "Enter the thickness of the standard in mm."),
            reference_path=ref_path, reference_q_unit=self.reference_unit.currentText(),
            temperature_c=self.temperature.value(), q_range=bounds,
            reference_fractional_sigma=_value(self.ref_uncertainty, "Reference uncertainty must be non-negative.",
                                              zero=True) / 100.0,
        )

    def _fit_or_stop(self):
        if self._stop is not None:
            self._cancel()
            self._status = "Fit stopped."
            self._render()
            return
        self._inputs_changed()
        try:
            request = self._request()
        except ValueError as error:
            self._error(str(error))
            return
        stop = self._stop = Event()
        self._status = ""
        self._render()
        worker = self.preview
        self.tasks.submit("fit", lambda: worker(request, cancelled=stop.is_set),
                          on_done=self._show_fit, on_error=self._error)

    def _cancel(self):
        if self._stop is not None:
            self._stop.set()
            self.tasks.cancel("fit")
            self._stop = None

    def _show_fit(self, result):
        self.calibration = result
        self._stop = None
        self._status = ""
        self._draw_fit()
        self._render()

    def _draw_fit(self):
        fit = self.calibration.fit
        self.overlay.set_curves([("Scaled standard", fit.q, fit.scaled), ("Reference", fit.q, fit.reference)],
                                markers=[True, False])
        self.residual.set_curves([("Residual", fit.q, fit.residual), ("0", [fit.q[0], fit.q[-1]], [0, 0])],
                                 markers=[True, False], legend=[False, False])

    def _error(self, message, _details=None):
        self._stop = None
        self._status = message
        self._render()

    def _apply(self, saved):
        if not saved and self.calibration is None:
            return
        try:
            thickness = _value(self.sample_thickness, SAMPLE_REQUIRED)
            if not 1e-6 <= thickness <= 1e5 or round(thickness, 6) != thickness:
                raise ValueError(SAMPLE_PRECISION)
            if saved:
                self.use_saved(thickness)
            else:
                self.apply(self.calibration, thickness)
        except (ValueError, OSError, RuntimeError) as error:
            self._error(str(error))
            return
        self.accept()

    def _render(self):
        running = self._stop is not None
        self.fit_button.setText(tr("Stop" if running else "Fit Scale"))
        self.apply_button.setEnabled(self.calibration is not None and not running)
        self.saved_button.setEnabled(self.use_saved is not None and not running)
        self.feedback.setText(tr("Reducing and fitting the standard…") if running else message_text(self._status))
        set_role(self.feedback, "warning" if self._status else "muted")
        summary = ""
        self.water_level.setText(f"dΣ/dΩ = {water_intensity(self.temperature.value()):.6g} cm⁻¹")
        if self.calibration is not None:
            fit = self.calibration.fit
            summary = trf("A_abs = {estimate} (1σ)\n{points} points from {low} to {high} Å⁻¹",
                          estimate=format_number(fit.factor, fit.sigma_factor), points=len(fit.q),
                          low=format_number(fit.q[0]), high=format_number(fit.q[-1]))
        self.fit_summary.setText(summary)

    def refresh_language(self):
        apply_to(self, current_language())
        if self.saved_record and self.use_saved:
            record = self.saved_record
            try:
                if not (math.isfinite(record["factor"]) and math.isfinite(record["sigma_factor"])):
                    raise ValueError("Invalid standard calibration.")
                summary = trf("Saved A_abs = {estimate} · {standard} · {date}",
                              estimate=format_number(record["factor"], record["sigma_factor"]), date=record["date_utc"],
                              standard=tr("Water") if record["standard"] == "Water" else record["standard"])
            except (KeyError, TypeError, ValueError):
                summary = tr("Invalid standard calibration.")
                self.use_saved = None
            self.saved_label.setText(summary)
        if self.calibration is not None:
            self._draw_fit()
        self._render()

    def _finished(self):
        self._cancel()
        runner = self.tasks
        if runner.is_busy():
            runner.busy_changed.connect(lambda busy: None if busy else runner.deleteLater())
        else:
            runner.deleteLater()
