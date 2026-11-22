"""Keep correction and uncertainty controls in step with profiles, loaded setups and Undo."""

from contextlib import ExitStack
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.detector_io import DETECTOR_SUFFIXES

from ..texts import message_text


class ReductionErrorsMixin:
    def _connect_reduction_errors(self):
        for check in (self.background_dark_check, self.background_dark_auto, self.flat_dark_check):
            check.toggled.connect(self._reference_options_changed)
        for spin in (self.background_dark_spin, self.flat_floor_spin):
            spin.valueChanged.connect(self._reference_options_changed)
        self.use_profile_response_check.toggled.connect(self._response_source_changed)
        self.error_model_combo.activated.connect(self._error_model_chosen)
        self.sigma_button.clicked.connect(self._choose_sigma)
        self.save_response_button.clicked.connect(self._save_response)
        for spin in (self.gain_spin, self.offset_spin):
            spin.valueChanged.connect(self._set_response)
        self.pool_scatter_check.toggled.connect(self._curve_errors_changed)
        self.poisson_variance_combo.activated.connect(self._curve_errors_changed)

    def _reference_options_changed(self, *_args):
        automatic = self.background_dark_auto.isChecked()
        self.view_model.set_reference_options(
            background_dark_scale=None if automatic else self.background_dark_spin.value(),
            background_already_dark_subtracted=self.background_dark_check.isChecked(),
            flat_already_dark_subtracted=self.flat_dark_check.isChecked(),
            flat_field_floor=self.flat_floor_spin.value(),
        )
        self.background_dark_auto.setEnabled(not self.background_dark_check.isChecked())
        self.background_dark_spin.setEnabled(self.background_dark_auto.isEnabled() and not automatic)
        self.background_dark_spin.setVisible(not automatic)
        self._options_changed()

    def _curve_errors_changed(self, *_args):
        self.view_model.set_curve_errors(self.pool_scatter_check.isChecked(), self.poisson_variance_combo.currentData())
        self._options_changed()

    def _response_source_changed(self, use_profile):
        if use_profile:
            self.view_model.set_detector_response(use_profile=True)
        else:
            response = self.view_model.detector_response()
            self.view_model.set_detector_response(
                use_profile=False, gain=response.counts_per_adu, offset=response.offset_adu,
                model=response.error_model, sigma_path=response.sigma_path,
            )
        self._sync_reduction_errors()
        self._options_changed()

    def _error_model_chosen(self, _index):
        if self.error_model_combo.currentData() == "sigma" and not self.view_model.state.corrections.sigma_path:
            self._choose_sigma()
        else:
            self._set_response()

    def _choose_sigma(self):
        patterns = " ".join("*" + suffix for suffix in (*DETECTOR_SUFFIXES, ".npy"))
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose a σ frame"), self._last_folder,
                                              trf("σ frames ({patterns});;All files (*)", patterns=patterns))
        if path:
            self._set_response(sigma_path=path)
        else:
            self._sync_reduction_errors()

    def _set_response(self, *_args, sigma_path=None):
        self.view_model.set_detector_response(
            use_profile=False, gain=self.gain_spin.value(), offset=self.offset_spin.value(),
            model=self.error_model_combo.currentData(),
            sigma_path=sigma_path or self.view_model.state.corrections.sigma_path,
        )
        self._sync_reduction_errors()
        self._options_changed()

    def _save_response(self):
        try:
            self.view_model.save_detector_response()
        except (ValueError, OSError) as exc:
            self._status(message_text(exc), "warning")
            return
        self._refresh_profiles()
        self._sync_reduction_errors()
        self._status(tr("Detector settings saved to the instrument profile."), "ok")
        self._options_changed()

    def _sync_reduction_errors(self):
        settings = self.view_model.state.corrections
        response = self.view_model.detector_response()
        spins = (
            (self.gain_spin, response.counts_per_adu), (self.offset_spin, response.offset_adu),
            (self.flat_floor_spin, settings.flat_field_floor),
            (self.background_dark_spin, settings.background_dark_scale if settings.background_dark_scale is not None
             else max(0, settings.dark_scale)),
        )
        checks = (
            (self.background_dark_check, settings.background_already_dark_subtracted),
            (self.background_dark_auto, settings.background_dark_scale is None),
            (self.flat_dark_check, settings.flat_already_dark_subtracted),
            (self.use_profile_response_check, settings.use_profile_response),
            (self.pool_scatter_check, settings.pooled_scatter),
        )
        combos = ((self.error_model_combo, response.error_model), (self.poisson_variance_combo, settings.poisson_variance))
        with ExitStack() as blockers:
            for widget, _ in (*spins, *checks, *combos):
                blockers.enter_context(QSignalBlocker(widget))
            for spin, value in spins:
                spin.setValue(value)
            for check, value in checks:
                check.setChecked(value)
            for combo, value in combos:
                combo.setCurrentIndex(combo.findData(value))
        for widget in (self.gain_spin, self.offset_spin, self.error_model_combo,
                       self.sigma_button, self.save_response_button):
            widget.setEnabled(not settings.use_profile_response)
        for widget in (self.sigma_label, self.sigma_button):
            widget.setVisible(response.error_model == "sigma")
        self.sigma_label.setText(Path(response.sigma_path).name if response.sigma_path else tr("No σ frame"))
        self.sigma_label.setToolTip(response.sigma_path or "")
        self.background_dark_auto.setEnabled(not settings.background_already_dark_subtracted)
        self.background_dark_spin.setEnabled(not settings.background_already_dark_subtracted)
        self.background_dark_spin.setVisible(settings.background_dark_scale is not None)
        self.pool_scatter_check.setEnabled(response.error_model in ("auto", "scatter"))
        self.poisson_variance_combo.setEnabled(response.error_model in ("auto", "poisson"))
        used = getattr(self.view_model.state.analysis, "error_model_used", None)
        labels = {"poisson": "Poisson", "scatter": "Scatter of the pixels", "sigma": "Given σ frame"}
        if used in labels:
            self.error_model_status.setText(trf("Last reduction used {model}.", model=tr(labels[used])))
        self.error_model_status.setVisible(used in labels)
