"""Local instance-model and label selection behavior for the SEM window."""

from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr

from ..application import SemExtractionSettings


class SemInstanceDetectionMixin:
    def _init_instance_detection(self):
        self._instance_status_cache = None
        self._instance_status_key = None
        defaults = self.view_model.instance_defaults()
        self.python_executable_edit.setText(str(defaults.get("python_executable") or ""))
        self.model_directory_edit.setText(str(defaults.get("model_directory") or ""))

    def _connect_instance_signals(self):
        self.detection_method_combo.currentIndexChanged.connect(self._instance_settings_changed)
        for edit in (self.python_executable_edit, self.model_directory_edit, self.label_path_edit):
            edit.textChanged.connect(self._instance_settings_changed)
        for control in (
            self.ai_scale_spin, self.probability_threshold_spin, self.overlap_threshold_spin,
            self.normalize_low_spin, self.normalize_high_spin,
        ):
            control.valueChanged.connect(self._instance_settings_changed)
        self.browse_python_button.clicked.connect(self._browse_instance_python)
        self.browse_instance_model_button.clicked.connect(self._browse_instance_model)
        self.browse_instance_labels_button.clicked.connect(self._browse_instance_labels)
        self.check_instance_resources_button.clicked.connect(self._check_instance_resources)

    def _instance_settings_changed(self, *_args):
        self._instance_status_key = None
        self._invalidate_measurements()

    def _instance_options(self):
        return {
            "method": self.detection_method_combo.currentData(),
            "python_executable": self.python_executable_edit.text().strip(),
            "model_directory": self.model_directory_edit.text().strip(),
            "label_path": self.label_path_edit.text().strip(),
            "ai_scale": self.ai_scale_spin.value(),
            "probability_threshold": self.probability_threshold_spin.value() or None,
            "overlap_threshold": self.overlap_threshold_spin.value() or None,
            "normalize_low": self.normalize_low_spin.value(),
            "normalize_high": self.normalize_high_spin.value(),
        }

    def _instance_probe_settings(self):
        # An unset calibration stays zero in this file-availability probe.
        # The actual extraction request always goes through _nm_per_pixel().
        if self.calibration_mode_combo.currentIndex() == 0:
            scale = self.nm_per_pixel_spin.value()
        else:
            pixels = self.scale_bar_pixels_spin.value()
            scale = self.scale_bar_nm_spin.value() / pixels if pixels > 0 else 0
        return SemExtractionSettings(nm_per_pixel=scale, **self._instance_options())

    def _instance_availability(self):
        options = self._instance_options()
        key = tuple(options.items())
        if key != self._instance_status_key:
            self._instance_status_cache = self.view_model.instance_status(self._instance_probe_settings())
            self._instance_status_key = key
        return self._instance_status_cache or {"available": False, "message": "Local instance detection is unavailable."}

    def _update_instance_controls(self, *, image, running):
        method = self.detection_method_combo.currentData()
        ai = method == "stardist"
        labels = method == "labels"
        self.detection_section.setVisible(self.view_model.state.source_kind != "csv")
        self.detection_section.setEnabled(image)
        self.detection_method_combo.setEnabled(image and self.view_model.state.image is not None)
        self.ai_resources_widget.setVisible(ai)
        self.labels_widget.setVisible(labels)
        self.instance_ai_options.setVisible(ai)
        self.ai_resources_widget.setEnabled(not running)
        self.labels_widget.setEnabled(not running)
        self.instance_ai_options.setEnabled(not running)
        threshold = method == "threshold"
        for control in (self.threshold_mode_combo, self.split_touching_check):
            control.setEnabled(threshold)
        self.polarity_combo.setEnabled(method in {"threshold", "stardist"})
        self.threshold_spin.setEnabled(threshold and self.threshold_mode_combo.currentIndex() == 1)
        self.peak_distance_spin.setEnabled(threshold and self.split_touching_check.isChecked())
        if not image:
            self.instance_status_label.clear()
            return True
        if threshold:
            self.instance_status_label.setText(tr("Threshold / watershed uses the stored image intensities."))
            return True
        if ai and self.normalize_low_spin.value() >= self.normalize_high_spin.value():
            self.instance_status_label.setText(tr("Normalization low percentile must be less than high percentile."))
            return False
        status = self._instance_availability()
        self.instance_status_label.setText(tr(str(status.get("message", ""))))
        return bool(status.get("available", False))

    def _check_instance_resources(self):
        self._instance_status_key = None
        self._update_controls()

    def _reset_instance_source_fields(self):
        self.label_path_edit.blockSignals(True)
        self.label_path_edit.clear()
        self.label_path_edit.blockSignals(False)
        self._instance_status_key = None

    def _browse_instance_python(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose local StarDist Python"), self.python_executable_edit.text(), tr("All files (*)"))
        if path:
            self.python_executable_edit.setText(path)

    def _browse_instance_model(self):
        path = QFileDialog.getExistingDirectory(self, tr("Choose local StarDist model directory"), self.model_directory_edit.text())
        if path:
            self.model_directory_edit.setText(path)

    def _browse_instance_labels(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open existing instance labels"), self.label_path_edit.text(),
            tr("Instance labels (*.tif *.tiff *.png);;All files (*)"),
        )
        if path:
            self.label_path_edit.setText(path)


__all__ = ["SemInstanceDetectionMixin"]
