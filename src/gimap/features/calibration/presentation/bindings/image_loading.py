"""Image Loading behavior for Calibration."""

from __future__ import annotations

import logging

from typing import Optional


from PyQt5.QtCore import QThread, QTimer

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from ...application import (
    AmbiguousImageDatasetError,
    DetectorImage,
)


from ..workers import ImageLoaderWorker

LOGGER = logging.getLogger(__name__)


class ImageLoadingMixin:
    """Own image loading presentation behavior."""

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Calibration Image",
            "",
            "Detector Images (*.nxs *.cbf);;NXS Files (*.nxs);;CBF Files (*.cbf)",
        )
        if path:
            self.load_image(self.view_model.normalize_path(path))

    def _load_path_edit(self) -> None:
        path = self.path_edit.text().strip().strip('"')
        if path:
            self.load_image(self.view_model.normalize_path(path))

    def load_image(self, path: str, dataset_path: Optional[str] = None) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        self.path_edit.setText(path)
        self.job_status.set_state("running", "Reading image...", progress=None)
        self._set_running(True)
        self._load_thread = QThread(self)
        self._load_worker = ImageLoaderWorker(path, self.view_model, dataset_path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._image_loaded)
        self._load_worker.failed.connect(lambda exc: self._image_failed(path, exc))
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._cleanup_loader)
        self._load_thread.start()

    def _image_loaded(self, image: DetectorImage) -> None:
        self.image = image
        self.result = None
        self._preview_cache.clear()
        self._reset_preview_view = True
        self.clean_preview_button.setChecked(False)
        self.manual_group.setChecked(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if image.energy_kev:
            self.energy_spin.setValue(image.energy_kev)
        if image.pixel_size_x_m:
            self.pixel_x_spin.setValue(image.pixel_size_x_m * 1e6)
        if image.pixel_size_y_m:
            self.pixel_y_spin.setValue(image.pixel_size_y_m * 1e6)
        if image.distance_m:
            self.estimated_distance_spin.setValue(image.distance_m * 1000.0)
        else:
            current_distance = self.view_model.current_geometry(
                {
                    "distance": 0.0,
                    "beam_center_x": 0.0,
                    "beam_center_y": 0.0,
                }
            )["distance"]
            if current_distance and float(current_distance) > 0:
                self.estimated_distance_spin.setValue(float(current_distance))
        if image.pixel_size_x_m and image.pixel_size_y_m:
            self.pixel_label.setText(
                f"{image.pixel_size_x_m * 1e6:.3f} × {image.pixel_size_y_m * 1e6:.3f} µm (metadata)"
            )
        else:
            self.pixel_label.setText("Not detected — enter in Advanced Settings")
            self.calibration_advanced_section.set_expanded(True)
        self.detector_label.setText(image.detector_name or "Not identified")
        detector_index = 0
        if image.detector_name:
            normalized = " ".join(image.detector_name.lower().split())
            for index in range(1, self.detector_combo.count()):
                model_name = self.detector_combo.itemData(index)
                if model_name and model_name != "custom" and model_name.lower() in normalized:
                    detector_index = index
                    break
        self.detector_combo.setCurrentIndex(detector_index)
        if not image.detector_name:
            self.detector_label.setText("Not identified — choose a detector model")
        detected_standards = self.view_model.detected_standard_keys(image.source_path)
        if len(detected_standards) == 1:
            standard_index = self.standard_combo.findData(detected_standards[0])
            if standard_index >= 0:
                self.standard_combo.setCurrentIndex(standard_index)
        elif len(detected_standards) > 1:
            auto_index = self.standard_combo.findData("auto")
            if auto_index >= 0:
                self.standard_combo.setCurrentIndex(auto_index)
        energy_note = ""
        if image.metadata.get("energy_source"):
            energy_note = " | energy from companion NXS"
        standard_note = ""
        if len(detected_standards) > 1:
            standard_note = " | multiple standard names found; comparing patterns automatically"
        self.stage_label.setText(
            f"Loaded {self.view_model.source_name(image.source_path)} — "
            f"{image.data.shape[1]} × {image.data.shape[0]} pixels"
            f"{energy_note}{standard_note}. "
            "Click Auto Calibration."
        )
        self.job_status.set_state(
            "succeeded",
            self.stage_label.text(),
            progress=1.0,
        )
        self.preview_info_label.setText(
            f"{self.view_model.source_name(image.source_path)}  ·  "
            f"{image.data.shape[1]} × {image.data.shape[0]} px"
        )
        self.candidate_table.setRowCount(0)
        self._clear_result_labels()
        self.redraw_preview()
        self._set_running(False)

    def _detector_model_changed(self) -> None:
        model_name = self.detector_combo.currentData()
        if not model_name or model_name == "custom":
            if model_name == "custom":
                self.calibration_advanced_section.set_expanded(True)
            return
        model = self.detector_models.get(model_name, {})
        pixel_x = model.get("pixel_size_x")
        pixel_y = model.get("pixel_size_y", pixel_x)
        if pixel_x:
            self.pixel_x_spin.setValue(float(pixel_x))
        if pixel_y:
            self.pixel_y_spin.setValue(float(pixel_y))
        self.pixel_label.setText(f"{float(pixel_x):.3f} × {float(pixel_y):.3f} µm ({model_name})")
        self.detector_label.setText(model_name)

    def _image_failed(self, path: str, exc: Exception) -> None:
        self.job_status.set_state("failed", "Failed to load image.", progress=0.0)
        if isinstance(exc, AmbiguousImageDatasetError):
            from PyQt5.QtWidgets import QInputDialog

            selected, ok = QInputDialog.getItem(
                self, "Select NXS Dataset", "Detector image dataset:", exc.paths, 0, False
            )
            if ok and selected:
                QTimer.singleShot(150, lambda: self.load_image(path, selected))
        else:
            QMessageBox.warning(self, "Calibration Image", str(exc))
        self._set_running(False)

    def _cleanup_loader(self) -> None:
        self._load_worker = None
        if self._load_thread is not None:
            self._load_thread.deleteLater()
        self._load_thread = None
        if self._close_when_idle and self._cal_thread is None:
            QTimer.singleShot(0, self.close)

    def _distance_range(self) -> tuple[float, float]:
        index = self.range_combo.currentIndex()
        if index == 1:
            return 500.0, 10000.0
        if index == 2:
            return 30.0, 1500.0
        if index == 3:
            low, high = self.custom_min_spin.value(), self.custom_max_spin.value()
            if low >= high:
                raise ValueError("Custom distance minimum must be smaller than the maximum.")
            return low, high
        return 30.0, 10000.0
