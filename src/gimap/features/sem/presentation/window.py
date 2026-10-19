"""Modeless SEM analysis window, also usable without a prediction model."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QWidget

from src.gimap.app.presentation import install_safe_wheel_behavior
from src.gimap.app.presentation.assets import app_icon
from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from ..application import SemComparisonSettings, SemCsvSettings, SemExtractionSettings
from .instance_detection import SemInstanceDetectionMixin
from .plotting import SemPlotPresenter
from .views import SemWindowView


class SemWindow(SemInstanceDetectionMixin, QWidget, SemWindowView):
    def __init__(self, view_model, parent=None):
        super().__init__(parent, Qt.Window)
        self.view_model = view_model
        self._runner = TaskRunner(self, max_threads=2)
        self._close_pending = False
        self._disposed = False
        self._status = ("idle", "Open an SEM image or a particle-size CSV to begin.", {}, 0.0)
        self.setupUi(self)
        self._init_instance_detection()
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose)
        install_safe_wheel_behavior(self)
        self.plotter = SemPlotPresenter(self.image_panel, self.distribution_panel)
        self._connect_signals()
        language_changed().connect(self.refresh_language)
        self._runner.busy_changed.connect(self._on_busy_changed)
        self._update_controls()
        self.refresh_language()

    def _connect_signals(self):
        self.open_image_button.clicked.connect(self._browse_image)
        self.open_csv_button.clicked.connect(self._browse_csv)
        self.run_button.clicked.connect(self._analyze)
        self.update_plot_button.clicked.connect(self._update_comparison)
        self.export_button.clicked.connect(self._browse_export)
        self.refresh_columns_button.clicked.connect(self._read_csv_columns)
        self.clear_prediction_button.clicked.connect(lambda: self.set_prediction(None))
        self.job_status.cancelRequested.connect(self._cancel_current)
        extraction_controls = (
            self.calibration_mode_combo, self.nm_per_pixel_spin, self.scale_bar_nm_spin,
            self.scale_bar_pixels_spin, self.threshold_mode_combo, self.threshold_spin,
            self.polarity_combo, self.min_area_spin, self.max_area_spin, self.circularity_spin,
            self.clear_border_check, self.split_touching_check, self.peak_distance_spin,
            self.roi_check, self.roi_x0_spin, self.roi_y0_spin, self.roi_x1_spin, self.roi_y1_spin,
            self.csv_quantity_combo, self.csv_unit_combo, self.csv_skip_spin,
        )
        for control in extraction_controls:
            signal = getattr(control, "valueChanged", None)
            if signal is None:
                signal = getattr(control, "currentIndexChanged", None)
            if signal is None:
                signal = control.toggled
            signal.connect(self._invalidate_measurements)
        self.csv_column_combo.currentTextChanged.connect(self._invalidate_measurements)
        self.csv_delimiter_edit.textChanged.connect(self._invalidate_measurements)
        for control in (self.bins_spin, self.radius_scale_spin, self.radius_min_spin, self.radius_max_spin):
            control.valueChanged.connect(self._invalidate_comparison)
        self.display_mode_combo.currentIndexChanged.connect(self._invalidate_comparison)
        self._connect_instance_signals()

    def set_prediction(self, radius_nm, probability=None, source="", metadata=None):
        self._cancel_current(show_status=False)
        self.view_model.set_prediction(radius_nm, probability, source, metadata)
        self.refresh_language()
        self._update_controls()
        if self.view_model.state.measurements is not None:
            self._update_comparison()

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open SEM image"), "",
            tr("SEM images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)"),
        )
        if path:
            self._open_source(path, "image")

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open SEM particle sizes"), "", tr("CSV files (*.csv *.tsv *.txt);;All files (*)"),
        )
        if path:
            self._open_source(path, "csv")

    def _open_source(self, path, kind):
        self._cancel_current(show_status=False)
        previous_path = self.view_model.state.source_path
        if not previous_path or Path(previous_path).resolve() != Path(path).resolve():
            for control in (self.nm_per_pixel_spin, self.scale_bar_nm_spin, self.scale_bar_pixels_spin):
                control.blockSignals(True)
                control.setValue(0)
                control.blockSignals(False)
            self._reset_instance_source_fields()
        self.view_model.select_source(path, kind)
        self.sem_source_edit.setText(str(path))
        self.csv_column_combo.blockSignals(True)
        self.csv_column_combo.clear()
        self.csv_column_combo.blockSignals(False)
        self.particles_table.set_rows(())
        self.plotter.render_image(None)
        self.plotter.render_distribution(None)
        self._refresh_summary()
        if kind == "image":
            self._submit(lambda: self.view_model.load_image(path), self._image_loaded, "Reading SEM image…")
        else:
            if Path(path).suffix.lower() == ".tsv":
                self.csv_delimiter_edit.setText("\\t")
            self._read_csv_columns()

    def _image_loaded(self, image):
        self.view_model.state.image = image
        height, width = image.image.shape
        for control, value in ((self.roi_x0_spin, 0), (self.roi_y0_spin, 0),
                               (self.roi_x1_spin, width), (self.roi_y1_spin, height)):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.plotter.render_image(image.image)
        self._set_status("succeeded", "SEM image loaded: {width} × {height} pixels. Enter its calibration before extraction.",
                         width=width, height=height, progress=1.0)
        self._update_controls()

    def _delimiter(self):
        text = self.csv_delimiter_edit.text()
        return "\t" if text == "\\t" else text

    def _read_csv_columns(self):
        if self.view_model.state.source_kind != "csv":
            return
        self.view_model.invalidate_measurements()
        self.particles_table.set_rows(())
        self.plotter.render_distribution(None)
        self._refresh_summary()
        path = self.view_model.state.source_path
        skip_rows, delimiter = self.csv_skip_spin.value(), self._delimiter()
        self._submit(
            lambda: self.view_model.inspect_csv(path, skip_rows=skip_rows, delimiter=delimiter),
            self._columns_loaded, "Reading CSV columns…",
        )

    def _columns_loaded(self, columns):
        self.view_model.state.columns = tuple(columns)
        self.csv_column_combo.blockSignals(True)
        self.csv_column_combo.clear()
        self.csv_column_combo.addItems(list(columns))
        self.csv_column_combo.setCurrentIndex(-1)
        self.csv_column_combo.setEditText("")
        self.csv_column_combo.blockSignals(False)
        self._set_status("succeeded", "CSV loaded. Select a size column, its quantity, and its unit.", progress=1.0)
        self._update_controls()

    def _nm_per_pixel(self):
        if self.calibration_mode_combo.currentIndex() == 0:
            scale = self.nm_per_pixel_spin.value()
        else:
            length, pixels = self.scale_bar_nm_spin.value(), self.scale_bar_pixels_spin.value()
            scale = length / pixels if pixels > 0 else 0
        if scale <= 0:
            raise ValueError("Enter a positive image calibration in nm per pixel or a known scale-bar length and pixel length.")
        return scale

    def _extraction_settings(self):
        roi = None
        if self.roi_check.isChecked():
            roi = (self.roi_x0_spin.value(), self.roi_y0_spin.value(),
                   self.roi_x1_spin.value(), self.roi_y1_spin.value())
        return SemExtractionSettings(
            nm_per_pixel=self._nm_per_pixel(), roi_xyxy=roi,
            threshold=self.threshold_spin.value() if self.threshold_mode_combo.currentIndex() == 1 else None,
            bright_particles=self.polarity_combo.currentIndex() == 0,
            min_area_px=self.min_area_spin.value(), max_area_px=self.max_area_spin.value() or None,
            min_circularity=self.circularity_spin.value(), exclude_border=self.clear_border_check.isChecked(),
            split_touching=self.split_touching_check.isChecked(), peak_min_distance_px=self.peak_distance_spin.value(),
            **self._instance_options(),
        )

    def _csv_settings(self):
        column = self.csv_column_combo.currentText().strip()
        if not column:
            raise ValueError("Select the CSV column containing particle sizes.")
        unit = self.csv_unit_combo.currentData()
        return SemCsvSettings(
            value_column=column, quantity="diameter" if self.csv_quantity_combo.currentIndex() == 0 else "radius",
            unit=unit, nm_per_pixel=self._nm_per_pixel() if unit == "px" else None,
            skip_rows=self.csv_skip_spin.value(), delimiter=self._delimiter(),
        )

    def _comparison_settings(self):
        return SemComparisonSettings(
            bins=self.bins_spin.value(), radius_scale=self.radius_scale_spin.value(),
            display_mode="paper" if self.display_mode_combo.currentIndex() == 1 else "probability",
            radius_min_nm=self.radius_min_spin.value(), radius_max_nm=self.radius_max_spin.value() or None,
        )

    def _analyze(self):
        try:
            if self.view_model.state.source_kind == "image":
                command = self.view_model.extraction_command(self._extraction_settings(), self._comparison_settings())
            elif self.view_model.state.source_kind == "csv":
                command = self.view_model.csv_command(self._csv_settings(), self._comparison_settings())
            else:
                raise ValueError("Open an SEM image or a particle-size CSV first.")
        except Exception as exc:
            self._failed(str(exc))
            return
        self._submit(command, self._analyzed, "Measuring SEM particles…")

    def _analyzed(self, result):
        measurements, comparison = result
        self.view_model.state.measurements = measurements
        self.view_model.state.comparison = comparison
        self._render_results()
        if len(measurements.radius_nm):
            self.result_tabs.setCurrentIndex(1)
            self._set_status("succeeded", "Measured {count} accepted particle(s). Review the outlines and particle table.",
                             count=len(measurements.radius_nm), progress=1.0)
        else:
            self.result_tabs.setCurrentIndex(0)
            self._set_status("succeeded", "No particles passed the filters. Review the outlines, detection settings, and image ROI.", progress=1.0)

    def _update_comparison(self):
        if self.view_model.state.measurements is None or not len(self.view_model.state.measurements.radius_nm):
            return
        try:
            command = self.view_model.comparison_command(self._comparison_settings())
        except Exception as exc:
            self._failed(str(exc))
            return
        self._submit(command, self._comparison_updated, "Updating particle-size distribution…")

    def _comparison_updated(self, comparison):
        self.view_model.state.comparison = comparison
        self._render_results()
        self._set_status("succeeded", "Particle-size distribution updated.", progress=1.0)

    def _render_results(self):
        state = self.view_model.state
        image = state.image.image if state.image is not None else None
        self.plotter.render_image(image, state.measurements)
        self.plotter.render_distribution(state.comparison)
        if state.measurements is not None:
            def value(number):
                return "—" if number is None else f"{number:.6g}"
            self.particles_table.set_rows(
                (p.label, tr("Yes") if p.accepted else tr("No"), value(p.radius_nm), value(p.diameter_nm),
                 value(p.area_px), value(p.circularity), value(p.centroid_x_px), value(p.centroid_y_px),
                 tr(p.rejection_reason)) for p in state.measurements.particles
            )
        self._refresh_summary()
        self._update_controls()

    def _refresh_summary(self):
        state = self.view_model.state
        if state.measurements is None:
            self.summary_label.setText(tr("No SEM measurements yet"))
        else:
            total = len(state.measurements.particles)
            count = len(state.measurements.radius_nm)
            self.summary_label.setText(trf("{accepted} accepted / {total} measured particles", accepted=count, total=total))
        if state.comparison is None:
            self.notes_label.clear()
        else:
            definition = (
                "R is the equivalent-circle radius. Original measurements and every comparison setting are included in the export."
                if state.source_kind == "image" else
                "CSV sizes are converted to radius in nm using the selected column, quantity, and unit. Original measurements and comparison settings are included in the export."
            )
            notes = [tr(definition)]
            metadata = state.comparison.metadata
            notes.append(trf("SEM radius scale: {scale}.", scale=f"{metadata.get('sem_radius_scale', 1):.6g}"))
            excluded = metadata.get("sem_count_outside_range", 0)
            prediction_outside = metadata.get("prediction_probability_outside_range")
            if excluded or (prediction_outside is not None and prediction_outside > 1e-12):
                notes.append(trf("Outside the displayed radius range: {count} SEM particles; prediction probability {probability}.",
                                 count=excluded, probability="—" if prediction_outside is None else f"{prediction_outside:.3%}"))
            if metadata.get("display_mode") == "paper":
                notes.append(tr("Paper display uses 60 SEM bins and a minimum-subtracted, peak-scaled prediction; amplitudes are not directly comparable probabilities."))
            self.notes_label.setText(" ".join(notes))

    def _invalidate_measurements(self, *_args):
        self._cancel_current(show_status=False)
        self.view_model.invalidate_measurements()
        self.particles_table.set_rows(())
        self.plotter.render_distribution(None)
        image = self.view_model.state.image
        self.plotter.render_image(image.image if image is not None else None)
        self._refresh_summary()
        self._update_controls()

    def _invalidate_comparison(self, *_args):
        self._cancel_current(show_status=False)
        self.view_model.state.comparison = None
        self.plotter.render_distribution(None)
        self._refresh_summary()
        self._update_controls()

    def _browse_export(self):
        folder = QFileDialog.getExistingDirectory(self, tr("Choose a folder for SEM results"), "")
        if folder:
            from datetime import datetime

            self._export_to(Path(folder) / ("SEM_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")))

    def _export_to(self, directory):
        if self.view_model.state.measurements is None:
            self._failed("Extract or import SEM particle sizes before exporting.")
            return
        if self.view_model.state.comparison is None and len(self.view_model.state.measurements.radius_nm):
            self._failed("Update the particle-size distribution before exporting.")
            return
        command = self.view_model.export_command(directory)
        self._submit(command, self._exported, "Exporting SEM figures, measurements, and provenance…")

    def _exported(self, paths):
        paths = tuple(paths)
        self._set_status("succeeded", "Exported {count} SEM result file(s).", count=len(paths), progress=1.0)
        if paths:
            from PyQt5.QtCore import QUrl
            from PyQt5.QtGui import QDesktopServices

            folder = str(Path(paths[0]).parent)
            show_toast(self, trf("SEM results saved to {folder}", folder=folder),
                       action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder))))

    def _submit(self, command, on_done, message):
        self._runner.cancel("sem")
        self.view_model.state.running = True
        self.view_model.state.error_message = ""
        self._set_status("running", message)
        self._update_controls()

        def delivered(result):
            self.view_model.state.running = False
            on_done(result)
            self._update_controls()

        self._runner.submit("sem", command, on_done=delivered, on_error=self._failed)

    def _failed(self, message, _details=""):
        self.view_model.state.running = False
        self.view_model.state.error_message = message
        self._set_status("failed", message, progress=0.0)
        self._update_controls()

    def _cancel_current(self, *, show_status=True):
        was_running = self.view_model.state.running
        self.view_model.cancel()
        self._runner.cancel("sem")
        self.view_model.state.running = False
        if was_running and show_status:
            self._set_status("cancelled", "Cancelled. Results from the pending operation will be discarded.", progress=0.0)
        self._update_controls()

    def _set_status(self, state, template, *, progress=None, **values):
        self._status = state, template, values, progress
        self.job_status.set_state(state, trf(template, **values), progress=progress)

    def _update_controls(self):
        state = self.view_model.state
        image = state.source_kind == "image"
        csv = state.source_kind == "csv"
        self.calibration_section.setVisible(not csv or self.csv_unit_combo.currentData() == "px")
        self.segmentation_section.setVisible(not csv)
        self.csv_section.setVisible(csv)
        self.calibration_section.setEnabled((image or (csv and self.csv_unit_combo.currentData() == "px")) and not state.running)
        self.segmentation_section.setEnabled(image and not state.running)
        self.csv_section.setEnabled(csv and not state.running)
        self.calibration_mode_combo.setEnabled(not state.running)
        direct = self.calibration_mode_combo.currentIndex() == 0
        self.nm_per_pixel_spin.setEnabled(direct)
        self.scale_bar_nm_spin.setEnabled(not direct)
        self.scale_bar_pixels_spin.setEnabled(not direct)
        instance_ready = self._update_instance_controls(image=image, running=state.running)
        for control in (self.roi_x0_spin, self.roi_y0_spin, self.roi_x1_spin, self.roi_y1_spin):
            control.setEnabled(self.roi_check.isChecked())
        ready = state.image is not None if image else bool(state.source_path) and csv
        needs_scale = image or (csv and self.csv_unit_combo.currentData() == "px")
        if needs_scale:
            try:
                self._nm_per_pixel()
            except ValueError:
                ready = False
        if csv and not self.csv_column_combo.currentText().strip():
            ready = False
        if image and not instance_ready:
            ready = False
        self.run_button.setEnabled(ready and not state.running)
        if csv:
            run_text = "Import particle sizes"
        else:
            run_text = {"threshold": "Extract particles", "stardist": "Detect particles (local AI)",
                        "labels": "Measure labeled particles"}.get(self.detection_method_combo.currentData(), "Extract particles")
        self.run_button.setText(tr(run_text))
        has_measurements = state.measurements is not None
        has_sizes = has_measurements and len(state.measurements.radius_nm) > 0
        self.update_plot_button.setEnabled(has_sizes and not state.running)
        self.export_button.setEnabled(has_measurements and (state.comparison is not None or not has_sizes) and not state.running)
        self.clear_prediction_button.setEnabled(state.prediction is not None)

    def refresh_language(self, *_args):
        state = self.view_model.state
        self.prediction_label.setText(
            tr("No prediction attached") if state.prediction is None else
            trf("Prediction attached: {source}", source=state.prediction.source or tr("Current prediction"))
        )
        status, template, values, progress = self._status
        self.job_status.set_state(status, trf(template, **values), progress=progress)
        self._refresh_summary()
        self._update_controls()
        if state.measurements is not None:
            self._render_results()

    def _on_busy_changed(self, busy):
        if not busy and self._close_pending:
            self.close()

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._cancel_current(show_status=False)
        self.plotter.dispose()
        try:
            language_changed().disconnect(self.refresh_language)
        except (TypeError, RuntimeError):
            pass

    def running(self):
        return self._runner.is_busy()

    def shutdown(self):
        """Wait for the worker's final signal before an owning main window is deleted."""
        self._close_pending = False
        self._cancel_current(show_status=False)
        finished = self._runner.wait(timeout_s=30)
        if finished:
            self.dispose()
        return finished

    def closeEvent(self, event):  # noqa: N802 - Qt API
        if self._runner.is_busy():
            self._cancel_current()
            self._close_pending = True
            self.hide()
            event.ignore()
            return
        self.dispose()
        event.accept()

    def showEvent(self, event):  # noqa: N802 - Qt API
        self._close_pending = False
        super().showEvent(event)


__all__ = ["SemWindow"]
