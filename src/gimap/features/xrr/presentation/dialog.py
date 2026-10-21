"""Modeless Tools window for streaming XRR extraction."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import QDialog, QMessageBox

from src.gimap.app.bootstrap import create_standalone_legacy_context
from src.gimap.app.presentation import install_safe_wheel_behavior
from src.gimap.app.presentation.assets import app_icon
from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.task_runner import WORKER_STACK_BYTES

from ..application import (
    SpecularGeometry,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrExtractionSettings,
    XrrSeriesSpec,
)
from .bindings.files import XrrFilesMixin
from .bindings.geometry_sources import GeometrySourcesMixin
from .pixel_coordinates import ROI_PIXEL_NOTE, extraction_coordinate, shown_roi_center
from .plotting import XrrPlotPresenter
from .views import XrrSeriesDialogView
from .workers import XrrExtractionWorker, XrrInspectWorker

ROI_CENTER_COLUMNS = (4, 5)
"""``ROI x`` and ``ROI y`` of the results table."""


class XrrSeriesDialog(XrrFilesMixin, GeometrySourcesMixin, QDialog, XrrSeriesDialogView):
    """The window. Its beam centre fields, table and preview are in the pixels of Analyze; the extraction's
    own numbers (``SpecularGeometry``, ``XrrPoint``) are reached only through ``pixel_coordinates``."""

    def __init__(self, parent=None, *, app_context=None, view_model=None):
        super().__init__(parent)
        self.app_context = (
            app_context
            or getattr(parent, "app_context", None)
            or create_standalone_legacy_context()
        )
        if view_model is None:
            from ..bootstrap import create_xrr_view_model

            view_model = create_xrr_view_model(self.app_context)
        self.view_model = view_model
        self.setupUi(self)
        self.setWindowTitle("XRR Series Extractor")
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint
        )
        install_safe_wheel_behavior(self)
        self.plotter = XrrPlotPresenter(
            self.live_panel,
            self.curve_panel,
            on_detector_click=self._on_detector_click,
        )
        self._inspect_thread = None
        self._inspect_worker = None
        self._extract_thread = None
        self._extract_worker = None
        self._last_preview = None
        self._last_preview_shape = None
        self._last_roi_center = None
        self._last_progress_total = 0
        self._cancel_requested = False
        self._close_when_idle = False
        self._run_geometry_sources: dict[str, str] = {}
        """Where each geometry value of the running (or last) extraction came from."""
        self.setAcceptDrops(True)
        self._init_geometry_sources()
        self._refresh_roi_column_tips()
        language_changed().connect(self._refresh_roi_column_tips)
        self._connect_signals()
        self._update_angle_controls()
        self._update_source_controls()

    def _connect_signals(self) -> None:
        self.source_picker.browseRequested.connect(self._browse_source)
        self.source_folder_button.clicked.connect(self._browse_folder)
        self.inspect_button.clicked.connect(self._inspect_series)
        self.run_button.clicked.connect(self._run_extraction)
        self.export_button.clicked.connect(self._export_curve)
        self.job_status.cancelRequested.connect(self._cancel_extraction)
        self.angle_mode_combo.currentIndexChanged.connect(self._update_angle_controls)
        self.source_kind_combo.currentIndexChanged.connect(self._update_source_controls)
        self.pick_center_button.toggled.connect(self._pick_center_toggled)
        self.log_y_check.toggled.connect(self._redraw_curve)
        for control in (self.center_x_spin, self.center_y_spin, self.radius_spin):
            control.valueChanged.connect(self._redraw_detector)

    def _series_spec(self) -> XrrSeriesSpec:
        path = self.source_picker.path()
        if not path:
            raise ValueError("Select an NXS module file or a CBF series first.")
        kinds = ("auto", "nxs", "cbf")
        angle_mode = "nxs_dataset" if self.angle_mode_combo.currentIndex() == 1 else "linear"
        dataset_path = self.angle_dataset_edit.text().strip()
        if angle_mode == "nxs_dataset" and not dataset_path:
            raise ValueError("Enter the NXS motor dataset path containing theta values.")
        return XrrSeriesSpec(
            source_path=Path(path),
            source_kind=kinds[self.source_kind_combo.currentIndex()],
            pattern=self.pattern_edit.text().strip() or "*.cbf",
            angle_mode=angle_mode,
            theta_start_deg=self.theta_start_spin.value(),
            theta_step_deg=self.theta_step_spin.value(),
            angle_dataset_path=dataset_path,
        )

    def _geometry(self) -> SpecularGeometry:
        """The geometry the extraction uses: the centre of the fields in the extraction's own pixels."""
        center_x, center_y = self.center_x_spin, self.center_y_spin
        return SpecularGeometry(
            distance_m=self.distance_spin.value() * 1e-3,
            energy_kev=self.energy_spin.value(),
            pixel_size_x_m=self.pixel_x_spin.value() * 1e-6,
            pixel_size_y_m=self.pixel_y_spin.value() * 1e-6,
            beam_center_x_px=extraction_coordinate(center_x.value(), center_x.decimals()),
            beam_center_y_px=extraction_coordinate(center_y.value(), center_y.decimals()),
            vertical_direction=-1 if self.direction_combo.currentIndex() == 0 else 1,
        )

    def _request(self) -> XrrExtractionRequest:
        aggregation = "sum" if self.aggregation_combo.currentIndex() == 0 else "mean"
        return XrrExtractionRequest(
            series=self._series_spec(),
            geometry=self._geometry(),
            extraction=XrrExtractionSettings(
                radius_px=self.radius_spin.value(),
                aggregation=aggregation,
            ),
        )

    def _inspect_series(self) -> None:
        if self._thread_running(self._inspect_thread) or self._thread_running(
            self._extract_thread
        ):
            return
        try:
            spec = self._series_spec()
        except Exception as exc:
            self._show_input_error(str(exc))
            return
        self.inspect_button.setEnabled(False)
        self.series_summary.setText(tr("Reading first frame…"))
        self._inspect_thread = QThread(self)
        self._inspect_thread.setStackSize(WORKER_STACK_BYTES)
        self._inspect_worker = XrrInspectWorker(self.view_model, spec)
        self._inspect_worker.moveToThread(self._inspect_thread)
        self._inspect_thread.started.connect(self._inspect_worker.run)
        self._inspect_worker.finished.connect(self._on_inspected)
        self._inspect_worker.failed.connect(self._on_inspect_failed)
        self._inspect_worker.finished.connect(self._inspect_thread.quit)
        self._inspect_worker.failed.connect(self._inspect_thread.quit)
        self._inspect_thread.finished.connect(self._cleanup_inspection)
        self._inspect_thread.start()

    def _on_inspected(self, inspection) -> None:
        frame = inspection.first_frame
        unused_values = self._apply_metadata(frame.metadata, frame.data.shape)
        self._show_inspection_summary(inspection, unused_values)
        if self.source_picker.path():  # a series that could be read: Browse… starts there next time
            self._remember_folder(self.source_picker.path())
        self._last_preview = frame.data
        self._last_preview_shape = tuple(frame.data.shape)
        self._last_roi_center = None
        self._show_first_frame_label()
        self._redraw_detector()

    def _on_inspect_failed(self, message: str) -> None:
        self.series_summary.setText(tr("Series inspection failed"))
        self.job_status.set_state("failed", message, progress=0.0)

    def _cleanup_inspection(self) -> None:
        self.inspect_button.setEnabled(True)
        if self._inspect_worker is not None:
            self._inspect_worker.deleteLater()
        if self._inspect_thread is not None:
            self._inspect_thread.deleteLater()
        self._inspect_worker = None
        self._inspect_thread = None
        if self._close_when_idle and not self._thread_running(self._extract_thread):
            self.close()

    def _run_extraction(self) -> None:
        if self._thread_running(self._extract_thread) or self._thread_running(
            self._inspect_thread
        ):
            return
        try:
            request = self._request()
        except Exception as exc:
            self._show_input_error(str(exc))
            return
        self._cancel_requested = False
        self._last_progress_total = 0
        self._run_geometry_sources = self.geometry_sources()
        self.view_model.state.result = None
        self.results_table.set_rows(())
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.inspect_button.setEnabled(False)
        self.job_status.set_state("running", "Starting XRR extraction…", progress=0.0)
        self._extract_thread = QThread(self)
        self._extract_thread.setStackSize(WORKER_STACK_BYTES)
        self._extract_worker = XrrExtractionWorker(self.view_model, request)
        self._extract_worker.moveToThread(self._extract_thread)
        self._extract_thread.started.connect(self._extract_worker.run)
        self._extract_worker.progress.connect(self._on_extraction_progress)
        self._extract_worker.finished.connect(self._on_extraction_finished)
        self._extract_worker.failed.connect(self._on_extraction_failed)
        self._extract_worker.finished.connect(self._extract_thread.quit)
        self._extract_worker.failed.connect(self._extract_thread.quit)
        self._extract_thread.finished.connect(self._cleanup_extraction)
        self._extract_thread.start()

    def _on_extraction_progress(self, progress) -> None:
        self._last_progress_total = progress.total
        self._last_preview = progress.preview
        self._last_preview_shape = progress.preview_shape
        self._last_roi_center = shown_roi_center(progress.point)
        self.live_frame_label.setText(
            f"{progress.completed}/{progress.total} · {progress.point.source_name} · "
            f"θ {progress.point.theta_deg:.6g}°"
        )
        self.job_status.set_state(
            "running",
            trf("Extracted frame {done} of {total}", done=progress.completed, total=progress.total),
            progress=progress.completed / max(1, progress.total),
        )
        self._redraw_detector()
        self._render_result(self.view_model.state.result)

    def _on_extraction_finished(self, result) -> None:
        result = self.view_model.state.result or result
        completed = len(result.points) if result is not None else 0
        if self._cancel_requested or completed < self._last_progress_total:
            self.job_status.set_state(
                "cancelled", trf("Stopped after {count} point(s).", count=completed), progress=0.0
            )
        else:
            # What the reader said about the files (frames left out …) stays in sight after the run.
            layout = getattr(result, "layout", None)
            remarks = ["⚠ " + trf(template, **values) for template, values in (layout.warnings if layout else ())]
            text = " ".join([trf("Extracted {count} XRR point(s).", count=completed), *remarks])
            self.job_status.set_state("succeeded", text, progress=1.0)
        self._render_result(result)

    def _on_extraction_failed(self, message: str) -> None:
        self.job_status.set_state("failed", message, progress=0.0)
        self._render_result(self.view_model.state.result)

    def _cleanup_extraction(self) -> None:
        self.run_button.setEnabled(True)
        self.inspect_button.setEnabled(True)
        if self._extract_worker is not None:
            self._extract_worker.deleteLater()
        if self._extract_thread is not None:
            self._extract_thread.deleteLater()
        self._extract_worker = None
        self._extract_thread = None
        if self._close_when_idle:
            self.close()

    def _cancel_extraction(self) -> None:
        if self._extract_worker is None:
            return
        self._cancel_requested = True
        self.job_status.set_state("running", "Stopping extraction…", progress=None)
        self._extract_worker.cancel()

    def _render_result(self, result: XrrExtractionResult | None) -> None:
        self.plotter.render_curve(result, log_y=self.log_y_check.isChecked())
        points = () if result is None else result.points
        rows = []
        for point in points:
            roi_x, roi_y = shown_roi_center(point)
            rows.append(
                (
                    point.sequence_index + 1,
                    f"{point.theta_deg:.8g}",
                    f"{point.qz_inv_angstrom:.8g}",
                    "—" if point.intensity is None else f"{point.intensity:.8g}",
                    f"{roi_x:.2f}",
                    f"{roi_y:.2f}",
                    point.valid_pixels,
                )
            )
        self.results_table.set_rows(rows)
        self.export_button.setEnabled(bool(points))

    def _refresh_roi_column_tips(self, *_args) -> None:
        """Which pixels ``ROI x`` and ``ROI y`` are (the exported CSV keeps the extraction's own numbers)."""
        for column in ROI_CENTER_COLUMNS:
            item = self.results_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tr(ROI_PIXEL_NOTE))

    def _redraw_curve(self) -> None:
        self._render_result(self.view_model.state.result)

    def _redraw_detector(self) -> None:
        """The frame, the ROI and the direct beam, all in the pixels of the fields."""
        if self._last_preview is None or self._last_preview_shape is None:
            return
        self.plotter.render_detector(
            self._last_preview,
            self._last_preview_shape,
            roi_center=self._last_roi_center,
            radius_px=self.radius_spin.value(),
            direct_center=(self.center_x_spin.value(), self.center_y_spin.value()),
            title=self.live_frame_label.text() or "Detector preview",
        )

    def _pick_center_toggled(self, active: bool) -> None:
        self.pick_center_button.setText(
            tr("Click the direct beam · Esc cancels") if active else tr("Pick direct-beam center")
        )
        self.live_canvas_cursor(active)

    def live_canvas_cursor(self, active: bool) -> None:
        self.plotter.live_canvas.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def _on_detector_click(self, event) -> None:
        if not self.pick_center_button.isChecked() or event.inaxes is not self.plotter.live_axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        # The axes are the pixels of the fields: the point clicked is the point entered.
        self.center_x_spin.setValue(float(event.xdata))
        self.center_y_spin.setValue(float(event.ydata))
        self.pick_center_button.setChecked(False)
        self._redraw_detector()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.pick_center_button.isChecked():
            self.pick_center_button.setChecked(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        """Esc closes through ``closeEvent``, which waits for a running worker thread.

        QDialog's own reject() hides the dialog without a close event; with WA_DeleteOnClose the
        running QThread child would then be deleted and take the application down.
        ``closeEvent`` does not call ``QDialog.closeEvent``, so this cannot recurse.
        """
        self.close()

    def _update_angle_controls(self) -> None:
        dataset_mode = self.angle_mode_combo.currentIndex() == 1
        self.theta_start_spin.setEnabled(not dataset_mode)
        self.theta_step_spin.setEnabled(not dataset_mode)
        self.angle_dataset_edit.setEnabled(dataset_mode)

    def _update_source_controls(self) -> None:
        nxs = self.source_kind_combo.currentIndex() == 1
        cbf = self.source_kind_combo.currentIndex() == 2
        self.pattern_edit.setEnabled(not nxs)
        if cbf and self.angle_mode_combo.currentIndex() == 1:
            self.angle_mode_combo.setCurrentIndex(0)
        self.angle_mode_combo.model().item(1).setEnabled(not cbf)

    def _show_input_error(self, message: str) -> None:
        """``message`` is the English of this window or of the reader; shown in the interface language."""
        self.job_status.set_state("failed", message, progress=0.0)
        QMessageBox.warning(self, tr("XRR Series Extractor"), tr(message))

    @staticmethod
    def _thread_running(thread) -> bool:
        return thread is not None and thread.isRunning()

    def closeEvent(self, event) -> None:
        if self._thread_running(self._extract_thread):
            self._close_when_idle = True
            self._cancel_extraction()
            event.ignore()
            return
        if self._thread_running(self._inspect_thread):
            self._close_when_idle = True
            event.ignore()
            return
        event.accept()


__all__ = ["XrrSeriesDialog"]
