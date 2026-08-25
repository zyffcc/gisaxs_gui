"""Insitu Persistence Preview for fitting presentation."""

from __future__ import annotations

import os


import datetime

from pathlib import Path

import numpy as np

from PyQt5.QtCore import QUrl

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from PyQt5.QtGui import QDesktopServices

from ..detector_data_access import analysis_image_for
from ...application import InSituSourceFrame


class InsituPersistencePreviewMixin:
    """Own insitu persistence preview behavior."""

    def _insitu_cache_dir(self) -> Path:
        return self.fitting_view_model.storage.insitu_cache_directory()

    def _insitu_session_cache_path(self) -> Path:
        return self.fitting_view_model.storage.insitu_session_path()

    def _reset_insitu_session_cache(self):
        self._insitu_workflow_processed_count = 0
        self._insitu_workflow_failed_count = 0
        self._insitu_workflow_results = []
        self.fitting_view_model.storage.reset_insitu_records()
        self._refresh_insitu_workflow_status()

    def _append_insitu_session_cache(self, record: dict):
        try:
            self.fitting_view_model.storage.append_insitu_record(record)
        except Exception as exc:
            self._log_insitu_workflow(f"Session cache write failed: {exc}", "ERROR")

    def _load_insitu_session_records(self) -> list[dict]:
        if self._insitu_workflow_results:
            rows = list(self._insitu_workflow_results)
            current = getattr(self, "_insitu_workflow_current_record", None)
            if isinstance(current, dict):
                rows.append(current.copy())
            return rows
        try:
            rows = self.fitting_view_model.storage.load_insitu_records()
        except Exception as exc:
            rows = []
            self._log_insitu_workflow(f"Session cache read failed: {exc}", "ERROR")
        current = getattr(self, "_insitu_workflow_current_record", None)
        if isinstance(current, dict):
            rows.append(current.copy())
        return rows

    def _export_insitu_records_to_csv(self, path: Path, rows: list[dict]):
        try:
            if not rows:
                return
            self.fitting_view_model.storage.export_insitu_records(path, rows)
        except Exception as exc:
            self._log_insitu_workflow(f"CSV export failed: {exc}", "ERROR")

    def _export_insitu_workflow_results(self):
        rows = self._load_insitu_session_records()
        if not rows:
            QMessageBox.information(
                self._insitu_workflow_parent_widget(),
                "Export Results",
                "No cached in-situ results to export.",
            )
            return
        default_name = f"in_situ_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self._insitu_workflow_parent_widget(),
            "Export In-situ Results",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filename:
            return
        self._export_insitu_records_to_csv(Path(filename), rows)
        self._log_insitu_workflow(f"Exported results to {filename}", "SUCCESS")

    def _clear_insitu_session_cache(self):
        if self._insitu_workflow_busy:
            QMessageBox.information(
                self._insitu_workflow_parent_widget(),
                "Clear Session Cache",
                "Stop the workflow before clearing the cache.",
            )
            return
        self._reset_insitu_session_cache()
        self._insitu_workflow_last_fit_params = None
        self._insitu_workflow_last_fit_status = "-"
        self._insitu_workflow_last_chi_square = None
        self._reset_insitu_heatmap_data()
        self._log_insitu_workflow("Session cache cleared")

    def _open_insitu_cache_folder(self):
        try:
            directory = self.fitting_view_model.storage.ensure_insitu_cache_directory()
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        except Exception as exc:
            self._log_insitu_workflow(f"Open cache folder failed: {exc}", "ERROR")

    def _draw_insitu_workflow_image_preview(self, image_data, file_path: str = ""):
        holder = getattr(self, "_insitu_workflow_canvas_image", None)
        auto_cut = self._insitu_workflow_settings().get("auto_cut", False)
        self._insitu_preview_image_data = image_data
        self._insitu_preview_file_path = file_path
        self._draw_insitu_image_on_holder(
            holder,
            image_data,
            title=(
                InSituSourceFrame.from_token(file_path).display_name
                if file_path
                else "Current image"
            ),
            selection=auto_cut,
        )

    def _draw_insitu_workflow_region_preview(self):
        analysis_image = analysis_image_for(self)
        if analysis_image is None:
            return
        self._draw_insitu_workflow_image_preview(
            analysis_image, self._insitu_workflow_processing_file or ""
        )

    def _draw_insitu_image_on_holder(
        self, holder, image_data, title: str = "", selection: bool = False
    ):
        try:
            if holder is None or not hasattr(holder, "_insitu_figure"):
                return
            fig = holder._insitu_figure
            canvas = holder._insitu_canvas
            fig.clear()
            ax = fig.add_subplot(111)
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            log_enabled = bool(
                widgets.get("preview_log") and widgets["preview_log"].isChecked()
            )
            processed = np.asarray(image_data, dtype=np.float32)
            if log_enabled:
                processed = np.log(
                    np.where(np.isfinite(processed), np.maximum(processed, 0.001), np.nan)
                )
            processed = np.flipud(processed)
            preview_data, _ = self._downsample_for_preview(processed, max_pixels=280_000)
            finite = processed[np.isfinite(processed)]
            if finite.size == 0:
                raise ValueError("Preview image contains no finite intensity values")
            auto_scale = bool(
                not widgets.get("preview_auto_scale")
                or widgets["preview_auto_scale"].isChecked()
            )
            if auto_scale:
                vmin, vmax = np.percentile(finite, (1.0, 99.0))
                self._set_insitu_preview_range(float(vmin), float(vmax))
            else:
                vmin = float(widgets["preview_vmin"].value())
                vmax = float(widgets["preview_vmax"].value())
                if vmax <= vmin:
                    vmin, vmax = np.percentile(finite, (1.0, 99.0))
            ax.imshow(
                preview_data,
                cmap=(
                    widgets["preview_colormap"].currentText()
                    if widgets.get("preview_colormap")
                    else "viridis"
                ),
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            show_roi = bool(
                selection
                and widgets.get("preview_show_roi")
                and widgets["preview_show_roi"].isChecked()
            )
            if show_roi:
                try:
                    valid, message = self._validate_current_cut_settings()
                    info = self._create_selection_from_current_cut_controls()
                    if info:
                        bounds = info.get("bounds", {})
                        scale_y = preview_data.shape[0] / max(1, processed.shape[0])
                        scale_x = preview_data.shape[1] / max(1, processed.shape[1])
                        from matplotlib.patches import Rectangle

                        rect = Rectangle(
                            (bounds.get("x_min", 0) * scale_x, bounds.get("y_min", 0) * scale_y),
                            max(1, (bounds.get("x_max", 0) - bounds.get("x_min", 0)) * scale_x),
                            max(1, (bounds.get("y_max", 0) - bounds.get("y_min", 0)) * scale_y),
                            linewidth=2,
                            edgecolor="#16803c" if valid else "#b00020",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
                    if not valid:
                        ax.text(
                            0.02,
                            0.96,
                            message,
                            transform=ax.transAxes,
                            color="#b00020",
                            fontsize=9,
                            va="top",
                            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                        )
                except Exception:
                    pass
            if widgets.get("preview_show_center") and widgets[
                "preview_show_center"
            ].isChecked():
                try:
                    page = getattr(self.ui, "fittingInsituSeriesPage", None)
                    controls = page.ui.workflowControls
                    center_x = float(controls.centerXSpinBox.value())
                    center_y = float(controls.centerYSpinBox.value())
                    scale_y = preview_data.shape[0] / max(1, processed.shape[0])
                    scale_x = preview_data.shape[1] / max(1, processed.shape[1])
                    ax.plot(
                        center_x * scale_x,
                        center_y * scale_y,
                        marker="+",
                        markersize=14,
                        markeredgewidth=2,
                        color="#ffb000",
                    )
                except Exception:
                    pass
            ax.set_title(title)
            ax.axis("off")
            fig.tight_layout(pad=0.3)
            canvas.draw_idle()
        except Exception:
            pass

    def _set_insitu_preview_range(self, vmin: float, vmax: float) -> None:
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        for key, value in (("preview_vmin", vmin), ("preview_vmax", vmax)):
            editor = widgets.get(key)
            if editor is not None:
                editor.blockSignals(True)
                editor.setValue(value)
                editor.blockSignals(False)

    def _on_insitu_preview_display_changed(self, *_args) -> None:
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        auto = bool(
            widgets.get("preview_auto_scale")
            and widgets["preview_auto_scale"].isChecked()
        )
        for key in ("preview_vmin", "preview_vmax"):
            if widgets.get(key) is not None:
                widgets[key].setEnabled(not auto)
        image_data = getattr(self, "_insitu_preview_image_data", None)
        if image_data is not None:
            self._draw_insitu_workflow_image_preview(
                image_data,
                getattr(self, "_insitu_preview_file_path", ""),
            )
