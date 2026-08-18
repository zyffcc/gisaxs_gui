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
        self._draw_insitu_image_on_holder(
            holder,
            image_data,
            title=os.path.basename(file_path) or "Current image",
            selection=auto_cut,
        )

    def _draw_insitu_workflow_region_preview(self):
        if getattr(self, "current_stack_data", None) is None:
            return
        self._draw_insitu_workflow_image_preview(
            self.current_stack_data, self._insitu_workflow_processing_file or ""
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
            processed, _ = self._prepare_image_data_for_display(image_data)
            processed = np.flipud(processed)
            preview_data, _ = self._downsample_for_preview(processed, max_pixels=280_000)
            vmin = self._current_vmin if self._current_vmin is not None else np.nanmin(processed)
            vmax = self._current_vmax if self._current_vmax is not None else np.nanmax(processed)
            ax.imshow(
                preview_data,
                cmap=self._image_colormap,
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            if selection:
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
            ax.set_title(title)
            ax.axis("off")
            fig.tight_layout(pad=0.3)
            canvas.draw_idle()
        except Exception:
            pass
