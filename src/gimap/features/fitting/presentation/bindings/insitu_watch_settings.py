"""Insitu Watch Settings for fitting presentation."""

from __future__ import annotations

import os

import numpy as np


from PyQt5.QtWidgets import (
    QFileDialog,
)

from src.gimap.shared.file_paths import normalize_path

from ..binding_primitives import (
    _ai_catalog,
)
from ..detector_data_access import analysis_image_for


class InsituWatchSettingsMixin:
    """Own insitu watch settings behavior."""

    def _is_auto_show_enabled(self) -> bool:
        return (
            hasattr(self.ui, "gisaxsInputAutoShowCheckBox")
            and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
        )

    def _get_stack_value_text(self) -> str:
        try:
            if hasattr(self.ui, "gisaxsInputStackValue"):
                return self.ui.gisaxsInputStackValue.text().strip()
        except Exception:
            pass
        return ""

    def _set_stack_value_text(self, value: str):
        try:
            stack_text = str(value).strip()
            if hasattr(self.ui, "gisaxsInputStackValue"):
                self.ui.gisaxsInputStackValue.setText(stack_text)
        except Exception:
            pass

    def _insitu_workflow_settings(self) -> dict:
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        return {
            "run_mode": widgets.get("run_mode").currentText()
            if widgets.get("run_mode")
            else "Process Existing Sequence",
            "auto_show": bool(widgets.get("auto_show").isChecked())
            if widgets.get("auto_show")
            else False,
            "auto_cut": bool(widgets.get("auto_cut").isChecked())
            if widgets.get("auto_cut")
            else False,
            "auto_fit": bool(widgets.get("auto_fit").isChecked())
            if widgets.get("auto_fit")
            else False,
            "use_previous": bool(widgets.get("use_previous").isChecked())
            if widgets.get("use_previous")
            else True,
            "full_auto_fit": bool(widgets.get("full_auto_fit").isChecked())
            if widgets.get("full_auto_fit")
            else False,
            "profile": (
                str(widgets.get("profile").currentText())
                if widgets.get("profile")
                else _ai_catalog(self).default_profile_name
            ),
            "auto_refine": bool(widgets.get("auto_refine").isChecked())
            if widgets.get("auto_refine")
            else False,
            "poll_interval": float(widgets.get("poll").value()) if widgets.get("poll") else 2.0,
            "fit_every": int(widgets.get("fit_every").value()) if widgets.get("fit_every") else 1,
            "ui_every": int(widgets.get("ui_every").value()) if widgets.get("ui_every") else 5,
            "wait_stable": bool(widgets.get("stable").isChecked())
            if widgets.get("stable")
            else True,
        }

    def _update_insitu_run_mode_ui(self):
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        mode = self._insitu_workflow_settings().get("run_mode", "Process Existing Sequence")
        is_live = mode == "Live Watch"
        for key, visible in (
            ("live_settings", is_live),
            ("sequence_settings", not is_live),
            ("start", is_live),
            ("process", not is_live),
            ("pause", True),
            ("stop", True),
        ):
            widget = widgets.get(key)
            if widget is not None:
                widget.setVisible(visible)
        self._refresh_insitu_workflow_status()

    def _populate_insitu_sequence_folder_default(self):
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        edit = widgets.get("sequence_folder")
        if edit is None:
            return
        try:
            current = self.current_parameters.get("imported_gisaxs_file", "")
            folder = os.path.dirname(current) if current else ""
            if folder:
                edit.setText(folder)
        except Exception:
            pass

    def _browse_insitu_sequence_folder(self):
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        edit = widgets.get("sequence_folder")
        start = edit.text().strip() if edit is not None else ""
        folder = QFileDialog.getExistingDirectory(
            self._insitu_workflow_parent_widget(), "Select In-situ Sequence Folder", start
        )
        if folder and edit is not None:
            edit.setText(normalize_path(folder))

    def _set_insitu_workflow_state(self, state: str, message: str = ""):
        self._insitu_workflow_state = state
        if message:
            self._log_insitu_workflow(message)
        self._refresh_insitu_workflow_status()
        if state == "Error":
            self._restore_single_analysis_runtime()

    def _log_insitu_workflow(self, message: str, level: str = "INFO"):
        text = f"[In-situ Workflow][{level}] {message}"
        try:
            self._add_fitting_message(
                text, level if level in ("INFO", "DEBUG", "ERROR", "WARN", "SUCCESS") else "INFO"
            )
        except Exception:
            try:
                self.status_updated.emit(text)
            except Exception:
                pass
        browser = (getattr(self, "_insitu_workflow_widgets", {}) or {}).get("log")
        if browser is not None:
            browser.append(text)
            try:
                document = browser.document()
                while document.blockCount() > 500:
                    cursor = browser.textCursor()
                    cursor.movePosition(cursor.Start)
                    cursor.select(cursor.BlockUnderCursor)
                    cursor.removeSelectedText()
                    cursor.deleteChar()
            except Exception:
                pass

    def _refresh_insitu_workflow_status(self):
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        labels = widgets.get("status_labels") or {}
        try:
            values = {
                "status": self._insitu_workflow_state,
                "run_mode": self._insitu_workflow_settings().get(
                    "run_mode", "Process Existing Sequence"
                ),
                "file": self._insitu_current_batch_label(),
                "processed": str(int(getattr(self, "_insitu_workflow_processed_count", 0))),
                "failed": str(int(getattr(self, "_insitu_workflow_failed_count", 0))),
                "queue": str(len(getattr(self, "_insitu_workflow_queue", []) or [])),
                "fit": str(getattr(self, "_insitu_workflow_last_fit_status", "-") or "-"),
                "chi": self._format_optional_float(
                    getattr(self, "_insitu_workflow_last_chi_square", None)
                ),
                "cache": str(self._insitu_session_cache_path()),
            }
            for key, value in values.items():
                if key in labels and labels[key] is not None:
                    if key == "file":
                        labels[key].setText(f"Current image: {value}")
                    elif key == "processed":
                        labels[key].setText(f"Done: {value}")
                    elif key == "failed":
                        labels[key].setText(f"Failed: {value}")
                    elif key == "queue":
                        labels[key].setText(f"Queue: {value}")
                    else:
                        labels[key].setText(value)
            start_btn = widgets.get("start")
            pause_btn = widgets.get("pause")
            stop_btn = widgets.get("stop")
            running = self._insitu_workflow_state in ("Watching", "Processing", "Paused")
            if start_btn is not None:
                start_btn.setEnabled(not running or self._insitu_workflow_state == "Paused")
                start_btn.setText(
                    "Resume" if self._insitu_workflow_state == "Paused" else "Start Watch"
                )
            process_btn = widgets.get("process")
            if process_btn is not None:
                process_btn.setEnabled(not running or self._insitu_workflow_state == "Paused")
                process_btn.setText(
                    "Resume" if self._insitu_workflow_state == "Paused" else "Start Process"
                )
            if pause_btn is not None:
                pause_btn.setEnabled(self._insitu_workflow_state in ("Watching", "Processing"))
            if stop_btn is not None:
                stop_btn.setEnabled(running or bool(getattr(self, "_insitu_workflow_busy", False)))
            page = getattr(self.ui, "fittingInsituSeriesPage", None)
            if page is not None:
                status_map = {
                    "Idle": "idle",
                    "Watching": "running",
                    "Processing": "running",
                    "Paused": "paused",
                    "Error": "failed",
                }
                total = int(getattr(self, "_insitu_workflow_processed_count", 0)) + len(
                    getattr(self, "_insitu_workflow_queue", []) or []
                )
                processed = int(getattr(self, "_insitu_workflow_processed_count", 0))
                progress = None if self._insitu_workflow_state == "Watching" else (
                    0.0 if total == 0 else processed / total
                )
                page.ui.jobStatus.set_state(
                    status_map.get(self._insitu_workflow_state, "idle"),
                    f"{processed} processed · "
                    f"{int(getattr(self, '_insitu_workflow_failed_count', 0))} failed",
                    progress=progress,
                )
                rows = self._load_insitu_session_records()
                page.render_records(rows)
                current = getattr(self, "_insitu_workflow_current_record", None)
                if isinstance(current, dict):
                    page.set_step_state(
                        "source", page._normalize_step_state(current.get("load_status"))
                    )
                    page.set_step_state(
                        "preprocess",
                        page._normalize_step_state(current.get("preprocess_status")),
                    )
                    page.set_step_state(
                        "geometry",
                        page._normalize_step_state(current.get("geometry_status")),
                    )
                    page.set_step_state(
                        "cut", page._normalize_step_state(current.get("cut_status"))
                    )
                    page.set_step_state(
                        "fit", page._normalize_step_state(current.get("fit_status"))
                    )
        except Exception:
            pass

    def _insitu_current_batch_label(self) -> str:
        batch = getattr(self, "_insitu_workflow_processing_batch", None) or []
        try:
            if len(batch) > 1:
                return f"{os.path.basename(batch[0])} -> {os.path.basename(batch[-1])} ({len(batch)} files)"
            if len(batch) == 1:
                return os.path.basename(batch[0])
        except Exception:
            pass
        return os.path.basename(self._insitu_workflow_processing_file or "-")

    def _format_optional_float(self, value):
        try:
            if value is None:
                return "-"
            value = float(value)
            return f"{value:.6g}" if np.isfinite(value) else "-"
        except Exception:
            return "-"

    def _refresh_insitu_workflow_step_styles(self):
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        cut_valid, _message = self._validate_current_cut_settings()
        show_valid = bool(
            analysis_image_for(self) is not None
            or self.current_parameters.get("imported_gisaxs_file")
        )
        fit_valid = self._has_active_fitting_template()
        previous_valid = self._insitu_workflow_last_fit_params is not None
        style_map = {
            "auto_show": "color: #16803c;" if show_valid else "color: #b00020;",
            "auto_cut": "color: #16803c;" if cut_valid else "color: #b00020;",
            "auto_fit": "color: #16803c;" if fit_valid else "color: #b00020;",
            "use_previous": "color: #16803c;" if previous_valid else "color: #b00020;",
            "full_auto_fit": "color: #16803c;" if fit_valid else "color: #b00020;",
            "auto_refine": "color: #16803c;"
            if self.fitting_view_model.storage.dependency_available("scipy")
            else "color: #b00020;",
        }
        for key, style in style_map.items():
            widget = widgets.get(key)
            if widget is None:
                continue
            widget.setStyleSheet(style if widget.isChecked() else "color: #202124;")
        auto_fit_enabled = bool(widgets.get("auto_fit") and widgets["auto_fit"].isChecked())
        for key in ("use_previous", "full_auto_fit", "auto_refine"):
            widget = widgets.get(key)
            if widget is not None:
                widget.setEnabled(auto_fit_enabled)
        heatmap_button = widgets.get("heatmap")
        if heatmap_button is not None:
            heatmap_button.setEnabled(
                bool(widgets.get("auto_cut") and widgets["auto_cut"].isChecked())
            )
        page = getattr(self.ui, "fittingInsituSeriesPage", None)
        if page is not None:
            page.set_step_state("source", "configured" if show_valid else "pending")
            page.set_step_state("preprocess", "configured")
            page.set_step_state("geometry", "configured")
            page.set_step_state("cut", "configured" if cut_valid else "error")
            page.set_step_state("fit", "configured" if fit_valid else "pending")
        self._draw_insitu_workflow_region_preview()

    def _validate_current_cut_settings(self):
        try:
            analysis_image = analysis_image_for(self)
            if analysis_image is None:
                return False, "No image loaded"
            geometry = self._insitu_cut_geometry()
            vertical_value = geometry.get("cut_vertical_px", 0.0)
            parallel_value = geometry.get("cut_parallel_px", 0.0)
            if vertical_value <= 0 or parallel_value <= 0:
                return False, "Cut width/height must be positive"
            info = self._create_selection_from_current_cut_controls()
            if not info:
                return False, "Cut ROI unavailable"
            bounds = info.get("bounds", {})
            height, width = analysis_image.shape
            if (
                bounds.get("x_max", 0) <= 0
                or bounds.get("y_max", 0) <= 0
                or bounds.get("x_min", width) >= width
                or bounds.get("y_min", height) >= height
            ):
                return False, "Cut ROI is outside the image"
            return True, "OK"
        except Exception as exc:
            return False, str(exc)
