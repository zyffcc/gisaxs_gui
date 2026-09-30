"""Settings, folder choice and status line of the In-situ curve series."""

from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import QFileDialog

from src.gimap.shared.file_paths import normalize_path

from ..views.insitu_workflow_controls import PLOT_ONLY


class InsituWatchSettingsMixin:
    """Own in-situ settings, the source folder and the status widgets."""

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

        def read(key, default, getter):
            widget = widgets.get(key)
            return getter(widget) if widget is not None else default

        fitting = read("fit_mode", 0, lambda widget: widget.currentIndex()) != PLOT_ONLY
        settings = {
            "run_mode": read(
                "run_mode", "Process Existing Sequence", lambda widget: widget.currentText()
            ),
            "auto_show": True,
            "auto_fit": fitting,
            "full_auto_fit": fitting,
            "use_previous": False,
            "auto_refine": False,
            "poll_interval": read("poll", 2.0, lambda widget: float(widget.value())),
            "fit_every": 1,
            "ui_every": read("ui_every", 5, lambda widget: int(widget.value())),
            "wait_stable": read("stable", True, lambda widget: bool(widget.isChecked())),
            "recursive": read("recursive", False, lambda widget: bool(widget.isChecked())),
        }
        view_model = getattr(self, "fitting_view_model", None)
        recipe = getattr(getattr(view_model, "insitu", None), "recipe", None)
        if recipe is not None and recipe.model.get("workflow_v5"):
            enabled = not recipe.model.get("extract_only", False)
            settings.update(auto_fit=enabled, full_auto_fit=enabled)
        return settings

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
        """Suggest the folder of the Single-analysis curve (usually Analyze's gimap_analysis)."""
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        edit = widgets.get("sequence_folder")
        if edit is None:
            return
        current = (getattr(self, "current_1d_data", None) or {}).get("file_path") or getattr(
            self, "current_1d_file_path", ""
        )
        folder = os.path.dirname(str(current)) if current else ""
        text = edit.text().strip()
        if folder and (not text or text == getattr(self, "_insitu_suggested_folder", None)):
            edit.setText(folder)
            self._insitu_suggested_folder = folder

    def _browse_insitu_sequence_folder(self):
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        edit = widgets.get("sequence_folder")
        start = edit.text().strip() if edit is not None else ""
        folder = QFileDialog.getExistingDirectory(
            self._insitu_workflow_parent_widget(), "Select the Folder of Curves", start
        )
        if folder and edit is not None:
            edit.setText(normalize_path(folder))

    def _set_insitu_workflow_state(self, state: str, message: str = ""):
        self._insitu_workflow_state = state
        if message:
            self._log_insitu_workflow(message)
        self._refresh_insitu_workflow_status()

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
            prefixes = {
                "file": "Current curve: ",
                "processed": "Done: ",
                "failed": "Failed: ",
                "queue": "Queue: ",
            }
            for key, value in values.items():
                label = labels.get(key)
                if label is not None:
                    label.setText(prefixes.get(key, "") + value)
            state = self._insitu_workflow_state
            running = state in ("Watching", "Processing", "Paused")
            start_btn = widgets.get("start")
            if start_btn is not None:
                start_btn.setEnabled(not running or state == "Paused")
                start_btn.setText("Resume" if state == "Paused" else "Start Watch")
            process_btn = widgets.get("process")
            if process_btn is not None:
                process_btn.setEnabled(not running or state == "Paused")
                process_btn.setText("Resume" if state == "Paused" else "Start Process")
            pause_btn = widgets.get("pause")
            if pause_btn is not None:
                pause_btn.setEnabled(state in ("Watching", "Processing"))
            stop_btn = widgets.get("stop")
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
                processed = int(getattr(self, "_insitu_workflow_processed_count", 0))
                total = processed + len(getattr(self, "_insitu_workflow_queue", []) or [])
                progress = None if state == "Watching" else (
                    0.0 if total == 0 else processed / total
                )
                page.ui.jobStatus.set_state(
                    status_map.get(state, "idle"),
                    f"{processed} processed · "
                    f"{int(getattr(self, '_insitu_workflow_failed_count', 0))} failed",
                    progress=progress,
                )
                page.render_records(self._load_insitu_session_records())
                current = getattr(self, "_insitu_workflow_current_record", None)
                if isinstance(current, dict):
                    page.set_step_state(
                        "source", page._normalize_step_state(current.get("load_status"))
                    )
                    page.set_step_state(
                        "fit", page._normalize_step_state(current.get("fit_status"))
                    )
        except Exception:
            pass

    def _insitu_current_batch_label(self) -> str:
        current = getattr(self, "_insitu_workflow_processing_file", None)
        try:
            return self._insitu_frame_for_token(current).display_name if current else "-"
        except Exception:
            return "-"

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
        page = getattr(self.ui, "fittingInsituSeriesPage", None)
        if page is None:
            return
        folder = widgets.get("sequence_folder")
        has_folder = folder is not None and bool(folder.text().strip())
        recipe = self.fitting_view_model.insitu.recipe
        page.set_step_state("source", "configured" if has_folder else "pending")
        page.set_step_state("fit", "configured" if recipe is not None else "pending")


__all__ = ["InsituWatchSettingsMixin"]
