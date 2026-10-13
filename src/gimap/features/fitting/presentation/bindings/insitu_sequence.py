"""Process or watch a folder of 1D curves and fit each one with the In-situ Recipe."""

from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from src.gimap.features.fitting.application import (
    DEFAULT_CURVE_PATTERN,
    DiscoverInSituFramesRequest,
    InSituSourceFrame,
)

_NUMBER = re.compile(r"(\d+)")
_ANALYZE_SUFFIXES = re.compile(r"(_sum\d+)?(_fit_input)?$", re.IGNORECASE)


def curve_number(path) -> int | None:
    """Sequence number of a curve file: the last number of its name, ignoring Analyze's suffixes.

    ``run_00042_fit_input.dat`` → 42, ``run_frame0007_sum10_fit_input.dat`` → 7.
    """
    stem = _ANALYZE_SUFFIXES.sub("", Path(path).stem)
    numbers = _NUMBER.findall(stem)
    return int(numbers[-1]) if numbers else None


class InsituSequenceMixin:
    """Queue curve files (existing or newly written) and hand them to curve processing."""

    def _has_active_fitting_template(self):
        try:
            active_shapes, _shape_configs = self._collect_active_particles()
            return bool(active_shapes)
        except Exception:
            return False

    def _insitu_source_folder(self) -> str:
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        edit = widgets.get("sequence_folder")
        folder = edit.text().strip() if edit is not None else ""
        if not folder:
            self._populate_insitu_sequence_folder_default()
            folder = edit.text().strip() if edit is not None else ""
        return folder

    def _warn_insitu(self, title: str, message: str) -> None:
        QMessageBox.warning(self._insitu_workflow_parent_widget(), title, message)

    def _reset_insitu_run(self) -> None:
        self._insitu_workflow_file_sizes = {}
        self._reset_insitu_session_cache()
        self._insitu_workflow_last_fit_params = None
        self._insitu_workflow_last_fit_status = "-"
        self._insitu_workflow_last_chi_square = None
        self._reset_insitu_heatmap_data()

    def _start_insitu_workflow(self):
        """Live Watch: fit every curve written into the folder from now on."""
        try:
            recipe_error = self._insitu_recipe_start_error()
            if recipe_error:
                self._warn_insitu("In-situ Recipe changed", recipe_error)
                return
            folder = self._insitu_source_folder()
            if not os.path.isdir(folder):
                self._warn_insitu("In-situ Workflow", f"Watch folder not found:\n{folder}")
                return
            settings = self._insitu_workflow_settings()
            if self._insitu_workflow_state != "Paused":
                self._insitu_workflow_queue = []
                self._insitu_workflow_seen = set(self._list_insitu_watch_files(folder))
                self._reset_insitu_run()
                self.fitting_view_model.insitu.start_insitu_workflow(())
            else:
                self.fitting_view_model.insitu.resume_insitu_workflow()
            self._insitu_workflow_stop_requested = False
            if self._insitu_workflow_timer is None:
                self._insitu_workflow_timer = QTimer()
                self._insitu_workflow_timer.setSingleShot(False)
                self._insitu_workflow_timer.timeout.connect(self._insitu_workflow_poll)
            self._insitu_workflow_timer.start(max(200, int(settings["poll_interval"] * 1000)))
            self._set_insitu_workflow_state("Watching", f"Watching {folder}")
            self._insitu_workflow_poll()
        except Exception as exc:
            self._set_insitu_workflow_state("Error", f"Start failed: {exc}")

    def _start_insitu_sequence_processing(self):
        """Process Existing Sequence: fit every matching curve already in the folder."""
        try:
            recipe_error = self._insitu_recipe_start_error()
            if recipe_error:
                self._warn_insitu("In-situ Recipe changed", recipe_error)
                return
            folder = self._insitu_source_folder()
            if not folder or not os.path.isdir(folder):
                self._warn_insitu("In-situ Workflow", f"Folder of curves not found:\n{folder}")
                return
            if self._insitu_workflow_state != "Paused":
                self._insitu_workflow_queue = self._build_insitu_sequence_file_list(folder)
                self._insitu_workflow_seen = set(self._insitu_workflow_queue)
                self._reset_insitu_run()
                if not self._insitu_workflow_queue:
                    QMessageBox.information(
                        self._insitu_workflow_parent_widget(),
                        "In-situ Workflow",
                        "No curves matched the folder, file pattern and range.",
                    )
                    return
                self.fitting_view_model.insitu.start_insitu_workflow(
                    tuple(self._insitu_workflow_queue)
                )
            else:
                self.fitting_view_model.insitu.resume_insitu_workflow()
            self._insitu_workflow_stop_requested = False
            if self._insitu_workflow_timer is not None:
                self._insitu_workflow_timer.stop()
            self._set_insitu_workflow_state(
                "Processing", f"Processing {len(self._insitu_workflow_queue)} curve(s)"
            )
            QTimer.singleShot(0, self._process_next_insitu_workflow_file)
        except Exception as exc:
            self._set_insitu_workflow_state("Error", f"Sequence start failed: {exc}")

    def _build_insitu_sequence_file_list(self, folder: str) -> list[str]:
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        files = self._list_insitu_watch_files(folder)
        start_value = (
            int(widgets.get("sequence_start").value()) if widgets.get("sequence_start") else 0
        )
        end_value = int(widgets.get("sequence_end").value()) if widgets.get("sequence_end") else 0
        step = max(
            1, int(widgets.get("sequence_step").value()) if widgets.get("sequence_step") else 1
        )
        filtered = []
        for path in files:
            index = curve_number(path)
            if start_value and (index is None or index < start_value):
                continue
            if end_value and (index is None or index > end_value):
                continue
            if start_value and index is not None and ((index - start_value) % step != 0):
                continue
            filtered.append(path)
        if not start_value and not end_value and step > 1:
            filtered = filtered[::step]
        return filtered

    def _pause_insitu_workflow(self):
        try:
            if self._insitu_workflow_timer is not None:
                self._insitu_workflow_timer.stop()
            self.fitting_view_model.insitu.pause_insitu_workflow()
            self._set_insitu_workflow_state("Paused", "Workflow paused after the current curve")
        except Exception:
            pass

    def _stop_insitu_workflow(self):
        try:
            if self._insitu_workflow_timer is not None:
                self._insitu_workflow_timer.stop()
            self._insitu_workflow_stop_requested = True
            self.fitting_view_model.insitu.cancel_insitu_workflow()
            self._insitu_workflow_queue = []
            self._insitu_workflow_busy = False
            self._insitu_workflow_processing_file = None
            self._cleanup_insitu_refine_worker()
            if getattr(self, "_insitu_workflow_ai_record", None) is not None:
                self._stop_ai_fitting_process()
                self._insitu_workflow_ai_record = None
                self._insitu_workflow_ai_then_refine = False
            self._set_insitu_workflow_state("Idle", "Watch stopped")
        except Exception:
            pass

    def _list_insitu_watch_files(self, folder: str):
        try:
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            pattern_edit = widgets.get("sequence_pattern")
            pattern = (pattern_edit.text().strip() if pattern_edit is not None else "") or (
                DEFAULT_CURVE_PATTERN
            )
            frames = self.fitting_view_model.storage.discover_insitu_frames(
                DiscoverInSituFramesRequest(
                    root=Path(folder),
                    pattern=pattern,
                    recursive=self._insitu_workflow_settings()["recursive"],
                )
            )
            self._insitu_discovered_frames = {frame.token: frame for frame in frames}
            return [frame.token for frame in frames]
        except Exception:
            return []

    def _insitu_workflow_poll(self):
        try:
            if self._insitu_workflow_state != "Watching":
                return
            folder = self._insitu_source_folder()
            if not folder or not os.path.isdir(folder):
                self._set_insitu_workflow_state("Error", "Watch folder is unavailable")
                return
            settings = self._insitu_workflow_settings()
            for path in self._list_insitu_watch_files(folder):
                if path in self._insitu_workflow_seen or path in self._insitu_workflow_queue:
                    continue
                if settings["wait_stable"] and not self._insitu_workflow_file_is_stable(path):
                    continue
                self._insitu_workflow_seen.add(path)
                self._insitu_workflow_queue.append(path)
                self.fitting_view_model.insitu.enqueue_insitu_files((path,))
                self._log_insitu_workflow(f"Queued {self._insitu_frame_for_token(path).display_name}")
            self._refresh_insitu_workflow_status()
            self._process_next_insitu_workflow_file()
        except Exception as exc:
            self._set_insitu_workflow_state("Error", f"Polling failed: {exc}")

    def _insitu_workflow_file_is_stable(self, path: str) -> bool:
        """A file counts once its size and time stayed the same over one poll interval."""
        try:
            stat = self._insitu_frame_for_token(path).path.stat()
            stats = (int(stat.st_size), float(stat.st_mtime))
            previous = self._insitu_workflow_file_sizes.get(path)
            self._insitu_workflow_file_sizes[path] = stats
            return previous == stats and stats[0] > 0
        except Exception:
            return False

    def _insitu_frame_for_token(self, token: str) -> InSituSourceFrame:
        cached = getattr(self, "_insitu_discovered_frames", {}) or {}
        return cached.get(token) or InSituSourceFrame.from_token(token)

    def _process_next_insitu_workflow_file(self):
        if self._insitu_workflow_busy or self._insitu_workflow_state not in (
            "Watching",
            "Processing",
        ):
            return
        if not self._insitu_workflow_queue:
            if self._insitu_workflow_state == "Processing":
                self._set_insitu_workflow_state("Idle", "Sequence processing complete")
            self._refresh_insitu_workflow_status()
            return
        workflow_record = self.fitting_view_model.insitu.begin_next_insitu_file(1)
        if workflow_record is None:
            # Compatibility for dynamic callers that filled the queue directly.
            self.fitting_view_model.insitu.start_insitu_workflow(tuple(self._insitu_workflow_queue))
            workflow_record = self.fitting_view_model.insitu.begin_next_insitu_file(1)
        if workflow_record is None:
            return
        paths = list(workflow_record.paths)
        del self._insitu_workflow_queue[: len(paths)]
        path = paths[0]
        self._insitu_workflow_busy = True
        self._insitu_workflow_processing_file = path
        self._insitu_workflow_processing_batch = paths
        record = self._new_insitu_workflow_record(paths, workflow_record=workflow_record)
        self._insitu_workflow_current_record = record
        self._refresh_insitu_workflow_status()
        self._log_insitu_workflow(f"Loading {self._insitu_frame_for_token(path).display_name}")
        self._process_insitu_curve(record, path)

    def _new_insitu_workflow_record(self, path_or_paths, workflow_record=None) -> dict:
        paths = (
            list(path_or_paths)
            if isinstance(path_or_paths, (list, tuple))
            else [str(path_or_paths)]
        )
        frame = self._insitu_frame_for_token(paths[0]) if paths else None
        recipe = self.fitting_view_model.insitu.recipe
        return {
            "file_index": (
                int(workflow_record.index)
                if workflow_record is not None
                else len(getattr(self, "_insitu_workflow_results", []) or []) + 1
            ),
            "file_name": frame.display_name if frame else "",
            "file_path": str(frame.path) if frame else "",
            "curve_number": curve_number(frame.path) if frame else None,
            "batch_paths": json.dumps(paths, ensure_ascii=False),
            "timestamp": (
                workflow_record.started_at
                if workflow_record is not None
                else datetime.datetime.now().isoformat(timespec="seconds")
            ),
            "run_mode": self._insitu_workflow_settings().get(
                "run_mode", "Process Existing Sequence"
            ),
            "recipe_version": recipe.version if recipe is not None else "",
            "load_status": "pending",
            "fit_status": "skipped",
            "chi_square": "",
            "fitted_parameters": "",
            "error_message": "",
        }


__all__ = ["InsituSequenceMixin", "curve_number"]
