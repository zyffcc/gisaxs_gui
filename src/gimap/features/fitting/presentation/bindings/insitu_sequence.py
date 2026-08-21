"""Insitu Sequence for fitting presentation."""

from __future__ import annotations

import os

import json

import re


import datetime

from pathlib import Path


from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.shared.file_paths import normalize_path


from ..binding_primitives import (
    InsituBatchImageLoader,
)
from ..detector_data_access import analysis_image_for


class InsituSequenceMixin:
    """Own insitu sequence behavior."""

    def _has_active_fitting_template(self):
        try:
            active_shapes, _shape_configs = self._collect_active_particles()
            return bool(active_shapes)
        except Exception:
            return False

    def _start_insitu_workflow(self):
        try:
            recipe_error = self._insitu_recipe_start_error()
            if recipe_error:
                QMessageBox.warning(
                    self._insitu_workflow_parent_widget(),
                    "In-situ Recipe changed",
                    recipe_error,
                )
                return
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            folder = (
                widgets.get("sequence_folder").text().strip()
                if widgets.get("sequence_folder")
                else ""
            )
            if not folder:
                self._populate_insitu_sequence_folder_default()
                folder = (
                    widgets.get("sequence_folder").text().strip()
                    if widgets.get("sequence_folder")
                    else ""
                )
            if not self.fitting_view_model.storage.is_remote_source(folder) and not os.path.isdir(
                folder
            ):
                QMessageBox.warning(
                    self._insitu_workflow_parent_widget(),
                    "In-situ Workflow",
                    f"Watch folder not found:\n{folder}",
                )
                return
            settings = self._insitu_workflow_settings()
            if not any(settings[key] for key in ("auto_show", "auto_cut", "auto_fit")):
                QMessageBox.information(
                    self._insitu_workflow_parent_widget(),
                    "In-situ Workflow",
                    "Enable at least one workflow step.",
                )
                return
            self._activate_insitu_recipe_runtime()
            if self._insitu_workflow_state != "Paused":
                self._insitu_workflow_queue = []
                if (
                    self.fitting_view_model.storage.is_remote_source(folder)
                    and normalize_path(folder) not in self._folder_image_scan_cache
                ):
                    self._scan_folder_images_for_file(folder)
                self._insitu_workflow_seen = set(self._list_insitu_watch_files(folder))
                self._insitu_workflow_file_sizes = {}
                self._reset_insitu_session_cache()
                self._insitu_workflow_last_fit_params = None
                self._insitu_workflow_last_fit_status = "-"
                self._insitu_workflow_last_chi_square = None
                self._reset_insitu_heatmap_data()
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
        try:
            recipe_error = self._insitu_recipe_start_error()
            if recipe_error:
                QMessageBox.warning(
                    self._insitu_workflow_parent_widget(),
                    "In-situ Recipe changed",
                    recipe_error,
                )
                return
            settings = self._insitu_workflow_settings()
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            folder = (
                widgets.get("sequence_folder").text().strip()
                if widgets.get("sequence_folder")
                else ""
            )
            if not folder:
                self._populate_insitu_sequence_folder_default()
                folder = (
                    widgets.get("sequence_folder").text().strip()
                    if widgets.get("sequence_folder")
                    else ""
                )
            if not folder or (
                not self.fitting_view_model.storage.is_remote_source(folder)
                and not os.path.isdir(folder)
            ):
                QMessageBox.warning(
                    self._insitu_workflow_parent_widget(),
                    "In-situ Workflow",
                    f"Sequence folder not found:\n{folder}",
                )
                return
            if not any(settings[key] for key in ("auto_show", "auto_cut", "auto_fit")):
                QMessageBox.information(
                    self._insitu_workflow_parent_widget(),
                    "In-situ Workflow",
                    "Enable at least one workflow step.",
                )
                return
            if self._insitu_workflow_state != "Paused":
                self._insitu_workflow_queue = self._build_insitu_sequence_file_list(folder)
                self._insitu_workflow_seen = set(self._insitu_workflow_queue)
                self._insitu_workflow_file_sizes = {}
                self._reset_insitu_session_cache()
                self._insitu_workflow_last_fit_params = None
                self._insitu_workflow_last_fit_status = "-"
                self._insitu_workflow_last_chi_square = None
                self._reset_insitu_heatmap_data()
                if not self._insitu_workflow_queue:
                    QMessageBox.information(
                        self._insitu_workflow_parent_widget(),
                        "In-situ Workflow",
                        "No files matched the sequence settings.",
                    )
                    return
                self.fitting_view_model.insitu.start_insitu_workflow(
                    tuple(self._insitu_workflow_queue)
                )
            else:
                self.fitting_view_model.insitu.resume_insitu_workflow()
            self._activate_insitu_recipe_runtime()
            self._insitu_workflow_stop_requested = False
            if self._insitu_workflow_timer is not None:
                self._insitu_workflow_timer.stop()
            self._set_insitu_workflow_state(
                "Processing", f"Processing {len(self._insitu_workflow_queue)} existing file(s)"
            )
            QTimer.singleShot(0, self._process_next_insitu_workflow_file)
        except Exception as exc:
            self._set_insitu_workflow_state("Error", f"Sequence start failed: {exc}")

    def _build_insitu_sequence_file_list(self, folder: str) -> list[str]:
        widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
        pattern = (
            widgets.get("sequence_pattern").text().strip()
            if widgets.get("sequence_pattern")
            else "*.cbf"
        )
        pattern = pattern or "*.cbf"
        if self.fitting_view_model.storage.is_remote_source(folder):
            cached = self._folder_image_scan_cache.get(normalize_path(folder))
            if cached is None:
                self._scan_folder_images_for_file(folder)
                self._set_insitu_workflow_state("Idle", "Scanning remote sequence folder...")
                return []
            files = [p for p in cached if Path(p).match(pattern)]
        else:
            try:
                matches = sorted(
                    Path(folder).glob(pattern), key=lambda p: self._natural_sort_key(str(p))
                )
            except Exception:
                matches = []
            files = [str(path) for path in matches if path.is_file()]
        start_value = (
            int(widgets.get("sequence_start").value()) if widgets.get("sequence_start") else 0
        )
        end_value = int(widgets.get("sequence_end").value()) if widgets.get("sequence_end") else 0
        step = max(
            1, int(widgets.get("sequence_step").value()) if widgets.get("sequence_step") else 1
        )

        # 函数说明：实现 index from name 相关逻辑。
        def index_from_name(path: str):
            match = re.search(r"(\d+)(?=\.[^.]+$|$)", os.path.basename(path))
            return int(match.group(1)) if match else None

        filtered = []
        for path in files:
            idx = index_from_name(path)
            if start_value and (idx is None or idx < start_value):
                continue
            if end_value and (idx is None or idx > end_value):
                continue
            if start_value and idx is not None and ((idx - start_value) % step != 0):
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
            self._set_insitu_workflow_state("Paused", "Workflow paused after the current operation")
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
            image_loader = getattr(self, "async_image_loader", None)
            if image_loader is not None and image_loader.isRunning():
                try:
                    image_loader.requestInterruption()
                    self.status_updated.emit("Requested cancellation of current image loading")
                except Exception:
                    pass
            loader = getattr(self, "_insitu_batch_loader", None)
            if loader is not None and loader.isRunning():
                try:
                    loader.requestInterruption()
                    loader.quit()
                except Exception:
                    pass
                self._insitu_batch_loader = None
            cut_worker = getattr(self, "_insitu_cut_worker", None)
            if cut_worker is not None and cut_worker.isRunning():
                try:
                    cut_worker.requestInterruption()
                    cut_worker.quit()
                except Exception:
                    pass
                self._insitu_cut_worker = None
            if getattr(self, "_insitu_workflow_ai_record", None) is not None:
                self._stop_ai_fitting_process()
                self._insitu_workflow_ai_record = None
                self._insitu_workflow_ai_then_refine = False
            self._set_insitu_workflow_state("Idle", "Watch stopped")
            self._restore_single_analysis_runtime()
        except Exception:
            pass

    def _list_insitu_watch_files(self, folder: str):
        try:
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            pattern = (
                widgets.get("sequence_pattern").text().strip()
                if widgets.get("sequence_pattern")
                else "*.cbf"
            ) or "*.cbf"
            if self.fitting_view_model.storage.is_remote_source(folder):
                cached = self._folder_image_scan_cache.get(normalize_path(folder))
                return [path for path in (cached or []) if Path(path).match(pattern)]
            return [
                str(path)
                for path in sorted(Path(folder).glob(pattern), key=lambda p: self._natural_sort_key(str(p)))
                if path.is_file()
            ]
        except Exception:
            return []

    def _insitu_workflow_poll(self):
        try:
            if self._insitu_workflow_state != "Watching":
                return
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            folder = (
                widgets.get("sequence_folder").text().strip()
                if widgets.get("sequence_folder")
                else ""
            )
            if not folder or (
                not self.fitting_view_model.storage.is_remote_source(folder)
                and not os.path.isdir(folder)
            ):
                self._set_insitu_workflow_state("Error", "Watch folder is unavailable")
                return
            if (
                self.fitting_view_model.storage.is_remote_source(folder)
                and normalize_path(folder) not in self._folder_image_scan_cache
            ):
                self._scan_folder_images_for_file(folder)
                self._set_insitu_workflow_state("Watching", "Scanning remote watch folder...")
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
                self._log_insitu_workflow(f"Queued {os.path.basename(path)}")
            self._refresh_insitu_workflow_status()
            self._process_next_insitu_workflow_file()
        except Exception as exc:
            self._set_insitu_workflow_state("Error", f"Polling failed: {exc}")

    def _insitu_workflow_file_is_stable(self, path: str) -> bool:
        try:
            if self.fitting_view_model.storage.is_remote_source(path):
                return True
            stat = os.stat(path)
            previous = self._insitu_workflow_file_sizes.get(path)
            current = (int(stat.st_size), float(stat.st_mtime))
            self._insitu_workflow_file_sizes[path] = current
            return previous == current and current[0] > 0
        except Exception:
            return False

    def _process_next_insitu_workflow_file(self):
        if self._insitu_workflow_busy or self._insitu_workflow_state not in (
            "Watching",
            "Processing",
        ):
            return
        if not self._insitu_workflow_queue:
            if self._insitu_workflow_state == "Processing":
                self._set_insitu_workflow_state("Idle", "Sequence processing complete")
                self._restore_single_analysis_runtime()
            self._refresh_insitu_workflow_status()
            return
        batch_size = max(1, int(self._insitu_workflow_settings().get("fit_every", 1)))
        workflow_record = self.fitting_view_model.insitu.begin_next_insitu_file(batch_size)
        if workflow_record is None:
            # Compatibility for dynamic callers that filled the legacy queue directly.
            self.fitting_view_model.insitu.start_insitu_workflow(tuple(self._insitu_workflow_queue))
            workflow_record = self.fitting_view_model.insitu.begin_next_insitu_file(batch_size)
        if workflow_record is None:
            return
        batch_paths = list(workflow_record.paths)
        del self._insitu_workflow_queue[: len(batch_paths)]
        path = batch_paths[0]
        self._insitu_workflow_busy = True
        self._insitu_workflow_processing_file = path
        self._insitu_workflow_processing_batch = batch_paths
        self._insitu_workflow_current_record = self._new_insitu_workflow_record(
            batch_paths, workflow_record=workflow_record
        )
        self._refresh_insitu_workflow_status()
        self._log_insitu_workflow(
            f"Loading batch of {len(batch_paths)} file(s): {os.path.basename(batch_paths[0])}"
            + (f" -> {os.path.basename(batch_paths[-1])}" if len(batch_paths) > 1 else "")
        )
        try:
            self.current_parameters["imported_gisaxs_file"] = path
            if hasattr(self.ui, "gisaxsInputImportButtonValue"):
                self.ui.gisaxsInputImportButtonValue.setText(path)
            self._scan_folder_images_for_file(path)
            self._load_insitu_workflow_batch_async(batch_paths)
            if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                if len(batch_paths) > 1:
                    self.ui.gisaxsInputStackDisplayLabel.setText(
                        f"In-situ workflow stack: {os.path.splitext(os.path.basename(batch_paths[0]))[0]} - "
                        f"{os.path.splitext(os.path.basename(batch_paths[-1]))[0]}"
                    )
                else:
                    self.ui.gisaxsInputStackDisplayLabel.setText(
                        f"In-situ workflow: {os.path.splitext(os.path.basename(path))[0]}"
                    )
        except Exception as exc:
            self._finalize_insitu_workflow_file(load_status="failed", error_message=str(exc))

    def _new_insitu_workflow_record(self, path_or_paths, workflow_record=None) -> dict:
        paths = (
            list(path_or_paths)
            if isinstance(path_or_paths, (list, tuple))
            else [str(path_or_paths)]
        )
        first = paths[0] if paths else ""
        last = paths[-1] if paths else first
        batch_name = (
            os.path.basename(first)
            if len(paths) == 1
            else f"{os.path.basename(first)} -> {os.path.basename(last)}"
        )
        return {
            "file_index": (
                int(workflow_record.index)
                if workflow_record is not None
                else len(getattr(self, "_insitu_workflow_results", []) or []) + 1
            ),
            "file_name": batch_name,
            "file_path": str(first),
            "batch_size": len(paths),
            "batch_files": json.dumps(
                [os.path.basename(path) for path in paths], ensure_ascii=False
            ),
            "batch_paths": json.dumps(paths, ensure_ascii=False),
            "timestamp": (
                workflow_record.started_at
                if workflow_record is not None
                else datetime.datetime.now().isoformat(timespec="seconds")
            ),
            "run_mode": self._insitu_workflow_settings().get(
                "run_mode", "Process Existing Sequence"
            ),
            "recipe_version": (
                self.fitting_view_model.insitu.recipe.version
                if self.fitting_view_model.insitu.recipe is not None
                else ""
            ),
            "load_status": "pending",
            "preprocess_status": "pending",
            "geometry_status": "pending",
            "cut_status": "skipped",
            "fit_status": "skipped",
            "chi_square": "",
            "fitted_parameters": "",
            "error_message": "",
        }

    def _load_insitu_workflow_batch_async(self, batch_paths: list[str]):
        try:
            mirror_fill_enabled = bool(getattr(self, "_mirror_fill_detector_gaps", False))
            if mirror_fill_enabled:
                self._log_insitu_workflow(
                    f"Mirror-filling detector gaps in canonical preprocessing "
                    f"(margin={int(getattr(self, '_mirror_gap_margin_px', 0))} px)"
                )
                if isinstance(self._insitu_workflow_current_record, dict):
                    self._insitu_workflow_current_record["mirror_fill_detector_gaps"] = True
                    self._insitu_workflow_current_record["mirror_gap_margin_px"] = int(
                        getattr(self, "_mirror_gap_margin_px", 0)
                    )
            loader = InsituBatchImageLoader(
                batch_paths,
                fitting_view_model=self.fitting_view_model,
                copy_remote_to_cache=self._remote_copy_enabled,
                cache_dir=self._remote_cache_dir,
                cache_limit_gb=self._remote_cache_limit_gb,
            )
            loader.image_loaded.connect(self._on_insitu_batch_image_loaded)
            loader.error_occurred.connect(self._on_insitu_batch_image_error)
            loader.progress_updated.connect(self._on_image_loading_progress)
            loader.remote_file_detected.connect(self._on_remote_file_detected)
            loader.copy_started.connect(self._on_remote_copy_started)
            loader.copy_finished.connect(self._on_remote_copy_finished)
            loader.finished.connect(lambda: setattr(self, "_insitu_batch_loader", None))
            self._insitu_batch_loader = loader
            loader.start()
        except Exception as exc:
            self._finalize_insitu_workflow_file(
                load_status="failed", error_message=str(exc), failed=True
            )

    def _on_insitu_batch_image_loaded(self, image_data, first_file_path: str):
        if (
            getattr(self, "_insitu_workflow_stop_requested", False)
            or not self._insitu_workflow_busy
        ):
            return
        try:
            batch_paths = getattr(self, "_insitu_workflow_processing_batch", None) or [
                first_file_path
            ]
            if len(batch_paths) > 1:
                self.status_updated.emit(
                    f"In-situ batch loading complete: {len(batch_paths)} files "
                    f"({os.path.basename(batch_paths[0])} -> {os.path.basename(batch_paths[-1])})"
                )
            if self._should_refresh_insitu_views_for_current_file():
                self._display_image(image_data)
            else:
                self._ingest_workflow_image_without_preview(image_data)
            analysis_image = analysis_image_for(self)
            if analysis_image is None:
                raise RuntimeError("Analysis image is not ready after in-situ preprocessing")
            self._after_insitu_workflow_image_loaded(analysis_image, first_file_path)
        except Exception as exc:
            self._finalize_insitu_workflow_file(
                load_status="failed", error_message=str(exc), failed=True
            )

    def _on_insitu_batch_image_error(self, message: str):
        if getattr(self, "_insitu_workflow_stop_requested", False):
            return
        self._finalize_insitu_workflow_file(
            load_status="failed", error_message=str(message), failed=True
        )

    def _after_insitu_workflow_image_loaded(self, image_data, file_path: str):
        if not self._insitu_workflow_busy or file_path != self._insitu_workflow_processing_file:
            return
        record = self._insitu_workflow_current_record or self._new_insitu_workflow_record(file_path)
        settings = self._insitu_workflow_settings()
        refresh_views = self._should_refresh_insitu_views_for_current_file()
        try:
            record["load_status"] = "ok"
            record["preprocess_status"] = "ok"
            record["geometry_status"] = "ok"
            widgets = getattr(self, "_insitu_workflow_widgets", {}) or {}
            image_label = widgets.get("image_label")
            if image_label is not None:
                image_label.setText(f"Current image: {file_path}")
            if refresh_views and (settings["auto_show"] or settings["auto_cut"]):
                self._draw_insitu_workflow_image_preview(image_data, file_path)

            if settings["auto_cut"]:
                self._start_insitu_cut_worker(
                    image_data, file_path, record, settings, refresh_views
                )
                return

            if settings["auto_fit"]:
                self._run_insitu_workflow_fit(record)
                return

            self._finalize_insitu_workflow_file(record=record)
        except Exception as exc:
            if settings["auto_cut"] and record.get("cut_status") == "skipped":
                record["cut_status"] = "failed"
            record["error_message"] = str(exc)
            self._finalize_insitu_workflow_file(record=record, failed=True)
