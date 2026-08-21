"""In-situ cut extraction and current-curve coordination."""

from __future__ import annotations

import os

import numpy as np
from PyQt5.QtCore import QTimer

from ..binding_primitives import (
    InsituCutWorker,
    _scientific_commands,
)


class InsituCutProcessingMixin:
    """In-situ cut extraction and current-curve coordination."""

    def _start_insitu_cut_worker(
        self, image_data, file_path: str, record: dict, settings: dict, refresh_views: bool
    ):
        try:
            valid, message = self._validate_current_cut_settings()
            if not valid:
                record["cut_status"] = f"failed: {message}"
                raise RuntimeError(message)
            geometry = self._insitu_cut_geometry()
            vertical_value = geometry.get("cut_vertical_px", 0.0)
            parallel_value = geometry.get("cut_parallel_px", 0.0)
            center_x = geometry.get("center_parallel_px", 0.0)
            center_y = geometry.get("center_vertical_px", 0.0)
            show_q_axis = self._should_show_q_axis()
            qy_mesh = qz_mesh = None
            if show_q_axis:
                qy_mesh, qz_mesh = self._get_cached_q_meshgrids()
                if qy_mesh is None or qz_mesh is None:
                    raise RuntimeError("Q-space meshgrids not available")
            try:
                method = (
                    self.ui.fitInterpolationMethodValue.currentText()
                    if hasattr(self.ui, "fitInterpolationMethodValue")
                    else self._interp_method_default
                )
            except Exception:
                method = self._interp_method_default
            payload = {
                "image_data": image_data,
                "vertical": vertical_value,
                "parallel": parallel_value,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "cut_type": "horizontal" if vertical_value <= parallel_value else "vertical",
                "show_q_axis": bool(show_q_axis),
                "qy_mesh": qy_mesh,
                "qz_mesh": qz_mesh,
                "horizontal_q_axis": self._horizontal_q_axis(),
                "n_points": self._resolve_cut_points(None),
                "method": method,
            }
            worker = InsituCutWorker(
                payload,
                _scientific_commands(self).insitu_cut,
            )
            worker.cut_finished.connect(
                lambda result, rec=record, fp=file_path, st=dict(settings), rv=bool(refresh_views): (
                    self._on_insitu_cut_finished(result, rec, fp, st, rv)
                )
            )
            worker.error_occurred.connect(
                lambda message, rec=record, st=dict(settings): self._on_insitu_cut_failed(
                    message, rec, st
                )
            )
            worker.finished.connect(lambda: setattr(self, "_insitu_cut_worker", None))
            self._insitu_cut_worker = worker
            record["cut_status"] = "cutting"
            self._log_insitu_workflow(f"Cut worker started for {os.path.basename(file_path)}")
            self._refresh_insitu_workflow_status()
            worker.start()
        except Exception as exc:
            record["cut_status"] = "failed"
            record["error_message"] = str(exc)
            self._finalize_insitu_workflow_file(record=record, failed=True)

    def _on_insitu_cut_finished(
        self, result: dict, record: dict, file_path: str, settings: dict, refresh_views: bool
    ):
        try:
            if getattr(
                self, "_insitu_workflow_stop_requested", False
            ) or self._insitu_workflow_state not in ("Watching", "Processing"):
                return
            x_coords = np.asarray(result.get("x_coords", []), dtype=float)
            y_values = np.asarray(result.get("y_intensity", []), dtype=float)
            if result.get("source") == "pixel":
                if result.get("cut_type") == "vertical":
                    self._last_vertical_cut_pixel_rows = x_coords.copy()
                    x_coords = self._convert_pixel_to_qz(x_coords)
                    x_label = r"$q_z$ (nm$^{-1}$)"
                else:
                    x_coords = self._convert_pixel_to_qy(x_coords)
                    x_label = self._horizontal_q_label()
            else:
                x_label = str(result.get("x_label") or r"$q$ (nm$^{-1}$)")
            old_suppress = getattr(self, "_suppress_workflow_plot_updates", False)
            self._suppress_workflow_plot_updates = not refresh_views
            try:
                self._plot_cut_result(
                    x_coords,
                    y_values,
                    x_label,
                    "Intensity (a.u.)",
                    str(result.get("title") or "Cut"),
                )
            finally:
                self._suppress_workflow_plot_updates = old_suppress
            if getattr(self, "current_cut_data", None) is None:
                raise RuntimeError("Cut did not produce data")
            excluded_count = self._apply_deleted_point_mask_to_current_cut()
            if excluded_count:
                record["deleted_points_applied"] = int(excluded_count)
                self._log_insitu_workflow(
                    f"Applied deleted-point mask: removed {excluded_count} point(s) from {os.path.basename(file_path)}"
                )
            self._append_insitu_heatmap_cut(
                self.current_cut_data.get("x_coords", []),
                self.current_cut_data.get("y_intensity", []),
            )
            record["cut_status"] = "ok"
            if refresh_views:
                self._draw_insitu_workflow_region_preview()
                self._draw_insitu_workflow_curve_preview()
            self._log_insitu_workflow(
                f"Cut completed for {os.path.basename(file_path)}: {result.get('points', len(x_coords))} point(s)",
                "SUCCESS",
            )
            if settings.get("auto_fit"):
                QTimer.singleShot(0, lambda rec=record: self._run_insitu_workflow_fit(rec))
                return
            self._finalize_insitu_workflow_file(record=record)
        except Exception as exc:
            self._on_insitu_cut_failed(str(exc), record, settings)

    def _on_insitu_cut_failed(self, message: str, record: dict, settings: dict):
        if getattr(
            self, "_insitu_workflow_stop_requested", False
        ) or self._insitu_workflow_state not in ("Watching", "Processing"):
            return
        record["cut_status"] = "failed"
        record["error_message"] = str(message)
        self._finalize_insitu_workflow_file(record=record, failed=True)

    def _should_refresh_insitu_views_for_current_file(self) -> bool:
        try:
            ui_every = max(1, int(self._insitu_workflow_settings().get("ui_every", 5)))
            index = int((self._insitu_workflow_current_record or {}).get("file_index", 1))
            return (index % ui_every) == 0 or not bool(getattr(self, "_insitu_workflow_queue", []))
        except Exception:
            return True

    def _apply_deleted_point_mask_to_current_cut(self) -> int:
        """Apply the global deleted-q mask to the current cut arrays in-place."""
        excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
        if not excluded or getattr(self, "current_cut_data", None) is None:
            return 0
        try:
            q_arr = np.asarray(self.current_cut_data.get("x_coords", []), dtype=float).reshape(-1)
            i_arr = np.asarray(self.current_cut_data.get("y_intensity", []), dtype=float).reshape(
                -1
            )
            n = min(q_arr.size, i_arr.size)
            if n <= 0:
                return 0
            q_arr, i_arr = q_arr[:n], i_arr[:n]
            keep = np.array(
                [
                    self._ai_q_key(q_val) not in excluded
                    and self._ai_q_key(abs(float(q_val))) not in excluded
                    for q_val in q_arr
                ],
                dtype=bool,
            )
            removed = int(n - np.sum(keep))
            if removed <= 0 or int(np.sum(keep)) <= 0:
                return 0
            q_filtered = q_arr[keep]
            i_filtered = i_arr[keep]
            self.current_cut_data["x_coords"] = q_filtered
            self.current_cut_data["y_intensity"] = i_filtered
            self.q = q_filtered
            self.I = i_filtered
            if isinstance(getattr(self, "cut", None), dict):
                self.cut["q"] = q_filtered
                self.cut["I"] = i_filtered
                meta = self.cut.setdefault("meta", {})
                meta["deleted_points_applied"] = removed
            return removed
        except Exception as exc:
            self._log_insitu_workflow(f"Deleted-point mask failed: {exc}", "ERROR")
            return 0

    def _run_insitu_workflow_fit(self, record: dict):
        settings = self._insitu_workflow_settings()
        try:
            if settings["use_previous"] and self._insitu_workflow_last_fit_params is not None:
                setup = self._build_manual_refine_setup()
                if setup is not None:
                    self._apply_manual_refine_result(setup, self._insitu_workflow_last_fit_params)

            if settings["full_auto_fit"]:
                self._insitu_workflow_ai_record = record
                self._insitu_workflow_ai_then_refine = bool(settings["auto_refine"])
                self._log_insitu_workflow("Full Auto Fit started")
                self._start_ai_prediction("full")
                thread = getattr(self, "_ai_job_thread", None)
                if thread is None:
                    self._insitu_workflow_ai_record = None
                    raise RuntimeError(
                        "Full Auto Fit did not start. Check the selected AI fitting model and input curve."
                    )
                return

            if settings["auto_refine"]:
                self._start_insitu_auto_refine(record)
                return

            before = getattr(self, "fitting", None)
            old_suppress = getattr(self, "_suppress_workflow_plot_updates", False)
            self._suppress_workflow_plot_updates = (
                not self._should_refresh_insitu_views_for_current_file()
            )
            try:
                self._perform_manual_fitting()
            finally:
                self._suppress_workflow_plot_updates = old_suppress
            if getattr(self, "fitting", None) is before and before is None:
                raise RuntimeError("Manual fitting did not produce a result")
            self._complete_insitu_workflow_fit(record, "ok")
        except Exception as exc:
            record["fit_status"] = "failed"
            record["error_message"] = str(exc)
            self._finalize_insitu_workflow_file(record=record, failed=True)
