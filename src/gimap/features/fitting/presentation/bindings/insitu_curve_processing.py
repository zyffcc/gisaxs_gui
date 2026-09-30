"""Load one curve of the in-situ series, show it, and fit it with the Recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer

from ...application import LoadCurveRequest


class InsituCurveProcessingMixin:
    """Make a series curve the current data, then run the Recipe's fit on it."""

    def _process_insitu_curve(self, record: dict, token: str) -> None:
        settings = self._insitu_workflow_settings()
        refresh_views = self._should_refresh_insitu_views_for_current_file()
        try:
            frame = self._insitu_frame_for_token(token)
            record["load_status"] = "loading"
            self._set_insitu_current_curve(frame.path)
            record["load_status"] = "ok"
            record["points"] = int(np.asarray(self.q).size)
            excluded = self._apply_deleted_point_mask_to_current_curve()
            if excluded:
                record["deleted_points_applied"] = int(excluded)
                self._log_insitu_workflow(
                    f"Applied deleted-point mask: removed {excluded} point(s) from {frame.display_name}"
                )
            self._append_insitu_heatmap_cut(self.q, self.I)
            label = (getattr(self, "_insitu_workflow_widgets", {}) or {}).get("image_label")
            if label is not None:
                label.setText(f"Current curve: {frame.display_name}")
            if refresh_views:
                self._draw_insitu_workflow_curve_preview()
            if settings.get("auto_fit"):
                QTimer.singleShot(0, lambda rec=record: self._run_insitu_workflow_fit(rec))
                return
            self._finalize_insitu_workflow_file(record=record)
        except Exception as exc:
            if record.get("load_status") in ("pending", "loading"):
                record["load_status"] = "failed"
            record["error_message"] = str(exc)
            self._finalize_insitu_workflow_file(record=record, failed=True)

    def _set_insitu_current_curve(self, path: Path) -> None:
        """The series curve becomes the fitted data (the previous frame's fit is dropped)."""
        outcome = self.fitting_view_model.load_curve(
            LoadCurveRequest(path=Path(path), q_source_unit=self._imported_1d_q_unit)
        )
        if outcome.error is not None:
            raise RuntimeError(f"[{outcome.error.code}] {outcome.error.message}")
        data = outcome.value
        self.q = data.q
        self.I = data.intensity
        self.current_1d_file_path = str(path)
        self.current_1d_data = {
            "q": data.q,
            "I": data.intensity,
            "err": data.error,
            "file_path": str(path),
            "q_source_unit": data.q_source_unit,
            "pixels": data.pixels,
            "observation": dict(data.observation),
        }
        self.data_source = "1d"
        self.display_mode = "normal"
        card = getattr(self.ui, "fittingCurveCard", None)
        if card is not None:
            card.show_curve(path, data.q, observation=data.observation, unit=data.q_source_unit)
        self.fitting = None
        self.I_fitting = None
        self.has_fitting_data = False

    def _should_refresh_insitu_views_for_current_file(self) -> bool:
        try:
            ui_every = max(1, int(self._insitu_workflow_settings().get("ui_every", 5)))
            index = int((self._insitu_workflow_current_record or {}).get("file_index", 1))
            return (index % ui_every) == 0 or not bool(getattr(self, "_insitu_workflow_queue", []))
        except Exception:
            return True

    def _apply_deleted_point_mask_to_current_curve(self) -> int:
        """Drop the globally deleted q points from the current curve (all columns together)."""
        excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
        data = getattr(self, "current_1d_data", None)
        if not excluded or not isinstance(data, dict):
            return 0
        q = np.asarray(data.get("q", []), dtype=float).reshape(-1)
        if q.size == 0:
            return 0
        keep = np.array(
            [
                self._ai_q_key(value) not in excluded
                and self._ai_q_key(abs(float(value))) not in excluded
                for value in q
            ],
            dtype=bool,
        )
        removed = int(q.size - np.count_nonzero(keep))
        if removed <= 0 or not keep.any():
            return 0
        for key in ("q", "I", "err", "pixels"):
            values = data.get(key)
            if values is not None and np.asarray(values).size == q.size:
                data[key] = np.asarray(values)[keep]
        self.q = data["q"]
        self.I = data["I"]
        return removed

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
                recipe = self.fitting_view_model.insitu.recipe
                workflow_options = recipe.model.get("workflow_v5", {}) if recipe else {}
                numerical = workflow_options.get("numerical", True)
                method = workflow_options.get("method", "model")
                mode = method if method in ("stable", "experimental") else ("full" if numerical else "fast")
                self._start_ai_prediction(mode)
                if getattr(self, "_ai_job_thread", None) is None:
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


__all__ = ["InsituCurveProcessingMixin"]
