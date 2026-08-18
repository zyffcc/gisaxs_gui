"""In-situ auto-refinement lifecycle and workflow completion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, QTimer

from ..binding_primitives import (
    ManualAutoRefineWorker,
    _scientific_commands,
)


class InsituRefinementLifecycleMixin:
    """In-situ auto-refinement lifecycle and workflow completion."""

    def _start_insitu_auto_refine(self, record: dict):
        if not self.fitting_view_model.storage.dependency_available("scipy"):
            raise RuntimeError("SciPy is required for Auto Refine")
        setup = self._build_manual_refine_setup()
        if setup is None:
            raise RuntimeError("Auto Refine setup failed")
        selected = self._insitu_auto_refine_selected_params(setup)
        if not selected:
            raise RuntimeError("No Auto Refine parameters selected")
        run_settings = self._ai_run_settings()
        options = {
            "max_nfev": int(run_settings.get("full_refine_max_nfev", 80)),
            "target_logrmse": float(run_settings.get("full_refine_target_logrmse", 0.0)),
            "ftol": float(run_settings.get("full_refine_ftol", 1e-8)),
            "xtol": float(run_settings.get("full_refine_xtol", 1e-8)),
            "gtol": float(run_settings.get("full_refine_gtol", 1e-8)),
            "progress_interval": int(run_settings.get("full_refine_progress_interval", 20)),
            "show_interval": 0,
        }
        self._cleanup_insitu_refine_worker()
        thread = QThread(self._insitu_workflow_parent_widget())
        worker = ManualAutoRefineWorker(self, setup, selected, options)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(lambda payload: self._on_insitu_refine_progress(payload))
        worker.finished.connect(
            lambda result, rec=record, st=setup: self._on_insitu_refine_finished(rec, st, result)
        )
        worker.failed.connect(
            lambda message, rec=record: self._on_insitu_refine_failed(rec, message)
        )
        self._insitu_workflow_refine_thread = thread
        self._insitu_workflow_refine_worker = worker
        self._log_insitu_workflow("Auto Refine started")
        thread.start()

    def _insitu_auto_refine_selected_params(self, setup):
        cached_rows = self._manual_refine_dialog_state()
        selected = []
        for desc in setup.get("params", []):
            name = str(desc.get("name", ""))
            cached = cached_rows.get(name, {}) if isinstance(cached_rows, dict) else {}
            checked = (
                bool(cached.get("checked", self._manual_refine_default_selected(name)))
                if isinstance(cached, dict)
                else self._manual_refine_default_selected(name)
            )
            if not checked:
                continue
            value = float(desc.get("value", 0.0))
            default_lo, default_hi = self._default_manual_refine_bounds(name, value)
            try:
                lo = (
                    float(cached.get("min", default_lo))
                    if isinstance(cached, dict)
                    else float(default_lo)
                )
                hi = (
                    float(cached.get("max", default_hi))
                    if isinstance(cached, dict)
                    else float(default_hi)
                )
            except Exception:
                lo, hi = float(default_lo), float(default_hi)
            if hi <= lo:
                hi = lo + max(abs(lo) * 0.1, 1e-9)
            selected.append((desc, lo, hi))
        return selected

    def _on_insitu_refine_progress(self, payload: dict):
        try:
            nfev = int(payload.get("nfev_est", payload.get("nfev", payload.get("calls", 0))) or 0)
            calls = int(payload.get("calls", 0) or 0)
            max_nfev = int(payload.get("max_nfev", 0) or 0)
            log_rmse = float(payload.get("current_log_rmse", np.nan))
            self._insitu_workflow_last_fit_status = f"Auto Refine running: eval {nfev}/{max_nfev}, calls {calls}, logRMSE {log_rmse:.6g}"
            record = getattr(self, "_insitu_workflow_current_record", None)
            if isinstance(record, dict):
                record["refine_nfev"] = nfev
                record["refine_calls"] = calls
                record["refine_max_nfev"] = max_nfev
                record["fit_status"] = "refining"
            now = time.perf_counter()
            if now - float(getattr(self, "_insitu_last_refine_ui_update", 0.0)) >= 0.5:
                self._insitu_last_refine_ui_update = now
                self._refresh_insitu_workflow_status()
            if now - float(getattr(self, "_insitu_last_refine_log_update", 0.0)) >= 2.0:
                self._insitu_last_refine_log_update = now
                self._log_insitu_workflow(self._insitu_workflow_last_fit_status)
        except Exception:
            pass

    def _on_insitu_refine_finished(self, record: dict, setup: dict, result: dict):
        if getattr(self, "_insitu_workflow_stop_requested", False):
            self._cleanup_insitu_refine_worker()
            return
        try:
            self._apply_manual_refine_result(
                setup, result.get("params"), apply_indices=result.get("selected_indices")
            )
            old_suppress = getattr(self, "_suppress_workflow_plot_updates", False)
            refresh_views = self._should_refresh_insitu_views_for_current_file()
            if refresh_views:
                self._suppress_workflow_plot_updates = False
                try:
                    self._perform_manual_fitting()
                finally:
                    self._suppress_workflow_plot_updates = old_suppress
            else:
                self._set_insitu_refine_fitting_result(setup, result.get("params"))
            self._insitu_workflow_last_fit_params = np.asarray(
                result.get("params"), dtype=float
            ).copy()
            record["refine_nfev"] = int(result.get("nfev_est", result.get("nfev", 0)) or 0)
            record["refine_calls"] = int(result.get("calls", 0) or 0)
            record["refine_max_nfev"] = int(result.get("max_nfev", 0) or 0)
            record["refine_log_rmse"] = float(result.get("final_log_rmse", np.nan))
            self._log_insitu_workflow(
                f"Auto Refine finished: eval {record['refine_nfev']}/{record['refine_max_nfev']}, "
                f"calls {record['refine_calls']}, logRMSE {record['refine_log_rmse']:.6g}",
                "SUCCESS",
            )
            self._complete_insitu_workflow_fit(record, "ok")
        except Exception as exc:
            self._on_insitu_refine_failed(record, str(exc))
        finally:
            self._cleanup_insitu_refine_worker()

    def _set_insitu_refine_fitting_result(self, setup: dict, params):
        """Store final refine fit arrays without repainting the GUI."""
        if params is None:
            raise RuntimeError("Auto Refine returned no parameters")
        params = np.asarray(params, dtype=float)
        q_raw = np.asarray(setup.get("q_raw", []), dtype=float)
        q_model = np.asarray(setup.get("q_model", []), dtype=float)
        y_fit = np.asarray(setup["model_func"](q_model, *params), dtype=float)
        n = min(q_raw.size, y_fit.size)
        if n <= 0:
            raise RuntimeError("Auto Refine produced an empty fitted curve")
        q_raw = q_raw[:n]
        y_fit = y_fit[:n]
        param_dict = {
            str(name): float(value) for name, value in zip(setup.get("param_names", []), params)
        }
        self.I_fitting = y_fit
        self.has_fitting_data = True
        self._has_fitting_data = True
        self.fitting = {
            "q": np.array(q_raw, copy=True),
            "I": np.array(y_fit, copy=True),
            "meta": {
                "shapes": list(setup.get("shapes", [])),
                "params": param_dict,
                "source": "insitu_auto_refine",
                "data_source": setup.get("q_source_kind"),
                "q_source_unit": self._get_q_source_unit(setup.get("q_source_kind")),
                "q_model_unit": "nm",
            },
        }
        self.display_mode = "fitting"
        self._display_mode = "fitting"
        self._fitting_mode_active = True

    def _on_insitu_refine_failed(self, record: dict, message: str):
        if getattr(self, "_insitu_workflow_stop_requested", False):
            self._cleanup_insitu_refine_worker()
            return
        try:
            record["fit_status"] = "failed"
            record["error_message"] = message
            self._insitu_workflow_last_fit_status = f"failed: {message}"
            self._finalize_insitu_workflow_file(record=record, failed=True)
        finally:
            self._cleanup_insitu_refine_worker()

    def _cleanup_insitu_refine_worker(self):
        worker = getattr(self, "_insitu_workflow_refine_worker", None)
        thread = getattr(self, "_insitu_workflow_refine_thread", None)
        try:
            if worker is not None:
                try:
                    worker.request_stop()
                except Exception:
                    pass
            if thread is not None:
                thread.quit()
                try:
                    thread.finished.connect(thread.deleteLater)
                except Exception:
                    thread.deleteLater()
        except Exception:
            pass
        self._insitu_workflow_refine_worker = None
        self._insitu_workflow_refine_thread = None

    def _on_insitu_ai_full_fit_finished(self, record: dict, exit_code: int, result=None):
        try:
            if exit_code != 0:
                raise RuntimeError(f"Full Auto Fit failed with exit code {exit_code}")
            output_dir = Path(getattr(self, "_ai_output_dir", "") or "")
            rows = (
                tuple(result.candidates)
                if result is not None
                else self.fitting_view_model.load_candidate_results(output_dir)
            )
            if not rows:
                raise RuntimeError("Full Auto Fit produced no candidates")
            if not self._load_ai_candidate_params(rows[0]):
                raise RuntimeError("Failed to load the best Full Auto Fit candidate")
            old_suppress = getattr(self, "_suppress_workflow_plot_updates", False)
            self._suppress_workflow_plot_updates = (
                not self._should_refresh_insitu_views_for_current_file()
            )
            try:
                self._perform_manual_fitting()
            finally:
                self._suppress_workflow_plot_updates = old_suppress
            self._log_insitu_workflow("Full Auto Fit best candidate loaded", "SUCCESS")
            if getattr(self, "_insitu_workflow_ai_then_refine", False):
                self._start_insitu_auto_refine(record)
                return
            self._complete_insitu_workflow_fit(record, "ok")
        except Exception as exc:
            record["fit_status"] = "failed"
            record["error_message"] = str(exc)
            self._insitu_workflow_last_fit_status = f"failed: {exc}"
            self._finalize_insitu_workflow_file(record=record, failed=True)
        finally:
            self._insitu_workflow_ai_record = None
            self._insitu_workflow_ai_then_refine = False

    def _complete_insitu_workflow_fit(self, record: dict, status: str):
        record["fit_status"] = status
        params = self._current_fitting_parameter_dict()
        record["fitted_parameters"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
        record.update(params)
        chi = self._calculate_current_chi_square()
        if chi is not None:
            record["chi_square"] = chi
        try:
            if self._insitu_workflow_last_fit_params is None:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if params_list:
                    self._insitu_workflow_last_fit_params = np.asarray(
                        params_list, dtype=float
                    ).copy()
        except Exception:
            pass
        self._insitu_workflow_last_fit_status = status
        self._insitu_workflow_last_chi_square = chi
        self._log_insitu_workflow(
            f"Fit completed: chi-square={self._format_optional_float(chi)}", "SUCCESS"
        )
        if self._should_refresh_insitu_views_for_current_file():
            self._draw_insitu_workflow_curve_preview()
        self._finalize_insitu_workflow_file(record=record)

    def _current_fitting_parameter_dict(self):
        try:
            if isinstance(getattr(self, "fitting", None), dict):
                params = self.fitting.get("meta", {}).get("params")
                if isinstance(params, dict):
                    return {str(k): float(v) for k, v in params.items()}
        except Exception:
            pass
        return {}

    def _calculate_current_chi_square(self):
        try:
            if not isinstance(getattr(self, "fitting", None), dict):
                return None
            fit_y = np.asarray(self.fitting.get("I", []), dtype=float).reshape(-1)
            exp_y = None
            if getattr(self, "current_cut_data", None) is not None:
                exp_y = np.asarray(
                    self.current_cut_data.get("y_intensity", []), dtype=float
                ).reshape(-1)
            elif getattr(self, "current_1d_data", None) is not None:
                exp_y = np.asarray(self.current_1d_data.get("I", []), dtype=float).reshape(-1)
            if exp_y is None:
                return None
            value = _scientific_commands(self).ai.chi_square(exp_y, fit_y)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _finalize_insitu_workflow_file(
        self,
        record: dict = None,
        load_status: str = None,
        error_message: str = "",
        failed: bool = False,
    ):
        record = record or self._insitu_workflow_current_record or {}
        refresh_views = self._should_refresh_insitu_views_for_current_file()
        if load_status is not None:
            record["load_status"] = load_status
        if error_message:
            record["error_message"] = error_message
        failed = bool(
            failed
            or record.get("error_message")
            or str(record.get("load_status", "")).startswith("failed")
            or str(record.get("fit_status", "")).startswith("failed")
        )
        if failed:
            self._log_insitu_workflow(
                f"{record.get('file_name', 'file')} failed: {record.get('error_message', '')}",
                "ERROR",
            )
        else:
            self._log_insitu_workflow(f"{record.get('file_name', 'file')} processed", "SUCCESS")
        serializable_record = json.loads(json.dumps(record, ensure_ascii=False, default=str))
        try:
            if failed:
                self.fitting_view_model.insitu.fail_insitu_file(
                    str(record.get("error_message", "failed")), serializable_record
                )
            else:
                self.fitting_view_model.insitu.complete_insitu_file(serializable_record)
            workflow_state = self.fitting_view_model.state.insitu_workflow
            self._insitu_workflow_processed_count = workflow_state.processed_count
            self._insitu_workflow_failed_count = workflow_state.failed_count
        except RuntimeError:
            # Retain the legacy callback contract for out-of-order dynamic callers.
            self._insitu_workflow_processed_count += 1
            if failed:
                self._insitu_workflow_failed_count += 1
        self._insitu_workflow_results.append(record.copy())
        self._append_insitu_session_cache(record)
        self._insitu_workflow_busy = False
        self._insitu_workflow_processing_file = None
        self._insitu_workflow_processing_batch = []
        self._insitu_workflow_current_record = None
        self._refresh_insitu_workflow_status()
        if refresh_views:
            self._schedule_insitu_trend_refresh()
        if not self._insitu_workflow_stop_requested and self._insitu_workflow_state in (
            "Watching",
            "Processing",
        ):
            QTimer.singleShot(15, self._process_next_insitu_workflow_file)
