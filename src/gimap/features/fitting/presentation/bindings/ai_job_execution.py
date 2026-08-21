"""Ai Job Execution for fitting presentation."""

from __future__ import annotations

import re

import time


from pathlib import Path


from PyQt5.QtCore import QThread, QUrl

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.fitting.application import (
    CandidateGenerationRequest,
)

from src.gimap.features.fitting.presentation.ai_worker import AiCandidateWorker

from PyQt5.QtGui import QDesktopServices

from ..binding_primitives import (
    _ai_catalog,
)


class AiJobExecutionMixin:
    """Own ai job execution behavior."""

    def _ai_prediction_output_root(self) -> Path:
        return Path.cwd() / "AI_Fitting_Output"

    def _ai_current_prediction_dir(self) -> Path:
        return self._ai_prediction_output_root() / "current_prediction"

    def _prepare_ai_prediction_io(self) -> tuple[Path, Path] | None:
        arrays = self._current_ai_curve_arrays()
        if arrays is None:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting",
                "No valid AI input curve is loaded. Load or cut a 1D curve before prediction.",
            )
            return None
        q_arr, i_arr, sigma_arr = arrays
        out_dir = self._ai_current_prediction_dir()
        input_csv = out_dir / "input_curve.csv"
        self._ai_prepared_curve = (q_arr, i_arr, sigma_arr)
        self._ai_output_dir = out_dir
        self._ai_input_csv = input_csv
        return input_csv, out_dir

    def _ai_exact_nonempty_arg(self):
        constraints = self.build_ai_constraints_json_from_ui()
        exact = constraints.get("exact_nonempty")
        try:
            exact = int(exact)
        except Exception:
            exact = None
        return exact if exact and exact > 0 else None

    def _start_ai_prediction(self, run_mode: str = "fast") -> None:
        thread = getattr(self, "_ai_job_thread", None)
        if thread is not None and thread.isRunning():
            self._set_ai_workspace_status("AI prediction is already running.", None)
            return

        model_path = self._selected_ai_model_path()
        if model_path is None or not model_path.exists():
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting",
                "Please import or select a valid AI fitting model first.",
            )
            return
        io_paths = self._prepare_ai_prediction_io()
        if io_paths is None:
            return
        _input_csv, output_dir = io_paths
        q_values, intensity, sigma = self._ai_prepared_curve
        profile = (
            _ai_catalog(self).profile("Fast") if run_mode == "fast" else self._current_ai_profile()
        )
        constraints = self.build_ai_constraints_json_from_ui()
        request = CandidateGenerationRequest(
            model_path=model_path,
            output_dir=output_dir,
            q=q_values,
            intensity=intensity,
            sigma=sigma,
            profile=profile.to_dict(),
            constraints=constraints,
            exact_nonempty=self._ai_exact_nonempty_arg(),
            clear_output_dir=True,
        )

        thread = QThread(self.main_window or self.ui)
        worker = AiCandidateWorker(
            self.fitting_view_model,
            request,
            refine=run_mode != "fast",
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_ai_job_progress)
        worker.completed.connect(self._on_ai_job_finished)
        worker.failed.connect(self._on_ai_job_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_ai_job_thread)
        self._ai_job_thread = thread
        self._ai_job_worker = worker
        self._ai_active_profile = profile
        self._ai_run_started_at = time.perf_counter()
        self._ai_run_cancelled = False
        self._ai_candidate_rows = []
        self._begin_fitting_step("fit", f"Running {profile.name} AI fitting")
        self._set_ai_running_state(True)
        self._set_ai_workspace_status(
            f"Starting {profile.name} AI fitting run...",
            0,
        )
        self._append_ai_log(f"JobRunner request: profile={profile.name}, model={model_path}")
        thread.start()

    def _set_ai_running_state(self, running: bool) -> None:
        for button in getattr(self, "_ai_action_buttons", []) or []:
            text = button.text()
            if text in ("Fast Predict", "Full Auto Fit", "Run AI Auto Fit"):
                button.setEnabled(not running)
        for name in ("aiFittingFastPredictButton", "aiFittingFullAutoFitButton"):
            button = getattr(self.ui, name, None)
            if button is not None:
                button.setEnabled(not running)
        main_stop = getattr(self.ui, "aiFittingStopButton", None)
        if main_stop is not None:
            main_stop.setEnabled(running)
        if self._ai_stop_button is not None:
            self._ai_stop_button.setEnabled(running)
        can_export = bool(getattr(self, "_ai_output_dir", None)) and not running
        if self._ai_open_output_button is not None:
            self._ai_open_output_button.setEnabled(bool(getattr(self, "_ai_output_dir", None)))
        if self._ai_export_output_button is not None:
            self._ai_export_output_button.setEnabled(can_export)
        main_export = getattr(self.ui, "aiFittingExportOutputButton", None)
        if main_export is not None:
            main_export.setEnabled(can_export)

    def _append_ai_log(self, text: str) -> None:
        text = str(text).rstrip()
        if not text:
            return
        if not isinstance(getattr(self, "_ai_log_lines", None), list):
            self._ai_log_lines = []
        self._ai_log_lines.append(text)
        if len(self._ai_log_lines) > 2000:
            self._ai_log_lines = self._ai_log_lines[-2000:]
        browser = getattr(self, "_ai_log_browser", None)
        if browser is not None:
            browser.append(text)
        out_dir = getattr(self, "_ai_output_dir", None)
        if out_dir:
            try:
                self.fitting_view_model.storage.append_ai_log(Path(out_dir), text)
            except Exception:
                pass

    def _on_ai_job_progress(self, progress) -> None:
        line = str(getattr(progress, "message", "") or "")
        if line:
            self._append_ai_log(line)
            self._handle_ai_process_text(line, append_log=False)
        else:
            self._set_ai_workspace_status(
                "AI fitting running...",
                int(100 * float(getattr(progress, "fraction", 0.0))),
            )

    def _handle_ai_process_text(self, text: str, *, append_log: bool = True) -> None:
        for line in str(text).splitlines():
            if append_log:
                self._append_ai_log(line)
            match = re.search(r"Progress\s+(\d+)/(\d+)", line)
            if match:
                current = int(match.group(1))
                total = max(1, int(match.group(2)))
                self._set_ai_workspace_status(
                    f"Sampling progress {current}/{total}",
                    int(current * 100 / total),
                )
                continue
            refine_match = re.search(
                r"refine\s+#(\d+)/(\d+)\s+nfev~(\d+)/(\d+)",
                line,
            )
            if refine_match:
                index = int(refine_match.group(1))
                total = max(1, int(refine_match.group(2)))
                nfev = int(refine_match.group(3))
                max_nfev = max(1, int(refine_match.group(4)))
                fraction = ((index - 1) + min(1.0, nfev / max_nfev)) / total
                self._set_ai_workspace_status(line[:180], int(100 * fraction))
            elif "Refine #" in line:
                self._set_ai_workspace_status(line[:180], None)
            elif line.startswith("Wrote "):
                self._set_ai_workspace_status(line, 100)

    def _on_ai_job_finished(self, result) -> None:
        self._set_ai_running_state(False)
        self._ai_output_dir = Path(result.output_dir)
        self._ai_candidate_rows = [dict(row) for row in result.candidates]
        self._append_ai_log(
            f"Run summary: profile={result.profile_name}, "
            f"runtime={result.runtime_seconds:.3f}s, "
            f"configured_candidates={result.configured_candidates}, "
            f"results={len(result.candidates)}, "
            f"best_logRMSE={result.best_log_rmse}"
        )
        workflow_record = getattr(self, "_insitu_workflow_ai_record", None)
        if workflow_record is not None:
            self._on_insitu_ai_full_fit_finished(
                workflow_record,
                int(result.exit_code),
                result=result,
            )
            return
        self._set_ai_workspace_status(
            f"AI fitting finished in {result.runtime_seconds:.2f}s. Output: {result.output_dir}",
            100,
        )
        self._complete_fitting_step(
            "fit", f"AI fitting completed · {len(result.candidates)} candidates"
        )
        self._show_ai_candidate_table(result.output_dir, rows=result.candidates)

    def _on_ai_job_error(self, code: str, message: str) -> None:
        self._set_ai_running_state(False)
        self._append_ai_log(f"AI fitting error [{code}]: {message}")
        workflow_record = getattr(self, "_insitu_workflow_ai_record", None)
        if workflow_record is not None:
            workflow_record["fit_status"] = "failed"
            workflow_record["error_message"] = f"AI job error [{code}]: {message}"
            self._insitu_workflow_ai_record = None
            self._insitu_workflow_ai_then_refine = False
            self._finalize_insitu_workflow_file(
                record=workflow_record,
                failed=True,
            )
            return
        if code == "cancelled":
            self._fail_fitting_step("fit", "AI fitting cancelled")
            self._set_ai_workspace_status("AI fitting cancelled.", 0)
        else:
            self._fail_fitting_step("fit", message)
            self._set_ai_workspace_status(
                f"AI fitting failed [{code}]: {message}",
                0,
            )

    def _cleanup_ai_job_thread(self) -> None:
        self._ai_job_worker = None
        self._ai_job_thread = None

    def _stop_ai_fitting_process(self) -> None:
        thread = getattr(self, "_ai_job_thread", None)
        if thread is None or not thread.isRunning():
            return
        self._append_ai_log("Stopping AI fitting job...")
        self._ai_run_cancelled = True
        if not self.fitting_view_model.cancel_ai_candidates():
            self._append_ai_log("AI fitting job is already stopping or finished.")

    def _stop_ai_fitting_on_budget(self) -> None:
        self._append_ai_log("AI fitting time budget reached; requesting cancellation.")
        self._stop_ai_fitting_process()

    def _open_ai_output_folder(self) -> None:
        out_dir = getattr(self, "_ai_output_dir", None)
        if out_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(out_dir))))
