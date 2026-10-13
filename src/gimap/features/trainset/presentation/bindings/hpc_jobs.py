"""Hpc Jobs coordination for Trainset."""

from __future__ import annotations


from pathlib import Path

from typing import Dict


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.features.trainset.application import (
    RegisterTrainsetModelRequest,
)


SLURM_STATE_WORDS = {
    "PENDING": "Pending",
    "RUNNING": "Running",
    "COMPLETED": "Completed",
    "FAILED": "Failed",
    "CANCELLED": "Cancelled",
    "TIMEOUT": "Timed out",
    "OUT_OF_MEMORY": "Out of memory",
}


class HpcJobsMixin:
    """Own hpc jobs presentation behavior."""

    def _storage_acceptance_changed(self, accepted: bool) -> None:
        self.page.preview_gate_table.item(3, 1).setText("Ready" if accepted else "Pending")

    def _missing_submission_gates(self):
        missing = []
        for row in range(self.page.preview_gate_table.rowCount()):
            gate = self.page.preview_gate_table.item(row, 0).text()
            state = self.page.preview_gate_table.item(row, 1).text()
            if state != "Ready":
                missing.append(gate)
        return missing

    def _test_connection(self) -> None:
        try:
            config = self._collect_config()
            self.page.connection_button.setEnabled(False)
            self._run_worker(
                lambda: self.trainset_view_model.check_remote_connection(config),
                lambda result: (
                    self.page.connection_button.setEnabled(True),
                    self.page.preview_gate_table.item(3, 1).setText("Ready"),
                    QMessageBox.information(
                        self.window, "Maxwell", trf("Connection successful: {result}", result=result)
                    ),
                ),
                "Maxwell connection failed",
                on_error=lambda: self.page.connection_button.setEnabled(True),
            )
        except Exception as exc:
            QMessageBox.warning(self.window, "Maxwell", str(exc))

    def _submit_maxwell(self) -> None:
        missing = self._missing_submission_gates()
        if missing:
            self.page.step_list.setCurrentRow(1)
            QMessageBox.warning(
                self.window,
                "Submission checks incomplete",
                tr("Complete these checks before submitting:")
                + "\n\n"
                + "\n".join(f"• {item}" for item in missing),
            )
            return
        if not self.package_dir or not self.package_dir.exists():
            self._prepare_hpc_job()
        if not self.package_dir:
            return
        reply = QMessageBox.question(
            self.window,
            "Submit to Maxwell",
            trf("Upload and submit this job package?\n\n{path}", path=self.package_dir),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            config = self._collect_config()

            self.page.hpc_submit_button.setEnabled(False)
            self._run_worker(
                lambda: self.trainset_view_model.submit_remote_job(config, self.package_dir),
                self._submission_finished,
                "Maxwell submission failed",
                on_error=lambda: self.page.hpc_submit_button.setEnabled(True),
            )
        except Exception as exc:
            QMessageBox.warning(self.window, "Maxwell", str(exc))

    def _submission_finished(self, jobs: Dict[str, str]) -> None:
        self.page.hpc_submit_button.setEnabled(True)
        job_id = jobs["train_job_id"]
        self.config["runtime"]["last_job_id"] = job_id
        self.page.job_id_label.setText(
            trf("Generate: {generate} · Train: {train}", generate=jobs["generate_job_id"], train=job_id)
        )
        self.page.job_state.setText("SUBMITTED")
        self.page.set_step_state(3, "Submitted")
        self.page.set_step_state(4, "Job {job}", job=job_id)
        self.page.step_list.setCurrentRow(4)
        self._result_sync_started = False
        self.monitor_timer.start()
        self.status_updated.emit(trf("Submitted Maxwell jobs: {jobs}", jobs=jobs))

    def _refresh_job(self) -> None:
        if self._remote_refresh_running:
            return
        job_id = str(self.config.get("runtime", {}).get("last_job_id", ""))
        if not job_id:
            self._load_local_metrics()
            return
        try:
            config = self._collect_config()
            self._remote_refresh_running = True

            self._run_worker(
                lambda: self.trainset_view_model.query_remote_job(config, job_id),
                self._job_refreshed,
                "Job refresh failed",
                on_error=lambda: setattr(self, "_remote_refresh_running", False),
            )
        except Exception as exc:
            self._remote_refresh_running = False
            QMessageBox.warning(self.window, "Job refresh", str(exc))

    def _job_refreshed(self, payload) -> None:
        self._remote_refresh_running = False
        status, log = payload
        normalized_state = (
            status.state.upper().split("+", 1)[0].split()[0] if status.state else "UNKNOWN"
        )
        self.page.job_state.setText(status.state)  # the scheduler's own code
        # The step line in words of the interface language; an unknown code as the scheduler wrote it.
        self.page.set_step_state(4, SLURM_STATE_WORDS.get(normalized_state, status.state))
        self.page.job_id_label.setText(
            trf(
                "Job ID: {job} · Elapsed: {elapsed} · MaxRSS: {rss}",
                job=status.job_id, elapsed=status.elapsed, rss=status.max_rss,
            )
        )
        self.page.job_log.setPlainText(log or status.raw)
        terminal = normalized_state in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
        }
        if terminal:
            self.monitor_timer.stop()
        if normalized_state == "COMPLETED" and not self._result_sync_started:
            self._result_sync_started = True
            self._sync_results()

    def _sync_results(self) -> None:
        try:
            config = self._collect_config()
            destination = self._workspace() / self.config["project"]["name"] / "results"
            self._run_worker(
                lambda: self.trainset_view_model.download_remote_results(config, destination),
                lambda _result: (
                    self._load_local_metrics(),
                    QMessageBox.information(
                        self.window,
                        "Results",
                        trf("Results synchronized to:\n{path}", path=destination),
                    ),
                ),
                "Result synchronization failed",
            )
        except Exception as exc:
            QMessageBox.warning(self.window, "Results", str(exc))

    def _register_best_model(self) -> None:
        config = self._collect_config()
        roots = []
        if self.package_dir:
            roots.append(self.package_dir / "results")
        last_project = str(config.get("runtime", {}).get("last_project_dir", "")).strip()
        if last_project:
            roots.append(Path(last_project) / "results")
        roots.append(self._workspace() / config["project"]["name"] / "results")

        model_path = self.trainset_view_model.find_trained_model(roots)
        if model_path is None:
            selected, _ = QFileDialog.getOpenFileName(
                self.window,
                "Select trained model",
                str(roots[0] if roots else self.project_root),
                "Models (*.keras *.h5 *.pt *.pth);;All files (*)",
            )
            if not selected:
                return
            model_path = Path(selected)

        registered = self.trainset_view_model.register_prediction_module(
            RegisterTrainsetModelRequest(
                config=config,
                model_path=Path(model_path),
                modules_root=self.project_root / "modules",
            )
        )
        if registered is None:
            QMessageBox.warning(
                self.window,
                "Model registration",
                self.trainset_view_model.state.error_message
                or "Failed to register prediction module",
            )
            return
        self.prediction_module_registered.emit(registered.module_name)
        QMessageBox.information(
            self.window,
            "Model registered",
            trf("Registered prediction module:\n{path}", path=registered.module_dir),
        )
