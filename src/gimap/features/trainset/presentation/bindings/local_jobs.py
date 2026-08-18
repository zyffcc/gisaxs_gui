"""Local Jobs coordination for Trainset."""

from __future__ import annotations


import sys

from pathlib import Path


from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.trainset.application import (
    PrepareTrainsetJobRequest,
    TrainsetLocalProcessRequest,
)


from ..background_tasks import _FunctionWorker


class LocalJobsMixin:
    """Own local jobs presentation behavior."""

    def _prepare_job(self, local: bool) -> None:
        config = self._collect_config()
        config["training"]["backend"] = "local" if local else "slurm"
        valid, errors, _warnings = self.trainset_view_model.validate_config(
            config,
            simulation_available=self.simulation_port.is_available(),
        )
        if not valid:
            QMessageBox.warning(self.window, "Job package blocked", "\n".join(errors))
            return
        try:
            self.package_dir = self.trainset_view_model.prepare_job_package(
                PrepareTrainsetJobRequest(
                    config,
                    self._workspace(),
                    self.project_root,
                )
            )
            if self.package_dir is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "Failed to prepare job package"
                )
            config["runtime"]["last_project_dir"] = str(self.package_dir)
            self.page.package_tree.setPlainText(
                f"Prepared: {self.package_dir}\n\n"
                "config.yaml\nmanifest.json\ngenerate_dataset.py\ntrain.py\nvalidate_config.py\n"
                "environment.yml\nslurm_generate.sh\nslurm_train.sh\nsrc/trainset/\nsrc/calibration/\ndataset/\nresults/\nlogs/"
            )
            self.page.set_step_state(3, "Package ready")
            self.status_updated.emit("Reproducible local/HPC job package prepared")
        except Exception as exc:
            QMessageBox.critical(self.window, "Prepare job failed", str(exc))

    def _ensure_package(self) -> bool:
        if self.package_dir and self.package_dir.exists():
            return True
        self._prepare_local_job()
        return bool(self.package_dir and self.package_dir.exists())

    def _start_local_process(self, arguments, follow_up=None) -> None:
        if not self._ensure_package():
            return
        if self.trainset_view_model.local_process_running():
            QMessageBox.information(
                self.window,
                "Local backend",
                "A local generation/training process is already running.",
            )
            return
        self._pending_local_arguments = follow_up
        self._local_paused = False
        self.page.local_pause_button.setText("Pause")
        python_executable = (
            self.page.fields["training.local_python"].text().strip() or sys.executable
        )
        started = self.trainset_view_model.start_local_process(
            TrainsetLocalProcessRequest(
                package_dir=self.package_dir,
                python_executable=Path(python_executable),
                arguments=tuple(str(value) for value in arguments),
            ),
            on_started=self._local_process_started,
            on_progress=self._local_process_progress,
            on_log=self.page.job_log.append,
            on_finished=self._local_process_finished,
            on_error=self.generation_error.emit,
        )
        if not started:
            QMessageBox.warning(
                self.window,
                "Local backend",
                self.trainset_view_model.state.error_message or "Failed to start local process",
            )
            return
        self.page.step_list.setCurrentRow(4)

    def _local_process_started(self) -> None:
        self.generation_started.emit()
        self.page.job_state.setText("RUNNING")
        self.page.set_step_state(4, "RUNNING")
        self.page.set_local_job_status("running", "Starting local process…", 0)
        self.page.local_pause_button.setEnabled(True)
        self.page.local_stop_button.setEnabled(True)

    def _local_process_progress(self, percent: int, message: str) -> None:
        self.page.set_local_job_status("running", message, percent)
        self.progress_updated.emit(percent)

    def _toggle_local_pause(self) -> None:
        if not self.trainset_view_model.local_process_running():
            return
        paused = not self._local_paused
        if not self.trainset_view_model.set_local_process_paused(paused):
            return
        self._local_paused = paused
        self.page.local_pause_button.setText("Resume" if self._local_paused else "Pause")
        self.page.set_local_job_status(
            "paused" if self._local_paused else "running",
            (
                "Paused safely between BornAgain simulations/batches."
                if self._local_paused
                else "Resuming local process…"
            ),
            self.page.local_progress.value(),
        )

    def _stop_local_process(self) -> None:
        if self.trainset_view_model.cancel_local_process():
            self.page.set_local_job_status(
                "running",
                "Stopping safely after the current simulation/batch…",
                self.page.local_progress.value(),
            )
            self.page.local_pause_button.setEnabled(False)

    def _run_local_physical_test(self) -> None:
        """Generate a small, genuinely physical BornAgain dataset."""
        samples = int(self._collect_config().get("training", {}).get("smoke_samples", 64))
        self._start_local_process(
            [
                "generate_dataset.py",
                "--samples",
                str(samples),
                "--mode",
                "full",
                "--output",
                str(self._dataset_output_dir()),
            ]
        )

    def _run_local_generation(self) -> None:
        count = int(self._collect_config()["dataset"]["number_of_samples"])
        self._start_local_process(
            [
                "generate_dataset.py",
                "--samples",
                str(count),
                "--mode",
                "full",
                "--output",
                str(self._dataset_output_dir()),
            ]
        )

    def _run_local_training(self) -> None:
        self._start_local_process(
            [
                "train.py",
                "--dataset",
                str(self._dataset_output_dir()),
                "--output",
                str(self._results_output_dir()),
            ]
        )

    def _run_local_smoke_test(self) -> None:
        config = self._collect_config()
        reference = str(config.get("project", {}).get("reference_file", ""))
        if not reference or not Path(reference).exists():
            QMessageBox.information(
                self.window,
                "Local smoke test",
                "Load a real reference image first. It is used only to test the local data/model pipeline.",
            )
            return
        self._prepare_local_job()
        if not self.package_dir:
            return
        samples = int(config.get("training", {}).get("smoke_samples", 64))
        epochs = int(config.get("training", {}).get("smoke_epochs", 2))
        self.page.job_log.clear()
        self.page.job_log.append(
            "LIGHTWEIGHT DEMO: reference-derived images test I/O and training only; they are not a physical BornAgain dataset."
        )
        self._start_local_process(
            [
                "generate_dataset.py",
                "--samples",
                str(samples),
                "--mode",
                "demo",
                "--output",
                str(self._dataset_output_dir()),
            ],
            follow_up=[
                "train.py",
                "--smoke",
                "--epochs",
                str(epochs),
                "--dataset",
                str(self._dataset_output_dir()),
                "--output",
                str(self._results_output_dir()),
            ],
        )

    def _local_process_finished(self, exit_code: int, _status=None) -> None:
        state = "COMPLETED" if exit_code == 0 else "FAILED"
        self.page.job_state.setText(state)
        self.page.set_step_state(4, state)
        self.page.local_pause_button.setEnabled(False)
        self.page.local_stop_button.setEnabled(False)
        self.page.local_pause_button.setText("Pause")
        self.page.set_local_job_status(
            "succeeded" if exit_code == 0 else "failed",
            "Completed." if exit_code == 0 else f"Stopped or failed (exit code {exit_code}).",
            100 if exit_code == 0 else self.page.local_progress.value(),
        )
        if exit_code == 0:
            self.generation_finished.emit()
            pending = self._pending_local_arguments
            self._pending_local_arguments = None
            if pending:
                self.page.job_log.append("Starting lightweight training…")
                QTimer.singleShot(0, lambda: self._start_local_process(pending))
                return
        else:
            self._pending_local_arguments = None
            self.generation_error.emit(f"Local process exited with code {exit_code}")
        self._load_local_metrics()

    def _load_local_metrics(self) -> None:
        if not self.package_dir:
            return
        records = self.trainset_view_model.load_metrics(
            self._results_output_dir() / "metrics.jsonl"
        )
        self.page.metrics_table.setRowCount(0)
        from PyQt5.QtWidgets import QTableWidgetItem

        for row, record in enumerate(records):
            self.page.metrics_table.insertRow(row)
            values = (
                record.get("epoch", ""),
                record.get("loss", record.get("train_loss", "")),
                record.get("val_loss", ""),
                record.get("lr", ""),
            )
            for column, value in enumerate(values):
                self.page.metrics_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _run_worker(self, function, success, title: str, on_error=None) -> None:
        worker = _FunctionWorker(function)
        worker.signals.finished.connect(success)

        def report_error(message: str) -> None:
            if on_error is not None:
                on_error()
            QMessageBox.critical(self.window, title, message)

        worker.signals.error.connect(report_error)
        self.thread_pool.start(worker)
