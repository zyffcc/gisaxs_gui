"""Validation Files coordination for Trainset."""

from __future__ import annotations


import sys

from pathlib import Path


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from src.gimap.features.trainset.application import (
    ModelContractRequest,
)


class ValidationFilesMixin:
    """Own validation files presentation behavior."""

    def _validate_and_report(self) -> bool:
        config = self._collect_config()
        valid, errors, warnings = self.trainset_view_model.validate_config(
            config,
            simulation_available=self.simulation_port.is_available(),
        )
        if valid:
            self.page.validation_badge.setText("Configuration valid")
            self.page.preview_gate_table.item(0, 1).setText("Ready")
            text = "Configuration is valid."
            if warnings:
                text += "\n\nWarnings:\n" + "\n".join(f"• {item}" for item in warnings)
            QMessageBox.information(self.window, "Validation", text)
        else:
            self.page.validation_badge.setText("Validation failed")
            QMessageBox.warning(
                self.window, "Validation", "\n".join(f"• {item}" for item in errors)
            )
        return valid

    def _validate_model_contract(self) -> None:
        try:
            config = self._collect_config()
            height, width = int(config["roi"]["height"]), int(config["roi"]["width"])
            outputs = len(self.catalog.trainable_names(config))
            if outputs < 1:
                raise ValueError("At least one physics parameter needs a non-zero range.")
            result = self.trainset_view_model.validate_model_contract(
                ModelContractRequest(
                    input_shape=(height, width, 1),
                    output_size=outputs,
                    model_config=config["model"],
                )
            )
            if result is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "Model validation failed"
                )
            if result.runtime_error is not None:
                summary = (
                    f"Static tensor contract\n\n{result.static_summary}\n\n"
                    f"TensorFlow forward pass unavailable: {result.runtime_error}"
                )
            else:
                summary = (
                    f"Forward pass OK\n\n{result.static_summary}\n\n"
                    f"Batch output: {result.output_shape}\n"
                    f"Trainable weights: {result.trainable_weights:,}"
                )
            self.page.model_summary.setPlainText(summary)
            self.page.preview_gate_table.item(2, 1).setText("Ready")
            self.page.set_step_state(2, "Contract ready")
        except Exception as exc:
            QMessageBox.warning(self.window, "Model validation", str(exc))

    def _save_project_dialog(self) -> None:
        config = self._collect_config()
        default = self.project_root / f"{config['project']['name']}.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Save trainset project", str(default), "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if path:
            self.trainset_view_model.save_project(config, Path(path))
            self.status_updated.emit(f"Saved trainset project: {path}")

    def _load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load trainset project",
            str(self.project_root),
            "Project configuration (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.set_parameters(self.trainset_view_model.load_project(Path(path)))
            self.status_updated.emit(f"Loaded trainset project: {path}")
        except Exception as exc:
            QMessageBox.critical(self.window, "Project load failed", str(exc))

    def _choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self.window, "Choose local trainset workspace", str(self.project_root)
        )
        if path:
            self.page.fields["project.workspace"].setText(path)

    def _choose_dataset_folder(self) -> None:
        current = self.page.fields["runtime.dataset_output_dir"].text().strip()
        path = QFileDialog.getExistingDirectory(
            self.window,
            "Choose generated dataset folder",
            current or str(self._workspace()),
        )
        if path:
            self.page.fields["runtime.dataset_output_dir"].setText(path)

    def _choose_results_folder(self) -> None:
        current = self.page.fields["runtime.results_output_dir"].text().strip()
        path = QFileDialog.getExistingDirectory(
            self.window,
            "Choose training results folder",
            current or str(self._workspace()),
        )
        if path:
            self.page.fields["runtime.results_output_dir"].setText(path)

    def _choose_cache_folder(self) -> None:
        current = self.page.fields["simulation.grid_cache.directory"].text().strip()
        path = QFileDialog.getExistingDirectory(
            self.window,
            "Choose BornAgain grid cache folder",
            current or str(self._workspace()),
        )
        if path:
            self.page.fields["simulation.grid_cache.directory"].setText(path)

    def _choose_local_python(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self.window,
            "Choose local Python executable",
            str(Path(sys.executable).parent),
            "Python executable (python.exe python);;All files (*)",
        )
        if selected:
            self.page.fields["training.local_python"].setText(selected)

    def _workspace(self) -> Path:
        configured = self.page.fields["project.workspace"].text().strip()
        return Path(configured) if configured else self.project_root / "trainset_jobs"

    def _dataset_output_dir(self) -> Path:
        configured = self.page.fields["runtime.dataset_output_dir"].text().strip()
        if configured:
            return Path(configured)
        return (
            self.package_dir or self._workspace() / self.page.project_name.text().strip()
        ) / "dataset"

    def _results_output_dir(self) -> Path:
        configured = self.page.fields["runtime.results_output_dir"].text().strip()
        if configured:
            return Path(configured)
        return (
            self.package_dir or self._workspace() / self.page.project_name.text().strip()
        ) / "results"

    def _prepare_local_job(self) -> None:
        self._prepare_job(local=True)

    def _prepare_hpc_job(self) -> None:
        self._prepare_job(local=False)
