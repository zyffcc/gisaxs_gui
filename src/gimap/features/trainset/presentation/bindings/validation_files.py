"""Validation Files coordination for Trainset."""

from __future__ import annotations


import sys

from pathlib import Path


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from src.gimap.app.presentation.i18n import tr, trf
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
            self.page.set_validation_state("Configuration valid", "ok")
            self.page.preview_gate_table.item(0, 1).setText("Ready")
            text = tr("Configuration is valid.")
            if warnings:
                text += "\n\n" + tr("Warnings:") + "\n" + "\n".join(f"• {item}" for item in warnings)
            QMessageBox.information(self.window, "Validation", text)
        else:
            self.page.set_validation_state("Validation failed", "error")
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
                self.page.texts.set(
                    self.page.model_summary,
                    "Static tensor contract\n\n{summary}\n\nTensorFlow forward pass unavailable: {error}",
                    setter="setPlainText",
                    summary=result.static_summary,
                    error=result.runtime_error,
                )
            else:
                self.page.texts.set(
                    self.page.model_summary,
                    "Forward pass OK\n\n{summary}\n\nBatch output: {shape}\nTrainable weights: {weights:,}",
                    setter="setPlainText",
                    summary=result.static_summary,
                    shape=result.output_shape,
                    weights=result.trainable_weights,
                )
            self.page.preview_gate_table.item(2, 1).setText("Ready")
            self.page.set_step_state(2, "Contract ready")
        except Exception as exc:
            QMessageBox.warning(self.window, "Model validation", str(exc))

    def _save_project_dialog(self) -> None:
        config = self._collect_config()
        folder = self._start_folder("project", self.project_root)
        default = Path(folder) / f"{config['project']['name']}.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self.window, tr("Save trainset project"), str(default), "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if path:
            self._remember_folder("project", path)
            self.trainset_view_model.save_project(config, Path(path))
            self.status_updated.emit(trf("Saved trainset project: {path}", path=path))

    def _load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            tr("Load trainset project"),
            self._start_folder("project", self.project_root),
            "Project configuration (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        self._remember_folder("project", path)
        try:
            # set_parameters marks the loaded design not validated (not "changed since validation").
            self.set_parameters(self.trainset_view_model.load_project(Path(path)))
            self.status_updated.emit(trf("Loaded trainset project: {path}", path=path))
        except Exception as exc:
            QMessageBox.critical(self.window, "Project load failed", str(exc))

    def _start_folder(self, kind: str, fallback) -> str:
        """Where a chooser opens: the folder it last ended in (user preferences), else ``fallback``."""
        return self.trainset_view_model.last_folder(kind) or str(fallback or "")

    def _remember_folder(self, kind: str, path) -> None:
        self.trainset_view_model.remember_folder(kind, path)

    def _choose_folder_into(self, field: str, kind: str, title: str, fallback) -> None:
        """A folder chooser for one Local Run path field: opens at the field's folder or the last one used."""
        current = self.page.fields[field].text().strip()
        start = current if current and Path(current).is_dir() else self._start_folder(kind, fallback)
        path = QFileDialog.getExistingDirectory(self.window, tr(title), start)
        if path:
            self._remember_folder(kind, path)
            self.page.fields[field].setText(path)

    def _choose_workspace(self) -> None:
        self._choose_folder_into(
            "project.workspace", "workspace", "Choose local trainset workspace", self.project_root
        )

    def _choose_dataset_folder(self) -> None:
        self._choose_folder_into(
            "runtime.dataset_output_dir", "dataset", "Choose generated dataset folder", self._workspace()
        )

    def _choose_results_folder(self) -> None:
        self._choose_folder_into(
            "runtime.results_output_dir", "results", "Choose training results folder", self._workspace()
        )

    def _choose_cache_folder(self) -> None:
        self._choose_folder_into(
            "simulation.grid_cache.directory", "cache", "Choose BornAgain grid cache folder", self._workspace()
        )

    def _choose_local_python(self) -> None:
        current = self.page.fields["training.local_python"].text().strip()
        start = str(Path(current).parent) if current else self._start_folder("python", Path(sys.executable).parent)
        selected, _ = QFileDialog.getOpenFileName(
            self.window,
            tr("Choose local Python executable"),
            start,
            "Python executable (python.exe python);;All files (*)",
        )
        if selected:
            self._remember_folder("python", selected)
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
