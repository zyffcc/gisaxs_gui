"""Fitting Parameter Persistence for fitting presentation."""

from __future__ import annotations

import datetime

import copy

from pathlib import Path

from PyQt5.QtCore import Qt, QPoint

from PyQt5.QtWidgets import (
    QFileDialog,
    QWidget,
    QMenu,
)


from src.gimap.shared.file_paths import normalize_path


class FittingParameterPersistenceMixin:
    """Own fitting parameter persistence behavior."""

    def _build_fitting_parameter_snapshot(self) -> dict:
        """Return a portable fitting-parameter snapshot."""
        try:
            self.save_particle_parameters()
        except Exception:
            pass
        model_section = {}
        try:
            model_section = copy.deepcopy(
                self.model_params_manager.get_parameter("fitting", None, {})
            )
        except Exception:
            model_section = {}
        return {
            "schema": "gimap_fitting_parameters_v1",
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "fitting": copy.deepcopy(self.get_parameters()),
            "model_parameters": {
                "fitting": model_section,
            },
        }

    def save_fitting_parameters_to_file(self, filepath: str) -> bool:
        """Save only Cut/Fitting parameters, including particle/global model params."""
        try:
            filepath = normalize_path(filepath)
            self.fitting_view_model.storage.save_parameter_snapshot(
                Path(filepath),
                self._build_fitting_parameter_snapshot(),
            )
            self._add_fitting_success(f"Fitting parameters saved to: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to save fitting parameters: {e}")
            return False

    def load_fitting_parameters_from_file(self, filepath: str) -> bool:
        """Load a Cut/Fitting parameter snapshot and refresh the fitting UI."""
        try:
            filepath = normalize_path(filepath)
            payload = self.fitting_view_model.storage.load_parameter_snapshot(Path(filepath))

            fitting_params = payload.get("fitting") if isinstance(payload, dict) else None
            if isinstance(fitting_params, dict):
                self.set_parameters(fitting_params)

            model_fitting = None
            if isinstance(payload, dict):
                model_params = payload.get("model_parameters")
                if isinstance(model_params, dict):
                    model_fitting = model_params.get("fitting")
                if model_fitting is None and (
                    "particles" in payload or "global_parameters" in payload
                ):
                    model_fitting = payload
                if (
                    model_fitting is None
                    and "fitting" in payload
                    and isinstance(payload.get("fitting"), dict)
                ):
                    maybe = payload["fitting"]
                    if "particles" in maybe or "global_parameters" in maybe:
                        model_fitting = maybe

            if isinstance(model_fitting, dict):
                self.model_params_manager.replace_section("fitting", copy.deepcopy(model_fitting))
                self.model_params_manager.save_parameters()
                self.reload_particle_parameters()

            self.parameters_changed.emit(self.current_parameters)
            self._add_fitting_success(f"Fitting parameters loaded from: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to load fitting parameters: {e}")
            return False

    def save_fitting_parameters_dialog(self) -> bool:
        filepath, _ = QFileDialog.getSaveFileName(
            self.main_window or self.ui,
            "Save Fitting Parameters",
            "config/fitting_parameters.json",
            "JSON Files (*.json);;All Files (*)",
        )
        return self.save_fitting_parameters_to_file(filepath) if filepath else False

    def load_fitting_parameters_dialog(self) -> bool:
        filepath, _ = QFileDialog.getOpenFileName(
            self.main_window or self.ui,
            "Load Fitting Parameters",
            "config/",
            "JSON Files (*.json);;All Files (*)",
        )
        return self.load_fitting_parameters_from_file(filepath) if filepath else False

    def _setup_fitting_parameters_context_menu(self) -> None:
        names = (
            "FittingControlsCard",
            "ModelParameterCard",
            "gisaxsFixedControlsStack",
            "gisaxsWorkAreaContents",
            "sampleParametersBox",
            "fitBox",
            "gisaxsFittingPageScrollAreaWidgetContents",
        )
        for name in names:
            widget = getattr(self.ui, name, None)
            if widget is None:
                root = getattr(self.ui, "centralwidget", None)
                widget = root.findChild(QWidget, name) if root is not None else None
            if widget is None:
                continue
            try:
                widget.setContextMenuPolicy(Qt.CustomContextMenu)
                widget.customContextMenuRequested.connect(
                    self._show_fitting_parameters_context_menu
                )
            except RuntimeError:
                # Some generated containers are intentionally replaced by the runtime layout wrapper.
                # PyQt keeps the Python attribute even after Qt deletes the C++ object.
                continue
            except Exception:
                pass

    def _show_fitting_parameters_context_menu(self, pos: QPoint) -> None:
        widget = self.sender()
        if widget is None:
            return
        try:
            global_pos = widget.mapToGlobal(pos)
        except RuntimeError:
            return
        menu = QMenu(widget)
        save_action = menu.addAction("Save Fitting Parameters...")
        load_action = menu.addAction("Load Fitting Parameters...")
        menu.addSeparator()
        export_particles_action = menu.addAction("Export Particle Parameters Only...")
        import_particles_action = menu.addAction("Import Particle Parameters Only...")
        reload_action = menu.addAction("Reload Parameters from Config")
        menu.addSeparator()
        ai_action = menu.addAction("Open AI Fitting Workspace...")
        action = menu.exec_(global_pos)
        if action == save_action:
            self.save_fitting_parameters_dialog()
        elif action == load_action:
            self.load_fitting_parameters_dialog()
        elif action == export_particles_action:
            filepath, _ = QFileDialog.getSaveFileName(
                self.main_window or self.ui,
                "Export Particle Parameters",
                "config/model_parameters_fitting.json",
                "JSON Files (*.json);;All Files (*)",
            )
            if filepath:
                self.export_particle_parameters(filepath)
        elif action == import_particles_action:
            filepath, _ = QFileDialog.getOpenFileName(
                self.main_window or self.ui,
                "Import Particle Parameters",
                "config/",
                "JSON Files (*.json);;All Files (*)",
            )
            if filepath:
                self.import_particle_parameters(filepath)
        elif action == reload_action:
            self.reload_particle_parameters()
        elif action == ai_action:
            self.open_ai_fitting_workspace()
