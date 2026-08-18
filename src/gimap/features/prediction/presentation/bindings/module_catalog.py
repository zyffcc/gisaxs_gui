"""Module Catalog coordination for Prediction."""

from __future__ import annotations

import os


from pathlib import Path

from typing import Dict, Optional


from PyQt5.QtCore import QSignalBlocker, QEvent, QTimer, QUrl

from PyQt5.QtGui import QDesktopServices

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from src.gimap.shared.file_paths import normalize_path


class ModuleCatalogMixin:
    """Own module catalog presentation behavior."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        try:
            combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
            if combo and obj is combo and event is not None:
                if event.type() in (QEvent.MouseButtonPress, QEvent.FocusIn):
                    # Refresh module list when user is about to open/select
                    self._refresh_modules()

        except Exception:
            pass
        try:
            return super().eventFilter(obj, event)
        except (RuntimeError, SystemError):
            # Qt may deliver a final event while the watched C++ widget is
            # already being destroyed. Treat that teardown event as unhandled
            # instead of letting a stale wrapper abort the application.
            return False

    def _populate_framework_combo(self, combo) -> None:
        available = self.detect_available_frameworks()
        options = [
            label
            for label in available.values()
            if self.is_framework_compatible(self._current_module, label)
        ]
        if not options:
            options = ["No compatible framework installed"]
        try:
            if options and not options[0].startswith("No compatible"):
                self.current_parameters["framework"] = options[0]
        except Exception:
            pass
        blocker = QSignalBlocker(combo)
        combo.clear()
        combo.addItems(options)
        combo.setEnabled(bool(options) and not options[0].startswith("No compatible"))
        del blocker
        self._refresh_framework_status()

    def detect_available_frameworks(self) -> Dict[str, str]:
        frameworks: Dict[str, str] = {}
        try:
            from importlib.metadata import version

            try:
                frameworks["tensorflow"] = f"tensorflow {version('tensorflow')}"
            except Exception:
                pass
            try:
                frameworks["torch"] = f"torch {version('torch')}"
            except Exception:
                pass
        except Exception:
            pass
        return frameworks

    def is_framework_compatible(
        self, module: Optional[Dict[str, object]], framework_text: str
    ) -> bool:
        framework = (framework_text or "").lower()
        if not framework or framework.startswith("no compatible"):
            return False
        spec = module or {}
        model_format = str(spec.get("model_format") or "").lower() if isinstance(spec, dict) else ""
        model_path = str(spec.get("model_path") or "").lower() if isinstance(spec, dict) else ""

        if any(token in model_format for token in ("torch", "pytorch")) or model_path.endswith(
            (".pt", ".pth")
        ):
            return "torch" in framework
        if (
            any(token in model_format for token in ("tensorflow", "keras", "savedmodel", "h5"))
            or model_path.endswith((".keras", ".h5"))
            or os.path.isdir(model_path)
        ):
            return "tensorflow" in framework
        return "tensorflow" in framework or "torch" in framework

    def refresh_framework_options_for_current_module(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is None:
            return
        current = combo.currentText()
        self._populate_framework_combo(combo)
        if current and self.is_framework_compatible(self._current_module, current):
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._refresh_framework_status()
        self._refresh_predict_readiness()

    def _framework_ready(self) -> bool:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is None:
            return False
        return combo.isEnabled() and self.is_framework_compatible(
            self._current_module, combo.currentText()
        )

    def _refresh_framework_status(self) -> None:
        label = getattr(self.ui, "gisaxsPredictFrameworkStatusLabel", None)
        if label is None:
            return
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        text = combo.currentText() if combo is not None else ""
        if self._framework_ready():
            label.setText(f"Framework OK: {text}")
            label.setStyleSheet("color: #166534;")
        elif text.startswith("No compatible"):
            label.setText("Framework missing or incompatible")
            label.setStyleSheet("color: #b91c1c;")
        else:
            label.setText("Framework incompatible")
            label.setStyleSheet("color: #b91c1c;")

    def _initialize_modules_ui(self) -> None:
        self._refresh_modules()
        # Restore last selected module if available
        module_name = self.current_parameters.get("module_name") or ""
        self._set_combobox_text("gisaxsPredictModuleSelectCombox", module_name)
        if module_name:
            self._on_module_selected(module_name)

    def _refresh_modules(self) -> None:
        modules = self._scan_modules()
        new_names = sorted(modules.keys())
        old_names = sorted(self._modules_by_name.keys())
        current_name = self.current_parameters.get("module_name", "")
        self._modules_by_name = modules
        self._modules_by_id = {m.get("id", name): m for name, m in modules.items()}
        if new_names != old_names:
            self._populate_module_combo()
        elif current_name and current_name in modules:
            self._current_module = modules[current_name]

    def _scan_modules(self) -> Dict[str, Dict[str, object]]:
        try:
            modules = self.prediction_view_model.discover_modules()
            return {
                module.name: self.prediction_view_model.module_display_values(module)
                for module in modules
            }
        except Exception as exc:
            self._append_status_message(f"Module scan failed: {exc}", level="ERROR")
            return {}

    def _parse_module_yaml(self, yaml_path: str) -> Optional[Dict[str, object]]:
        try:
            module = self.prediction_view_model.load_module(Path(yaml_path))
            return (
                self.prediction_view_model.module_display_values(module)
                if module is not None
                else None
            )
        except Exception:
            return None

    def _populate_module_combo(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        if combo is None:
            return
        current = combo.currentText()
        names = sorted(self._modules_by_name.keys())
        blocker = QSignalBlocker(combo)
        combo.clear()
        combo.addItems(names)
        # Try restore
        idx = combo.findText(self.current_parameters.get("module_name", ""))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif current:
            idx2 = combo.findText(current)
            if idx2 >= 0:
                combo.setCurrentIndex(idx2)
        del blocker

    def _select_model_folder(self, start_dir: str = "") -> str:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select TensorFlow SavedModel Folder",
            start_dir or "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        return os.path.abspath(normalize_path(folder)) if folder else ""

    def _on_module_selected(self, name: str) -> None:
        if not name:
            return
        spec = self._modules_by_name.get(name)
        if not spec:
            return
        self._current_module = spec
        self.prediction_view_model.select_module(name)
        self.current_parameters["module_name"] = spec.get("name", name)
        self.current_parameters["module_model_path"] = ""
        self._current_model = None
        self._set_model_status_color("gray", "Not loaded")
        self.refresh_framework_options_for_current_module()

        # The selected module owns its model path. Do not inherit a previous
        # module's model path here, or the wrong model can be silently loaded.
        model_path = spec.get("model_path") or ""
        if not model_path or not os.path.exists(model_path):
            self._load_module_mask(self._current_module)
            self._persist_parameters()
            self._append_status_message(
                "Module selected. Use Import Model to choose and load a model.", level="INFO"
            )
            self._refresh_predict_readiness()
            return

        # Persist chosen model path in session parameters (not writing back to YAML)
        abs_model = os.path.abspath(model_path)
        self.current_parameters["module_model_path"] = abs_model
        self._current_module["model_path"] = abs_model

        # Load mask if available
        self._load_module_mask(self._current_module)

        self._persist_parameters()
        self._append_status_message(f"Module selected: {self.current_parameters['module_name']}")

    def _load_module_mask(self, spec: Dict[str, object]) -> None:
        self._current_mask = None
        mask_path = spec.get("mask_path") if isinstance(spec, dict) else None
        if not isinstance(mask_path, str) or not mask_path:
            return
        mask_path = normalize_path(mask_path)
        self._current_mask = self.prediction_view_model.load_mask(Path(mask_path))
        if self._current_mask is not None:
            self._append_status_message(f"Mask loaded: {os.path.basename(mask_path)}")
            return
        message = self.prediction_view_model.state.error_message or "Failed to load mask"
        if message == "Mask file found but unsupported format (only .npy)":
            self._append_status_message(message, level="WARN")
        else:
            self._append_status_message(f"Failed to load mask: {message}", level="ERROR")

    def _on_edit_module_clicked(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        name = combo.currentText().strip() if combo else ""
        spec = self._modules_by_name.get(name) if name else None
        yaml_path = spec.get("yaml_path") if isinstance(spec, dict) else None
        if not isinstance(yaml_path, str) or not os.path.isfile(yaml_path):
            QMessageBox.information(
                self.main_window, "File Missing", "module.yaml not found for this module."
            )
            return
        try:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(yaml_path)):
                raise OSError("The operating system did not accept the file request")
            self._start_module_edit_watch(yaml_path)
        except Exception as exc:
            QMessageBox.warning(
                self.main_window, "Open Failed", f"Cannot open file:\n{yaml_path}\n\n{exc}"
            )

    def _start_module_edit_watch(self, yaml_path: str) -> None:
        try:
            self._module_edit_watch_mtime = os.path.getmtime(yaml_path)
        except OSError:
            self._module_edit_watch_mtime = None
        self._module_edit_watch_path = yaml_path
        self._module_edit_watch_ticks = 0
        if self._module_edit_watch_timer is None:
            self._module_edit_watch_timer = QTimer(self)
            self._module_edit_watch_timer.timeout.connect(self._check_module_edit_watch)
        self._module_edit_watch_timer.start(1000)
        self._append_status_message("Watching module.yaml for saved edits...")

    def _check_module_edit_watch(self) -> None:
        path = self._module_edit_watch_path
        if not path:
            return
        self._module_edit_watch_ticks += 1
        if self._module_edit_watch_ticks > 300:
            if self._module_edit_watch_timer:
                self._module_edit_watch_timer.stop()
            self._module_edit_watch_path = None
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if self._module_edit_watch_mtime is not None and mtime == self._module_edit_watch_mtime:
            return

        if self._module_edit_watch_timer:
            self._module_edit_watch_timer.stop()
        self._module_edit_watch_path = None
        self._module_edit_watch_mtime = mtime
        selected_name = self.current_parameters.get("module_name", "")
        self._refresh_modules()
        if selected_name and selected_name in self._modules_by_name:
            self._current_module = self._modules_by_name[selected_name]
            self._load_module_mask(self._current_module)
        self._append_status_message("module.yaml saved; module settings reloaded.")

    def _on_reload_module_config_clicked(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        selected_name = combo.currentText().strip() if combo else ""
        old_spec = (
            self._modules_by_name.get(selected_name) if selected_name else self._current_module
        )
        old_yaml_path = old_spec.get("yaml_path") if isinstance(old_spec, dict) else None
        old_model_path = ""
        if isinstance(self._current_module, dict):
            old_model_path = str(self._current_module.get("model_path") or "")

        self._refresh_modules()

        refreshed_spec = None
        if isinstance(old_yaml_path, str) and old_yaml_path:
            old_yaml_abs = os.path.normcase(os.path.abspath(old_yaml_path))
            for spec in self._modules_by_name.values():
                yaml_path = spec.get("yaml_path") if isinstance(spec, dict) else None
                if (
                    isinstance(yaml_path, str)
                    and os.path.normcase(os.path.abspath(yaml_path)) == old_yaml_abs
                ):
                    refreshed_spec = spec
                    break

        if refreshed_spec is None and selected_name:
            refreshed_spec = self._modules_by_name.get(selected_name)

        if not isinstance(refreshed_spec, dict):
            QMessageBox.warning(
                self.main_window,
                "Reload Config",
                "Could not reload the selected module. Please check module.yaml.",
            )
            self._append_status_message(
                "Module config reload failed: selected module not found after scan.", level="ERROR"
            )
            return

        new_name = str(refreshed_spec.get("name") or selected_name)
        new_model_path = str(refreshed_spec.get("model_path") or "")
        self._current_module = refreshed_spec
        self.current_parameters["module_name"] = new_name

        if combo is not None and combo.findText(new_name) >= 0:
            blocker = QSignalBlocker(combo)
            combo.setCurrentText(new_name)
            del blocker

        if new_model_path:
            self.current_parameters["module_model_path"] = os.path.abspath(new_model_path)
        else:
            self.current_parameters["module_model_path"] = ""
            self._current_model = None
            self._set_model_status_color("gray", "Not loaded")
        if (
            old_model_path
            and new_model_path
            and os.path.abspath(old_model_path) != os.path.abspath(new_model_path)
        ):
            self._current_model = None
            self._set_model_status_color("gray", "Not loaded")

        self.refresh_framework_options_for_current_module()
        self._load_module_mask(self._current_module)
        self._persist_parameters()
        self._refresh_predict_readiness()

        steps = refreshed_spec.get("preprocess_steps")
        step_text = (
            ", ".join(str(s) for s in steps) if isinstance(steps, list) and steps else "default"
        )
        self._append_status_message(
            f"Module config reloaded: {new_name}; preprocess steps: {step_text}"
        )

    def _on_model_import_clicked(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        name = combo.currentText().strip() if combo else ""
        spec = self._modules_by_name.get(name) if name else None
        if not spec:
            QMessageBox.information(self.main_window, "No Module", "Please select a module first.")
            return
        model_path = (spec.get("model_path") or "") if isinstance(spec, dict) else ""
        if not model_path or not os.path.exists(model_path):
            model_path = self._select_model_folder(
                spec.get("folder", "") if isinstance(spec, dict) else ""
            )
            if not model_path:
                return
            self.current_parameters["module_model_path"] = model_path
            self._current_module = spec
            self._current_module["model_path"] = model_path
            self._write_model_path_to_yaml(self._current_module, model_path)
            self.refresh_framework_options_for_current_module()
        else:
            model_path = os.path.abspath(model_path)
            self.current_parameters["module_model_path"] = model_path

        self._append_status_message("Loading model (this may take a while)...")
        self.progress_updated.emit(5)
        self._model_loading = True
        self._model_cancel_requested = False
        self._set_model_status_color("red", "Loading...")
        btn_import = getattr(self.ui, "gisaxsPredictModelImportButton", None)
        if btn_import:
            btn_import.setEnabled(False)

        def _load():
            self._append_status_message(f"Loading model from: {model_path}")
            try:
                model = self.prediction_view_model.inspect_model(
                    Path(model_path),
                    allow_unsafe_lambda=True,
                )
                if model is None:
                    raise RuntimeError(
                        self.prediction_view_model.state.error_message or "Model validation failed"
                    )
            except Exception as exc:
                self._append_status_message(
                    f"Failed to load model from: {model_path} | {exc}",
                    level="ERROR",
                )
                return None, str(exc)
            self._append_status_message(
                f"Model successfully validated in isolated worker: {model.artifact_path}"
            )
            return model, None

        def _run():
            model, err = _load()
            self.model_load_finished.emit(model, err or "", model_path)

        import threading as _threading

        self._model_loader_thread = _threading.Thread(target=_run, daemon=True)
        self._model_loader_thread.start()

    def _on_model_load_finished(self, model: object, err: str, model_path: str) -> None:
        """Finalize model loading on the Qt UI thread."""
        expected_model_path = str(self.current_parameters.get("module_model_path") or "")
        if expected_model_path and os.path.abspath(model_path) != os.path.abspath(
            expected_model_path
        ):
            self._append_status_message(
                f"Ignored stale model load result from: {model_path}",
                level="WARN",
            )
            self._model_loading = False
            btn = getattr(self.ui, "gisaxsPredictModelImportButton", None)
            if btn:
                btn.setEnabled(True)
            self._refresh_predict_readiness()
            return
        if not expected_model_path:
            self._append_status_message(
                f"Ignored model load result because no module model is selected: {model_path}",
                level="WARN",
            )
            self._model_loading = False
            btn = getattr(self.ui, "gisaxsPredictModelImportButton", None)
            if btn:
                btn.setEnabled(True)
            self._refresh_predict_readiness()
            return
        if err:
            self._append_status_message(f"Model load failed: {err}", level="ERROR")
            self.progress_updated.emit(0)
            self._current_model = None
            self._model_loading = False
            self._set_model_status_color("gray", "Not loaded")
        else:
            self._current_model = model
            self.current_parameters["module_model_path"] = model_path
            if self._current_module is not None:
                self._current_module["model_path"] = model_path
            if self._model_cancel_requested:
                self._current_model = None
                self._append_status_message("Model load canceled.")
                self.progress_updated.emit(0)
                self._set_model_status_color("gray", "Canceled")
            else:
                self._append_status_message("Model loaded successfully.")
                self.progress_updated.emit(100)
                self._set_model_status_color("green", "Loaded")
            self._model_loading = False

        btn = getattr(self.ui, "gisaxsPredictModelImportButton", None)
        if btn:
            btn.setEnabled(True)
        self._persist_parameters()
        self._refresh_predict_readiness()

    def _write_model_path_to_yaml(self, spec: Dict[str, object], model_path: str) -> None:
        module = spec.get("_prediction_module") if isinstance(spec, dict) else None
        if module is None:
            return
        if self.prediction_view_model.update_model_path(module, Path(model_path)):
            yaml_path = spec.get("yaml_path", "module.yaml")
            self._append_status_message(f"Updated model_path in {os.path.basename(str(yaml_path))}")
            return
        self._append_status_message(
            self.prediction_view_model.state.error_message or "Failed to update module.yaml",
            level="ERROR",
        )

    def _run_gisaxs_predict(self) -> None:
        self._execute_prediction()
