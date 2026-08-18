"""Ai Model Controls for fitting presentation."""

from __future__ import annotations

import os


from pathlib import Path


from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QComboBox,
    QLabel,
    QPushButton,
)


from ..binding_primitives import (
    _ai_catalog,
    _scientific_commands,
)


class AiModelControlsMixin:
    """Own ai model controls behavior."""

    def _refresh_ai_fitting_models(self) -> None:
        models = self._scan_ai_fitting_models()
        self._ai_fitting_models = models
        for combo in (
            getattr(self, "_ai_model_combo", None),
            getattr(self.ui, "aiFittingModelComboBox", None),
        ):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            for model in models:
                combo.addItem(model.display_name, str(model.artifact_path))
            combo.blockSignals(False)
        self._restore_ai_model_selection()
        if models:
            self._set_ai_workspace_status(f"Found {len(models)} AI fitting model(s).", 0)
            selected_path = self._selected_ai_model_path()
            info = next(
                (
                    item
                    for item in models
                    if selected_path is not None and item.artifact_path == selected_path
                ),
                models[0],
            )
            label = getattr(self, "_ai_model_status_label", None)
            if label is not None:
                label.setText(
                    f"Checkpoint: {info.artifact_type} | version {info.version} | K={list(info.contract.supported_k)} "
                    f"| max_points={info.contract.max_points} | training={info.training_status.get('state', 'unknown')}"
                )
        else:
            self._set_ai_workspace_status(
                "No AI fitting model found in modules/Fitting_1D_Model/ or modules/Fitting_1D_model/",
                0,
            )

    def _browse_ai_fitting_model(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main_window or self.ui,
            "Select AI Fitting Model Folder",
            os.path.join(os.getcwd(), "modules", "Fitting_1D_Model"),
        )
        if not folder:
            return
        found = _ai_catalog(self).discover_model(Path(folder))
        if not found:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting Model",
                "Selected folder must contain a .keras artifact or a TensorFlow SavedModel root/subfolder.",
            )
            return
        settings = self._ai_fitting_settings()
        extra = settings.get("extra_model_paths")
        extra = extra if isinstance(extra, list) else []
        if folder not in extra:
            extra.append(folder)
        self._save_ai_fitting_settings(
            extra_model_paths=extra, last_selected_model=str(found[0].artifact_path)
        )
        self._refresh_ai_fitting_models()
        self._set_ai_workspace_status(f"Selected model: {found[0].artifact_path}", 0)

    def _restore_ai_model_selection(self) -> None:
        selected = self._ai_fitting_settings().get("last_selected_model")
        if not selected:
            preferred = next(
                (
                    model
                    for model in getattr(self, "_ai_fitting_models", [])
                    if model.model_id == "gisaxs-k1-k4-phys-constraints"
                ),
                None,
            )
            selected = str(preferred.artifact_path) if preferred is not None else None
            if selected:
                self._save_ai_fitting_settings(last_selected_model=selected)
        if not selected:
            return
        for combo in (
            getattr(self, "_ai_model_combo", None),
            getattr(self.ui, "aiFittingModelComboBox", None),
        ):
            if combo is None:
                continue
            for i in range(combo.count()):
                if combo.itemData(i) == selected:
                    combo.setCurrentIndex(i)
                    break

    def _restore_ai_workspace_settings(self) -> None:
        mode = str(self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(
            " Prediction", ""
        )
        combo = getattr(self, "_ai_constraint_combo", None)
        if combo is not None:
            idx = combo.findText(str(mode))
            combo.blockSignals(True)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        k_value = str(self._ai_fitting_settings().get("fixed_k", 1))
        k_combo = getattr(self, "_ai_constraint_k_combo", None)
        if k_combo is not None:
            idx = k_combo.findText(k_value)
            k_combo.blockSignals(True)
            k_combo.setCurrentIndex(idx if idx >= 0 else 0)
            k_combo.blockSignals(False)
        self._sync_ai_constraint_controls(str(mode))

    def _restore_main_ai_settings(self) -> None:
        mode = str(self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(
            " Prediction", ""
        )
        combo = getattr(self.ui, "aiFittingConstraintComboBox", None)
        if combo is not None:
            label = "Free Prediction" if mode == "Free" else str(mode)
            idx = combo.findText(label)
            combo.blockSignals(True)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        k_value = str(self._ai_fitting_settings().get("fixed_k", 1))
        k_combo = getattr(self.ui, "aiFittingFixedKComboBox", None)
        if k_combo is not None:
            idx = k_combo.findText(k_value)
            k_combo.blockSignals(True)
            k_combo.setCurrentIndex(idx if idx >= 0 else 0)
            k_combo.blockSignals(False)
        self._sync_ai_constraint_controls(str(mode))

    def _on_ai_model_selected(self, index: int) -> None:
        combo = self.sender()
        if combo is None or index < 0:
            return
        path = combo.itemData(index)
        if path:
            self._save_ai_fitting_settings(last_selected_model=str(path))
            self._sync_ai_model_combos(str(path))
            info = next(
                (
                    item
                    for item in getattr(self, "_ai_fitting_models", [])
                    if str(item.artifact_path) == str(path)
                ),
                None,
            )
            label = getattr(self, "_ai_model_status_label", None)
            if label is not None and info is not None:
                state = info.training_status.get("state", "unknown")
                label.setText(
                    f"Checkpoint: {info.artifact_type} | version {info.version} | K={list(info.contract.supported_k)} "
                    f"| max_points={info.contract.max_points} | training={state}"
                )

    def _sync_ai_model_combos(self, selected_path: str) -> None:
        for combo in (
            getattr(self.ui, "aiFittingModelComboBox", None),
            getattr(self, "_ai_model_combo", None),
        ):
            if combo is None:
                continue
            for i in range(combo.count()):
                if combo.itemData(i) == selected_path:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    break

    def _on_ai_constraint_mode_changed(self, mode: str) -> None:
        mode = str(mode).replace(" Prediction", "")
        self._save_ai_fitting_settings(last_constraint_mode=mode)
        self._sync_ai_constraint_combos(mode)
        self._sync_ai_constraint_controls(mode)
        if mode == "Fixed Combination" and not self._ai_fixed_combination():
            QTimer.singleShot(0, self._show_ai_fixed_combination_dialog)

    def _on_ai_fixed_k_changed(self, text: str) -> None:
        try:
            value = int(text)
        except Exception:
            value = 1
        self._save_ai_fitting_settings(fixed_k=value)
        for combo in (
            getattr(self.ui, "aiFittingFixedKComboBox", None),
            getattr(self, "_ai_constraint_k_combo", None),
        ):
            if combo is None:
                continue
            idx = combo.findText(str(value))
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

    def _sync_ai_constraint_combos(self, mode: str) -> None:
        for combo, free_label in (
            (getattr(self.ui, "aiFittingConstraintComboBox", None), "Free Prediction"),
            (getattr(self, "_ai_constraint_combo", None), "Free"),
        ):
            if combo is None:
                continue
            label = free_label if mode == "Free" else str(mode)
            idx = combo.findText(label)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

    def _sync_ai_constraint_controls(self, mode: str | None = None) -> None:
        mode = str(mode or self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(
            " Prediction", ""
        )
        show_k = mode == "Fixed K"
        show_combo = mode == "Fixed Combination"
        for widget in (
            getattr(self.ui, "aiFittingFixedKComboBox", None),
            getattr(self, "_ai_constraint_k_combo", None),
        ):
            if widget is not None:
                widget.setVisible(show_k)
        label = self._ai_fixed_combination_label()
        for widget in (
            getattr(self.ui, "aiFittingCombinationButton", None),
            getattr(self, "_ai_constraint_combination_button", None),
        ):
            if widget is not None:
                widget.setVisible(show_combo)
                widget.setText(label)

    def _ai_fixed_combination(self) -> list[str]:
        constraints = self._ai_run_settings().get("parameter_constraints", {})
        components = constraints.get("components") if isinstance(constraints, dict) else None
        return [str(c) for c in components] if isinstance(components, list) else []

    def _ai_constraint_geometries(self) -> list[str]:
        mode = str(self._ai_fitting_settings().get("last_constraint_mode", "Free")).replace(
            " Prediction", ""
        )
        if mode == "Fixed Combination":
            geometries = self._ai_fixed_combination()
        elif mode == "Current Manual Model":
            geometries = []
            try:
                geometries = [
                    _scientific_commands(self).ai.normalize_geometry(
                        self.get_particle_shape(widget_id)
                    )
                    for widget_id in self._iter_particle_widget_ids()
                    if self.get_particle_shape(widget_id) not in (None, "None")
                ]
            except Exception:
                geometries = []
        else:
            geometries = ["sphere", "cylinder", "vertical_cylinder"]
        return sorted(
            {_scientific_commands(self).ai.normalize_geometry(item) for item in geometries if item}
        )

    def _ai_fixed_combination_label(self) -> str:
        components = self._ai_fixed_combination()
        if not components:
            return "Choose Combination..."
        display = [str(c).replace("_", " ").title() for c in components]
        return " + ".join(display)

    def _save_ai_fixed_combination(self, components: list[str]) -> None:
        settings_constraints = self._ai_run_settings().get("parameter_constraints", {})
        constraints_payload = (
            dict(settings_constraints) if isinstance(settings_constraints, dict) else {}
        )
        if components:
            constraints_payload["components"] = components
        else:
            constraints_payload.pop("components", None)
        self._save_ai_fitting_settings(parameter_constraints=constraints_payload)
        self._sync_ai_constraint_controls("Fixed Combination")

    def _show_ai_fixed_combination_dialog(self) -> None:
        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("Fixed Combination")
        dialog.resize(420, 260)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the component sequence for Fixed Combination:", dialog))
        current = self._ai_fixed_combination()
        choices = [
            ("None", ""),
            ("Sphere", "sphere"),
            ("Cylinder", "cylinder"),
            ("Vertical Cylinder", "vertical_cylinder"),
        ]
        combos = []
        for idx in range(4):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Slot {idx + 1}:", dialog))
            combo = QComboBox(dialog)
            for label, value in choices:
                combo.addItem(label, value)
            if idx < len(current):
                found = combo.findData(current[idx])
                combo.setCurrentIndex(found if found >= 0 else 0)
            row.addWidget(combo, 1)
            layout.addLayout(row)
            combos.append(combo)

        buttons = QHBoxLayout()
        save = QPushButton("Save", dialog)
        clear = QPushButton("Clear", dialog)
        cancel = QPushButton("Cancel", dialog)
        buttons.addWidget(save)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        # 函数说明：实现 selected 组件 相关逻辑。
        def selected_components() -> list[str]:
            return [str(combo.currentData()) for combo in combos if combo.currentData()]

        # 函数说明：保存选区。
        def save_selection() -> None:
            components = selected_components()
            if not components:
                QMessageBox.information(
                    dialog, "Fixed Combination", "Select at least one component."
                )
                return
            self._save_ai_fixed_combination(components)
            self._set_ai_workspace_status(
                f"Fixed combination: {self._ai_fixed_combination_label()}", None
            )
            dialog.accept()

        # 函数说明：清除选区。
        def clear_selection() -> None:
            self._save_ai_fixed_combination([])
            dialog.accept()

        save.clicked.connect(save_selection)
        clear.clicked.connect(clear_selection)
        cancel.clicked.connect(dialog.reject)
        dialog.exec_()
