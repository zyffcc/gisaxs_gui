"""Ai Constraints for fitting presentation."""

from __future__ import annotations

import json


from PyQt5.QtWidgets import (
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QTextBrowser,
    QDialog,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from src.gimap.features.fitting.application import (
    ConstraintSet,
)

from ..binding_primitives import (
    COMPONENT_PARAMETER_SCHEMAS,
)


class AiConstraintsMixin:
    """Own ai constraints behavior."""

    def _load_ai_candidate_params(self, row: dict, *, refresh_plot: bool = True) -> bool:
        reviewed = self.fitting_view_model.review_candidates(
            [row],
            self._ai_run_settings().get("constraint_set"),
        )
        candidate = reviewed[0] if reviewed else dict(row)
        violations = candidate.get("constraint_violations") or []
        if violations:
            detail = "\n".join(f"- {message}" for message in violations)
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting Constraint Violation",
                "The selected candidate violates enabled physical constraints:\n" + detail,
            )
            return False
        try:
            mapping = self.fitting_view_model.map_candidate_parameters(row)
            while len(self._iter_particle_widget_ids()) < len(mapping.components):
                self._on_add_particle_clicked()
            for widget_id in self._iter_particle_widget_ids():
                self.set_particle_shape(widget_id, "None")

            widget_ids = self._iter_particle_widget_ids()
            for idx, component in enumerate(mapping.components):
                widget_id = widget_ids[idx]
                shape = component.shape
                if shape not in COMPONENT_PARAMETER_SCHEMAS:
                    shape = "Sphere"
                particle_id = f"particle_{widget_id}"
                self.model_params_manager.set_particle_shape("fitting", particle_id, shape)
                self.model_params_manager.set_particle_enabled("fitting", particle_id, True)
                for parameter_name, value in component.parameters.items():
                    self.model_params_manager.set_particle_parameter(
                        "fitting",
                        particle_id,
                        shape,
                        parameter_name,
                        float(value),
                    )

            for key, value in mapping.global_parameters.items():
                self.model_params_manager.set_global_parameter(
                    "fitting",
                    key,
                    float(value),
                )
            self.model_params_manager.save_parameters()
            if not self.reload_particle_parameters():
                raise RuntimeError(
                    "candidate parameters were saved but could not be reloaded into the GUI"
                )
            if refresh_plot:
                self._perform_manual_fitting()
            self._set_ai_workspace_status(
                f"Loaded AI candidate #{row.get('rank', '')}: {row.get('combination', '')}", None
            )
            return True
        except Exception as exc:
            QMessageBox.warning(
                self.main_window or self.ui,
                "AI Fitting",
                f"Failed to load candidate parameters:\n{exc}",
            )
            return False

    def build_ai_constraints_json_from_ui(self) -> dict:
        mode = self._ai_fitting_settings().get("last_constraint_mode", "Free")
        workspace_combo = getattr(self, "_ai_constraint_combo", None)
        if workspace_combo is not None:
            mode = workspace_combo.currentText()
        main_combo = getattr(self.ui, "aiFittingConstraintComboBox", None)
        if main_combo is not None and workspace_combo is None:
            mode = main_combo.currentText().replace(" Prediction", "")
        settings_constraints = self._ai_run_settings().get("parameter_constraints", {})
        payload = {"mode": mode, "constraints": {}}
        if isinstance(settings_constraints, dict):
            for key in ("type_parameter_ranges", "global_ranges", "parameter_ranges"):
                value = settings_constraints.get(key)
                if value:
                    payload[key] = value
        if mode == "Fixed K":
            try:
                payload["exact_nonempty"] = int(self._ai_fitting_settings().get("fixed_k", 1))
            except Exception:
                payload["exact_nonempty"] = 1
        elif mode == "Current Manual Model":
            shapes = []
            try:
                for widget_id in self._iter_particle_widget_ids():
                    shape = self.get_particle_shape(widget_id)
                    if shape and shape != "None":
                        shapes.append(shape.lower().replace(" ", "_"))
            except Exception:
                shapes = []
            payload["components"] = shapes
            payload["exact_nonempty"] = len(shapes) if shapes else None
        elif mode == "Fixed Combination":
            components = (
                settings_constraints.get("components")
                if isinstance(settings_constraints, dict)
                else None
            )
            payload["components"] = components if isinstance(components, list) else []
        constraint_set = ConstraintSet.from_dict(self._ai_run_settings().get("constraint_set"))
        spacing_rule = str(self._ai_run_settings().get("d_spacing_rule", "max_diameter"))
        payload["d_constraint"] = constraint_set.d_constraint_payload(spacing_rule)
        payload["physical_constraints"] = constraint_set.to_dict()
        return payload

    def _show_advanced_constraints_dialog(self) -> None:
        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("Advanced Constraints")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Constraints are filtered by geometry and are applied to AI proposals, bounds, refinement and final validation.",
            dialog,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        geometries = self._ai_constraint_geometries()
        geometry_label = QLabel(
            "Applicable geometry: "
            + (", ".join(name.replace("_", " ") for name in geometries) or "none"),
            dialog,
        )
        geometry_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(geometry_label)

        physical_set = ConstraintSet.from_dict(self._ai_run_settings().get("constraint_set"))
        applicable = physical_set.applicable(geometries)
        physical_table = QTableWidget(len(applicable), 4, dialog)
        physical_table.setHorizontalHeaderLabels(
            ["Enable", "Constraint", "Formula / meaning", "Margin"]
        )
        physical_table.verticalHeader().setVisible(False)
        physical_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        physical_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        physical_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        physical_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        physical_widgets = []
        for row_idx, (definition, option) in enumerate(applicable):
            enabled = QCheckBox(physical_table)
            enabled.setChecked(option.enabled)
            margin_box = QDoubleSpinBox(physical_table)
            margin_box.setDecimals(6)
            margin_box.setRange(definition.minimum_margin, definition.maximum_margin)
            margin_box.setSingleStep(0.001)
            margin_box.setValue(option.margin)
            margin_box.setEnabled(definition.maximum_margin > definition.minimum_margin)
            formula = QTableWidgetItem(f"{definition.formula}\n{definition.meaning}")
            formula.setToolTip(definition.meaning)
            physical_table.setCellWidget(row_idx, 0, enabled)
            physical_table.setItem(row_idx, 1, QTableWidgetItem(definition.label))
            physical_table.setItem(row_idx, 2, formula)
            physical_table.setCellWidget(row_idx, 3, margin_box)
            physical_widgets.append((definition, enabled, margin_box))
        physical_table.setMaximumHeight(190)
        layout.addWidget(physical_table)

        d_rule_row = QHBoxLayout()
        d_rule_row.addWidget(QLabel("Multi-component D rule:", dialog))
        d_rule_combo = QComboBox(dialog)
        d_rule_combo.addItem("Maximum exclusion size", "max_diameter")
        d_rule_combo.addItem("Mean exclusion size", "mean_diameter")
        current_d_rule = str(self._ai_run_settings().get("d_spacing_rule", "max_diameter"))
        d_rule_combo.setCurrentIndex(max(0, d_rule_combo.findData(current_d_rule)))
        d_rule_row.addWidget(d_rule_combo)
        d_rule_row.addStretch(1)
        layout.addLayout(d_rule_row)

        rows = [
            ("type", "R", 1.0, 100.0),
            ("type", "sigma_R", 0.02, 90.0),
            ("type", "h", 2.0, 500.0),
            ("type", "sigma_h", 0.04, 400.0),
            ("type", "D", 3.0, 500.0),
            ("type", "sigma_D", 0.06, 400.0),
            ("global", "BG", 1e-18, 1e8),
            ("global", "sigma_Res", 0.002, 0.3),
            ("global", "nu_Res", 1.0, 10.0),
            ("global", "int_Res", 1e-18, 1e8),
            ("global", "k", 1e-2, 1e6),
        ]
        stored = self._ai_run_settings().get("parameter_constraints", {})
        type_ranges = stored.get("type_parameter_ranges", {}) if isinstance(stored, dict) else {}
        global_ranges = stored.get("global_ranges", {}) if isinstance(stored, dict) else {}

        table = QTableWidget(len(rows), 5, dialog)
        table.setHorizontalHeaderLabels(["Apply", "Scope", "Parameter", "Min", "Max"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

        row_widgets = []
        for row_idx, (scope, name, default_lo, default_hi) in enumerate(rows):
            if scope == "type":
                existing = None
                for type_name in ("sphere", "cylinder", "vertical_cylinder"):
                    ranges = type_ranges.get(type_name, {}) if isinstance(type_ranges, dict) else {}
                    if name in ranges:
                        existing = ranges[name]
                        break
            else:
                existing = global_ranges.get(name) if isinstance(global_ranges, dict) else None
            enabled = isinstance(existing, (list, tuple)) and len(existing) == 2
            lo = float(existing[0]) if enabled else float(default_lo)
            hi = float(existing[1]) if enabled else float(default_hi)

            check = QCheckBox(table)
            check.setChecked(enabled)
            min_box = QDoubleSpinBox(table)
            max_box = QDoubleSpinBox(table)
            for spin in (min_box, max_box):
                spin.setDecimals(8)
                spin.setRange(0.0, 1e12)
                spin.setSingleStep(max(abs(default_hi - default_lo) / 100.0, 1e-6))
                spin.setMinimumWidth(120)
            min_box.setValue(lo)
            max_box.setValue(hi)
            table.setCellWidget(row_idx, 0, check)
            table.setItem(
                row_idx, 1, QTableWidgetItem("Component" if scope == "type" else "Global")
            )
            table.setItem(row_idx, 2, QTableWidgetItem(name))
            table.setCellWidget(row_idx, 3, min_box)
            table.setCellWidget(row_idx, 4, max_box)
            row_widgets.append((scope, name, check, min_box, max_box))

        layout.addWidget(table, 1)

        preview = QTextBrowser(dialog)
        preview.setMaximumHeight(120)
        preview.setPlainText(
            json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False)
        )
        layout.addWidget(preview)

        # 函数说明：实现 collect constraints 相关逻辑。
        def collect_constraints() -> dict:
            type_constraints = {}
            global_constraints = {}
            for scope, name, check, min_box, max_box in row_widgets:
                if not check.isChecked():
                    continue
                lo = float(min_box.value())
                hi = float(max_box.value())
                if hi < lo:
                    lo, hi = hi, lo
                if scope == "type":
                    for type_name in ("sphere", "cylinder", "vertical_cylinder"):
                        type_constraints.setdefault(type_name, {})[name] = [lo, hi]
                else:
                    global_constraints[name] = [lo, hi]
            payload = {}
            if type_constraints:
                payload["type_parameter_ranges"] = type_constraints
            if global_constraints:
                payload["global_ranges"] = global_constraints
            return payload

        def collect_physical_constraints() -> dict:
            payload = physical_set.to_dict()
            for definition, enabled, margin_box in physical_widgets:
                payload[definition.id] = {
                    "enabled": bool(enabled.isChecked()),
                    "margin": float(margin_box.value()),
                }
            return payload

        # 函数说明：刷新预览。
        def refresh_preview() -> None:
            settings = self._ai_fitting_settings()
            old_constraints = settings.get("parameter_constraints")
            self._save_ai_fitting_settings(parameter_constraints=collect_constraints())
            preview.setPlainText(
                json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False)
            )
            self._save_ai_fitting_settings(
                parameter_constraints=old_constraints if isinstance(old_constraints, dict) else {}
            )

        for _scope, _name, check, min_box, max_box in row_widgets:
            check.toggled.connect(lambda _=False: refresh_preview())
            min_box.valueChanged.connect(lambda _=0.0: refresh_preview())
            max_box.valueChanged.connect(lambda _=0.0: refresh_preview())

        save = QPushButton("Save Constraints", dialog)
        clear = QPushButton("Clear All", dialog)
        close = QPushButton("Close", dialog)

        # 函数说明：保存constraints。
        def save_constraints() -> None:
            constraints_payload = collect_constraints()
            self._save_ai_fitting_settings(
                parameter_constraints=constraints_payload,
                constraint_set=collect_physical_constraints(),
                d_spacing_rule=str(d_rule_combo.currentData()),
            )
            preview.setPlainText(
                json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False)
            )
            self._set_ai_workspace_status("Advanced parameter constraints saved.", None)
            dialog.accept()

        # 函数说明：清除constraints。
        def clear_constraints() -> None:
            for _scope, _name, check, _min_box, _max_box in row_widgets:
                check.setChecked(False)
            defaults = ConstraintSet.defaults().to_dict()
            for definition, enabled, margin_box in physical_widgets:
                default_option = defaults[definition.id]
                enabled.setChecked(bool(default_option["enabled"]))
                margin_box.setValue(float(default_option["margin"]))
            d_rule_combo.setCurrentIndex(max(0, d_rule_combo.findData("max_diameter")))
            self._save_ai_fitting_settings(
                parameter_constraints={},
                constraint_set=defaults,
                d_spacing_rule="max_diameter",
            )
            preview.setPlainText(
                json.dumps(self.build_ai_constraints_json_from_ui(), indent=2, ensure_ascii=False)
            )

        save.clicked.connect(save_constraints)
        clear.clicked.connect(clear_constraints)
        close.clicked.connect(dialog.reject)
        row = QHBoxLayout()
        row.addWidget(save)
        row.addWidget(clear)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        dialog.exec_()
