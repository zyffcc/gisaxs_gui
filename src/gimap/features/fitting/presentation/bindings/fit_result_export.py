"""Fit Result Export for fitting presentation."""

from __future__ import annotations


from pathlib import Path


from PyQt5.QtWidgets import (
    QFileDialog,
    QInputDialog,
)

from src.gimap.features.fitting.application import (
    ExportFitResultRequest,
)

from ..binding_primitives import (
    _scientific_commands,
)


class FitResultExportMixin:
    """Own fit result export behavior."""

    def _get_fitting_parameter_comment_lines(self):
        """No description."""
        lines = ["# Fitting Parameters Begin"]
        try:
            import re

            shapes = []
            param_dict = None
            param_source = "current_ui_snapshot"
            widget_ids = list(getattr(self, "_last_active_particle_ids", []) or [])

            if isinstance(getattr(self, "fitting", None), dict):
                meta = self.fitting.get("meta", {})
                fit_shapes = meta.get("shapes")
                fit_params = meta.get("params")
                if fit_shapes and fit_params:
                    shapes = [str(shape).lower() for shape in fit_shapes]
                    param_dict = {str(k): float(v) for k, v in dict(fit_params).items()}
                    param_source = "last_fitting_result"

            if not shapes:
                shapes, widget_ids = self._collect_active_particles()

            if not param_dict and shapes:
                shape_list, params_list = self._get_last_fitting_spec_and_params(
                    fallback_shapes=shapes
                )
                if shape_list and params_list:
                    shapes = list(shape_list)
                    param_dict = {
                        str(name): float(value)
                        for name, value in zip(
                            _scientific_commands(self).model.parameter_names(shapes),
                            params_list,
                        )
                    }

            if not shapes or not param_dict:
                lines.append("# Parameter Source: unavailable")
                lines.append("# No fitting parameter snapshot available")
                lines.append("# Fitting Parameters End")
                return lines

            template = _scientific_commands(self).model.parameter_names(shapes)
            lines.append(f"# Parameter Source: {param_source}")
            lines.append(f"# Active Shapes: {', '.join(shapes)}")

            grouped_particle_params = {}
            global_parameter_names = []
            for template_name in template:
                match = re.match(r"^(.*?)(\d+)$", str(template_name))
                if match:
                    param_base = match.group(1)
                    particle_index = int(match.group(2))
                    grouped_particle_params.setdefault(particle_index, []).append(
                        (template_name, param_base)
                    )
                else:
                    global_parameter_names.append(template_name)

            for particle_index in sorted(grouped_particle_params.keys()):
                shape = (
                    shapes[particle_index - 1] if particle_index - 1 < len(shapes) else "unknown"
                )
                widget_id = (
                    widget_ids[particle_index - 1]
                    if particle_index - 1 < len(widget_ids)
                    else particle_index
                )
                lines.append(f"# Particle {particle_index}: widget_id={widget_id}, shape={shape}")
                for template_name, _param_base in grouped_particle_params[particle_index]:
                    if template_name in param_dict:
                        lines.append(
                            f"#   {template_name} = {float(param_dict[template_name]):.10g}"
                        )

            if global_parameter_names:
                lines.append("# Global Parameters:")
                for template_name in global_parameter_names:
                    if template_name in param_dict:
                        lines.append(
                            f"#   {template_name} = {float(param_dict[template_name]):.10g}"
                        )

        except Exception as e:
            lines.append(f"# Fitting parameter export error: {e}")

        lines.append("# Fitting Parameters End")
        return lines

    def _build_export_header_lines(self, choice: str, data_name: str):
        """No description."""
        lines = []
        try:
            from datetime import datetime

            q_source_kind = None
            if choice == "Cut Data":
                q_source_kind = "cut"
            elif choice == "1D File Data":
                q_source_kind = "1d"
            elif choice == "Fitting Data" and isinstance(getattr(self, "fitting", None), dict):
                q_source_kind = self.fitting.get("meta", {}).get(
                    "data_source", getattr(self, "data_source", None)
                )

            lines.append("# GIMaP Export")
            lines.append(f"# Export Time: {datetime.now().isoformat(timespec='seconds')}")
            lines.append(f"# Data Type: {choice}")
            lines.append(f"# Export Name: {data_name}")
            lines.append(f"# Display Mode: {getattr(self, 'display_mode', 'normal')}")
            lines.append(f"# Log X: {self._is_fit_log_x_enabled()}")
            lines.append(f"# Log Y: {self._is_fit_log_y_enabled()}")
            lines.append(f"# Normalize: {self._is_fit_norm_enabled()}")
            lines.append(f"# Axis Filter: {self._get_independent_axis_filter_mode()}")
            lines.append(f"# Raw q Source Unit: {self._get_q_source_unit(q_source_kind)}")
            lines.append("# Internal Model q Unit: nm^-1")
            lines.append(f"# q Unit: {self._get_q_unit_label(mathtext=False)}")
            lines.append(
                f"# X Column: {self._build_q_axis_label(filter_mode='all', mathtext=False)}"
            )
            lines.append("# Y Column: Intensity (a.u.)")

            if self._roi_min is not None and self._roi_max is not None:
                lines.append(
                    f"# ROI Range: {float(self._roi_min):.10g} -> {float(self._roi_max):.10g}"
                )

            if choice == "1D File Data" and getattr(self, "current_1d_data", None) is not None:
                file_path = self.current_1d_data.get("file_path")
                if file_path:
                    lines.append(f"# 1D File: {file_path}")
            elif choice == "Cut Data" and getattr(self, "cut", None) is not None:
                cut_meta = self.cut.get("meta", {}) if isinstance(self.cut, dict) else {}
                title = cut_meta.get("title")
                if title:
                    lines.append(f"# Cut Title: {title}")

        except Exception:
            pass

        lines.extend(self._get_fitting_parameter_comment_lines())
        return lines

    def _export_fitting_data(self):
        """Fitting"""
        try:
            import numpy as np

            if not hasattr(self.ui, "fitGraphicsView") or self.ui.fitGraphicsView is None:
                self._add_fitting_error("fitGraphicsView is not available")
                return

            options = []
            if getattr(self, "cut", None) is not None:
                options.append("Cut Data")
            if getattr(self, "fitting", None) is not None:
                options.append("Fitting Data")
            if getattr(self, "current_1d_data", None) is not None:
                options.append("1D File Data")
            if not options:
                self._add_fitting_error(
                    "No available data to export (need Cut, Fitting, or 1D data)"
                )
                return

            default_index = 0
            choice, ok = QInputDialog.getItem(
                None, "Select Data to Export", "Data source:", options, default_index, False
            )
            if not ok:
                return

            x_data = None
            y_data = None
            data_name = ""
            q_source_kind = None
            if choice == "Cut Data" and self.cut is not None:
                x_data = np.array(self.cut.get("q", []))
                y_data = np.array(self.cut.get("I", []))
                data_name = "Cut_Data"
                q_source_kind = "cut"
            elif choice == "Fitting Data" and self.fitting is not None:
                x_data = np.array(self.fitting.get("q", []))
                y_data = np.array(self.fitting.get("I", []))
                data_name = "Fitting_Data"
                q_source_kind = self.fitting.get("meta", {}).get(
                    "data_source", getattr(self, "data_source", None)
                )
            elif choice == "1D File Data" and self.current_1d_data is not None:
                x_data = np.array(self.current_1d_data.get("q", []))
                y_data = np.array(self.current_1d_data.get("I", []))
                data_name = "1D_File_Data"
                q_source_kind = "1d"
            else:
                self._add_fitting_error("Selected data is not available to export")
                return

            filename, _ = QFileDialog.getSaveFileName(
                None,
                f"Export {data_name}",
                f"{data_name}.txt",
                "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
            )

            if not filename:
                return

            min_length = min(len(x_data), len(y_data))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]

            x_data = self._convert_q_values_for_display(x_data, source=q_source_kind)
            x_column_name = self._build_q_axis_label(filter_mode="all", mathtext=False)
            y_column_name = "Intensity (a.u.)"
            header_lines = self._build_export_header_lines(choice, data_name)
            outcome = self.fitting_view_model.export_fit_result(
                ExportFitResultRequest(
                    path=Path(filename),
                    q=x_data,
                    intensity=y_data,
                    header_lines=tuple(header_lines),
                    x_column_name=x_column_name,
                    y_column_name=y_column_name,
                )
            )
            if outcome.error is not None:
                raise RuntimeError(f"[{outcome.error.code}] {outcome.error.message}")

            self._add_fitting_success(f"{data_name} exported successfully to: {filename}")

        except Exception as e:
            self._add_fitting_error(f"Export failed: {str(e)}")
