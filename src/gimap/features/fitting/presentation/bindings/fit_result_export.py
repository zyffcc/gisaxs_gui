"""Fit Result Export for fitting presentation."""

from __future__ import annotations


import re
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QFileDialog,
    QDialog,
)

from src.gimap.features.fitting.application import (
    ExportCurveFigureRequest,
    ExportFitResultRequest,
    FigureSeries,
)
from src.gimap.shared.figures import FIGURE_FILE_FILTER

from ..binding_primitives import (
    _scientific_commands,
)
from ..export_dialog import FittingDataExportDialog


class FitResultExportMixin:
    """Own fit result export behavior."""

    def _export_plot(self):
        """Save the curve plot as shown (data, model, components) as a publication figure."""
        series, axes_labels = self._plotted_figure_series()
        if not series:
            self._add_fitting_error("Nothing to export: load a curve first")
            return
        source = (getattr(self, "current_1d_data", None) or {}).get("file_path")
        default = (
            str(Path(source).with_name(f"{Path(source).stem}_plot.png"))
            if source
            else "fitting_plot.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Plot", default, FIGURE_FILE_FILTER
        )
        if not path:
            return
        outcome = self.fitting_view_model.storage.export_curve_figure(
            ExportCurveFigureRequest(path=Path(path), series=series, **axes_labels)
        )
        if outcome.succeeded:
            self._add_fitting_success(f"Plot exported: {outcome.value}")
            self.status_updated.emit(f"Plot exported to {outcome.value.name}")
        else:
            self._add_fitting_error(f"Could not export the plot: {outcome.error.message}")

    def _plotted_figure_series(self):
        """The data and model layers drawn on the curve plot, its axis labels and scales.

        Read from the plot itself so every render path exports what is shown; the
        fitting-range guides (unlabelled lines) are left out.
        """
        from matplotlib.colors import to_hex

        figure = getattr(self, "_current_fit_figure", None)
        if figure is None or not figure.axes:
            return (), {}
        axes = figure.axes[0]
        series = []
        for artist in axes.collections:
            label = str(artist.get_label())
            offsets = np.asarray(artist.get_offsets(), dtype=float)
            if label.startswith("_") or offsets.ndim != 2 or not offsets.size:
                continue
            colors = artist.get_facecolor()
            color = to_hex(colors[0]) if len(colors) else "#1f4e9c"
            series.append(FigureSeries(label, offsets[:, 0], offsets[:, 1], color, "scatter"))
        for line in axes.lines:
            label = str(line.get_label())
            if label.startswith("_"):
                continue
            series.append(
                FigureSeries(
                    label,
                    np.asarray(line.get_xdata(), dtype=float),
                    np.asarray(line.get_ydata(), dtype=float),
                    to_hex(line.get_color()),
                    "line",
                )
            )
        x_scale = axes.get_xscale()
        return tuple(series), dict(
            # "[Fold overlay]"-style notes explain the screen view, not the figure.
            x_label=re.sub(r"\s*\[[^\]]*\]", "", axes.get_xlabel()),
            y_label=axes.get_ylabel(),
            x_scale=x_scale if x_scale in ("linear", "log", "symlog") else "linear",
            log_y=axes.get_yscale() == "log",
        )

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
                if meta.get("source") == "native_v5":
                    # Versioned workflow candidates must never be described by
                    # unrelated manual-model controls or their parameter schema.
                    import json

                    candidates = meta.get("side_candidates") or [meta.get("candidate")]
                    candidates = [row for row in candidates if isinstance(row, dict)]
                    lines.append("# Parameter Source: native_v5_candidate_snapshot")
                    if not candidates:
                        lines.append("# No native workflow candidate snapshot available")
                    for row in candidates:
                        lines.append(f"# Side: {row.get('side', 'not recorded')}")
                        lines.append(f"# Candidate Source: {row.get('best_source', 'not recorded')}")
                        lines.append(f"# Forward Version: {row.get('forward_version', 'not recorded')}")
                        if row.get("model_id"):
                            lines.append(f"# Model ID: {row['model_id']}")
                        lines.append("# Units: " + json.dumps(row.get("unit_contract", {}), sort_keys=True))
                        lines.append("# Component weights are model amplitude weights, not probabilities or volume fractions")
                        for index, component in enumerate(row.get("components", []), 1):
                            lines.append(f"# Particle {index}: shape={component.get('type', 'not recorded')}")
                            values = {**component.get("params", {})}
                            for name in ("type_id", "amplitude", "weight"):
                                if component.get(name) is not None:
                                    values[name] = component[name]
                            for name, value in values.items():
                                if value is not None:
                                    lines.append(f"#   component_{index}_{name} = {float(value):.17g}")
                            if "structure_factor" in component:
                                lines.append(f"#   component_{index}_structure_factor = {bool(component['structure_factor'])}")
                        lines.append("# Global Parameters:")
                        for name, value in row.get("global_params", {}).items():
                            if value is not None:
                                lines.append(f"#   {name} = {float(value):.17g}")
                        # Legacy conditional V5 amplitudes also require these
                        # original reference coefficients and normalization.
                        for name in ("normalizer", "forward_reference"):
                            if row.get(name) is not None:
                                lines.append(f"# {name}: " + json.dumps(row[name], sort_keys=True))
                    lines.append("# Fitting Parameters End")
                    return lines
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
            if choice == "Curve Data":
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
            lines.append(f"# q Branch: {self._get_q_branch()}")
            lines.append(f"# q Combination: {self._get_q_combination_mode()}")
            lines.append(f"# X Scale: {self._get_x_axis_scale()}")
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

            if choice == "Curve Data" and getattr(self, "current_1d_data", None) is not None:
                file_path = self.current_1d_data.get("file_path")
                if file_path:
                    lines.append(f"# Curve File: {file_path}")

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
            if getattr(self, "fitting", None) is not None:
                options.append("Fitting Data")
            if getattr(self, "current_1d_data", None) is not None:
                options.append("Curve Data")
            if not options:
                self._add_fitting_error("No data to export: load a curve or run a fit first")
                return

            dialog = FittingDataExportDialog(tuple(options), self.main_window)
            if dialog.exec_() != QDialog.Accepted:
                return
            selection = dialog.selection()
            choice = selection.source

            x_data = None
            y_data = None
            data_name = ""
            q_source_kind = None
            if choice == "Fitting Data" and self.fitting is not None:
                x_data = np.array(self.fitting.get("q", []))
                y_data = np.array(self.fitting.get("I", []))
                data_name = "Fitting_Data"
                q_source_kind = self.fitting.get("meta", {}).get(
                    "data_source", getattr(self, "data_source", None)
                )
            elif choice == "Curve Data" and self.current_1d_data is not None:
                x_data = np.array(self.current_1d_data.get("q", []))
                y_data = np.array(self.current_1d_data.get("I", []))
                data_name = "Curve_Data"
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

            if selection.preparation != "raw" and choice != "Fitting Data":
                prepared = self._prepare_signed_q_data(x_data, y_data)
                x_data, y_data = prepared.q, prepared.intensity
            if selection.preparation == "fitting" and self._roi_active():
                lower, upper = sorted((float(self._roi_min), float(self._roi_max)))
                roi = (x_data >= lower) & (x_data <= upper)
                x_data, y_data = x_data[roi], y_data[roi]
                if x_data.size < 2:
                    raise ValueError("The current fitting region contains fewer than two points")

            x_data = self._convert_q_values_for_display(x_data, source=q_source_kind)
            x_column_name = self._build_q_axis_label(filter_mode="all", mathtext=False)
            y_column_name = "Intensity (a.u.)"
            header_lines = self._build_export_header_lines(choice, data_name)
            header_lines.append(f"# Export Representation: {selection.preparation}")
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
