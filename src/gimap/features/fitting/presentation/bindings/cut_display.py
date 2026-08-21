"""Cut Display for fitting presentation."""

from __future__ import annotations

import numpy as np

from PyQt5.QtWidgets import (
    QMessageBox,
)


from ..binding_primitives import (
    _scientific_commands,
    is_matplotlib_available,
)
from ..state import CurveViewState


_Q_VIEW_RULES = {
    "signed": ("both", "separate"),
    "positive": ("positive", "separate"),
    "negative": ("negative", "separate"),
    "negative_abs": ("negative", "fold"),
    "fold": ("both", "fold"),
    "average": ("both", "average"),
}


class CutDisplayMixin:
    """Own cut display behavior."""

    def _plot_cut_result(self, x_coords, y_intensity, x_label, y_label, title):
        """No description."""
        try:
            x_arr = np.asarray(x_coords, dtype=float)
            y_arr = np.asarray(y_intensity, dtype=float)
            rows_for_debug = None
            if "vertical" in str(title).lower():
                candidate_rows = getattr(self, "_last_vertical_cut_pixel_rows", None)
                try:
                    if candidate_rows is not None and len(candidate_rows) == len(x_arr):
                        rows_for_debug = candidate_rows
                except Exception:
                    rows_for_debug = None
            x_arr, y_arr, _ = self._sort_filter_cut_pairs(
                x_arr,
                y_arr,
                context=title,
                pixel_rows=rows_for_debug,
                log_vertical=("vertical" in str(title).lower()),
            )
            self.q = x_arr
            self.I = y_arr
            self.data_source = "cut"

            self.current_cut_data = {
                "x_coords": x_arr.copy() if hasattr(x_arr, "copy") else list(x_arr),
                "y_intensity": y_arr.copy() if hasattr(y_arr, "copy") else list(y_arr),
                "x_label": x_label,
                "y_label": y_label,
                "title": title,
                "q_source_unit": "nm",
            }
            try:
                import time

                self.cut = {
                    "q": x_arr.copy() if hasattr(x_arr, "copy") else np.array(x_arr),
                    "I": y_arr.copy() if hasattr(y_arr, "copy") else np.array(y_arr),
                    "meta": {
                        "x_label": x_label,
                        "y_label": y_label,
                        "title": title,
                        "timestamp": time.time(),
                        "source": "cut",
                        "q_source_unit": "nm",
                        "q_model_unit": "nm",
                    },
                }
            except Exception:
                self.cut = {"q": x_arr, "I": y_arr, "meta": {"source": "cut"}}

            if getattr(self, "_suppress_workflow_plot_updates", False):
                return

            if not is_matplotlib_available():
                QMessageBox.warning(
                    self.main_window,
                    "Missing Library",
                    "matplotlib library is required for plotting.\nPlease install it using: pip install matplotlib",
                )
                return

            options = self._display_manager.get_display_options()

            if "q" in x_label.lower():
                self._update_GUI_image("normal")
                self._update_outside_window("normal")
            else:
                self._plot_cut_data_legacy(x_arr, y_arr, x_label, y_label, title, options)

            self.status_updated.emit(f"Cut result plotted: {title}")

        except Exception as e:
            self.status_updated.emit(f"Plot failed: {str(e)}")
            QMessageBox.critical(
                self.main_window, "Plot Error", f"Failed to plot cut result:\n{str(e)}"
            )

    def _plot_cut_data_legacy(self, x_coords, y_intensity, x_label, y_label, title, options):
        """No description."""
        try:
            y_data = np.array(y_intensity)
            if options["normalize"]:
                max_intensity = np.max(y_data)
                if max_intensity > 0:
                    y_data = y_data / max_intensity
                    y_label = "Normalized Intensity"

            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return

            fig = Figure(figsize=(8, 6), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            self._plot_cut_data_with_log_handling(
                ax, x_coords, y_data, options["log_x"], markersize=4, linewidth=1.5
            )

            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_label, fontsize=13)
            ax.set_title(title, fontsize=15)
            ax.grid(True, alpha=0.3)

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.8)
            ax.tick_params(axis="both", which="both", width=1.6, labelsize=12)

            self._apply_x_axis_scale(ax)
            if options["log_y"]:
                ax.set_yscale("log")

            fig.tight_layout()

            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(
                self._active_curve_graphics_view(), proxy_widget, keep_aspect=True
            )

            if self.independent_fit_window is not None and self.independent_fit_window.isVisible():
                self.independent_fit_window.update_plot(
                    x_coords,
                    y_intensity,
                    x_label,
                    y_label,
                    title,
                    log_x=options["log_x"],
                    log_y=options["log_y"],
                    normalize=options["normalize"],
                    x_scale=self._get_x_axis_scale(),
                )

            self.status_updated.emit(f"Cut result plotted: {title}")

        except Exception as e:
            self.status_updated.emit(f"Legacy plot cut data error: {str(e)}")

    def _get_checkbox_state(self, checkbox_name: str, default_value: bool = False) -> bool:
        """No description."""
        try:
            if hasattr(self.ui, checkbox_name):
                checkbox = getattr(self.ui, checkbox_name)
                return checkbox.isChecked()
            return default_value
        except Exception:
            return default_value

    def _is_fit_log_x_enabled(self):
        """Return the user's Log X intent, independent of the resolved scale."""
        return self._get_checkbox_state("fitLogXCheckBox", False)

    def _get_x_axis_scale(self):
        if not self._is_fit_log_x_enabled():
            return "linear"
        return "symlog" if self._get_q_view_mode() in {"signed", "negative"} else "log"

    def _is_fit_log_y_enabled(self):
        """No description."""
        return self._get_checkbox_state("fitLogYCheckBox", False)

    def _is_fit_norm_enabled(self):
        """No description."""
        return self._get_checkbox_state("fitNormCheckBox", False)

    def _get_q_display_unit(self):
        """Return the unit stored in the shared curve view state."""
        state = getattr(getattr(self, "fitting_view_model", None), "state", None)
        curve_view = getattr(state, "curve_view", None)
        unit = getattr(curve_view, "q_unit", "nm")
        return unit if unit in ("angstrom", "nm") else "nm"

    def _get_q_source_unit(self, source=None):
        """No description."""
        try:
            if isinstance(source, dict):
                unit = str(source.get("q_source_unit", "")).lower()
                if unit in ("nm", "angstrom"):
                    return unit
                source = source.get("data_source")

            if source is None:
                source = getattr(self, "data_source", None)

            source_text = str(source or "").lower()
            if "cut" in source_text:
                return "nm"
            if "fit" in source_text and isinstance(getattr(self, "fitting", None), dict):
                meta = self.fitting.get("meta", {})
                unit = str(meta.get("q_source_unit", "")).lower()
                if unit in ("nm", "angstrom"):
                    return unit
            if "1d" in source_text:
                return getattr(self, "_imported_1d_q_unit", "angstrom")
        except Exception:
            pass
        return getattr(self, "_imported_1d_q_unit", "angstrom")

    def _get_q_display_scale(self):
        """No description."""
        return 0.1 if self._get_q_display_unit() == "angstrom" else 1.0

    def _get_q_unit_label(self, mathtext: bool = True):
        """No description."""
        if self._get_q_display_unit() == "nm":
            return "nm$^{-1}$" if mathtext else "nm^-1"
        return r"$\AA^{-1}$" if mathtext else "Angstrom^-1"

    def _convert_q_values_for_model(self, q_values, source=None):
        """No description."""
        return _scientific_commands(self).curve.q_for_model(
            q_values, self._get_q_source_unit(source)
        )

    def _convert_q_values_for_display(self, q_values, source=None):
        """No description."""
        return _scientific_commands(self).curve.q_for_display(
            q_values,
            self._get_q_source_unit(source),
            self._get_q_display_unit(),
        )

    def _build_q_axis_label(
        self, filter_mode: str = "all", absolute: bool = False, mathtext: bool = True
    ):
        """No description."""
        unit_label = self._get_q_unit_label(mathtext=mathtext)
        combination = self._get_q_combination_mode()
        base = "|q|" if absolute or combination in ("fold", "average") else "q"
        suffix = ""
        if filter_mode == "positive":
            suffix = " [Positive Only]"
        elif filter_mode == "negative":
            suffix = " [Negative Only]"
        if combination == "fold":
            suffix += " [Fold overlay]"
        elif combination == "average":
            suffix += " [Average ±q]"
        return f"{base} ({unit_label}){suffix}"

    def _get_q_combination_mode(self):
        return _Q_VIEW_RULES[self._get_q_view_mode()][1]

    def _get_q_branch(self):
        return _Q_VIEW_RULES[self._get_q_view_mode()][0]

    def _get_q_view_mode(self):
        combo = getattr(self.ui, "fitQViewModeComboBox", None)
        value = combo.currentData() if combo is not None else "signed"
        return value if value in _Q_VIEW_RULES else "signed"

    @staticmethod
    def _q_view_mode_from_legacy(branch: str, combination: str) -> str:
        reverse = {
            ("both", "separate"): "signed",
            ("positive", "separate"): "positive",
            ("negative", "separate"): "negative",
            ("negative", "fold"): "negative_abs",
            ("both", "fold"): "fold",
            ("both", "average"): "average",
        }
        return reverse.get((branch, combination), "signed")

    def _is_positive_only_enabled(self):
        """Update axis filtering when the Positive Only option changes."""
        for owner, name in (
            (self.ui, "fitRegionPositiveOnlyCheckBox"),
            (self.ui, "PositiveOnlyCheckBox"),
        ):
            try:
                if owner is not None and hasattr(owner, name) and getattr(owner, name).isChecked():
                    return True
            except Exception:
                pass
        return False

    def _is_negative_only_enabled(self):
        """No description."""
        for owner, name in (
            (self.ui, "fitRegionNegativeOnlyCheckBox"),
        ):
            try:
                if owner is not None and hasattr(owner, name) and getattr(owner, name).isChecked():
                    return True
            except Exception:
                pass
        return False

    def _get_independent_axis_filter_mode(self):
        """No description."""
        return {"both": "all", "positive": "positive", "negative": "negative"}[
            self._get_q_branch()
        ]

    def _sync_axis_filter_controls(self):
        """No description."""
        if getattr(self, "_syncing_axis_filter", False):
            return

        self._syncing_axis_filter = True
        try:
            sender = self.sender()
            mode = self._get_independent_axis_filter_mode()
            q_view_combo = getattr(self.ui, "fitQViewModeComboBox", None)
            if sender is q_view_combo:
                mode = self._get_independent_axis_filter_mode()

            positive_widgets = [
                (self.ui, "fitRegionPositiveOnlyCheckBox"),
                (self.ui, "PositiveOnlyCheckBox"),
            ]
            negative_widgets = [
                (self.ui, "fitRegionNegativeOnlyCheckBox"),
            ]

            if sender is not None:
                for owner, name in positive_widgets:
                    if (
                        owner is not None
                        and hasattr(owner, name)
                        and sender is getattr(owner, name)
                    ):
                        mode = "positive" if sender.isChecked() else "all"
                        break
                for owner, name in negative_widgets:
                    if (
                        owner is not None
                        and hasattr(owner, name)
                        and sender is getattr(owner, name)
                    ):
                        mode = "negative" if sender.isChecked() else "all"
                        break

            # 函数说明：设置checked。
            def _set_checked(owner, name, checked):
                try:
                    if owner is None or not hasattr(owner, name):
                        return
                    widget = getattr(owner, name)
                    widget.blockSignals(True)
                    widget.setChecked(bool(checked))
                    widget.blockSignals(False)
                except Exception:
                    pass

            for owner, name in positive_widgets:
                _set_checked(owner, name, mode == "positive")
            for owner, name in negative_widgets:
                _set_checked(owner, name, mode == "negative")
            if q_view_combo is not None and sender is not q_view_combo:
                target = {"all": "signed", "positive": "positive", "negative": "negative"}[
                    mode
                ]
                index = q_view_combo.findData(target)
                q_view_combo.blockSignals(True)
                q_view_combo.setCurrentIndex(max(0, index))
                q_view_combo.blockSignals(False)
        finally:
            self._syncing_axis_filter = False

    def _filter_q_data_for_independent_display(self, q_data, y_data=None):
        """No description."""
        if y_data is None:
            q_array = np.asarray([] if q_data is None else q_data, dtype=float)
            y_array = np.zeros(q_array.shape, dtype=float)
            prepared = self._prepare_signed_q_data(q_array, y_array)
            prepared_y = None
        else:
            prepared = self._prepare_signed_q_data(q_data, y_data)
            prepared_y = prepared.intensity
        filter_mode = self._get_independent_axis_filter_mode()
        return prepared.q, prepared.q.copy(), prepared_y, filter_mode

    def _prepare_signed_q_data(self, q_data, intensity_data):
        return _scientific_commands(self).curve.prepare_signed(
            q_data,
            intensity_data,
            branch=self._get_q_branch(),
            combination=self._get_q_combination_mode(),
        )

    def _get_fit_y_range_mode(self):
        """Return the y-limit policy shared by both curve projections."""
        state = getattr(getattr(self, "fitting_view_model", None), "state", None)
        curve_view = getattr(state, "curve_view", None)
        mode = getattr(curve_view, "y_range", "all")
        return mode if mode in ("experimental", "fitting", "all") else "all"

    def _current_curve_view_state(self, *, sync_window: bool = True) -> CurveViewState:
        """Capture all curve display controls as the single presentation state."""
        previous = getattr(self.fitting_view_model.state, "curve_view", CurveViewState())
        state = CurveViewState(
            q_mode=self._get_q_view_mode(),
            layer_mode=self._get_curve_view_mode(),
            log_x=self._is_fit_log_x_enabled(),
            log_y=self._is_fit_log_y_enabled(),
            normalize=self._is_fit_norm_enabled(),
            q_unit=previous.q_unit,
            y_range=previous.y_range,
        )
        self.fitting_view_model.update_curve_view(state)
        window = getattr(self, "independent_fit_window", None)
        if sync_window and window is not None and hasattr(window, "set_curve_view_state"):
            window.set_curve_view_state(state)
        return state

    def _apply_curve_view_state(self, state: CurveViewState, *, refresh: bool = True) -> None:
        """Apply state from the independent window back to the embedded controls."""
        if not isinstance(state, CurveViewState):
            return
        previous = getattr(self.fitting_view_model.state, "curve_view", CurveViewState())
        controls = (
            (getattr(self.ui, "fitQViewModeComboBox", None), state.q_mode),
            (getattr(self.ui, "fitCurveViewModeComboBox", None), state.layer_mode),
        )
        for combo, value in controls:
            if combo is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                old_block = combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(old_block)
        for name, checked in (
            ("fitLogXCheckBox", state.log_x),
            ("fitLogYCheckBox", state.log_y),
            ("fitNormCheckBox", state.normalize),
        ):
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            old_block = widget.blockSignals(True)
            widget.setChecked(bool(checked))
            widget.blockSignals(old_block)
        self.fitting_view_model.update_curve_view(state)
        window = getattr(self, "independent_fit_window", None)
        if window is not None and hasattr(window, "set_curve_view_state"):
            window.set_curve_view_state(state)
        if not refresh:
            return
        if previous.q_mode != state.q_mode:
            self._on_q_preparation_changed()
        else:
            self._update_q_view_hint()
            self._refresh_curve_layers()

    def _on_independent_curve_view_state_changed(self, state: CurveViewState) -> None:
        self._apply_curve_view_state(state)

    def _valid_y_values_for_limits(self, y_values, log_y=False):
        """No description."""
        try:
            return _scientific_commands(self).curve.valid_y_for_limits(y_values, log_y=log_y)
        except Exception:
            return np.asarray([], dtype=float)

    def _apply_fit_y_axis_limits(
        self, ax, experimental_y=None, fitting_y=None, extra_y_values=None, log_y=False
    ):
        """No description."""
        try:
            mode = self._get_fit_y_range_mode()
            y_sources = []

            if mode == "experimental":
                y_sources.append(experimental_y)
            elif mode == "fitting":
                y_sources.append(fitting_y)
            else:
                y_sources.extend([experimental_y, fitting_y])
                if extra_y_values:
                    y_sources.extend(extra_y_values)

            valid_parts = [
                self._valid_y_values_for_limits(values, log_y=log_y) for values in y_sources
            ]
            valid_parts = [values for values in valid_parts if values.size > 0]
            if not valid_parts:
                return

            values = np.concatenate(valid_parts)
            y_min = float(np.min(values))
            y_max = float(np.max(values))
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                return

            if log_y:
                if y_min <= 0 or y_max <= 0:
                    return
                if y_min == y_max:
                    ax.set_ylim(y_min / 1.5, y_max * 1.5)
                else:
                    ax.set_ylim(y_min / 1.08, y_max * 1.08)
                return

            if y_min == y_max:
                pad = abs(y_min) * 0.08 if y_min != 0 else 1.0
            else:
                pad = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - pad, y_max + pad)
        except Exception:
            pass

    def _normalize_intensity_data(self, I_data):
        """No description."""
        return _scientific_commands(self).curve.normalize_intensity(I_data)

    def _apply_log_scales(self, ax, log_x=False, log_y=False):
        """No description."""
        self._apply_x_axis_scale(ax)
        if log_y:
            ax.set_yscale("log")

    def _apply_x_axis_scale(self, ax):
        scale = self._get_x_axis_scale()
        if scale == "symlog":
            q = np.asarray(getattr(self, "q", []), dtype=float)
            nonzero = np.abs(q[np.isfinite(q) & (q != 0)])
            linthresh = float(np.min(nonzero) * 0.5) if nonzero.size else 1e-6
            ax.set_xscale("symlog", linthresh=max(linthresh, 1e-12))
        else:
            ax.set_xscale(scale)

    def _has_valid_data(self):
        """,I"""
        return (
            hasattr(self, "q")
            and hasattr(self, "I")
            and self.q is not None
            and self.I is not None
            and len(self.q) > 0
            and len(self.I) > 0
        )
