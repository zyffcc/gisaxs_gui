"""Roi Curve Processing for fitting presentation."""

from __future__ import annotations

import numpy as np

from ..binding_primitives import (
    _scientific_commands,
)


class RoiCurveProcessingMixin:
    """Own roi curve processing behavior."""

    def _get_current_fit_axes(self):
        """Try to get the current Matplotlib Axes used by the in-GUI 1D plot."""
        try:
            fig = getattr(self, "_current_fit_figure", None)
            if fig is None:
                return None
            axes = getattr(fig, "axes", None)
            if not axes:
                return None
            return axes[0] if len(axes) > 0 else None
        except Exception:
            return None

    def _compute_display_xmin_for_log(self) -> float:
        """Compute a safe lower X bound to use when log-x is enabled.

        Priority:
        1) If current Axes exist, use its left xlim (must be > 0).
        2) Else, use min positive finite q from current data (full q preferred).
        3) Fallback to a tiny positive epsilon.
        """
        # 1) Try current axes xlim
        try:
            ax = self._get_current_fit_axes()
            if ax is not None:
                x0, _ = ax.get_xlim()
                if np.isfinite(x0) and x0 > 0:
                    return float(x0)
        except Exception:
            pass
        # 2) Use data-based min positive q
        try:
            q_all = None
            if self.q is not None:
                q_all = np.asarray(self.q)
            elif self.current_cut_data is not None and "x_coords" in self.current_cut_data:
                q_all = np.asarray(self.current_cut_data["x_coords"])
            elif self.current_1d_data is not None and "q" in self.current_1d_data:
                q_all = np.asarray(self.current_1d_data["q"])
            if q_all is not None and q_all.size > 0:
                pos = q_all[np.isfinite(q_all) & (q_all > 0)]
                if pos.size > 0:
                    return float(np.min(pos))
        except Exception:
            pass
        # 3) Fallback
        return 1e-12

    def _adjust_roi_bounds_for_log_x(self):
        """When log-x is enabled, ensure ROI slider/spin ranges start from the displayed x-axis min (>0).

        Also clamp current ROI values to the new bounds.
        """
        try:
            self._sync_roi_controls_to_current_display(reset_to_domain=False)
            return
        except Exception:
            pass

        try:
            log_x = self._is_fit_log_x_enabled()
        except Exception:
            log_x = False

        if not hasattr(self.ui, "fitFittingRegionSlider"):
            return

        try:
            s = self.ui.fitFittingRegionSlider
            # Determine full max from existing tracking or data
            q_max = None
            if self._q_full_max is not None:
                q_max = float(self._q_full_max)
            else:
                try:
                    q_all = np.asarray(self.q) if self.q is not None else None
                    if q_all is not None and q_all.size > 0:
                        q_max = float(np.nanmax(q_all[np.isfinite(q_all)]))
                except Exception:
                    q_max = None
            if q_max is None:
                return

            if log_x:
                xmin = self._compute_display_xmin_for_log()
                # Ensure xmin < q_max
                if not np.isfinite(xmin) or xmin <= 0 or xmin >= q_max:
                    xmin = min(max(1e-12, (q_max * 1e-6)), q_max * 0.5)  # conservative fallback
                new_min, new_max = float(xmin), float(q_max)
            else:
                # Restore to full valid data bounds if known
                q_min = float(self._q_full_min) if self._q_full_min is not None else None
                if q_min is None:
                    try:
                        q_all = np.asarray(self.q) if self.q is not None else None
                        if q_all is not None and q_all.size > 0:
                            q_min = float(np.nanmin(q_all[np.isfinite(q_all)]))
                    except Exception:
                        q_min = None
                if q_min is None:
                    return
                new_min, new_max = float(q_min), float(q_max)

            # Update control ranges and clamp current ROI values
            self._updating_roi_controls = True
            try:
                s.setRangeF(new_min, new_max)
                # Clamp current ROI values
                if self._roi_min is None or self._roi_max is None:
                    cur_min, cur_max = new_min, new_max
                else:
                    cur_min = max(new_min, min(float(self._roi_min), new_max))
                    cur_max = max(cur_min, min(float(self._roi_max), new_max))
                self._roi_min, self._roi_max = cur_min, cur_max
                s.setMinValueF(cur_min)
                s.setMaxValueF(cur_max)
                # Update spinbox ranges to match
                if hasattr(self.ui, "fitFittingRegionMinValue"):
                    self.ui.fitFittingRegionMinValue.setRange(new_min, new_max)
                    self.ui.fitFittingRegionMinValue.setValue(cur_min)
                if hasattr(self.ui, "fitFittingRegionMaxValue"):
                    self.ui.fitFittingRegionMaxValue.setRange(new_min, new_max)
                    self.ui.fitFittingRegionMaxValue.setValue(cur_max)
            finally:
                self._updating_roi_controls = False
        except Exception:
            pass

    def _on_points_num_finished(self):
        n = None
        try:
            if hasattr(self.ui, "fitDataPointsNumValue"):
                n = int(self.ui.fitDataPointsNumValue.value())
        except Exception:
            n = None
        if n is None:
            return
        if n < 10:
            n = 10
        # Keep a stable in-controller cache for repeated cuts
        try:
            self._points_num_current = int(n)
        except Exception:
            self._points_num_current = int(self._points_num_default)
        # Persist
        try:
            self.preferences.set("fit.points_num", int(n))
            self.preferences.save()
        except Exception:
            pass
        was_fitting = self._is_in_fitting_mode() if hasattr(self, "_is_in_fitting_mode") else False

        if getattr(self, "data_source", None) == "cut":
            self._perform_cut(points_override=n)
        elif getattr(self, "data_source", None) == "1d":
            self._resample_1d(n_points=n)

        if was_fitting:
            self._perform_manual_fitting()

    def _on_interp_method_changed(self, method: str):
        meth = method or "Linear"
        try:
            self.preferences.set("fit.interp_method", meth)
            self.preferences.save()
        except Exception:
            pass
        if self.data_source == "1d" and self.q is not None:
            self._resample_1d(n_points=len(self.q), method=meth, keep_same_count=True)
        elif self.data_source == "cut":
            self._perform_cut()

    def _resample_1d(self, n_points: int, method: str = None, keep_same_count: bool = False):
        if self.current_1d_data is None or self.q is None or self.I is None:
            return
        q0 = np.asarray(self.current_1d_data.get("q", self.q))
        I0 = np.asarray(self.current_1d_data.get("I", self.I))
        if q0.size < 2:
            return
        method = method or (
            self.ui.fitInterpolationMethodValue.currentText()
            if hasattr(self.ui, "fitInterpolationMethodValue")
            else "Linear"
        )
        if keep_same_count:
            n_points = len(self.q)
        # Fallback to stable current points if not valid
        try:
            n_points = int(max(10, n_points))
        except Exception:
            n_points = int(max(10, getattr(self, "_points_num_current", self._points_num_default)))
        q_new = np.linspace(q0.min(), q0.max(), int(max(10, n_points)))
        I_new = self._interpolate_series(q0, I0, q_new, method)
        self.q, self.I = q_new, I_new
        if self._q_full_min is None or self._q_full_max is None:
            self._initialize_roi_from_current_q()
        self._apply_roi_to_data_and_refresh()

    def _interpolate_series(self, x, y, x_new, method: str):
        return _scientific_commands(self).cut.interpolate(x, y, x_new, method)

    def _log_cut_debug(self, message: str):
        """Send cut diagnostics to the fitting log without interrupting the cut flow."""
        try:
            self._add_fitting_message(message, "INFO")
        except Exception:
            try:
                self.status_updated.emit(message)
            except Exception:
                pass

    def _get_axis_filter_debug_count(self, q_values):
        try:
            mode = self._get_independent_axis_filter_mode()
        except Exception:
            mode = "all"
        q_arr = np.asarray(q_values, dtype=float)
        if mode == "positive":
            count = int(np.sum(q_arr > 0))
        elif mode == "negative":
            count = int(np.sum(q_arr < 0))
        else:
            count = int(q_arr.size)
        return mode, count

    def _sort_filter_cut_pairs(
        self,
        x_values,
        intensity_values,
        context: str = "cut",
        pixel_rows=None,
        log_vertical: bool = False,
    ):
        """Keep cut x/intensity arrays paired while removing bad values and sorting by x."""
        raw_x = np.asarray(x_values, dtype=float).reshape(-1)
        if log_vertical and raw_x.size:
            self._log_cut_debug(
                f"{context}: first/last q before sorting = {raw_x[0]:.8g}, {raw_x[-1]:.8g}"
            )
        x_arr, y_arr, rows_arr = _scientific_commands(self).cut.sort_filter(
            x_values,
            intensity_values,
            context=context,
            pixel_rows=pixel_rows,
            on_diagnostic=self._log_cut_debug,
        )
        if log_vertical:
            monotonic = bool(np.all(np.diff(x_arr) >= 0)) if x_arr.size > 1 else True
            self._log_cut_debug(
                f"{context}: first/last q after sorting = {x_arr[0]:.8g}, "
                f"{x_arr[-1]:.8g}; monotonic={monotonic}"
            )
            try:
                max_idx = int(np.nanargmax(y_arr))
                if rows_arr is not None and rows_arr.size:
                    self._log_cut_debug(
                        f"{context}: max intensity row={int(rows_arr[max_idx])}, "
                        f"q={x_arr[max_idx]:.8g}"
                    )
                else:
                    self._log_cut_debug(f"{context}: max intensity q={x_arr[max_idx]:.8g}")
            except Exception as exc:
                self._log_cut_debug(f"{context}: max intensity diagnostic failed: {exc}")
            mode, count = self._get_axis_filter_debug_count(x_arr)
            self._log_cut_debug(
                f"{context}: points after {mode} axis filter would be {count}/{x_arr.size}"
            )
        return x_arr, y_arr, rows_arr

    def _filter_cut_pairs_for_active_axis(self, q_values, intensity_values, context="cut"):
        """Apply Positive/Negative Only before resampling so all points cover the visible domain."""
        try:
            mode = self._get_independent_axis_filter_mode()
        except Exception:
            mode = "all"
        q_arr, intensity_arr = _scientific_commands(self).cut.filter_axis(
            q_values,
            intensity_values,
            mode,
            context=context,
        )
        self._log_cut_debug(
            f"{context}: {q_arr.size} native point(s) remain after {mode} axis filtering."
        )
        return q_arr, intensity_arr
