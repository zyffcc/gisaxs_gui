"""Roi Range Controls for fitting presentation."""

from __future__ import annotations

import numpy as np


class RoiRangeControlsMixin:
    """Own roi range controls behavior."""

    def _roi_active(self) -> bool:
        if not getattr(self, "_roi_controls_enabled", True):
            return False
        return (
            self._roi_min is not None
            and self._roi_max is not None
            and self._q_full_min is not None
            and self._q_full_max is not None
            and (
                abs(self._roi_min - self._q_full_min) > 1e-12
                or abs(self._roi_max - self._q_full_max) > 1e-12
            )
        )

    def _get_roi_active_arrays(self):
        """Return (q_plot, I_plot) using ROI subset if active, else full arrays."""
        if self.q is None or self.I is None:
            return None, None

        # Helper: filter out non-finite pairs to avoid None/inf/NaN issues
        # 函数说明：实现 过滤 finite 相关逻辑。
        def _filter_finite(q_arr, I_arr):
            q_arr = np.asarray(q_arr)
            I_arr = np.asarray(I_arr)
            mask = np.isfinite(q_arr) & np.isfinite(I_arr)
            if not np.any(mask):
                return q_arr[:0], I_arr[:0]
            return q_arr[mask], I_arr[mask]

        if (
            self._roi_active()
            and self.q_ROI is not None
            and self.I_ROI is not None
            and len(self.q_ROI) > 0
        ):
            return _filter_finite(self.q_ROI, self.I_ROI)
        return _filter_finite(self.q, self.I)

    def _draw_roi_guides_if_active(self, ax):
        try:
            if not getattr(self, "_roi_controls_enabled", True):
                return
            if not self._roi_active():
                return
            q_bounds = self._convert_q_values_for_display(
                np.array([float(self._roi_min), float(self._roi_max)], dtype=float),
                source=getattr(self, "data_source", None),
            )
            if q_bounds.size >= 2:
                try:
                    if self._get_independent_axis_filter_mode() == "negative":
                        q_bounds = np.abs(q_bounds)
                except Exception:
                    pass
                q_bounds = np.sort(q_bounds)
                ax.axvline(
                    float(q_bounds[0]), color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )
                ax.axvline(
                    float(q_bounds[1]), color="red", linestyle="--", linewidth=1.2, alpha=0.8
                )
        except Exception:
            pass

    def _current_q_has_negative_values(self) -> bool:
        try:
            q = np.asarray(self.q) if self.q is not None else None
            if q is None or q.size == 0:
                return False
            return bool(np.any(np.isfinite(q) & (q < 0)))
        except Exception:
            return False

    def _roi_editing_should_be_enabled(self) -> bool:
        """ROI is ambiguous only when log-x folds positive and negative q together."""
        try:
            return not (
                self._is_fit_log_x_enabled()
                and self._current_q_has_negative_values()
                and self._get_independent_axis_filter_mode() == "all"
            )
        except Exception:
            return True

    def _set_roi_controls_enabled(self, enabled: bool):
        self._roi_controls_enabled = bool(enabled)
        for name in (
            "fitFittingRegionSlider",
            "fitFittingRegionMinValue",
            "fitFittingRegionMaxValue",
        ):
            try:
                if hasattr(self.ui, name):
                    widget = getattr(self.ui, name)
                    widget.setEnabled(bool(enabled))
                    if enabled:
                        widget.setToolTip("")
                    else:
                        widget.setToolTip(
                            "Log-X with both +q and -q is ambiguous. Select Positive Only or Negative Only first."
                        )
            except Exception:
                pass
        try:
            if hasattr(self.ui, "fitRegionEditHintLabel"):
                self.ui.fitRegionEditHintLabel.setVisible(not bool(enabled))
        except Exception:
            pass

    def _get_roi_domain_bounds(self):
        if self.q is None or self.I is None:
            return None
        q_all = np.asarray(self.q)
        I_all = np.asarray(self.I)
        valid = np.isfinite(q_all) & np.isfinite(I_all)
        if not np.any(valid):
            return None

        q_valid = q_all[valid]
        log_x = self._is_fit_log_x_enabled()
        filter_mode = self._get_independent_axis_filter_mode()
        if filter_mode == "positive":
            q_valid = q_valid[q_valid > 0]
        elif filter_mode == "negative":
            q_valid = q_valid[q_valid < 0]
        elif log_x and not self._current_q_has_negative_values():
            q_valid = q_valid[q_valid > 0]

        if q_valid.size == 0:
            return None
        return float(np.min(q_valid)), float(np.max(q_valid))

    def _roi_controls_use_abs_negative(self) -> bool:
        try:
            return self._get_independent_axis_filter_mode() == "negative"
        except Exception:
            return False

    def _roi_data_to_control_range(self, q_min: float, q_max: float):
        if self._roi_controls_use_abs_negative():
            vals = np.sort(np.abs(np.array([q_min, q_max], dtype=float)))
            return float(vals[0]), float(vals[1])
        return float(q_min), float(q_max)

    def _roi_data_to_control_values(self, q_min: float, q_max: float):
        return self._roi_data_to_control_range(q_min, q_max)

    def _roi_control_to_data_values(self, vmin: float, vmax: float):
        if self._roi_controls_use_abs_negative():
            lo, hi = sorted((abs(float(vmin)), abs(float(vmax))))
            return -hi, -lo
        return float(vmin), float(vmax)

    def _nearest_roi_control_value(self, value: float):
        try:
            q = np.asarray(self.q) if self.q is not None else None
            if q is None or q.size == 0:
                return float(value)
            finite = q[np.isfinite(q)]
            if finite.size == 0:
                return float(value)
            if self._roi_controls_use_abs_negative():
                finite = np.abs(finite[finite < 0])
            if finite.size == 0:
                return float(value)
            return float(finite[np.argmin(np.abs(finite - value))])
        except Exception:
            return float(value)

    def _sync_roi_controls_to_current_display(self, reset_to_domain: bool = False):
        """Update ROI bounds/editability to match the current Fitting Plot display."""
        enabled = self._roi_editing_should_be_enabled()
        self._set_roi_controls_enabled(enabled)

        bounds = self._get_roi_domain_bounds()
        if bounds is None:
            return
        q_min, q_max = bounds

        self._q_full_min, self._q_full_max = q_min, q_max
        if reset_to_domain or self._roi_min is None or self._roi_max is None or not enabled:
            self._roi_min, self._roi_max = q_min, q_max
        else:
            self._roi_min = max(q_min, min(float(self._roi_min), q_max))
            self._roi_max = max(self._roi_min, min(float(self._roi_max), q_max))

        self._updating_roi_controls = True
        try:
            control_min, control_max = self._roi_data_to_control_range(q_min, q_max)
            control_roi_min, control_roi_max = self._roi_data_to_control_values(
                self._roi_min, self._roi_max
            )
            if hasattr(self.ui, "fitFittingRegionSlider"):
                s = self.ui.fitFittingRegionSlider
                s.setRangeF(control_min, control_max)
                s.setMinValueF(control_roi_min)
                s.setMaxValueF(control_roi_max)
            if hasattr(self.ui, "fitFittingRegionMinValue"):
                self.ui.fitFittingRegionMinValue.setRange(control_min, control_max)
                self.ui.fitFittingRegionMinValue.setValue(control_roi_min)
            if hasattr(self.ui, "fitFittingRegionMaxValue"):
                self.ui.fitFittingRegionMaxValue.setRange(control_min, control_max)
                self.ui.fitFittingRegionMaxValue.setValue(control_roi_max)
        finally:
            self._updating_roi_controls = False

    def _setup_fitting_region_controls(self):
        """Wire up ROI slider/spinboxes and interpolation widgets.
        - Initialize defaults from user settings
        - Connect slider (live) and spinboxes (editingFinished)
        - Defer actual ROI initialization to first import/cut
        """
        # Slider
        if hasattr(self.ui, "fitFittingRegionSlider"):
            try:
                self.ui.fitFittingRegionSlider.setDecimals(4)
            except Exception:
                pass
            try:
                self.ui.fitFittingRegionSlider.rangeChangedF.connect(self._on_roi_slider_changed)
            except Exception:
                if hasattr(self.ui.fitFittingRegionSlider, "rangeChanged"):
                    self.ui.fitFittingRegionSlider.rangeChanged.connect(
                        self._on_roi_slider_changed_int
                    )

        # Min/Max spinboxes
        if hasattr(self.ui, "fitFittingRegionMinValue"):
            try:
                self.ui.fitFittingRegionMinValue.setDecimals(4)
            except Exception:
                pass
            self.ui.fitFittingRegionMinValue.editingFinished.connect(self._on_roi_spin_finished)
        if hasattr(self.ui, "fitFittingRegionMaxValue"):
            try:
                self.ui.fitFittingRegionMaxValue.setDecimals(4)
            except Exception:
                pass
            self.ui.fitFittingRegionMaxValue.editingFinished.connect(self._on_roi_spin_finished)

        # Points number
        if hasattr(self.ui, "fitDataPointsNumValue"):
            try:
                self.ui.fitDataPointsNumValue.setRange(10, 5000)
                self.ui.fitDataPointsNumValue.setSingleStep(1)
                self.ui.fitDataPointsNumValue.setValue(int(max(10, self._points_num_current)))
            except Exception:
                pass
            try:
                # 函数说明：实现 dp after commit 相关逻辑。
                def _dp_after_commit(info, value):
                    try:
                        self._points_num_current = int(value)
                    except Exception:
                        self._points_num_current = int(self._points_num_default)
                    was_fitting = (
                        self._is_in_fitting_mode()
                        if hasattr(self, "_is_in_fitting_mode")
                        else False
                    )

                    if getattr(self, "data_source", None) == "cut":
                        self._perform_cut(points_override=int(self._points_num_current))
                    elif getattr(self, "data_source", None) == "1d":
                        self._resample_1d(n_points=int(self._points_num_current))

                    if was_fitting:
                        self._perform_manual_fitting()

                mode = self._signal_mode_overrides.get(
                    "fitDataPointsNumValue", self._default_signal_mode
                )
                self.param_trigger_manager.register_parameter_widget(
                    widget=self.ui.fitDataPointsNumValue,
                    widget_id="meta_fit_points_num",
                    category="fit_controls",
                    immediate_handler=lambda v: None,
                    delayed_handler=None,
                    connect_signals=True,
                    meta={
                        "persist": "settings",
                        "key_path": ("fitting", "fit.points_num"),
                        "debounce_ms": 0,
                        "epsilon_abs": 0,
                        "epsilon_rel": 0,
                        "after_commit": _dp_after_commit,
                        "trigger_fit": False,
                        "connect_mode": mode,
                    },
                )
            except Exception:
                self.ui.fitDataPointsNumValue.editingFinished.connect(self._on_points_num_finished)

        # Interpolation method
        if hasattr(self.ui, "fitInterpolationMethodValue"):
            combo = self.ui.fitInterpolationMethodValue
            try:
                combo.clear()
                combo.addItems(["Linear", "Quadratic", "Spline"])
                idx = combo.findText(self._interp_method_default)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            except Exception:
                pass
            combo.currentTextChanged.connect(self._on_interp_method_changed)

    def _initialize_roi_from_current_q(self, force_full: bool = False):
        """Initialize or refresh ROI bounds from current q/I arrays.

        - force_full=True resets ROI to full valid range regardless of previous ROI.
        - q_min/q_max are computed from pairs where both q and I are finite (exclude None/inf/NaN).
        """
        if self.q is None or self.I is None:
            return
        q_all = np.asarray(self.q)
        I_all = np.asarray(self.I)
        valid = np.isfinite(q_all) & np.isfinite(I_all)
        if not np.any(valid):
            return
        q_valid = q_all[valid]
        q_min, q_max = float(np.min(q_valid)), float(np.max(q_valid))
        # Update full range tracking to valid bounds
        prev_full = (self._q_full_min, self._q_full_max)
        self._q_full_min, self._q_full_max = q_min, q_max
        # Decide whether to reset ROI to full
        if force_full or self._roi_min is None or self._roi_max is None:
            self._roi_min, self._roi_max = q_min, q_max
        else:
            # If previous ROI is outside new bounds or full bounds changed notably, clamp or reset
            changed = (
                prev_full[0] is None
                or prev_full[1] is None
                or abs(prev_full[0] - q_min) > 1e-12
                or abs(prev_full[1] - q_max) > 1e-12
            )
            if changed:
                self._roi_min, self._roi_max = q_min, q_max
            else:
                # Clamp ROI into new bounds
                self._roi_min = max(q_min, min(self._roi_min, q_max))
                self._roi_max = max(self._roi_min, min(self._roi_max, q_max))
        # Update UI controls
        self._updating_roi_controls = True
        try:
            if hasattr(self.ui, "fitFittingRegionSlider"):
                s = self.ui.fitFittingRegionSlider
                s.setRangeF(q_min, q_max)
                s.setMinValueF(self._roi_min)
                s.setMaxValueF(self._roi_max)
            if hasattr(self.ui, "fitFittingRegionMinValue"):
                self.ui.fitFittingRegionMinValue.setRange(q_min, q_max)
                self.ui.fitFittingRegionMinValue.setValue(self._roi_min)
            if hasattr(self.ui, "fitFittingRegionMaxValue"):
                self.ui.fitFittingRegionMaxValue.setRange(q_min, q_max)
                self.ui.fitFittingRegionMaxValue.setValue(self._roi_max)
        finally:
            self._updating_roi_controls = False
        self._sync_roi_controls_to_current_display(reset_to_domain=force_full)

    def _on_roi_slider_changed_int(self, imin, imax):
        s = self.ui.fitFittingRegionSlider
        dec = 2
        try:
            dec = s.decimals()
        except Exception:
            pass
        scale = 10**dec
        self._on_roi_slider_changed(imin / scale, imax / scale)

    def _on_roi_slider_changed(self, vmin: float, vmax: float):
        if self._updating_roi_controls:
            return
        if not getattr(self, "_roi_controls_enabled", True):
            self._sync_roi_controls_to_current_display(reset_to_domain=True)
            return
        self._slider_is_source = True
        try:
            vmin = self._nearest_roi_control_value(float(vmin))
            vmax = self._nearest_roi_control_value(float(vmax))
            control_min, control_max = self._roi_data_to_control_range(
                self._q_full_min if self._q_full_min is not None else vmin,
                self._q_full_max if self._q_full_max is not None else vmax,
            )
            vmin = max(control_min, min(vmin, vmax))
            vmax = min(control_max, max(vmax, vmin))
            self._roi_min, self._roi_max = self._roi_control_to_data_values(vmin, vmax)
            # Update spinboxes
            self._updating_roi_controls = True
            if hasattr(self.ui, "fitFittingRegionMinValue"):
                self.ui.fitFittingRegionMinValue.setValue(vmin)
            if hasattr(self.ui, "fitFittingRegionMaxValue"):
                self.ui.fitFittingRegionMaxValue.setValue(vmax)
        finally:
            self._updating_roi_controls = False
            self._slider_is_source = False
            self._schedule_roi_refresh()

    def _on_roi_spin_finished(self):
        if self._updating_roi_controls:
            return
        if not getattr(self, "_roi_controls_enabled", True):
            self._sync_roi_controls_to_current_display(reset_to_domain=True)
            return
        vmin = (
            float(self.ui.fitFittingRegionMinValue.value())
            if hasattr(self.ui, "fitFittingRegionMinValue")
            else self._roi_min
        )
        vmax = (
            float(self.ui.fitFittingRegionMaxValue.value())
            if hasattr(self.ui, "fitFittingRegionMaxValue")
            else self._roi_max
        )
        control_min, control_max = self._roi_data_to_control_range(
            self._q_full_min if self._q_full_min is not None else vmin,
            self._q_full_max if self._q_full_max is not None else vmax,
        )
        vmin = max(control_min, vmin)
        vmax = min(control_max, vmax)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        self._roi_min, self._roi_max = self._roi_control_to_data_values(vmin, vmax)
        # Update slider
        self._updating_roi_controls = True
        try:
            if hasattr(self.ui, "fitFittingRegionSlider"):
                s = self.ui.fitFittingRegionSlider
                if self._q_full_min is not None and self._q_full_max is not None:
                    s.setRangeF(control_min, control_max)
                s.setMinValueF(vmin)
                s.setMaxValueF(vmax)
        finally:
            self._updating_roi_controls = False
            self._apply_roi_to_data_and_refresh()

    def _schedule_roi_refresh(self):
        """No description."""
        try:
            from PyQt5.QtCore import QTimer

            if self._roi_update_timer is None:
                self._roi_update_timer = QTimer()
                self._roi_update_timer.setSingleShot(True)
                self._roi_update_timer.timeout.connect(self._apply_roi_to_data_and_refresh)
            delay = int(getattr(self, "_roi_debounce_ms", 140))
            self._roi_update_timer.start(max(20, delay))
        except Exception:
            self._apply_roi_to_data_and_refresh()

    def _apply_roi_to_data_and_refresh(self):
        if self.q is None or self.I is None:
            return
        self._sync_roi_controls_to_current_display(reset_to_domain=False)
        q = np.asarray(self.q)
        I = np.asarray(self.I)
        # Always drop non-finite pairs before ROI masking
        valid = np.isfinite(q) & np.isfinite(I)
        q = q[valid]
        I = I[valid]
        if (
            not getattr(self, "_roi_controls_enabled", True)
            or self._roi_min is None
            or self._roi_max is None
        ):
            self.q_ROI, self.I_ROI = q, I
        else:
            mask = (q >= self._roi_min) & (q <= self._roi_max)
            if not np.any(mask):
                self.q_ROI, self.I_ROI = q, I
            else:
                self.q_ROI = q[mask]
                self.I_ROI = I[mask]
        # Redraw displays
        try:
            current_mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            if current_mode == "fitting" or (
                hasattr(self, "_is_in_fitting_mode") and self._is_in_fitting_mode()
            ):
                self._update_gui_fitting_display()
                current_mode = "fitting"
            self._update_GUI_image(current_mode)
            self._update_outside_window(current_mode)
        except Exception:
            pass
