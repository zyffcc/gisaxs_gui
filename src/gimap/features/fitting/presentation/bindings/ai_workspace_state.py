"""Ai Workspace State for fitting presentation."""

from __future__ import annotations


from pathlib import Path

import numpy as np


from src.gimap.features.fitting.application import (
    ConstraintSet,
)

from ..binding_primitives import (
    _ai_catalog,
    _scientific_commands,
)


class AiWorkspaceStateMixin:
    """Own ai workspace state behavior."""

    def _set_ai_workspace_status(self, text: str, progress: int = None) -> None:
        main_label = getattr(self.ui, "aiFittingStatusLabel", None) or getattr(
            self.ui, "fitMethodInfoLabel", None
        )
        if main_label is not None:
            main_label.setText(f"Status: {text}")
        label = getattr(self, "_ai_status_label", None)
        if label is not None:
            label.setText(f"Status: {text}")
        bar = getattr(self, "_ai_progress", None)
        if bar is not None and progress is not None:
            bar.setValue(int(progress))
        browser = getattr(self, "_ai_log_browser", None)
        if browser is not None:
            browser.append(text)

    def _ai_workspace_placeholder(self, action_name: str) -> None:
        if action_name == "Advanced Constraints":
            self._show_advanced_constraints_dialog()
            return
        if action_name == "Show Results":
            self._show_ai_candidate_table()
            return
        self._set_ai_workspace_status(
            f"{action_name} is available after a prediction run.",
            0,
        )

    def _reset_ai_workspace_defaults(self) -> None:
        self._set_ai_profile(_ai_catalog(self).default_profile_name)
        self._save_ai_fitting_settings(
            constraint_set=ConstraintSet.defaults().to_dict(),
            d_spacing_rule="max_diameter",
            parameter_constraints={},
        )
        combo = getattr(self, "_ai_constraint_combo", None)
        if combo is not None:
            combo.setCurrentText("Free")
        self._set_ai_workspace_status("Balanced profile and model-default constraints restored.", 0)

    def _selected_ai_model_path(self) -> Path | None:
        for combo in (
            getattr(self, "_ai_model_combo", None),
            getattr(self.ui, "aiFittingModelComboBox", None),
        ):
            if combo is None or combo.currentIndex() < 0:
                continue
            data = combo.itemData(combo.currentIndex())
            if data:
                return Path(str(data))
        selected = self._ai_fitting_settings().get("last_selected_model")
        return Path(str(selected)) if selected else None

    def _current_ai_curve_arrays(self, apply_exclusions: bool = True):
        """Map current legacy curve state to the pure AI curve preparation service."""
        axis_filter = self._get_independent_axis_filter_mode()
        excluded = (
            set(getattr(self, "_ai_excluded_input_q", set()) or set())
            if apply_exclusions
            else set()
        )
        roi = None
        if (
            getattr(self, "_roi_controls_enabled", True)
            and self._roi_min is not None
            and self._roi_max is not None
        ):
            roi = (float(self._roi_min), float(self._roi_max))

        sources = []
        if self.q_ROI is not None and self.I_ROI is not None:
            sources.append((self.q_ROI, self.I_ROI, None, None))
        if self.q is not None and self.I is not None:
            sources.append((self.q, self.I, None, roi))
        if isinstance(getattr(self, "current_1d_data", None), dict):
            data = self.current_1d_data
            sources.append((data.get("q", []), data.get("I", []), data.get("err"), roi))
        if isinstance(getattr(self, "cut", None), dict):
            sources.append((self.cut.get("q", []), self.cut.get("I", []), None, roi))

        for q_values, intensities, sigma, source_roi in sources:
            try:
                curve = _scientific_commands(self).ai.prepare_curve(
                    q_values,
                    intensities,
                    sigma,
                    axis_filter=axis_filter,
                    roi=source_roi,
                    excluded_q=excluded,
                    minimum_points=16,
                )
                return curve.q, curve.intensity, curve.sigma
            except (TypeError, ValueError):
                continue
        return None

    def _ai_q_key(self, q_value) -> str:
        return _scientific_commands(self).ai.q_key(q_value)

    def _filter_ai_excluded_points_for_display(self, q_arr, *value_arrays):
        excluded = getattr(self, "_ai_excluded_input_q", set()) or set()
        if not excluded:
            return (q_arr, *value_arrays)
        try:
            q_np = np.asarray(q_arr)
            keep = np.array(
                [
                    self._ai_q_key(q_val) not in excluded
                    and self._ai_q_key(abs(float(q_val))) not in excluded
                    for q_val in q_np
                ],
                dtype=bool,
            )
            if int(np.sum(keep)) == 0:
                return (q_arr, *value_arrays)
            filtered = [q_np[keep]]
            for arr in value_arrays:
                if arr is None:
                    filtered.append(None)
                    continue
                arr_np = np.asarray(arr)
                if arr_np.shape[0] == q_np.shape[0]:
                    filtered.append(arr_np[keep])
                else:
                    filtered.append(arr)
            return tuple(filtered)
        except Exception:
            return (q_arr, *value_arrays)
