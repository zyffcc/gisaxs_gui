"""Ai Workspace State for fitting presentation."""

from __future__ import annotations



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
