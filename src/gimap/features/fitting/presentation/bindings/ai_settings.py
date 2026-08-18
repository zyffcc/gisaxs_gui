"""Ai Settings for fitting presentation."""

from __future__ import annotations


import copy

from pathlib import Path


from src.gimap.features.fitting.application import (
    ConstraintSet,
)

from ..binding_primitives import (
    _ai_catalog,
)


class AiSettingsMixin:
    """Own ai settings behavior."""

    def _ai_fitting_settings(self) -> dict:
        try:
            settings = self.preferences.get("ai_fitting", {})
            return settings if isinstance(settings, dict) else {}
        except Exception:
            return {}

    def _save_ai_fitting_settings(self, **updates) -> None:
        try:
            settings = self._ai_fitting_settings()
            settings.update(updates)
            self.preferences.set("ai_fitting", settings)
            self.preferences.save()
        except Exception:
            pass

    def _default_ai_run_settings(self) -> dict:
        return {
            "profile": _ai_catalog(self).default_profile_name,
            "profile_overrides": {},
            "random_seed": 123,
            "time_budget_seconds": _ai_catalog(self).profile().time_budget_seconds,
            "constraint_set": ConstraintSet.defaults().to_dict(),
            "d_spacing_rule": "max_diameter",
            "full_num_samples": 2000,
            "full_top_k": 20,
            "full_refine_top_n": 5,
            "full_refine_max_nfev": 80,
            "full_refine_progress_interval": 20,
            "full_refine_ftol": 1e-8,
            "full_refine_xtol": 1e-8,
            "full_refine_gtol": 1e-8,
            "full_refine_stall_patience": 0,
            "full_refine_stall_tol": 1e-4,
            "full_refine_target_logrmse": 0.08,
            "full_sampling_std": 0.005,
            "fast_num_samples": 128,
            "fast_top_k": 20,
            "fast_progress_interval": 16,
            "parameter_constraints": {},
        }

    def _ai_run_settings(self) -> dict:
        settings = self._default_ai_run_settings()
        stored = self._ai_fitting_settings()
        for key in settings:
            if key in stored:
                settings[key] = stored[key]
        return settings

    def _restore_ai_session_settings(self, payload) -> None:
        """Migrate saved AI settings while keeping old sessions loadable."""
        if not isinstance(payload, dict):
            return
        defaults = self._default_ai_run_settings()
        updates = {key: copy.deepcopy(value) for key, value in payload.items() if key in defaults}
        if not updates:
            return
        self._save_ai_fitting_settings(**updates)
        self._restore_ai_run_settings_to_widgets()
        self._sync_workspace_ai_run_widgets()

    def _current_ai_profile(self):
        settings = self._ai_run_settings()
        selected = str(settings.get("profile") or _ai_catalog(self).default_profile_name)
        if not _ai_catalog(self).has_profile(selected):
            selected = _ai_catalog(self).default_profile_name
        profile = _ai_catalog(self).profile(selected)
        overrides = settings.get("profile_overrides")
        if isinstance(overrides, dict) and overrides:
            allowed = set(profile.to_dict()) - {"name"}
            cleaned = {key: value for key, value in overrides.items() if key in allowed}
            if cleaned:
                profile = profile.with_updates(**cleaned)
        seed = int(settings.get("random_seed", profile.random_seed))
        budget = settings.get("time_budget_seconds", profile.time_budget_seconds)
        if budget in ("", 0, 0.0):
            budget = None
        return (
            profile.with_updates(random_seed=seed, time_budget_seconds=budget)
            if (seed != profile.random_seed or budget != profile.time_budget_seconds)
            else profile
        )

    def _set_ai_profile(self, name: str) -> None:
        if not _ai_catalog(self).has_profile(name):
            return
        profile = _ai_catalog(self).profile(name)
        self._save_ai_fitting_settings(
            profile=name,
            profile_overrides={},
            random_seed=profile.random_seed,
            time_budget_seconds=profile.time_budget_seconds,
        )
        self._sync_workspace_ai_run_widgets()
        label = getattr(self, "_ai_profile_state_label", None)
        if label is not None:
            label.setText(name)
        self._set_ai_workspace_status(f"{name} profile restored.", None)

    def _mark_ai_profile_custom(self, **updates) -> None:
        settings = self._ai_run_settings()
        overrides = settings.get("profile_overrides")
        overrides = dict(overrides) if isinstance(overrides, dict) else {}
        overrides.update(updates)
        self._save_ai_fitting_settings(profile_overrides=overrides)
        label = getattr(self, "_ai_profile_state_label", None)
        if label is not None:
            label.setText("Custom")

    def _connect_ai_fitting_settings_widgets(self) -> None:
        self._restore_ai_run_settings_to_widgets()
        widget_map = {
            "aiFittingSamplesSpinBox": "candidate_count",
            "aiFittingRefineTopNSpinBox": "refinement_count",
            "aiFittingRefineMaxEvalSpinBox": "max_evaluations",
            "aiFittingProgressEverySpinBox": "progress_interval",
            "aiFittingRefineFtolSpinBox": "tolerance",
            "aiFittingRefineXtolSpinBox": "tolerance",
            "aiFittingRefineGtolSpinBox": "tolerance",
            "aiFittingSamplingStdSpinBox": "sampling_std",
            "aiFittingTargetLogRmseSpinBox": "target_log_rmse",
        }
        for widget_name, setting_key in widget_map.items():
            widget = getattr(self.ui, widget_name, None)
            if widget is None or widget.property("aiSettingConnected"):
                continue
            widget.valueChanged.connect(
                lambda value, key=setting_key: self._mark_ai_profile_custom(**{key: value})
            )
            widget.setProperty("aiSettingConnected", True)

    def _restore_ai_run_settings_to_widgets(self) -> None:
        profile = self._current_ai_profile()
        widget_map = {
            "aiFittingSamplesSpinBox": profile.candidate_count,
            "aiFittingRefineTopNSpinBox": profile.refinement_count,
            "aiFittingRefineMaxEvalSpinBox": profile.max_evaluations,
            "aiFittingProgressEverySpinBox": profile.progress_interval,
            "aiFittingRefineFtolSpinBox": profile.tolerance,
            "aiFittingRefineXtolSpinBox": profile.tolerance,
            "aiFittingRefineGtolSpinBox": profile.tolerance,
            "aiFittingSamplingStdSpinBox": profile.sampling_std,
            "aiFittingTargetLogRmseSpinBox": profile.target_log_rmse,
        }
        for widget_name, value in widget_map.items():
            widget = getattr(self.ui, widget_name, None)
            if widget is None:
                continue
            try:
                widget.blockSignals(True)
                widget.setValue(value)
            finally:
                widget.blockSignals(False)

    def _sync_workspace_ai_run_widgets(self) -> None:
        profile = self._current_ai_profile()
        workspace_map = {
            "_ai_full_samples_spin": profile.candidate_count,
            "_ai_refine_top_n_spin": profile.refinement_count,
            "_ai_refine_max_nfev_spin": profile.max_evaluations,
            "_ai_progress_every_spin": profile.progress_interval,
            "_ai_refine_ftol_spin": profile.tolerance,
            "_ai_refine_xtol_spin": profile.tolerance,
            "_ai_refine_gtol_spin": profile.tolerance,
            "_ai_stall_patience_spin": profile.stall_patience,
            "_ai_stall_tol_spin": profile.stall_tolerance,
            "_ai_sampling_std_spin": profile.sampling_std,
            "_ai_target_logrmse_spin": profile.target_log_rmse,
        }
        for attr, value in workspace_map.items():
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            try:
                widget.blockSignals(True)
                widget.setValue(value)
            finally:
                widget.blockSignals(False)

    def _ai_fitting_base_dirs(self) -> list:
        settings = self._ai_fitting_settings()
        stored = settings.get("model_base_dirs")
        dirs = []
        if isinstance(stored, list):
            dirs.extend(Path(p) for p in stored if isinstance(p, str) and p.strip())
        dirs.extend(_ai_catalog(self).default_model_directories(Path.cwd()))
        extra = settings.get("extra_model_paths")
        if isinstance(extra, list):
            dirs.extend(Path(p) for p in extra if isinstance(p, str) and p.strip())
        unique = []
        seen = set()
        for path in dirs:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _scan_ai_fitting_models(self) -> list:
        return list(_ai_catalog(self).discover_models(self._ai_fitting_base_dirs()))
