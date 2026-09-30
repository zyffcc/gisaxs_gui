"""Adapt Fitting widgets to explicit, framework-neutral In-situ Recipe commands."""

from __future__ import annotations

import json

from ...application import (
    InSituFittingPolicy,
    InSituTrackingPolicy,
    SingleAnalysisRecipeSnapshot,
)


class InsituRecipeBindingMixin:
    """Capture the Single-analysis curve fit as a versioned Recipe for the series."""

    def _capture_current_insitu_recipe(self):
        """Adapt current widgets to a framework-neutral explicit Recipe command."""
        try:
            curve = getattr(self, "current_1d_data", None)
            if not isinstance(curve, dict) or getattr(self, "q", None) is None:
                raise ValueError(
                    "Open one representative curve in Single analysis first "
                    "(Analyze ▸ Send to Fitting, or Open Curve…)."
                )
            payload = self._current_insitu_recipe_payload()
            snapshot = SingleAnalysisRecipeSnapshot(
                experiment_setup=payload["experiment_setup"],
                preprocessing=payload["preprocessing"],
                cut=payload["cut"],
                model=payload["model"],
                tracking=InSituTrackingPolicy(center="fixed", yoneda="fixed"),
                fitting=InSituFittingPolicy(
                    initialization="ai_each_frame",
                    refinement="plot_only",
                    failure="continue",
                ),
                note=f"Captured from representative curve: {curve.get('file_path', '')}",
            )
            recipe = self.fitting_view_model.insitu.create_recipe_from_single(snapshot)
            page = getattr(self.ui, "fittingInsituSeriesPage", None)
            if page is not None:
                page.render_recipe(recipe)
            self._populate_insitu_sequence_folder_default()
            self._refresh_insitu_workflow_step_styles()
            self._add_fitting_success(
                f"In-situ Recipe v{recipe.version} captured from Single analysis"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self._add_fitting_error(str(exc))

    def _current_insitu_recipe_payload(self) -> dict[str, dict[str, object]]:
        """The curve source (what was fitted) and the model (how it was fitted)."""
        curve = getattr(self, "current_1d_data", None) or {}
        observation = curve.get("observation") or {}
        parameter_snapshot = self._build_fitting_parameter_snapshot()
        fitting_values = parameter_snapshot.get("fitting", {})
        model = {
            "workflow_v5": self._workflow_options() if hasattr(self, "_workflow_options") else {},
            "workflow_input_selection": (
                self._workflow_input_selection() if hasattr(self, "_workflow_input_selection") else {}
            ),
            "schema": parameter_snapshot.get("schema", "gimap_fitting_parameters_v1"),
            "model_parameters": parameter_snapshot.get("model_parameters", {}),
            "fitting_params": (
                fitting_values.get("fitting_params", {})
                if isinstance(fitting_values, dict)
                else {}
            ),
        }
        return {
            "experiment_setup": {},
            "preprocessing": {},
            "cut": {
                "source": "curve",
                "q_source_unit": str(curve.get("q_source_unit", "angstrom")),
                "observation": dict(observation),
            },
            "model": model,
        }

    def _current_ui_matches_insitu_recipe(self) -> bool:
        recipe = self.fitting_view_model.insitu.recipe
        if recipe is None:
            return True
        try:
            current = self._current_insitu_recipe_payload()["model"]
            return json.dumps(current, sort_keys=True, ensure_ascii=False) == json.dumps(
                recipe.to_dict()["model"], sort_keys=True, ensure_ascii=False
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _insitu_recipe_start_error(self) -> str:
        dialog = getattr(self, "_workflow_v5_dialog", None)
        if dialog is not None and dialog.job is not None:
            return "Finish or cancel the 1D text batch before starting an in-situ sequence."
        if getattr(self, "_ai_job_thread", None) is not None:
            return "Wait for the current 1D fitting job to finish before starting a sequence."
        recipe = self.fitting_view_model.insitu.recipe
        if recipe is None:
            return "Capture the current Single analysis as an In-situ Recipe first."
        if recipe.model.get("workflow_v5"):
            # V5 does not reuse mutable Single model widgets: its conditions and
            # input selection are fully captured in the Recipe.
            return ""
        if self._current_ui_matches_insitu_recipe():
            return ""
        return (
            f"The Single analysis model changed after Recipe v{recipe.version} was captured. "
            "Capture the intended model again before running the series."
        )


__all__ = ["InsituRecipeBindingMixin"]
