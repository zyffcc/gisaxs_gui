"""Adapt Fitting widgets to explicit, framework-neutral In-situ Recipe commands."""

from __future__ import annotations

import json
import math

from ...application import (
    InSituFittingPolicy,
    InSituTrackingPolicy,
    SingleAnalysisRecipeSnapshot,
)


def _recipe_detector_geometry(panel, authoritative):
    """Preserve scientific precision unless a geometry control was actually changed."""
    visible = panel.current_settings()
    values = {}
    for recipe_key, attribute, control_name in (
        ("distance_mm", "distance", "distance_spinbox"),
        ("grazing_angle_deg", "grazing_angle", "angle_spinbox"),
        ("wavelength_nm", "wavelength", "wavelength_spinbox"),
        ("beam_center_x_px", "beam_center_x", "beam_center_x_spinbox"),
        ("beam_center_y_px", "beam_center_y", "beam_center_y_spinbox"),
        ("pixel_size_x_um", "pixel_size_x", "pixel_size_x_spinbox"),
        ("pixel_size_y_um", "pixel_size_y", "pixel_size_y_spinbox"),
    ):
        value = float(getattr(visible, attribute))
        control = getattr(panel, control_name, None)
        if authoritative is not None and control is not None:
            exact = float(getattr(authoritative, attribute))
            # Read-only projection: never re-apply rounded widgets during capture.
            # A deliberate edit differs by at least one representable control step.
            half_step = 0.5 * 10.0 ** -control.decimals()
            if math.isfinite(exact) and abs(value - exact) <= half_step + 1e-12:
                value = exact
        values[recipe_key] = value
    return values


class InsituRecipeBindingMixin:
    """Adapt Single widgets to versioned, framework-neutral Recipe commands."""

    def _capture_current_insitu_recipe(self):
        """Adapt current widgets to a framework-neutral explicit Recipe command."""
        try:
            if self.fitting_view_model.state.image_status != "ready":
                raise ValueError(
                    "Load and verify one representative file in Single analysis first."
                )
            payload = self._current_insitu_recipe_payload()
            source_path = str(
                self.current_parameters.get("imported_gisaxs_file", "")
            )
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
                note=f"Captured from representative file: {source_path}",
            )
            recipe = self.fitting_view_model.insitu.create_recipe_from_single(snapshot)
            page = getattr(self.ui, "fittingInsituSeriesPage", None)
            if page is not None:
                page.render_recipe(recipe)
            self._populate_insitu_sequence_folder_default()
            self._add_fitting_success(
                f"In-situ Recipe v{recipe.version} captured from Single analysis"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self._add_fitting_error(str(exc))

    def _current_insitu_recipe_payload(self) -> dict[str, dict[str, object]]:
        detector_panel = getattr(self.ui, "fittingDetectorSetupPanel", None)
        if detector_panel is None:
            raise ValueError("Detector setup is not available")
        load_detector = getattr(self.fitting_view_model, "load_detector_settings", None)
        authoritative = load_detector() if callable(load_detector) else None
        detector_geometry = _recipe_detector_geometry(detector_panel, authoritative)
        parameter_snapshot = self._build_fitting_parameter_snapshot()
        fitting_values = parameter_snapshot.get("fitting", {})
        model = {
            "workflow_v5": self._workflow_options() if hasattr(self, "_workflow_options") else {},
            "workflow_input_selection": self._workflow_input_selection() if hasattr(self,"_workflow_input_selection") else {},
            "schema": parameter_snapshot.get("schema", "gimap_fitting_parameters_v1"),
            "model_parameters": parameter_snapshot.get("model_parameters", {}),
            "fitting_params": (
                fitting_values.get("fitting_params", {})
                if isinstance(fitting_values, dict)
                else {}
            ),
        }
        return {
            "experiment_setup": detector_geometry,
            "preprocessing": {
                "invalid_margin_px": int(getattr(self, "_invalid_margin_px", 3)),
                "flip_ud": bool(getattr(self, "_flip_ud", False)),
                "threshold_enabled": bool(
                    getattr(self, "_threshold_mask_enabled", False)
                ),
                "threshold_min": float(
                    getattr(self, "_threshold_mask_min", -1e12)
                ),
                "threshold_max": float(
                    getattr(self, "_threshold_mask_max", 1e12)
                ),
                "mirror_fill_gaps": bool(
                    getattr(self, "_mirror_fill_detector_gaps", False)
                ),
                "mirror_gap_margin_px": int(
                    getattr(self, "_mirror_gap_margin_px", 0)
                ),
            },
            "cut": {
                "center_vertical_px": self.ui.gisaxsInputCenterVerticalValue.value(),
                "center_parallel_px": self.ui.gisaxsInputCenterParallelValue.value(),
                "cut_vertical_px": self.ui.gisaxsInputCutLineVerticalValue.value(),
                "cut_parallel_px": self.ui.gisaxsInputCutLineParallelValue.value(),
                "auto_horizontal_thickness_px": int(
                    self.ui.gisaxsAutoYonedaCutThicknessSpinBox.value()
                ),
            },
            "model": model,
        }

    def _current_ui_matches_insitu_recipe(self) -> bool:
        recipe = self.fitting_view_model.insitu.recipe
        if recipe is None:
            return True
        try:
            current = self._current_insitu_recipe_payload()
            expected = recipe.to_dict()
            categories = ("experiment_setup", "preprocessing", "cut", "model")
            return all(
                json.dumps(current[name], sort_keys=True, ensure_ascii=False)
                == json.dumps(expected[name], sort_keys=True, ensure_ascii=False)
                for name in categories
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
            # V5 does not reuse mutable Single model widgets: its conditions/input
            # selection are fully captured, just like detector and cut geometry.
            return ""
        if recipe.source == "insitu_edit":
            try:
                current_model = self._current_insitu_recipe_payload()["model"]
                if json.dumps(current_model, sort_keys=True, ensure_ascii=False) == json.dumps(
                    recipe.to_dict()["model"], sort_keys=True, ensure_ascii=False
                ):
                    return ""
            except (AttributeError, TypeError, ValueError):
                pass
            return (
                f"The Single analysis model changed after Recipe v{recipe.version} was created. "
                "Capture the intended model again before running the series."
            )
        if self._current_ui_matches_insitu_recipe():
            return ""
        return (
            f"Single analysis has changed since Recipe v{recipe.version} was captured. "
            "Return to In-situ series and explicitly capture a new Recipe before running."
        )


__all__ = ["InsituRecipeBindingMixin"]
