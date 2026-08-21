"""Apply an In-situ Recipe at the legacy scientific execution seam."""

from __future__ import annotations


class InsituRecipeRuntimeMixin:
    """Keep In-situ runtime values separate from visible Single controls."""

    _DETECTOR_SETTINGS = (
        ("fitting", "detector.distance", "distance_mm", 2565.0),
        ("beam", "grazing_angle", "grazing_angle_deg", 0.4),
        ("beam", "wavelength", "wavelength_nm", 0.1045),
        ("fitting", "detector.beam_center_x", "beam_center_x_px", 0.0),
        ("fitting", "detector.beam_center_y", "beam_center_y_px", 0.0),
        ("fitting", "detector.pixel_size_x", "pixel_size_x_um", 172.0),
        ("fitting", "detector.pixel_size_y", "pixel_size_y_um", 172.0),
    )

    def _activate_insitu_recipe_runtime(self) -> None:
        recipe = self.fitting_view_model.insitu.recipe
        if recipe is None:
            raise ValueError("Capture an In-situ Recipe before processing.")
        if getattr(self, "_insitu_runtime_snapshot", None) is None:
            self._insitu_runtime_snapshot = {
                "preprocessing": {
                    "_flip_ud": bool(getattr(self, "_flip_ud", False)),
                    "_threshold_mask_enabled": bool(
                        getattr(self, "_threshold_mask_enabled", False)
                    ),
                    "_threshold_mask_min": float(
                        getattr(self, "_threshold_mask_min", -1e12)
                    ),
                    "_threshold_mask_max": float(
                        getattr(self, "_threshold_mask_max", 1e12)
                    ),
                    "_mirror_fill_detector_gaps": bool(
                        getattr(self, "_mirror_fill_detector_gaps", False)
                    ),
                    "_mirror_gap_margin_px": int(
                        getattr(self, "_mirror_gap_margin_px", 0)
                    ),
                },
                "detector": {
                    (group, key): self.fitting_view_model.get_setting(
                        group, key, default
                    )
                    for group, key, _recipe_key, default in self._DETECTOR_SETTINGS
                },
            }
        preprocess = recipe.preprocessing
        self._flip_ud = bool(preprocess.get("flip_ud", False))
        self._threshold_mask_enabled = bool(
            preprocess.get("threshold_enabled", False)
        )
        self._threshold_mask_min = float(preprocess.get("threshold_min", -1e12))
        self._threshold_mask_max = float(preprocess.get("threshold_max", 1e12))
        self._mirror_fill_detector_gaps = bool(
            preprocess.get("mirror_fill_gaps", False)
        )
        self._mirror_gap_margin_px = int(preprocess.get("mirror_gap_margin_px", 0))
        for group, key, recipe_key, default in self._DETECTOR_SETTINGS:
            self.fitting_view_model.set_setting(
                group,
                key,
                float(recipe.experiment_setup.get(recipe_key, default)),
            )
        self._q_mesh_cache_key = None

    def _restore_single_analysis_runtime(self) -> None:
        snapshot = getattr(self, "_insitu_runtime_snapshot", None)
        if not snapshot:
            return
        for name, value in snapshot["preprocessing"].items():
            setattr(self, name, value)
        for (group, key), value in snapshot["detector"].items():
            self.fitting_view_model.set_setting(group, key, value)
        self._insitu_runtime_snapshot = None
        self._q_mesh_cache_key = None

    def _insitu_cut_geometry(self) -> dict[str, float]:
        recipe = self.fitting_view_model.insitu.recipe
        if recipe is None:
            return {}
        return {
            "center_vertical_px": float(recipe.cut.get("center_vertical_px", 0.0)),
            "center_parallel_px": float(recipe.cut.get("center_parallel_px", 0.0)),
            "cut_vertical_px": float(recipe.cut.get("cut_vertical_px", 0.0)),
            "cut_parallel_px": float(recipe.cut.get("cut_parallel_px", 0.0)),
        }


__all__ = ["InsituRecipeRuntimeMixin"]
