from __future__ import annotations

import json

import pytest

from src.gimap.features.fitting.application import (
    CreateInSituRecipe,
    ReviseInSituRecipe,
    ReviseInSituRecipeRequest,
    SingleAnalysisRecipeSnapshot,
)
from src.gimap.features.fitting.domain import (
    InSituFittingPolicy,
    InSituProcessingRecipe,
    InSituTrackingPolicy,
)
from src.gimap.features.fitting.presentation.bindings.insitu_recipe_runtime import (
    InsituRecipeRuntimeMixin,
)


def _snapshot() -> SingleAnalysisRecipeSnapshot:
    return SingleAnalysisRecipeSnapshot(
        experiment_setup={"distance_mm": 2000.0, "pixel_um": [172.0, 172.0]},
        preprocessing={"flip_ud": True, "mirror_fill": False},
        cut={"center": [512.0, 256.0], "width_px": 5},
        model={"shapes": ["sphere"], "parameters": {"R": 12.0}},
        tracking=InSituTrackingPolicy(center="previous_success", yoneda="fixed"),
        fitting=InSituFittingPolicy(
            initialization="previous_success",
            refinement="every_n",
            refine_every_n=5,
            failure="continue",
        ),
        note="Validated representative frame",
    )


def test_create_recipe_detaches_single_analysis_values_and_is_json_serializable():
    source = {"distance_mm": 2000.0, "nested": {"values": [1, 2]}}
    snapshot = SingleAnalysisRecipeSnapshot(
        experiment_setup=source,
        preprocessing={},
        cut={},
        model={},
    )

    recipe = CreateInSituRecipe(lambda: "2026-08-21T10:00:00").execute(snapshot)
    source["distance_mm"] = 999.0
    source["nested"]["values"].append(3)

    assert recipe.version == 1
    assert recipe.source == "single_analysis"
    assert recipe.experiment_setup["distance_mm"] == 2000.0
    assert recipe.experiment_setup["nested"]["values"] == (1, 2)
    json.dumps(recipe.to_dict())


def test_revise_recipe_creates_child_without_mutating_previous_version():
    original = CreateInSituRecipe(lambda: "created").execute(_snapshot())
    revision = ReviseInSituRecipe(lambda: "revised").execute(
        ReviseInSituRecipeRequest(
            current=original,
            preprocessing={"flip_ud": False, "mirror_fill": True},
            scope="future",
            note="Mirror fill enabled after frame 10",
        )
    )

    assert original.version == 1
    assert original.preprocessing["mirror_fill"] is False
    assert revision.recipe.version == 2
    assert revision.recipe.parent_version == 1
    assert revision.recipe.source == "insitu_edit"
    assert revision.recipe.preprocessing["mirror_fill"] is True
    assert revision.scope == "future"


def test_explicit_single_recapture_creates_next_recipe_version():
    creator = CreateInSituRecipe(iter(("first", "second")).__next__)
    first = creator.execute(_snapshot())
    second = creator.execute(_snapshot(), first)

    assert second.version == 2
    assert second.parent_version == 1
    assert second.source == "single_analysis"


def test_selected_and_future_scope_requires_explicit_selection():
    recipe = CreateInSituRecipe(lambda: "created").execute(_snapshot())

    with pytest.raises(ValueError, match="selected frame"):
        ReviseInSituRecipe().execute(
            ReviseInSituRecipeRequest(
                current=recipe,
                scope="selected_and_future",
            )
        )


def test_recipe_round_trip_preserves_policies_and_nested_values():
    recipe = CreateInSituRecipe(lambda: "created").execute(_snapshot())

    restored = InSituProcessingRecipe.from_dict(recipe.to_dict())

    assert restored.to_dict() == recipe.to_dict()
    assert restored.tracking.center == "previous_success"
    assert restored.fitting.refinement == "every_n"
    assert restored.fitting.refine_every_n == 5


def test_recipe_rejects_runtime_objects_at_the_boundary():
    with pytest.raises(ValueError, match="JSON serializable"):
        CreateInSituRecipe().execute(
            SingleAnalysisRecipeSnapshot(
                experiment_setup={"bad": object()},
                preprocessing={},
                cut={},
                model={},
            )
        )


def test_recipe_runtime_applies_insitu_values_and_restores_single_state():
    recipe = CreateInSituRecipe(lambda: "created").execute(
        SingleAnalysisRecipeSnapshot(
            experiment_setup={
                "distance_mm": 3100.0,
                "grazing_angle_deg": 0.31,
                "wavelength_nm": 0.12,
                "beam_center_x_px": 411.0,
                "beam_center_y_px": 233.0,
                "pixel_size_x_um": 75.0,
                "pixel_size_y_um": 76.0,
            },
            preprocessing={
                "flip_ud": True,
                "threshold_enabled": True,
                "threshold_min": 2.0,
                "threshold_max": 900.0,
                "mirror_fill_gaps": True,
                "mirror_gap_margin_px": 3,
            },
            cut={
                "center_vertical_px": 22.0,
                "center_parallel_px": 33.0,
                "cut_vertical_px": 5.0,
                "cut_parallel_px": 7.0,
            },
            model={},
        )
    )

    class FakeViewModel:
        def __init__(self):
            self.insitu = type("InSitu", (), {"recipe": recipe})()
            self.values = {
                ("fitting", "detector.distance"): 2000.0,
                ("beam", "grazing_angle"): 0.2,
                ("beam", "wavelength"): 0.1,
                ("fitting", "detector.beam_center_x"): 10.0,
                ("fitting", "detector.beam_center_y"): 11.0,
                ("fitting", "detector.pixel_size_x"): 172.0,
                ("fitting", "detector.pixel_size_y"): 172.0,
            }

        def get_setting(self, group, key, default=None):
            return self.values.get((group, key), default)

        def set_setting(self, group, key, value):
            self.values[(group, key)] = value

    runtime = InsituRecipeRuntimeMixin()
    runtime.fitting_view_model = FakeViewModel()
    runtime._flip_ud = False
    runtime._threshold_mask_enabled = False
    runtime._threshold_mask_min = -1e12
    runtime._threshold_mask_max = 1e12
    runtime._mirror_fill_detector_gaps = False
    runtime._mirror_gap_margin_px = 0
    runtime._q_mesh_cache_key = "single"
    runtime._insitu_runtime_snapshot = None

    runtime._activate_insitu_recipe_runtime()

    assert runtime._flip_ud is True
    assert runtime._mirror_fill_detector_gaps is True
    assert runtime.fitting_view_model.values[("fitting", "detector.distance")] == 3100.0
    assert runtime._insitu_cut_geometry()["cut_parallel_px"] == 7.0

    runtime._restore_single_analysis_runtime()

    assert runtime._flip_ud is False
    assert runtime._mirror_fill_detector_gaps is False
    assert runtime.fitting_view_model.values[("fitting", "detector.distance")] == 2000.0
