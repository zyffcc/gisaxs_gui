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
