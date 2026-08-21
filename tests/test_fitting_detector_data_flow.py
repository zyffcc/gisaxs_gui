from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.gimap.features.fitting.application import FittingImageCalculations
from src.gimap.features.fitting.presentation.detector_data_access import (
    analysis_image_for,
    analysis_revision_for,
)
from src.gimap.features.fitting.presentation.state import FittingState
from src.gimap.features.fitting.presentation.workflow_view_model import (
    FittingWorkflowViewModelMixin,
)


ROOT = Path(__file__).resolve().parents[1]
FITTING_SOURCE = ROOT / "src" / "gimap" / "features" / "fitting"


def test_prepare_detector_image_builds_one_read_only_analysis_revision():
    raw = np.array(
        [[10.0, -1.0, 30.0, 40.0, 50.0], [60.0, -1.0, 80.0, 90.0, 100.0]],
        dtype=np.float32,
    )
    original = raw.copy()

    state = FittingImageCalculations().prepare(
        raw,
        revision=7,
        flip_ud=True,
        mirror_fill_gaps=True,
        mirror_center_x=2.0,
    )

    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(state.raw_image, original)
    np.testing.assert_array_equal(
        state.analysis_image,
        [[60.0, 90.0, 80.0, 90.0, 100.0], [10.0, 40.0, 30.0, 40.0, 50.0]],
    )
    assert state.revision == 7
    assert state.mirror_filled_gap_pixels == 2
    assert state.raw_image.flags.writeable is False
    assert state.analysis_image.flags.writeable is False
    assert not np.shares_memory(state.raw_image, state.analysis_image)


def test_preprocessing_rebuilds_from_raw_instead_of_accumulating_transforms():
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    calculations = FittingImageCalculations()

    flipped = calculations.prepare(raw, revision=1, flip_ud=True)
    restored = calculations.prepare(flipped.raw_image, revision=2, flip_ud=False)

    np.testing.assert_array_equal(flipped.analysis_image, np.flipud(raw))
    np.testing.assert_array_equal(restored.analysis_image, raw)
    np.testing.assert_array_equal(flipped.raw_image, raw)


def test_stack_gap_value_is_mirror_filled_in_the_canonical_analysis_image():
    summed_raw = np.array([[20.0, -2.0, 60.0, 80.0, 100.0]], dtype=np.float32)

    state = FittingImageCalculations().prepare(
        summed_raw,
        revision=3,
        mirror_fill_gaps=True,
        mirror_center_x=2.0,
        mirror_gap_value=-2.0,
    )

    np.testing.assert_array_equal(state.analysis_image, [[20.0, 80.0, 60.0, 80.0, 100.0]])


def test_preview_and_scientific_consumers_resolve_the_same_analysis_object():
    state = FittingImageCalculations().prepare(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        revision=11,
        flip_ud=True,
    )
    owner = SimpleNamespace(
        current_detector_image=state,
        current_analysis_image=np.zeros_like(state.analysis_image),
        current_stack_data=np.ones_like(state.analysis_image),
    )

    assert analysis_image_for(owner) is state.analysis_image
    assert analysis_revision_for(owner) == 11


def test_analysis_revision_invalidates_an_existing_cut_and_is_recorded_on_new_cut():
    owner = SimpleNamespace(state=FittingState(analysis_revision=4, cut_status="ready"))

    FittingWorkflowViewModelMixin.accept_analysis_revision(owner, 5)

    assert owner.state.analysis_revision == 5
    assert owner.state.cut_status == "stale"
    assert owner.state.workflow.step("center").status == "available"
    FittingWorkflowViewModelMixin.complete_workflow_step(owner, "cut", "Cut ready")
    assert owner.state.cut_status == "ready"
    assert owner.state.cut_result_analysis_revision == 5

    previous = owner.state
    FittingWorkflowViewModelMixin.accept_analysis_revision(owner, 5)
    assert owner.state == previous


def test_fitting_source_has_no_hidden_raw_or_legacy_analysis_consumers():
    raw_alias_owners = {
        Path("presentation/bindings/image_display_loading.py"),
        Path("presentation/bindings/image_display_options.py"),
        Path("presentation/view_binding.py"),
    }
    stack_alias_owners = {
        Path("presentation/bindings/image_display_options.py"),
        Path("presentation/detector_data_access.py"),
        Path("presentation/view_binding.py"),
    }

    raw_consumers = set()
    stack_consumers = set()
    display_dependencies = set()
    presentation_preprocessing = set()
    for path in FITTING_SOURCE.rglob("*.py"):
        relative = path.relative_to(FITTING_SOURCE)
        source = path.read_text(encoding="utf-8")
        if "current_raw_image" in source:
            raw_consumers.add(relative)
        if "current_stack_data" in source:
            stack_consumers.add(relative)
        if relative.parts[0] in {"application", "domain"} and "_get_current_display_image" in source:
            display_dependencies.add(relative)
        if relative.parts[0] == "presentation" and (
            ".image.transform(" in source or ".image.mirror_gaps(" in source
        ):
            presentation_preprocessing.add(relative)

    assert raw_consumers <= raw_alias_owners
    assert stack_consumers <= stack_alias_owners
    assert not display_dependencies
    assert not presentation_preprocessing
