from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    BoundsFirstLabel,
    BoundsProvenance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_dataset import (
    sample_solution_label,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import ProfiledBranchCodec
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_component_slots import (
    MAX_CANONICAL_ASSIGNMENTS,
    canonicalize_component_slots,
    component_slot_equivalence_classes,
    component_physical_dictionary_key,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    gui_component_to_latent,
    latent_component_to_gui,
)


def _sphere(radius: float, *, d: float | None = None):
    return gui_component_to_latent(
        GuiComponentParameters(
            SPHERE,
            R=radius,
            sigma_R=0.1 * radius,
            D=d,
            sigma_D=None if d is None else 0.1 * d,
        )
    )


def _bounds(r_low, r_high, *, d=None, allow_d_absent=False):
    return GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(r_low, r_high),
        sigma_R=ClosedInterval(0.08 * r_low, 0.12 * r_high),
        D=None if d is None else ClosedInterval(*d),
        sigma_D=None if d is None else ClosedInterval(0.08 * d[0], 0.12 * d[1]),
        allow_D_absent=allow_d_absent,
    )


def _codec(bounds, d_present):
    return ProfiledBranchCodec.build(
        (SPHERE,) * len(bounds),
        bounds,
        d_present,
    )


def test_overlapping_bounds_choose_one_physical_lexicographic_representative():
    codec = _codec((_bounds(5.0, 25.0),) * 3, (False, False, False))
    components = (_sphere(20.0), _sphere(10.0), _sphere(15.0))
    expected = tuple(sorted(components, key=component_physical_dictionary_key))
    intensities = (ClosedInterval(0.0, 1.0),) * 3

    results = {
        canonicalize_component_slots(
            codec,
            order,
            None,
            component_intensity_bounds=intensities,
        ).components
        for order in permutations(components)
    }
    assert results == {expected}
    assignment = canonicalize_component_slots(
        codec,
        components,
        None,
        component_intensity_bounds=intensities,
    )
    assert assignment.components == expected
    assert assignment.slot_equivalence_classes == ((0, 1, 2),)
    assert assignment.exchangeability_context_complete
    assert codec.decode(assignment.coordinates)[0] == pytest.approx(expected)
    with pytest.raises(ValueError, match="unsupported component-slot"):
        replace(assignment, version="stale-v1")


def test_four_component_permutation_invariance_is_bounded_by_four_factorial():
    codec = _codec((_bounds(5.0, 30.0),) * 4, (False,) * 4)
    components = tuple(_sphere(value) for value in (20.0, 10.0, 25.0, 15.0))
    expected = tuple(sorted(components, key=component_physical_dictionary_key))
    intensities = (ClosedInterval(0.0, 1.0),) * 4

    for order in permutations(components):
        assert (
            canonicalize_component_slots(
                codec,
                order,
                None,
                component_intensity_bounds=intensities,
            ).components
            == expected
        )
    assert MAX_CANONICAL_ASSIGNMENTS == 24


def test_nonoverlapping_user_bounds_override_unavailable_lexicographic_slot_order():
    large_slot = _bounds(15.0, 25.0)
    small_slot = _bounds(5.0, 12.0)
    codec = _codec((large_slot, small_slot), (False, False))
    small, large = _sphere(10.0), _sphere(20.0)

    with pytest.raises(ValueError, match="no shape/D-compatible.*feasible"):
        canonicalize_component_slots(codec, (small, large), None)
    second = canonicalize_component_slots(codec, (large, small), None)
    assert second.components == (large, small)
    assert second.slot_equivalence_classes == ((0,), (1,))
    assert not second.exchangeability_context_complete


def test_overlapping_geometry_and_unequal_int_ranges_preserve_distinct_labels():
    codec = _codec(
        (_bounds(5.0, 25.0), _bounds(10.0, 30.0)),
        (False, False),
    )
    first, second = _sphere(15.0), _sphere(20.0)
    intensities = (ClosedInterval(0.0, 0.4), ClosedInterval(0.6, 1.0))

    forward = canonicalize_component_slots(
        codec,
        (first, second),
        None,
        component_intensity_bounds=intensities,
    )
    reverse = canonicalize_component_slots(
        codec,
        (second, first),
        None,
        component_intensity_bounds=intensities,
    )

    assert component_slot_equivalence_classes(
        codec,
        component_intensity_bounds=intensities,
    ) == ((0,), (1,))
    assert forward.components == (first, second)
    assert reverse.components == (second, first)
    assert forward.components != reverse.components
    assert forward.coordinates != reverse.coordinates


def test_missing_amplitude_context_is_conservative_even_for_identical_geometry():
    codec = _codec((_bounds(5.0, 25.0),) * 2, (False, False))
    first, second = _sphere(15.0), _sphere(20.0)

    forward = canonicalize_component_slots(codec, (first, second), None)
    reverse = canonicalize_component_slots(codec, (second, first), None)

    assert component_slot_equivalence_classes(codec) == ((0,), (1,))
    assert forward.components == (first, second)
    assert reverse.components == (second, first)
    assert not forward.exchangeability_context_complete


def test_shape_and_d_signature_are_not_exchangeable_even_for_equal_shapes():
    codec = _codec(
        (
            _bounds(8.0, 12.0),
            _bounds(8.0, 12.0, d=(50.0, 80.0)),
        ),
        (False, True),
    )
    absent = _sphere(10.0)
    present = _sphere(10.0, d=60.0)

    with pytest.raises(ValueError, match="no shape/D-compatible.*feasible"):
        canonicalize_component_slots(
            codec,
            (present, absent),
            None,
            component_intensity_bounds=(ClosedInterval(0.0, 1.0),) * 2,
        )
    assignment = canonicalize_component_slots(
        codec,
        (absent, present),
        None,
        component_intensity_bounds=(ClosedInterval(0.0, 1.0),) * 2,
    )
    assert assignment.components == (absent, present)


def test_no_user_bounds_feasible_assignment_fails_closed():
    codec = _codec(
        (_bounds(5.0, 12.0), _bounds(5.0, 12.0)),
        (False, False),
    )
    with pytest.raises(ValueError, match="no shape/D-compatible.*feasible"):
        canonicalize_component_slots(codec, (_sphere(15.0), _sphere(20.0)), None)


def test_bounds_first_sampler_and_label_contract_share_canonical_slots():
    bounds = BoundsProvenance.create(
        bounds_seed=91,
        generation_attempt=0,
        range_regime="wide",
        placement="interior",
        component_bounds=(_bounds(5.0, 25.0), _bounds(5.0, 25.0)),
        d_present=(False, False),
        resolution_bounds=None,
    )
    label = sample_solution_label(bounds, local_target_seed=101)
    codec = bounds.local_codec()
    latent, resolution = codec.decode(label.local_target_unit)
    canonical = canonicalize_component_slots(codec, latent, resolution)

    assert canonical.components == latent
    assert codec.latent_components_to_gui(latent) == label.truth_components

    reversed_latent = tuple(reversed(latent))
    reversed_local = bounds.local_codec().encode(reversed_latent, None)
    reversed_global = bounds.global_reference_codec().encode(reversed_latent, None)
    reversed_label = BoundsFirstLabel(
        task_kind="in_domain_solution",
        bounds=bounds,
        local_target_unit=reversed_local.unit_cube,
        global_reference_unit=reversed_global.unit_cube,
        truth_components=tuple(latent_component_to_gui(item) for item in reversed_latent),
        truth_resolution=None,
    )
    assert reversed_label.truth_components != label.truth_components
